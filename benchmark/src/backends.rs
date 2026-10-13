//! Reproducible v2.2 backend comparison. Graph, TaskPool and independent output
//! oracle are built outside observation. Every path includes request construction,
//! evaluator initialization, result reads and output retention; entity DFS counts
//! all independent evaluators and loss of shared dependencies. Persistent caches
//! are Off. Timing and allocation executables use the same workload/observer API.
use crate::measurement::Observer;
use bevy_tasks::TaskPool;
use gekijyou_bevy::BevyExecutor;
use gekijyou_computation::{
    CachePolicy, EvalStats, Evaluator, FrameIndex, GraphState, Node, NodeCacheConfig,
    PlannerConfig, PlannerStats, RequestBatch, SerialExecutor,
};
use std::{
    hint::black_box,
    io::{self, Write},
};

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Dfs,
    PlannerSerial,
    PlannerParallel(usize),
    EntityParallelDfs,
}
impl Backend {
    pub const ALL: [Self; 5] = [
        Self::Dfs,
        Self::PlannerSerial,
        Self::PlannerParallel(1),
        Self::PlannerParallel(4),
        Self::EntityParallelDfs,
    ];
    fn label(self) -> String {
        match self {
            Self::Dfs => "dfs".into(),
            Self::PlannerSerial => "planner_serial".into(),
            Self::PlannerParallel(n) => format!("planner_parallel_{n}"),
            Self::EntityParallelDfs => "entity_parallel_dfs".into(),
        }
    }
}
// Pure integer work: the same per-closure arithmetic order in every backend.
fn compute(phase: u64, actor: u64, rounds: u64) -> u64 {
    let mut v = phase.wrapping_add(actor);
    for n in 0..rounds {
        v = black_box(v.rotate_left(7).wrapping_mul(31).wrapping_add(n));
    }
    v
}
pub struct BackendScene {
    state: GraphState,
    phase: Node<u64>,
    nodes: Vec<Node<u64>>,
    frames: usize,
    oracle: Vec<u64>,
}
pub struct BackendResult {
    pub values: Vec<u64>,
    pub calls: u64,
    pub initialized_evaluators: usize,
    pub planner: PlannerStats,
}
impl BackendScene {
    pub fn new(actors: usize, frames: usize, rounds: u64, periodic: bool) -> Self {
        let mut state = GraphState::new();
        let (phase, nodes) = state
            .edit(|e| {
                e.set_default_eq::<u64>();
                let phase = e.map(
                    e.frame(),
                    move |f: &FrameIndex| if periodic { f.0 % 8 } else { f.0 },
                )?;
                let nodes = (0..actors)
                    .map(|a| e.map(phase, move |p: &u64| compute(*p, a as u64, rounds)))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((phase, nodes))
            })
            .unwrap();
        // Separate oracle implementation, not graph traversal or the worker kernel.
        let mut oracle = Vec::new();
        for f in 0..frames {
            for a in 0..actors {
                let mut v = (if periodic { f as u64 % 8 } else { f as u64 }).wrapping_add(a as u64);
                for n in 0..rounds {
                    v = v.rotate_left(7).wrapping_mul(31).wrapping_add(n);
                }
                oracle.push(v);
            }
        }
        Self {
            state,
            phase,
            nodes,
            frames,
            oracle,
        }
    }
    fn evaluator(&self) -> Evaluator {
        let mut eval = Evaluator::new(&self.state);
        let off = NodeCacheConfig {
            memo: CachePolicy::Off,
            context: CachePolicy::Off,
        };
        eval.configure_cache(self.phase, off).unwrap();
        for &n in &self.nodes {
            eval.configure_cache(n, off).unwrap();
        }
        eval
    }
    pub fn verify(&self, result: &BackendResult) {
        assert_eq!(result.values, self.oracle);
    }
    pub fn run(&self, backend: Backend, pool: &TaskPool) -> BackendResult {
        if matches!(backend, Backend::EntityParallelDfs) {
            // Each entity owns an evaluator over the same immutable global graph.
            // Creation/configuration and duplicated phase work are inside the scope.
            let values: Vec<(usize, Vec<u64>, EvalStats)> = pool.scope(|scope| {
                for (actor, &node) in self.nodes.iter().enumerate() {
                    scope.spawn(async move {
                        let mut eval = Evaluator::new(&self.state);
                        let off = NodeCacheConfig {
                            memo: CachePolicy::Off,
                            context: CachePolicy::Off,
                        };
                        eval.configure_cache(self.phase, off).unwrap();
                        eval.configure_cache(node, off).unwrap();
                        let mut batch = RequestBatch::new();
                        let ids: Vec<_> = (0..self.frames)
                            .map(|f| batch.at(FrameIndex(f as u64)).request(node))
                            .collect();
                        let results = eval
                            .bind(self.state.snapshot())
                            .unwrap()
                            .evaluate(batch)
                            .unwrap();
                        let values = ids
                            .into_iter()
                            .map(|id| *results.get(id).unwrap())
                            .collect();
                        (actor, values, eval.stats())
                    });
                }
            });
            let calls = values.iter().map(|(_, _, s)| s.computation_calls).sum();
            let mut reordered = vec![0; self.frames * self.nodes.len()];
            for (actor, values, _) in values {
                for (f, v) in values.into_iter().enumerate() {
                    reordered[f * self.nodes.len() + actor] = v;
                }
            }
            return BackendResult {
                values: reordered,
                calls,
                initialized_evaluators: self.nodes.len(),
                planner: PlannerStats::default(),
            };
        }
        let mut eval = self.evaluator();
        let mut batch = RequestBatch::new();
        let mut ids = Vec::new();
        for f in 0..self.frames {
            for &n in &self.nodes {
                ids.push(batch.at(FrameIndex(f as u64)).request(n));
            }
        }
        let config = PlannerConfig {
            frame_window: self.frames,
            comparison_budget: 16,
            ..Default::default()
        };
        let (results, planner) = match backend {
            Backend::Dfs => (
                eval.bind(self.state.snapshot())
                    .unwrap()
                    .evaluate(batch)
                    .unwrap(),
                PlannerStats::default(),
            ),
            Backend::PlannerSerial => eval
                .bind(self.state.snapshot())
                .unwrap()
                .evaluate_planned(batch, config, &mut SerialExecutor)
                .unwrap(),
            Backend::PlannerParallel(chunk) => eval
                .bind(self.state.snapshot())
                .unwrap()
                .evaluate_planned(batch, config, &mut BevyExecutor::new(pool, chunk))
                .unwrap(),
            _ => unreachable!(),
        };
        BackendResult {
            values: ids.iter().map(|&id| *results.get(id).unwrap()).collect(),
            calls: eval.stats().computation_calls,
            initialized_evaluators: 1,
            planner,
        }
    }
}

pub fn measure(
    observer: &mut impl Observer,
    out: &mut impl Write,
    samples: usize,
) -> io::Result<()> {
    let pool = bevy_tasks::TaskPoolBuilder::new().num_threads(4).build();
    writeln!(
        out,
        "observer\tsample\tcase\trounds\tbackend\tactors\tframes\tns\tallocs\tpeak_extra_bytes\tlive_delta_bytes\tinitialized_evaluators\tcalls\tregistry_comparisons\tpeak_resolutions\tpeak_registry\tpeak_in_flight"
    )?;
    for rounds in [0, 20000] {
        for periodic in [false, true] {
            let scene = BackendScene::new(32, 64, rounds, periodic);
            for backend in Backend::ALL {
                scene.verify(&scene.run(backend, &pool));
            }
            for sample in 0..samples {
                for offset in 0..Backend::ALL.len() {
                    let backend = Backend::ALL[(sample + offset) % Backend::ALL.len()];
                    let mut result = None;
                    let observed = observer.observe(|| {
                        let measured = scene.run(backend, &pool);
                        let checksum = measured
                            .values
                            .iter()
                            .fold(0_u64, |a, b| a.wrapping_add(*b));
                        result = Some(measured);
                        checksum
                    });
                    let result = result.unwrap();
                    scene.verify(&result);
                    let elapsed = observed
                        .elapsed
                        .map(|d| d.as_nanos().to_string())
                        .unwrap_or_else(|| "NA".into());
                    let (allocs, peak, live) = observed
                        .allocations
                        .map(|a| {
                            (
                                a.allocs.to_string(),
                                a.peak_extra.to_string(),
                                a.live_delta.to_string(),
                            )
                        })
                        .unwrap_or_else(|| ("NA".into(), "NA".into(), "NA".into()));
                    writeln!(
                        out,
                        "{}\t{sample}\t{}\t{rounds}\t{}\t32\t64\t{elapsed}\t{allocs}\t{peak}\t{live}\t{}\t{}\t{}\t{}\t{}\t{}",
                        observer.name(),
                        if periodic { "periodic8" } else { "distinct" },
                        backend.label(),
                        result.initialized_evaluators,
                        result.calls,
                        result.planner.equivalence_comparisons,
                        result.planner.peak_resolution_tasks,
                        result.planner.peak_registry_entries,
                        result.planner.peak_in_flight
                    )?;
                }
            }
        }
    }
    Ok(())
}
