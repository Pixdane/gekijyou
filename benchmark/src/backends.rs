//! Reproducible backend comparison over one unified load matrix.
//!
//! Graph, task pool and the independent output oracle are built outside
//! observation. Every path includes request construction, evaluator
//! initialization, result reads and output retention; independent entity DFS
//! counts all evaluators and the loss of shared dependencies. Graph and closures
//! never appear in the oracle, which is computed from direct integer expressions.
//!
//! Persistent caches are selected by [`CacheMode`]; cold runs create and destroy
//! their evaluators inside observation while warm runs reuse a preheated set
//! prepared outside it. Timing and allocation executables share this workload and
//! observer API; allocation observation excludes the asynchronous backend because
//! a process-global counter cannot bound a detached physical future's lifetime.
//! Setup, preflight verification and pool lifetime stay outside the observed
//! region. See [`measure`] for the complete matrix and measurement boundaries.

use crate::measurement::Observer;
use bevy_tasks::{TaskPool, TaskPoolBuilder};
use gekijyou_bevy::{AsyncDriver, BevyExecutor, TaskLimits, TaskStats};
use gekijyou_computation::{
    CachePolicy, EvalStats, EvaluationResults, EvaluationSnapshot, Evaluator, FrameIndex,
    GraphState, Node, NodeCacheConfig, PlannerConfig, PlannerStats, RequestBatch, RequestId,
    SerialExecutor,
};
use std::{
    collections::HashSet,
    hint::black_box,
    io::{self, Write},
    sync::{Arc, Condvar, Mutex, mpsc},
    thread::ThreadId,
    time::{Duration, Instant},
};

/// Wall-clock or allocation observation. `name()` selects the participating
/// backends: allocation accounting cannot bound the asynchronous backend.
pub fn measure(
    observer: &mut impl Observer,
    out: &mut impl Write,
    samples: usize,
) -> io::Result<()> {
    writeln!(
        out,
        "observer\tsample\tcase\trounds\tbackend\tactors\tframes\tns\tallocs\tpeak_extra_bytes\tlive_delta_bytes\tinitialized_evaluators\tcalls\tregistry_comparisons\tpeak_resolutions\tpeak_registry\tpeak_in_flight\ttopology\tframe_pattern\tmemo\tcontext\tcache_lifetime\tmeasurement_boundary\tframe_window\tmax_in_flight\tmax_pending\tchunk_size\tmemo_hits\tcontext_hits\tresolution_visits\tidentity_hits\tcontent_hits\tworker_jobs_spawned\tdispatched\tterminal"
    )?;
    // The process-global allocation counter cannot delimit a detached physical
    // future, so allocation observation never runs the asynchronous backend.
    let backends: &[Backend] = if observer.name() == "alloc" {
        &Backend::SYNC
    } else {
        &Backend::ALL
    };
    for rounds in ROUNDS {
        for periodic in PERIODIC {
            for topology in Topology::ALL {
                for frame_pattern in FramePattern::ALL {
                    for cache in CacheMode::ALL {
                        for lifetime in CacheLifetime::ALL {
                            let scene = BackendScene::new(SceneConfig {
                                actors: ACTORS,
                                frames: FRAMES,
                                rounds,
                                periodic,
                                topology,
                                frame_pattern,
                            });
                            // Preflight: complete oracle check in its own pool, joined
                            // and dropped before any observation begins.
                            {
                                let pool = benchmark_pool();
                                for &backend in backends {
                                    let mut prepared =
                                        scene.prepare(backend, &pool, cache, lifetime);
                                    let result = prepared.run(&pool);
                                    scene.verify(&result);
                                }
                            }
                            for sample in 0..samples {
                                for offset in 0..backends.len() {
                                    let backend = backends[(sample + offset) % backends.len()];
                                    // Warm preheats in an independent pool that is joined
                                    // and dropped before the measurement pool exists. Cold
                                    // `prepare` creates nothing, so it shares the
                                    // measurement pool created here.
                                    let (pool, mut prepared) = match lifetime {
                                        CacheLifetime::Warm => {
                                            let warmup = benchmark_pool();
                                            let prepared =
                                                scene.prepare(backend, &warmup, cache, lifetime);
                                            drop(warmup);
                                            (benchmark_pool(), prepared)
                                        }
                                        CacheLifetime::Cold => {
                                            let pool = benchmark_pool();
                                            let prepared =
                                                scene.prepare(backend, &pool, cache, lifetime);
                                            (pool, prepared)
                                        }
                                    };
                                    let mut measured = None;
                                    // Request construction, evaluation, output reads and
                                    // retention are inside observation; the checksum is
                                    // consumed here, setup stays outside.
                                    let observed = observer.observe(|| {
                                        let result = prepared.run(&pool);
                                        let checksum = result
                                            .values
                                            .iter()
                                            .fold(0_u64, |acc, value| acc.wrapping_add(*value));
                                        measured = Some(result);
                                        checksum
                                    });
                                    // Join the measurement pool before the warm prepared
                                    // run drops, so teardown never pollutes the next row.
                                    drop(pool);
                                    drop(prepared);
                                    let result = measured.unwrap();
                                    scene.verify(&result);
                                    write_row(
                                        observer.name(),
                                        out,
                                        Row {
                                            sample,
                                            rounds,
                                            periodic,
                                            backend,
                                            cache,
                                            lifetime,
                                            topology,
                                            frame_pattern,
                                            result: &result,
                                            observed: &observed,
                                        },
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

const ACTORS: usize = 32;
const FRAMES: usize = 64;
const ROUNDS: [u64; 2] = [0, 20000];
const PERIODIC: [bool; 2] = [false, true];

struct Row<'a> {
    sample: usize,
    rounds: u64,
    periodic: bool,
    backend: Backend,
    cache: CacheMode,
    lifetime: CacheLifetime,
    topology: Topology,
    frame_pattern: FramePattern,
    result: &'a BackendResult,
    observed: &'a crate::measurement::Observation,
}

fn write_row(observer: &str, out: &mut impl Write, row: Row<'_>) -> io::Result<()> {
    let result = row.result;
    let elapsed = row
        .observed
        .elapsed
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|| "NA".into());
    let (allocs, peak, live) = row
        .observed
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
        "{observer}\t{}\t{}\t{}\t{}\t{ACTORS}\t{FRAMES}\t{elapsed}\t{allocs}\t{peak}\t{live}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        row.sample,
        if row.periodic {
            "periodic8"
        } else {
            "distinct"
        },
        row.rounds,
        row.backend.label(),
        result.initialized_evaluators,
        result.calls,
        result.planner.equivalence_comparisons,
        result.planner.peak_resolution_tasks,
        result.planner.peak_registry_entries,
        result.planner.peak_in_flight,
        row.topology.label(),
        row.frame_pattern.label(),
        row.cache.memo(),
        row.cache.context(),
        row.lifetime.label(),
        measurement_boundary(row.backend, row.lifetime),
        row.backend.frame_window(),
        row.backend.max_in_flight(),
        row.backend.max_pending(),
        row.backend.chunk_size(),
        result.eval.memo_hits,
        result.eval.context_hits,
        result.eval.resolution_visits,
        result.planner.identity_hits,
        result.planner.content_hits,
        result.tasks.worker_jobs_spawned,
        result.tasks.dispatched,
        result.tasks.terminal,
    )
}

fn measurement_boundary(backend: Backend, lifetime: CacheLifetime) -> &'static str {
    match (backend, lifetime) {
        (Backend::Async(_), CacheLifetime::Cold) => "async_logical_results_cold",
        (Backend::Async(_), CacheLifetime::Warm) => "async_logical_results_warm",
        (_, CacheLifetime::Cold) => "cold_full",
        (_, CacheLifetime::Warm) => "warm_incremental",
    }
}

// ---------------------------------------------------------------------------
// Load description
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Dfs,
    PlannerSerial,
    PlannerParallel(usize),
    Async(usize),
    EntityParallelDfs,
}

impl Backend {
    /// Every backend, including the asynchronous one.
    pub const ALL: [Self; 8] = [
        Self::Dfs,
        Self::PlannerSerial,
        Self::PlannerParallel(1),
        Self::PlannerParallel(4),
        Self::Async(1),
        Self::Async(4),
        Self::Async(16),
        Self::EntityParallelDfs,
    ];
    /// Synchronous backends whose load is fully reclaimed when `run` returns.
    pub const SYNC: [Self; 5] = [
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
            Self::Async(n) => format!("async_{n}"),
            Self::EntityParallelDfs => "entity_parallel_dfs".into(),
        }
    }

    fn frame_window(self) -> usize {
        match self {
            Self::PlannerSerial | Self::PlannerParallel(_) => PLANNER_FRAME_WINDOW,
            Self::Async(_) => ASYNC_FRAME_WINDOW,
            Self::Dfs | Self::EntityParallelDfs => 0,
        }
    }

    fn max_in_flight(self) -> usize {
        match self {
            Self::PlannerSerial | Self::PlannerParallel(_) => PLANNER_MAX_IN_FLIGHT,
            Self::Async(_) => ASYNC_MAX_IN_FLIGHT,
            Self::Dfs | Self::EntityParallelDfs => 0,
        }
    }

    fn max_pending(self) -> usize {
        match self {
            Self::Async(_) => ASYNC_MAX_PENDING,
            _ => 0,
        }
    }

    fn chunk_size(self) -> usize {
        match self {
            Self::PlannerParallel(n) | Self::Async(n) => n,
            _ => 0,
        }
    }
}

const PLANNER_FRAME_WINDOW: usize = 16;
const PLANNER_MAX_IN_FLIGHT: usize = 64;
const PLANNER_COMPARISON_BUDGET: usize = 64;
const ASYNC_FRAME_WINDOW: usize = 16;
const ASYNC_MAX_IN_FLIGHT: usize = 64;
const ASYNC_MAX_PENDING: usize = 64;
const ASYNC_LIMITS: TaskLimits = TaskLimits {
    max_in_flight: ASYNC_MAX_IN_FLIGHT,
    max_pending: ASYNC_MAX_PENDING,
};

fn planner_config() -> PlannerConfig {
    PlannerConfig {
        frame_window: PLANNER_FRAME_WINDOW,
        max_in_flight: PLANNER_MAX_IN_FLIGHT,
        comparison_budget: PLANNER_COMPARISON_BUDGET,
        ..Default::default()
    }
}

/// Memo/context policy pair applied to every scene computation. `LruLru` uses
/// capacity 16 for both caches. Raw frames and inputs stay uncached in every
/// mode; hits are reported, never assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMode {
    OffOff,
    LastOff,
    LastLast,
    LruLru,
}

impl CacheMode {
    pub const ALL: [Self; 4] = [Self::OffOff, Self::LastOff, Self::LastLast, Self::LruLru];

    fn config(self) -> NodeCacheConfig {
        NodeCacheConfig {
            memo: self.memo_policy(),
            context: self.context_policy(),
        }
    }

    fn memo_policy(self) -> CachePolicy {
        match self {
            Self::OffOff => CachePolicy::Off,
            Self::LastOff | Self::LastLast => CachePolicy::Last,
            Self::LruLru => CachePolicy::Lru {
                capacity: CACHE_CAPACITY,
            },
        }
    }

    fn context_policy(self) -> CachePolicy {
        match self {
            Self::OffOff | Self::LastOff => CachePolicy::Off,
            Self::LastLast => CachePolicy::Last,
            Self::LruLru => CachePolicy::Lru {
                capacity: CACHE_CAPACITY,
            },
        }
    }

    /// TSV label of the memo policy; shared with the asynchronous measurement binary.
    pub fn memo(self) -> &'static str {
        policy_label(self.memo_policy())
    }

    /// TSV label of the context policy.
    pub fn context(self) -> &'static str {
        policy_label(self.context_policy())
    }
}

const CACHE_CAPACITY: usize = 16;

fn policy_label(policy: CachePolicy) -> &'static str {
    match policy {
        CachePolicy::Off => "off",
        CachePolicy::Last => "last",
        CachePolicy::Lru { .. } => "lru16",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLifetime {
    Cold,
    Warm,
}

impl CacheLifetime {
    pub const ALL: [Self; 2] = [Self::Cold, Self::Warm];

    fn label(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    SharedPhase,
    Diamond,
    Chain,
}

impl Topology {
    pub const ALL: [Self; 3] = [Self::SharedPhase, Self::Diamond, Self::Chain];

    /// TSV label; shared with the asynchronous measurement binary.
    pub fn label(self) -> &'static str {
        match self {
            Self::SharedPhase => "shared_phase",
            Self::Diamond => "diamond",
            Self::Chain => "chain",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramePattern {
    Ordered,
    ShuffledDuplicates,
    LocalRepeat,
}

impl FramePattern {
    pub const ALL: [Self; 3] = [Self::Ordered, Self::ShuffledDuplicates, Self::LocalRepeat];

    /// TSV label; shared with the asynchronous measurement binary.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::ShuffledDuplicates => "shuffled_duplicates",
            Self::LocalRepeat => "local_repeat",
        }
    }

    /// Requested frame index for logical row `index`; the oracle uses the same map.
    fn frame(self, index: usize, frames: usize) -> u64 {
        match self {
            Self::Ordered => index as u64,
            Self::ShuffledDuplicates => ((index * 17 % (frames / 2).max(1)) * 7) as u64,
            Self::LocalRepeat => (((index / 4) * 11) % (frames / 4).max(1)) as u64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SceneConfig {
    pub actors: usize,
    pub frames: usize,
    pub rounds: u64,
    pub periodic: bool,
    pub topology: Topology,
    pub frame_pattern: FramePattern,
}

/// Chain depth shared by the graph and the oracle.
const CHAIN_STEPS: u64 = 64;

fn compute(phase: u64, actor: u64, rounds: u64) -> u64 {
    let mut v = phase.wrapping_add(actor);
    for n in 0..rounds {
        v = black_box(v.rotate_left(7).wrapping_mul(31).wrapping_add(n));
    }
    v
}

/// Ordered `(a, b, a, b)` merge with coefficients `1, 2, 3, 4`.
fn merge_ordered(a: u64, b: u64) -> u64 {
    a.wrapping_add(b.wrapping_mul(2))
        .wrapping_add(a.wrapping_mul(3))
        .wrapping_add(b.wrapping_mul(4))
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

pub struct BackendScene {
    state: GraphState,
    /// Every computation node handled to an evaluator for cache configuration.
    computations: Vec<Node<u64>>,
    actor_nodes: Vec<Node<u64>>,
    frame_map: Vec<u64>,
    oracle: Vec<u64>,
    config: SceneConfig,
}

/// One measured run. `calls` and `eval` are this run's increments (stats are
/// reset at run start; warm caches persist), `planner`/`tasks` come from the
/// run's own plan/driver, and `values` is the full output vector. Non-async
/// backends report default `tasks`. `initialized_evaluators` counts evaluators
/// created inside the observation: 1 for shared Cold, `actors` for Cold
/// EntityParallelDfs, 0 for Warm and empty loads.
pub struct BackendResult {
    pub values: Vec<u64>,
    pub calls: u64,
    pub initialized_evaluators: usize,
    pub planner: PlannerStats,
    pub eval: EvalStats,
    pub tasks: TaskStats,
}

impl BackendResult {
    fn empty() -> Self {
        Self {
            values: Vec::new(),
            calls: 0,
            initialized_evaluators: 0,
            planner: PlannerStats::default(),
            eval: EvalStats::default(),
            tasks: TaskStats::default(),
        }
    }
}

/// Private evaluator ownership; avoids a one-element `Vec` for shared backends.
enum Evaluators {
    Shared(Evaluator),
    PerActor(Vec<Evaluator>),
}

impl Evaluators {
    fn reset_stats(&mut self) {
        match self {
            Self::Shared(evaluator) => {
                evaluator.reset_stats();
            }
            Self::PerActor(evaluators) => {
                for evaluator in evaluators {
                    evaluator.reset_stats();
                }
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::Shared(_) => 1,
            Self::PerActor(evaluators) => evaluators.len(),
        }
    }
}

pub struct PreparedRun<'a> {
    scene: &'a BackendScene,
    backend: Backend,
    cache: CacheMode,
    evaluators: Option<Evaluators>,
}

impl BackendScene {
    pub fn new(config: SceneConfig) -> Self {
        let mut state = GraphState::new();
        let (computations, actor_nodes) = state
            .edit(|e| {
                e.set_default_eq::<u64>();
                let phase = e.map(
                    e.frame(),
                    move |f: &FrameIndex| {
                        if config.periodic { f.0 % 8 } else { f.0 }
                    },
                )?;
                let mut computations = vec![phase];
                let source = match config.topology {
                    Topology::SharedPhase => phase,
                    Topology::Diamond => {
                        let a = e.map(phase, |p: &u64| p.wrapping_mul(3).wrapping_add(1))?;
                        let b = e.map(phase, |p: &u64| p.rotate_left(5) ^ 7)?;
                        computations.push(a);
                        computations.push(b);
                        // Repeated edges keep the ordered coefficient structure.
                        let merged =
                            e.map((a, b, a, b), |x: &u64, y: &u64, z: &u64, w: &u64| {
                                x.wrapping_add(y.wrapping_mul(2))
                                    .wrapping_add(z.wrapping_mul(3))
                                    .wrapping_add(w.wrapping_mul(4))
                            })?;
                        computations.push(merged);
                        merged
                    }
                    Topology::Chain => {
                        let mut current = phase;
                        for k in 1..=CHAIN_STEPS {
                            current =
                                e.map(current, move |x: &u64| x.wrapping_add(k).rotate_left(3))?;
                            computations.push(current);
                        }
                        current
                    }
                };
                let actor_nodes = (0..config.actors)
                    .map(|actor| {
                        e.map(source, move |p: &u64| {
                            compute(*p, actor as u64, config.rounds)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                computations.extend(actor_nodes.iter().copied());
                Ok((computations, actor_nodes))
            })
            .unwrap();

        let frame_map: Vec<u64> = (0..config.frames)
            .map(|index| config.frame_pattern.frame(index, config.frames))
            .collect();
        // Independent oracle: direct expressions and loops, never the graph closures.
        let mut oracle = Vec::with_capacity(frame_map.len() * config.actors);
        for &frame in &frame_map {
            let phase = if config.periodic { frame % 8 } else { frame };
            let source = match config.topology {
                Topology::SharedPhase => phase,
                Topology::Diamond => {
                    let a = phase.wrapping_mul(3).wrapping_add(1);
                    let b = phase.rotate_left(5) ^ 7;
                    merge_ordered(a, b)
                }
                Topology::Chain => {
                    let mut x = phase;
                    for k in 1..=CHAIN_STEPS {
                        x = x.wrapping_add(k).rotate_left(3);
                    }
                    x
                }
            };
            for actor in 0..config.actors {
                let mut v = source.wrapping_add(actor as u64);
                for n in 0..config.rounds {
                    v = v.rotate_left(7).wrapping_mul(31).wrapping_add(n);
                }
                oracle.push(v);
            }
        }

        Self {
            state,
            computations,
            actor_nodes,
            frame_map,
            oracle,
            config,
        }
    }

    /// The scene's measured configuration.
    pub fn config(&self) -> SceneConfig {
        self.config
    }

    /// Expected output for every request in `request_batch`, in request order.
    pub fn oracle(&self) -> &[u64] {
        &self.oracle
    }

    /// Immutable snapshot of the scene graph; shared by every evaluator built
    /// from this scene.
    pub fn snapshot(&self) -> EvaluationSnapshot {
        self.state.snapshot()
    }

    /// One evaluator with all computations configured for `cache`. Raw frames
    /// and inputs are never cached. Public for standalone binaries that drive
    /// their own planner/AsyncDriver loops.
    pub fn new_evaluator(&self, cache: CacheMode) -> Evaluator {
        self.evaluator(cache)
    }

    /// One request per actor per frame in scene order, plus the matching
    /// request ids. Public for standalone binaries; `verify` consumes the
    /// results in this order against `oracle`.
    pub fn request_batch(&self) -> (RequestBatch, Vec<RequestId<u64>>) {
        let mut batch = RequestBatch::new();
        let mut ids = Vec::with_capacity(self.frame_map.len() * self.actor_nodes.len());
        for &frame in &self.frame_map {
            for &node in &self.actor_nodes {
                ids.push(batch.at(FrameIndex(frame)).request(node));
            }
        }
        (batch, ids)
    }

    /// Panics unless the run's outputs equal the oracle item by item.
    pub fn verify(&self, result: &BackendResult) {
        assert_eq!(
            result.values, self.oracle,
            "backend output differs from oracle"
        );
    }

    fn is_empty(&self) -> bool {
        self.config.actors == 0 || self.config.frames == 0
    }

    fn evaluator(&self, cache: CacheMode) -> Evaluator {
        let mut evaluator = Evaluator::new(&self.state);
        let config = cache.config();
        for &node in &self.computations {
            evaluator.configure_cache(node, config).unwrap();
        }
        evaluator
    }

    fn build_evaluators(&self, backend: Backend, cache: CacheMode) -> Evaluators {
        match backend {
            Backend::EntityParallelDfs => Evaluators::PerActor(
                (0..self.config.actors)
                    .map(|_| self.evaluator(cache))
                    .collect(),
            ),
            _ => Evaluators::Shared(self.evaluator(cache)),
        }
    }

    /// Builds one prepared run. Cold keeps no evaluator state: `run` creates
    /// and destroys evaluators inside the observation. Warm preheats once here
    /// (outside observation, on the given pool), discards preheat results and
    /// resets statistics, so `run` only measures incremental reuse against the
    /// surviving warm caches. Cold and Warm are compared within their own
    /// lifetime only. Empty loads (`actors == 0 || frames == 0`) skip
    /// evaluator creation entirely and report zero statistics. The pool must
    /// come from [`benchmark_pool`] (or be fully initialized) so worker
    /// thread-local setup never lands inside the observation.
    pub fn prepare<'a>(
        &'a self,
        backend: Backend,
        pool: &TaskPool,
        cache: CacheMode,
        lifetime: CacheLifetime,
    ) -> PreparedRun<'a> {
        let evaluators = match lifetime {
            CacheLifetime::Warm if !self.is_empty() => {
                let mut evaluators = self.build_evaluators(backend, cache);
                let _ = self.execute(&mut evaluators, backend, pool, 0);
                evaluators.reset_stats();
                Some(evaluators)
            }
            _ => None,
        };
        PreparedRun {
            scene: self,
            backend,
            cache,
            evaluators,
        }
    }

    fn execute(
        &self,
        evaluators: &mut Evaluators,
        backend: Backend,
        pool: &TaskPool,
        initialized_evaluators: usize,
    ) -> BackendResult {
        evaluators.reset_stats();
        match (backend, evaluators) {
            (Backend::EntityParallelDfs, Evaluators::PerActor(evaluators)) => {
                self.run_entities(evaluators, pool, initialized_evaluators)
            }
            (Backend::EntityParallelDfs, _) => {
                unreachable!("entity backend needs per-actor evaluators")
            }
            (_, Evaluators::Shared(evaluator)) => {
                self.run_shared(evaluator, backend, pool, initialized_evaluators)
            }
            (_, Evaluators::PerActor(_)) => unreachable!("shared backend needs one evaluator"),
        }
    }

    fn run_shared(
        &self,
        evaluator: &mut Evaluator,
        backend: Backend,
        pool: &TaskPool,
        initialized_evaluators: usize,
    ) -> BackendResult {
        let snapshot = self.state.snapshot();
        let (batch, ids) = self.request_batch();
        let (results, planner, tasks) = match backend {
            Backend::Dfs => (
                evaluator.bind(snapshot).unwrap().evaluate(batch).unwrap(),
                PlannerStats::default(),
                TaskStats::default(),
            ),
            Backend::PlannerSerial => {
                let (results, planner) = evaluator
                    .bind(snapshot)
                    .unwrap()
                    .evaluate_planned(batch, planner_config(), &mut SerialExecutor)
                    .unwrap();
                (results, planner, TaskStats::default())
            }
            Backend::PlannerParallel(chunk) => {
                let (results, planner) = evaluator
                    .bind(snapshot)
                    .unwrap()
                    .evaluate_planned(batch, planner_config(), &mut BevyExecutor::new(pool, chunk))
                    .unwrap();
                (results, planner, TaskStats::default())
            }
            Backend::Async(chunk) => self.run_async(evaluator, pool, snapshot, batch, chunk),
            Backend::EntityParallelDfs => unreachable!("entity backend uses run_entities"),
        };
        let eval = evaluator.stats();
        let values = ids.iter().map(|&id| *results.get(id).unwrap()).collect();
        BackendResult {
            values,
            calls: eval.computation_calls,
            initialized_evaluators,
            planner,
            eval,
            tasks,
        }
    }

    /// Nonblocking driver loop bounded by a hang deadline. The measured region ends
    /// once every result is delivered and the logical task quota is reclaimed; it
    /// never waits for physical executor teardown and does not reduce the number of
    /// completion messages sent per logical task.
    fn run_async(
        &self,
        evaluator: &mut Evaluator,
        pool: &TaskPool,
        snapshot: EvaluationSnapshot,
        batch: RequestBatch,
        chunk: usize,
    ) -> (EvaluationResults, PlannerStats, TaskStats) {
        let mut driver = AsyncDriver::new(snapshot.clone(), ASYNC_LIMITS, chunk);
        driver
            .begin(evaluator, snapshot, batch, planner_config())
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let (results, planner) = loop {
            if let Some(done) = driver.poll(evaluator, pool).unwrap() {
                break done;
            }
            assert!(
                Instant::now() < deadline,
                "asynchronous backend did not deliver results within 30s"
            );
            std::thread::yield_now();
        };
        let tasks = driver.stats();
        assert_eq!(
            driver.in_flight(),
            0,
            "results delivered while logical tasks were still running"
        );
        (results, planner, tasks)
    }

    fn run_entities(
        &self,
        evaluators: &mut [Evaluator],
        pool: &TaskPool,
        initialized_evaluators: usize,
    ) -> BackendResult {
        let per_actor: Vec<(usize, Vec<u64>, EvalStats)> = pool.scope(|scope| {
            // One disjoint mutable borrow per actor, taken directly from the slice;
            // no intermediate container is allocated inside the observation.
            for (actor, evaluator) in evaluators.iter_mut().enumerate() {
                let node = self.actor_nodes[actor];
                let frame_map = &self.frame_map;
                let state = &self.state;
                scope.spawn(async move {
                    let mut batch = RequestBatch::new();
                    let ids: Vec<_> = frame_map
                        .iter()
                        .map(|&frame| batch.at(FrameIndex(frame)).request(node))
                        .collect();
                    let results = evaluator
                        .bind(state.snapshot())
                        .unwrap()
                        .evaluate(batch)
                        .unwrap();
                    let values = ids
                        .into_iter()
                        .map(|id| *results.get(id).unwrap())
                        .collect();
                    (actor, values, evaluator.stats())
                });
            }
        });
        let eval = sum_eval_stats(per_actor.iter().map(|(_, _, stats)| *stats));
        let mut reordered = vec![0_u64; self.frame_map.len() * self.actor_nodes.len()];
        for (actor, values, _) in per_actor {
            for (index, value) in values.into_iter().enumerate() {
                reordered[index * self.actor_nodes.len() + actor] = value;
            }
        }
        BackendResult {
            values: reordered,
            calls: eval.computation_calls,
            initialized_evaluators,
            planner: PlannerStats::default(),
            eval,
            tasks: TaskStats::default(),
        }
    }
}

impl PreparedRun<'_> {
    /// Executes one observed run and returns its result. Warm runs reuse
    /// their preheated evaluators (no evaluators are created inside the
    /// observation); Cold runs create and destroy their evaluators inside the
    /// observation. The observation clock is owned by the observer: request,
    /// evaluation, output reads/retention are all inside its span. For async
    /// runs the reported boundary is logical result delivery plus terminal
    /// reclamation (30s hang guard); physical executor teardown stays outside.
    pub fn run(&mut self, pool: &TaskPool) -> BackendResult {
        let scene = self.scene;
        if scene.is_empty() {
            return BackendResult::empty();
        }
        match &mut self.evaluators {
            // Warm: reuse the preheated evaluators, so no evaluators are
            // created inside observation and warm caches keep serving.
            Some(evaluators) => scene.execute(evaluators, self.backend, pool, 0),
            // Cold: create and destroy the evaluators inside observation.
            None => {
                let mut evaluators = scene.build_evaluators(self.backend, self.cache);
                let initialized = evaluators.count();
                scene.execute(&mut evaluators, self.backend, pool, initialized)
            }
        }
    }
}

fn sum_eval_stats(stats: impl IntoIterator<Item = EvalStats>) -> EvalStats {
    stats
        .into_iter()
        .fold(EvalStats::default(), |total, stats| EvalStats {
            memo_hits: total.memo_hits + stats.memo_hits,
            memo_misses: total.memo_misses + stats.memo_misses,
            memo_comparisons: total.memo_comparisons + stats.memo_comparisons,
            computation_calls: total.computation_calls + stats.computation_calls,
            context_hits: total.context_hits + stats.context_hits,
            context_misses: total.context_misses + stats.context_misses,
            context_comparisons: total.context_comparisons + stats.context_comparisons,
            resolution_visits: total.resolution_visits + stats.resolution_visits,
            definition_syncs: total.definition_syncs + stats.definition_syncs,
            projection_builds: total.projection_builds + stats.projection_builds,
        })
}

// ---------------------------------------------------------------------------
// Pool initialization
// ---------------------------------------------------------------------------

/// Hang-detection bound for worker/handoff waits; never a scheduling delay.
const HANG: Duration = Duration::from_secs(30);

struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

/// Poisoning is recovered instead of propagated, matching the repository's
/// existing `std::sync` convention; no new locking dependency is introduced.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait<'a>(gate: &Gate, open: std::sync::MutexGuard<'a, bool>) -> std::sync::MutexGuard<'a, bool> {
    gate.changed
        .wait(open)
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Gate {
    fn new() -> Self {
        Self {
            open: Mutex::new(false),
            changed: Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut open = lock(&self.open);
        while !*open {
            open = wait(self, open);
        }
    }

    fn release(&self) {
        *lock(&self.open) = true;
        self.changed.notify_all();
    }
}

/// Opens the gate when it drops. Declared after the pool so a failed assertion
/// still releases blocked workers before the pool joins its threads.
struct GateGuard(Arc<Gate>);

impl Drop for GateGuard {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// Four real workers with all caller-visible thread state initialized outside any
/// observation. Each worker is confirmed inside the executor by a gated blocking
/// job, not merely by its thread-spawn callback.
pub fn benchmark_pool() -> TaskPool {
    let (spawned_tx, spawned_rx) = mpsc::channel::<()>();
    let spawned_tx = Arc::new(Mutex::new(spawned_tx));
    let notify = Arc::clone(&spawned_tx);
    let pool = TaskPoolBuilder::new()
        .num_threads(4)
        .on_thread_spawn(move || {
            let _ = lock(&notify).send(());
        })
        .build();
    for _ in 0..4 {
        spawned_rx
            .recv_timeout(HANG)
            .expect("pool worker never entered its thread body");
    }

    let gate = Arc::new(Gate::new());
    let guard = GateGuard(Arc::clone(&gate));
    let (started_tx, started_rx) = mpsc::channel::<ThreadId>();
    let mut handles = Vec::with_capacity(4);
    for _ in 0..4 {
        let gate = Arc::clone(&gate);
        let started_tx = started_tx.clone();
        handles.push(pool.spawn(async move {
            let _ = started_tx.send(std::thread::current().id());
            // Synchronous block: the executor runner is created before this point.
            gate.wait();
        }));
    }
    drop(started_tx);
    // ThreadId is unordered but hashable; distinct ids prove four runners exist.
    let mut threads = HashSet::with_capacity(4);
    for _ in 0..4 {
        threads.insert(
            started_rx
                .recv_timeout(HANG)
                .expect("pool worker never entered the executor"),
        );
    }
    assert_eq!(
        threads.len(),
        4,
        "pool workers did not span four distinct threads"
    );
    gate.release();
    for handle in handles {
        bevy_tasks::block_on(handle);
    }
    // Prime caller-thread-local state the same way a measured run would touch it.
    let _ = pool.scope(|scope| {
        scope.spawn(async {});
    });
    drop(guard);
    pool
}

// ---------------------------------------------------------------------------
// Workload kernel
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_load_returns_zero_statistics_without_work() {
        let pool = benchmark_pool();
        for config in [
            SceneConfig {
                actors: 0,
                frames: 16,
                rounds: 4,
                periodic: true,
                topology: Topology::Chain,
                frame_pattern: FramePattern::Ordered,
            },
            SceneConfig {
                actors: 4,
                frames: 0,
                rounds: 4,
                periodic: false,
                topology: Topology::Diamond,
                frame_pattern: FramePattern::LocalRepeat,
            },
        ] {
            let scene = BackendScene::new(config);
            assert!(scene.oracle().is_empty());
            // Empty load never creates a work evaluator or dispatches a load job:
            // the whole contract is empty values plus zero evaluation, planner and
            // task statistics, for both cache lifetimes and every backend.
            for backend in Backend::ALL {
                for cache in CacheMode::ALL {
                    for lifetime in CacheLifetime::ALL {
                        let mut prepared = scene.prepare(backend, &pool, cache, lifetime);
                        let result = prepared.run(&pool);
                        assert!(result.values.is_empty());
                        assert_eq!(result.calls, 0);
                        assert_eq!(result.initialized_evaluators, 0);
                        assert_eq!(result.eval, EvalStats::default());
                        assert_eq!(result.planner, PlannerStats::default());
                        assert_eq!(result.tasks, TaskStats::default());
                    }
                }
            }
        }
    }
}
