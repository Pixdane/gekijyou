//! Four P2 loads over the existing 1,689-node scene; no rendering or GPU work.
use super::fixture::{Input, Viewport, Workload, direct_checksum, inputs};
use gekijyou::computation::{CachePolicy, EvalStats, Evaluator, FrameIndex};
use std::hint::black_box;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Case {
    AllMiss,
    HighHit,
    History,
    GraphEdit,
}

pub const CASES: [Case; 4] = [Case::AllMiss, Case::HighHit, Case::History, Case::GraphEdit];
pub const POLICIES: [(&str, CachePolicy); 3] = [
    ("transient", CachePolicy::Transient),
    ("last", CachePolicy::Last),
    ("lru8", CachePolicy::Lru { capacity: 8 }),
];

impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Self::AllMiss => "all_miss",
            Self::HighHit => "high_hit",
            Self::History => "history4",
            Self::GraphEdit => "graph_edit",
        }
    }

    pub fn datasets(self, contexts: usize) -> Vec<Vec<Input>> {
        match self {
            // 32 distinct states exceed every tested cache capacity. Input-dependent
            // nodes miss; the two immutable zero-input nodes can still hit.
            Self::AllMiss => (0..32).map(|variant| inputs(contexts, variant)).collect(),
            _ => (0..4)
                .map(|variant| {
                    (0..contexts)
                        .map(|index| Input {
                            // High-hit/edit vary Frame within one phase. History
                            // alternates four phases, testing old direct-input tuples.
                            frame: FrameIndex(if self == Self::History {
                                120 + variant * 4 + index as u64 % 4
                            } else {
                                120 + (variant + index as u64) % 4
                            }),
                            viewport: Viewport {
                                width: 1920,
                                height: 1080,
                            },
                            seed: 7,
                        })
                        .collect()
                })
                .collect(),
        }
    }
}

pub fn evaluate(workload: &Workload, evaluator: &mut Evaluator, data: &[Input]) -> u64 {
    let batch = workload.prepare(black_box(data));
    let output = workload.evaluate(evaluator, batch);
    black_box(&output).checksum()
}

/// Verify complete values against the unchanged direct implementation, including
/// the changed definition and duplicate-request storage; not just checksums.
pub fn verify(workload: &mut Workload, case: Case, datasets: &[Vec<Input>]) {
    let mut evaluator = workload.evaluator();
    for data in datasets.iter().chain(datasets) {
        if case == Case::GraphEdit {
            workload.replace_phase();
        }
        let output = workload.evaluate(&mut evaluator, workload.prepare(data));
        let direct = workload.direct(data);
        let expected: Vec<_> = direct.iter().map(|v| v.snapshot()).collect();
        assert_eq!(output.snapshots(), expected);
        assert_eq!(output.checksum(), direct_checksum(&direct));
        assert!(output.duplicates_share_storage());
    }
}

/// Preload a cycle of reusable states. Miss loads only preload the previous state.
/// Warmup and graph construction are outside timing/allocation scopes.
pub fn warm(workload: &Workload, case: Case, datasets: &[Vec<Input>]) -> Evaluator {
    let mut evaluator = workload.evaluator();
    let warm_data = if case == Case::AllMiss {
        &datasets[datasets.len() - 1..]
    } else {
        datasets
    };
    for data in warm_data {
        black_box(evaluate(workload, &mut evaluator, data));
    }
    evaluator.reset_stats();
    evaluator
}

pub fn hit_rate(stats: EvalStats) -> f64 {
    let queries = stats.memo_hits + stats.memo_misses;
    if queries == 0 {
        0.0
    } else {
        stats.memo_hits as f64 / queries as f64
    }
}
