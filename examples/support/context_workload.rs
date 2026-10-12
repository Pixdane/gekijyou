//! P3 context loads reuse the exact P2 scene definitions and direct oracle.
use super::fixture::{Config, Input, ValueComparison, Viewport, Workload, direct_checksum};
use gekijyou::computation::{CachePolicy, EvalStats, Evaluator, FrameIndex};
use std::hint::black_box;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Case {
    Repeat,
    Continuous,
    Scrub,
    EditFirst,
    EditSecond,
    UnrelatedEdit,
    Cold,
}
pub const CASES: [Case; 7] = [
    Case::Repeat,
    Case::Continuous,
    Case::Scrub,
    Case::EditFirst,
    Case::EditSecond,
    Case::UnrelatedEdit,
    Case::Cold,
];
pub const CONFIGS: [(&str, CachePolicy, usize); 5] = [
    ("last_ctx0", CachePolicy::Last, 0),
    ("last_ctx1", CachePolicy::Last, 1),
    ("last_ctx4", CachePolicy::Last, 4),
    ("last_ctx16", CachePolicy::Last, 16),
    ("lru8_ctx0", CachePolicy::Lru { capacity: 8 }, 0),
];
impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Self::Repeat => "repeat",
            Self::Continuous => "continuous",
            Self::Scrub => "scrub3",
            Self::EditFirst => "edit_first",
            Self::EditSecond => "edit_second",
            Self::UnrelatedEdit => "unrelated_edit_first",
            Self::Cold => "cold_first",
        }
    }
    pub fn edits(self) -> bool {
        matches!(
            self,
            Self::EditFirst | Self::EditSecond | Self::UnrelatedEdit
        )
    }
    pub fn datasets(self, contexts: usize, count: usize) -> Vec<Vec<Input>> {
        let variants = match self {
            Self::Continuous => count,
            Self::Scrub => 3,
            _ => 1,
        };
        (0..variants)
            .map(|i| {
                (0..contexts)
                    .map(|index| Input {
                        frame: FrameIndex(match self {
                            Self::Continuous => (i * contexts + index) as u64,
                            Self::Scrub => [30, 60, 90][i],
                            _ => 30,
                        }),
                        viewport: Viewport {
                            width: 1920,
                            height: 1080,
                        },
                        seed: 7,
                    })
                    .collect()
            })
            .collect()
    }
    pub fn replace(self, workload: &mut Workload) {
        if self == Self::UnrelatedEdit {
            workload.replace_unrelated();
        } else {
            workload.replace_phase();
        }
    }
}
pub fn workload(rounds: usize, policy: CachePolicy, capacity: usize) -> Workload {
    let mut workload = Workload::configured(
        Config {
            actors: 32,
            stages: 16,
            rounds,
        },
        policy,
        ValueComparison::Content,
    );
    assert_eq!(workload.node_count(), 1689);
    workload.set_context_capacity(capacity);
    workload
}
pub fn evaluate(workload: &Workload, evaluator: &mut Evaluator, data: &[Input]) -> u64 {
    let output = workload.evaluate(evaluator, workload.prepare(black_box(data)));
    black_box(&output).checksum()
}
pub fn warm(workload: &Workload, case: Case, data: &[Vec<Input>]) -> Evaluator {
    let mut evaluator = workload.evaluator();
    if case != Case::Cold {
        if case == Case::Continuous {
            let mut previous = data[0].clone();
            for input in &mut previous {
                input.frame = FrameIndex(u64::MAX);
            }
            black_box(evaluate(workload, &mut evaluator, &previous));
        } else {
            for d in data {
                black_box(evaluate(workload, &mut evaluator, d));
            }
        }
    }
    evaluator.reset_stats();
    evaluator
}
pub fn verify(workload: &mut Workload, case: Case, datasets: &[Vec<Input>]) {
    let mut evaluator = workload.evaluator();
    let data = if case == Case::Continuous {
        &datasets[..datasets.len().min(8)]
    } else {
        datasets
    };
    for data in data.iter().chain(data) {
        if case.edits() {
            case.replace(workload);
        }
        let output = workload.evaluate(&mut evaluator, workload.prepare(data));
        let direct = workload.direct(data);
        assert_eq!(
            output.snapshots(),
            direct.iter().map(|d| d.snapshot()).collect::<Vec<_>>()
        );
        assert_eq!(output.checksum(), direct_checksum(&direct));
        assert!(output.duplicates_share_storage());
    }
}
pub fn context_rate(stats: EvalStats) -> f64 {
    rate(stats.context_hits, stats.context_misses)
}
pub fn memo_rate(stats: EvalStats) -> f64 {
    rate(stats.memo_hits, stats.memo_misses)
}
fn rate(hits: u64, misses: u64) -> f64 {
    if hits + misses == 0 {
        0.0
    } else {
        hits as f64 / (hits + misses) as f64
    }
}
