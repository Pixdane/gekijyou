//! Workload definitions shared by correctness, timing and allocation runs.
use crate::scene::{Config, Input, ValueComparison, Viewport, Workload, direct_checksum, inputs};
use gekijyou_computation::p35::{CachePolicy, Evaluator, FrameIndex};
use std::hint::black_box;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suite {
    Scene,
    Memo,
    Context,
}
impl Suite {
    pub const ALL: [Self; 3] = [Self::Scene, Self::Memo, Self::Context];
    pub fn name(self) -> &'static str {
        match self {
            Self::Scene => "scene",
            Self::Memo => "memo",
            Self::Context => "context",
        }
    }
    pub fn cases(self) -> &'static [Case] {
        match self {
            Self::Scene => &[Case::AllMiss],
            Self::Memo => &[Case::AllMiss, Case::HighHit, Case::History, Case::GraphEdit],
            Self::Context => &[
                Case::Repeat,
                Case::Continuous,
                Case::Scrub,
                Case::EditFirst,
                Case::EditSecond,
                Case::UnrelatedEdit,
                Case::Cold,
            ],
        }
    }
    pub fn configurations(self) -> &'static [Configuration] {
        match self {
            Self::Scene => &SCENE_CONFIGS,
            Self::Memo => &MEMO_CONFIGS,
            Self::Context => &CONTEXT_CONFIGS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Direct,
    Evaluate,
    EndToEnd,
}
impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Evaluate => "evaluate",
            Self::EndToEnd => "end_to_end",
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Configuration {
    pub name: &'static str,
    pub policy: CachePolicy,
    pub context_capacity: usize,
    pub mode: Mode,
}
const fn configuration(
    name: &'static str,
    policy: CachePolicy,
    context_capacity: usize,
    mode: Mode,
) -> Configuration {
    Configuration {
        name,
        policy,
        context_capacity,
        mode,
    }
}
const SCENE_CONFIGS: [Configuration; 3] = [
    configuration("direct", CachePolicy::Transient, 0, Mode::Direct),
    configuration("evaluate", CachePolicy::Transient, 0, Mode::Evaluate),
    configuration("end_to_end", CachePolicy::Transient, 0, Mode::EndToEnd),
];
const MEMO_CONFIGS: [Configuration; 5] = [
    configuration("transient", CachePolicy::Transient, 0, Mode::EndToEnd),
    configuration("last", CachePolicy::Last, 0, Mode::EndToEnd),
    configuration("lru1", CachePolicy::Lru { capacity: 1 }, 0, Mode::EndToEnd),
    configuration("lru4", CachePolicy::Lru { capacity: 4 }, 0, Mode::EndToEnd),
    configuration("lru8", CachePolicy::Lru { capacity: 8 }, 0, Mode::EndToEnd),
];
const CONTEXT_CONFIGS: [Configuration; 5] = [
    configuration("last_ctx0", CachePolicy::Last, 0, Mode::EndToEnd),
    configuration("last_ctx1", CachePolicy::Last, 1, Mode::EndToEnd),
    configuration("last_ctx4", CachePolicy::Last, 4, Mode::EndToEnd),
    configuration("last_ctx16", CachePolicy::Last, 16, Mode::EndToEnd),
    configuration(
        "lru8_ctx0",
        CachePolicy::Lru { capacity: 8 },
        0,
        Mode::EndToEnd,
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    AllMiss,
    HighHit,
    History,
    GraphEdit,
    Repeat,
    Continuous,
    Scrub,
    EditFirst,
    EditSecond,
    UnrelatedEdit,
    Cold,
}
impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Self::AllMiss => "all_miss",
            Self::HighHit => "high_hit",
            Self::History => "history4",
            Self::GraphEdit => "graph_edit",
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
            Self::GraphEdit | Self::EditFirst | Self::EditSecond | Self::UnrelatedEdit
        )
    }
    pub fn datasets(self, contexts: usize, iterations: usize) -> Vec<Vec<Input>> {
        assert!(contexts > 0 && iterations > 0);
        if self == Self::AllMiss {
            return (0..32).map(|v| inputs(contexts, v)).collect();
        }
        let variants = match self {
            Self::Continuous => iterations,
            Self::Scrub => 3,
            Self::HighHit | Self::History | Self::GraphEdit => 4,
            _ => 1,
        };
        (0..variants)
            .map(|i| {
                (0..contexts)
                    .map(|index| Input {
                        frame: FrameIndex(match self {
                            Self::HighHit | Self::GraphEdit => 120 + (i + index) as u64 % 4,
                            Self::History => 120 + i as u64 * 4 + index as u64 % 4,
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
        assert!(self.edits());
        if self == Self::UnrelatedEdit {
            workload.replace_unrelated();
        } else {
            workload.replace_phase();
        }
    }
    /// Cold starts empty; reusable histories are preloaded. Miss/continuous loads
    /// preload only a predecessor rather than any measured states.
    pub fn warm(self, workload: &Workload, data: &[Vec<Input>]) -> Evaluator {
        let mut evaluator = workload.evaluator();
        match self {
            Self::Cold => {}
            Self::AllMiss => {
                black_box(evaluate(workload, &mut evaluator, data.last().unwrap()));
            }
            Self::Continuous => {
                let mut previous = data[0].clone();
                for input in &mut previous {
                    input.frame = FrameIndex(u64::MAX);
                }
                black_box(evaluate(workload, &mut evaluator, &previous));
            }
            _ => {
                for d in data {
                    black_box(evaluate(workload, &mut evaluator, d));
                }
            }
        }
        evaluator.reset_stats();
        evaluator
    }
    /// Compare complete values, not only a checksum; verify duplicate ownership too.
    /// Every measured dataset is checked before timing, including continuous frames.
    pub fn verify(self, workload: &mut Workload, data: &[Vec<Input>]) {
        let mut evaluator = workload.evaluator();
        for d in data.iter().chain(data) {
            if self.edits() {
                self.replace(workload);
            }
            let output = workload.evaluate(&mut evaluator, workload.prepare(d));
            let direct = workload.direct(d);
            assert_eq!(
                output.snapshots(),
                direct.iter().map(|v| v.snapshot()).collect::<Vec<_>>()
            );
            assert_eq!(output.checksum(), direct_checksum(&direct));
            assert!(output.duplicates_share_storage());
        }
    }
}

pub fn workload(config: Config, selected: Configuration) -> Workload {
    let mut workload = Workload::configured(config, selected.policy, ValueComparison::Content);
    workload.set_context_capacity(selected.context_capacity);
    workload
}
pub fn evaluate(workload: &Workload, evaluator: &mut Evaluator, data: &[Input]) -> u64 {
    workload
        .evaluate(evaluator, workload.prepare(black_box(data)))
        .checksum()
}
