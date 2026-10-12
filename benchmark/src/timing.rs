//! Wall-clock observer with no allocation instrumentation.
use crate::runner::{Observation, Observer};
use std::{hint::black_box, time::Instant};
#[derive(Default)]
pub struct Timer;
impl Observer for Timer {
    fn name(&self) -> &'static str {
        "time"
    }
    fn observe(&mut self, operation: impl FnOnce() -> u64) -> Observation {
        let start = Instant::now();
        black_box(operation());
        Observation {
            elapsed: Some(start.elapsed()),
            allocations: None,
        }
    }
}
