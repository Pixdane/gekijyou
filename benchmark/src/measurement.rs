//! Shared measurement types; instrumentation is confined to each executable.
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Allocations {
    pub allocs: u64,
    pub reallocs: u64,
    pub deallocs: u64,
    pub requested_bytes: u64,
    pub live_delta: i64,
    pub peak_extra: i64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Observation {
    pub elapsed: Option<Duration>,
    pub allocations: Option<Allocations>,
}
/// Observers must execute the closure once and finish their scope before returning.
/// The checksum is consumed inside that scope; setup and reporting stay outside it.
pub trait Observer {
    fn name(&self) -> &'static str;
    fn observe(&mut self, operation: impl FnOnce() -> u64) -> Observation;
}
