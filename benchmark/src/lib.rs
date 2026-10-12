//! Reusable workloads and measurement boundaries for the computation engine.
//!
//! The library contains no global allocation instrumentation. The `timing` and
//! `allocations` binaries supply different observers to the same runner.

pub mod options;
pub mod runner;
pub mod scenario;
pub mod scene;
pub mod timing;
