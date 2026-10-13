//! Current evaluator workloads with independent full-output checks.
//!
//! Run `cargo run --release -p benchmark --bin timing` for wall-clock samples
//! and `cargo run --release -p benchmark --bin allocations` for requested-layout
//! allocation counts and live peaks. Both use the same rotating backend matrix.
//! Run them sequentially on the same machine and toolchain; allocation accounting
//! includes worker threads but excludes allocator metadata and does not measure RSS.
//! The `snapshot_cost` and `bevy_scene` executables separately measure snapshot/edit
//! costs and ECS input extraction, frame preparation and output application.
//!
//! Persistent caches are disabled for backend comparisons. Independent entity DFS
//! creates one evaluator per actor and repeats shared phase work; it uses TaskPool
//! jobs rather than Bevy query parallel iteration. Graph, pool and oracle setup stay
//! outside observation; request construction, evaluator setup and output retention
//! are included. See [`backends`] for the complete workload and measurement boundary.

pub mod backends;
pub mod measurement;
pub mod timing;
