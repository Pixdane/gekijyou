//! P2 direct-input memo policy benchmark (retained unchanged workloads/output).
#[path = "support/bench_options.rs"]
mod bench_options;
#[path = "support/scene_workload.rs"]
mod fixture;
#[path = "support/memo_workload.rs"]
mod memo;
#[path = "support/memo_timing.rs"]
mod timing;
fn main() {
    timing::run(bench_options::TimingOptions::from_env());
}
