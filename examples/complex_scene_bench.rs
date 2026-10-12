//! Retains P2 policy comparisons and adds P3 context projection measurements.
#[path = "support/bench_options.rs"]
mod bench_options;
#[path = "support/context_workload.rs"]
mod context_load;
#[path = "support/context_timing.rs"]
mod context_timing;
#[path = "support/scene_workload.rs"]
mod fixture;
#[path = "support/memo_workload.rs"]
mod memo;
#[path = "support/memo_timing.rs"]
mod p2;
fn main() {
    let options = bench_options::TimingOptions::from_env();
    p2::run(options);
    context_timing::run(options);
}
