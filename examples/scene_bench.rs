//! Original P1 complex-scene timing workload.
#[path = "support/scene_workload.rs"]
mod fixture;
#[path = "support/scene_timing.rs"]
mod timing;

#[path = "support/bench_options.rs"]
mod bench_options;

fn main() {
    timing::run(bench_options::TimingOptions::from_env());
}
