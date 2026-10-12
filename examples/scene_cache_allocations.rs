//! Allocation counts only. This allocator is not used by the CPU profiling binary.
#[path = "support/scene_workload.rs"]
mod fixture;

#[path = "support/memo_workload.rs"]
mod memo;

use fixture::{Config, ValueComparison, Workload, inputs};
use gekijyou::computation::CachePolicy;
use memo::{CASES, Case, POLICIES};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// The wrapper preserves System's allocation contract; tracking itself only uses
// atomics and never allocates or unwinds from an allocator method.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        result
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        result
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, new_size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            REALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.load(Ordering::Relaxed) {
            DEALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count(f: impl FnOnce() -> u64) -> [usize; 4] {
    for counter in [&ALLOCS, &REALLOCS, &DEALLOCS, &BYTES] {
        counter.store(0, Ordering::Relaxed);
    }
    TRACK.store(true, Ordering::Relaxed);
    black_box(f());
    TRACK.store(false, Ordering::Relaxed);
    [
        ALLOCS.load(Ordering::Relaxed),
        REALLOCS.load(Ordering::Relaxed),
        DEALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    ]
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("Run with --release");
        std::process::exit(2);
    }
    println!(
        "# Allocation instrumentation only; no timings reported. Counts include successful alloc/alloc_zeroed and realloc separately."
    );
    println!(
        "# Requested bytes sum allocation sizes and realloc new sizes; not peak memory or RSS. Graph/datasets created outside tracking."
    );
    println!(
        "# warm32 preloads states; cold32 starts with an empty evaluator. graph_edit includes replace allocations."
    );
    println!(
        "case\trounds\tcontexts\tpolicy\tstate\tbatches\tallocs_per_batch\treallocs_per_batch\tdeallocs_per_batch\tbytes_per_batch\tcomputation_calls_per_batch\thit_rate\tcomparisons_per_batch"
    );
    for rounds in [0, 128] {
        for contexts in [1, 64] {
            // Legacy fixture comparison: content equality unregistered, like P1.
            let legacy = Workload::new(Config {
                actors: 32,
                stages: 16,
                rounds,
            });
            assert_eq!(legacy.node_count(), 1689);
            let data = inputs(contexts, 19);
            let direct = legacy.direct(&data);
            black_box(fixture::direct_checksum(&direct));
            let mut evaluator = legacy.evaluator();
            black_box(memo::evaluate(&legacy, &mut evaluator, &data));
            evaluator.reset_stats();
            let counters = count(|| memo::evaluate(&legacy, &mut evaluator, &data));
            let stats = evaluator.stats();
            println!(
                "legacy_identity\t{rounds}\t{contexts}\tlast\twarm1\t1\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}",
                counters[0],
                counters[1],
                counters[2],
                counters[3],
                stats.computation_calls,
                memo::hit_rate(stats),
                stats.memo_comparisons
            );
            for case in CASES {
                let data = case.datasets(contexts);
                for (name, policy) in POLICIES {
                    let mut workload = Workload::configured(
                        Config {
                            actors: 32,
                            stages: 16,
                            rounds,
                        },
                        policy,
                        ValueComparison::Content,
                    );
                    // Full input cycles are verified by the timing example and tests.
                    memo::verify(&mut workload, case, &data[..1]);
                    for state in ["cold32", "warm32"] {
                        let mut evaluator = if state == "cold32" {
                            workload.evaluator()
                        } else {
                            memo::warm(&workload, case, &data)
                        };
                        let counters = count(|| {
                            let mut sum = 0_u64;
                            for i in 0..32 {
                                if case == Case::GraphEdit {
                                    workload.replace_phase();
                                }
                                sum = sum.wrapping_add(memo::evaluate(
                                    &workload,
                                    &mut evaluator,
                                    &data[i % data.len()],
                                ));
                            }
                            sum
                        });
                        let stats = evaluator.stats();
                        let average = |count: usize| count as f64 / 32.0;
                        println!(
                            "{}\t{rounds}\t{contexts}\t{name}\t{state}\t32\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.3}",
                            case.name(),
                            average(counters[0]),
                            average(counters[1]),
                            average(counters[2]),
                            average(counters[3]),
                            stats.computation_calls as f64 / 32.0,
                            memo::hit_rate(stats),
                            stats.memo_comparisons as f64 / 32.0
                        );
                    }
                }
            }
            // Use explicit Transient for a P1-equivalent retention/count comparison.
            let workload = Workload::configured(
                Config {
                    actors: 32,
                    stages: 16,
                    rounds,
                },
                CachePolicy::Transient,
                ValueComparison::Identity,
            );
            let mut evaluator = workload.evaluator();
            let counters = count(|| memo::evaluate(&workload, &mut evaluator, &data));
            println!(
                "legacy_identity\t{rounds}\t{contexts}\ttransient\tcold1\t1\t{}\t{}\t{}\t{}\t{}\t0.000000\t0",
                counters[0],
                counters[1],
                counters[2],
                counters[3],
                evaluator.stats().computation_calls
            );
        }
    }
}
