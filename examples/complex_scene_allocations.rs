//! Allocation-only P3 instrumentation, separate from all timing runs.
#[path = "support/context_workload.rs"]
mod context_load;
#[path = "support/counting_allocator.rs"]
mod counting;
#[path = "support/scene_workload.rs"]
mod fixture;
use context_load::{CASES, CONFIGS, Case};
use std::hint::black_box;

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("Run with --release");
        std::process::exit(2);
    }
    println!(
        "# Successful alloc/alloc_zeroed, realloc separately; cumulative requested bytes include each realloc new_size, not live/peak bytes."
    );
    println!(
        "# Graph/data/warmup outside tracking. Edit rows exclude replace; edit_second also excludes the first evaluation."
    );
    println!(
        "# cold_first starts empty, warm32 is preloaded. repeat_evaluate_only prepares consumed batches outside tracking."
    );
    println!(
        "# For fully memo/context-hit repeat_evaluate_only, temporary_map_bytes subtracts the two fixed outer/result Vec buffers from evaluation bytes."
    );
    println!(
        "p3_case\trounds\tcontexts\tconfig\tstate\tbatches\tallocs_per_batch\treallocs_per_batch\tdeallocs_per_batch\tbytes_per_batch\tcontext_hit_rate\tmemo_hit_rate\tresolution_visits_per_batch\tcomputation_calls_per_batch\ttemporary_map_bytes_per_batch\treplace_allocs_per_batch\treplace_reallocs_per_batch\treplace_bytes_per_batch"
    );
    for rounds in [0, 128] {
        for contexts in [1, 64] {
            for case in CASES {
                let data = case.datasets(contexts, 32);
                for (name, policy, capacity) in CONFIGS {
                    let mut workload = context_load::workload(rounds, policy, capacity);
                    context_load::verify(&mut workload, case, &data);
                    let mut evaluator = context_load::warm(&workload, case, &data);
                    let batches = if case == Case::Cold { 1 } else { 32 };
                    let mut counters = [0_usize; 4];
                    let mut edit_counters = [0_usize; 4];
                    let mut totals = gekijyou::computation::EvalStats::default();
                    for i in 0..batches {
                        if case.edits() {
                            // Allocation scope matches evaluation timing, not validation.
                            let counts = counting::count(|| {
                                case.replace(&mut workload);
                                0
                            });
                            for (sum, n) in edit_counters.iter_mut().zip(counts) {
                                *sum += n;
                            }
                            if case == Case::EditSecond {
                                black_box(context_load::evaluate(
                                    &workload,
                                    &mut evaluator,
                                    &data[0],
                                ));
                                evaluator.reset_stats();
                            }
                        }
                        let counts = counting::count(|| {
                            context_load::evaluate(&workload, &mut evaluator, &data[i % data.len()])
                        });
                        for (sum, n) in counters.iter_mut().zip(counts) {
                            *sum += n;
                        }
                        // Capture executed counters each time rather than inferring counts.
                        let stats = evaluator.reset_stats();
                        accumulate_stats(&mut totals, stats);
                    }
                    let stats = std::mem::take(&mut totals);
                    let average = |n: usize| n as f64 / batches as f64;
                    println!(
                        "{}\t{rounds}\t{contexts}\t{name}\t{}\t{batches}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.6}\t{:.3}\t{:.3}\tNA\t{:.3}\t{:.3}\t{:.3}",
                        case.name(),
                        if case == Case::Cold {
                            "cold1"
                        } else {
                            "warm32"
                        },
                        average(counters[0]),
                        average(counters[1]),
                        average(counters[2]),
                        average(counters[3]),
                        context_load::context_rate(stats),
                        context_load::memo_rate(stats),
                        stats.resolution_visits as f64 / batches as f64,
                        stats.computation_calls as f64 / batches as f64,
                        average(edit_counters[0]),
                        average(edit_counters[1]),
                        average(edit_counters[3])
                    );
                    if case == Case::Repeat {
                        let prepared: Vec<_> =
                            (0..32).map(|_| workload.prepare(&data[0])).collect();
                        let counts = counting::count(|| {
                            let mut checksum = 0_u64;
                            for batch in prepared {
                                checksum = checksum.wrapping_add(
                                    workload.evaluate(&mut evaluator, batch).checksum(),
                                );
                            }
                            checksum
                        });
                        let stats = evaluator.reset_stats();
                        assert_eq!(stats.computation_calls, 0);
                        // With all caches preloaded, evaluation allocates only outer per-
                        // context maps, output Arc vector, and dense temporary map buffers.
                        type Value = std::sync::Arc<dyn std::any::Any + Send + Sync>;
                        let fixed = contexts
                            * (std::mem::size_of::<
                                slotmap::SecondaryMap<gekijyou::computation::NodeId, Value>,
                            >() + 6 * std::mem::size_of::<Value>());
                        let temporary_bytes = counts[3]
                            .checked_sub(32 * fixed)
                            .expect("Fixed buffers allocated");
                        if capacity != 0 {
                            let empty_map_bytes = counting::count(|| {
                                black_box(slotmap::SecondaryMap::<
                                    gekijyou::computation::NodeId,
                                    Value,
                                >::new());
                                0
                            })[3];
                            assert_eq!(temporary_bytes, 32 * contexts * empty_map_bytes);
                        }
                        println!(
                            "repeat_evaluate_only\t{rounds}\t{contexts}\t{name}\twarm32\t32\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.6}\t{:.3}\t0.000\t{:.3}\t0.000\t0.000\t0.000",
                            counts[0] as f64 / 32.0,
                            counts[1] as f64 / 32.0,
                            counts[2] as f64 / 32.0,
                            counts[3] as f64 / 32.0,
                            context_load::context_rate(stats),
                            context_load::memo_rate(stats),
                            stats.resolution_visits as f64 / 32.0,
                            temporary_bytes as f64 / 32.0
                        );
                    }
                }
            }
        }
    }
}
fn accumulate_stats(s: &mut gekijyou::computation::EvalStats, t: gekijyou::computation::EvalStats) {
    s.memo_hits += t.memo_hits;
    s.memo_misses += t.memo_misses;
    s.memo_comparisons += t.memo_comparisons;
    s.computation_calls += t.computation_calls;
    s.context_hits += t.context_hits;
    s.context_misses += t.context_misses;
    s.context_comparisons += t.context_comparisons;
    s.resolution_visits += t.resolution_visits;
}
