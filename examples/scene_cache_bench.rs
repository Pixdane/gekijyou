//! Run with --release -- --samples 9 --sample-ms 40.
//! Compares cache policies over four scene-state workloads.
#[path = "support/bench_options.rs"]
mod bench_options;
#[path = "support/scene_workload.rs"]
mod fixture;
#[path = "support/memo_workload.rs"]
mod memo;

use fixture::{Config, ValueComparison, Workload};
use gekijyou::computation::EvalStats;
use memo::{CASES, Case, POLICIES};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

struct Measurement {
    elapsed: Duration,
    edits: Duration,
    stats: EvalStats,
}

fn measure(
    workload: &mut Workload,
    case: Case,
    data: &[Vec<fixture::Input>],
    iterations: usize,
) -> Measurement {
    let mut evaluator = memo::warm(workload, case, data);
    let mut edits = Duration::ZERO;
    let mut checksum = 0_u64;
    let start = Instant::now();
    for i in 0..iterations {
        if case == Case::GraphEdit {
            let edit_start = Instant::now();
            workload.replace_phase();
            edits += edit_start.elapsed();
        }
        checksum = checksum.wrapping_add(memo::evaluate(
            workload,
            &mut evaluator,
            &data[i % data.len()],
        ));
    }
    let elapsed = start.elapsed();
    black_box(checksum);
    // Warmed Evaluator construction and final destruction are outside timing.
    Measurement {
        elapsed,
        edits,
        stats: evaluator.stats(),
    }
}

fn main() {
    let bench_options::TimingOptions { samples, target } = bench_options::TimingOptions::from_env();
    println!(
        "# Cache workloads: 32 actors x 16 stages, exact content semantics set before node creation."
    );
    println!(
        "# End-to-end includes Context/Batch creation, reads, checksum, result drop. Warmup excluded."
    );
    println!(
        "# graph_edit also includes replace validation; edit_us reports that component. Transient has no memo queries."
    );
    println!(
        "case\trounds\tcontexts\tpolicy\titerations\tmedian_batch_us\tmin_batch_us\tmax_batch_us\tmedian_edit_us\tcomputation_calls_per_batch\thits_per_batch\tmisses_per_batch\thit_rate\tcomparisons_per_batch"
    );
    for rounds in [0, 128] {
        for contexts in [1, 64] {
            for case in CASES {
                let data = case.datasets(contexts);
                let mut workloads: Vec<_> = POLICIES
                    .iter()
                    .map(|(_, policy)| {
                        Workload::configured(
                            Config {
                                actors: 32,
                                stages: 16,
                                rounds,
                            },
                            *policy,
                            ValueComparison::Content,
                        )
                    })
                    .collect();
                for workload in &mut workloads {
                    assert_eq!(workload.node_count(), 1689);
                    memo::verify(workload, case, &data);
                }
                // Use the same multiple-of-32 batch count for each policy, ensuring
                // identical input distribution, miss cycles and history access.
                let trial = measure(&mut workloads[0], case, &data, 32).elapsed;
                let cap = (4096 / contexts).max(32);
                let iterations = ((target.as_secs_f64() / trial.as_secs_f64()).ceil() as usize
                    * 32)
                    .clamp(32, cap);
                let mut measurements: [Vec<Measurement>; 3] = std::array::from_fn(|_| Vec::new());
                for sample in 0..samples {
                    // Rotate policies to reduce systematic order/thermal bias.
                    for offset in 0..3 {
                        let index = (sample + offset) % 3;
                        measurements[index].push(measure(
                            &mut workloads[index],
                            case,
                            &data,
                            iterations,
                        ));
                    }
                }
                for (index, (policy, _)) in POLICIES.iter().enumerate() {
                    let values = &mut measurements[index];
                    values.sort_by_key(|m| m.elapsed);
                    let median = &values[samples / 2];
                    let per_batch = |count: u64| count as f64 / iterations as f64;
                    let us = |time: Duration| time.as_secs_f64() * 1e6 / iterations as f64;
                    println!(
                        "{}\t{rounds}\t{contexts}\t{policy}\t{iterations}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.3}",
                        case.name(),
                        us(median.elapsed),
                        us(values[0].elapsed),
                        us(values[samples - 1].elapsed),
                        us(median.edits),
                        per_batch(median.stats.computation_calls),
                        per_batch(median.stats.memo_hits),
                        per_batch(median.stats.memo_misses),
                        memo::hit_rate(median.stats),
                        per_batch(median.stats.memo_comparisons)
                    );
                }
            }
        }
    }
}
