use super::{
    bench_options::TimingOptions,
    context_load::{self, CASES, CONFIGS, Case},
    fixture::{Input, Workload},
};
use gekijyou::computation::EvalStats;
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
    data: &[Vec<Input>],
    iterations: usize,
) -> Measurement {
    let mut evaluator = context_load::warm(workload, case, data);
    let mut edits = Duration::ZERO;
    let mut elapsed = Duration::ZERO;
    let mut total = 0_u64;
    let mut second_stats = EvalStats::default();
    if case.edits() {
        for _ in 0..iterations {
            let start = Instant::now();
            case.replace(workload);
            edits += start.elapsed();
            if case == Case::EditSecond {
                black_box(context_load::evaluate(workload, &mut evaluator, &data[0]));
                evaluator.reset_stats();
            }
            let start = Instant::now();
            total = total.wrapping_add(context_load::evaluate(workload, &mut evaluator, &data[0]));
            elapsed += start.elapsed();
            if case == Case::EditSecond {
                add_stats(&mut second_stats, evaluator.reset_stats());
            }
        }
    } else {
        let start = Instant::now();
        for i in 0..iterations {
            total = total.wrapping_add(context_load::evaluate(
                workload,
                &mut evaluator,
                &data[i % data.len()],
            ));
        }
        elapsed = start.elapsed();
    }
    black_box(total);
    let stats = if case == Case::EditSecond {
        second_stats
    } else {
        evaluator.stats()
    };
    Measurement {
        elapsed,
        edits,
        stats,
    }
}
fn add_stats(s: &mut EvalStats, t: EvalStats) {
    s.memo_hits += t.memo_hits;
    s.memo_misses += t.memo_misses;
    s.memo_comparisons += t.memo_comparisons;
    s.computation_calls += t.computation_calls;
    s.context_hits += t.context_hits;
    s.context_misses += t.context_misses;
    s.context_comparisons += t.context_comparisons;
    s.resolution_visits += t.resolution_visits;
}
pub fn run(options: TimingOptions) {
    let TimingOptions { samples, target } = options;
    println!(
        "# P3: 32 actors x 16 stages, 1689 nodes, 6 requests/context. Five requested output boundaries opt in."
    );
    println!(
        "# End-to-end includes Context/Batch creation, reads/checksum/drop; graph/datasets/warmup/evaluator drop excluded."
    );
    println!(
        "# Edit timings exclude replace (reported separately); edit_second also excludes first evaluation. cold_first: empty evaluator, 1 batch/sample."
    );
    println!(
        "# Continuous frames strictly increase within each sample; scrape timeline = 30/60/90. Stats/rates describe executed queries only (no-query rate=0)."
    );
    println!(
        "p3_case\trounds\tcontexts\tconfig\titerations\tmedian_batch_us\tmin_batch_us\tmax_batch_us\tmedian_replace_us\tcontext_hits_per_batch\tcontext_misses_per_batch\tcontext_hit_rate\tcontext_comparisons_per_batch\tmemo_hits_per_batch\tmemo_misses_per_batch\tmemo_hit_rate\tmemo_comparisons_per_batch\tresolution_visits_per_batch\tcomputation_calls_per_batch"
    );
    for rounds in [0, 128] {
        for contexts in [1, 64] {
            for case in CASES {
                let cap = (4096 / contexts).max(32);
                let data = case.datasets(contexts, cap);
                let mut workloads: Vec<_> = CONFIGS
                    .iter()
                    .map(|(_, p, c)| context_load::workload(rounds, *p, *c))
                    .collect();
                for workload in &mut workloads {
                    context_load::verify(workload, case, &data);
                }
                let iterations = if case == Case::Cold {
                    1
                } else {
                    let trial = measure(&mut workloads[0], case, &data, 32).elapsed;
                    ((target.as_secs_f64() / trial.as_secs_f64()).ceil() as usize * 32)
                        .clamp(32, cap)
                };
                let mut measurements: Vec<Vec<Measurement>> =
                    (0..CONFIGS.len()).map(|_| Vec::new()).collect();
                for sample in 0..samples {
                    for offset in 0..CONFIGS.len() {
                        let i = (sample + offset) % CONFIGS.len();
                        measurements[i].push(measure(&mut workloads[i], case, &data, iterations));
                    }
                }
                for (i, (name, _, _)) in CONFIGS.iter().enumerate() {
                    let values = &mut measurements[i];
                    values.sort_by_key(|m| m.elapsed);
                    let median = &values[samples / 2];
                    let s = median.stats;
                    let avg = |n: u64| n as f64 / iterations as f64;
                    let us = |d: Duration| d.as_secs_f64() * 1e6 / iterations as f64;
                    println!(
                        "{}\t{rounds}\t{contexts}\t{name}\t{iterations}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.3}\t{:.3}\t{:.3}",
                        case.name(),
                        us(median.elapsed),
                        us(values[0].elapsed),
                        us(values[samples - 1].elapsed),
                        us(median.edits),
                        avg(s.context_hits),
                        avg(s.context_misses),
                        context_load::context_rate(s),
                        avg(s.context_comparisons),
                        avg(s.memo_hits),
                        avg(s.memo_misses),
                        context_load::memo_rate(s),
                        avg(s.memo_comparisons),
                        avg(s.resolution_visits),
                        avg(s.computation_calls)
                    );
                }
            }
        }
    }
}
