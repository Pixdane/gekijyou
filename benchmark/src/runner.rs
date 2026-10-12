//! Common sample lifecycle and raw reporting; observers define what is measured.
use crate::{
    options::Options,
    scenario::{Case, Configuration, Mode, Suite, evaluate, workload},
    scene::{Config, Input, Workload, direct_checksum},
};
use gekijyou_computation::EvalStats;
use std::{
    hint::black_box,
    io::{self, Write},
    time::Duration,
};

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
impl Observation {
    fn add(&mut self, other: Self) {
        if let Some(elapsed) = other.elapsed {
            *self.elapsed.get_or_insert(Duration::ZERO) += elapsed;
        }
        if let Some(b) = other.allocations {
            let a = self.allocations.get_or_insert_with(Allocations::default);
            a.allocs += b.allocs;
            a.reallocs += b.reallocs;
            a.deallocs += b.deallocs;
            a.requested_bytes += b.requested_bytes;
            a.live_delta += b.live_delta;
            a.peak_extra = a.peak_extra.max(b.peak_extra);
        }
    }
}
/// Observers must execute the closure once and finish their scope before returning.
/// The checksum is consumed inside that scope; setup and reporting stay outside it.
pub trait Observer {
    fn name(&self) -> &'static str;
    fn observe(&mut self, operation: impl FnOnce() -> u64) -> Observation;
}
#[derive(Debug)]
pub struct Sample {
    pub evaluation: Observation,
    pub replacement: Observation,
    pub stats: EvalStats,
    pub batches: usize,
    /// A whole loop for ordinary samples, one Batch for edit/cold samples.
    pub peak_scope: &'static str,
}

pub fn measure_sample(
    observer: &mut impl Observer,
    workload: &mut Workload,
    case: Case,
    selected: Configuration,
    data: &[Vec<Input>],
    iterations: usize,
) -> Sample {
    assert!(iterations > 0 && !data.is_empty());
    assert!(case != Case::Continuous || data.len() >= iterations);
    let mut evaluator = case.warm(workload, data);
    let batches = if case == Case::Cold { 1 } else { iterations };
    let mut replacement = Observation::default();
    let mut evaluation = Observation::default();
    let mut stats = EvalStats::default();
    if case.edits() {
        assert_eq!(selected.mode, Mode::EndToEnd);
        for i in 0..batches {
            replacement.add(observer.observe(|| {
                case.replace(workload);
                0
            }));
            if case == Case::EditSecond {
                black_box(evaluate(workload, &mut evaluator, &data[i % data.len()]));
                evaluator.reset_stats();
            }
            evaluation.add(
                observer.observe(|| evaluate(workload, &mut evaluator, &data[i % data.len()])),
            );
            add_stats(&mut stats, evaluator.reset_stats());
        }
    } else {
        // evaluate_batch consumes its inputs: all preparation is outside observation.
        let prepared = if selected.mode == Mode::Evaluate {
            (0..batches)
                .map(|i| workload.prepare(&data[i % data.len()]))
                .collect()
        } else {
            Vec::new()
        };
        evaluation = observer.observe(|| {
            let mut checksum = 0_u64;
            match selected.mode {
                Mode::Direct => {
                    for i in 0..batches {
                        let output = workload.direct(black_box(&data[i % data.len()]));
                        checksum = checksum.wrapping_add(direct_checksum(black_box(&output)));
                    }
                }
                Mode::Evaluate => {
                    for batch in prepared {
                        checksum = checksum.wrapping_add(
                            workload
                                .evaluate(&mut evaluator, black_box(batch))
                                .checksum(),
                        );
                    }
                }
                Mode::EndToEnd => {
                    for i in 0..batches {
                        checksum = checksum.wrapping_add(evaluate(
                            workload,
                            &mut evaluator,
                            &data[i % data.len()],
                        ));
                    }
                }
            }
            checksum
        });
        stats = evaluator.stats();
    }
    Sample {
        evaluation,
        replacement,
        stats,
        batches,
        peak_scope: if case.edits() || case == Case::Cold {
            "batch"
        } else {
            "sample"
        },
    }
}
fn add_stats(a: &mut EvalStats, b: EvalStats) {
    a.memo_hits += b.memo_hits;
    a.memo_misses += b.memo_misses;
    a.memo_comparisons += b.memo_comparisons;
    a.computation_calls += b.computation_calls;
    a.context_hits += b.context_hits;
    a.context_misses += b.context_misses;
    a.context_comparisons += b.context_comparisons;
    a.resolution_visits += b.resolution_visits;
}

/// Verify each workload before measuring; rotate configurations within each sample.
/// Every result row is a raw sample, in execution order, with explicit scope labels.
pub fn run(
    options: &Options,
    observer: &mut impl Observer,
    writer: &mut impl Write,
) -> io::Result<()> {
    options
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    writeln!(
        writer,
        "measurement\tsuite\tcase\tconfig\tmode\trounds\tcontexts\tnodes\tsample\torder\tbatches\tpeak_scope\tbatch_us\treplace_us\tallocs_per_batch\treallocs_per_batch\tdeallocs_per_batch\trequested_bytes_per_batch\tlive_delta_bytes\tpeak_extra_bytes\treplace_allocs_per_batch\treplace_reallocs_per_batch\treplace_bytes_per_batch\tmemo_hits_per_batch\tmemo_misses_per_batch\tmemo_comparisons_per_batch\tcontext_hits_per_batch\tcontext_misses_per_batch\tcontext_comparisons_per_batch\tcomputation_calls_per_batch\tresolution_visits_per_batch"
    )?;
    for &rounds in &options.rounds {
        for &contexts in &options.contexts {
            for &suite in &options.suites {
                let selected: Vec<_> = suite
                    .configurations()
                    .iter()
                    .copied()
                    .filter(|&c| options.selected_configuration(c))
                    .collect();
                if selected.is_empty() {
                    continue;
                }
                for &case in suite.cases().iter().filter(|&&c| options.selected_case(c)) {
                    let data = case.datasets(contexts, options.iterations);
                    let mut workloads: Vec<_> = selected
                        .iter()
                        .map(|&configuration| {
                            workload(
                                Config {
                                    actors: 32,
                                    stages: 16,
                                    rounds,
                                },
                                configuration,
                            )
                        })
                        .collect();
                    for workload in &mut workloads {
                        assert_eq!(workload.node_count(), 1689);
                        case.verify(workload, &data);
                    }
                    for sample in 0..options.samples {
                        for order in 0..selected.len() {
                            let index = (sample + order) % selected.len();
                            let measured = measure_sample(
                                observer,
                                &mut workloads[index],
                                case,
                                selected[index],
                                &data,
                                options.iterations,
                            );
                            write_sample(
                                writer,
                                observer.name(),
                                suite,
                                case,
                                selected[index],
                                rounds,
                                contexts,
                                workloads[index].node_count(),
                                sample,
                                order,
                                &measured,
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn write_sample(
    writer: &mut impl Write,
    measurement: &str,
    suite: Suite,
    case: Case,
    configuration: Configuration,
    rounds: usize,
    contexts: usize,
    nodes: usize,
    sample: usize,
    order: usize,
    result: &Sample,
) -> io::Result<()> {
    let per = |n: u64| format!("{:.6}", n as f64 / result.batches as f64);
    let duration = |d: Option<Duration>| {
        d.map_or_else(
            || "NA".into(),
            |d| format!("{:.6}", d.as_secs_f64() * 1e6 / result.batches as f64),
        )
    };
    let allocations = result.evaluation.allocations;
    let flow =
        |field: fn(Allocations) -> u64| allocations.map_or_else(|| "NA".into(), |a| per(field(a)));
    let live = |field: fn(Allocations) -> i64| {
        allocations.map_or_else(|| "NA".into(), |a| field(a).to_string())
    };
    let edit = |field: fn(Allocations) -> u64| {
        result.replacement.allocations.map_or_else(
            || {
                if allocations.is_some() {
                    "0.000000".into()
                } else {
                    "NA".into()
                }
            },
            |a| per(field(a)),
        )
    };
    let s = result.stats;
    writeln!(
        writer,
        "{measurement}\t{}\t{}\t{}\t{}\t{rounds}\t{contexts}\t{nodes}\t{sample}\t{order}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        suite.name(),
        case.name(),
        configuration.name,
        configuration.mode.name(),
        result.batches,
        result.peak_scope,
        duration(result.evaluation.elapsed),
        duration(if case.edits() {
            result.replacement.elapsed
        } else {
            result.evaluation.elapsed.map(|_| Duration::ZERO)
        }),
        flow(|a| a.allocs),
        flow(|a| a.reallocs),
        flow(|a| a.deallocs),
        flow(|a| a.requested_bytes),
        live(|a| a.live_delta),
        live(|a| a.peak_extra),
        edit(|a| a.allocs),
        edit(|a| a.reallocs),
        edit(|a| a.requested_bytes),
        per(s.memo_hits),
        per(s.memo_misses),
        per(s.memo_comparisons),
        per(s.context_hits),
        per(s.context_misses),
        per(s.context_comparisons),
        per(s.computation_calls),
        per(s.resolution_visits)
    )
}
