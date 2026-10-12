//! Run with: cargo run --release --example scene_bench
//! No timing thresholds are asserted; this executable reports local measurements.
use super::fixture::{Config, Input, Workload, direct_checksum, inputs};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
enum Mode {
    Direct,
    GraphEvaluate,
    GraphEndToEnd,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::GraphEvaluate => "graph_evaluate",
            Self::GraphEndToEnd => "graph_end_to_end",
        }
    }
}

fn measure(
    workload: &Workload,
    datasets: &[Vec<Input>],
    iterations: usize,
    mode: Mode,
) -> Duration {
    let mut evaluator = workload.evaluator();
    // evaluate_batch consumes its input: recreate all batches outside the timer
    // for evaluate-only measurements. No evaluated result is reused.
    let prepared = if matches!(mode, Mode::GraphEvaluate) {
        (0..iterations)
            .map(|i| workload.prepare(&datasets[i % datasets.len()]))
            .collect()
    } else {
        Vec::new()
    };
    let mut sink = 0_u64;
    let start = Instant::now();
    match mode {
        Mode::Direct => {
            for i in 0..iterations {
                let result = workload.direct(black_box(&datasets[i % datasets.len()]));
                sink = sink.wrapping_add(direct_checksum(black_box(&result)));
                // The result, including its Arcs, is dropped inside the timer.
            }
        }
        Mode::GraphEvaluate => {
            for batch in prepared {
                let result = workload.evaluate(&mut evaluator, black_box(batch));
                sink = sink.wrapping_add(black_box(&result).checksum());
            }
        }
        Mode::GraphEndToEnd => {
            for i in 0..iterations {
                let batch = workload.prepare(black_box(&datasets[i % datasets.len()]));
                let result = workload.evaluate(&mut evaluator, batch);
                sink = sink.wrapping_add(black_box(&result).checksum());
            }
        }
    }
    black_box(sink);
    start.elapsed()
}

fn calibrate(workload: &Workload, datasets: &[Vec<Input>], mode: Mode, target: Duration) -> usize {
    // Cap prepared-context memory; direct measurements do not prebuild batches.
    let cap = if matches!(mode, Mode::Direct) {
        65_536
    } else {
        (4096 / datasets[0].len()).max(1)
    };
    let mut iterations = 1;
    loop {
        let elapsed = measure(workload, datasets, iterations, mode);
        if elapsed >= target || iterations == cap {
            return iterations;
        }
        let multiplier = (target.as_nanos() / elapsed.as_nanos().max(1)).clamp(2, 16) as usize;
        iterations = iterations.saturating_mul(multiplier).min(cap);
    }
}

pub fn run(options: super::bench_options::TimingOptions) {
    let super::bench_options::TimingOptions { samples, target } = options;
    println!(
        "# actors=32 stages=16 requests_per_context=6 samples={samples} target_sample_ms={}",
        target.as_millis()
    );
    println!(
        "# Times include typed result reads, checksum and result destruction; graph_evaluate excludes Context/Batch construction."
    );
    println!(
        "rounds\tcontexts\tnodes\tmode\titerations\tmedian_batch_us\tmin_batch_us\tmax_batch_us\tmedian_context_us\tvs_direct"
    );
    for rounds in [0, 128] {
        let workload = Workload::new(Config {
            actors: 32,
            stages: 16,
            rounds,
        });
        for contexts in [1, 16, 64] {
            let datasets: Vec<_> = (0..32).map(|variant| inputs(contexts, variant)).collect();
            // Exact complete-value comparison before timing, not merely a hash.
            let mut evaluator = workload.evaluator();
            for data in &datasets {
                let graph = workload.evaluate(&mut evaluator, workload.prepare(data));
                let direct = workload.direct(data);
                let expected: Vec<_> = direct.iter().map(|output| output.snapshot()).collect();
                assert_eq!(graph.snapshots(), expected);
                assert_eq!(graph.checksum(), direct_checksum(&direct));
                assert!(graph.duplicates_share_storage());
            }
            let modes = [Mode::Direct, Mode::GraphEvaluate, Mode::GraphEndToEnd];
            let iterations: Vec<_> = modes
                .iter()
                .map(|&mode| calibrate(&workload, &datasets, mode, target))
                .collect();
            let mut timings = [Vec::new(), Vec::new(), Vec::new()];
            for sample in 0..samples {
                // Rotate measurement order to reduce a systematic warmup/thermal bias.
                for offset in 0..modes.len() {
                    let index = (sample + offset) % modes.len();
                    let elapsed = measure(&workload, &datasets, iterations[index], modes[index]);
                    timings[index]
                        .push(elapsed.as_secs_f64() * 1_000_000.0 / iterations[index] as f64);
                }
            }
            for timing in &mut timings {
                timing.sort_by(f64::total_cmp);
            }
            let direct_median = timings[0][samples / 2];
            for (index, mode) in modes.into_iter().enumerate() {
                let values = &timings[index];
                let median = values[samples / 2];
                println!(
                    "{rounds}\t{contexts}\t{}\t{}\t{}\t{median:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                    workload.node_count(),
                    mode.name(),
                    iterations[index],
                    values[0],
                    values[samples - 1],
                    median / contexts as f64,
                    median / direct_median
                );
            }
        }
    }
}
