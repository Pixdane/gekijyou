//! Explicit asynchronous execution measurements, separate from ECS adaptation.
//!
//! The `complete` case measures begin/poll/total latency over the same load matrix
//! as the backend comparison: the graph, request batch and independent oracle come
//! from [`BackendScene`], every configuration is cold, and each sample gets a new
//! evaluator, driver and [`benchmark_pool`]. Timing stops when the complete result
//! set is delivered and the logical task quota is reclaimed; it excludes physical
//! executor teardown (the pool is joined and dropped before the next sample) and it
//! does not claim a lower completion-message count.
//!
//! Three gated cases follow the same row format. Their `total_ns` also waits on an
//! explicit gate or on failure reclamation, so it is diagnostic only and never used
//! for speed ranking; their `measurement_boundary` column reads `gated_diagnostic`
//! so no row is ever compared against the result-completion boundary of the
//! complete matrix, and the memoized replacement samples are labelled `warm`:
//! `cancel_replace` reuses the original cancellation loop, `cached_replacement_zero_slots`
//! delivers a memoized chain replacement while a gated worker still holds the only
//! slot, and `single_task_panic` fails one task of four and then completes a safe
//! sub-batch on the same driver. Every gate is released by an RAII releaser that is
//! declared after the pool and armed before the first fallible wait, so a failed
//! assertion still frees the workers before the pool joins.
//!
//! Full independent output checks run after each timed region. Run with `--release`.

use benchmark::backends::{
    BackendScene, CacheMode, FramePattern, SceneConfig, Topology, benchmark_pool,
};
use bevy_tasks::TaskPool;
use gekijyou_bevy::{AsyncDriver, RequestGeneration, TaskLimits, TaskStats};
use gekijyou_computation::{
    CachePolicy, EvalError, EvaluationResults, Evaluator, FrameIndex, GraphState, NodeCacheConfig,
    PlannerConfig, RequestBatch,
};
use std::{
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};

const ACTORS: usize = 32;
const FRAMES: usize = 64;
const SAMPLES: usize = 9;
const ROUNDS: [u64; 2] = [0, 20000];
const CHUNKS: [usize; 3] = [1, 4, 16];
/// Logical quota for the matrix case and for the panic case.
const QUOTA: TaskLimits = TaskLimits {
    max_in_flight: 64,
    max_pending: 64,
};
/// Hang-detection bound for every wait; never a scheduling delay.
const HANG: Duration = Duration::from_secs(30);
/// Boundary of the complete matrix: results are delivered and the logical quota
/// is reclaimed, without physical executor teardown.
const ASYNC_BOUNDARY: &str = "async_logical_results_cold";
/// Boundary of the gated cases: their totals also wait on a gate or on failure
/// reclamation, so they are diagnostics rather than a result-completion boundary.
const GATED_BOUNDARY: &str = "gated_diagnostic";

fn planner_config() -> PlannerConfig {
    PlannerConfig {
        frame_window: 16,
        max_in_flight: 64,
        comparison_budget: 64,
        ..Default::default()
    }
}

fn off_off() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Off,
        context: CachePolicy::Off,
    }
}

fn last_off() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Last,
        context: CachePolicy::Off,
    }
}

// ---------------------------------------------------------------------------
// Gate with an RAII releaser
// ---------------------------------------------------------------------------

struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

/// Poisoning is recovered instead of propagated, matching the repository's
/// existing `std::sync` convention; no extra locking dependency is introduced.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Gate {
    fn new() -> Self {
        Self {
            open: Mutex::new(false),
            changed: Condvar::new(),
        }
    }

    fn is_open(&self) -> bool {
        *lock(&self.open)
    }

    fn wait(&self) {
        let mut open = lock(&self.open);
        while !*open {
            open = self
                .changed
                .wait(open)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn release(&self) {
        *lock(&self.open) = true;
        self.changed.notify_all();
    }
}

/// Opens the gate when it drops. Declared after every pool so that a failed
/// assertion still releases blocked workers before the pool joins its threads.
struct GateReleaser(Arc<Gate>);

impl Drop for GateReleaser {
    fn drop(&mut self) {
        self.0.release();
    }
}

// ---------------------------------------------------------------------------
// Unified row format
// ---------------------------------------------------------------------------

/// Factors shared by every case; gated cases report `n/a` for matrix factors.
struct Config<'a> {
    topology: &'a str,
    frame_pattern: &'a str,
    periodic: bool,
    memo: &'a str,
    context: &'a str,
    cache_lifetime: &'a str,
    measurement_boundary: &'a str,
    frame_window: usize,
    max_in_flight: usize,
    max_pending: usize,
    chunk_size: usize,
}

struct Row<'a> {
    sample: usize,
    case: &'a str,
    rounds: u64,
    actors: usize,
    frames: usize,
    begin_ns: u128,
    cancel_ns: u128,
    poll_ns: u128,
    total_ns: u128,
    calls: u64,
    config: Config<'a>,
    stats: TaskStats,
}

const HEADER: &str = "sample\tcase\trounds\tactors\tframes\tbegin_ns\tcancel_ns\tpoll_ns\ttotal_ns\tdispatched\tterminal\tobsolete\tcanceled_pending\tpeak_in_flight\tcalls\ttopology\tframe_pattern\tperiodic\tmemo\tcontext\tcache_lifetime\tmeasurement_boundary\tframe_window\tmax_in_flight\tmax_pending\tchunk_size\tworker_jobs_spawned\tfailed";

fn print_row(row: &Row<'_>) {
    let stats = row.stats;
    let config = &row.config;
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        row.sample,
        row.case,
        row.rounds,
        row.actors,
        row.frames,
        row.begin_ns,
        row.cancel_ns,
        row.poll_ns,
        row.total_ns,
        stats.dispatched,
        stats.terminal,
        stats.obsolete,
        stats.canceled_pending,
        stats.peak_in_flight,
        row.calls,
        config.topology,
        config.frame_pattern,
        config.periodic,
        config.memo,
        config.context,
        config.cache_lifetime,
        config.measurement_boundary,
        config.frame_window,
        config.max_in_flight,
        config.max_pending,
        config.chunk_size,
        stats.worker_jobs_spawned,
        stats.failed,
    );
}

/// Polls until the active request publishes its results; the deadline only
/// detects a hang.
fn drive_to_completion(
    driver: &mut AsyncDriver,
    evaluator: &mut Evaluator,
    pool: &TaskPool,
    message: &str,
) -> (u128, EvaluationResults) {
    let mut poll_ns = 0;
    let deadline = Instant::now() + HANG;
    loop {
        let tick = Instant::now();
        let outcome = driver
            .poll(evaluator, pool)
            .expect("async sample failed unexpectedly");
        poll_ns += tick.elapsed().as_nanos();
        if let Some((results, _)) = outcome {
            return (poll_ns, results);
        }
        assert!(Instant::now() < deadline, "{message}");
        std::thread::yield_now();
    }
}

// ---------------------------------------------------------------------------
// Case: complete
// ---------------------------------------------------------------------------

fn complete_samples() {
    for topology in Topology::ALL {
        for frame_pattern in FramePattern::ALL {
            for periodic in [false, true] {
                for rounds in ROUNDS {
                    for cache in CacheMode::ALL {
                        let scene = BackendScene::new(SceneConfig {
                            actors: ACTORS,
                            frames: FRAMES,
                            rounds,
                            periodic,
                            topology,
                            frame_pattern,
                        });
                        for chunk in CHUNKS {
                            for sample in 0..SAMPLES {
                                let pool = benchmark_pool();
                                let mut evaluator = scene.new_evaluator(cache);
                                let mut driver = AsyncDriver::new(scene.snapshot(), QUOTA, chunk);
                                let (batch, ids) = scene.request_batch();
                                let start = Instant::now();
                                let tick = Instant::now();
                                driver
                                    .begin(
                                        &mut evaluator,
                                        scene.snapshot(),
                                        batch,
                                        planner_config(),
                                    )
                                    .unwrap();
                                let begin_ns = tick.elapsed().as_nanos();
                                let (poll_ns, results) = drive_to_completion(
                                    &mut driver,
                                    &mut evaluator,
                                    &pool,
                                    "complete sample stalled",
                                );
                                let total_ns = start.elapsed().as_nanos();
                                let stats = driver.stats();
                                assert_eq!(
                                    driver.in_flight(),
                                    0,
                                    "complete sample left logical tasks running"
                                );
                                let calls = evaluator.stats().computation_calls;
                                drop(driver);
                                // Join the pool before verifying and before the next sample.
                                drop(pool);
                                let values: Vec<u64> =
                                    ids.iter().map(|&id| *results.get(id).unwrap()).collect();
                                assert_eq!(
                                    values,
                                    scene.oracle(),
                                    "async complete sample differed from the oracle"
                                );
                                assert_eq!(stats.failed, 0);
                                assert_eq!(stats.terminal, stats.dispatched);
                                assert!(stats.peak_in_flight <= QUOTA.max_in_flight);
                                print_row(&Row {
                                    sample,
                                    case: "complete",
                                    rounds,
                                    actors: ACTORS,
                                    frames: FRAMES,
                                    begin_ns,
                                    cancel_ns: 0,
                                    poll_ns,
                                    total_ns,
                                    calls,
                                    config: Config {
                                        topology: topology.label(),
                                        frame_pattern: frame_pattern.label(),
                                        periodic,
                                        memo: cache.memo(),
                                        context: cache.context(),
                                        cache_lifetime: "cold",
                                        measurement_boundary: ASYNC_BOUNDARY,
                                        frame_window: 16,
                                        max_in_flight: QUOTA.max_in_flight,
                                        max_pending: QUOTA.max_pending,
                                        chunk_size: chunk,
                                    },
                                    stats,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Case: cancel_replace
// ---------------------------------------------------------------------------

fn cancellation_samples() {
    for sample in 0..SAMPLES {
        let pool = benchmark_pool();
        let gate = Arc::new(Gate::new());
        let _releaser = GateReleaser(Arc::clone(&gate));
        let (start_tx, start_rx) = mpsc::channel();
        let mut state = GraphState::new();
        let node = state
            .edit(|e| {
                let gate = Arc::clone(&gate);
                let start_tx = start_tx.clone();
                e.map((), move || {
                    if !gate.is_open() {
                        let _ = start_tx.send(());
                    }
                    gate.wait();
                    1_u64
                })
            })
            .unwrap();
        let mut evaluator = Evaluator::new(&state);
        evaluator.configure_cache(node, off_off()).unwrap();
        let limits = TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        };
        let mut driver = AsyncDriver::new(state.snapshot(), limits, 1);
        let mut batch = RequestBatch::new();
        batch.at(FrameIndex(100)).request(node);
        let mut generation: RequestGeneration = driver
            .begin(&mut evaluator, state.snapshot(), batch, planner_config())
            .unwrap();
        driver.poll(&mut evaluator, &pool).unwrap();
        start_rx
            .recv_timeout(HANG)
            .expect("old worker never entered the gate");

        // Replace the definition and keep cancelling a successor that cannot run
        // while the gated worker still holds the only logical slot.
        state.edit(|e| e.replace(node, (), || 3_u64)).unwrap();
        let start = Instant::now();
        let mut cancel_ns = 0;
        let mut poll_ns = 0;
        let mut id = None;
        for _ in 0..20 {
            let mut batch = RequestBatch::new();
            id = Some(batch.at(FrameIndex(20)).request(node));
            let tick = Instant::now();
            assert!(driver.cancel(generation));
            generation = driver
                .begin(&mut evaluator, state.snapshot(), batch, planner_config())
                .unwrap();
            cancel_ns += tick.elapsed().as_nanos();
            let tick = Instant::now();
            assert!(driver.poll(&mut evaluator, &pool).unwrap().is_none());
            poll_ns += tick.elapsed().as_nanos();
            assert_eq!(driver.in_flight(), 1);
        }
        gate.release();
        let deadline = Instant::now() + HANG;
        let results = loop {
            let tick = Instant::now();
            let outcome = driver.poll(&mut evaluator, &pool).unwrap();
            poll_ns += tick.elapsed().as_nanos();
            if let Some((results, _)) = outcome {
                break results;
            }
            assert!(Instant::now() < deadline, "cancel sample stalled");
            std::thread::yield_now();
        };
        let total_ns = start.elapsed().as_nanos();
        assert_eq!(results.get(id.unwrap()).unwrap(), &3);
        let stats = driver.stats();
        assert_eq!(stats.terminal, 2);
        assert_eq!(stats.obsolete, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.peak_in_flight, 1);
        assert_eq!(driver.in_flight(), 0);
        let calls = evaluator.stats().computation_calls;
        drop(driver);
        drop(pool);
        print_row(&Row {
            sample,
            case: "cancel_replace",
            rounds: 0,
            actors: 1,
            frames: 1,
            begin_ns: 0,
            cancel_ns,
            poll_ns,
            total_ns,
            calls,
            config: Config {
                topology: "n/a",
                frame_pattern: "n/a",
                periodic: false,
                memo: "off",
                context: "off",
                cache_lifetime: "cold",
                measurement_boundary: GATED_BOUNDARY,
                frame_window: 16,
                max_in_flight: limits.max_in_flight,
                max_pending: limits.max_pending,
                chunk_size: 1,
            },
            stats,
        });
    }
}

// ---------------------------------------------------------------------------
// Case: cached_replacement_zero_slots
// ---------------------------------------------------------------------------

fn cached_replacement_samples() {
    for sample in 0..SAMPLES {
        let pool = benchmark_pool();
        let gate = Arc::new(Gate::new());
        let _releaser = GateReleaser(Arc::clone(&gate));
        let (start_tx, start_rx) = mpsc::channel();
        let mut state = GraphState::new();
        // A gated old node holds the only slot while the replacement resolves an
        // already memoized Constant(10) -> +1 -> *2 chain.
        let (gated, constant, plus_one, tail) = state
            .edit(|e| {
                let gate = Arc::clone(&gate);
                let start_tx = start_tx.clone();
                let gated = e.map((), move || {
                    if !gate.is_open() {
                        let _ = start_tx.send(());
                    }
                    gate.wait();
                    1_u64
                })?;
                let constant = e.constant(10_u64);
                let plus_one = e.map(constant, |x: &u64| x.wrapping_add(1))?;
                let tail = e.map(plus_one, |x: &u64| x.wrapping_mul(2))?;
                Ok((gated, constant, plus_one, tail))
            })
            .unwrap();
        let mut evaluator = Evaluator::new(&state);
        // The gated old node stays Off/Off, while only the chain is memoized. The
        // replacement request is therefore served entirely by the warmed chain memo
        // in the zero-slot case, which is exactly why it dispatches nothing.
        evaluator.configure_cache(gated, off_off()).unwrap();
        for &node in &[constant, plus_one, tail] {
            evaluator.configure_cache(node, last_off()).unwrap();
        }
        let warmed = evaluator
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(42), tail)
            .unwrap();
        assert_eq!(*warmed, 22);

        let limits = TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        };
        let mut driver = AsyncDriver::new(state.snapshot(), limits, 1);
        let mut batch = RequestBatch::new();
        batch.at(FrameIndex(7)).request(gated);
        let tick = Instant::now();
        driver
            .begin(&mut evaluator, state.snapshot(), batch, planner_config())
            .unwrap();
        let begin_ns = tick.elapsed().as_nanos();
        let mut poll_ns = 0;
        let tick = Instant::now();
        assert!(driver.poll(&mut evaluator, &pool).unwrap().is_none());
        poll_ns += tick.elapsed().as_nanos();
        start_rx
            .recv_timeout(HANG)
            .expect("old worker never entered the gate");
        let dispatched_before = driver.stats().dispatched;
        assert_eq!(dispatched_before, 1);

        // Replacement with zero dispatch slots available.
        let start = Instant::now();
        let mut batch = RequestBatch::new();
        let id = batch.at(FrameIndex(42)).request(tail);
        let tick = Instant::now();
        driver
            .begin(&mut evaluator, state.snapshot(), batch, planner_config())
            .unwrap();
        let cancel_ns = tick.elapsed().as_nanos();
        let (extra_poll_ns, results) = drive_to_completion(
            &mut driver,
            &mut evaluator,
            &pool,
            "replacement did not deliver while the gate stayed closed",
        );
        poll_ns += extra_poll_ns;
        assert!(
            !gate.is_open(),
            "replacement delivered only after the gate opened"
        );
        assert_eq!(*results.get(id).unwrap(), 22);
        assert_eq!(
            driver.stats().dispatched,
            dispatched_before,
            "the replacement dispatched new work"
        );

        // Release the gate and reclaim the obsolete terminal.
        gate.release();
        let deadline = Instant::now() + HANG;
        while driver.in_flight() > 0 {
            driver.poll(&mut evaluator, &pool).unwrap();
            assert!(
                Instant::now() < deadline,
                "obsolete worker was not reclaimed"
            );
            std::thread::yield_now();
        }
        let total_ns = start.elapsed().as_nanos();
        let stats = driver.stats();
        assert_eq!(stats.obsolete, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.terminal, stats.dispatched);
        let calls = evaluator.stats().computation_calls;
        drop(driver);
        drop(pool);
        print_row(&Row {
            sample,
            case: "cached_replacement_zero_slots",
            rounds: 0,
            actors: 1,
            frames: 1,
            begin_ns,
            cancel_ns,
            poll_ns,
            total_ns,
            calls,
            config: Config {
                topology: "n/a",
                frame_pattern: "n/a",
                periodic: false,
                memo: "last",
                context: "off",
                cache_lifetime: "warm",
                measurement_boundary: GATED_BOUNDARY,
                frame_window: 16,
                max_in_flight: limits.max_in_flight,
                max_pending: limits.max_pending,
                chunk_size: 1,
            },
            stats,
        });
    }
}

// ---------------------------------------------------------------------------
// Case: single_task_panic
// ---------------------------------------------------------------------------

fn single_task_panic_samples() {
    for chunk in CHUNKS {
        for sample in 0..SAMPLES {
            let pool = benchmark_pool();
            let gate = Arc::new(Gate::new());
            let _releaser = GateReleaser(Arc::clone(&gate));
            let (start_tx, start_rx) = mpsc::channel();
            let mut state = GraphState::new();
            // Four independent computations in creation order: one gated, one that
            // panics exactly once, and two healthy siblings.
            let nodes = state
                .edit(|e| {
                    let gate = Arc::clone(&gate);
                    let start_tx = start_tx.clone();
                    let first = e.map((), move || {
                        if !gate.is_open() {
                            let _ = start_tx.send(());
                        }
                        gate.wait();
                        10_u64
                    })?;
                    let failing = e.map((), || -> u64 { panic!("single task panic") })?;
                    let third = e.map((), || 30_u64)?;
                    let fourth = e.map((), || 40_u64)?;
                    Ok([first, failing, third, fourth])
                })
                .unwrap();
            let mut evaluator = Evaluator::new(&state);
            for &node in &nodes {
                evaluator.configure_cache(node, off_off()).unwrap();
            }
            let mut driver = AsyncDriver::new(state.snapshot(), QUOTA, chunk);

            let mut batch = RequestBatch::new();
            // The failed request publishes no values, so its request identities are
            // not retained; the bad node is requested exactly this once.
            for &node in &nodes {
                batch.at(FrameIndex(0)).request(node);
            }
            let start = Instant::now();
            let tick = Instant::now();
            let failed_generation = driver
                .begin(&mut evaluator, state.snapshot(), batch, planner_config())
                .unwrap();
            let begin_ns = tick.elapsed().as_nanos();
            let mut poll_ns = 0;
            let tick = Instant::now();
            assert!(
                driver.poll(&mut evaluator, &pool).unwrap().is_none(),
                "first poll published results"
            );
            poll_ns += tick.elapsed().as_nanos();
            start_rx
                .recv_timeout(HANG)
                .expect("gated worker never started");

            gate.release();
            let deadline = Instant::now() + HANG;
            loop {
                let tick = Instant::now();
                let outcome = driver.poll(&mut evaluator, &pool);
                poll_ns += tick.elapsed().as_nanos();
                match outcome {
                    Ok(None) => {}
                    Ok(Some(_)) => panic!("a failed request published successful results"),
                    Err(error) => {
                        assert_eq!(error, EvalError::WorkerPanicked);
                        break;
                    }
                }
                assert!(
                    Instant::now() < deadline,
                    "the panicking worker never failed the request"
                );
                std::thread::yield_now();
            }
            // Successful payloads of the failed batch may be dropped; only the
            // logical quota contract is asserted here.
            let deadline = Instant::now() + HANG;
            while driver.in_flight() > 0 {
                driver.poll(&mut evaluator, &pool).unwrap();
                assert!(
                    Instant::now() < deadline,
                    "logical quota was not reclaimed after the failure"
                );
                std::thread::yield_now();
            }
            let failed_stats = driver.stats();
            assert_eq!(failed_stats.dispatched, 4);
            assert_eq!(failed_stats.terminal, 4);
            assert_eq!(failed_stats.failed, 1);
            let tick = Instant::now();
            assert!(
                !driver.cancel(failed_generation),
                "a failed generation canceled its finished request"
            );
            let mut cancel_ns = tick.elapsed().as_nanos();

            // The same driver and evaluator accept a safe sub-batch; the bad node is
            // never requested again.
            let mut batch = RequestBatch::new();
            let safe_ids = [
                batch.at(FrameIndex(0)).request(nodes[0]),
                batch.at(FrameIndex(0)).request(nodes[2]),
                batch.at(FrameIndex(0)).request(nodes[3]),
            ];
            driver
                .begin(&mut evaluator, state.snapshot(), batch, planner_config())
                .unwrap();
            let tick = Instant::now();
            assert!(
                !driver.cancel(failed_generation),
                "the stale failed generation canceled the new request"
            );
            cancel_ns += tick.elapsed().as_nanos();
            let (extra_poll_ns, results) = drive_to_completion(
                &mut driver,
                &mut evaluator,
                &pool,
                "safe sub-batch did not complete",
            );
            poll_ns += extra_poll_ns;
            let total_ns = start.elapsed().as_nanos();
            assert_eq!(*results.get(safe_ids[0]).unwrap(), 10);
            assert_eq!(*results.get(safe_ids[1]).unwrap(), 30);
            assert_eq!(*results.get(safe_ids[2]).unwrap(), 40);
            assert_eq!(driver.in_flight(), 0);
            let stats = driver.stats();
            assert_eq!(stats.dispatched, 7);
            assert_eq!(stats.terminal, 7);
            assert_eq!(stats.failed, 1);
            let calls = evaluator.stats().computation_calls;
            drop(driver);
            drop(pool);
            print_row(&Row {
                sample,
                case: "single_task_panic",
                rounds: 0,
                actors: 1,
                frames: 1,
                begin_ns,
                cancel_ns,
                poll_ns,
                total_ns,
                calls,
                config: Config {
                    topology: "n/a",
                    frame_pattern: "n/a",
                    periodic: false,
                    memo: "off",
                    context: "off",
                    cache_lifetime: "cold",
                    measurement_boundary: GATED_BOUNDARY,
                    frame_window: 16,
                    max_in_flight: QUOTA.max_in_flight,
                    max_pending: QUOTA.max_pending,
                    chunk_size: chunk,
                },
                stats,
            });
        }
    }
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("Use --release");
        std::process::exit(2);
    }
    println!("{HEADER}");
    complete_samples();
    cancellation_samples();
    cached_replacement_samples();
    single_task_panic_samples();
}
