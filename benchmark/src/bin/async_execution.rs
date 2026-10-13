//! Explicit asynchronous execution measurements, separate from ECS adaptation.
//! Graph, TaskPool, evaluator and batch construction stay outside observation.
//! begin_ns and poll_ns measure coordinator calls (including cache commits);
//! total_ns additionally includes worker/queue latency until complete publication.
//! Cancellation uses a gated old worker to measure replacement and cross-session
//! quota retention deterministically; the gate affects timing, never output values.
//! Full independent output checks happen after measurement. Run with --release.
use bevy_tasks::{TaskPool, TaskPoolBuilder};
use gekijyou_bevy::{AsyncDriver, TaskLimits};
use gekijyou_computation::{Evaluator, FrameIndex, GraphState, PlannerConfig, RequestBatch};
use std::{
    hint::black_box,
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};
fn limits() -> TaskLimits {
    TaskLimits {
        max_in_flight: 4,
        max_pending: 16,
    }
}
fn complete_samples(pool: &TaskPool, rounds: u64) {
    let mut state = GraphState::new();
    let nodes = state
        .edit(|e| {
            (0..32)
                .map(|actor| {
                    e.map(e.frame(), move |f: &FrameIndex| {
                        let mut v = f.0.wrapping_add(actor);
                        for n in 0..rounds {
                            v = black_box(v.rotate_left(7).wrapping_mul(31).wrapping_add(n));
                        }
                        v
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap();
    let oracle: Vec<_> = (0..64_u64)
        .flat_map(|frame| {
            (0..32_u64).map(move |actor| {
                let mut v = frame.wrapping_add(actor);
                for n in 0..rounds {
                    v = v.rotate_left(7).wrapping_mul(31).wrapping_add(n);
                }
                v
            })
        })
        .collect();
    for sample in 0..9 {
        let mut evaluator = Evaluator::new(&state);
        let mut driver = AsyncDriver::new(state.snapshot(), limits());
        let mut batch = RequestBatch::new();
        let mut ids = Vec::new();
        for f in 0..64 {
            for &node in &nodes {
                ids.push(batch.at(FrameIndex(f)).request(node));
            }
        }
        let start = Instant::now();
        let begin = Instant::now();
        driver
            .begin(
                &mut evaluator,
                state.snapshot(),
                batch,
                PlannerConfig::default(),
            )
            .unwrap();
        let begin_ns = begin.elapsed().as_nanos();
        let mut poll_ns = 0;
        let deadline = Instant::now() + Duration::from_secs(30);
        let results = loop {
            let tick = Instant::now();
            let result = driver.poll(&mut evaluator, pool).unwrap();
            poll_ns += tick.elapsed().as_nanos();
            if let Some((results, _)) = result {
                break results;
            }
            assert!(Instant::now() < deadline, "Async benchmark stalled");
            std::thread::yield_now();
        };
        let total_ns = start.elapsed().as_nanos();
        for (id, expected) in ids.into_iter().zip(&oracle) {
            assert_eq!(results.get(id).unwrap(), expected);
        }
        assert_eq!(driver.in_flight(), 0);
        let stats = driver.stats();
        println!(
            "{sample}\tcomplete\t{rounds}\t32\t64\t{begin_ns}\t0\t{poll_ns}\t{total_ns}\t{}\t{}\t{}\t{}\t{}\t{}",
            stats.dispatched,
            stats.terminal,
            stats.obsolete,
            stats.canceled_pending,
            stats.peak_in_flight,
            evaluator.stats().computation_calls
        );
    }
}
fn cancellation_samples(pool: &TaskPool) {
    for sample in 0..9 {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (start_tx, start_rx) = mpsc::channel();
        let mut state = GraphState::new();
        let node = state
            .edit(|e| {
                let gate = gate.clone();
                e.map((), move || {
                    start_tx.send(()).unwrap();
                    let (lock, ready) = &*gate;
                    let mut open = lock.lock().unwrap();
                    while !*open {
                        open = ready.wait(open).unwrap();
                    }
                    1_u64
                })
            })
            .unwrap();
        let mut evaluator = Evaluator::new(&state);
        let mut driver = AsyncDriver::new(
            state.snapshot(),
            TaskLimits {
                max_in_flight: 1,
                max_pending: 1,
            },
        );
        let mut batch = RequestBatch::new();
        batch.at(FrameIndex(100)).request(node);
        let mut generation = driver
            .begin(
                &mut evaluator,
                state.snapshot(),
                batch,
                PlannerConfig::default(),
            )
            .unwrap();
        driver.poll(&mut evaluator, pool).unwrap();
        start_rx.recv_timeout(Duration::from_secs(5)).unwrap();
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
                .begin(
                    &mut evaluator,
                    state.snapshot(),
                    batch,
                    PlannerConfig::default(),
                )
                .unwrap();
            cancel_ns += tick.elapsed().as_nanos();
            let tick = Instant::now();
            assert!(driver.poll(&mut evaluator, pool).unwrap().is_none());
            poll_ns += tick.elapsed().as_nanos();
            assert_eq!(driver.in_flight(), 1);
        }
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
        let deadline = Instant::now() + Duration::from_secs(5);
        let results = loop {
            let tick = Instant::now();
            let result = driver.poll(&mut evaluator, pool).unwrap();
            poll_ns += tick.elapsed().as_nanos();
            if let Some((results, _)) = result {
                break results;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        };
        let total_ns = start.elapsed().as_nanos();
        assert_eq!(results.get(id.unwrap()).unwrap(), &3);
        let stats = driver.stats();
        assert_eq!(stats.terminal, 2);
        assert_eq!(stats.obsolete, 1);
        assert_eq!(stats.peak_in_flight, 1);
        assert_eq!(driver.in_flight(), 0);
        println!(
            "{sample}\tcancel_replace\t0\t1\t1\t0\t{cancel_ns}\t{poll_ns}\t{total_ns}\t{}\t{}\t{}\t{}\t{}\t{}",
            stats.dispatched,
            stats.terminal,
            stats.obsolete,
            stats.canceled_pending,
            stats.peak_in_flight,
            evaluator.stats().computation_calls
        );
    }
}
fn main() {
    if cfg!(debug_assertions) {
        eprintln!("Use --release");
        std::process::exit(2);
    }
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    println!(
        "sample\tcase\trounds\tactors\tframes\tbegin_ns\tcancel_ns\tpoll_ns\ttotal_ns\tdispatched\tterminal\tobsolete\tcanceled_pending\tpeak_in_flight\tcalls"
    );
    for rounds in [0, 20000] {
        complete_samples(&pool, rounds);
    }
    cancellation_samples(&pool);
}
