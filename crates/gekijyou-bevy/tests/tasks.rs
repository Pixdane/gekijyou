use bevy_tasks::TaskPoolBuilder;
use gekijyou_bevy::{RequestGeneration, TaskCoordinator, TaskId, TaskLimits, TaskStatus};
use gekijyou_computation::{EvalError, EvaluationSnapshot, GraphState};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn edits_and_repeated_cancellation_keep_old_workers_counted_until_terminal() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let mut state = GraphState::new();
    let drops = Arc::new(AtomicUsize::new(0));
    let input = state
        .edit(|e| {
            let i = e.input::<Probe>();
            e.set(i, Probe(drops.clone()))?;
            Ok(i)
        })
        .unwrap();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        },
        1,
    );
    let (start_tx, start_rx) = mpsc::channel();
    let (end_tx, end_rx) = mpsc::channel();
    struct Releaser {
        end_tx: mpsc::Sender<()>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            // Opens the gate before the pool joins so a failed assertion can
            // never strand the worker.
            let _ = self.end_tx.send(());
        }
    }
    // Armed (constructed) before the first fallible wait below.
    let _releaser = Releaser {
        end_tx: end_tx.clone(),
    };
    let old = tasks
        .submit(&pool, move |snapshot| {
            start_tx.send(()).unwrap();
            end_rx.recv().unwrap();
            assert!(snapshot.get(input).unwrap().is_some());
            7_u64
        })
        .unwrap();
    start_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    state.edit(|e| e.set(input, Probe(drops.clone()))).unwrap();
    for _ in 0..20 {
        let generation = RequestGeneration::new();
        tasks.set_session(state.snapshot(), generation).unwrap();
        assert_eq!(tasks.in_flight(), 1);
        assert_eq!(tasks.status(old), Some(TaskStatus::CanceledRunning));
        tasks.submit(&pool, |_| 8).unwrap();
        assert!(tasks.submit(&pool, |_| 9).is_none());
    }
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    end_tx.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut values = Vec::new();
    while tasks.in_flight() > 0 || tasks.pending() > 0 {
        values.extend(tasks.poll(&pool));
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].result, Ok(8));
    assert_eq!(tasks.stats().terminal, 2);
    assert_eq!(tasks.stats().obsolete, 1);
    assert_eq!(tasks.stats().peak_in_flight, 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
#[test]
fn worker_panic_is_terminal_and_a_new_job_can_use_the_slot() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        },
        1,
    );
    tasks.submit(&pool, |_| panic!("worker")).unwrap();
    tasks.submit(&pool, |_| 3).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut finished = Vec::new();
    while tasks.in_flight() > 0 || tasks.pending() > 0 {
        finished.extend(tasks.poll(&pool));
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(finished.len(), 2);
    assert!(
        finished
            .iter()
            .any(|f| f.result == Err(EvalError::WorkerPanicked))
    );
    assert!(finished.iter().any(|f| f.result == Ok(3)));
    assert_eq!(tasks.stats().terminal, 2);
    assert_eq!(tasks.stats().failed, 1);
    assert!(tasks.poll(&pool).is_empty());
    assert_eq!(tasks.in_flight(), 0);
}
#[test]
fn session_definitions_cannot_roll_back_even_after_old_completion() {
    let mut state = GraphState::new();
    let old = state.snapshot();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<()>::new(
        old.clone(),
        generation,
        TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        },
        1,
    );
    state
        .edit(|e| {
            e.constant(1_u64);
            Ok(())
        })
        .unwrap();
    tasks.set_session(state.snapshot(), generation).unwrap();
    assert!(matches!(
        tasks.set_session(old, generation),
        Err(EvalError::StaleGraphDefinition { .. })
    ));
}
fn drain<R: Send + 'static>(
    tasks: &mut TaskCoordinator<R>,
    pool: &bevy_tasks::TaskPool,
) -> Vec<gekijyou_bevy::FinishedTask<R>> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut finished = Vec::new();
    while tasks.in_flight() > 0 || tasks.pending() > 0 {
        finished.extend(tasks.poll(pool));
        assert!(Instant::now() < deadline, "chunk drain hung");
        std::thread::yield_now();
    }
    finished
}

#[test]
fn batch_submission_chunks_physically_and_preserves_logical_semantics() {
    let state = GraphState::new();
    let snapshot = state.snapshot();
    for (chunk, expected_jobs_first, expected_jobs_final) in [(1, 4, 5), (4, 1, 2)] {
        let pool = TaskPoolBuilder::new().num_threads(4).build();
        let generation = RequestGeneration::new();
        let mut tasks = TaskCoordinator::<u64>::new(
            snapshot.clone(),
            generation,
            TaskLimits {
                max_in_flight: 4,
                max_pending: 4,
            },
            chunk,
        );
        let ids: Vec<TaskId> = tasks
            .submit_batch_fallible(&pool, (0..5).map(move |n| move |_| Ok(n as u64)))
            .unwrap();
        assert_eq!(ids.len(), 5);
        // After submit, before any poll: 4 logical running, 1 pending.
        assert_eq!(tasks.in_flight(), 4);
        assert_eq!(tasks.pending(), 1);
        assert_eq!(tasks.stats().dispatched, 4);
        assert_eq!(
            tasks.stats().worker_jobs_spawned,
            expected_jobs_first,
            "chunk {chunk} first dispatch"
        );
        let finished = drain(&mut tasks, &pool);
        assert_eq!(finished.len(), 5);
        let mut values = std::collections::HashMap::new();
        for f in finished {
            values.insert(f.id, f.result.unwrap());
        }
        for (n, &id) in ids.iter().enumerate() {
            assert_eq!(values[&id], n as u64, "chunk {chunk} value {n}");
        }
        let stats = tasks.stats();
        assert_eq!(stats.dispatched, 5);
        assert_eq!(stats.terminal, 5);
        assert_eq!(
            stats.worker_jobs_spawned, expected_jobs_final,
            "chunk {chunk} final"
        );
    }
}

#[test]
fn partial_chunk_member_panic_does_not_skip_remaining_members() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 3,
            max_pending: 0,
        },
        4,
    );
    let ids: Vec<TaskId> = tasks
        .submit_batch_fallible(
            &pool,
            (0..3).map(|n| {
                move |_| {
                    if n == 1 {
                        panic!("middle member");
                    }
                    Ok(n as u64)
                }
            }),
        )
        .unwrap();
    // Only a partial chunk (3 < 4) is admitted under limits 3+0.
    assert_eq!(tasks.in_flight(), 3);
    assert_eq!(tasks.pending(), 0);
    let finished = drain(&mut tasks, &pool);
    assert_eq!(finished.len(), 3);
    let mut results = std::collections::HashMap::new();
    for f in finished {
        results.insert(f.id, f.result);
    }
    assert_eq!(results[&ids[0]], Ok(0));
    assert_eq!(results[&ids[1]], Err(EvalError::WorkerPanicked));
    assert_eq!(results[&ids[2]], Ok(2));
    assert_eq!(tasks.stats().failed, 1);
    assert_eq!(tasks.stats().terminal, 3);
    assert_eq!(tasks.stats().worker_jobs_spawned, 1);
}

#[test]
fn full_chunk_member_panic_still_produces_sibling_results() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 4,
            max_pending: 4,
        },
        4,
    );
    let ids: Vec<TaskId> = tasks
        .submit_batch_fallible(
            &pool,
            (0..4).map(|n| {
                move |_| {
                    if n == 1 {
                        panic!("second member");
                    }
                    Ok((n * 10) as u64)
                }
            }),
        )
        .unwrap();
    // A complete 4-member chunk is admitted and spawned as one physical job.
    assert_eq!(tasks.in_flight(), 4);
    assert_eq!(tasks.pending(), 0);
    assert_eq!(tasks.stats().worker_jobs_spawned, 1);
    let finished = drain(&mut tasks, &pool);
    assert_eq!(finished.len(), 4);
    let mut results = std::collections::HashMap::new();
    for f in finished {
        results.insert(f.id, f.result);
    }
    assert_eq!(results[&ids[0]], Ok(0));
    assert_eq!(results[&ids[1]], Err(EvalError::WorkerPanicked));
    assert_eq!(results[&ids[2]], Ok(20));
    assert_eq!(results[&ids[3]], Ok(30));
    let stats = tasks.stats();
    assert_eq!(stats.failed, 1);
    assert_eq!(stats.terminal, 4);
    assert_eq!(stats.dispatched, 4);
    assert_eq!(stats.worker_jobs_spawned, 1);
}

#[test]
fn oversized_batch_is_rejected_whole_without_side_effects() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 2,
            max_pending: 1,
        },
        1,
    );
    let dropped = Arc::new(AtomicUsize::new(0));
    let dropped_outer = dropped.clone();
    struct Marker(Arc<AtomicUsize>);
    impl Drop for Marker {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let before = tasks.stats();
    // Captured resources must not be retained by a rejected batch. The Markers
    // are created EAGERLY (collect) and captured by each job closure; a
    // whole-batch rejection drops the closures with the vector, so all four
    // Markers observe their drop immediately after the failed call.
    let jobs: Vec<_> = (0..4)
        .map({
            let dropped = dropped.clone();
            move |_| {
                let marker = Marker(dropped.clone());
                move |_: EvaluationSnapshot| {
                    let _kept = marker;
                    Ok::<u64, EvalError>(7)
                }
            }
        })
        .collect();
    let ids = tasks.submit_batch_fallible(&pool, jobs);
    assert!(ids.is_none(), "4 > available 3 must reject whole batch");
    assert_eq!(tasks.stats(), before);
    assert_eq!(tasks.in_flight(), 0);
    assert_eq!(tasks.pending(), 0);
    assert_eq!(
        dropped_outer.load(Ordering::SeqCst),
        4,
        "rejected closures must be dropped, none retained"
    );
    // Exact fit is accepted: 3 == available.
    let ids = tasks
        .submit_batch_fallible(&pool, (0..3).map(|n| move |_| Ok(n as u64)))
        .unwrap();
    assert_eq!(ids.len(), 3);
    let finished = drain(&mut tasks, &pool);
    assert_eq!(finished.len(), 3);
    let mut values = std::collections::HashMap::new();
    for f in finished {
        values.insert(f.id, f.result.unwrap());
    }
    for (n, id) in ids.iter().enumerate() {
        assert_eq!(values[id], n as u64, "exact-fit value {n}");
    }
    let stats = tasks.stats();
    assert_eq!(stats.dispatched, 3);
    assert_eq!(stats.terminal, 3);
}

#[test]
fn canceled_full_chunk_keeps_quota_and_recovers_after_gate_release() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    struct Releaser {
        gate: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    // RAII releaser declared after the pool: on failure paths the gate opens
    // (and wakes all members) before the pool join can block forever.
    let _releaser = Releaser { gate: gate.clone() };
    impl Drop for Releaser {
        fn drop(&mut self) {
            *self.gate.0.lock().unwrap() = true;
            self.gate.1.notify_all();
        }
    }
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 4,
            max_pending: 4,
        },
        4,
    );
    let (started_tx, started_rx) = mpsc::channel();
    let gate_for_jobs = gate.clone();
    let ids: Vec<TaskId> = tasks
        .submit_batch_fallible(
            &pool,
            (0..4).map(move |_| {
                let started_tx = started_tx.clone();
                let gate = gate_for_jobs.clone();
                move |_| {
                    started_tx.send(()).unwrap();
                    let (lock, ready) = &*gate;
                    let mut open = lock.lock().unwrap();
                    while !*open {
                        open = ready.wait(open).unwrap();
                    }
                    Ok::<u64, EvalError>(1)
                }
            }),
        )
        .unwrap();
    // One physical job runs the four dispatched members FIFO: the first
    // blocks on the gate, the rest are running but have not executed yet.
    // The plan only requires confirming the
    // first member entered before cancel.
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("first chunk member started");
    // Cancel while the full chunk is gate-blocked: quota stays held.
    tasks.cancel();
    assert_eq!(tasks.in_flight(), 4);
    assert_eq!(tasks.status(ids[0]), Some(TaskStatus::CanceledRunning));
    // Twenty generation changes do not release the quota either.
    for _ in 0..20 {
        tasks
            .set_session(state.snapshot(), RequestGeneration::new())
            .unwrap();
        assert_eq!(tasks.in_flight(), 4);
    }
    // Release the gate; every obsolete worker reaches one terminal state.
    {
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while tasks.in_flight() > 0 {
        tasks.poll_completions();
        assert!(Instant::now() < deadline, "obsolete drain hung");
        std::thread::yield_now();
    }
    let stats = tasks.stats();
    assert_eq!(stats.terminal, 4);
    assert_eq!(stats.obsolete, 4);
}

#[test]
fn empty_batch_dispatches_pending_after_poll_completions_only() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        generation,
        TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        },
        4,
    );
    // Two tasks; only the first can dispatch (in_flight 1), the second pends.
    let ids: Vec<TaskId> = tasks
        .submit_batch_fallible(&pool, [10u64, 20].into_iter().map(|v| move |_| Ok(v)))
        .unwrap();
    assert_eq!(tasks.in_flight(), 1);
    assert_eq!(tasks.pending(), 1);
    // Reclaim the first task WITHOUT dispatching: loop poll_completions until
    // the worker's terminal message has actually arrived (yield, never sleep,
    // never poll(&pool) which would dispatch the queued second task).
    let deadline = Instant::now() + Duration::from_secs(5);
    let finished = loop {
        let finished = tasks.poll_completions();
        if !finished.is_empty() {
            break finished;
        }
        assert!(Instant::now() < deadline, "first completion never arrived");
        std::thread::yield_now();
    };
    assert_eq!(finished.len(), 1);
    assert_eq!(*finished[0].result.as_ref().unwrap(), 10);
    assert_eq!(tasks.in_flight(), 0);
    assert_eq!(tasks.pending(), 1, "empty batch must do the dispatch");
    // The empty batch is accepted and dispatches the pending task.
    let empty: Vec<TaskId> = tasks
        .submit_batch_fallible(
            &pool,
            std::iter::empty::<fn(EvaluationSnapshot) -> Result<u64, EvalError>>(),
        )
        .unwrap();
    assert!(empty.is_empty());
    assert_eq!(tasks.in_flight(), 1);
    assert_eq!(tasks.pending(), 0);
    let stats = tasks.stats();
    assert_eq!(stats.dispatched, 2);
    assert_eq!(stats.worker_jobs_spawned, 2);
    let finished = drain(&mut tasks, &pool);
    assert_eq!(finished.len(), 1);
    assert_eq!(*finished[0].result.as_ref().unwrap(), 20);
    assert_eq!(finished[0].id, ids[1]);
    let stats = tasks.stats();
    assert_eq!(stats.dispatched, 2);
    assert_eq!(stats.terminal, 2);
}

#[test]
fn zero_chunk_constructor_is_rejected() {
    let state = GraphState::new();
    let generation = RequestGeneration::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        TaskCoordinator::<u64>::new(
            state.snapshot(),
            generation,
            TaskLimits {
                max_in_flight: 1,
                max_pending: 1,
            },
            0,
        )
    }));
    assert!(result.is_err(), "chunk size 0 must panic at construction");
}
