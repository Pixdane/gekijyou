use bevy_tasks::TaskPoolBuilder;
use gekijyou_bevy::{AnimationRuntime, TaskCoordinator, TaskLimits, TaskStatus};
use gekijyou_computation::{EvalError, GraphState};
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
    let mut runtime = AnimationRuntime::new(1);
    let mut tasks = TaskCoordinator::new(
        state.snapshot(),
        runtime.generation(),
        TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        },
    );
    let (start_tx, start_rx) = mpsc::channel();
    let (end_tx, end_rx) = mpsc::channel();
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
        runtime.pause();
        tasks
            .set_session(state.snapshot(), runtime.generation())
            .unwrap();
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
    let runtime = AnimationRuntime::new(1);
    let mut tasks = TaskCoordinator::<u64>::new(
        state.snapshot(),
        runtime.generation(),
        TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        },
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
    let runtime = AnimationRuntime::new(1);
    let mut tasks = TaskCoordinator::<()>::new(
        old.clone(),
        runtime.generation(),
        TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        },
    );
    state
        .edit(|e| {
            e.constant(1_u64);
            Ok(())
        })
        .unwrap();
    tasks
        .set_session(state.snapshot(), runtime.generation())
        .unwrap();
    assert!(matches!(
        tasks.set_session(old, runtime.generation()),
        Err(EvalError::StaleGraphDefinition { .. })
    ));
}
