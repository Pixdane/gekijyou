use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_tasks::{TaskPool, TaskPoolBuilder};
use bevy_transform::components::Transform;
use gekijyou_bevy::{AsyncDriver, BevyExecutor, Bindings, OutputField, TaskLimits};
use gekijyou_computation::{
    CachePolicy, EvalError, EvaluationResults, Evaluator, FrameIndex, GraphError, GraphState,
    NodeCacheConfig, PlannerConfig, RequestBatch,
};
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
fn off() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Off,
        context: CachePolicy::Off,
    }
}
fn limits() -> TaskLimits {
    TaskLimits {
        max_in_flight: 1,
        max_pending: 1,
    }
}
fn complete(driver: &mut AsyncDriver, eval: &mut Evaluator, pool: &TaskPool) -> EvaluationResults {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some((results, _)) = driver.poll(eval, pool).unwrap() {
            return results;
        }
        assert!(Instant::now() < deadline, "request did not complete");
        std::thread::yield_now();
    }
}

#[test]
fn parallel_chunks_match_the_frame_oracle() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let mut state = GraphState::new();
    let nodes = state
        .edit(|e| {
            (0..24)
                .map(|n| e.map(e.frame(), move |f: &FrameIndex| f.0 + n))
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap();
    for chunk in [1, 4, 16] {
        let mut eval = Evaluator::new(&state);
        for &node in &nodes {
            eval.configure_cache(node, off()).unwrap();
        }
        let mut batch = RequestBatch::new();
        let mut ids = Vec::new();
        for f in [30, 120, 30, 300] {
            for (n, &node) in nodes.iter().enumerate() {
                ids.push((f + n as u64, batch.at(FrameIndex(f)).request(node)));
            }
        }
        let (results, _) = eval
            .bind(state.snapshot())
            .unwrap()
            .evaluate_planned(
                batch,
                PlannerConfig::default(),
                &mut BevyExecutor::new(&pool, chunk),
            )
            .unwrap();
        for (expected, id) in ids {
            assert_eq!(*results.get(id).unwrap(), expected);
        }
        assert_eq!(eval.stats().computation_calls, 72);
    }
}
#[test]
fn panic_inside_a_chunk_fails_successors_without_stranding_other_tasks() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let nodes = state
        .edit(|e| {
            let mut nodes = Vec::new();
            for n in 0..8 {
                let calls = calls.clone();
                nodes.push(e.map((), move || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    assert!(n != 3, "chunk worker panic");
                    n as u64
                })?);
            }
            Ok(nodes)
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    let mut batch = RequestBatch::new();
    for &n in &nodes {
        batch.at(FrameIndex(0)).request(n);
    }
    assert!(matches!(
        eval.bind(state.snapshot()).unwrap().evaluate_planned(
            batch,
            PlannerConfig::default(),
            &mut BevyExecutor::new(&pool, 8)
        ),
        Err(EvalError::WorkerPanicked)
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 8);
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), nodes[0])
            .unwrap(),
        0
    );
}

#[test]
fn definition_edits_and_repeated_replacement_keep_old_worker_quota_until_terminal() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let mut state = GraphState::new();
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (start_tx, start_rx) = mpsc::channel();
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
                Vec3::ONE
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(node, off()).unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_output(
            &state.snapshot(),
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    let mut driver = AsyncDriver::new(state.snapshot(), limits());
    let mut old = bindings
        .request(&state.snapshot(), &[FrameIndex(0)])
        .unwrap();
    let old_generation = driver
        .begin(
            &mut eval,
            state.snapshot(),
            old.take_batch().unwrap(),
            PlannerConfig::default(),
        )
        .unwrap();
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    start_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    state
        .edit(|e| e.replace(node, (), || Vec3::splat(2.)))
        .unwrap();
    let mut bound = None;
    for _ in 0..20 {
        let mut next = bindings
            .request(&state.snapshot(), &[FrameIndex(0)])
            .unwrap();
        driver
            .begin(
                &mut eval,
                state.snapshot(),
                next.take_batch().unwrap(),
                PlannerConfig::default(),
            )
            .unwrap();
        assert!(!driver.cancel(old_generation));
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert_eq!(driver.in_flight(), 1);
        bound = Some(next);
    }
    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    let results = complete(&mut driver, &mut eval, &pool);
    bindings
        .apply(
            &mut world,
            &state.snapshot(),
            &bound.unwrap(),
            &results,
            FrameIndex(0),
        )
        .unwrap();
    assert_eq!(
        world.get::<Transform>(entity).unwrap().translation,
        Vec3::splat(2.)
    );
    let stats = driver.stats();
    assert_eq!(stats.peak_in_flight, 1);
    assert_eq!(stats.terminal, 2);
    assert_eq!(stats.obsolete, 1);
    assert!(stats.canceled_pending >= 19);
    assert_eq!(eval.stats().computation_calls, 1);
}

#[test]
fn worker_failure_is_terminal_and_never_delivers_partial_results() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut state = GraphState::new();
    let (good, bad) = state
        .edit(|e| Ok((e.constant(1_u64), e.map((), || -> u64 { panic!("async") })?)))
        .unwrap();
    let mut eval = Evaluator::new(&state);
    let mut driver = AsyncDriver::new(state.snapshot(), limits());
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(good);
    batch.at(FrameIndex(0)).request(bad);
    driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match driver.poll(&mut eval, &pool) {
            Ok(None) => (),
            Err(EvalError::WorkerPanicked) => break,
            _ => panic!("unexpected terminal result"),
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(driver.stats().failed, 1);
    let mut retry = RequestBatch::new();
    let id = retry.at(FrameIndex(0)).request(good);
    driver
        .begin(&mut eval, state.snapshot(), retry, PlannerConfig::default())
        .unwrap();
    assert_eq!(complete(&mut driver, &mut eval, &pool).get(id).unwrap(), &1);
}

#[test]
fn same_snapshot_requests_have_distinct_identities_and_old_cancel_tokens_are_harmless() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(3_u64))).unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    let mut driver = AsyncDriver::new(snapshot.clone(), limits());
    let mut old = RequestBatch::new();
    old.at(FrameIndex(0)).request(node);
    let generation = driver
        .begin(&mut eval, snapshot.clone(), old, PlannerConfig::default())
        .unwrap();
    driver.poll(&mut eval, &pool).unwrap();
    let mut new = RequestBatch::new();
    let id = new.at(FrameIndex(1)).request(node);
    let successor = driver
        .begin(&mut eval, snapshot.clone(), new, PlannerConfig::default())
        .unwrap();
    assert_ne!(generation, successor);
    assert!(!driver.cancel(generation));
    let results = complete(&mut driver, &mut eval, &pool);
    assert_eq!(results.get(id).unwrap(), &3);
    assert_eq!(results.snapshot_id(), snapshot.id());
    assert_eq!(driver.in_flight(), 0);
}

#[test]
fn cancellation_is_explicit_and_preserves_existing_evaluator_history() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut state = GraphState::new();
    let node = state
        .edit(|e| e.map(e.frame(), |f: &FrameIndex| f.0))
        .unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    assert_eq!(
        *eval
            .bind(snapshot.clone())
            .unwrap()
            .sample(FrameIndex(100), node)
            .unwrap(),
        100
    );
    let mut driver = AsyncDriver::new(snapshot.clone(), limits());
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(20)).request(node);
    let generation = driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    assert!(driver.cancel(generation));
    assert!(!driver.cancel(generation));
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    assert_eq!(driver.in_flight(), 0);
    assert_eq!(
        *eval
            .bind(snapshot.clone())
            .unwrap()
            .sample(FrameIndex(100), node)
            .unwrap(),
        100
    );
    assert_eq!(eval.stats().computation_calls, 1);
    assert_eq!(state.snapshot().id(), snapshot.id());
}

#[test]
fn invalid_replacement_does_not_cancel_a_valid_request() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(8_u64))).unwrap();
    let mut eval = Evaluator::new(&state);
    let mut driver = AsyncDriver::new(state.snapshot(), limits());
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(20)).request(node);
    let generation = driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    let foreign = GraphState::new();
    assert_eq!(
        driver.begin(
            &mut eval,
            foreign.snapshot(),
            RequestBatch::new(),
            PlannerConfig::default()
        ),
        Err(EvalError::ForeignGraph)
    );
    assert_eq!(driver.generation(), generation);
    let invalid = PlannerConfig {
        frame_window: 0,
        ..Default::default()
    };
    assert_eq!(
        driver.begin(&mut eval, state.snapshot(), RequestBatch::new(), invalid),
        Err(EvalError::InvalidPlannerConfig)
    );
    assert_eq!(driver.generation(), generation);
    let mut foreign = GraphState::new();
    let foreign_node = foreign.edit(|e| Ok(e.constant(3_u64))).unwrap();
    let deleted = state.edit(|e| Ok(e.constant(9_u64))).unwrap();
    state.edit(|e| e.remove(deleted)).unwrap();
    for (node, expected) in [
        (
            foreign_node,
            GraphError::ForeignNode {
                node: foreign_node.id(),
            },
        ),
        (deleted, GraphError::MissingNode { node: deleted.id() }),
    ] {
        let mut invalid = RequestBatch::new();
        invalid.at(FrameIndex(20)).request(node);
        assert_eq!(
            driver.begin(
                &mut eval,
                state.snapshot(),
                invalid,
                PlannerConfig::default()
            ),
            Err(EvalError::Graph(expected))
        );
        assert_eq!(driver.generation(), generation);
        assert_eq!(eval.stats().definition_syncs, 0);
    }
    assert_eq!(complete(&mut driver, &mut eval, &pool).get(id).unwrap(), &8);
}

#[test]
fn obsolete_payload_drop_panic_terminates_active_request_and_allows_retry() {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::atomic::AtomicBool,
    };
    struct Value(Arc<AtomicBool>);
    impl Drop for Value {
        fn drop(&mut self) {
            assert!(
                !self.0.swap(false, Ordering::Relaxed),
                "obsolete payload destructor"
            );
        }
    }
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let flag = Arc::new(AtomicBool::new(true));
    let (started_tx, started_rx) = mpsc::channel();
    let mut state = GraphState::new();
    let (old, current) = state
        .edit(|e| {
            let gate = gate.clone();
            let flag = flag.clone();
            let old = e.map((), move || {
                started_tx.send(()).unwrap();
                let (lock, ready) = &*gate;
                let mut open = lock.lock().unwrap();
                while !*open {
                    open = ready.wait(open).unwrap();
                }
                Value(flag.clone())
            })?;
            Ok((old, e.constant(42_u64)))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    let mut driver = AsyncDriver::new(state.snapshot(), limits());
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(old);
    let old_generation = driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(driver.cancel(old_generation));
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(current);
    let generation = driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            while flag.load(Ordering::Relaxed) {
                driver.poll(&mut eval, &pool).unwrap();
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }))
        .is_err()
    );
    assert!(!flag.load(Ordering::Relaxed));
    assert_eq!(driver.in_flight(), 0);
    assert_eq!(driver.stats().terminal, 1);
    assert_eq!(driver.stats().obsolete, 1);
    assert!(
        !driver.cancel(generation),
        "unwinding must make the active request terminal"
    );
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    assert_eq!(
        driver.stats().dispatched,
        1,
        "queued successors must be canceled"
    );
    assert_eq!(driver.stats().canceled_pending, 1);
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(0)).request(current);
    driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    assert_eq!(
        complete(&mut driver, &mut eval, &pool).get(id).unwrap(),
        &42
    );
    assert_eq!(driver.stats().terminal, 2);
}

#[test]
fn async_completed_selection_can_be_held_and_applied_repeatedly() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let mut state = GraphState::new();
    let node = state
        .edit(|e| e.map(e.frame(), |f: &FrameIndex| Vec3::splat(f.0 as f32)))
        .unwrap();
    let snapshot = state.snapshot();
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let mut bindings = Bindings::new();
    let output = bindings
        .bind_output(
            &snapshot,
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    let mut bound = bindings
        .request_selected(
            &snapshot,
            &[FrameIndex(100), FrameIndex(20), FrameIndex(300)],
            &[output],
        )
        .unwrap();
    let mut eval = Evaluator::new(&state);
    let mut driver = AsyncDriver::new(snapshot.clone(), limits());
    driver
        .begin(
            &mut eval,
            snapshot.clone(),
            bound.take_batch().unwrap(),
            PlannerConfig::default(),
        )
        .unwrap();
    let results = complete(&mut driver, &mut eval, &pool);
    drop(driver);
    drop(eval);
    for f in [300, 20, 100, 20] {
        bindings
            .apply(&mut world, &snapshot, &bound, &results, FrameIndex(f))
            .unwrap();
        assert_eq!(
            world.get::<Transform>(entity).unwrap().translation,
            Vec3::splat(f as f32)
        );
    }
}

#[test]
fn cache_commit_panic_reclaims_worker_quota_and_allows_a_fresh_request() {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::atomic::AtomicBool,
    };
    struct Value {
        frame: u64,
        panic: Arc<AtomicBool>,
    }
    impl Drop for Value {
        fn drop(&mut self) {
            assert!(
                !self.panic.swap(false, Ordering::Relaxed),
                "memo victim destructor"
            );
        }
    }
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let flag = Arc::new(AtomicBool::new(false));
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            let panic = flag.clone();
            e.map(e.frame(), move |f: &FrameIndex| Value {
                frame: f.0,
                panic: panic.clone(),
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    drop(
        eval.bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), node)
            .unwrap(),
    );
    let mut driver = AsyncDriver::new(
        state.snapshot(),
        TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        },
    );
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(1)).request(node);
    driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    driver.poll(&mut eval, &pool).unwrap();
    flag.store(true, Ordering::Relaxed);
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(
        catch_unwind(AssertUnwindSafe(|| loop {
            driver.poll(&mut eval, &pool).unwrap();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }))
        .is_err()
    );
    assert!(!flag.load(Ordering::Relaxed));
    assert_eq!(driver.in_flight(), 0);
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(1)).request(node);
    driver
        .begin(&mut eval, state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    assert_eq!(
        complete(&mut driver, &mut eval, &pool)
            .get(id)
            .unwrap()
            .frame,
        1
    );
    assert_eq!(driver.stats().terminal, 2);
    assert_eq!(eval.stats().computation_calls, 3);
}

#[test]
fn explicit_serial_and_parallel_planner_results_apply_to_ecs() {
    use bevy_math::Quat;
    use gekijyou_computation::SerialExecutor;
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let mut state = GraphState::new();
    let (position, rotation) = state
        .edit(|e| {
            Ok((
                e.map(e.frame(), |f: &FrameIndex| Vec3::splat(f.0 as f32))?,
                e.constant(Quat::from_rotation_z(0.5)),
            ))
        })
        .unwrap();
    let snapshot = state.snapshot();
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let mut bindings = Bindings::new();
    bindings
        .bind_output(
            &snapshot,
            entity,
            position,
            OutputField::transform_translation(),
        )
        .unwrap();
    bindings
        .bind_output(
            &snapshot,
            entity,
            rotation,
            OutputField::transform_rotation(),
        )
        .unwrap();
    for parallel in [false, true] {
        let mut bound = bindings
            .request(&snapshot, &[FrameIndex(100), FrameIndex(20)])
            .unwrap();
        let batch = bound.take_batch().unwrap();
        let mut eval = Evaluator::new(&state);
        let mut scope = eval.bind(snapshot.clone()).unwrap();
        let (results, _) = if parallel {
            scope.evaluate_planned(
                batch,
                PlannerConfig::default(),
                &mut BevyExecutor::new(&pool, 4),
            )
        } else {
            scope.evaluate_planned(batch, PlannerConfig::default(), &mut SerialExecutor)
        }
        .unwrap();
        for f in [100, 20] {
            bindings
                .apply(&mut world, &snapshot, &bound, &results, FrameIndex(f))
                .unwrap();
            let transform = world.get::<Transform>(entity).unwrap();
            assert_eq!(transform.translation, Vec3::splat(f as f32));
            assert_eq!(transform.rotation, Quat::from_rotation_z(0.5));
        }
    }
}
