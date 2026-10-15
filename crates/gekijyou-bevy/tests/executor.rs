use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_tasks::{TaskPool, TaskPoolBuilder};
use bevy_transform::components::Transform;
use gekijyou_bevy::{AsyncDriver, BevyExecutor, Bindings, OutputField, TaskLimits};
use gekijyou_computation::{
    CachePolicy, EvalError, EvaluationResults, Evaluator, FrameIndex, GraphError, GraphState, Node,
    NodeCacheConfig, PlannerConfig, RequestBatch, RequestId,
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
    struct Releaser {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            let (lock, ready) = &*self.gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }
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
    // Declared after the pool; armed before the first poll that can block a
    // worker on the gate, so failures never strand the TaskPool join.
    let _releaser = Releaser { gate: gate.clone() };
    let mut bindings = Bindings::new();
    bindings
        .bind_output(
            &state.snapshot(),
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    let mut driver = AsyncDriver::new(state.snapshot(), limits(), 1);
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
    let mut driver = AsyncDriver::new(state.snapshot(), limits(), 1);
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
    let mut driver = AsyncDriver::new(snapshot.clone(), limits(), 1);
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
    let mut driver = AsyncDriver::new(snapshot.clone(), limits(), 1);
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
    let mut driver = AsyncDriver::new(state.snapshot(), limits(), 1);
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
    struct Releaser {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            let (lock, ready) = &*self.gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }
    // Declared after the pool, armed before the first poll that can strand a
    // worker on the gate.
    let _releaser = Releaser { gate: gate.clone() };
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
    let mut driver = AsyncDriver::new(state.snapshot(), limits(), 1);
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
    let mut driver = AsyncDriver::new(snapshot.clone(), limits(), 1);
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
        1,
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

#[test]
fn gated_old_worker_does_not_block_fresh_cached_chain_delivery() {
    struct Releaser {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            let (lock, ready) = &*self.gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    // Declared after the pool so reverse drop opens the gate before join.
    let _releaser = Releaser { gate: gate.clone() };
    let (started_tx, started_rx) = mpsc::channel();
    let mut state = GraphState::new();
    let (gated, chain_nodes) = state
        .edit(|e| {
            let gate = gate.clone();
            let gated = e.map((), move || {
                started_tx.send(()).unwrap();
                let (lock, ready) = &*gate;
                let mut open = lock.lock().unwrap();
                while !*open {
                    open = ready.wait(open).unwrap();
                }
                0_u64
            })?;
            let ten = e.constant(10_u64);
            let plus = e.map(ten, |x: &u64| x + 1)?;
            let times = e.map(plus, |x: &u64| x * 2)?;
            Ok((gated, (ten, plus, times)))
        })
        .unwrap();
    let (ten, plus, times) = chain_nodes;
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(gated, off()).unwrap();
    let chain_warm = NodeCacheConfig {
        memo: CachePolicy::Last,
        context: CachePolicy::Off,
    };
    for node in [ten, plus, times] {
        eval.configure_cache(node, chain_warm).unwrap();
    }
    // Warm the chain once on the same evaluator.
    {
        let mut warm_batch = RequestBatch::new();
        warm_batch.at(FrameIndex(0)).request(times);
        let mut plan = eval
            .plan(state.snapshot(), warm_batch, PlannerConfig::default())
            .unwrap();
        let mut completed = Vec::new();
        while !plan.is_complete() {
            let tasks = eval.advance(&mut plan, completed).unwrap();
            completed = tasks.into_iter().map(|t| t.execute()).collect();
        }
        plan.finish().unwrap();
    }
    assert_eq!(eval.cache_stats().memo_entries, 3);
    let snapshot = state.snapshot();
    let mut driver = AsyncDriver::new(
        snapshot.clone(),
        TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        },
        1,
    );
    let mut old_batch = RequestBatch::new();
    old_batch.at(FrameIndex(0)).request(gated);
    let old_generation = driver
        .begin(
            &mut eval,
            snapshot.clone(),
            old_batch,
            PlannerConfig::default(),
        )
        .unwrap();
    // The gated node claims the single in-flight slot.
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    // Replace with a batch requesting only the cached chain tail.
    let mut new_batch = RequestBatch::new();
    let tail_id = new_batch.at(FrameIndex(42)).request(times);
    let new_generation = driver
        .begin(
            &mut eval,
            snapshot.clone(),
            new_batch,
            PlannerConfig::default(),
        )
        .unwrap();
    assert_ne!(old_generation, new_generation);
    // While the gate stays closed the cached chain still delivers.
    let results = complete(&mut driver, &mut eval, &pool);
    assert_eq!(results.get(tail_id).unwrap(), &22);
    assert_eq!(driver.in_flight(), 1);
    let stats = driver.stats();
    assert_eq!(stats.dispatched, 1);
    // Release the gate and reclaim the old worker.
    {
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let stats = driver.stats();
    assert_eq!(stats.dispatched, 1);
    assert_eq!(stats.terminal, 1);
    assert_eq!(stats.obsolete, 1);
}
fn async_chunk_oracle(layers: u64, frame: u64) -> u64 {
    let mut x = frame;
    for k in 1..=layers {
        x = x.wrapping_add(k).rotate_left(3);
    }
    x
}

#[test]
fn async_chunks_cover_multi_layer_graph_across_repeated_frames() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let mut state = GraphState::new();
    let tail = state
        .edit(|e| {
            let frame = e.frame();
            let mut current = e.map(frame, |f: &FrameIndex| f.0)?;
            for k in 1..=64u64 {
                current = e.map(current, move |x: &u64| x.wrapping_add(k).rotate_left(3))?;
            }
            Ok(current)
        })
        .unwrap();
    let snapshot = state.snapshot();
    // Frames are generated once; DFS runs one batch through one evaluator and
    // its per-frame results are the cross-check the chunk matrix compares to.
    let frames: Vec<u64> = (0..24).map(|i| (((i * 17) % 12) * 7) as u64).collect();
    let dfs_expected: Vec<u64> = {
        let mut dfs_eval = Evaluator::new(&state);
        let mut batch = RequestBatch::new();
        let ids: Vec<_> = frames
            .iter()
            .map(|&f| batch.at(FrameIndex(f)).request(tail))
            .collect();
        let results = dfs_eval
            .bind(snapshot.clone())
            .unwrap()
            .evaluate(batch)
            .unwrap();
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| {
                let dfs_value = *results.get(id).unwrap();
                assert_eq!(
                    dfs_value,
                    async_chunk_oracle(64, frames[i]),
                    "DFS frame {} must match the oracle",
                    frames[i]
                );
                dfs_value
            })
            .collect()
    };
    for chunk in [1usize, 4, 16] {
        let mut eval = Evaluator::new(&state);
        let mut driver = AsyncDriver::new(
            snapshot.clone(),
            TaskLimits {
                max_in_flight: 64,
                max_pending: 64,
            },
            chunk,
        );
        // Repeated and out-of-order frames exercise window reuse per chunk.
        let mut batch = RequestBatch::new();
        let mut expectations = Vec::new();
        for &f in &frames {
            expectations.push((f, batch.at(FrameIndex(f)).request(tail)));
        }
        driver
            .begin(
                &mut eval,
                snapshot.clone(),
                batch,
                PlannerConfig {
                    max_in_flight: 64,
                    ..Default::default()
                },
            )
            .unwrap();
        let (results, _) = loop {
            match driver.poll(&mut eval, &pool).unwrap() {
                Some(done) => break done,
                None => std::thread::yield_now(),
            }
        };
        for ((f, id), dfs_value) in expectations.into_iter().zip(&dfs_expected) {
            let planner_value = *results.get(id).unwrap();
            assert_eq!(
                planner_value, *dfs_value,
                "chunk {chunk} frame {f}: planner vs DFS"
            );
            assert_eq!(
                planner_value,
                async_chunk_oracle(64, f),
                "chunk {chunk} frame {f}: oracle"
            );
        }
    }
}

#[test]
fn chunk_member_failure_reclaims_all_quota_and_allows_a_new_request() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            // One frame-dependent node: frame 1 panics, every other frame is a
            // plain +1. Ready order sorts by node then frame, so frame 1 is
            // the SECOND member of the four-frame chunk.
            e.map(e.frame(), |f: &FrameIndex| -> u64 {
                if f.0 == 1 {
                    panic!("chunk failure");
                }
                f.0 + 1
            })
        })
        .unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(node, off()).unwrap();
    let mut driver = AsyncDriver::new(
        snapshot.clone(),
        TaskLimits {
            max_in_flight: 64,
            max_pending: 64,
        },
        4,
    );
    let mut batch = RequestBatch::new();
    let mut ids = Vec::new();
    for f in 0..4_u64 {
        ids.push((f, batch.at(FrameIndex(f)).request(node)));
    }
    driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match driver.poll(&mut eval, &pool) {
            Err(EvalError::WorkerPanicked) => break,
            Ok(None) => (),
            Ok(Some(_)) => panic!("failed request must not deliver results"),
            Err(error) => panic!("unexpected error {error}"),
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    // The failing chunk's remaining members hold quota until their own
    // terminals arrive; drain completely before asserting totals. One poll
    // is not required to reclaim every not-yet-arrived completion.
    let deadline = Instant::now() + Duration::from_secs(5);
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let stats = driver.stats();
    assert_eq!(stats.failed, 1);
    assert_eq!(stats.dispatched, 4);
    assert_eq!(stats.terminal, 4);
    // A fresh request on the same driver/evaluator succeeds.
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(5)).request(node);
    driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    let results = complete(&mut driver, &mut eval, &pool);
    assert_eq!(*results.get(id).unwrap(), 6);
}

#[test]
fn canceled_full_chunk_does_not_pollute_new_definition_results() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    struct Releaser {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            let (lock, ready) = &*self.gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }
    let (started_tx, started_rx) = mpsc::channel();
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            let gate = gate.clone();
            e.map(e.frame(), move |f: &FrameIndex| {
                // All four logical members are dispatched and running; the
                // first blocks the gate inside the one physical job, so the
                // remaining three have not yet executed their closures.
                let (lock, ready) = &*gate;
                let mut open = lock.lock().unwrap();
                if f.0 == 0 {
                    let _ = started_tx.send(());
                    while !*open {
                        open = ready.wait(open).unwrap();
                    }
                }
                1_u64
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(node, off()).unwrap();
    // Declared after the pool, armed before the first fallible wait.
    let _releaser = Releaser { gate: gate.clone() };
    let snapshot = state.snapshot();
    let mut driver = AsyncDriver::new(
        snapshot.clone(),
        TaskLimits {
            max_in_flight: 64,
            max_pending: 64,
        },
        4,
    );
    let mut batch = RequestBatch::new();
    // Exactly one window of four frames: the planner emits four logical tasks
    // which the coordinator admits as one full chunk (one physical job).
    for f in 0..4 {
        batch.at(FrameIndex(f)).request(node);
    }
    let old_generation = driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    // Four logical members dispatched, first member confirmed inside the gate.
    assert_eq!(driver.in_flight(), 4);
    // Definition switch while the full chunk runs.
    state.edit(|e| e.replace(node, (), || 42_u64)).unwrap();
    let new_snapshot = state.snapshot();
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(0)).request(node);
    driver
        .begin(
            &mut eval,
            new_snapshot.clone(),
            batch,
            PlannerConfig::default(),
        )
        .unwrap();
    assert_ne!(old_generation, driver.generation());
    // The old generation token is dead after the replacement; cancel must
    // refuse it (the successor owns the session now).
    assert!(!driver.cancel(old_generation));
    // The new definition result must be the new value, never polluted by the
    // canceled chunk (the replaced plan's ready keys are dropped by cancel).
    let results = complete(&mut driver, &mut eval, &pool);
    assert_eq!(*results.get(id).unwrap(), 42);
    assert_eq!(results.snapshot_id(), new_snapshot.id());
    // Release the gate: all four obsolete logical members keep their quota
    // until their natural terminal messages, then release exactly once each.
    {
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let stats = driver.stats();
    assert_eq!(stats.obsolete, 4);
    // One obsolete old chunk (4 logical) + one terminal from the new request.
    assert_eq!(stats.terminal, 5);
}
#[test]
fn chunk4_obsolete_payload_drop_panic_keeps_residue_and_allows_replacement() {
    use std::sync::atomic::AtomicBool;
    struct Releaser {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            let (lock, ready) = &*self.gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    // Two independent gate positions: frame 0 blocks on gate 1, frame 1
    // (running behind it in the same physical job) blocks on gate 2.
    let gate1 = Arc::new((Mutex::new(false), Condvar::new()));
    let gate2 = Arc::new((Mutex::new(false), Condvar::new()));
    // Declared after the pool; both armed before the first fallible wait.
    let _releaser1 = Releaser {
        gate: gate1.clone(),
    };
    let _releaser2 = Releaser {
        gate: gate2.clone(),
    };
    // Armed exactly once: the frame-0 obsolete payload destructor panics.
    let flag = Arc::new(AtomicBool::new(true));
    // The replacement definition produces Payloads guarded by a fresh,
    // never-armed flag so reclaimed successors never panic on drop.
    let safe_flag = Arc::new(AtomicBool::new(false));
    let (started_tx, started_rx) = mpsc::channel();
    struct Payload {
        value: u64,
        destructor: Arc<AtomicBool>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            assert!(
                !self.destructor.swap(false, Ordering::Relaxed),
                "obsolete payload destructor"
            );
        }
    }
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            let gate1 = gate1.clone();
            let gate2 = gate2.clone();
            let flag = flag.clone();
            e.map(e.frame(), move |f: &FrameIndex| {
                let wait = |gate: &Arc<(Mutex<bool>, Condvar)>| {
                    let (lock, ready) = &**gate;
                    let mut open = lock.lock().unwrap();
                    while !*open {
                        open = ready.wait(open).unwrap();
                    }
                };
                match f.0 {
                    0 => {
                        let _ = started_tx.send(());
                        wait(&gate1);
                        Payload {
                            value: f.0,
                            destructor: flag.clone(),
                        }
                    }
                    1 => {
                        let _ = started_tx.send(());
                        wait(&gate2);
                        Payload {
                            value: f.0,
                            destructor: flag.clone(),
                        }
                    }
                    _ => Payload {
                        value: f.0,
                        destructor: flag.clone(),
                    },
                }
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(node, off()).unwrap();
    let snapshot = state.snapshot();
    let mut driver = AsyncDriver::new(
        snapshot.clone(),
        TaskLimits {
            max_in_flight: 4,
            max_pending: 4,
        },
        4,
    );
    let mut batch = RequestBatch::new();
    for f in 0..4 {
        batch.at(FrameIndex(f)).request(node);
    }
    let old_generation = driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    // Dispatch: four logical members; frame 0 blocks gate 1, frame 1 waits
    // gate 2, frames 2/3 run to completion right away in the same FIFO pass.
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("frame 0 never entered gate 1");
    // Replace with a fresh request BEFORE opening any gate, and poll once so
    // the new plan actually queues its task behind the occupied quota.
    let new_snapshot = {
        let safe_flag = safe_flag.clone();
        let _ = state
            .edit(|e| {
                e.replace(node, (), move || Payload {
                    value: 42,
                    destructor: safe_flag.clone(),
                })
            })
            .unwrap();
        state.snapshot()
    };
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(node);
    let new_generation = driver
        .begin(
            &mut eval,
            new_snapshot.clone(),
            batch,
            PlannerConfig::default(),
        )
        .unwrap();
    assert_ne!(old_generation, new_generation);
    // Four physical dispatch slots are occupied (running = 4), so the poll
    // admits the replacement's new task only into the pending queue.
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    let stats = driver.stats();
    assert_eq!(stats.dispatched, 4, "replacement must stay undispatched");
    // Release gate 1 only: frame 0 finishes and its obsolete payload
    // destructor panics exactly once (shared flag consumed).
    {
        let (lock, ready) = &*gate1;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("frame 1 never entered gate 2");
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loop {
            driver.poll(&mut eval, &pool).unwrap();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }))
        .is_err(),
        "obsolete payload destructor must panic the poll"
    );
    // The panicking poll terminated the active request and canceled its one
    // queued replacement task; frames 1/2/3 remain running as residue.
    assert!(!flag.load(Ordering::Relaxed));
    assert_eq!(driver.in_flight(), 3);
    let stats = driver.stats();
    assert_eq!(
        stats.canceled_pending, 1,
        "queued replacement must be canceled"
    );
    // The canceled active request refuses its token.
    assert!(!driver.cancel(new_generation));
    // Release gate 2 (the flag is exhausted; no further payload panics): all
    // four obsolete members drain to natural terminals.
    {
        let (lock, ready) = &*gate2;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let stats = driver.stats();
    assert_eq!(stats.obsolete, 4);
    assert_eq!(stats.terminal, 4);
    // A fresh request after full reclamation computes the NEW definition and
    // must deliver the actual new value 42 for the new snapshot.
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(0)).request(node);
    driver
        .begin(
            &mut eval,
            new_snapshot.clone(),
            batch,
            PlannerConfig::default(),
        )
        .unwrap();
    let results = complete(&mut driver, &mut eval, &pool);
    let payload = results.get_shared(id).unwrap();
    assert_eq!(payload.value, 42);
    assert_eq!(results.snapshot_id(), new_snapshot.id());
    drop(payload);
    assert!(!safe_flag.load(Ordering::Relaxed));
    assert_eq!(driver.in_flight(), 0);
    let stats = driver.stats();
    assert_eq!(stats.terminal, 5);
}
#[test]
fn chunk4_cache_commit_panic_terminates_plan_and_keeps_residue() {
    use std::sync::atomic::AtomicBool;
    struct Releaser {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }
    impl Drop for Releaser {
        fn drop(&mut self) {
            let (lock, ready) = &*self.gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }
    struct Victim {
        frame: u64,
        panic: Arc<AtomicBool>,
    }
    impl Drop for Victim {
        fn drop(&mut self) {
            assert!(
                !self.panic.swap(false, Ordering::Relaxed),
                "memo victim destructor"
            );
        }
    }
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let _releaser = Releaser { gate: gate.clone() };
    let flag = Arc::new(AtomicBool::new(false));
    let (started_tx, started_rx) = mpsc::channel();
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            let gate = gate.clone();
            let flag = flag.clone();
            e.map(e.frame(), move |f: &FrameIndex| {
                if f.0 == 2 {
                    // Ready order sorts by node then frame: frames 1,2,3,4 run
                    // FIFO, so frame 2 is the SECOND member. Holding it lets
                    // the frame-1 completion commit first and evict the warm
                    // victim while the plan is still active.
                    let (lock, ready) = &*gate;
                    let mut open = lock.lock().unwrap();
                    let _ = started_tx.send(());
                    while !*open {
                        open = ready.wait(open).unwrap();
                    }
                }
                Victim {
                    frame: f.0,
                    panic: flag.clone(),
                }
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    // The node stays Memo Last / Context Off for the whole fixture; policy is
    // read at commit time, so the warm frame-0 victim occupies the single
    // memo entry until a later commit evicts it.
    eval.configure_cache(
        node,
        NodeCacheConfig {
            memo: CachePolicy::Last,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    let mut warm_batch = RequestBatch::new();
    warm_batch.at(FrameIndex(0)).request(node);
    {
        let mut plan = eval
            .plan(state.snapshot(), warm_batch, PlannerConfig::default())
            .unwrap();
        let mut completed = Vec::new();
        while !plan.is_complete() {
            let tasks = eval.advance(&mut plan, completed).unwrap();
            completed = tasks.into_iter().map(|t| t.execute()).collect();
        }
        plan.finish().unwrap();
    }
    let calls_after_warm = eval.stats().computation_calls;
    assert_eq!(calls_after_warm, 1);
    // Arm the flag: the NEXT eviction of the frame-0 victim panics. Commits
    // still write into the same memo store (policy is read at commit time)
    // and evict the armed victim.
    flag.store(true, Ordering::Relaxed);
    let snapshot = state.snapshot();
    let mut driver = AsyncDriver::new(
        snapshot.clone(),
        TaskLimits {
            max_in_flight: 4,
            max_pending: 4,
        },
        4,
    );
    let mut batch = RequestBatch::new();
    for f in 1..5_u64 {
        batch.at(FrameIndex(f)).request(node);
    }
    let failed_generation = driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    // Frames 1..4 dispatch as one chunk; frame 2 blocks the gate after frame 1
    // delivered its completion.
    driver.poll(&mut eval, &pool).unwrap();
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loop {
            driver.poll(&mut eval, &pool).unwrap();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }))
        .is_err(),
        "evicted warm victim must panic the commit"
    );
    assert!(!flag.load(Ordering::Relaxed));
    // The commit panic terminated the plan; the frame-2 worker plus frames 3/4
    // remain as residue running.
    assert_eq!(driver.in_flight(), 3);
    // The failed generation cannot cancel its already-terminal plan.
    assert!(!driver.cancel(failed_generation));
    // Release the gate; every residue worker reaches one natural terminal.
    {
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let stats = driver.stats();
    // All four members dispatched and terminated (their results were never
    // published: the plan died at the commit panic).
    assert_eq!(stats.dispatched, 4);
    assert_eq!(stats.terminal, 4);
    // computation_calls counts accepted planned completions only: warm(1) +
    // f1(1). The canceled remainder (f2/f3/f4) runs as obsolete workers and
    // never publishes or counts.
    assert_eq!(eval.stats().computation_calls, calls_after_warm + 1);
    // A fresh request after the failure recomputes frame 1 (its memo entry
    // was evicted by the panicking commit) and delivers the real value.
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(1)).request(node);
    driver
        .begin(&mut eval, snapshot.clone(), batch, PlannerConfig::default())
        .unwrap();
    let results = complete(&mut driver, &mut eval, &pool);
    assert_eq!(results.get_shared(id).unwrap().frame, 1);
    assert_eq!(driver.in_flight(), 0);
    let stats = driver.stats();
    assert_eq!(stats.terminal, 5);
    // accepted planned completions: warm(1) + f1(1) + fresh f1(1) = 3 — the
    // same count contract as the original chunk-1 cache-commit fixture.
    assert_eq!(eval.stats().computation_calls, calls_after_warm + 2);
}

/// Lock-free counters shared by the scene closures.
struct BatchCounters {
    calls: AtomicUsize,
    items: AtomicUsize,
    max_len: AtomicUsize,
    scalar_runs: AtomicUsize,
    first_event: AtomicUsize,
}
impl Default for BatchCounters {
    fn default() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            items: AtomicUsize::new(0),
            max_len: AtomicUsize::new(0),
            scalar_runs: AtomicUsize::new(0),
            first_event: AtomicUsize::new(0),
        }
    }
}
/// Cooperative first-execution markers: whichever of the scalar or batch run
/// reaches its counter first claims the slot.
const EVENT_SCALAR: usize = 1;
const EVENT_BATCH: usize = 2;

/// Batch-kernel scene shared by the coalescing and ordering tests.
///
/// `actors` parameterized nodes share one `BatchKernel`. The optional non-batch
/// scalar node is inserted before them, so the canonical ready order (sorted by
/// node id) places it ahead of the coalesced batch run in the same chunk.
struct BatchScene {
    state: GraphState,
    scalar: Option<Node<u64>>,
    actors: Vec<Node<u64>>,
    counters: Arc<BatchCounters>,
}
impl BatchScene {
    fn new(actors: u64, with_scalar: bool) -> Self {
        let counters = Arc::new(BatchCounters::default());
        let mut state = GraphState::new();
        let (scalar, actor_nodes) = {
            let counters = counters.clone();
            state
                .edit(move |e| {
                    let source = e.input::<u64>();
                    e.set(source, 7)?;
                    let source_node = source.node();
                    let scalar = if with_scalar {
                        let counters = counters.clone();
                        Some(e.map(source_node, move |x: &u64| {
                            counters.scalar_runs.fetch_add(1, Ordering::Relaxed);
                            let _ = counters.first_event.compare_exchange(
                                0,
                                EVENT_SCALAR,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            );
                            x + 1
                        })?)
                    } else {
                        None
                    };
                    let kernel =
                        e.batch_kernel(move |items: &[(&u64, &u64)], out: &mut Vec<u64>| {
                            counters.calls.fetch_add(1, Ordering::Relaxed);
                            counters.items.fetch_add(items.len(), Ordering::Relaxed);
                            counters.max_len.fetch_max(items.len(), Ordering::Relaxed);
                            let _ = counters.first_event.compare_exchange(
                                0,
                                EVENT_BATCH,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            );
                            for &(&actor, &input) in items {
                                out.push(input.wrapping_add(actor).wrapping_mul(3));
                            }
                        });
                    let actors = (0..actors)
                        .map(|actor| e.map_batch(&kernel, actor, source_node))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((scalar, actors))
                })
                .unwrap()
        };
        Self {
            state,
            scalar,
            actors: actor_nodes,
            counters,
        }
    }

    fn reset(&self) {
        for counter in [
            &self.counters.calls,
            &self.counters.items,
            &self.counters.max_len,
            &self.counters.scalar_runs,
            &self.counters.first_event,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    fn calls(&self) -> usize {
        self.counters.calls.load(Ordering::Relaxed)
    }

    fn items(&self) -> usize {
        self.counters.items.load(Ordering::Relaxed)
    }

    fn max_len(&self) -> usize {
        self.counters.max_len.load(Ordering::Relaxed)
    }

    fn scalar_runs(&self) -> usize {
        self.counters.scalar_runs.load(Ordering::Relaxed)
    }

    fn first_event(&self) -> usize {
        self.counters.first_event.load(Ordering::Relaxed)
    }

    /// Requests every scene node once at frame 0, paired with its oracle value.
    fn request_all(&self) -> (RequestBatch, Vec<(u64, RequestId<u64>)>) {
        let mut batch = RequestBatch::new();
        let mut expects = Vec::new();
        if let Some(scalar) = self.scalar {
            let id = batch.at(FrameIndex(0)).request(scalar);
            expects.push((8, id));
        }
        for (actor, &node) in self.actors.iter().enumerate() {
            let id = batch.at(FrameIndex(0)).request(node);
            expects.push(((7 + actor as u64).wrapping_mul(3), id));
        }
        (batch, expects)
    }
}

#[test]
fn bevy_executor_batch_kernel_coalesces_up_to_chunk_size() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let scene = BatchScene::new(16, false);
    for chunk in [1usize, 4, 16] {
        scene.reset();
        let mut eval = Evaluator::new(&scene.state);
        let snapshot = scene.state.snapshot();
        let (batch, expects) = scene.request_all();
        let (results, _) = eval
            .bind(snapshot)
            .unwrap()
            .evaluate_planned(
                batch,
                PlannerConfig::default(),
                &mut BevyExecutor::new(&pool, chunk),
            )
            .unwrap();
        for (expected, id) in expects {
            assert_eq!(*results.get(id).unwrap(), expected, "chunk {chunk}");
        }
        // Coalescing is physical only: each node is still evaluated once.
        assert_eq!(eval.stats().computation_calls, 16);
        assert_eq!(scene.calls(), 16 / chunk, "chunk {chunk}");
        assert!(
            scene.max_len() <= chunk,
            "chunk {chunk} exceeded the chunk size"
        );
        assert_eq!(scene.max_len(), chunk, "chunk {chunk}");
        assert_eq!(scene.items(), 16, "chunk {chunk}");
    }
}

#[test]
fn async_driver_batch_kernel_coalesces_up_to_chunk_size_and_task_limits() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let scene = BatchScene::new(16, false);
    for chunk in [1usize, 4, 16] {
        scene.reset();
        let mut eval = Evaluator::new(&scene.state);
        let mut driver = AsyncDriver::new(
            scene.state.snapshot(),
            TaskLimits {
                max_in_flight: 16,
                max_pending: 32,
            },
            chunk,
        );
        let (batch, expects) = scene.request_all();
        driver
            .begin(
                &mut eval,
                scene.state.snapshot(),
                batch,
                PlannerConfig::default(),
            )
            .unwrap();
        let results = complete(&mut driver, &mut eval, &pool);
        for (expected, id) in expects {
            assert_eq!(*results.get(id).unwrap(), expected, "chunk {chunk}");
        }
        assert!(
            driver.stats().peak_in_flight <= 16,
            "chunk {chunk}: {:?}",
            driver.stats()
        );
        assert!(
            scene.max_len() <= chunk,
            "chunk {chunk} exceeded the chunk size"
        );
        assert_eq!(scene.items(), 16, "chunk {chunk}");
        // The first dispatch always fills the chunk, so peak coalescing is
        // bounded by the chunk size rather than per-task dispatch.
        assert_eq!(scene.max_len(), chunk, "chunk {chunk}");
    }

    // A tight in-flight limit caps coalescing below the chunk size while the
    // admitted work still respects the total task quota.
    scene.reset();
    let mut eval = Evaluator::new(&scene.state);
    let mut driver = AsyncDriver::new(
        scene.state.snapshot(),
        TaskLimits {
            max_in_flight: 2,
            max_pending: 64,
        },
        16,
    );
    let (batch, expects) = scene.request_all();
    driver
        .begin(
            &mut eval,
            scene.state.snapshot(),
            batch,
            PlannerConfig::default(),
        )
        .unwrap();
    let results = complete(&mut driver, &mut eval, &pool);
    for (expected, id) in expects {
        assert_eq!(*results.get(id).unwrap(), expected);
    }
    assert!(driver.stats().peak_in_flight <= 2, "{:?}", driver.stats());
    assert!(
        scene.max_len() <= 2,
        "limits must cap coalescing below the chunk size"
    );
    assert_eq!(scene.max_len(), 2);
    assert_eq!(scene.items(), 16);
}

#[test]
fn async_driver_cancellation_stops_batch_dispatch_within_limits() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    let scene = BatchScene::new(16, false);
    let mut eval = Evaluator::new(&scene.state);
    let mut driver = AsyncDriver::new(
        scene.state.snapshot(),
        TaskLimits {
            max_in_flight: 2,
            max_pending: 2,
        },
        16,
    );
    let (batch, _) = scene.request_all();
    let generation = driver
        .begin(
            &mut eval,
            scene.state.snapshot(),
            batch,
            PlannerConfig::default(),
        )
        .unwrap();
    // The first poll admits the four-task budget, but only one physical chunk
    // fits inside `max_in_flight`; the remaining two tasks stay pending.
    assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
    assert_eq!(driver.stats().dispatched, 2);
    assert!(driver.cancel(generation));
    assert!(!driver.cancel(generation));
    assert_eq!(driver.stats().canceled_pending, 2);
    // Canceling gates the plan immediately. The single running chunk still
    // drains to a natural terminal, but the canceled pending tasks never run.
    let deadline = Instant::now() + Duration::from_secs(5);
    while driver.in_flight() > 0 {
        assert!(driver.poll(&mut eval, &pool).unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(scene.calls(), 1, "canceled pending tasks must not run");
    assert_eq!(
        scene.max_len(),
        2,
        "the dispatched chunk coalesces both admitted tasks"
    );
    assert_eq!(scene.items(), 2);
    let stats = driver.stats();
    assert_eq!(stats.terminal, 2);
    assert_eq!(stats.obsolete, 2);
    assert_eq!(stats.failed, 0);
    // A canceled generation cannot be revived.
    assert!(!driver.cancel(generation));
}

#[test]
fn non_batch_tasks_finish_before_gated_batch_successors_in_one_chunk() {
    let pool = TaskPoolBuilder::new().num_threads(4).build();
    // One non-batch scalar plus eight batch nodes: nine tasks in a single chunk.
    let scene = BatchScene::new(8, true);

    // Synchronous parallel executor: the scalar runs, then the coalesced run.
    scene.reset();
    let mut eval = Evaluator::new(&scene.state);
    let (batch, expects) = scene.request_all();
    let (results, _) = eval
        .bind(scene.state.snapshot())
        .unwrap()
        .evaluate_planned(
            batch,
            PlannerConfig::default(),
            &mut BevyExecutor::new(&pool, 16),
        )
        .unwrap();
    for (expected, id) in expects {
        assert_eq!(*results.get(id).unwrap(), expected);
    }
    assert_eq!(scene.scalar_runs(), 1);
    assert_eq!(scene.calls(), 1);
    assert_eq!(
        scene.first_event(),
        EVENT_SCALAR,
        "BevyExecutor must finish the non-batch task before the batch run"
    );

    // Nonblocking planner adapter: the scalar completion is delivered before
    // the coalesced batch run inside the same physical chunk.
    scene.reset();
    let mut eval = Evaluator::new(&scene.state);
    let mut driver = AsyncDriver::new(
        scene.state.snapshot(),
        TaskLimits {
            max_in_flight: 16,
            max_pending: 16,
        },
        16,
    );
    let (batch, expects) = scene.request_all();
    driver
        .begin(
            &mut eval,
            scene.state.snapshot(),
            batch,
            PlannerConfig::default(),
        )
        .unwrap();
    let results = complete(&mut driver, &mut eval, &pool);
    for (expected, id) in expects {
        assert_eq!(*results.get(id).unwrap(), expected);
    }
    assert_eq!(scene.scalar_runs(), 1);
    assert_eq!(scene.calls(), 1);
    assert_eq!(
        scene.first_event(),
        EVENT_SCALAR,
        "AsyncDriver must finish the non-batch task before the batch run"
    );
}
