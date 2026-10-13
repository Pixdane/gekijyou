use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_tasks::TaskPoolBuilder;
use bevy_transform::components::Transform;
use gekijyou_bevy::{
    AnimationRuntime, AsyncDriver, BevyExecutor, BindingError, Bindings, FrameStatus, OutputField,
    TaskLimits,
};
use gekijyou_computation::{
    CachePolicy, EvalError, Evaluator, FrameIndex, GraphState, NodeCacheConfig, PlannerConfig,
    RequestBatch,
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
fn runtime_cancel_keeps_old_quota_and_never_applies_old_definition() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    let mut runtime = AnimationRuntime::new(2);
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (start_tx, start_rx) = mpsc::channel();
    let node = runtime
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
    runtime.configure_cache(node, off()).unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_output(
            runtime.snapshot(),
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    runtime
        .enable_async(TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        })
        .unwrap();
    runtime
        .begin_prepare(&bindings, &[FrameIndex(0)], PlannerConfig::default())
        .unwrap();
    assert!(!runtime.poll_prepare(&pool, &bindings).unwrap());
    start_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    runtime
        .edit(|e| e.replace(node, (), || Vec3::splat(2.)))
        .unwrap();
    for _ in 0..20 {
        runtime
            .begin_prepare(&bindings, &[FrameIndex(0)], PlannerConfig::default())
            .unwrap();
        runtime.poll_prepare(&pool, &bindings).unwrap();
        assert_eq!(runtime.async_in_flight(), 1);
        assert!(bindings.apply(&mut world, &runtime, FrameIndex(0)).is_err());
    }
    {
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while runtime.prepared().status(FrameIndex(0)) != FrameStatus::Ready {
        runtime.poll_prepare(&pool, &bindings).unwrap();
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    bindings.apply(&mut world, &runtime, FrameIndex(0)).unwrap();
    assert_eq!(
        world.get::<Transform>(entity).unwrap().translation,
        Vec3::splat(2.)
    );
    let stats = runtime.async_task_stats().unwrap();
    assert_eq!(stats.peak_in_flight, 1);
    assert_eq!(stats.terminal, 2);
    assert_eq!(stats.obsolete, 1);
    assert!(stats.canceled_pending >= 19);
}
#[test]
fn async_worker_failure_is_visible_as_a_failed_complete_frame() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut runtime = AnimationRuntime::new(1);
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let node = runtime
        .edit(|e| e.map((), || -> Vec3 { panic!("async") }))
        .unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_output(
            runtime.snapshot(),
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    runtime
        .enable_async(TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        })
        .unwrap();
    runtime
        .begin_prepare(&bindings, &[FrameIndex(0)], PlannerConfig::default())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match runtime.poll_prepare(&pool, &bindings) {
            Ok(_) => (),
            Err(BindingError::Evaluation(EvalError::WorkerPanicked)) => break,
            other => panic!("Unexpected poll {other:?}"),
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(runtime.async_in_flight(), 0);
    assert_eq!(runtime.async_task_stats().unwrap().failed, 1);
    assert_eq!(
        runtime.prepared().status(FrameIndex(0)),
        FrameStatus::Failed
    );
    assert!(bindings.apply(&mut world, &runtime, FrameIndex(0)).is_err());
}
#[test]
fn async_driver_replacing_a_plan_in_the_same_generation_discards_old_messages() {
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(3_u64))).unwrap();
    let mut eval = Evaluator::new(&state);
    let runtime = AnimationRuntime::new(1);
    let mut driver = AsyncDriver::new(
        state.snapshot(),
        runtime.generation(),
        TaskLimits {
            max_in_flight: 1,
            max_pending: 1,
        },
    );
    let mut old = RequestBatch::new();
    old.at(FrameIndex(0)).request(node);
    driver
        .begin(
            &mut eval,
            state.snapshot(),
            runtime.generation(),
            old,
            PlannerConfig::default(),
        )
        .unwrap();
    driver.poll(&mut eval, &pool).unwrap();
    let mut new = RequestBatch::new();
    let id = new.at(FrameIndex(1)).request(node);
    driver
        .begin(
            &mut eval,
            state.snapshot(),
            runtime.generation(),
            new,
            PlannerConfig::default(),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Some((results, _)) = driver.poll(&mut eval, &pool).unwrap() {
            assert_eq!(results.get(id).unwrap(), &3);
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(driver.in_flight(), 0);
}

#[test]
fn cache_commit_panic_reclaims_worker_quota_and_allows_a_fresh_request() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::AtomicBool;
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
    #[derive(Component)]
    struct Output(u64);
    let pool = TaskPoolBuilder::new().num_threads(1).build();
    let mut world = World::new();
    let entity = world.spawn(Output(0)).id();
    let flag = Arc::new(AtomicBool::new(false));
    let mut runtime = AnimationRuntime::new(1);
    let node = runtime
        .edit(|e| {
            let panic = flag.clone();
            e.map(e.frame(), move |f: &FrameIndex| Value {
                frame: f.0,
                panic: panic.clone(),
            })
        })
        .unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_output(
            runtime.snapshot(),
            entity,
            node,
            OutputField::new(|c: &mut Output, v: &Value| c.0 = v.frame),
        )
        .unwrap();
    runtime.prepare(&bindings, &[FrameIndex(0)]).unwrap();
    runtime
        .enable_async(TaskLimits {
            max_in_flight: 1,
            max_pending: 0,
        })
        .unwrap();
    runtime
        .begin_prepare(&bindings, &[FrameIndex(1)], PlannerConfig::default())
        .unwrap();
    runtime.poll_prepare(&pool, &bindings).unwrap();
    flag.store(true, Ordering::Relaxed);
    let deadline = Instant::now() + Duration::from_secs(2);
    assert!(
        catch_unwind(AssertUnwindSafe(|| loop {
            runtime.poll_prepare(&pool, &bindings).unwrap();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }))
        .is_err()
    );
    assert!(!flag.load(Ordering::Relaxed));
    assert_eq!(runtime.async_in_flight(), 0);
    assert_eq!(
        runtime.frame_status(&bindings, FrameIndex(1)),
        FrameStatus::Failed
    );
    runtime
        .begin_prepare(&bindings, &[FrameIndex(1)], PlannerConfig::default())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while runtime.frame_status(&bindings, FrameIndex(1)) != FrameStatus::Ready {
        runtime.poll_prepare(&pool, &bindings).unwrap();
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    bindings.apply(&mut world, &runtime, FrameIndex(1)).unwrap();
    assert_eq!(world.get::<Output>(entity).unwrap().0, 1);
    assert_eq!(runtime.async_task_stats().unwrap().terminal, 2);
    assert_eq!(runtime.stats().computation_calls, 3);
}
