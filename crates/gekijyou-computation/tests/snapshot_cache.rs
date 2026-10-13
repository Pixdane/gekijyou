use gekijyou_computation::{
    CachePolicy, EvalError, Evaluator, FrameIndex, GraphError, GraphState, NodeCacheConfig,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

fn config(memo: CachePolicy, context: CachePolicy) -> NodeCacheConfig {
    NodeCacheConfig { memo, context }
}
#[test]
fn content_history_survives_input_changes_with_independent_cache_layers() {
    for (memo, context) in [
        (CachePolicy::Lru { capacity: 3 }, CachePolicy::Off),
        (CachePolicy::Off, CachePolicy::Lru { capacity: 3 }),
        (CachePolicy::Last, CachePolicy::Lru { capacity: 3 }),
    ] {
        let mut state = GraphState::new();
        let (input, node) = state
            .edit(|e| {
                e.set_default_eq::<u64>();
                let i = e.input();
                e.set(i, 10_u64)?;
                let n = e.map(i.node(), |x: &u64| x * 2)?;
                Ok((i, n))
            })
            .unwrap();
        let epoch = state.snapshot().definition_epoch();
        let mut eval = Evaluator::new(&state);
        eval.configure_cache(node, config(memo, context)).unwrap();
        for value in [10, 20, 10] {
            state.edit(|e| e.set(input, value)).unwrap();
            assert_eq!(state.snapshot().definition_epoch(), epoch);
            assert_eq!(
                *eval
                    .bind(state.snapshot())
                    .unwrap()
                    .sample(FrameIndex(300), node)
                    .unwrap(),
                value * 2
            );
        }
        assert_eq!(eval.stats().computation_calls, 2);
        assert_eq!(eval.stats().definition_syncs, 0);
        if context != CachePolicy::Off {
            assert_eq!(eval.stats().projection_builds, 1);
            assert_eq!(eval.stats().context_hits, 1);
        }
    }
}
#[test]
fn identity_history_only_reuses_the_same_shared_object() {
    let mut state = GraphState::new();
    let (input, node) = state
        .edit(|e| {
            let i = e.input::<u64>();
            let n = e.map(i.node(), |x: &u64| x * 2)?;
            Ok((i, n))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        node,
        config(CachePolicy::Off, CachePolicy::Lru { capacity: 8 }),
    )
    .unwrap();
    let shared = Arc::new(10_u64);
    for value in [shared.clone(), Arc::new(10), Arc::new(20), shared.clone()] {
        state.edit(|e| e.set_shared(input, value)).unwrap();
        eval.bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), node)
            .unwrap();
    }
    assert_eq!(eval.stats().computation_calls, 3);
    assert_eq!(eval.stats().context_hits, 1);
}
#[test]
fn newer_definition_clears_context_and_keeps_unchanged_direct_memo() {
    let mut state = GraphState::new();
    let (source, target) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let s = e.constant(5_u64);
            let t = e.map(s, |x: &u64| x + 1)?;
            Ok((s, t))
        })
        .unwrap();
    let old = state.snapshot();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(target, config(CachePolicy::Last, CachePolicy::Last))
        .unwrap();
    assert_eq!(
        *eval
            .bind(old.clone())
            .unwrap()
            .sample(FrameIndex(0), target)
            .unwrap(),
        6
    );
    state.edit(|e| e.replace(source, (), || 5_u64)).unwrap();
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), target)
            .unwrap(),
        6
    );
    assert_eq!(eval.stats().context_hits, 0);
    assert_eq!(eval.stats().memo_hits, 1);
    assert!(matches!(
        eval.bind(old.clone()),
        Err(EvalError::StaleGraphDefinition { .. })
    ));
    assert_eq!(
        *Evaluator::from_snapshot(&old)
            .bind(old)
            .unwrap()
            .sample(FrameIndex(0), target)
            .unwrap(),
        6
    );
    state
        .edit(|e| {
            e.constant(99_u32);
            Ok(())
        })
        .unwrap();
    eval.bind(state.snapshot())
        .unwrap()
        .sample(FrameIndex(0), target)
        .unwrap();
    assert_eq!(eval.stats().projection_builds, 3);
}
#[test]
fn policies_validate_and_change_each_layer_without_mutating_graph() {
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(2_u64))).unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    assert_eq!(
        eval.configure_cache(
            node,
            config(CachePolicy::Lru { capacity: 0 }, CachePolicy::Off)
        ),
        Err(GraphError::InvalidCacheCapacity)
    );
    assert!(
        eval.configure_cache(
            state.graph().frame(),
            config(CachePolicy::Last, CachePolicy::Last)
        )
        .is_err()
    );
    eval.configure_cache(node, config(CachePolicy::Last, CachePolicy::Last))
        .unwrap();
    eval.bind(snapshot.clone())
        .unwrap()
        .sample(FrameIndex(0), node)
        .unwrap();
    assert_eq!(eval.cache_stats().memo_entries, 1);
    assert_eq!(eval.cache_stats().context_entries, 1);
    eval.configure_cache(node, config(CachePolicy::Off, CachePolicy::Last))
        .unwrap();
    assert_eq!(eval.cache_stats().memo_entries, 0);
    assert_eq!(eval.cache_stats().context_entries, 1);
    eval.configure_cache(node, config(CachePolicy::Off, CachePolicy::Off))
        .unwrap();
    assert_eq!(eval.cache_stats().context_entries, 0);
    assert_eq!(state.snapshot().id(), snapshot.id());
}
#[test]
fn missing_roots_preserve_dependency_failure_order() {
    let mut state = GraphState::new();
    let (input, early, output) = state
        .edit(|e| {
            let i = e.input::<u64>();
            e.set(i, 2)?;
            let early = e.map(e.frame(), |f: &FrameIndex| {
                if f.0 == 1 {
                    panic!("first dependency");
                }
                1_u64
            })?;
            let out = e.map((early, i.node()), |a: &u64, b: &u64| a + b)?;
            Ok((i, early, out))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(output, config(CachePolicy::Last, CachePolicy::Last))
        .unwrap();
    eval.bind(state.snapshot())
        .unwrap()
        .sample(FrameIndex(0), output)
        .unwrap();
    state
        .edit(|e| {
            e.remove(input.node())?;
            let missing = e.input::<u64>();
            e.replace(output, (early, missing.node()), |a: &u64, b: &u64| a + b)
        })
        .unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            eval.bind(state.snapshot())
                .unwrap()
                .sample(FrameIndex(1), output)
                .unwrap();
        }))
        .is_err()
    );
    assert!(matches!(
        eval.bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), output),
        Err(EvalError::MissingInput { .. })
    ));
}
#[test]
fn comparator_unwind_preserves_valid_history_and_frozen_semantics() {
    let panic = Arc::new(AtomicBool::new(false));
    let mut state = GraphState::new();
    let (input, node) = state
        .edit(|e| {
            let panic = panic.clone();
            e.set_default_equivalence::<u64, _>(move |a, b| {
                assert!(!panic.load(Ordering::Relaxed), "comparison");
                a == b
            });
            let i = e.input();
            e.set(i, 1_u64)?;
            let n = e.map(i.node(), |x: &u64| x * 2)?;
            Ok((i, n))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        node,
        config(
            CachePolicy::Lru { capacity: 2 },
            CachePolicy::Lru { capacity: 2 },
        ),
    )
    .unwrap();
    eval.bind(state.snapshot())
        .unwrap()
        .sample(FrameIndex(0), node)
        .unwrap();
    state
        .edit(|e| {
            e.set_default_equivalence::<u64, _>(|_, _| false);
            e.set(input, 1)
        })
        .unwrap();
    panic.store(true, Ordering::Relaxed);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            eval.bind(state.snapshot())
                .unwrap()
                .sample(FrameIndex(0), node)
                .unwrap();
        }))
        .is_err()
    );
    assert_eq!(eval.cache_stats().context_entries, 1);
    panic.store(false, Ordering::Relaxed);
    eval.bind(state.snapshot())
        .unwrap()
        .sample(FrameIndex(0), node)
        .unwrap();
    assert_eq!(eval.stats().computation_calls, 1);
}
struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
#[test]
fn deleted_nodes_release_caches_while_results_and_old_snapshots_survive() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            let drops = drops.clone();
            e.map((), move || Probe(drops.clone()))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(node, config(CachePolicy::Last, CachePolicy::Last))
        .unwrap();
    let old = state.snapshot();
    let output = eval
        .bind(old.clone())
        .unwrap()
        .sample(FrameIndex(0), node)
        .unwrap();
    state.edit(|e| e.remove(node)).unwrap();
    eval.bind(state.snapshot()).unwrap();
    assert_eq!(eval.cache_stats().memo_entries, 0);
    assert_eq!(eval.cache_stats().context_entries, 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(output);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let value = Evaluator::from_snapshot(&old)
        .bind(old)
        .unwrap()
        .sample(FrameIndex(0), node)
        .unwrap();
    drop(value);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}
#[test]
fn dfs_reports_a_supported_depth_limit_and_can_be_reused() {
    let mut state = GraphState::new();
    let (root, deep) = state
        .edit(|e| {
            let root = e.constant(0_u64);
            let mut n = root;
            for _ in 0..300 {
                n = e.map(n, |x: &u64| x + 1)?;
            }
            Ok((root, n))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    assert!(matches!(
        eval.bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), deep),
        Err(EvalError::DepthLimit { .. })
    ));
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), root)
            .unwrap(),
        0
    );
}
