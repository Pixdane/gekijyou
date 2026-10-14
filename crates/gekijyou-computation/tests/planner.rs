use gekijyou_computation::{
    CachePolicy, EvalError, Evaluator, FrameIndex, GraphError, GraphState, NodeCacheConfig,
    PlannerConfig, RequestBatch, SerialExecutor,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
fn off() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Off,
        context: CachePolicy::Off,
    }
}
#[test]
fn invalid_new_definition_batch_preserves_old_scope_and_cache_history() {
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(7_u64))).unwrap();
    let old = state.snapshot();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        node,
        NodeCacheConfig {
            memo: CachePolicy::Last,
            context: CachePolicy::Last,
        },
    )
    .unwrap();
    assert_eq!(
        *eval
            .bind(old.clone())
            .unwrap()
            .sample(FrameIndex(0), node)
            .unwrap(),
        7
    );
    let stats = eval.stats();
    let caches = eval.cache_stats();
    let deleted = state.edit(|e| Ok(e.constant(9_u64))).unwrap();
    state.edit(|e| e.remove(deleted)).unwrap();
    let mut foreign = GraphState::new();
    let foreign_node = foreign.edit(|e| Ok(e.constant(3_u64))).unwrap();
    for (invalid_node, expected) in [
        (
            foreign_node,
            GraphError::ForeignNode {
                node: foreign_node.id(),
            },
        ),
        (deleted, GraphError::MissingNode { node: deleted.id() }),
    ] {
        let mut batch = RequestBatch::new();
        batch.at(FrameIndex(0)).request(invalid_node);
        assert!(
            matches!(eval.plan(state.snapshot(), batch, PlannerConfig::default()), Err(EvalError::Graph(error)) if error == expected)
        );
        assert_eq!(eval.stats(), stats);
        assert_eq!(eval.cache_stats(), caches);
    }
    assert_eq!(
        *eval.bind(old).unwrap().sample(FrameIndex(0), node).unwrap(),
        7
    );
    assert_eq!(eval.stats().context_hits, stats.context_hits + 1);
    assert_eq!(eval.stats().computation_calls, stats.computation_calls);
}
#[test]
fn planner_matches_dfs_and_shares_content_then_identity_within_a_window() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let (phase, expensive, output) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let phase = e.map(e.frame(), |f: &FrameIndex| f.0 % 3)?;
            let calls = calls.clone();
            let expensive = e.map(phase, move |p: &u64| {
                calls.fetch_add(1, Ordering::Relaxed);
                p * 10
            })?;
            let output = e.map(expensive, |p: &u64| p + 1)?;
            Ok((phase, expensive, output))
        })
        .unwrap();
    let frames = [120, 30, 121, 122, 300, 120];
    let mut dfs = Evaluator::new(&state);
    let mut planner = Evaluator::new(&state);
    for n in [phase, expensive, output] {
        dfs.configure_cache(n, off()).unwrap();
        planner.configure_cache(n, off()).unwrap();
    }
    let expected: Vec<_> = frames
        .iter()
        .map(|&f| {
            *dfs.bind(state.snapshot())
                .unwrap()
                .sample(FrameIndex(f), output)
                .unwrap()
        })
        .collect();
    calls.store(0, Ordering::Relaxed);
    let mut batch = RequestBatch::new();
    let ids: Vec<_> = frames
        .iter()
        .map(|&f| batch.at(FrameIndex(f)).request(output))
        .collect();
    let (results, stats) = planner
        .bind(state.snapshot())
        .unwrap()
        .evaluate_planned(batch, PlannerConfig::default(), &mut SerialExecutor)
        .unwrap();
    for (id, value) in ids.iter().zip(expected) {
        assert_eq!(*results.get(*id).unwrap(), value);
    }
    assert!(Arc::ptr_eq(
        &results.get_shared(ids[0]).unwrap(),
        &results.get_shared(ids[5]).unwrap()
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert!(stats.content_hits > 0);
    assert!(stats.identity_hits > 0);
    assert_eq!(stats.windows, 1);
    assert_eq!(planner.cache_stats().memo_entries, 0);
}
#[test]
fn static_nodes_share_in_a_window_and_may_repeat_across_windows_with_off_caches() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut state = GraphState::new();
    let (constant, output) = state
        .edit(|e| {
            let calls = calls.clone();
            let c = e.map((), move || {
                calls.fetch_add(1, Ordering::Relaxed);
                5_u64
            })?;
            let o = e.map((c, e.frame()), |x: &u64, f: &FrameIndex| x + f.0)?;
            Ok((c, o))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(constant, off()).unwrap();
    eval.configure_cache(output, off()).unwrap();
    let mut batch = RequestBatch::new();
    let ids: Vec<_> = (0..7)
        .map(|f| batch.at(FrameIndex(f)).request(output))
        .collect();
    let config = PlannerConfig {
        frame_window: 2,
        max_in_flight: 1,
        ..Default::default()
    };
    let (results, stats) = eval
        .bind(state.snapshot())
        .unwrap()
        .evaluate_planned(batch, config, &mut SerialExecutor)
        .unwrap();
    for (f, id) in ids.into_iter().enumerate() {
        assert_eq!(*results.get(id).unwrap(), 5 + f as u64);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 4);
    assert_eq!(stats.windows, 4);
    assert_eq!(stats.peak_in_flight, 1);
}
#[test]
fn zero_content_budget_preserves_strict_identity_sharing_without_false_merges() {
    let mut state = GraphState::new();
    let (phase, node) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let p = e.map(e.frame(), |f: &FrameIndex| f.0 % 2)?;
            let n = e.map(p, |p: &u64| p * 10)?;
            Ok((p, n))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(phase, off()).unwrap();
    eval.configure_cache(node, off()).unwrap();
    let mut batch = RequestBatch::new();
    let ids: Vec<_> = (0..32)
        .map(|f| batch.at(FrameIndex(f)).request(node))
        .collect();
    let (results, stats) = eval
        .bind(state.snapshot())
        .unwrap()
        .evaluate_planned(
            batch,
            PlannerConfig {
                frame_window: 32,
                comparison_budget: 0,
                ..Default::default()
            },
            &mut SerialExecutor,
        )
        .unwrap();
    assert_eq!(stats.equivalence_comparisons, 0);
    assert_eq!(stats.compute_tasks, 64);
    for (f, id) in ids.into_iter().enumerate() {
        assert_eq!(*results.get(id).unwrap(), (f as u64 % 2) * 10);
    }
}
#[test]
fn registry_budget_failure_is_terminal_and_releases_temporary_results() {
    let mut state = GraphState::new();
    let node = state
        .edit(|e| e.map(e.frame(), |f: &FrameIndex| f.0))
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(node, off()).unwrap();
    let mut batch = RequestBatch::new();
    for f in 0..4 {
        batch.at(FrameIndex(f)).request(node);
    }
    let mut plan = eval
        .plan(
            state.snapshot(),
            batch,
            PlannerConfig {
                max_registry_entries: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(matches!(
        eval.advance(&mut plan, Vec::new()),
        Err(EvalError::PlannerBudgetExceeded { .. })
    ));
    assert!(plan.failure().is_some());
    assert!(plan.finish().is_err());
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(9), node)
            .unwrap(),
        9
    );
}
#[test]
fn old_plan_stops_reads_and_cannot_commit_after_definition_switch() {
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(1_u64))).unwrap();
    let mut eval = Evaluator::new(&state);
    let old = state.snapshot();
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(node);
    let mut plan = eval.plan(old, batch, PlannerConfig::default()).unwrap();
    let tasks = eval.advance(&mut plan, Vec::new()).unwrap();
    assert_eq!(tasks.len(), 1);
    state.edit(|e| e.replace(node, (), || 2_u64)).unwrap();
    eval.bind(state.snapshot()).unwrap();
    assert!(matches!(
        eval.advance(&mut plan, tasks.into_iter().map(|t| t.execute()).collect()),
        Err(EvalError::StaleGraphDefinition { .. })
    ));
    assert_eq!(eval.cache_stats().memo_entries, 0);
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), node)
            .unwrap(),
        2
    );
}
#[test]
fn cache_policy_changed_after_dispatch_is_observed_at_completion() {
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(7_u64))).unwrap();
    let mut eval = Evaluator::new(&state);
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(0)).request(node);
    let mut plan = eval
        .plan(state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    let tasks = eval.advance(&mut plan, Vec::new()).unwrap();
    eval.configure_cache(node, off()).unwrap();
    eval.advance(&mut plan, tasks.into_iter().map(|t| t.execute()).collect())
        .unwrap();
    assert_eq!(*plan.finish().unwrap().0.get(id).unwrap(), 7);
    assert_eq!(eval.cache_stats().memo_entries, 0);
}
#[test]
fn worker_panic_fails_the_plan_and_never_publishes_partial_results() {
    let mut state = GraphState::new();
    let (panic, next) = state
        .edit(|e| {
            let p = e.map((), || -> u64 { panic!("worker") })?;
            let n = e.map(p, |x: &u64| x + 1)?;
            Ok((p, n))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(panic, off()).unwrap();
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(next);
    assert!(matches!(
        eval.bind(state.snapshot()).unwrap().evaluate_planned(
            batch,
            PlannerConfig::default(),
            &mut SerialExecutor
        ),
        Err(EvalError::WorkerPanicked)
    ));
    assert_eq!(eval.cache_stats().memo_entries, 0);
}
