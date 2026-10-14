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
#[test]
fn missing_unset_input_reports_originating_request_frame_not_frame_zero() {
    let mut state = GraphState::new();
    let (input, output) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let input = e.input::<u64>();
            let mid = e.map(input.node(), |x: &u64| x + 1)?;
            let output = e.map(mid, |x: &u64| x * 2)?;
            Ok((input, output))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    for target in [input.node(), output] {
        let mut batch = RequestBatch::new();
        batch.at(FrameIndex(42)).request(target);
        let error = eval
            .bind(state.snapshot())
            .unwrap()
            .evaluate_planned(batch, PlannerConfig::default(), &mut SerialExecutor)
            .err()
            .unwrap();
        assert_eq!(
            error,
            EvalError::MissingInput {
                frame: FrameIndex(42),
                node: input.node().id()
            }
        );
        assert_eq!(
            eval.bind(state.snapshot())
                .unwrap()
                .sample(FrameIndex(42), target),
            Err(EvalError::MissingInput {
                frame: FrameIndex(42),
                node: input.node().id()
            })
        );
    }
}
#[test]
fn static_dependency_missing_input_reports_first_reached_request_frame() {
    let mut state = GraphState::new();
    let (healthy, missing_input, output) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let constant = e.map((), || 5_u64)?;
            let input = e.input::<u64>();
            let missing = e.map((constant, input.node()), |c: &u64, x: &u64| c + x)?;
            let healthy = e.map(constant, |c: &u64| c + 1)?;
            let output = e.map(
                (healthy, missing, e.frame()),
                |h: &u64, m: &u64, f: &FrameIndex| h + m + f.0,
            )?;
            Ok((healthy, input, output))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(healthy, off()).unwrap();
    for frame_window in [1, PlannerConfig::default().frame_window] {
        let mut batch = RequestBatch::new();
        batch.at(FrameIndex(7)).request(healthy);
        batch.at(FrameIndex(43)).request(output);
        batch.at(FrameIndex(42)).request(output);
        let config = PlannerConfig {
            frame_window,
            ..Default::default()
        };
        let error = eval
            .bind(state.snapshot())
            .unwrap()
            .evaluate_planned(batch, config, &mut SerialExecutor)
            .err()
            .unwrap();
        assert_eq!(
            error,
            EvalError::MissingInput {
                frame: FrameIndex(43),
                node: missing_input.node().id()
            }
        );
    }
}

#[test]
fn full_registry_defers_misses_and_delivers_cached_hits_with_zero_slots() {
    let mut state = GraphState::new();
    let (miss_a, miss_b, downstream) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            // Creation order fixes the sorted Ready order: miss_a first.
            let miss_a = e.map((), || 1_u64)?;
            let miss_b = e.map((), || 2_u64)?;
            let downstream = e.map(miss_a, |x: &u64| x + 1)?;
            Ok((miss_a, miss_b, downstream))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(miss_a, off()).unwrap();
    eval.configure_cache(miss_b, off()).unwrap();
    // Memo-only: the downstream stays out of the start-window context lookup
    // and is instead delivered through the Ready drain as a real memo hit.
    eval.configure_cache(
        downstream,
        NodeCacheConfig {
            memo: CachePolicy::Last,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    // Warm the cached downstream (executes miss_a's closure and downstream once).
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), downstream)
            .unwrap(),
        2
    );
    let memo_hits_before = eval.stats().memo_hits;
    let context_hits_before = eval.stats().context_hits;

    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(miss_a);
    batch.at(FrameIndex(0)).request(miss_b);
    batch.at(FrameIndex(0)).request(downstream);
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
    // Zero slots: both misses defer; the registry must not trip while no task
    // can actually be created.
    let tasks = eval.advance_with_budget(&mut plan, Vec::new(), 0).unwrap();
    assert!(tasks.is_empty());
    assert!(!plan.is_complete());
    assert_eq!(plan.stats().compute_tasks, 0);
    // One slot: miss_a dispatches and fills the registry; miss_b defers behind
    // the per-call dispatch budget before any registry check.
    let tasks = eval.advance_with_budget(&mut plan, Vec::new(), 1).unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(plan.stats().compute_tasks, 1);
    // The completion is delivered with zero slots: miss_b stays deferred while
    // the cached downstream hit advances. No error, no new task.
    let completions: Vec<_> = tasks.into_iter().map(|t| t.execute()).collect();
    let tasks = eval.advance_with_budget(&mut plan, completions, 0).unwrap();
    assert!(tasks.is_empty());
    assert!(!plan.is_complete());
    // miss_a's completion unblocks the downstream, which is delivered as a
    // memo hit in the same Ready drain while miss_b stays deferred.
    assert_eq!(eval.stats().memo_hits, memo_hits_before + 1);
    assert_eq!(eval.stats().context_hits, context_hits_before);
    assert_eq!(plan.stats().compute_tasks, 1);
    // The delivered hit stored its output: a later sample of `downstream`
    // re-executes only the Off/Off miss_a (miss_a has no cache) and then hits
    // the downstream memo. 2 warm + 1 planned + 1 sample re-execution.
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), downstream)
            .unwrap(),
        2
    );
    assert_eq!(eval.stats().computation_calls, 4);
    // Only now, with a slot for the deferred miss against the full registry,
    // does the capacity error fire at the real miss.
    assert!(matches!(
        eval.advance_with_budget(&mut plan, Vec::new(), 1),
        Err(EvalError::PlannerBudgetExceeded { limit: 1 })
    ));
    assert!(plan.failure().is_some());
    assert!(plan.finish().is_err());
    // A fresh request still evaluates.
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), miss_b)
            .unwrap(),
        2
    );
}

#[test]
fn zero_budget_advance_completes_from_memo_hits_without_new_tasks() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let (constant, output) = state
        .edit(|e| {
            let calls = calls.clone();
            let c = e.map((), move || {
                calls.fetch_add(1, Ordering::Relaxed);
                5_u64
            })?;
            let o = e.map(c, |x: &u64| x + 1)?;
            Ok((c, o))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    let warm = NodeCacheConfig {
        memo: CachePolicy::Last,
        context: CachePolicy::Off,
    };
    eval.configure_cache(constant, warm).unwrap();
    eval.configure_cache(output, warm).unwrap();
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), output)
            .unwrap(),
        6
    );
    let warm_calls = eval.stats().computation_calls;
    assert_eq!(warm_calls, 2);
    let mut batch = RequestBatch::new();
    let id = batch.at(FrameIndex(0)).request(output);
    let mut plan = eval
        .plan(state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    let tasks = eval.advance_with_budget(&mut plan, Vec::new(), 0).unwrap();
    assert!(tasks.is_empty());
    assert!(plan.is_complete());
    assert_eq!(plan.stats().compute_tasks, 0);
    assert_eq!(eval.stats().computation_calls, warm_calls);
    let (results, _) = plan.finish().unwrap();
    assert_eq!(*results.get(id).unwrap(), 6);
}

#[test]
fn zero_budget_processes_hits_behind_an_uncached_miss_and_defers_only_misses() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let (uncached, cached, downstream) = state
        .edit(|e| {
            let uncached_calls = calls.clone();
            let uncached = e.map((), move || {
                uncached_calls.fetch_add(1, Ordering::Relaxed);
                100_u64
            })?;
            let cached = e.map((), || 200_u64)?;
            let downstream = e.map(cached, |x: &u64| x + 1)?;
            Ok((uncached, cached, downstream))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(uncached, off()).unwrap();
    let warm = NodeCacheConfig {
        memo: CachePolicy::Last,
        context: CachePolicy::Off,
    };
    eval.configure_cache(cached, warm).unwrap();
    eval.configure_cache(downstream, warm).unwrap();
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), downstream)
            .unwrap(),
        201
    );
    let hits_before = eval.stats().memo_hits;
    let mut batch = RequestBatch::new();
    let miss_id = batch.at(FrameIndex(0)).request(uncached);
    let out_id = batch.at(FrameIndex(0)).request(downstream);
    let mut plan = eval
        .plan(state.snapshot(), batch, PlannerConfig::default())
        .unwrap();
    // Zero quota: the leading uncached miss must not stop the cached hits.
    let tasks = eval.advance_with_budget(&mut plan, Vec::new(), 0).unwrap();
    assert!(tasks.is_empty());
    assert!(!plan.is_complete());
    assert_eq!(eval.stats().memo_hits, hits_before + 2);
    // One slot: only the true miss is dispatched.
    let tasks = eval.advance_with_budget(&mut plan, Vec::new(), 1).unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(plan.stats().compute_tasks, 1);
    let completions: Vec<_> = tasks.into_iter().map(|t| t.execute()).collect();
    let tasks = eval.advance_with_budget(&mut plan, completions, 0).unwrap();
    assert!(tasks.is_empty());
    assert!(plan.is_complete());
    let (results, _) = plan.finish().unwrap();
    assert_eq!(*results.get(miss_id).unwrap(), 100);
    assert_eq!(*results.get(out_id).unwrap(), 201);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn zero_dispatch_merges_second_frame_onto_running_compute_via_waiter() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let (phase, expensive) = state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let p = e.map(e.frame(), |_f: &FrameIndex| 7_u64)?;
            let expensive_calls = calls.clone();
            let n = e.map(p, move |x: &u64| {
                expensive_calls.fetch_add(1, Ordering::Relaxed);
                x * 10
            })?;
            Ok((p, n))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(phase, off()).unwrap();
    eval.configure_cache(expensive, off()).unwrap();
    let mut batch = RequestBatch::new();
    let id_a = batch.at(FrameIndex(120)).request(expensive);
    let id_b = batch.at(FrameIndex(121)).request(expensive);
    let mut plan = eval
        .plan(
            state.snapshot(),
            batch,
            PlannerConfig {
                max_in_flight: 1,
                ..Default::default()
            },
        )
        .unwrap();
    // The first phase task fills the only in-flight slot; the second phase is
    // deferred and dispatched on the next advance.
    let tasks = eval.advance_with_budget(&mut plan, Vec::new(), 1).unwrap();
    assert_eq!(tasks.len(), 1);
    let completions: Vec<_> = tasks.into_iter().map(|t| t.execute()).collect();
    let tasks = eval.advance_with_budget(&mut plan, completions, 1).unwrap();
    assert_eq!(tasks.len(), 1);
    let completions: Vec<_> = tasks.into_iter().map(|t| t.execute()).collect();
    // Both phase completions were delivered one at a time; the first child now
    // fills the quota and the second child must merge onto that running compute.
    let tasks = eval.advance_with_budget(&mut plan, completions, 1).unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(plan.stats().content_hits, 1);
    assert_eq!(plan.stats().peak_in_flight, 1);
    // Policies are read at commit time: switching the child to Last/Last makes
    // its single completion publish one output for both frames.
    eval.configure_cache(
        expensive,
        NodeCacheConfig {
            memo: CachePolicy::Last,
            context: CachePolicy::Last,
        },
    )
    .unwrap();
    let completions: Vec<_> = tasks.into_iter().map(|t| t.execute()).collect();
    let tasks = eval.advance_with_budget(&mut plan, completions, 0).unwrap();
    assert!(tasks.is_empty());
    assert!(plan.is_complete());
    let (results, _) = plan.finish().unwrap();
    assert_eq!(*results.get(id_a).unwrap(), 70);
    assert_eq!(*results.get(id_b).unwrap(), 70);
    assert!(Arc::ptr_eq(
        &results.get_shared(id_a).unwrap(),
        &results.get_shared(id_b).unwrap()
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn zero_dispatch_deferred_misses_count_toward_the_ready_budget() {
    let mut state = GraphState::new();
    let (miss, parent, child_a, child_b) = state
        .edit(|e| {
            let miss = e.map((), || 1_u64)?;
            let parent = e.map((), || 2_u64)?;
            let a = e.map(parent, |x: &u64| x + 1)?;
            let b = e.map(parent, |x: &u64| x + 2)?;
            Ok((miss, parent, a, b))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(miss, off()).unwrap();
    eval.configure_cache(
        parent,
        NodeCacheConfig {
            memo: CachePolicy::Last,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    eval.configure_cache(child_a, off()).unwrap();
    eval.configure_cache(child_b, off()).unwrap();
    // Warm only the parent constant.
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), parent)
            .unwrap(),
        2
    );
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(miss);
    let id_a = batch.at(FrameIndex(0)).request(child_a);
    let id_b = batch.at(FrameIndex(0)).request(child_b);
    let mut plan = eval
        .plan(
            state.snapshot(),
            batch,
            PlannerConfig {
                max_ready_tasks: 2,
                ..Default::default()
            },
        )
        .unwrap();
    // Ready starts as [miss, parent]; the parent hit unlocks both children, so
    // the queue holds 2 children plus the deferred miss: 3 > 2 must fail.
    assert!(matches!(
        eval.advance_with_budget(&mut plan, Vec::new(), 0),
        Err(EvalError::PlannerBudgetExceeded { limit: 2 })
    ));
    assert!(plan.failure().is_some());
    // A fresh request still evaluates after the terminal budget error.
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), miss)
            .unwrap(),
        1
    );
    let _ = (id_a, id_b);
}
