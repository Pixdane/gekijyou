use gekijyou_computation::{
    BatchResults, CachePolicy, Context, EvalError, EvalStats, Evaluator, FrameIndex, Graph, Node,
    RequestBatch, RequestId,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

fn frame(graph: &Graph, n: u64) -> Context {
    let mut context = Context::new(graph);
    context.set(graph, graph.frame(), FrameIndex(n)).unwrap();
    context
}
fn sample<T: 'static>(
    graph: &Graph,
    evaluator: &mut Evaluator,
    context: Context,
    target: Node<T>,
) -> (BatchResults, RequestId<T>) {
    let mut batch = RequestBatch::new(graph);
    let id = batch.add_context(context).unwrap();
    let request = batch.request(id, target).unwrap();
    (evaluator.evaluate_batch(graph, batch).unwrap(), request)
}
fn failing<T: 'static>(
    graph: &Graph,
    evaluator: &mut Evaluator,
    context: Context,
    target: Node<T>,
) -> Result<BatchResults, EvalError> {
    let mut batch = RequestBatch::new(graph);
    let id = batch.add_context(context).unwrap();
    batch.request(id, target).unwrap();
    evaluator.evaluate_batch(graph, batch)
}

#[test]
fn identical_independent_contexts_skip_entire_chain_and_direct_memo() {
    let mut graph = Graph::new();
    let a = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    let b = graph.map(a, |n: &u64| n + 2);
    let target = graph.map(b, |n: &u64| n * 3);
    graph.set_context_cache_capacity(target, 8).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let (old, first) = sample(&graph, &mut evaluator, frame(&graph, 10), target);
    assert_eq!(old.get(first), Ok(&36));
    assert_eq!(evaluator.stats().resolution_visits, 4);
    assert_eq!(evaluator.stats().context_misses, 1);
    evaluator.reset_stats();
    let mut batch = RequestBatch::new(&graph);
    for _ in 0..2 {
        let id = batch.add_context(frame(&graph, 10)).unwrap();
        batch.request(id, target).unwrap();
        batch.request(id, target).unwrap();
    }
    evaluator.evaluate_batch(&graph, batch).unwrap();
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            context_hits: 4,
            context_comparisons: 4,
            resolution_visits: 4,
            ..EvalStats::default()
        }
    );
    assert_eq!(evaluator.reset_stats().context_hits, 4);
    let (new, next) = sample(&graph, &mut evaluator, frame(&graph, 10), target);
    assert!(std::ptr::eq(
        old.get(first).unwrap(),
        new.get(next).unwrap()
    ));
}

#[test]
fn roots_use_source_semantics_and_ignore_unrelated_or_unassigned_inputs() {
    #[derive(PartialEq, Eq)]
    struct Viewport(u64);
    let mut graph = Graph::new();
    graph.set_default_eq::<Viewport>();
    let viewport = graph.input::<Viewport>();
    let unrelated = graph.input::<u64>();
    // Output u64 deliberately has only identity semantics.
    let target = graph.map((graph.frame(), viewport), |f: &FrameIndex, v: &Viewport| {
        f.0 + v.0
    });
    graph.set_context_cache_capacity(target, 4).unwrap();
    // Later changes cannot alter the input's frozen comparator.
    graph.set_default_equivalence::<Viewport, _>(|_, _| false);
    let mut evaluator = Evaluator::new(&graph);
    for unrelated_value in [Some(1), Some(2), None] {
        let mut context = frame(&graph, 10);
        context.set(&graph, viewport, Viewport(20)).unwrap();
        if let Some(n) = unrelated_value {
            context.set(&graph, unrelated, n).unwrap();
        }
        let (result, request) = sample(&graph, &mut evaluator, context, target);
        assert_eq!(result.get(request), Ok(&30));
    }
    assert_eq!(evaluator.stats().context_hits, 2);
    assert_eq!(evaluator.stats().context_comparisons, 4);
    assert_eq!(evaluator.stats().computation_calls, 1);
}

#[test]
fn changed_frame_misses_context_but_equal_phase_hits_direct_memo() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let phase = graph.map(graph.frame(), |f: &FrameIndex| f.0 / 30);
    let expensive = graph.map(phase, |n: &u64| n * 100);
    graph.set_context_cache_capacity(expensive, 4).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, frame(&graph, 30), expensive);
    evaluator.reset_stats();
    let (result, request) = sample(&graph, &mut evaluator, frame(&graph, 31), expensive);
    assert_eq!(result.get(request), Ok(&100));
    assert_eq!(evaluator.stats().context_hits, 0);
    assert_eq!(evaluator.stats().context_misses, 1);
    assert_eq!(evaluator.stats().memo_hits, 1);
    assert_eq!(evaluator.stats().computation_calls, 1);
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, frame(&graph, 31), expensive);
    assert_eq!(evaluator.stats().context_hits, 1);
    assert_eq!(evaluator.stats().resolution_visits, 1);
}

#[test]
fn diamond_duplicate_edges_and_more_than_four_roots() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let roots: Vec<_> = (0..6).map(|_| graph.input::<u64>()).collect();
    let a = graph.map(
        (roots[0], roots[1], roots[2], roots[0]),
        |a: &u64, b: &u64, c: &u64, d: &u64| a + b + c + d,
    );
    let b = graph.map(
        (roots[3], roots[4], roots[5], roots[1]),
        |a: &u64, b: &u64, c: &u64, d: &u64| a + b + c + d,
    );
    let target = graph.map((a, b, a), |a: &u64, b: &u64, c: &u64| a + b + c);
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for _ in 0..2 {
        let mut context = Context::new(&graph);
        for (i, &root) in roots.iter().enumerate() {
            context.set(&graph, root, i as u64).unwrap();
        }
        let (result, request) = sample(&graph, &mut evaluator, context, target);
        assert_eq!(result.get(request), Ok(&19));
    }
    assert_eq!(evaluator.stats().context_hits, 1);
    assert_eq!(evaluator.stats().context_comparisons, 6);
}

#[test]
fn zero_input_computation_has_empty_projection() {
    let mut graph = Graph::new();
    let target = graph.map((), || 7_u64);
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    assert_eq!(evaluator.stats().context_hits, 1);
    assert_eq!(evaluator.stats().context_comparisons, 0);
    assert_eq!(evaluator.stats().computation_calls, 1);
}

#[test]
fn replacing_upstream_invalidates_context_and_preserves_direct_history() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let a = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    let target = graph.map(a, |n: &u64| n + 1);
    graph.set_context_cache_capacity(target, 2).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let (old, request) = sample(&graph, &mut evaluator, frame(&graph, 3), target);
    graph
        .replace(a, graph.frame(), |f: &FrameIndex| f.0 + 1)
        .unwrap();
    evaluator.reset_stats();
    let (new, next) = sample(&graph, &mut evaluator, frame(&graph, 3), target);
    assert_eq!(new.get(next), Ok(&5));
    assert_eq!(old.get(request), Ok(&4));
    assert_eq!(evaluator.stats().context_hits, 0);
    assert_eq!(evaluator.stats().computation_calls, 2);
    // Same output from a replaced definition must still reuse target's direct Memo.
    graph
        .replace(a, graph.frame(), |f: &FrameIndex| f.0 + 1)
        .unwrap();
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, frame(&graph, 3), target);
    assert_eq!(evaluator.stats().context_misses, 1);
    assert_eq!(evaluator.stats().memo_hits, 1);
    assert_eq!(evaluator.stats().computation_calls, 1);
}

#[test]
fn changed_connections_rebuild_roots_and_require_new_input() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let seed = graph.input::<u64>();
    let upstream = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    let target = graph.map(upstream, |n: &u64| n * 10);
    graph.set_context_cache_capacity(target, 4).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, frame(&graph, 2), target);
    graph
        .replace(
            upstream,
            (graph.frame(), seed),
            |f: &FrameIndex, s: &u64| f.0 + s,
        )
        .unwrap();
    assert!(matches!(
        failing(&graph, &mut evaluator, frame(&graph, 2), target),
        Err(EvalError::MissingInput { .. })
    ));
    for seed_value in [1, 2, 1] {
        let mut context = frame(&graph, 2);
        context.set(&graph, seed, seed_value).unwrap();
        let (result, request) = sample(&graph, &mut evaluator, context, target);
        assert_eq!(result.get(request), Ok(&((2 + seed_value) * 10)));
    }
    assert_eq!(evaluator.stats().context_hits, 1);
    // Reconnect to a constant: old external roots must disappear as well.
    graph.replace(upstream, (), || 9_u64).unwrap();
    let (result, request) = sample(&graph, &mut evaluator, Context::new(&graph), target);
    assert_eq!(result.get(request), Ok(&90));
}

#[test]
fn additions_and_direct_policy_edits_preserve_context_but_unrelated_replace_invalidates() {
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, frame(&graph, 2), target);
    let unrelated = graph.constant(0_u64);
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, frame(&graph, 2), target);
    assert_eq!(evaluator.stats().context_hits, 1);
    graph.replace(unrelated, (), || 1_u64).unwrap();
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, frame(&graph, 2), target);
    assert_eq!(evaluator.stats().context_misses, 1);
    assert_eq!(evaluator.stats().computation_calls, 1);
}

#[test]
fn lru_promotes_and_evicts_historical_projections() {
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    graph.set_context_cache_capacity(target, 2).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for (f, hit) in [(1, false), (2, false), (1, true), (3, false), (2, false)] {
        evaluator.reset_stats();
        let (result, request) = sample(&graph, &mut evaluator, frame(&graph, f), target);
        assert_eq!(result.get(request), Ok(&f));
        assert_eq!(evaluator.stats().context_hits, u64::from(hit));
        assert_eq!(evaluator.stats().computation_calls, u64::from(!hit));
    }
}

#[test]
fn capacity_changes_are_lazy_local_and_independent_of_direct_memo() {
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    let other = graph.map(graph.frame(), |f: &FrameIndex| f.0 + 1);
    for node in [target, other] {
        graph.set_context_cache_capacity(node, 2).unwrap();
    }
    let mut evaluator = Evaluator::new(&graph);
    for node in [target, other] {
        sample(&graph, &mut evaluator, frame(&graph, 1), node);
    }
    graph.set_context_cache_capacity(target, 2).unwrap();
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, frame(&graph, 1), target);
    assert_eq!(evaluator.stats().context_hits, 1);
    for capacity in [1, 0, 2] {
        graph.set_context_cache_capacity(target, capacity).unwrap();
        evaluator.reset_stats();
        sample(&graph, &mut evaluator, frame(&graph, 1), target);
        assert_eq!(evaluator.stats().context_hits, 0);
        assert_eq!(evaluator.stats().context_misses, u64::from(capacity != 0));
        assert_eq!(evaluator.stats().memo_hits, 1);
        sample(&graph, &mut evaluator, frame(&graph, 1), other);
        assert_eq!(evaluator.stats().context_hits, 1);
    }
}

#[test]
fn missing_required_roots_fall_back_to_dfs_without_queries_or_entries() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let a = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    let missing = graph.input::<u64>();
    let target = graph.map((a, missing), |a: &u64, b: &u64| a + b);
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let mut valid = frame(&graph, 1);
    valid.set(&graph, missing, 2).unwrap();
    sample(&graph, &mut evaluator, valid, target);
    evaluator.reset_stats();
    assert!(matches!(
        failing(&graph, &mut evaluator, frame(&graph, 1), target),
        Err(EvalError::MissingInput { .. })
    ));
    assert_eq!(evaluator.stats().context_hits, 0);
    assert_eq!(evaluator.stats().context_misses, 0);
    assert_eq!(evaluator.stats().resolution_visits, 4);
    assert_eq!(evaluator.stats().memo_hits, 1);
    let mut valid = frame(&graph, 1);
    valid.set(&graph, missing, 2).unwrap();
    sample(&graph, &mut evaluator, valid, target);
    assert_eq!(evaluator.stats().context_hits, 1);
}

#[test]
fn missing_root_does_not_preempt_an_earlier_panicking_dependency() {
    let mut graph = Graph::new();
    let a = graph.map(graph.frame(), |_: &FrameIndex| -> u64 {
        panic!("earlier dependency")
    });
    let missing = graph.input::<u64>();
    let target = graph.map((a, missing), |a: &u64, b: &u64| a + b);
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let attempt = catch_unwind(AssertUnwindSafe(|| {
        failing(&graph, &mut evaluator, frame(&graph, 1), target)
    }));
    assert!(attempt.is_err());
    assert_eq!(evaluator.stats().context_misses, 0);
    assert_eq!(evaluator.stats().computation_calls, 1);
    graph
        .replace(a, graph.frame(), |f: &FrameIndex| f.0)
        .unwrap();
    let mut context = frame(&graph, 1);
    context.set(&graph, missing, 2).unwrap();
    let (result, request) = sample(&graph, &mut evaluator, context, target);
    assert_eq!(result.get(request), Ok(&3));
}

#[test]
fn closure_panic_publishes_no_entry_and_successful_upstream_caches_survive() {
    let mut graph = Graph::new();
    let a = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    graph.set_context_cache_capacity(a, 1).unwrap();
    let target = graph.map(a, |n: &u64| {
        assert_ne!(*n, 1, "injected panic");
        n + 10
    });
    graph.set_context_cache_capacity(target, 2).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for _ in 0..2 {
        assert!(
            catch_unwind(AssertUnwindSafe(|| sample(
                &graph,
                &mut evaluator,
                frame(&graph, 1),
                target
            )))
            .is_err()
        );
    }
    assert_eq!(evaluator.stats().context_misses, 3); // target twice, upstream once
    assert_eq!(evaluator.stats().context_hits, 1); // successful upstream retained
    let (result, request) = sample(&graph, &mut evaluator, frame(&graph, 2), target);
    assert_eq!(result.get(request), Ok(&12));
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, frame(&graph, 2), target);
    assert_eq!(evaluator.stats().context_hits, 1);
    assert_eq!(evaluator.stats().resolution_visits, 1);
}

#[test]
fn comparator_panic_preserves_entries_and_evaluator_recovers() {
    let panic = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&panic);
    let mut graph = Graph::new();
    graph.set_default_equivalence::<u64, _>(move |a, b| {
        assert!(!flag.load(Ordering::Relaxed));
        a == b
    });
    let input = graph.input::<u64>();
    let target = graph.map(input, |n: &u64| *n);
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    graph.set_context_cache_capacity(target, 2).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let context = |n| {
        let mut c = Context::new(&graph);
        c.set(&graph, input, n).unwrap();
        c
    };
    for n in [1, 2] {
        sample(&graph, &mut evaluator, context(n), target);
    }
    panic.store(true, Ordering::Relaxed);
    assert!(
        catch_unwind(AssertUnwindSafe(|| sample(
            &graph,
            &mut evaluator,
            context(1),
            target
        )))
        .is_err()
    );
    panic.store(false, Ordering::Relaxed);
    sample(&graph, &mut evaluator, context(3), target); // evicts 1, not 2
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, context(2), target);
    assert_eq!(evaluator.stats().context_hits, 1);
    sample(&graph, &mut evaluator, context(1), target);
    assert_eq!(evaluator.stats().context_misses, 1);
}

#[test]
fn unregistered_input_remains_identity_compared() {
    struct Value(u64);
    let mut graph = Graph::new();
    let input = graph.input::<Value>();
    let target = graph.map(input, |v: &Value| v.0);
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    graph.set_context_cache_capacity(target, 2).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for _ in 0..2 {
        let mut context = Context::new(&graph);
        context.set(&graph, input, Value(1)).unwrap();
        sample(&graph, &mut evaluator, context, target);
    }
    assert_eq!(evaluator.stats().context_hits, 0);
    assert_eq!(evaluator.stats().context_misses, 2);
}

#[test]
fn non_clone_results_survive_eviction_replacement_and_owner_destruction() {
    struct Value(u64);
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |f: &FrameIndex| Value(f.0));
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let (old, request) = sample(&graph, &mut evaluator, frame(&graph, 1), target);
    let (hit, hit_request) = sample(&graph, &mut evaluator, frame(&graph, 1), target);
    assert!(std::ptr::eq(
        old.get(request).unwrap(),
        hit.get(hit_request).unwrap()
    ));
    sample(&graph, &mut evaluator, frame(&graph, 2), target);
    graph.replace(target, (), || Value(99)).unwrap();
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    drop(evaluator);
    drop(graph);
    assert_eq!(old.get(request).unwrap().0, 1);
    assert_eq!(hit.get(hit_request).unwrap().0, 1);
}

#[test]
fn foreign_graph_contexts_batches_and_evaluators_are_rejected() {
    let mut graph = Graph::new();
    let target = graph.constant(7_u64);
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    let foreign = Graph::new();
    let mut batch = RequestBatch::new(&graph);
    assert_eq!(
        batch.add_context(Context::new(&foreign)),
        Err(EvalError::ForeignGraph)
    );
    assert!(matches!(
        evaluator.evaluate_batch(&foreign, RequestBatch::new(&foreign)),
        Err(EvalError::ForeignGraph)
    ));
    assert!(matches!(
        evaluator.evaluate_batch(&graph, RequestBatch::new(&foreign)),
        Err(EvalError::ForeignGraph)
    ));
}

#[test]
fn all_four_cache_configurations_produce_identical_outputs_and_distinct_stats() {
    for direct in [false, true] {
        for context_cache in [false, true] {
            let mut graph = Graph::new();
            graph.set_default_eq::<u64>();
            let a = graph.map(graph.frame(), |f: &FrameIndex| f.0 + 1);
            let target = graph.map(a, |n: &u64| n * 10);
            let policy = if direct {
                CachePolicy::Last
            } else {
                CachePolicy::Transient
            };
            for node in [a, target] {
                graph.set_cache_policy(node, policy).unwrap();
            }
            graph
                .set_context_cache_capacity(target, usize::from(context_cache))
                .unwrap();
            let mut evaluator = Evaluator::new(&graph);
            for _ in 0..2 {
                let (result, request) = sample(&graph, &mut evaluator, frame(&graph, 1), target);
                assert_eq!(result.get(request), Ok(&20));
            }
            let stats = evaluator.stats();
            assert_eq!(stats.context_hits, u64::from(context_cache));
            assert_eq!(stats.context_misses, u64::from(context_cache));
            assert_eq!(
                stats.memo_hits,
                if direct && !context_cache { 2 } else { 0 }
            );
            assert_eq!(
                stats.computation_calls,
                if direct || context_cache { 2 } else { 4 }
            );
            assert_eq!(stats.resolution_visits, if context_cache { 4 } else { 6 });
        }
    }
}

#[test]
fn disabling_context_retention_releases_old_output_on_next_visit() {
    struct Value(Arc<()>);
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |_: &FrameIndex| Value(Arc::new(())));
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let (result, request) = sample(&graph, &mut evaluator, frame(&graph, 1), target);
    let old = Arc::downgrade(&result.get(request).unwrap().0);
    drop(result);
    graph.set_context_cache_capacity(target, 0).unwrap();
    assert!(old.upgrade().is_some()); // Lazy synchronization has not run yet.
    let (next, request) = sample(&graph, &mut evaluator, frame(&graph, 1), target);
    assert!(old.upgrade().is_none());
    let new = Arc::downgrade(&next.get(request).unwrap().0);
    drop(next);
    assert!(new.upgrade().is_none()); // Disabled caches retain no entry.
}
