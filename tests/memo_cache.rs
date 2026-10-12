use gekijyou::computation::{
    BatchResults, CachePolicy, Context, EvalError, EvalStats, Evaluator, FrameIndex, Graph,
    GraphError, Node, RequestBatch, RequestId,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

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

fn frame(graph: &Graph, n: u64) -> Context {
    let mut context = Context::new(graph);
    context.set(graph, graph.frame(), FrameIndex(n)).unwrap();
    context
}

#[test]
fn different_frames_reuse_equal_direct_inputs_and_report_exact_stats() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let phase = graph.map(graph.frame(), |f: &FrameIndex| f.0 / 30);
    graph
        .set_cache_policy(phase, CachePolicy::Transient)
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let expensive = graph.map(phase, move |p: &u64| {
        counter.fetch_add(1, Ordering::Relaxed);
        p * 100
    });
    let mut evaluator = Evaluator::new(&graph);
    let (old, request) = sample(&graph, &mut evaluator, frame(&graph, 30), expensive);
    let (new, next) = sample(&graph, &mut evaluator, frame(&graph, 31), expensive);
    assert_eq!(old.get(request), Ok(&100));
    assert!(std::ptr::eq(
        old.get(request).unwrap(),
        new.get(next).unwrap()
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            memo_hits: 1,
            memo_misses: 1,
            memo_comparisons: 1,
            computation_calls: 3,
            resolution_visits: 6,
            ..EvalStats::default()
        }
    );
    let previous = evaluator.reset_stats();
    assert_eq!(previous.memo_hits, 1);
    assert_eq!(evaluator.stats(), EvalStats::default());
    sample(&graph, &mut evaluator, frame(&graph, 32), expensive);
    assert_eq!(evaluator.stats().memo_hits, 1); // Resetting stats retains cache.
}

#[test]
fn last_and_lru_differ_on_historical_inputs() {
    for (policy, expected_computations, expected_hits) in [
        (CachePolicy::Last, 7, 0),
        (CachePolicy::Lru { capacity: 2 }, 5, 2),
    ] {
        let mut graph = Graph::new();
        graph.set_default_eq::<u64>();
        let phase = graph.map(graph.frame(), |f: &FrameIndex| f.0 / 30);
        graph
            .set_cache_policy(phase, CachePolicy::Transient)
            .unwrap();
        let target = graph.map(phase, |p: &u64| p * 7);
        graph.set_cache_policy(target, policy).unwrap();
        let mut evaluator = Evaluator::new(&graph);
        // Hit 1 promotes it, so inserting 3 evicts 2; then 1 still hits.
        for f in [30, 60, 31, 90, 32, 61, 91] {
            let (results, request) = sample(&graph, &mut evaluator, frame(&graph, f), target);
            assert_eq!(results.get(request), Ok(&(f / 30 * 7)));
        }
        let stats = evaluator.stats();
        assert_eq!(stats.computation_calls, 7 + expected_computations);
        assert_eq!(stats.memo_hits, expected_hits);
        assert_eq!(stats.memo_misses, expected_computations);
    }
}

#[test]
fn transient_and_zero_capacity_share_only_within_a_context() {
    for policy in [CachePolicy::Transient, CachePolicy::Lru { capacity: 0 }] {
        let mut graph = Graph::new();
        let target = graph.map((), || 7_u64);
        graph.set_cache_policy(target, policy).unwrap();
        let mut evaluator = Evaluator::new(&graph);
        for _ in 0..2 {
            let mut batch = RequestBatch::new(&graph);
            for _ in 0..2 {
                let id = batch.add_context(Context::new(&graph)).unwrap();
                batch.request(id, target).unwrap();
                batch.request(id, target).unwrap();
            }
            evaluator.evaluate_batch(&graph, batch).unwrap();
        }
        assert_eq!(
            evaluator.stats(),
            EvalStats {
                computation_calls: 4,
                resolution_visits: 8,
                ..EvalStats::default()
            }
        );
    }
}

#[test]
fn identity_semantics_reuse_shared_objects_but_not_equal_new_allocations() {
    struct Value(u64); // Neither Eq nor Clone.
    let mut graph = Graph::new();
    let shared = graph.constant(Value(7));
    graph
        .set_cache_policy(shared, CachePolicy::Transient)
        .unwrap();
    let fresh = graph.map((), || Value(7));
    graph
        .set_cache_policy(fresh, CachePolicy::Transient)
        .unwrap();
    let shared_target = graph.map(shared, |v: &Value| v.0);
    let fresh_target = graph.map(fresh, |v: &Value| v.0);
    let mut evaluator = Evaluator::new(&graph);
    for _ in 0..2 {
        sample(&graph, &mut evaluator, Context::new(&graph), shared_target);
        sample(&graph, &mut evaluator, Context::new(&graph), fresh_target);
    }
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            memo_hits: 1,
            memo_misses: 3,
            memo_comparisons: 2,
            computation_calls: 7,
            resolution_visits: 8,
            ..EvalStats::default()
        }
    );
}

#[test]
fn float_bits_and_creation_snapshot_control_source_equality() {
    let mut graph = Graph::new();
    let early = graph.input::<f64>();
    graph.set_default_equivalence::<f64, _>(|a, b| a.to_bits() == b.to_bits());
    let late = graph.input::<f64>();
    // Output nodes have no u64 content comparator: input semantics still drive lookup.
    let early_target = graph.map(early, |v: &f64| v.to_bits());
    let late_target = graph.map(late, |v: &f64| v.to_bits());
    let mut evaluator = Evaluator::new(&graph);
    let nan = f64::from_bits(0x7ff8_0000_0000_0042);
    for (input, target, expected_hits) in [(early, early_target, 0), (late, late_target, 2)] {
        evaluator.reset_stats();
        for value in [
            0.0_f64,
            0.0,
            -0.0,
            nan,
            nan,
            f64::from_bits(nan.to_bits() + 1),
        ] {
            let mut context = Context::new(&graph);
            context.set(&graph, input, value).unwrap();
            let (results, request) = sample(&graph, &mut evaluator, context, target);
            assert_eq!(results.get(request), Ok(&value.to_bits()));
        }
        assert_eq!(evaluator.stats().memo_hits, expected_hits);
        assert_eq!(evaluator.stats().computation_calls, 6 - expected_hits);
    }
}

#[test]
fn zero_input_computations_reuse_without_value_comparisons() {
    struct Value(u64);
    for (policy, calls) in [
        (CachePolicy::Last, 1),
        (CachePolicy::Lru { capacity: 2 }, 1),
        (CachePolicy::Transient, 3),
        (CachePolicy::Lru { capacity: 0 }, 3),
    ] {
        let mut graph = Graph::new();
        let node = graph.map((), || Value(9));
        graph.set_cache_policy(node, policy).unwrap();
        let mut evaluator = Evaluator::new(&graph);
        for _ in 0..3 {
            let (results, request) = sample(&graph, &mut evaluator, Context::new(&graph), node);
            assert_eq!(results.get(request).unwrap().0, 9);
        }
        assert_eq!(evaluator.stats().computation_calls, calls);
        assert_eq!(evaluator.stats().memo_comparisons, 0);
        assert_eq!(evaluator.stats().memo_hits, if calls == 1 { 2 } else { 0 });
    }
}

#[test]
fn replacing_upstream_preserves_downstream_history_and_old_results() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let a = graph.map(graph.frame(), |f: &FrameIndex| f.0 / 30);
    let b = graph.map(a, |v: &u64| v * 10);
    let c = graph.map(b, |v: &u64| v + 1);
    for node in [a, b, c] {
        graph
            .set_cache_policy(node, CachePolicy::Lru { capacity: 2 })
            .unwrap();
    }
    let mut evaluator = Evaluator::new(&graph);
    let (old, old_request) = sample(&graph, &mut evaluator, frame(&graph, 30), c);
    sample(&graph, &mut evaluator, frame(&graph, 60), c);
    evaluator.reset_stats();
    // Same Frame used in an old A entry now produces a different historical value.
    graph
        .replace(a, graph.frame(), |f: &FrameIndex| f.0 / 30 + 1)
        .unwrap();
    let (new, request) = sample(&graph, &mut evaluator, frame(&graph, 30), c);
    assert_eq!(new.get(request), Ok(&21));
    assert_eq!(old.get(old_request), Ok(&11));
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            memo_hits: 2,
            memo_misses: 1,
            memo_comparisons: 2,
            computation_calls: 1,
            resolution_visits: 4,
            ..EvalStats::default()
        }
    );
    // Replacing B itself must invalidate even though its input is identical.
    graph.replace(b, a, |v: &u64| v * 100).unwrap();
    evaluator.reset_stats();
    let (new, request) = sample(&graph, &mut evaluator, frame(&graph, 30), c);
    assert_eq!(new.get(request), Ok(&201));
    assert_eq!(evaluator.stats().computation_calls, 2);
    assert_eq!(old.get(old_request), Ok(&11));
}

#[test]
fn replacing_input_edges_invalidates_definition_and_uses_forward_dfs() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let first = graph.constant(3_u64);
    let target = graph.map(first, |v: &u64| v + 1);
    let later = graph.constant(3_u64); // Same value must not hide the definition edit.
    let forward = graph.map(later, |v: &u64| *v);
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    graph.replace(target, forward, |v: &u64| v + 9).unwrap();
    evaluator.reset_stats();
    let (results, request) = sample(&graph, &mut evaluator, Context::new(&graph), target);
    assert_eq!(results.get(request), Ok(&12));
    assert_eq!(evaluator.stats().computation_calls, 3);
    assert_eq!(evaluator.stats().memo_hits, 0);
}

#[test]
fn policy_transitions_apply_lazily_locally_and_preserve_definition() {
    let mut graph = Graph::new();
    let target = graph.map((), || 1_u64);
    let unaffected = graph.map((), || 2_u64);
    let mut evaluator = Evaluator::new(&graph);
    sample(&graph, &mut evaluator, Context::new(&graph), unaffected);
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    for policy in [
        CachePolicy::Lru { capacity: 16 },
        CachePolicy::Lru { capacity: 4 },
        CachePolicy::Last,
        CachePolicy::Lru { capacity: 1 },
        CachePolicy::Last,
        CachePolicy::Lru { capacity: 2 },
        CachePolicy::Transient,
        CachePolicy::Last,
        CachePolicy::Lru { capacity: 0 },
        CachePolicy::Last,
    ] {
        graph.set_cache_policy(target, policy).unwrap();
        evaluator.reset_stats();
        sample(&graph, &mut evaluator, Context::new(&graph), unaffected);
        sample(&graph, &mut evaluator, Context::new(&graph), target);
        assert_eq!(evaluator.stats().computation_calls, 1);
        assert_eq!(evaluator.stats().memo_hits, 1);
    }
    graph.set_cache_policy(target, CachePolicy::Last).unwrap();
    evaluator.reset_stats();
    sample(&graph, &mut evaluator, Context::new(&graph), target);
    assert_eq!(evaluator.stats().memo_hits, 1);
}

#[test]
fn shrinking_memo_cannot_resurrect_evicted_history() {
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    graph
        .set_cache_policy(target, CachePolicy::Lru { capacity: 16 })
        .unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for f in 0..16 {
        sample(&graph, &mut evaluator, frame(&graph, f), target);
    }
    graph
        .set_cache_policy(target, CachePolicy::Lru { capacity: 4 })
        .unwrap();
    evaluator.reset_stats();
    for f in [0, 1, 2, 3, 0, 4, 1] {
        sample(&graph, &mut evaluator, frame(&graph, f), target);
    }
    assert_eq!(evaluator.stats().memo_hits, 1);
    assert_eq!(evaluator.stats().computation_calls, 6);
}

#[test]
fn non_clone_inputs_outputs_and_repeated_tuple_edges_remain_shared() {
    #[derive(PartialEq, Eq)]
    struct Value(u64); // Eq without Clone.
    struct Output([u64; 4]);
    let mut graph = Graph::new();
    graph.set_default_eq::<Value>();
    let a = graph.input::<Value>();
    let b = graph.input::<Value>();
    let target = graph.map(
        (a, b, a, b),
        |a: &Value, b: &Value, again_a: &Value, again_b: &Value| {
            assert!(std::ptr::eq(a, again_a));
            assert!(std::ptr::eq(b, again_b));
            Output([a.0, b.0, again_a.0, again_b.0])
        },
    );
    let mut evaluator = Evaluator::new(&graph);
    let mut retained = Vec::new();
    for _ in 0..2 {
        let mut context = Context::new(&graph);
        context.set(&graph, a, Value(3)).unwrap();
        context.set(&graph, b, Value(7)).unwrap();
        retained.push(sample(&graph, &mut evaluator, context, target));
    }
    let (old, first) = &retained[0];
    let (new, second) = &retained[1];
    assert_eq!(old.get(*first).unwrap().0, [3, 7, 3, 7]);
    assert!(std::ptr::eq(
        old.get(*first).unwrap(),
        new.get(*second).unwrap()
    ));
    assert_eq!(evaluator.stats().memo_comparisons, 4);
    assert_eq!(evaluator.stats().computation_calls, 1);
}

#[test]
fn successful_cache_survives_errors_and_panics_without_partial_entries() {
    let mut graph = Graph::new();
    let good = graph.map((), || 7_u64);
    let missing = graph.input::<u64>();
    // Failure injection only; the successful path is a constant pure function.
    let panic_now = Arc::new(AtomicBool::new(true));
    let flag = panic_now.clone();
    let failing = graph.map(good, move |v: &u64| {
        assert!(!flag.load(Ordering::Relaxed), "injected panic");
        v + 1
    });
    let mut evaluator = Evaluator::new(&graph);
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    batch.request(id, good).unwrap();
    batch.request(id, missing).unwrap();
    assert!(matches!(
        evaluator.evaluate_batch(&graph, batch),
        Err(EvalError::MissingInput { .. })
    ));
    assert_eq!(evaluator.stats().computation_calls, 1);

    for _ in 0..2 {
        let mut batch = RequestBatch::new(&graph);
        let id = batch.add_context(Context::new(&graph)).unwrap();
        batch.request(id, failing).unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| evaluator.evaluate_batch(&graph, batch))).is_err()
        );
    }
    assert_eq!(evaluator.stats().computation_calls, 3); // Both panics invoked the closure.
    assert_eq!(evaluator.stats().memo_hits, 2); // Only good can hit.
    assert_eq!(evaluator.stats().memo_misses, 3);
    panic_now.store(false, Ordering::Relaxed);
    let (results, request) = sample(&graph, &mut evaluator, Context::new(&graph), failing);
    assert_eq!(results.get(request), Ok(&8));
    sample(&graph, &mut evaluator, Context::new(&graph), failing);
    assert_eq!(evaluator.stats().computation_calls, 4);
    assert_eq!(evaluator.stats().memo_hits, 5);
    let mut context = Context::new(&graph);
    context.set(&graph, missing, 11).unwrap();
    let (results, request) = sample(&graph, &mut evaluator, context, missing);
    assert_eq!(results.get(request), Ok(&11));
}

#[test]
fn cached_consumer_still_requires_its_dependencies_and_current_context_inputs() {
    let mut graph = Graph::new();
    let input = graph.input::<u64>();
    let target = graph.map(input, |v: &u64| v + 1);
    let mut evaluator = Evaluator::new(&graph);
    let mut context = Context::new(&graph);
    context.set(&graph, input, 7).unwrap();
    sample(&graph, &mut evaluator, context, target);
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    batch.request(id, target).unwrap();
    evaluator.reset_stats();
    assert!(matches!(
        evaluator.evaluate_batch(&graph, batch),
        Err(EvalError::MissingInput { .. })
    ));
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            resolution_visits: 2,
            ..EvalStats::default()
        }
    );
}

#[test]
fn memo_reuses_across_contexts_without_counting_duplicate_requests_as_hits() {
    let mut graph = Graph::new();
    graph.set_default_eq::<u64>();
    let phase = graph.map(graph.frame(), |f: &FrameIndex| f.0 / 30);
    graph
        .set_cache_policy(phase, CachePolicy::Transient)
        .unwrap();
    let target = graph.map(phase, |p: &u64| p * 10);
    let mut batch = RequestBatch::new(&graph);
    let mut requests = Vec::new();
    for f in [30, 31] {
        let id = batch.add_context(frame(&graph, f)).unwrap();
        requests.push(batch.request(id, target).unwrap());
        requests.push(batch.request(id, target).unwrap());
    }
    let mut evaluator = Evaluator::new(&graph);
    let results = evaluator.evaluate_batch(&graph, batch).unwrap();
    for request in requests {
        assert_eq!(results.get(request), Ok(&10));
    }
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            memo_hits: 1,
            memo_misses: 1,
            memo_comparisons: 1,
            computation_calls: 3,
            resolution_visits: 8,
            ..EvalStats::default()
        }
    );
}

#[test]
fn input_policy_is_rejected_and_current_context_value_is_preserved() {
    let mut graph = Graph::new();
    let input = graph.input::<u64>();
    let mut evaluator = Evaluator::new(&graph);
    for policy in [
        CachePolicy::Last,
        CachePolicy::Lru { capacity: 4 },
        CachePolicy::Transient,
    ] {
        assert!(matches!(
            graph.set_cache_policy(input, policy),
            Err(GraphError::InputNodeCachePolicy { .. })
        ));
        for n in [7, 11] {
            let mut context = Context::new(&graph);
            context.set(&graph, input, n).unwrap();
            let (results, request) = sample(&graph, &mut evaluator, context, input);
            assert_eq!(results.get(request), Ok(&n));
        }
    }
    assert_eq!(
        evaluator.stats(),
        EvalStats {
            resolution_visits: 6,
            ..EvalStats::default()
        }
    );
}

#[test]
fn external_shared_resource_identity_requires_explicit_equivalence() {
    struct Resource(u64);
    for explicit_identity in [false, true] {
        let resource = Arc::new(Resource(7));
        let mut graph = Graph::new();
        if explicit_identity {
            graph.set_default_equivalence::<Arc<Resource>, _>(Arc::ptr_eq);
        }
        let input = graph.input::<Arc<Resource>>();
        let target = graph.map(input, |r: &Arc<Resource>| r.0);
        let mut evaluator = Evaluator::new(&graph);
        for resource in [resource.clone(), resource, Arc::new(Resource(7))] {
            let mut context = Context::new(&graph);
            context.set(&graph, input, resource).unwrap();
            let (result, request) = sample(&graph, &mut evaluator, context, target);
            assert_eq!(result.get(request), Ok(&7));
        }
        let expected_hits = u64::from(explicit_identity);
        assert_eq!(evaluator.stats().memo_hits, expected_hits);
        assert_eq!(evaluator.stats().computation_calls, 3 - expected_hits);
    }
}

#[test]
fn comparator_panic_preserves_history_and_evaluator_can_be_reused() {
    let panic_now = Arc::new(AtomicBool::new(false));
    let flag = panic_now.clone();
    let mut graph = Graph::new();
    graph.set_default_equivalence::<u64, _>(move |a, b| {
        assert!(!flag.load(Ordering::Relaxed), "injected comparator panic");
        a == b
    });
    let input = graph.input::<u64>();
    let target = graph.map(input, |v: &u64| v * 10);
    graph
        .set_cache_policy(target, CachePolicy::Lru { capacity: 2 })
        .unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for n in [1, 2] {
        let mut context = Context::new(&graph);
        context.set(&graph, input, n).unwrap();
        sample(&graph, &mut evaluator, context, target);
    }
    evaluator.reset_stats();
    panic_now.store(true, Ordering::Relaxed);
    let mut context = Context::new(&graph);
    context.set(&graph, input, 1).unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sample(&graph, &mut evaluator, context, target);
        }))
        .is_err()
    );
    assert_eq!(evaluator.stats().computation_calls, 0);
    assert_eq!(
        evaluator.stats().memo_hits + evaluator.stats().memo_misses,
        0
    );
    panic_now.store(false, Ordering::Relaxed);
    for n in [1, 2] {
        let mut context = Context::new(&graph);
        context.set(&graph, input, n).unwrap();
        let (result, request) = sample(&graph, &mut evaluator, context, target);
        assert_eq!(result.get(request), Ok(&(n * 10)));
    }
    assert_eq!(evaluator.stats().memo_hits, 2);
    assert_eq!(evaluator.stats().computation_calls, 0);
}

#[test]
fn eviction_releases_cached_inputs_while_retained_results_own_outputs() {
    let mut graph = Graph::new();
    let input = graph.input::<Arc<()>>();
    let target = graph.map(input, |_: &Arc<()>| Arc::new(()));
    let mut evaluator = Evaluator::new(&graph);
    let resource = Arc::new(());
    let input_weak = Arc::downgrade(&resource);
    let mut context = Context::new(&graph);
    context.set(&graph, input, resource).unwrap();
    let (old, request) = sample(&graph, &mut evaluator, context, target);
    let output_weak = Arc::downgrade(old.get(request).unwrap());
    assert!(input_weak.upgrade().is_some());
    let mut context = Context::new(&graph);
    context.set(&graph, input, Arc::new(())).unwrap();
    sample(&graph, &mut evaluator, context, target);
    assert!(input_weak.upgrade().is_none());
    assert!(output_weak.upgrade().is_some());
    drop(old);
    assert!(output_weak.upgrade().is_none());
}

#[test]
fn lazy_invalidation_and_evaluator_drop_release_cached_values() {
    for replace_definition in [false, true] {
        let mut graph = Graph::new();
        let input = graph.input::<Arc<()>>();
        let target = graph.map(input, |_: &Arc<()>| Arc::new(()));
        let mut evaluator = Evaluator::new(&graph);
        let resource = Arc::new(());
        let input_weak = Arc::downgrade(&resource);
        let mut context = Context::new(&graph);
        context.set(&graph, input, resource).unwrap();
        let (old, request) = sample(&graph, &mut evaluator, context, target);
        let output_weak = Arc::downgrade(old.get(request).unwrap());
        drop(old);
        if replace_definition {
            graph
                .replace(target, input, |_: &Arc<()>| Arc::new(()))
                .unwrap();
        } else {
            graph
                .set_cache_policy(target, CachePolicy::Transient)
                .unwrap();
        }
        // Graph edits do not mutate an evaluator; old entries survive until visited.
        assert!(input_weak.upgrade().is_some());
        assert!(output_weak.upgrade().is_some());
        let mut context = Context::new(&graph);
        context.set(&graph, input, Arc::new(())).unwrap();
        let (new, request) = sample(&graph, &mut evaluator, context, target);
        assert!(input_weak.upgrade().is_none());
        assert!(output_weak.upgrade().is_none());
        let new_weak = Arc::downgrade(new.get(request).unwrap());
        drop(new);
        if replace_definition {
            assert!(new_weak.upgrade().is_some());
        }
        drop(evaluator);
        assert!(new_weak.upgrade().is_none());
    }
}
