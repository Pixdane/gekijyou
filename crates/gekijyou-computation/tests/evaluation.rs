#![cfg(feature = "p35-baseline")]
use gekijyou_computation::p35::{
    CachePolicy, Context, EvalError, Evaluator, FrameIndex, Graph, GraphError, RequestBatch,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[test]
fn frame_and_two_contexts_produce_independent_results() {
    let mut graph = Graph::new();
    let target = graph.map(graph.frame(), |frame: &FrameIndex| frame.0 * 2);
    let mut batch = RequestBatch::new(&graph);
    let mut requests = Vec::new();
    for frame in [3, 11] {
        let mut context = Context::new(&graph);
        context
            .set(&graph, graph.frame(), FrameIndex(frame))
            .unwrap();
        let id = batch.add_context(context).unwrap();
        requests.push(batch.request(id, target).unwrap());
    }
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(results.get(requests[1]), Ok(&22));
    assert_eq!(results.get(requests[0]), Ok(&6));
}

#[test]
fn diamond_multiple_targets_and_duplicate_requests_share_once_per_context() {
    let mut graph = Graph::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let shared = graph.map(graph.frame(), move |frame: &FrameIndex| {
        counter.fetch_add(1, Ordering::Relaxed);
        frame.0 + 1
    });
    graph
        .set_cache_policy(shared, CachePolicy::Transient)
        .unwrap();
    let left = graph.map(shared, |n: &u64| n * 2);
    let right = graph.map(shared, |n: &u64| n * 3);
    let diamond = graph.map((left, right), |l: &u64, r: &u64| l + r);
    let mut batch = RequestBatch::new(&graph);
    let mut context = Context::new(&graph);
    context.set(&graph, graph.frame(), FrameIndex(4)).unwrap();
    let id = batch.add_context(context).unwrap();
    let first = batch.request(id, diamond).unwrap();
    let upstream = batch.request(id, shared).unwrap();
    let second = batch.request(id, diamond).unwrap();
    let branch = batch.request(id, left).unwrap();
    assert_ne!(first, second);
    let mut equal_context = Context::new(&graph);
    equal_context
        .set(&graph, graph.frame(), FrameIndex(4))
        .unwrap();
    let other = batch.add_context(equal_context).unwrap();
    let other_request = batch.request(other, diamond).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(results.get(first), Ok(&25));
    assert_eq!(results.get(upstream), Ok(&5));
    assert_eq!(results.get(branch), Ok(&10));
    assert_eq!(results.get(other_request), Ok(&25));
    assert!(std::ptr::eq(
        results.get(first).unwrap(),
        results.get(second).unwrap()
    ));
}

#[test]
fn zero_input_non_clone_constant_input_and_request_handles_work() {
    #[derive(Debug, PartialEq, Eq)]
    struct Owned(String);
    let mut graph = Graph::new();
    let constant = graph.constant(Owned("constant".into()));
    let input = graph.input::<Owned>();
    let zero = graph.map((), || Owned("zero".into()));
    let target = graph.map(
        (constant, input, zero),
        |a: &Owned, b: &Owned, c: &Owned| format!("{}/{}/{}", a.0, b.0, c.0),
    );
    let mut context = Context::new(&graph);
    context.set(&graph, input, Owned("input".into())).unwrap();
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(context).unwrap();
    let value = batch.request(id, constant).unwrap();
    // Copy and Clone must not impose bounds on Owned.
    let copied = value;
    #[allow(clippy::clone_on_copy)]
    let cloned = value.clone();
    assert_eq!(format!("{value:?}"), format!("{copied:?}"));
    let combined = batch.request(id, target).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(results.get(copied), Ok(&Owned("constant".into())));
    assert_eq!(results.get(cloned), results.get(value));
    assert_eq!(results.get(combined).unwrap(), "constant/input/zero");
}

#[test]
fn repeated_edges_keep_argument_order_and_multiplicity() {
    let mut graph = Graph::new();
    let a = graph.constant(2_u32);
    let b = graph.constant(7_u32);
    let target = graph.map((b, a, b, a), |b: &u32, a: &u32, b2: &u32, a2: &u32| {
        [*b, *a, *b2, *a2]
    });
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    let request = batch.request(id, target).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(results.get(request), Ok(&[7, 2, 7, 2]));
}

#[test]
fn batch_uses_current_definition_and_forward_dependencies() {
    let mut graph = Graph::new();
    let early = graph.constant(1_u64);
    let target = graph.map(early, |n: &u64| n * 10);
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    let request = batch.request(id, target).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let later = graph.constant(9_u64);
    graph.replace(early, later, |n: &u64| n + 1).unwrap();
    let results = evaluator.evaluate_batch(&graph, batch).unwrap();
    assert_eq!(results.get(request), Ok(&100));
}

#[test]
fn inputs_are_lazy_and_repeated_set_replaces_value() {
    let mut graph = Graph::new();
    let input = graph.input::<String>();
    let missing = graph.input::<u64>();
    let unreachable = graph.map(missing, |n: &u64| n + 1);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    graph.map((), move || {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    let mut context = Context::new(&graph);
    context.set(&graph, input, "old".into()).unwrap();
    context.set(&graph, input, "new".into()).unwrap();
    assert!(matches!(
        context.set(&graph, unreachable, 5),
        Err(EvalError::NotInput { .. })
    ));
    let mut batch = RequestBatch::new(&graph);
    batch.add_context(Context::new(&graph)).unwrap(); // Unused and unassigned.
    let id = batch.add_context(context).unwrap();
    let request = batch.request(id, input).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(results.get(request).unwrap(), "new");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn newly_required_missing_input_reports_the_context() {
    let mut graph = Graph::new();
    let target = graph.constant(1_u64);
    let input = graph.input::<u64>();
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    batch.request(id, target).unwrap();
    graph.replace(target, input, |n: &u64| *n).unwrap();
    let result = Evaluator::new(&graph).evaluate_batch(&graph, batch);
    assert!(matches!(result, Err(EvalError::MissingInput { context, .. }) if context == id));
}

#[test]
fn foreign_graphs_are_rejected_at_each_boundary() {
    let graph = Graph::new();
    let foreign = Graph::new();
    let mut context = Context::new(&graph);
    assert!(matches!(
        context.set(&foreign, foreign.frame(), FrameIndex(0)),
        Err(EvalError::ForeignGraph)
    ));
    assert!(matches!(
        context.set(&graph, foreign.frame(), FrameIndex(0)),
        Err(EvalError::Graph(GraphError::ForeignNode { .. }))
    ));
    let mut batch = RequestBatch::new(&graph);
    assert!(matches!(
        batch.add_context(Context::new(&foreign)),
        Err(EvalError::ForeignGraph)
    ));
    let id = batch.add_context(context).unwrap();
    assert!(matches!(
        batch.request(id, foreign.frame()),
        Err(EvalError::Graph(GraphError::ForeignNode { .. }))
    ));
    let mut evaluator = Evaluator::new(&graph);
    assert!(matches!(
        evaluator.evaluate_batch(&foreign, batch),
        Err(EvalError::ForeignGraph)
    ));
    assert!(matches!(
        evaluator.evaluate_batch(&graph, RequestBatch::new(&foreign)),
        Err(EvalError::ForeignGraph)
    ));
}

#[test]
fn context_and_request_ids_cannot_cross_batches() {
    let mut graph = Graph::new();
    let target = graph.constant(42_u64);
    let mut a = RequestBatch::new(&graph);
    let mut b = RequestBatch::new(&graph);
    let ca = a.add_context(Context::new(&graph)).unwrap();
    let cb = b.add_context(Context::new(&graph)).unwrap();
    assert!(matches!(
        b.request(ca, target),
        Err(EvalError::ForeignContext)
    ));
    let ra = a.request(ca, target).unwrap();
    let rb = b.request(cb, target).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let a_results = evaluator.evaluate_batch(&graph, a).unwrap();
    let b_results = evaluator.evaluate_batch(&graph, b).unwrap();
    assert_eq!(a_results.get(rb), Err(EvalError::ForeignRequest));
    assert_eq!(b_results.get(ra), Err(EvalError::ForeignRequest));
    assert_eq!(a_results.get(ra), Ok(&42));
    assert_eq!(b_results.get(rb), Ok(&42));
}

#[test]
fn results_survive_replacement_and_graph_and_evaluator_destruction() {
    let (results, request) = {
        let mut graph = Graph::new();
        let target = graph.constant(String::from("old"));
        let mut batch = RequestBatch::new(&graph);
        let id = batch.add_context(Context::new(&graph)).unwrap();
        let request = batch.request(id, target).unwrap();
        let mut evaluator = Evaluator::new(&graph);
        let results = evaluator.evaluate_batch(&graph, batch).unwrap();
        graph.replace(target, (), || "new".to_string()).unwrap();
        assert_eq!(results.get(request).unwrap(), "old");
        let mut batch = RequestBatch::new(&graph);
        let id = batch.add_context(Context::new(&graph)).unwrap();
        let new_request = batch.request(id, target).unwrap();
        let new_results = evaluator.evaluate_batch(&graph, batch).unwrap();
        assert_eq!(new_results.get(new_request).unwrap(), "new");
        (results, request)
    };
    assert_eq!(results.get(request).unwrap(), "old");
}

#[test]
fn failed_replace_preserves_evaluation() {
    let mut graph = Graph::new();
    let first = graph.constant(2_u64);
    let second = graph.map(first, |n: &u64| n + 3);
    assert_eq!(
        graph.replace(first, second, |n: &u64| *n),
        Err(GraphError::Cycle)
    );
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    let request = batch.request(id, second).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(results.get(request), Ok(&5));
}

#[test]
fn no_requests_return_success_without_evaluation() {
    let graph = Graph::new();
    let mut evaluator = Evaluator::new(&graph);
    assert!(
        evaluator
            .evaluate_batch(&graph, RequestBatch::new(&graph))
            .is_ok()
    );
    let mut batch = RequestBatch::new(&graph);
    batch.add_context(Context::new(&graph)).unwrap();
    assert!(evaluator.evaluate_batch(&graph, batch).is_ok());
}

#[test]
fn transient_computations_are_not_memoized_across_batches() {
    let mut graph = Graph::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let target = graph.map((), move || {
        counter.fetch_add(1, Ordering::Relaxed);
        42_u64
    });
    graph
        .set_cache_policy(target, CachePolicy::Transient)
        .unwrap();
    let mut evaluator = Evaluator::new(&graph);
    for _ in 0..2 {
        let mut batch = RequestBatch::new(&graph);
        let id = batch.add_context(Context::new(&graph)).unwrap();
        let request = batch.request(id, target).unwrap();
        let results = evaluator.evaluate_batch(&graph, batch).unwrap();
        assert_eq!(results.get(request), Ok(&42));
    }
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn panic_unwinds_batch_and_evaluator_can_be_reused() {
    let mut graph = Graph::new();
    let dropped = Arc::new(AtomicUsize::new(0));
    struct Tracked(Arc<AtomicUsize>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let tracker = dropped.clone();
    let temporary = graph.map((), move || Tracked(tracker.clone()));
    graph
        .set_cache_policy(temporary, CachePolicy::Transient)
        .unwrap();
    let panicking = graph.map(temporary, |_: &Tracked| -> u64 {
        panic!("computation panic")
    });
    let good = graph.constant(9_u64);
    let mut evaluator = Evaluator::new(&graph);
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    batch.request(id, panicking).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| evaluator.evaluate_batch(&graph, batch))).is_err());
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    let request = batch.request(id, good).unwrap();
    let results = evaluator.evaluate_batch(&graph, batch).unwrap();
    assert_eq!(results.get(request), Ok(&9));
}

#[test]
fn failed_batch_releases_already_computed_values_and_can_be_retried() {
    let mut graph = Graph::new();
    let drops = Arc::new(AtomicUsize::new(0));
    struct Tracked(Arc<AtomicUsize>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let tracker = drops.clone();
    let first = graph.map((), move || Tracked(tracker.clone()));
    graph
        .set_cache_policy(first, CachePolicy::Transient)
        .unwrap();
    let missing = graph.input::<u64>();
    let mut evaluator = Evaluator::new(&graph);
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(Context::new(&graph)).unwrap();
    batch.request(id, first).unwrap();
    batch.request(id, missing).unwrap();
    assert!(matches!(
        evaluator.evaluate_batch(&graph, batch),
        Err(EvalError::MissingInput { .. })
    ));
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let mut context = Context::new(&graph);
    context.set(&graph, missing, 7).unwrap();
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(context).unwrap();
    let request = batch.request(id, missing).unwrap();
    let results = evaluator.evaluate_batch(&graph, batch).unwrap();
    assert_eq!(results.get(request), Ok(&7));
}
