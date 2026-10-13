use super::*;
use crate::p35::{ComputationDefinition, ComputationInputs, FrameIndex, Node};
use slotmap::Key;
use std::{
    any::TypeId,
    sync::atomic::{AtomicUsize, Ordering},
};

#[test]
fn dependencies_beyond_inline_capacity_preserve_order_and_repeated_values() {
    use super::super::node::Dependency;
    struct Value(u64);
    struct FiveInputs([Node<Value>; 5]);
    impl ComputationInputs<(), [u64; 5]> for FiveInputs {
        fn build_computation(&self, _: ()) -> ComputationDefinition<[u64; 5]> {
            let inputs = self
                .0
                .iter()
                .map(|node| Dependency {
                    graph_id: node.graph_id,
                    source: node.id,
                    type_id: TypeId::of::<Value>(),
                })
                .collect();
            ComputationDefinition::new(
                inputs,
                Box::new(|args| {
                    assert_eq!(args.len(), 5);
                    assert!(std::ptr::eq(args[0], args[2]));
                    assert!(std::ptr::eq(args[1], args[4]));
                    let output: [u64; 5] =
                        std::array::from_fn(|i| args[i].downcast_ref::<Value>().unwrap().0);
                    Arc::new(output)
                }),
            )
        }
    }
    let mut graph = Graph::new();
    let a = graph.constant(Value(2));
    let b = graph.constant(Value(7));
    let c = graph.constant(Value(3));
    let target = graph.map(FiveInputs([b, a, b, c, a]), ());
    let mut evaluator = Evaluator::new(&graph);
    for _ in 0..2 {
        let mut batch = RequestBatch::new(&graph);
        let context = batch.add_context(Context::new(&graph)).unwrap();
        let request = batch.request(context, target).unwrap();
        let results = evaluator.evaluate_batch(&graph, batch).unwrap();
        assert_eq!(results.get(request), Ok(&[7, 2, 7, 3, 2]));
    }
    assert_eq!(evaluator.stats().computation_calls, 4);
    assert_eq!(evaluator.stats().memo_hits, 4);
    assert_eq!(evaluator.stats().memo_comparisons, 5);
}

#[test]
fn erased_output_type_is_checked_before_publishing() {
    // Only module-internal code can construct this inconsistent definition.
    // Public adapters cannot forge the erased output type.
    struct WrongOutput;
    impl ComputationInputs<(), u64> for WrongOutput {
        fn build_computation(&self, _: ()) -> ComputationDefinition<u64> {
            ComputationDefinition::new(Box::new([]), Box::new(|_| Arc::new("wrong")))
        }
    }
    for capacity in [0, 1] {
        let mut graph = Graph::new();
        let node = graph.map(WrongOutput, ());
        graph.set_context_cache_capacity(node, capacity).unwrap();
        let mut evaluator = Evaluator::new(&graph);
        for _ in 0..2 {
            let mut batch = RequestBatch::new(&graph);
            let context = batch.add_context(Context::new(&graph)).unwrap();
            batch.request(context, node).unwrap();
            assert!(matches!(evaluator.evaluate_batch(&graph, batch),
            Err(EvalError::Graph(GraphError::TypeMismatch { node: id })) if id == node.id));
        }
        assert_eq!(evaluator.stats().computation_calls, 2);
        assert_eq!(evaluator.stats().memo_misses, 2);
        assert_eq!(evaluator.stats().memo_hits, 0);
        assert_eq!(
            evaluator.stats().context_misses,
            if capacity == 0 { 0 } else { 2 }
        );
        assert_eq!(evaluator.stats().context_hits, 0);
    }
}

#[test]
fn all_targets_are_validated_before_any_closure_executes() {
    let mut graph = Graph::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let target = graph.map((), move || {
        counter.fetch_add(1, Ordering::Relaxed);
        7_u64
    });
    let mut evaluator = Evaluator::new(&graph);
    for missing in [false, true] {
        let mut batch = RequestBatch::new(&graph);
        let context = batch.add_context(Context::new(&graph)).unwrap();
        batch.request(context, target).unwrap();
        let node = if missing { NodeId::null() } else { target.id };
        let wrong: Node<String> = Node::new(graph.id(), node);
        batch.request(context, wrong).unwrap();
        let result = evaluator.evaluate_batch(&graph, batch);
        let expected = if missing {
            GraphError::MissingNode { node }
        } else {
            GraphError::TypeMismatch { node }
        };
        assert!(matches!(result, Err(EvalError::Graph(error)) if error == expected));
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn context_hit_does_not_expand_temporary_result_map() {
    let mut graph = Graph::new();
    for _ in 0..10_000 {
        graph.constant(0_u64);
    }
    let target = graph.map(graph.frame(), |f: &FrameIndex| f.0);
    graph.set_context_cache_capacity(target, 1).unwrap();
    let mut evaluator = Evaluator::new(&graph);
    let mut context = Context::new(&graph);
    context.set(&graph, graph.frame(), FrameIndex(7)).unwrap();
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(context).unwrap();
    batch.request(id, target).unwrap();
    evaluator.evaluate_batch(&graph, batch).unwrap();
    let mut context = Context::new(&graph);
    context.set(&graph, graph.frame(), FrameIndex(7)).unwrap();
    let mut batch = RequestBatch::new(&graph);
    let id = batch.add_context(context).unwrap();
    let context = batch.context(id).unwrap();
    let mut values = SecondaryMap::new();
    evaluator.reset_stats();
    let output = evaluator
        .evaluate_node(&graph, context, id, &mut values, target.id)
        .unwrap();
    assert_eq!(output.downcast_ref::<u64>(), Some(&7));
    assert_eq!(values.len(), 0);
    assert_eq!(values.capacity(), 0);
    assert_eq!(evaluator.stats().resolution_visits, 1);
}
