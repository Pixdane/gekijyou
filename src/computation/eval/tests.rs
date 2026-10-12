use super::*;
use crate::computation::{ComputationDefinition, ComputationInputs, Node};
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
    let mut batch = RequestBatch::new(&graph);
    let context = batch.add_context(Context::new(&graph)).unwrap();
    let request = batch.request(context, target).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    assert_eq!(results.get(request), Ok(&[7, 2, 7, 3, 2]));
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
    let mut graph = Graph::new();
    let node = graph.map(WrongOutput, ());
    let mut batch = RequestBatch::new(&graph);
    let context = batch.add_context(Context::new(&graph)).unwrap();
    batch.request(context, node).unwrap();
    assert!(
        matches!(Evaluator::new(&graph).evaluate_batch(&graph, batch),
        Err(EvalError::Graph(GraphError::TypeMismatch { node: id })) if id == node.id)
    );
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
