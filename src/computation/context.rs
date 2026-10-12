use super::{
    error::EvalError,
    graph::Graph,
    node::{GraphId, Node, NodeId, NodeKind},
    value::SharedValue,
};
use slotmap::SecondaryMap;
use std::{any::TypeId, sync::Arc};

/// External input values for one graph, without borrowing its definitions.
///
/// Values must obey the module's semantic immutability contract. Unassigned inputs
/// are allowed; only inputs reached during evaluation need values. Registering a
/// context with [`RequestBatch::add_context`](super::RequestBatch::add_context)
/// transfers ownership and freezes its assignments for that batch.
///
/// ```compile_fail
/// use gekijyou::computation::{Context, FrameIndex, Graph, RequestBatch};
/// let graph = Graph::new();
/// let mut context = Context::new(&graph);
/// let mut batch = RequestBatch::new(&graph);
/// batch.add_context(context).unwrap();
/// context.set(&graph, graph.frame(), FrameIndex(1)).unwrap();
/// ```
pub struct Context {
    pub(super) graph_id: GraphId,
    values: SecondaryMap<NodeId, SharedValue>,
}

impl Context {
    /// Creates an empty context bound to this graph's identity, not its revisions.
    pub fn new(graph: &Graph) -> Self {
        Self {
            graph_id: graph.id(),
            values: SecondaryMap::new(),
        }
    }

    /// Sets or replaces an external input, validating before changing any value.
    ///
    /// The context, graph and node must belong to the same graph. Computation
    /// nodes, including constants, cannot receive external values.
    pub fn set<T: Send + Sync + 'static>(
        &mut self,
        graph: &Graph,
        node: Node<T>,
        value: T,
    ) -> Result<(), EvalError> {
        if self.graph_id != graph.id() {
            return Err(EvalError::ForeignGraph);
        }
        graph.check_node(node.graph_id, node.id, TypeId::of::<T>())?;
        if !matches!(graph.node_record(node.id)?.kind, NodeKind::Input) {
            return Err(EvalError::NotInput { node: node.id });
        }
        self.values.insert(node.id, Arc::new(value));
        Ok(())
    }

    pub(super) fn value(&self, node: NodeId) -> Option<&SharedValue> {
        self.values.get(node)
    }
}

#[cfg(test)]
mod tests {
    use super::super::error::GraphError;
    use super::*;
    use slotmap::Key;

    #[test]
    fn invalid_typed_and_missing_handles_do_not_change_assignments() {
        let mut graph = Graph::new();
        let input = graph.input::<u64>();
        let mut context = Context::new(&graph);
        context.set(&graph, input, 17).unwrap();
        let wrong: Node<String> = Node::new(graph.id(), input.id);
        assert_eq!(
            context.set(&graph, wrong, "wrong".into()),
            Err(EvalError::Graph(GraphError::TypeMismatch {
                node: input.id
            }))
        );
        let missing: Node<u64> = Node::new(graph.id(), NodeId::null());
        assert_eq!(
            context.set(&graph, missing, 19),
            Err(EvalError::Graph(GraphError::MissingNode {
                node: missing.id
            }))
        );
        assert_eq!(
            context.value(input.id).unwrap().downcast_ref::<u64>(),
            Some(&17)
        );
    }
}
