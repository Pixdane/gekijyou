use super::{
    Graph, GraphDefinitionEpoch, GraphError, GraphId, Input, Node, NodeId, graph::check_input,
    node::next_id, value::SharedValue,
};
use std::{
    collections::HashMap,
    sync::{Arc, atomic::AtomicU64},
};
pub(crate) type InputData = HashMap<NodeId, SharedValue>;

/// Unique identity of a published calculation basis; clones retain it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotId(u64);
impl SnapshotId {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(next_id(&NEXT))
    }
}

/// Frozen graph and ordinary inputs. No cache, mutable world or mutable graph access.
#[derive(Clone)]
pub struct EvaluationSnapshot {
    pub(crate) id: SnapshotId,
    pub(crate) graph: Arc<Graph>,
    pub(crate) inputs: Arc<InputData>,
}
impl EvaluationSnapshot {
    pub fn id(&self) -> SnapshotId {
        self.id
    }
    pub fn graph_id(&self) -> GraphId {
        self.graph.id
    }
    pub fn definition_epoch(&self) -> GraphDefinitionEpoch {
        self.graph.epoch
    }
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn depends_on_frame<T: 'static>(&self, node: Node<T>) -> Result<bool, GraphError> {
        self.graph.check(node)?;
        Ok(self.graph.node_record(node.id)?.depends_on_frame)
    }
    pub fn get<T: Send + Sync + 'static>(&self, input: Input<T>) -> Result<Option<&T>, GraphError> {
        check_input(&self.graph, input)?;
        self.inputs
            .get(&input.0.id)
            .map(|v| {
                v.downcast_ref::<T>()
                    .ok_or(GraphError::TypeMismatch { node: input.0.id })
            })
            .transpose()
    }
    pub fn get_shared<T: Send + Sync + 'static>(
        &self,
        input: Input<T>,
    ) -> Result<Option<Arc<T>>, GraphError> {
        check_input(&self.graph, input)?;
        self.inputs
            .get(&input.0.id)
            .map(|v| {
                v.clone()
                    .downcast::<T>()
                    .map_err(|_| GraphError::TypeMismatch { node: input.0.id })
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphState;
    #[test]
    fn stable_reads_allocate_nothing_and_input_edit_preserves_graph_storage() {
        let mut state = GraphState::new();
        let input = state
            .edit(|e| {
                let i = e.input::<Vec<u8>>();
                e.set(i, vec![1; 1024])?;
                Ok(i)
            })
            .unwrap();
        let old = state.snapshot();
        assert_eq!(
            crate::test_allocations::count(|| {
                for _ in 0..1000 {
                    std::hint::black_box(state.snapshot());
                }
            }),
            0
        );
        state.edit(|e| e.set(input, vec![2; 1024])).unwrap();
        let new = state.snapshot();
        assert!(Arc::ptr_eq(&old.graph, &new.graph));
        assert!(!Arc::ptr_eq(&old.inputs, &new.inputs));
        assert_eq!(old.definition_epoch(), new.definition_epoch());
    }
}
