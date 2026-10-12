use super::{
    error::GraphError,
    graph::Graph,
    node::{NodeId, NodeKind},
    value::SharedValue,
};
use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

struct ContextEntry {
    root_values: Box<[SharedValue]>,
    output: SharedValue,
}

/// Node-local external inputs in fixed order, plus oldest-to-newest history.
/// Bounds retained entry count, not the byte size of the objects owned by Arcs.
pub(super) struct NodeContextCache {
    pub(super) replacement_epoch: u64,
    capacity: usize,
    pub(super) roots: Box<[NodeId]>,
    entries: VecDeque<ContextEntry>,
}

impl NodeContextCache {
    pub(super) fn new(graph: &Graph, node: NodeId, capacity: usize) -> Result<Self, GraphError> {
        Ok(Self {
            replacement_epoch: graph.replacement_epoch(),
            capacity,
            roots: root_inputs(graph, node)?,
            entries: VecDeque::new(),
        })
    }

    pub(super) fn synchronize_capacity(&mut self, capacity: usize) {
        if self.capacity != capacity {
            self.entries.clear();
            self.capacity = capacity;
        }
    }

    /// Borrow current context inputs through the callback; hits allocate no input
    /// buffer and do not clone every input Arc. Compare before changing recency so
    /// errors and unwinding comparators leave the queue untouched.
    pub(super) fn lookup<E>(
        &mut self,
        mut equivalent: impl FnMut(NodeId, &SharedValue) -> Result<bool, E>,
    ) -> Result<Option<SharedValue>, E> {
        for index in (0..self.entries.len()).rev() {
            let entry = &self.entries[index];
            let mut matches = true;
            for (&root, previous) in self.roots.iter().zip(&entry.root_values) {
                if !equivalent(root, previous)? {
                    matches = false;
                    break;
                }
            }
            if matches {
                let entry = self.entries.remove(index).expect("Existing context entry");
                let output = Arc::clone(&entry.output);
                self.entries.push_back(entry);
                return Ok(Some(output));
            }
        }
        Ok(None)
    }

    /// Only publish after DFS has successfully resolved and validated the output.
    pub(super) fn insert(&mut self, root_values: Box<[SharedValue]>, output: SharedValue) {
        debug_assert_eq!(root_values.len(), self.roots.len());
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(ContextEntry {
            root_values,
            output,
        });
    }
}

/// Iterative DFS in dependency order. Membership is never iterated, so randomized
/// HashSet order cannot affect the projection. Shared ancestors and repeated edges
/// are visited once. Structural constants contribute no external inputs.
fn root_inputs(graph: &Graph, node: NodeId) -> Result<Box<[NodeId]>, GraphError> {
    let mut stack = vec![node];
    let mut visited = HashSet::new();
    let mut roots = Vec::new();
    while let Some(node) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }
        match &graph.node_record(node)?.kind {
            NodeKind::Input => roots.push(node),
            NodeKind::Computation { inputs, .. } => {
                stack.extend(inputs.iter().rev().map(|input| input.source));
            }
        }
    }
    Ok(roots.into_boxed_slice())
}

#[cfg(test)]
mod tests;
