use super::{Graph, GraphError, NodeId, node::NodeKind, value::SharedValue};
use std::collections::{HashSet, VecDeque};
struct Entry {
    roots: Box<[SharedValue]>,
    output: SharedValue,
}
pub(crate) struct NodeContextCache {
    pub(crate) roots: Box<[NodeId]>,
    capacity: usize,
    entries: VecDeque<Entry>,
}
impl NodeContextCache {
    pub(crate) fn new(graph: &Graph, node: NodeId, capacity: usize) -> Result<Self, GraphError> {
        let mut stack = vec![node];
        let mut visited = HashSet::new();
        let mut roots = Vec::new();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            match &graph.node_record(id)?.kind {
                NodeKind::Input | NodeKind::Frame => roots.push(id),
                NodeKind::Computation { inputs, .. } => {
                    stack.extend(inputs.iter().rev().map(|i| i.source))
                }
            }
        }
        Ok(Self {
            roots: roots.into_boxed_slice(),
            capacity,
            entries: VecDeque::new(),
        })
    }
    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }
    pub(crate) fn lookup<E>(
        &mut self,
        mut equivalent: impl FnMut(NodeId, &SharedValue) -> Result<bool, E>,
    ) -> Result<Option<SharedValue>, E> {
        for index in (0..self.entries.len()).rev() {
            let entry = &self.entries[index];
            let mut matches = true;
            for (&id, old) in self.roots.iter().zip(&entry.roots) {
                if !equivalent(id, old)? {
                    matches = false;
                    break;
                }
            }
            if matches {
                let entry = self.entries.remove(index).unwrap();
                let result = entry.output.clone();
                self.entries.push_back(entry);
                return Ok(Some(result));
            }
        }
        Ok(None)
    }
    pub(crate) fn insert(&mut self, roots: Box<[SharedValue]>, output: SharedValue) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(Entry { roots, output });
    }
}
