use super::{
    error::GraphError,
    inputs::ComputationInputs,
    node::{
        Dependency, FrameIndex, GraphDefinitionEpoch, GraphId, Input, Node, NodeId, NodeKind,
        NodeRecord, next_id,
    },
    snapshot::{EvaluationSnapshot, InputData, SnapshotId},
    value::{SharedValue, ValueSemantics},
};
use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
    sync::{Arc, atomic::AtomicU64},
};

/// Immutable graph definition. Mutation is available only through GraphState::edit.
#[derive(Clone)]
pub struct Graph {
    pub(crate) id: GraphId,
    pub(crate) nodes: HashMap<NodeId, Arc<NodeRecord>>,
    frame: Node<FrameIndex>,
    defaults: HashMap<TypeId, ValueSemantics>,
    pub(crate) epoch: GraphDefinitionEpoch,
    // Shared across staged copies: rolling back a transaction cannot reuse a handle.
    allocator: Arc<AtomicU64>,
}
impl Graph {
    fn new() -> Self {
        let id = GraphId::new();
        let frame = Node::new(id, NodeId(0));
        let record = NodeRecord {
            type_id: TypeId::of::<FrameIndex>(),
            revision: 0,
            kind: NodeKind::Frame,
            semantics: ValueSemantics::by_eq::<FrameIndex>(),
            depends_on_frame: true,
        };
        Self {
            id,
            nodes: HashMap::from([(frame.id, Arc::new(record))]),
            frame,
            defaults: HashMap::new(),
            epoch: GraphDefinitionEpoch(0),
            allocator: Arc::new(AtomicU64::new(1)),
        }
    }
    pub fn frame(&self) -> Node<FrameIndex> {
        self.frame
    }
    pub fn id(&self) -> GraphId {
        self.id
    }
    pub fn definition_epoch(&self) -> GraphDefinitionEpoch {
        self.epoch
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    /// Graphs always contain the dedicated frame node.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn check<T: 'static>(&self, node: Node<T>) -> Result<(), GraphError> {
        self.check_node(node.graph_id, node.id, TypeId::of::<T>())
    }
    pub(crate) fn node_record(&self, id: NodeId) -> Result<&NodeRecord, GraphError> {
        self.nodes
            .get(&id)
            .map(AsRef::as_ref)
            .ok_or(GraphError::MissingNode { node: id })
    }
    pub(crate) fn check_node(
        &self,
        graph: GraphId,
        node: NodeId,
        ty: TypeId,
    ) -> Result<(), GraphError> {
        if graph != self.id {
            return Err(GraphError::ForeignNode { node });
        }
        if self.node_record(node)?.type_id != ty {
            return Err(GraphError::TypeMismatch { node });
        }
        Ok(())
    }
    fn check_inputs(&self, inputs: &[Dependency]) -> Result<(), GraphError> {
        for i in inputs {
            self.check_node(i.graph_id, i.source, i.type_id)?;
        }
        Ok(())
    }
    /// Checks every connected and disconnected subgraph without executing closures.
    pub fn validate(&self) -> Result<(), GraphError> {
        self.topology().map(|_| ())
    }
    fn topology(&self) -> Result<Vec<NodeId>, GraphError> {
        if !matches!(self.node_record(self.frame.id)?.kind, NodeKind::Frame) {
            return Err(GraphError::InvalidFrame);
        }
        let mut pending = HashMap::new();
        let mut children: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut ready = VecDeque::new();
        for (&id, record) in &self.nodes {
            let inputs = match &record.kind {
                NodeKind::Computation { inputs, .. } => inputs.as_ref(),
                _ => &[],
            };
            self.check_inputs(inputs)?;
            pending.insert(id, inputs.len());
            if inputs.is_empty() {
                ready.push_back(id);
            }
            for dep in inputs {
                children.entry(dep.source).or_default().push(id);
            }
        }
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_front() {
            order.push(id);
            for &child in children.get(&id).into_iter().flatten() {
                let count = pending.get_mut(&child).unwrap();
                *count -= 1;
                if *count == 0 {
                    ready.push_back(child);
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(GraphError::Cycle);
        }
        Ok(order)
    }
    fn update_frame_dependence(&mut self) -> Result<(), GraphError> {
        for id in self.topology()? {
            let dependent = match &self.nodes[&id].kind {
                NodeKind::Frame => true,
                NodeKind::Input => false,
                NodeKind::Computation { inputs, .. } => inputs
                    .iter()
                    .any(|i| self.nodes[&i.source].depends_on_frame),
            };
            if self.nodes[&id].depends_on_frame != dependent {
                Arc::make_mut(self.nodes.get_mut(&id).unwrap()).depends_on_frame = dependent;
            }
        }
        Ok(())
    }
    fn insert<T: 'static>(&mut self, kind: NodeKind) -> Node<T> {
        let id = NodeId(next_id(&self.allocator));
        let record = NodeRecord {
            type_id: TypeId::of::<T>(),
            revision: 0,
            kind,
            semantics: self
                .defaults
                .get(&TypeId::of::<T>())
                .cloned()
                .unwrap_or_default(),
            depends_on_frame: false,
        };
        self.nodes.insert(id, Arc::new(record));
        Node::new(self.id, id)
    }
}

/// Sole owner of the current editable graph and ordinary inputs.
///
/// Transactions stage changes and publish only after closure success and complete
/// validation. Snapshot reads clone Arcs without assigning a new version. Graph
/// metadata uses copy-on-write; closures and large inputs are never deep copied.
///
/// ```
/// use gekijyou_computation::{GraphState, FrameIndex};
/// let mut state = GraphState::new();
/// let (offset, position) = state.edit(|e| {
///     let offset = e.input::<u64>();
///     let position = e.map((e.frame(), offset.node()), |f: &FrameIndex, x: &u64| f.0 + x)?;
///     e.set(offset, 10)?;
///     Ok((offset, position))
/// })?;
/// let frozen = state.snapshot();
/// state.edit(|e| e.set(offset, 20))?;
/// assert_eq!(frozen.get(offset)?, Some(&10));
/// # Ok::<(), gekijyou_computation::GraphError>(())
/// ```
pub struct GraphState {
    graph: Arc<Graph>,
    inputs: Arc<InputData>,
    snapshot_id: SnapshotId,
}
impl GraphState {
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Graph::new()),
            inputs: Arc::new(HashMap::new()),
            snapshot_id: SnapshotId::new(),
        }
    }
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn snapshot(&self) -> EvaluationSnapshot {
        EvaluationSnapshot {
            id: self.snapshot_id,
            graph: self.graph.clone(),
            inputs: self.inputs.clone(),
        }
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
        self.snapshot().get_shared(input)
    }
    /// An error or panic drops all staged changes. Newly allocated node identities
    /// remain burned even if a handle escapes the failed transaction.
    pub fn edit<R>(
        &mut self,
        f: impl FnOnce(&mut GraphEdit) -> Result<R, GraphError>,
    ) -> Result<R, GraphError> {
        self.try_edit(f)
    }
    /// Transaction adapter for integration errors. `E` preserves caller errors
    /// while graph validation failures convert through `From<GraphError>`.
    pub fn try_edit<R, E: From<GraphError>>(
        &mut self,
        f: impl FnOnce(&mut GraphEdit) -> Result<R, E>,
    ) -> Result<R, E> {
        let mut edit = GraphEdit {
            graph: self.graph.clone(),
            inputs: self.inputs.clone(),
            definition_changed: false,
            inputs_changed: false,
        };
        let result = f(&mut edit)?;
        if edit.definition_changed {
            let epoch = self
                .graph
                .epoch
                .0
                .checked_add(1)
                .ok_or(GraphError::DefinitionEpochExhausted)?;
            let graph = Arc::make_mut(&mut edit.graph);
            graph.update_frame_dependence()?;
            graph.epoch = GraphDefinitionEpoch(epoch);
        }
        // Changed graph inputs are validated before any of the staged state is published.
        if edit.definition_changed || edit.inputs_changed {
            for (&node, value) in edit.inputs.iter() {
                let record = edit.graph.node_record(node)?;
                if !matches!(record.kind, NodeKind::Input) {
                    return Err(GraphError::NotInput { node }.into());
                }
                if record.type_id != value.as_ref().type_id() {
                    return Err(GraphError::TypeMismatch { node }.into());
                }
            }
            let id = SnapshotId::new();
            // Publish the entire basis before any old user-owned value is dropped.
            // A panicking destructor cannot leave graph and inputs from different commits.
            let old = std::mem::replace(
                self,
                Self {
                    graph: edit.graph,
                    inputs: edit.inputs,
                    snapshot_id: id,
                },
            );
            drop(old);
        } else {
            // Defaults affect future nodes only and do not alter the calculation basis.
            self.graph = edit.graph;
        }
        Ok(result)
    }
}
impl Default for GraphState {
    fn default() -> Self {
        Self::new()
    }
}

/// Controlled staged graph/input edits. Raw mutable storage is never exposed.
pub struct GraphEdit {
    graph: Arc<Graph>,
    inputs: Arc<InputData>,
    definition_changed: bool,
    inputs_changed: bool,
}
impl GraphEdit {
    pub fn frame(&self) -> Node<FrameIndex> {
        self.graph.frame()
    }
    pub fn input<T: Send + Sync + 'static>(&mut self) -> Input<T> {
        self.definition_changed = true;
        Input(Arc::make_mut(&mut self.graph).insert(NodeKind::Input))
    }
    pub fn set_default_eq<T: Eq + 'static>(&mut self) {
        Arc::make_mut(&mut self.graph)
            .defaults
            .insert(TypeId::of::<T>(), ValueSemantics::by_eq::<T>());
    }
    /// Frozen on node creation; replacing a node preserves its existing semantics.
    pub fn set_default_equivalence<T: 'static, F: Fn(&T, &T) -> bool + Send + Sync + 'static>(
        &mut self,
        f: F,
    ) {
        Arc::make_mut(&mut self.graph)
            .defaults
            .insert(TypeId::of::<T>(), ValueSemantics::by_equivalence::<T, F>(f));
    }
    pub fn map<I: ComputationInputs<F, O>, F, O: 'static>(
        &mut self,
        inputs: I,
        f: F,
    ) -> Result<Node<O>, GraphError> {
        let definition = inputs.build_computation(f);
        let (inputs, computation) = definition.into_parts();
        self.graph.check_inputs(&inputs)?;
        self.definition_changed = true;
        Ok(
            Arc::make_mut(&mut self.graph).insert(NodeKind::Computation {
                inputs: inputs.into(),
                computation: computation.into(),
            }),
        )
    }
    pub fn constant<T: Send + Sync + 'static>(&mut self, value: T) -> Node<T> {
        let value: SharedValue = Arc::new(value);
        self.definition_changed = true;
        Arc::make_mut(&mut self.graph).insert(NodeKind::Computation {
            inputs: Arc::new([]),
            computation: Arc::new(move |args| {
                assert!(args.is_empty());
                value.clone()
            }),
        })
    }
    /// Dependency validity is checked now; whole-transaction acyclicity at commit.
    pub fn replace<I: ComputationInputs<F, O>, F, O: 'static>(
        &mut self,
        node: Node<O>,
        inputs: I,
        f: F,
    ) -> Result<(), GraphError> {
        self.graph.check(node)?;
        let record = self.graph.node_record(node.id)?;
        if !matches!(record.kind, NodeKind::Computation { .. }) {
            return Err(GraphError::NotComputation { node: node.id });
        }
        let revision = record
            .revision
            .checked_add(1)
            .ok_or(GraphError::RevisionExhausted { node: node.id })?;
        let (inputs, computation) = inputs.build_computation(f).into_parts();
        self.graph.check_inputs(&inputs)?;
        let record = Arc::make_mut(
            Arc::make_mut(&mut self.graph)
                .nodes
                .get_mut(&node.id)
                .unwrap(),
        );
        record.kind = NodeKind::Computation {
            inputs: inputs.into(),
            computation: computation.into(),
        };
        record.revision = revision;
        self.definition_changed = true;
        Ok(())
    }
    pub fn set<T: Send + Sync + 'static>(
        &mut self,
        input: Input<T>,
        value: T,
    ) -> Result<(), GraphError> {
        self.set_shared(input, Arc::new(value))
    }
    /// Stores this Arc directly, without wrapping it in a second Arc.
    pub fn set_shared<T: Send + Sync + 'static>(
        &mut self,
        input: Input<T>,
        value: Arc<T>,
    ) -> Result<(), GraphError> {
        check_input(&self.graph, input)?;
        let value: SharedValue = value;
        if self
            .inputs
            .get(&input.0.id)
            .is_some_and(|old| Arc::ptr_eq(old, &value))
        {
            return Ok(());
        }
        Arc::make_mut(&mut self.inputs).insert(input.0.id, value);
        self.inputs_changed = true;
        Ok(())
    }
    /// Downstream references must all be removed or reconnected by transaction end.
    /// This allows deleting a connected set in either order. Old snapshots survive.
    pub fn remove<T: 'static>(&mut self, node: Node<T>) -> Result<(), GraphError> {
        self.graph.check(node)?;
        if node.id == self.graph.frame.id {
            return Err(GraphError::InvalidFrame);
        }
        Arc::make_mut(&mut self.graph).nodes.remove(&node.id);
        if self.inputs.contains_key(&node.id) {
            Arc::make_mut(&mut self.inputs).remove(&node.id);
            self.inputs_changed = true;
        }
        self.definition_changed = true;
        Ok(())
    }
}
pub(crate) fn check_input<T: 'static>(graph: &Graph, input: Input<T>) -> Result<(), GraphError> {
    graph.check(input.node())?;
    if !matches!(graph.node_record(input.0.id)?.kind, NodeKind::Input) {
        return Err(GraphError::NotInput { node: input.0.id });
    }
    Ok(())
}
