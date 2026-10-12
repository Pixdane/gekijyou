use super::{
    error::GraphError,
    inputs::{ComputationDefinition, ComputationInputs},
    node::{CachePolicy, Dependency, FrameIndex, GraphId, Node, NodeId, NodeKind, NodeRecord},
    value::{ErasedComputation, SharedValue, ValueSemantics},
};
use slotmap::{SecondaryMap, SlotMap};
use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
    sync::Arc,
};

/// An animation computation graph with typed nodes and explicit dependencies.
///
/// Each graph starts with a frame-index input for animation sampling. Obtain it
/// with [`Graph::frame`] and add computations with [`Graph::map`]. New nodes depend
/// on existing nodes; [`Graph::replace`] may reconnect computations to any existing
/// nodes as long as the graph remains acyclic.
///
/// Graph identities validate handle ownership at runtime; they are not stable
/// object identifiers across project rebuilds.
pub struct Graph {
    id: GraphId,
    nodes: SlotMap<NodeId, NodeRecord>,
    frame: Node<FrameIndex>,
    default_value_semantics: HashMap<TypeId, ValueSemantics>,
    // Counts successful replacements only; node additions do not change this epoch.
    replacement_epoch: u64,
}

impl Graph {
    /// Creates a graph containing only its frame input node.
    ///
    /// Each call allocates a distinct graph identity. [`Default::default`] has
    /// the same behavior.
    ///
    /// # Panics
    ///
    /// Panics if the process-local graph identity counter is exhausted, rather
    /// than reusing an existing identity.
    pub fn new() -> Self {
        let id = GraphId::new();
        let mut nodes = SlotMap::with_key();
        let frame_semantics = ValueSemantics::by_eq::<FrameIndex>();
        let default_value_semantics =
            HashMap::from([(TypeId::of::<FrameIndex>(), frame_semantics.clone())]);
        let frame = Node::new(
            id,
            nodes.insert(NodeRecord {
                type_id: TypeId::of::<FrameIndex>(),
                revision: 0,
                kind: NodeKind::Input,
                semantics: frame_semantics,
                cache_policy: CachePolicy::default(),
            }),
        );

        Self {
            id,
            nodes,
            frame,
            default_value_semantics,
            replacement_epoch: 0,
        }
    }

    /// Returns a handle to the frame input node.
    ///
    /// Returns the handle by value without retaining a graph borrow, allowing
    /// `graph.map(graph.frame(), |f: &FrameIndex| f.0)`.
    /// Does not set or read an actual frame input value.
    pub fn frame(&self) -> Node<FrameIndex> {
        self.frame
    }

    /// Declares an external input whose value is supplied by a context.
    pub fn input<T: Send + Sync + 'static>(&mut self) -> Node<T> {
        let record = self.record::<T>(NodeKind::Input);
        let id = self.nodes.insert(record);
        Node::new(self.id, id)
    }

    /// Registers dependencies and a closure, returning the typed output handle.
    ///
    /// Built-in inputs support `()`, a single node, and tuples of one to four nodes.
    /// The closure receives shared references in input order and returns `O`.
    /// Repeated input nodes are legal. This does not invoke `f`; input handles
    /// remain available for other branches.
    ///
    /// Built-in adapters require input types to be `'static` and the output and
    /// closure to be `Send + Sync + 'static`. Use `move` to capture owned data;
    /// this does not extend the lifetime of references it contains. Copying node
    /// handles does not require input values to be copyable.
    ///
    /// `Fn + Send + Sync` does not guarantee purity or determinism. Model external
    /// inputs explicitly and uphold the module's computation and value contracts.
    ///
    /// # Panics
    ///
    /// Panics if an input belongs to another graph, is missing, or has the wrong
    /// type. Invalid dependencies are rejected before inserting the node.
    pub fn map<I, F, O>(&mut self, inputs: I, f: F) -> Node<O>
    where
        I: ComputationInputs<F, O>,
        O: 'static,
    {
        self.insert_computation(inputs.build_computation(f))
    }

    /// Stores a constant as a zero-input computation, without requiring `T: Clone`.
    ///
    /// Its closure returns the same shared value on each invocation. The graph
    /// retains the value even when no evaluator caches its result.
    pub fn constant<T: Send + Sync + 'static>(&mut self, value: T) -> Node<T> {
        let value: SharedValue = Arc::new(value);
        let computation: Box<ErasedComputation> = Box::new(move |args| {
            assert!(args.is_empty(), "Invalid input count");
            Arc::clone(&value)
        });
        self.insert_computation(ComputationDefinition::<T>::new(Box::new([]), computation))
    }

    /// Sets `T::eq` as the default equivalence for subsequently created nodes of `T`.
    ///
    /// Existing nodes, including the built-in frame input, retain their semantics.
    /// `Eq` alone does not establish that every downstream closure respects it;
    /// callers must uphold the module's substitution and immutability contracts.
    pub fn set_default_eq<T: Eq + 'static>(&mut self) {
        self.default_value_semantics
            .insert(TypeId::of::<T>(), ValueSemantics::by_eq::<T>());
    }

    /// Sets the default equivalence for subsequently created nodes of `T`.
    ///
    /// Existing nodes retain their semantics, including after replacement. The
    /// comparator must be stable, reflexive, symmetric and transitive; published
    /// values must be semantically immutable, and every downstream computation
    /// must permit substituting equivalent inputs. General tolerance comparisons
    /// usually violate these requirements and must not be used as cache equality.
    /// Shared-object identity always counts as equivalent without invoking `f`.
    ///
    /// ```
    /// use gekijyou::computation::Graph;
    /// let mut graph = Graph::new();
    /// graph.set_default_equivalence::<f64, _>(|a, b| a.to_bits() == b.to_bits());
    /// let input = graph.input::<f64>();
    /// ```
    pub fn set_default_equivalence<T: 'static, F>(&mut self, f: F)
    where
        F: Fn(&T, &T) -> bool + Send + Sync + 'static,
    {
        self.default_value_semantics
            .insert(TypeId::of::<T>(), ValueSemantics::by_equivalence::<T, F>(f));
    }

    /// Configures persistent result retention for a computation node.
    ///
    /// Validates graph ownership, existence and type. Does not change the node's
    /// definition, revision or the graph's replacement epoch. Each evaluator
    /// applies the latest policy lazily on the next visit, discarding that node's
    /// entries when its applied policy differs in variant or capacity, including
    /// `Last` to `Lru { capacity: 1 }`. Reapplying the same policy retains entries.
    /// External inputs always read the current Context; configuring them returns
    /// [`GraphError::InputNodeCachePolicy`] without changing any metadata.
    pub fn set_cache_policy<T: 'static>(
        &mut self,
        node: Node<T>,
        policy: CachePolicy,
    ) -> Result<(), GraphError> {
        self.check_node(node.graph_id, node.id, TypeId::of::<T>())?;
        if matches!(self.nodes[node.id].kind, NodeKind::Input) {
            return Err(GraphError::InputNodeCachePolicy { node: node.id });
        }
        self.nodes[node.id].cache_policy = policy;
        Ok(())
    }

    /// Replaces a computation's closure and inputs, preserving its typed handle.
    ///
    /// `inputs` supports `()`, a single node, or tuples of one to four nodes.
    /// The closure is registered without being invoked. Output type, value
    /// semantics, and cache policy are preserved. Input nodes cannot be replaced.
    ///
    /// Increments this node's revision and the graph's replacement epoch on success.
    /// Returns an error for invalid handles, dependencies, cycles, or exhausted
    /// version counters, leaving the graph unchanged.
    /// Validation takes O(nodes + edges) time and space.
    pub fn replace<I, F, O>(&mut self, target: Node<O>, inputs: I, f: F) -> Result<(), GraphError>
    where
        I: ComputationInputs<F, O>,
        O: 'static,
    {
        self.replace_computation(target, inputs.build_computation(f))
    }

    /// Checks frame identity, dependency ownership, existence, types, and acyclicity.
    ///
    /// Checks the entire graph, including disconnected nodes, without executing
    /// closures. Takes O(nodes + edges) time and space; repeated edges are legal.
    pub fn validate(&self) -> Result<(), GraphError> {
        self.validate_with_replacement(None)
    }

    // Structural roots include every external input and zero-input computation.
    // Derive them from node definitions so graph edits need no separate root state.
    fn roots(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().filter_map(|(id, record)| {
            let is_root = match &record.kind {
                NodeKind::Input => true,
                NodeKind::Computation { inputs, .. } => inputs.is_empty(),
            };
            is_root.then_some(id)
        })
    }

    fn record<T: 'static>(&self, kind: NodeKind) -> NodeRecord {
        NodeRecord {
            type_id: TypeId::of::<T>(),
            revision: 0,
            kind,
            semantics: self
                .default_value_semantics
                .get(&TypeId::of::<T>())
                .cloned()
                .unwrap_or_default(),
            cache_policy: CachePolicy::default(),
        }
    }

    pub(super) fn id(&self) -> GraphId {
        self.id
    }

    pub(super) fn node_record(&self, id: NodeId) -> Result<&NodeRecord, GraphError> {
        self.nodes
            .get(id)
            .ok_or(GraphError::MissingNode { node: id })
    }

    pub(super) fn check_node(
        &self,
        graph_id: GraphId,
        id: NodeId,
        type_id: TypeId,
    ) -> Result<(), GraphError> {
        if graph_id != self.id {
            return Err(GraphError::ForeignNode { node: id });
        }
        let record = self
            .nodes
            .get(id)
            .ok_or(GraphError::MissingNode { node: id })?;
        if record.type_id != type_id {
            return Err(GraphError::TypeMismatch { node: id });
        }
        Ok(())
    }

    fn check_inputs(&self, inputs: &[Dependency]) -> Result<(), GraphError> {
        for input in inputs {
            self.check_node(input.graph_id, input.source, input.type_id)?;
        }
        Ok(())
    }

    // The typed definition preserves output identity across the erasure boundary.
    // Check every edge before insertion so rejection is non-mutating.
    fn insert_computation<T: 'static>(&mut self, definition: ComputationDefinition<T>) -> Node<T> {
        let (inputs, computation) = definition.into_parts();
        self.check_inputs(&inputs)
            .unwrap_or_else(|error| panic!("{error}"));
        let record = self.record::<T>(NodeKind::Computation {
            inputs,
            computation,
        });
        let id = self.nodes.insert(record);
        Node::new(self.id, id)
    }

    fn replace_computation<T: 'static>(
        &mut self,
        target: Node<T>,
        definition: ComputationDefinition<T>,
    ) -> Result<(), GraphError> {
        self.check_node(target.graph_id, target.id, TypeId::of::<T>())?;
        if matches!(self.nodes[target.id].kind, NodeKind::Input) {
            return Err(GraphError::InputNodeReplacement { node: target.id });
        }
        let (inputs, computation) = definition.into_parts();
        self.validate_with_replacement(Some((target.id, &inputs)))?;
        let revision = self.nodes[target.id]
            .revision
            .checked_add(1)
            .ok_or(GraphError::RevisionExhausted { node: target.id })?;
        let replacement_epoch = self
            .replacement_epoch
            .checked_add(1)
            .ok_or(GraphError::ReplacementEpochExhausted)?;
        // All fallible checks precede the commit. Existing semantics and policy
        // belong to the node's output contract, not to its replacement closure.
        self.nodes[target.id].kind = NodeKind::Computation {
            inputs,
            computation,
        };
        self.nodes[target.id].revision = revision;
        self.replacement_epoch = replacement_epoch;
        Ok(())
    }

    fn validate_with_replacement(
        &self,
        replacement: Option<(NodeId, &[Dependency])>,
    ) -> Result<(), GraphError> {
        self.check_node(
            self.frame.graph_id,
            self.frame.id,
            TypeId::of::<FrameIndex>(),
        )?;
        if !matches!(self.nodes[self.frame.id].kind, NodeKind::Input) {
            return Err(GraphError::InvalidFrame);
        }

        let mut pending: SecondaryMap<NodeId, usize> = SecondaryMap::new();
        let mut dependents: SecondaryMap<NodeId, Vec<NodeId>> = SecondaryMap::new();
        let mut ready: VecDeque<_> = self
            .roots()
            .filter(|&id| replacement.is_none_or(|(target, _)| id != target))
            .collect();
        // The target's current root status is excluded above. Seed it only when
        // the candidate definition has zero inputs, before committing the edit.
        if let Some((target, inputs)) = replacement
            && inputs.is_empty()
        {
            ready.push_back(target);
        }
        for (id, record) in &self.nodes {
            let inputs = match replacement {
                Some((target, inputs)) if id == target => inputs,
                _ => match &record.kind {
                    NodeKind::Input => &[],
                    NodeKind::Computation { inputs, .. } => inputs.as_ref(),
                },
            };
            self.check_inputs(inputs)?;
            pending.insert(id, inputs.len());
            for input in inputs {
                // Count and notify per edge, preserving duplicate dependencies.
                dependents
                    .entry(input.source)
                    .unwrap()
                    .or_default()
                    .push(id);
            }
        }

        let mut visited = 0;
        while let Some(id) = ready.pop_front() {
            visited += 1;
            if let Some(children) = dependents.get(id) {
                for &child in children {
                    pending[child] -= 1;
                    if pending[child] == 0 {
                        ready.push_back(child);
                    }
                }
            }
        }
        if visited != self.nodes.len() {
            return Err(GraphError::Cycle);
        }
        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
