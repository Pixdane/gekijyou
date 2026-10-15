use bevy_ecs::{component::Mutable, prelude::*};
use bevy_math::{Quat, Vec3};
use bevy_transform::components::Transform;
use gekijyou_computation::{
    EvalError, EvaluationResults, EvaluationSnapshot, FrameIndex, GraphEdit, GraphError, GraphId,
    GraphState, Input, Node, NodeId, RequestBatch, RequestBatchId, RequestTicket,
};
use rustc_hash::FxHashMap as HashMap;
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BindingError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Evaluation(#[from] EvalError),
    #[error("An input already has an authoritative source")]
    InputConflict,
    #[error("Output component write ranges overlap")]
    OutputConflict,
    #[error("Input/output ranges would implicitly feed output back into input")]
    FeedbackConflict,
    #[error("Input source entity no longer exists")]
    MissingEntity,
    #[error("Bound component is missing")]
    MissingComponent,
    #[error("Request belongs to another binding registry")]
    ForeignBindings,
    #[error("Bindings changed since this request was built")]
    StaleBindings,
    #[error("Application snapshot does not match the request/results")]
    SnapshotMismatch,
    #[error("Output handle does not belong to this registry or has been removed")]
    UnknownOutput,
    #[error("Frame was not included in this bound request")]
    MissingFrame,
    #[error("The request batch has already been taken")]
    BatchTaken,
}

/// Custom components default to Whole. Fine fields are an explicit user assertion
/// that differently named ranges can be read/written independently and safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteRange {
    Whole,
    Field(&'static str),
}
impl WriteRange {
    fn overlaps(self, other: Self) -> bool {
        self == Self::Whole || other == Self::Whole || self == other
    }
}

type Reader<C, T> = dyn Fn(&C) -> T + Send + Sync;
type SharedReader<C, T> = dyn Fn(&C) -> Arc<T> + Send + Sync;
enum InputReader<C, T> {
    Owned(Arc<Reader<C, T>>),
    Shared(Arc<SharedReader<C, T>>),
}

/// Typed ECS extractor; no reflection or dynamic field offsets.
pub struct InputField<C, T> {
    read: InputReader<C, T>,
    range: WriteRange,
}
impl<C, T> InputField<C, T> {
    pub fn new(f: impl Fn(&C) -> T + Send + Sync + 'static) -> Self {
        Self {
            read: InputReader::Owned(Arc::new(f)),
            range: WriteRange::Whole,
        }
    }
    /// Freezes an immutable shared resource directly, without deep cloning T or
    /// creating `Arc<Arc<T>>`. Reusing the same Arc preserves snapshot identity.
    pub fn shared(f: impl Fn(&C) -> Arc<T> + Send + Sync + 'static) -> Self {
        Self {
            read: InputReader::Shared(Arc::new(f)),
            range: WriteRange::Whole,
        }
    }
    pub fn with_range(mut self, range: WriteRange) -> Self {
        self.range = range;
        self
    }
}
impl InputField<Transform, Vec3> {
    pub fn transform_translation() -> Self {
        Self::new(|t: &Transform| t.translation).with_range(WriteRange::Field("translation"))
    }
}

/// Typed ECS writer; custom bindings conservatively own the whole component.
/// Writers must preserve component invariants if they panic. Apply prevalidates
/// values/components, but cannot roll back earlier ECS writes after a user panic.
pub struct OutputField<C, T> {
    write: Arc<Writer<C, T>>,
    range: WriteRange,
    marker: PhantomData<fn(T) -> T>,
}
type Writer<C, T> = dyn Fn(&mut C, &T) + Send + Sync;
impl<C, T> OutputField<C, T> {
    pub fn new(f: impl Fn(&mut C, &T) + Send + Sync + 'static) -> Self {
        Self {
            write: Arc::new(f),
            range: WriteRange::Whole,
            marker: PhantomData,
        }
    }
    pub fn with_range(mut self, range: WriteRange) -> Self {
        self.range = range;
        self
    }
}
impl<C: Clone> OutputField<C, C> {
    pub fn component() -> Self {
        Self::new(|c: &mut C, v: &C| *c = v.clone())
    }
}
impl OutputField<Transform, Vec3> {
    pub fn transform_translation() -> Self {
        Self::new(|c: &mut Transform, v: &Vec3| c.translation = *v)
            .with_range(WriteRange::Field("translation"))
    }
    pub fn transform_scale() -> Self {
        Self::new(|c: &mut Transform, v: &Vec3| c.scale = *v).with_range(WriteRange::Field("scale"))
    }
}
impl OutputField<Transform, Quat> {
    pub fn transform_rotation() -> Self {
        Self::new(|c: &mut Transform, v: &Quat| c.rotation = *v)
            .with_range(WriteRange::Field("rotation"))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NodeKey {
    graph: GraphId,
    node: NodeId,
}
impl NodeKey {
    pub(crate) fn of<T>(node: Node<T>) -> Self {
        Self {
            graph: node.graph_id(),
            node: node.id(),
        }
    }
}
struct Destination {
    entity: Entity,
    component: TypeId,
    range: WriteRange,
}
impl Destination {
    fn overlaps(&self, other: &Self) -> bool {
        self.entity == other.entity
            && self.component == other.component
            && self.range.overlaps(other.range)
    }
}
type Check = dyn Fn(&EvaluationSnapshot) -> Result<(), GraphError> + Send + Sync;
type SyncInput = dyn Fn(&World, &mut GraphEdit) -> Result<(), BindingError> + Send + Sync;
type ValidateComponent = dyn Fn(&World) -> Result<(), BindingError> + Send + Sync;
type RequestNode = dyn Fn(&mut RequestBatch, FrameIndex) -> RequestTicket + Send + Sync;
type ValidateTicket =
    dyn Fn(&EvaluationResults, RequestTicket) -> Result<(), EvalError> + Send + Sync;
type ApplyTicket =
    dyn Fn(&mut World, &EvaluationResults, RequestTicket) -> Result<(), BindingError> + Send + Sync;

struct InputEntry {
    key: NodeKey,
    source: Destination,
    check: Box<Check>,
    sync: Box<SyncInput>,
}
struct OutputEntry {
    id: u64,
    node: NodeKey,
    target: Destination,
    check: Box<Check>,
    request: Box<RequestNode>,
    validate_component: Box<ValidateComponent>,
    validate_result: Box<ValidateTicket>,
    apply: Box<ApplyTicket>,
}
#[derive(Clone, Copy)]
struct PlannedOutput {
    entry: usize,
    ticket: RequestTicket,
}

/// Stable output selection handle. Pruning never reuses a removed handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputBinding {
    registry: u64,
    id: u64,
}

/// Frozen mapping between an explicit output/frame selection and its typed batch.
/// Taking the batch leaves the mapping available for repeated or delayed Apply.
/// This retains the immutable computation basis, with no final-output frame cache.
pub struct BoundRequest {
    batch_id: RequestBatchId,
    batch: Option<RequestBatch>,
    snapshot: EvaluationSnapshot,
    registry: u64,
    revision: u64,
    frame_slices: HashMap<FrameIndex, std::ops::Range<usize>>,
    outputs: Vec<PlannedOutput>,
}
impl BoundRequest {
    pub fn snapshot(&self) -> &EvaluationSnapshot {
        &self.snapshot
    }
    pub fn take_batch(&mut self) -> Result<RequestBatch, BindingError> {
        self.batch.take().ok_or(BindingError::BatchTaken)
    }
}

/// Long-lived Node bindings. Request tickets belong to each explicit bound request.
/// One input has one authoritative source; outputs check conservative write ranges.
#[derive(Resource)]
pub struct Bindings {
    identity: u64,
    inputs: Vec<InputEntry>,
    outputs: Vec<OutputEntry>,
    output_index: HashMap<u64, usize>,
    revision: u64,
    graph: Option<GraphId>,
}
impl Default for Bindings {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            identity: NEXT
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("Binding identity exhausted"),
            inputs: Vec::new(),
            outputs: Vec::new(),
            output_index: HashMap::default(),
            revision: 0,
            graph: None,
        }
    }
}
impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    fn advance(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("Binding revision exhausted");
    }
    pub fn bind_input<C: Component, T: Send + Sync + 'static>(
        &mut self,
        snapshot: &EvaluationSnapshot,
        entity: Entity,
        input: Input<T>,
        field: InputField<C, T>,
    ) -> Result<(), BindingError> {
        snapshot.get(input)?;
        if self.graph.is_some_and(|id| id != snapshot.graph_id()) {
            return Err(GraphError::ForeignNode {
                node: input.node().id(),
            }
            .into());
        }
        let key = NodeKey::of(input.node());
        let source = Destination {
            entity,
            component: TypeId::of::<C>(),
            range: field.range,
        };
        if self.inputs.iter().any(|i| i.key == key) {
            return Err(BindingError::InputConflict);
        }
        if self.outputs.iter().any(|o| o.target.overlaps(&source)) {
            return Err(BindingError::FeedbackConflict);
        }
        self.advance();
        self.graph = Some(snapshot.graph_id());
        self.inputs.push(InputEntry {
            key,
            source,
            check: Box::new(move |s| s.get(input).map(|_| ())),
            sync: Box::new(move |world, edit| {
                if world.get_entity(entity).is_err() {
                    return Err(BindingError::MissingEntity);
                }
                let c = world
                    .get::<C>(entity)
                    .ok_or(BindingError::MissingComponent)?;
                match &field.read {
                    InputReader::Owned(read) => edit.set(input, read(c))?,
                    InputReader::Shared(read) => edit.set_shared(input, read(c))?,
                };
                Ok(())
            }),
        });
        Ok(())
    }
    pub fn bind_output<C: Component<Mutability = Mutable>, T: Send + Sync + 'static>(
        &mut self,
        snapshot: &EvaluationSnapshot,
        entity: Entity,
        node: Node<T>,
        field: OutputField<C, T>,
    ) -> Result<OutputBinding, BindingError> {
        snapshot.graph().check(node)?;
        if self.graph.is_some_and(|id| id != snapshot.graph_id()) {
            return Err(GraphError::ForeignNode { node: node.id() }.into());
        }
        let target = Destination {
            entity,
            component: TypeId::of::<C>(),
            range: field.range,
        };
        if self.outputs.iter().any(|o| o.target.overlaps(&target)) {
            return Err(BindingError::OutputConflict);
        }
        if self.inputs.iter().any(|i| i.source.overlaps(&target)) {
            return Err(BindingError::FeedbackConflict);
        }
        self.advance();
        self.graph = Some(snapshot.graph_id());
        let node_key = NodeKey::of(node);
        let write = field.write;
        self.output_index.insert(self.revision, self.outputs.len());
        self.outputs.push(OutputEntry {
            id: self.revision,
            node: node_key,
            target,
            check: Box::new(move |s| s.graph().check(node)),
            request: Box::new(move |batch, frame| batch.at(frame).request(node).ticket()),
            validate_component: Box::new(move |world| {
                if world.get_entity(entity).is_err() {
                    return Ok(());
                }
                if world.get::<C>(entity).is_none() {
                    return Err(BindingError::MissingComponent);
                }
                Ok(())
            }),
            validate_result: Box::new(move |results, ticket| {
                results.get_by_ticket::<T>(ticket).map(|_| ())
            }),
            apply: Box::new(move |world, results, ticket| {
                // Despawn after request construction is harmless; never retarget.
                if world.get_entity(entity).is_err() {
                    return Ok(());
                }
                let value = results.get_by_ticket::<T>(ticket)?;
                let mut component = world
                    .get_mut::<C>(entity)
                    .ok_or(BindingError::MissingComponent)?;
                write(&mut component, value);
                Ok(())
            }),
        });
        Ok(OutputBinding {
            registry: self.identity,
            id: self.revision,
        })
    }
    /// Clear invalid bindings without deleting shared graph nodes.
    pub fn prune(&mut self, world: &World, snapshot: &EvaluationSnapshot) -> usize {
        let before = self.inputs.len() + self.outputs.len();
        self.inputs
            .retain(|i| world.get_entity(i.source.entity).is_ok() && (i.check)(snapshot).is_ok());
        self.outputs
            .retain(|o| world.get_entity(o.target.entity).is_ok() && (o.check)(snapshot).is_ok());
        let removed = before - self.inputs.len() - self.outputs.len();
        if removed > 0 {
            self.output_index.clear();
            for (idx, output) in self.outputs.iter().enumerate() {
                self.output_index.insert(output.id, idx);
            }
            self.advance();
        }
        removed
    }
    /// Extracts all bound inputs in one transaction. Failure or panic publishes
    /// no partial input changes. Callers decide when ECS changes warrant a sync.
    pub fn sync_inputs(&self, world: &World, state: &mut GraphState) -> Result<(), BindingError> {
        if self
            .graph
            .is_some_and(|id| id != state.snapshot().graph_id())
        {
            return Err(EvalError::ForeignGraph.into());
        }
        state.try_edit(|edit| {
            for input in &self.inputs {
                (input.sync)(world, edit)?;
            }
            Ok(())
        })
    }

    /// Requests every registered output at the explicitly supplied frames.
    pub fn request(
        &self,
        snapshot: &EvaluationSnapshot,
        frames: &[FrameIndex],
    ) -> Result<BoundRequest, BindingError> {
        self.request_entries(snapshot, frames, (0..self.outputs.len()).collect())
    }

    /// Completeness belongs to this selection, not all outputs in the registry.
    /// Duplicate handles/frames are ignored; equal Frame/Node pairs share one typed
    /// request ticket even when their values are written to several entities.
    pub fn request_selected(
        &self,
        snapshot: &EvaluationSnapshot,
        frames: &[FrameIndex],
        outputs: &[OutputBinding],
    ) -> Result<BoundRequest, BindingError> {
        let mut entries = Vec::new();
        for output in outputs {
            if output.registry != self.identity {
                return Err(BindingError::UnknownOutput);
            }
            let &index = self
                .output_index
                .get(&output.id)
                .ok_or(BindingError::UnknownOutput)?;
            if !entries.contains(&index) {
                entries.push(index);
            }
        }
        self.request_entries(snapshot, frames, entries)
    }
    fn request_entries(
        &self,
        snapshot: &EvaluationSnapshot,
        frames: &[FrameIndex],
        entries: Vec<usize>,
    ) -> Result<BoundRequest, BindingError> {
        if self.graph.is_some_and(|id| id != snapshot.graph_id()) {
            return Err(EvalError::ForeignGraph.into());
        }
        for &entry in &entries {
            (self.outputs[entry].check)(snapshot)?;
        }
        let mut batch = RequestBatch::new();
        let mut tickets: HashMap<(FrameIndex, NodeKey), RequestTicket> = HashMap::default();
        let mut frame_slices: HashMap<FrameIndex, std::ops::Range<usize>> = HashMap::default();
        let mut outputs = Vec::with_capacity(frames.len().saturating_mul(entries.len()));
        for &frame in frames {
            if frame_slices.contains_key(&frame) {
                continue;
            }
            let start = outputs.len();
            for &entry in &entries {
                let output_entry = &self.outputs[entry];
                let ticket = *tickets
                    .entry((frame, output_entry.node))
                    .or_insert_with(|| (output_entry.request)(&mut batch, frame));
                outputs.push(PlannedOutput { entry, ticket });
            }
            frame_slices.insert(frame, start..outputs.len());
        }
        Ok(BoundRequest {
            batch_id: batch.id(),
            batch: Some(batch),
            snapshot: snapshot.clone(),
            registry: self.identity,
            revision: self.revision,
            frame_slices,
            outputs,
        })
    }

    /// Applies an owned completed result against an explicit computation basis.
    /// Pass the current GraphState snapshot to reject obsolete input/definition
    /// results, or intentionally retain the request snapshot for offline consumption.
    /// All selected nodes, typed tickets, components and identities are checked
    /// before writing. Despawned destinations are skipped. Custom writer panics do
    /// not roll back earlier writes. Apply never evaluates or touches caches.
    pub fn apply(
        &self,
        world: &mut World,
        snapshot: &EvaluationSnapshot,
        request: &BoundRequest,
        results: &EvaluationResults,
        frame: FrameIndex,
    ) -> Result<(), BindingError> {
        if request.registry != self.identity {
            return Err(BindingError::ForeignBindings);
        }
        if request.revision != self.revision {
            return Err(BindingError::StaleBindings);
        }
        if snapshot.graph_id() != request.snapshot.graph_id() {
            return Err(EvalError::ForeignGraph.into());
        }
        if snapshot.id() != request.snapshot.id() || results.snapshot_id() != request.snapshot.id()
        {
            return Err(BindingError::SnapshotMismatch);
        }
        if results.batch_id() != request.batch_id {
            return Err(EvalError::ForeignRequest.into());
        }
        let range = request
            .frame_slices
            .get(&frame)
            .cloned()
            .ok_or(BindingError::MissingFrame)?;
        for output in &request.outputs[range.clone()] {
            let entry = &self.outputs[output.entry];
            (entry.check)(snapshot)?;
            (entry.validate_component)(world)?;
            (entry.validate_result)(results, output.ticket)?;
        }
        for output in &request.outputs[range] {
            let entry = &self.outputs[output.entry];
            (entry.apply)(world, results, output.ticket)?;
        }
        Ok(())
    }
}
