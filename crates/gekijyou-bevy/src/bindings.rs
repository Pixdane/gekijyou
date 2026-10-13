use crate::{AnimationRuntime, runtime::PreparedFrame};
use bevy_ecs::{component::Mutable, prelude::*};
use bevy_math::{Quat, Vec3};
use bevy_transform::components::Transform;
use gekijyou_computation::{
    EvalError, EvaluationResults, EvaluationSnapshot, FrameIndex, GraphEdit, GraphError, GraphId,
    Input, Node, NodeId, RequestBatch, SharedValue,
};
use std::{any::TypeId, collections::HashSet, marker::PhantomData, sync::Arc};
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
    #[error("Input/output ranges would implicitly feed playback output back into input")]
    FeedbackConflict,
    #[error("Input source entity no longer exists")]
    MissingEntity,
    #[error("Bound component is missing")]
    MissingComponent,
    #[error("Enable the async driver exactly once before async preparation")]
    AsyncConfiguration,
    #[error("Pause playback before publishing graph/input edits")]
    Playing,
    #[error("The complete current snapshot/generation/binding frame is not prepared")]
    FrameNotReady,
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
type ApplyOutput = dyn Fn(&mut World, &PreparedFrame) -> Result<(), BindingError> + Send + Sync;
type ReadOutput = dyn Fn(&EvaluationResults) -> Result<SharedValue, EvalError> + Send + Sync;
type RequestOutput = dyn Fn(&mut RequestBatch, FrameIndex) -> Box<ReadOutput> + Send + Sync;
struct InputEntry {
    key: NodeKey,
    source: Destination,
    check: Box<Check>,
    sync: Box<SyncInput>,
}
struct OutputEntry {
    key: NodeKey,
    target: Destination,
    check: Box<Check>,
    request: Box<RequestOutput>,
    validate_component: Box<ValidateComponent>,
    apply: Box<ApplyOutput>,
}
pub(crate) struct PlannedOutput {
    pub(crate) frame: FrameIndex,
    pub(crate) key: NodeKey,
    pub(crate) read: Box<ReadOutput>,
}

/// Long-lived Node bindings. RequestIds are generated only for a preparation batch.
/// One input has one authoritative source; outputs check conservative write ranges.
#[derive(Resource, Default)]
pub struct Bindings {
    inputs: Vec<InputEntry>,
    outputs: Vec<OutputEntry>,
    revision: u64,
    graph: Option<GraphId>,
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
    ) -> Result<(), BindingError> {
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
        self.outputs.push(OutputEntry {
            key: NodeKey::of(node),
            target,
            check: Box::new(move |s| s.graph().check(node)),
            request: Box::new(move |batch, frame| {
                let id = batch.at(frame).request(node);
                Box::new(move |r| r.get_shared(id).map(|v| v as SharedValue))
            }),
            validate_component: Box::new(move |world| {
                if world.get_entity(entity).is_err() {
                    return Ok(());
                }
                if world.get::<C>(entity).is_none() {
                    return Err(BindingError::MissingComponent);
                }
                Ok(())
            }),
            apply: Box::new(move |world, frame| {
                // Despawn between registration/preparation and application is harmless.
                if world.get_entity(entity).is_err() {
                    return Ok(());
                }
                let value = frame
                    .values
                    .get(&NodeKey::of(node))
                    .and_then(|v| v.downcast_ref::<T>())
                    .ok_or(BindingError::FrameNotReady)?;
                let mut component = world
                    .get_mut::<C>(entity)
                    .ok_or(BindingError::MissingComponent)?;
                (field.write)(&mut component, value);
                Ok(())
            }),
        });
        Ok(())
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
            self.advance();
        }
        removed
    }
    pub(crate) fn sync(&self, world: &World, edit: &mut GraphEdit) -> Result<(), BindingError> {
        for input in &self.inputs {
            (input.sync)(world, edit)?;
        }
        Ok(())
    }
    pub(crate) fn plan(
        &self,
        snapshot: &EvaluationSnapshot,
        frames: &[FrameIndex],
    ) -> Result<(RequestBatch, Vec<PlannedOutput>), BindingError> {
        for output in &self.outputs {
            (output.check)(snapshot)?;
        }
        let mut batch = RequestBatch::new();
        let mut requests = Vec::new();
        let mut seen = HashSet::new();
        for &frame in frames {
            for output in &self.outputs {
                if seen.insert((frame, output.key)) {
                    requests.push(PlannedOutput {
                        frame,
                        key: output.key,
                        read: (output.request)(&mut batch, frame),
                    });
                }
            }
        }
        Ok((batch, requests))
    }
    /// Checks complete frame identity/requirements before any component is changed.
    /// This method cannot evaluate or query computation caches.
    pub fn apply(
        &self,
        world: &mut World,
        runtime: &AnimationRuntime,
        frame: FrameIndex,
    ) -> Result<(), BindingError> {
        let prepared = runtime.ready_frame(self.revision, frame)?;
        for output in &self.outputs {
            (output.check)(runtime.snapshot())?;
            (output.validate_component)(world)?;
        }
        for output in &self.outputs {
            (output.apply)(world, prepared)?;
        }
        Ok(())
    }
}
