use crate::{BindingError, Bindings, bindings::NodeKey};
use bevy_ecs::prelude::*;
use gekijyou_computation::{
    EvaluationSnapshot, Evaluator, FrameIndex, GraphEdit, GraphError, GraphState, Node,
    SharedValue, SnapshotId,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

/// Playback request identity; unrelated to snapshot identity and cache keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RuntimeGeneration(pub(crate) u64);
impl RuntimeGeneration {
    pub(crate) fn next(self) -> Self {
        Self(self.0.checked_add(1).expect("Runtime generation exhausted"))
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaybackState {
    #[default]
    Paused,
    Playing,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStatus {
    Ready,
    Pending,
    Failed,
    Missing,
}
pub(crate) struct PreparedFrame {
    pub(crate) values: HashMap<NodeKey, SharedValue>,
}
/// Bounded final-output retention, with no computation or equivalence queries.
/// Only complete frames are installed, associated with snapshot and generation.
pub struct PreparedResults {
    snapshot: SnapshotId,
    generation: RuntimeGeneration,
    binding: u64,
    capacity: usize,
    frames: HashMap<FrameIndex, PreparedFrame>,
    order: VecDeque<FrameIndex>,
    states: HashMap<FrameIndex, FrameStatus>,
}
impl PreparedResults {
    fn new(snapshot: SnapshotId, generation: RuntimeGeneration, capacity: usize) -> Self {
        assert!(capacity > 0, "Prepared frame capacity must be positive");
        Self {
            snapshot,
            generation,
            binding: 0,
            capacity,
            frames: HashMap::new(),
            order: VecDeque::new(),
            states: HashMap::new(),
        }
    }
    fn reset(&mut self, snapshot: SnapshotId, generation: RuntimeGeneration, binding: u64) {
        self.snapshot = snapshot;
        self.generation = generation;
        self.binding = binding;
        self.frames.clear();
        self.order.clear();
        self.states.clear();
    }
    pub fn status(&self, frame: FrameIndex) -> FrameStatus {
        if self.frames.contains_key(&frame) {
            FrameStatus::Ready
        } else {
            self.states
                .get(&frame)
                .copied()
                .unwrap_or(FrameStatus::Missing)
        }
    }
    pub fn snapshot_id(&self) -> SnapshotId {
        self.snapshot
    }
    pub fn generation(&self) -> RuntimeGeneration {
        self.generation
    }
    /// Explicit expected identities prevent interpreting old buffered outputs as current.
    pub fn get<T: 'static>(
        &self,
        snapshot: SnapshotId,
        generation: RuntimeGeneration,
        binding_revision: u64,
        frame: FrameIndex,
        node: Node<T>,
    ) -> Option<&T> {
        if snapshot != self.snapshot
            || generation != self.generation
            || binding_revision != self.binding
        {
            return None;
        }
        self.frames
            .get(&frame)?
            .values
            .get(&NodeKey::of(node))?
            .downcast_ref()
    }
    fn install(&mut self, frame: FrameIndex, values: HashMap<NodeKey, SharedValue>) {
        if let Some(i) = self.order.iter().position(|&f| f == frame) {
            self.order.remove(i);
        }
        if self.order.len() >= self.capacity
            && let Some(old) = self.order.pop_front()
        {
            self.frames.remove(&old);
            self.states.remove(&old);
        }
        self.frames.insert(frame, PreparedFrame { values });
        self.states.remove(&frame);
        self.order.push_back(frame);
    }
    fn state(&mut self, frame: FrameIndex, status: FrameStatus) {
        // Pending/failed metadata is independently bounded, like completed frames.
        if !self.states.contains_key(&frame)
            && self.states.len() >= self.capacity
            && let Some(old) = self.states.keys().next().copied()
        {
            self.states.remove(&old);
        }
        self.frames.remove(&frame);
        self.order.retain(|&f| f != frame);
        self.states.insert(frame, status);
    }
}

/// Minimal animation facade. Graph/input edits require Paused, while preparation
/// works in either state. Old snapshots remain valid independently of this runtime.
#[derive(Resource)]
pub struct AnimationRuntime {
    state: GraphState,
    evaluator: Evaluator,
    snapshot: EvaluationSnapshot,
    generation: RuntimeGeneration,
    playback: PlaybackState,
    frame: FrameIndex,
    prepared: PreparedResults,
    pub(crate) last_error: Option<BindingError>,
}
impl AnimationRuntime {
    pub fn new(buffer_frames: usize) -> Self {
        let state = GraphState::new();
        let evaluator = Evaluator::new(&state);
        let snapshot = state.snapshot();
        let generation = RuntimeGeneration::default();
        let prepared = PreparedResults::new(snapshot.id(), generation, buffer_frames);
        Self {
            state,
            evaluator,
            snapshot,
            generation,
            playback: PlaybackState::Paused,
            frame: FrameIndex(0),
            prepared,
            last_error: None,
        }
    }
    pub fn snapshot(&self) -> &EvaluationSnapshot {
        &self.snapshot
    }
    pub fn generation(&self) -> RuntimeGeneration {
        self.generation
    }
    pub fn playback(&self) -> PlaybackState {
        self.playback
    }
    pub fn frame(&self) -> FrameIndex {
        self.frame
    }
    pub fn prepared(&self) -> &PreparedResults {
        &self.prepared
    }
    pub fn last_error(&self) -> Option<BindingError> {
        self.last_error
    }
    pub fn configure_cache<T: 'static>(
        &mut self,
        node: Node<T>,
        config: gekijyou_computation::NodeCacheConfig,
    ) -> Result<(), BindingError> {
        self.evaluator.bind(self.snapshot.clone())?;
        self.evaluator.configure_cache(node, config)?;
        Ok(())
    }
    pub fn stats(&self) -> gekijyou_computation::EvalStats {
        self.evaluator.stats()
    }
    fn invalidate(&mut self) {
        self.generation = self.generation.next();
        self.prepared.reset(self.snapshot.id(), self.generation, 0);
    }
    pub fn play(&mut self) {
        self.invalidate();
        self.playback = PlaybackState::Playing;
    }
    pub fn pause(&mut self) {
        self.playback = PlaybackState::Paused;
        self.invalidate();
    }
    pub fn seek(&mut self, frame: FrameIndex) {
        self.frame = frame;
        self.invalidate();
    }
    pub fn edit<R>(
        &mut self,
        f: impl FnOnce(&mut GraphEdit) -> Result<R, GraphError>,
    ) -> Result<R, BindingError> {
        if self.playback != PlaybackState::Paused {
            return Err(BindingError::Playing);
        }
        let result = self.state.edit(f)?;
        let snapshot = self.state.snapshot();
        if snapshot.id() != self.snapshot.id() {
            self.snapshot = snapshot;
            self.invalidate();
        }
        Ok(result)
    }
    pub fn sync_inputs(&mut self, world: &World, bindings: &Bindings) -> Result<(), BindingError> {
        if self.playback != PlaybackState::Paused {
            return Err(BindingError::Playing);
        }
        self.state.try_edit(|edit| bindings.sync(world, edit))?;
        let snapshot = self.state.snapshot();
        if snapshot.id() != self.snapshot.id() {
            self.snapshot = snapshot;
            self.invalidate();
        }
        Ok(())
    }
    pub fn prepare(
        &mut self,
        bindings: &Bindings,
        frames: &[FrameIndex],
    ) -> Result<(), BindingError> {
        if self.prepared.binding != bindings.revision() {
            self.prepared
                .reset(self.snapshot.id(), self.generation, bindings.revision());
        }
        for &frame in frames {
            self.prepared.state(frame, FrameStatus::Pending);
        }
        // Evaluation unwinding cannot publish a partial frame; pending metadata is
        // changed to failure even when the caller catches the original panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (batch, requests) = bindings.plan(&self.snapshot, frames)?;
            let results = self
                .evaluator
                .bind(self.snapshot.clone())?
                .evaluate(batch)?;
            let mut complete: HashMap<FrameIndex, HashMap<NodeKey, SharedValue>> =
                frames.iter().map(|&f| (f, HashMap::new())).collect();
            for request in requests {
                complete
                    .get_mut(&request.frame)
                    .unwrap()
                    .insert(request.key, (request.read)(&results)?);
            }
            Ok::<_, BindingError>(complete)
        }));
        match result {
            Ok(Ok(mut complete)) => {
                for &frame in frames {
                    if let Some(values) = complete.remove(&frame) {
                        self.prepared.install(frame, values);
                    }
                }
                self.last_error = None;
                Ok(())
            }
            Ok(Err(error)) => {
                for &frame in frames {
                    self.prepared.state(frame, FrameStatus::Failed);
                }
                self.last_error = Some(error);
                Err(error)
            }
            Err(panic) => {
                for &frame in frames {
                    self.prepared.state(frame, FrameStatus::Failed);
                }
                std::panic::resume_unwind(panic)
            }
        }
    }
    pub(crate) fn ready_frame(
        &self,
        binding: u64,
        frame: FrameIndex,
    ) -> Result<&PreparedFrame, BindingError> {
        if self.prepared.snapshot != self.snapshot.id()
            || self.prepared.generation != self.generation
            || self.prepared.binding != binding
        {
            return Err(BindingError::FrameNotReady);
        }
        self.prepared
            .frames
            .get(&frame)
            .ok_or(BindingError::FrameNotReady)
    }
    pub fn sample_shared<T: Send + Sync + 'static>(
        &self,
        bindings: &Bindings,
        frame: FrameIndex,
        node: Node<T>,
    ) -> Option<Arc<T>> {
        self.ready_frame(bindings.revision(), frame).ok()?;
        self.prepared
            .frames
            .get(&frame)?
            .values
            .get(&NodeKey::of(node))?
            .clone()
            .downcast()
            .ok()
    }
}
impl Default for AnimationRuntime {
    fn default() -> Self {
        Self::new(8)
    }
}
