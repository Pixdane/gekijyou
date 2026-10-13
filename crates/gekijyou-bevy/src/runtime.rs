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
        let old_frames = std::mem::take(&mut self.frames);
        self.order.clear();
        self.states.clear();
        drop(old_frames);
    }
    /// Status of this stored result set. Use AnimationRuntime::frame_status to
    /// include current binding requirements when deciding whether to apply.
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
    asynchronous: Option<AsyncPreparation>,
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
            asynchronous: None,
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
    /// Current snapshot/generation/binding readiness, without performing evaluation.
    pub fn frame_status(&self, bindings: &Bindings, frame: FrameIndex) -> FrameStatus {
        if self.prepared.snapshot != self.snapshot.id()
            || self.prepared.generation != self.generation
            || self.prepared.binding != bindings.revision()
        {
            return FrameStatus::Missing;
        }
        self.prepared.status(frame)
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
        if let Some(asynchronous) = &mut self.asynchronous {
            asynchronous
                .driver
                .cancel(self.snapshot.clone(), self.generation)
                .expect("Runtime definitions only advance");
            asynchronous.requests.clear();
            asynchronous.frames.clear();
        }
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
        self.prepare_backend(bindings, frames, None).map(|_| ())
    }
    /// Optional planned synchronous preparation, using either serial or Bevy executor.
    pub fn prepare_planned(
        &mut self,
        bindings: &Bindings,
        frames: &[FrameIndex],
        config: gekijyou_computation::PlannerConfig,
        executor: &mut impl gekijyou_computation::Executor,
    ) -> Result<gekijyou_computation::PlannerStats, BindingError> {
        self.prepare_backend(bindings, frames, Some((config, executor)))
    }
    fn prepare_backend(
        &mut self,
        bindings: &Bindings,
        frames: &[FrameIndex],
        backend: Option<(
            gekijyou_computation::PlannerConfig,
            &mut dyn gekijyou_computation::Executor,
        )>,
    ) -> Result<gekijyou_computation::PlannerStats, BindingError> {
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
            let (results, stats) = match backend {
                Some((config, executor)) => self
                    .evaluator
                    .bind(self.snapshot.clone())?
                    .evaluate_planned(batch, config, executor)?,
                None => (
                    self.evaluator
                        .bind(self.snapshot.clone())?
                        .evaluate(batch)?,
                    gekijyou_computation::PlannerStats::default(),
                ),
            };
            let mut complete: HashMap<FrameIndex, HashMap<NodeKey, SharedValue>> =
                frames.iter().map(|&f| (f, HashMap::new())).collect();
            for request in requests {
                complete
                    .get_mut(&request.frame)
                    .unwrap()
                    .insert(request.key, (request.read)(&results)?);
            }
            Ok::<_, BindingError>((complete, stats))
        }));
        match result {
            Ok(Ok((mut complete, stats))) => {
                for &frame in frames {
                    if let Some(values) = complete.remove(&frame) {
                        self.prepared.install(frame, values);
                    }
                }
                self.last_error = None;
                Ok(stats)
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

struct AsyncPreparation {
    driver: crate::AsyncDriver,
    requests: Vec<crate::bindings::PlannedOutput>,
    frames: Vec<FrameIndex>,
    binding: u64,
}
impl AnimationRuntime {
    /// Installs one persistent cross-generation quota tracker. Reinstalling would
    /// forget canceled running workers, so it is rejected rather than resetting it.
    pub fn enable_async(&mut self, limits: crate::TaskLimits) -> Result<(), BindingError> {
        if self.asynchronous.is_some() || limits.max_in_flight == 0 {
            return Err(BindingError::AsyncConfiguration);
        }
        self.asynchronous = Some(AsyncPreparation {
            driver: crate::AsyncDriver::new(self.snapshot.clone(), self.generation, limits),
            requests: Vec::new(),
            frames: Vec::new(),
            binding: 0,
        });
        Ok(())
    }
    /// Replaces the current playback request without waiting for old workers.
    /// All cache queries and dependency expansion remain on the polling coordinator.
    pub fn begin_prepare(
        &mut self,
        bindings: &Bindings,
        frames: &[FrameIndex],
        config: gekijyou_computation::PlannerConfig,
    ) -> Result<(), BindingError> {
        if self.asynchronous.is_none() {
            return Err(BindingError::AsyncConfiguration);
        }
        self.invalidate();
        self.prepared
            .reset(self.snapshot.id(), self.generation, bindings.revision());
        let (batch, requests) = bindings.plan(&self.snapshot, frames)?;
        let asynchronous = self.asynchronous.as_mut().unwrap();
        asynchronous.driver.begin(
            &mut self.evaluator,
            self.snapshot.clone(),
            self.generation,
            batch,
            config,
        )?;
        asynchronous.requests = requests;
        asynchronous.frames = frames.to_vec();
        asynchronous.binding = bindings.revision();
        for &frame in frames {
            self.prepared.state(frame, FrameStatus::Pending);
        }
        self.last_error = None;
        Ok(())
    }
    /// Nonblocking progress. Obsolete worker messages are reclaimed even when no
    /// current frame is awaiting preparation. Ready means the required set is complete.
    pub fn poll_prepare(
        &mut self,
        pool: &bevy_tasks::TaskPool,
        bindings: &Bindings,
    ) -> Result<bool, BindingError> {
        let Some(asynchronous) = self.asynchronous.as_mut() else {
            return Err(BindingError::AsyncConfiguration);
        };
        if asynchronous.binding != bindings.revision() && !asynchronous.frames.is_empty() {
            self.invalidate();
            return Err(BindingError::FrameNotReady);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            asynchronous.driver.poll(&mut self.evaluator, pool)
        }));
        match result {
            Ok(Ok(None)) => Ok(false),
            Ok(Ok(Some((results, _stats)))) => {
                if results.snapshot_id() != self.snapshot.id()
                    || asynchronous.driver.generation() != self.generation
                {
                    return Ok(false);
                }
                let mut complete: HashMap<FrameIndex, HashMap<NodeKey, SharedValue>> = asynchronous
                    .frames
                    .iter()
                    .map(|&f| (f, HashMap::new()))
                    .collect();
                for request in &asynchronous.requests {
                    complete
                        .get_mut(&request.frame)
                        .unwrap()
                        .insert(request.key, (request.read)(&results)?);
                }
                for frame in std::mem::take(&mut asynchronous.frames) {
                    if let Some(values) = complete.remove(&frame) {
                        self.prepared.install(frame, values);
                    }
                }
                asynchronous.requests.clear();
                self.last_error = None;
                Ok(true)
            }
            Ok(Err(error)) => {
                for frame in std::mem::take(&mut asynchronous.frames) {
                    self.prepared.state(frame, FrameStatus::Failed);
                }
                asynchronous.requests.clear();
                let error = BindingError::Evaluation(error);
                self.last_error = Some(error);
                Err(error)
            }
            Err(panic) => {
                for frame in std::mem::take(&mut asynchronous.frames) {
                    self.prepared.state(frame, FrameStatus::Failed);
                }
                asynchronous.requests.clear();
                std::panic::resume_unwind(panic)
            }
        }
    }
    pub fn async_task_stats(&mut self) -> Option<crate::TaskStats> {
        self.asynchronous.as_mut().map(|a| a.driver.stats())
    }
    pub fn async_in_flight(&mut self) -> usize {
        self.asynchronous
            .as_mut()
            .map(|a| a.driver.in_flight())
            .unwrap_or(0)
    }
}
