use super::{
    EvalError, EvaluationResults, EvaluationScope, EvaluationSnapshot, Evaluator, FrameIndex,
    NodeId, RequestBatch, SharedValue,
    node::{GraphDefinitionEpoch, NodeKind, next_id},
    value::ErasedComputation,
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, atomic::AtomicU64},
};

/// Active-window limits are independent of the size of the requested output batch.
/// Capacity errors explicitly fail a plan rather than dropping strict sharing entries.
#[derive(Clone, Copy, Debug)]
pub struct PlannerConfig {
    pub frame_window: usize,
    pub max_resolution_tasks: usize,
    pub max_registry_entries: usize,
    pub max_ready_tasks: usize,
    pub max_in_flight: usize,
    /// Maximum content-equivalence comparisons per compute-registry lookup.
    pub comparison_budget: usize,
}
impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            frame_window: 16,
            max_resolution_tasks: 65536,
            max_registry_entries: 65536,
            max_ready_tasks: 65536,
            max_in_flight: 256,
            comparison_budget: 64,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlannerStats {
    pub windows: u64,
    pub resolution_tasks: u64,
    pub compute_tasks: u64,
    pub identity_hits: u64,
    pub content_hits: u64,
    pub equivalence_comparisons: u64,
    pub peak_resolution_tasks: usize,
    pub peak_registry_entries: usize,
    pub peak_ready_tasks: usize,
    pub peak_in_flight: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeTaskId(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PlanId(u64);
impl PlanId {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(next_id(&NEXT))
    }
}

/// Immutable, ready pure-function work. Workers cannot access graph editing,
/// evaluator caches, planner mutation, or ECS. Snapshot retention is explicit.
/// Consuming a task produces exactly one completion, including worker panic.
pub struct ComputeTask {
    id: ComputeTaskId,
    plan: PlanId,
    node: NodeId,
    snapshot: EvaluationSnapshot,
    computation: Arc<ErasedComputation>,
    inputs: Box<[SharedValue]>,
}
impl ComputeTask {
    pub fn id(&self) -> ComputeTaskId {
        self.id
    }
    pub fn node(&self) -> NodeId {
        self.node
    }
    pub fn snapshot_id(&self) -> super::SnapshotId {
        self.snapshot.id()
    }
    pub fn definition_epoch(&self) -> GraphDefinitionEpoch {
        self.snapshot.definition_epoch()
    }
    /// Worker panic is a terminal failure notification, never a cached graph value.
    pub fn execute(self) -> ComputeCompletion {
        let id = self.id;
        let plan = self.plan;
        let result = catch_unwind(AssertUnwindSafe(move || {
            let args: smallvec::SmallVec<[&dyn std::any::Any; 4]> = self
                .inputs
                .iter()
                .map(|v| v.as_ref() as &dyn std::any::Any)
                .collect();
            let output = (self.computation)(&args);
            if output.as_ref().type_id() != self.snapshot.graph.node_record(self.node)?.type_id {
                return Err(super::GraphError::TypeMismatch { node: self.node }.into());
            }
            drop(args);
            drop(self);
            Ok(output)
        }))
        .unwrap_or(Err(EvalError::WorkerPanicked));
        ComputeCompletion { id, plan, result }
    }
}
/// Opaque completion identities cannot be forged into another plan's cache commit.
pub struct ComputeCompletion {
    id: ComputeTaskId,
    plan: PlanId,
    result: Result<SharedValue, EvalError>,
}
impl ComputeCompletion {
    pub fn id(&self) -> ComputeTaskId {
        self.id
    }
}
/// Executors run only already-ready work and return one terminal message per task.
pub trait Executor {
    fn execute(&mut self, tasks: Vec<ComputeTask>) -> Vec<ComputeCompletion>;
}
pub struct SerialExecutor;
impl Executor for SerialExecutor {
    fn execute(&mut self, tasks: Vec<ComputeTask>) -> Vec<ComputeCompletion> {
        tasks.into_iter().map(ComputeTask::execute).collect()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ResolutionKey {
    node: NodeId,
    frame: Option<FrameIndex>,
}
impl ResolutionKey {
    fn frame(self) -> FrameIndex {
        self.frame.unwrap_or(FrameIndex(0))
    }
}
struct Resolution {
    dependencies: Vec<ResolutionKey>,
    pending: usize,
    waiting: Option<ComputeTaskId>,
    value: Option<SharedValue>,
}
struct ComputeEntry {
    node: NodeId,
    inputs: Box<[SharedValue]>,
    waiters: Vec<ResolutionKey>,
    running: bool,
    output: Option<SharedValue>,
}
#[derive(PartialEq, Eq, Hash)]
struct IdentityKey {
    node: NodeId,
    pointers: Vec<usize>,
}
impl IdentityKey {
    fn new(node: NodeId, inputs: &[SharedValue]) -> Self {
        Self {
            node,
            pointers: inputs
                .iter()
                .map(|v| Arc::as_ptr(v) as *const () as usize)
                .collect(),
        }
    }
}
#[derive(Default)]
struct Window {
    resolutions: HashMap<ResolutionKey, Resolution>,
    children: HashMap<ResolutionKey, Vec<ResolutionKey>>,
    ready: VecDeque<ResolutionKey>,
    computes: HashMap<ComputeTaskId, ComputeEntry>,
    by_node: HashMap<NodeId, Vec<ComputeTaskId>>,
    identities: HashMap<IdentityKey, ComputeTaskId>,
    frames: HashMap<FrameIndex, SharedValue>,
    targets: Vec<(usize, ResolutionKey)>,
    running: usize,
}

/// Coordinator-owned planning session. All temporary registries are limited to an
/// active frame window; requested final outputs survive window teardown. Off caches
/// may recompute static nodes in subsequent windows. Async callers advance only on
/// the coordinator, dispatch returned tasks, then pass completions back to advance.
pub struct EvaluationPlan {
    id: PlanId,
    snapshot: EvaluationSnapshot,
    batch: Option<RequestBatch>,
    config: PlannerConfig,
    frames: Vec<FrameIndex>,
    next_frame: usize,
    next_task: u64,
    window: Option<Window>,
    outputs: Vec<Option<SharedValue>>,
    stats: PlannerStats,
    error: Option<EvalError>,
    complete: bool,
}
impl EvaluationPlan {
    pub fn snapshot(&self) -> &EvaluationSnapshot {
        &self.snapshot
    }
    pub fn stats(&self) -> PlannerStats {
        self.stats
    }
    pub fn is_complete(&self) -> bool {
        self.complete
    }
    pub fn failure(&self) -> Option<EvalError> {
        self.error
    }
    pub fn cancel(&mut self) {
        self.error = Some(EvalError::PlanCanceled);
        let window = self.window.take();
        let outputs = std::mem::take(&mut self.outputs);
        drop((window, outputs));
    }
    pub fn finish(mut self) -> Result<(EvaluationResults, PlannerStats), EvalError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.complete {
            return Err(EvalError::PlanIncomplete);
        }
        let outputs = self
            .outputs
            .into_iter()
            .map(|v| v.expect("Complete requested output"))
            .collect();
        Ok((
            self.batch
                .take()
                .unwrap()
                .finish(self.snapshot.id(), outputs),
            self.stats,
        ))
    }
    fn key(&self, node: NodeId, frame: FrameIndex) -> Result<ResolutionKey, EvalError> {
        Ok(ResolutionKey {
            node,
            frame: self
                .snapshot
                .graph
                .node_record(node)?
                .depends_on_frame
                .then_some(frame),
        })
    }
    fn start_window(&mut self, evaluator: &mut Evaluator) -> Result<Window, EvalError> {
        let end = (self.next_frame + self.config.frame_window).min(self.frames.len());
        let active: HashSet<_> = self.frames[self.next_frame..end].iter().copied().collect();
        self.next_frame = end;
        let mut window = Window::default();
        for (index, r) in self.batch.as_ref().unwrap().requests.iter().enumerate() {
            if active.contains(&r.frame) {
                window.targets.push((index, self.key(r.node, r.frame)?));
            }
        }
        let mut stack: Vec<_> = window.targets.iter().map(|(_, k)| *k).rev().collect();
        while let Some(key) = stack.pop() {
            if window.resolutions.contains_key(&key) {
                continue;
            }
            if window.resolutions.len() >= self.config.max_resolution_tasks {
                return Err(EvalError::PlannerBudgetExceeded {
                    limit: self.config.max_resolution_tasks,
                });
            }
            let frame = window
                .frames
                .entry(key.frame())
                .or_insert_with(|| Arc::new(key.frame()))
                .clone();
            let record = self.snapshot.graph.node_record(key.node)?;
            let cached = evaluator.context_lookup(&self.snapshot, &frame, key.node)?;
            let (value, dependencies) = if let Some(output) = cached {
                (Some(output), Vec::new())
            } else {
                match &record.kind {
                    NodeKind::Frame => (Some(frame), Vec::new()),
                    NodeKind::Input => (
                        Some(self.snapshot.inputs.get(&key.node).cloned().ok_or(
                            EvalError::MissingInput {
                                frame: key.frame(),
                                node: key.node,
                            },
                        )?),
                        Vec::new(),
                    ),
                    NodeKind::Computation { inputs, .. } => {
                        let dependencies = inputs
                            .iter()
                            .map(|i| self.key(i.source, key.frame()))
                            .collect::<Result<Vec<_>, _>>()?;
                        stack.extend(dependencies.iter().rev().copied());
                        (None, dependencies)
                    }
                }
            };
            window.resolutions.insert(
                key,
                Resolution {
                    dependencies,
                    pending: 0,
                    waiting: None,
                    value,
                },
            );
        }
        let keys: Vec<_> = window.resolutions.keys().copied().collect();
        for key in keys {
            if window.resolutions[&key].value.is_some() {
                continue;
            }
            let dependencies = window.resolutions[&key].dependencies.clone();
            let mut count = 0;
            for dep in dependencies {
                if window.resolutions[&dep].value.is_none() {
                    count += 1;
                    window.children.entry(dep).or_default().push(key);
                }
            }
            window.resolutions.get_mut(&key).unwrap().pending = count;
            if count == 0 {
                window.ready.push_back(key);
            }
        }
        // HashMap iteration never selects computation order; normalize ready nodes.
        let mut ready: Vec<_> = window.ready.drain(..).collect();
        ready.sort_by_key(|k| (k.node, k.frame.map(|f| f.0)));
        window.ready.extend(ready);
        if window.ready.len() > self.config.max_ready_tasks {
            return Err(EvalError::PlannerBudgetExceeded {
                limit: self.config.max_ready_tasks,
            });
        }
        self.stats.windows += 1;
        self.stats.resolution_tasks += window.resolutions.len() as u64;
        self.stats.peak_resolution_tasks = self
            .stats
            .peak_resolution_tasks
            .max(window.resolutions.len());
        self.stats.peak_ready_tasks = self.stats.peak_ready_tasks.max(window.ready.len());
        Ok(window)
    }
    fn deliver(
        &mut self,
        evaluator: &mut Evaluator,
        window: &mut Window,
        key: ResolutionKey,
        value: SharedValue,
    ) -> Result<(), EvalError> {
        let frame = window.frames.get(&key.frame()).unwrap();
        evaluator.context_insert(&self.snapshot, frame, key.node, value.clone());
        let resolution = window.resolutions.get_mut(&key).unwrap();
        resolution.value = Some(value);
        resolution.waiting = None;
        for &child in window.children.get(&key).into_iter().flatten() {
            let record = window.resolutions.get_mut(&child).unwrap();
            record.pending -= 1;
            if record.pending == 0 {
                window.ready.push_back(child);
            }
        }
        if window.ready.len() > self.config.max_ready_tasks {
            return Err(EvalError::PlannerBudgetExceeded {
                limit: self.config.max_ready_tasks,
            });
        }
        self.stats.peak_ready_tasks = self.stats.peak_ready_tasks.max(window.ready.len());
        Ok(())
    }
    fn find_compute(
        &mut self,
        window: &Window,
        key: ResolutionKey,
        inputs: &[SharedValue],
    ) -> Result<Option<ComputeTaskId>, EvalError> {
        if let Some(&id) = window.identities.get(&IdentityKey::new(key.node, inputs)) {
            self.stats.identity_hits += 1;
            return Ok(Some(id));
        }
        let mut budget = self.config.comparison_budget;
        let dependencies = &window.resolutions[&key].dependencies;
        for &id in window.by_node.get(&key.node).into_iter().flatten().rev() {
            let candidate = &window.computes[&id];
            let mut matched = true;
            for (position, (current, old)) in inputs.iter().zip(candidate.inputs.iter()).enumerate()
            {
                if budget == 0 {
                    return Ok(None);
                }
                budget -= 1;
                self.stats.equivalence_comparisons += 1;
                if !self
                    .snapshot
                    .graph
                    .node_record(dependencies[position].node)?
                    .semantics
                    .equivalent(current, old)
                {
                    matched = false;
                    break;
                }
            }
            if matched {
                self.stats.content_hits += 1;
                return Ok(Some(id));
            }
        }
        Ok(None)
    }
    fn advance_inner(
        &mut self,
        evaluator: &mut Evaluator,
        completed: Vec<ComputeCompletion>,
        dispatch_slots: usize,
    ) -> Result<Vec<ComputeTask>, EvalError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.snapshot.graph_id() != evaluator.graph.id() {
            return Err(EvalError::ForeignGraph);
        }
        if self.snapshot.definition_epoch() != evaluator.graph.definition_epoch() {
            return Err(EvalError::StaleGraphDefinition {
                requested: self.snapshot.definition_epoch(),
                active: evaluator.graph.definition_epoch(),
            });
        }
        if self.complete {
            return Ok(Vec::new());
        }
        let mut window = match self.window.take() {
            Some(w) => w,
            None => self.start_window(evaluator)?,
        };
        for completion in completed {
            if completion.plan != self.id {
                return Err(EvalError::ForeignPlanCompletion);
            }
            let Some(entry) = window.computes.get_mut(&completion.id) else {
                return Err(EvalError::ForeignPlanCompletion);
            };
            if !entry.running {
                continue;
            }
            evaluator.record_planned_call();
            entry.running = false;
            window.running -= 1;
            let output = completion.result?;
            let node = entry.node;
            let inputs = entry.inputs.clone();
            let waiters = std::mem::take(&mut entry.waiters);
            entry.output = Some(output.clone());
            // Active-definition checks precede every persistent commit. Policies
            // are read now, so disabling caches while a worker runs stays effective.
            evaluator.memo_insert(
                node,
                self.snapshot.graph.node_record(node)?,
                &inputs,
                output.clone(),
            );
            for key in waiters {
                self.deliver(evaluator, &mut window, key, output.clone())?;
            }
        }
        loop {
            let mut tasks = Vec::new();
            while window.running < self.config.max_in_flight && tasks.len() < dispatch_slots {
                let Some(key) = window.ready.pop_front() else {
                    break;
                };
                let record = self.snapshot.graph.nodes[&key.node].clone();
                let inputs: Box<[SharedValue]> = window.resolutions[&key]
                    .dependencies
                    .iter()
                    .map(|k| window.resolutions[k].value.as_ref().unwrap().clone())
                    .collect();
                if let Some(output) =
                    evaluator.memo_lookup(&self.snapshot.graph, key.node, &record, &inputs)?
                {
                    self.deliver(evaluator, &mut window, key, output)?;
                    continue;
                }
                if let Some(id) = self.find_compute(&window, key, &inputs)? {
                    window
                        .identities
                        .insert(IdentityKey::new(key.node, &inputs), id);
                    if let Some(output) = window.computes[&id].output.clone() {
                        self.deliver(evaluator, &mut window, key, output)?;
                    } else {
                        window.resolutions.get_mut(&key).unwrap().waiting = Some(id);
                        window.computes.get_mut(&id).unwrap().waiters.push(key);
                    }
                    continue;
                }
                if window.computes.len() >= self.config.max_registry_entries {
                    return Err(EvalError::PlannerBudgetExceeded {
                        limit: self.config.max_registry_entries,
                    });
                }
                let id = ComputeTaskId(self.next_task);
                self.next_task = self
                    .next_task
                    .checked_add(1)
                    .expect("Compute task identity exhausted");
                let computation = match &record.kind {
                    NodeKind::Computation { computation, .. } => computation.clone(),
                    _ => unreachable!(),
                };
                tasks.push(ComputeTask {
                    id,
                    plan: self.id,
                    node: key.node,
                    snapshot: self.snapshot.clone(),
                    computation,
                    inputs: inputs.clone(),
                });
                window
                    .identities
                    .insert(IdentityKey::new(key.node, &inputs), id);
                window.by_node.entry(key.node).or_default().push(id);
                window.computes.insert(
                    id,
                    ComputeEntry {
                        node: key.node,
                        inputs,
                        waiters: vec![key],
                        running: true,
                        output: None,
                    },
                );
                window.resolutions.get_mut(&key).unwrap().waiting = Some(id);
                window.running += 1;
                self.stats.compute_tasks += 1;
                self.stats.peak_registry_entries =
                    self.stats.peak_registry_entries.max(window.computes.len());
                self.stats.peak_in_flight = self.stats.peak_in_flight.max(window.running);
            }
            if !tasks.is_empty() || window.running > 0 || !window.ready.is_empty() {
                self.window = Some(window);
                return Ok(tasks);
            }
            for &(index, key) in &window.targets {
                self.outputs[index] = Some(
                    window.resolutions[&key]
                        .value
                        .clone()
                        .ok_or(EvalError::PlanIncomplete)?,
                );
            }
            if self.next_frame == self.frames.len() {
                self.complete = true;
                return Ok(Vec::new());
            }
            // Releasing this window releases every identity key's backing Arc.
            // No identity index or intermediate value survives into the next one.
            window = self.start_window(evaluator)?;
        }
    }
}
impl Evaluator {
    pub fn plan(
        &mut self,
        snapshot: EvaluationSnapshot,
        batch: RequestBatch,
        config: PlannerConfig,
    ) -> Result<EvaluationPlan, EvalError> {
        if config.frame_window == 0
            || config.max_resolution_tasks == 0
            || config.max_registry_entries == 0
            || config.max_ready_tasks == 0
            || config.max_in_flight == 0
        {
            return Err(EvalError::InvalidPlannerConfig);
        }
        self.bind(snapshot.clone())?;
        for r in &batch.requests {
            snapshot.graph.check_node(r.graph, r.node, r.ty)?;
        }
        let mut seen = HashSet::new();
        let frames = batch
            .requests
            .iter()
            .filter_map(|r| seen.insert(r.frame).then_some(r.frame))
            .collect();
        let count = batch.len();
        Ok(EvaluationPlan {
            id: PlanId::new(),
            snapshot,
            batch: Some(batch),
            config,
            frames,
            next_frame: 0,
            next_task: 0,
            window: None,
            outputs: vec![None; count],
            stats: PlannerStats::default(),
            error: None,
            complete: count == 0,
        })
    }
    /// Read/commit coordination always remains here. Errors and unwinding make the
    /// plan terminal and release registry Arcs; running workers own separate data.
    pub fn advance(
        &mut self,
        plan: &mut EvaluationPlan,
        completed: Vec<ComputeCompletion>,
    ) -> Result<Vec<ComputeTask>, EvalError> {
        self.advance_with_budget(plan, completed, usize::MAX)
    }
    /// Async adapters limit newly issued work to their available global quota.
    /// Zero permits completion processing but does not dispatch new computations.
    pub fn advance_with_budget(
        &mut self,
        plan: &mut EvaluationPlan,
        completed: Vec<ComputeCompletion>,
        dispatch_slots: usize,
    ) -> Result<Vec<ComputeTask>, EvalError> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            plan.advance_inner(self, completed, dispatch_slots)
        }));
        match result {
            Ok(Ok(tasks)) => Ok(tasks),
            Ok(Err(error)) => {
                plan.cancel();
                plan.error = Some(error);
                Err(error)
            }
            Err(panic) => {
                plan.cancel();
                std::panic::resume_unwind(panic)
            }
        }
    }
}
impl EvaluationScope<'_> {
    pub fn evaluate_planned(
        &mut self,
        batch: RequestBatch,
        config: PlannerConfig,
        executor: &mut (impl Executor + ?Sized),
    ) -> Result<(EvaluationResults, PlannerStats), EvalError> {
        let mut plan = self.evaluator.plan(self.snapshot.clone(), batch, config)?;
        let mut completed = Vec::new();
        while !plan.is_complete() {
            let tasks = self.evaluator.advance(&mut plan, completed)?;
            if tasks.is_empty() && !plan.is_complete() {
                plan.cancel();
                return Err(EvalError::PlanIncomplete);
            }
            completed = executor.execute(tasks);
        }
        plan.finish()
    }
}
