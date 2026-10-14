use crate::{RequestGeneration, TaskCoordinator, TaskLimits, TaskStats};
use bevy_tasks::TaskPool;
use gekijyou_computation::{
    ComputeCompletion, ComputeTask, EvalError, EvaluationPlan, EvaluationResults,
    EvaluationSnapshot, Evaluator, Executor, PlannerConfig, PlannerStats, RequestBatch,
};
use std::sync::Mutex;

/// Synchronous parallel batches on Bevy's TaskPool. The coordinator performs all
/// cache work before/after scope execution. Tasks are chunked without reordering
/// arithmetic inside closures. Every task catches its own panic, so one failing
/// task cannot leave other tasks in its chunk without terminal notifications.
pub struct BevyExecutor<'a> {
    pool: &'a TaskPool,
    chunk_size: usize,
}
impl<'a> BevyExecutor<'a> {
    pub fn new(pool: &'a TaskPool, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be positive");
        Self { pool, chunk_size }
    }
}
impl Executor for BevyExecutor<'_> {
    fn execute(&mut self, tasks: Vec<ComputeTask>) -> Vec<ComputeCompletion> {
        let mut tasks = tasks.into_iter();
        let mut chunks = Vec::new();
        loop {
            let chunk: Vec<_> = tasks.by_ref().take(self.chunk_size).collect();
            if chunk.is_empty() {
                break;
            }
            chunks.push(chunk);
        }
        let completed: Vec<Vec<ComputeCompletion>> = self.pool.scope(|scope| {
            for chunk in chunks {
                scope.spawn(async move { chunk.into_iter().map(ComputeTask::execute).collect() });
            }
        });
        completed.into_iter().flatten().collect()
    }
}

/// Nonblocking planner adapter. One tracker persists across generation changes,
/// so canceled workers cannot evade the global task quota. The Mutex wraps only
/// coordinator-owned lifecycle metadata for Bevy Resource's Sync bound; workers
/// never acquire it and never read or write the evaluator cache.
///
/// ```
/// use bevy_tasks::TaskPoolBuilder;
/// use gekijyou_bevy::{AsyncDriver, TaskLimits};
/// use gekijyou_computation::{GraphState, Evaluator, FrameIndex, RequestBatch, PlannerConfig};
/// let mut state = GraphState::new();
/// let node = state.edit(|e| e.map(e.frame(), |f: &FrameIndex| f.0))?;
/// let mut evaluator = Evaluator::new(&state);
/// let mut driver = AsyncDriver::new(state.snapshot(), TaskLimits { max_in_flight: 2, max_pending: 4 }, 1);
/// let pool = TaskPoolBuilder::new().num_threads(2).build();
/// let mut batch = RequestBatch::new();
/// let id = batch.at(FrameIndex(20)).request(node);
/// let generation = driver.begin(&mut evaluator, state.snapshot(), batch, PlannerConfig::default())?;
/// let results = loop {
///     if let Some((results, _)) = driver.poll(&mut evaluator, &pool)? { break results; }
///     std::thread::yield_now();
/// };
/// assert_eq!(results.get(id)?, &20);
/// assert!(!driver.cancel(generation)); // A delivered request is already finished.
/// # Ok::<(), gekijyou_computation::EvalError>(())
/// ```
pub struct AsyncDriver {
    tracker: Mutex<TaskCoordinator<ComputeCompletion>>,
    plan: Option<EvaluationPlan>,
    generation: RequestGeneration,
}
impl AsyncDriver {
    /// `chunk_size` only groups logical tasks into physical TaskPool jobs;
    /// quotas, terminals and per-task statistics stay logical. Zero panics
    /// at construction.
    pub fn new(snapshot: EvaluationSnapshot, limits: TaskLimits, chunk_size: usize) -> Self {
        let generation = RequestGeneration::new();
        Self {
            tracker: Mutex::new(TaskCoordinator::new(
                snapshot, generation, limits, chunk_size,
            )),
            plan: None,
            generation,
        }
    }
    fn tracker(&mut self) -> &mut TaskCoordinator<ComputeCompletion> {
        self.tracker
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub fn generation(&self) -> RequestGeneration {
        self.generation
    }
    pub fn stats(&mut self) -> TaskStats {
        self.tracker().stats()
    }
    pub fn in_flight(&mut self) -> usize {
        self.tracker().in_flight()
    }
    /// Explicitly cancels the active request. An old token cannot cancel its
    /// successor. Running workers retain quota and immutable data until terminal.
    pub fn cancel(&mut self, generation: RequestGeneration) -> bool {
        if generation != self.generation || self.plan.is_none() {
            return false;
        }
        self.tracker().cancel();
        if let Some(mut plan) = self.plan.take() {
            plan.cancel();
        }
        true
    }
    /// Replaces the active request with a fresh identity, even on the same snapshot.
    /// Cache histories are managed by Evaluator, independently of cancellation.
    pub fn begin(
        &mut self,
        evaluator: &mut Evaluator,
        snapshot: EvaluationSnapshot,
        batch: RequestBatch,
        config: PlannerConfig,
    ) -> Result<RequestGeneration, EvalError> {
        // Validate before canceling a valid existing request.
        self.tracker().validate_snapshot(&snapshot)?;
        let plan = evaluator.plan(snapshot.clone(), batch, config)?;
        let generation = RequestGeneration::new();
        self.tracker().set_session(snapshot, generation)?;
        if let Some(mut old) = self.plan.take() {
            old.cancel();
        }
        self.plan = Some(plan);
        self.generation = generation;
        Ok(generation)
    }
    /// Polling always reclaims obsolete completions, even when no plan is active.
    /// Errors fail the entire request; completed pure cache entries may survive.
    /// Panics also terminate the active request before resuming unwinding,
    /// including panics while dropping obsolete completion payloads.
    pub fn poll(
        &mut self,
        evaluator: &mut Evaluator,
        pool: &TaskPool,
    ) -> Result<Option<(EvaluationResults, PlannerStats)>, EvalError> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.poll_inner(evaluator, pool)
        }));
        match outcome {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => {
                self.fail_active_request();
                Err(error)
            }
            Err(panic) => {
                self.fail_active_request();
                std::panic::resume_unwind(panic);
            }
        }
    }
    fn fail_active_request(&mut self) {
        // Detach first: dropping pending jobs or plan values may run user code.
        let plan = self.plan.take();
        self.tracker().cancel();
        if let Some(mut plan) = plan {
            plan.cancel();
        }
    }
    fn poll_inner(
        &mut self,
        evaluator: &mut Evaluator,
        pool: &TaskPool,
    ) -> Result<Option<(EvaluationResults, PlannerStats)>, EvalError> {
        let finished = self.tracker().poll_completions();
        let mut completed = Vec::new();
        for task in finished {
            completed.push(task.result?);
        }
        let slots = self.tracker().available();
        let Some(plan) = self.plan.as_mut() else {
            return Ok(None);
        };
        let tasks = evaluator.advance_with_budget(plan, completed, slots)?;
        if plan.is_complete() {
            return self.plan.take().unwrap().finish().map(Some);
        }
        // The batch entry dispatches, including empty batches, so already
        // queued work drains without a separate dispatch_pending call.
        assert!(
            self.tracker()
                .submit_batch_fallible(
                    pool,
                    tasks.into_iter().map(|task| move |_| {
                        let completion = task.execute();
                        if let Some(error) = completion.failure() {
                            Err(error)
                        } else {
                            Ok(completion)
                        }
                    })
                )
                .is_some(),
            "Reserved task admission failed"
        );
        Ok(None)
    }
}
