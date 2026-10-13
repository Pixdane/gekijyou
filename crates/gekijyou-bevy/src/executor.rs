use crate::{RuntimeGeneration, TaskCoordinator, TaskLimits, TaskStats};
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
pub struct AsyncDriver {
    tracker: Mutex<TaskCoordinator<ComputeCompletion>>,
    plan: Option<EvaluationPlan>,
    generation: RuntimeGeneration,
}
impl AsyncDriver {
    pub fn new(
        snapshot: EvaluationSnapshot,
        generation: RuntimeGeneration,
        limits: TaskLimits,
    ) -> Self {
        Self {
            tracker: Mutex::new(TaskCoordinator::new(snapshot, generation, limits)),
            plan: None,
            generation,
        }
    }
    fn tracker(&mut self) -> &mut TaskCoordinator<ComputeCompletion> {
        self.tracker
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub fn generation(&self) -> RuntimeGeneration {
        self.generation
    }
    pub fn stats(&mut self) -> TaskStats {
        self.tracker().stats()
    }
    pub fn in_flight(&mut self) -> usize {
        self.tracker().in_flight()
    }
    pub fn cancel(
        &mut self,
        snapshot: EvaluationSnapshot,
        generation: RuntimeGeneration,
    ) -> Result<(), EvalError> {
        if let Some(mut plan) = self.plan.take() {
            plan.cancel();
        }
        self.tracker().set_session(snapshot, generation)?;
        self.tracker().cancel();
        self.generation = generation;
        Ok(())
    }
    pub fn begin(
        &mut self,
        evaluator: &mut Evaluator,
        snapshot: EvaluationSnapshot,
        generation: RuntimeGeneration,
        batch: RequestBatch,
        config: PlannerConfig,
    ) -> Result<(), EvalError> {
        self.cancel(snapshot.clone(), generation)?;
        self.plan = Some(evaluator.plan(snapshot, batch, config)?);
        Ok(())
    }
    /// Polling always reclaims obsolete completions, even when no plan is active.
    /// Errors fail the entire request; completed pure cache entries may survive.
    pub fn poll(
        &mut self,
        evaluator: &mut Evaluator,
        pool: &TaskPool,
    ) -> Result<Option<(EvaluationResults, PlannerStats)>, EvalError> {
        let finished = self.tracker().poll_completions();
        let mut completed = Vec::new();
        for task in finished {
            match task.result {
                Ok(value) => completed.push(value),
                Err(error) => {
                    if let Some(mut plan) = self.plan.take() {
                        plan.cancel();
                    }
                    self.tracker().cancel();
                    return Err(error);
                }
            }
        }
        let slots = self.tracker().available();
        let Some(plan) = self.plan.as_mut() else {
            return Ok(None);
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.advance_with_budget(plan, completed, slots)
        }));
        let tasks = match outcome {
            Ok(Ok(tasks)) => tasks,
            Ok(Err(error)) => {
                self.plan = None;
                self.tracker().cancel();
                return Err(error);
            }
            Err(panic) => {
                self.plan = None;
                self.tracker().cancel();
                std::panic::resume_unwind(panic);
            }
        };
        let plan = self.plan.as_mut().unwrap();
        if plan.is_complete() {
            return self.plan.take().unwrap().finish().map(Some);
        }
        self.tracker().dispatch_pending(pool);
        for task in tasks {
            // Slots were reserved by the coordinator budget. No other caller can
            // submit while this exclusive method runs; admission is deterministic.
            assert!(
                self.tracker()
                    .submit_fallible(pool, move |_| {
                        let completion = task.execute();
                        if let Some(error) = completion.failure() {
                            Err(error)
                        } else {
                            Ok(completion)
                        }
                    })
                    .is_some(),
                "Reserved task admission failed"
            );
        }
        Ok(None)
    }
}
