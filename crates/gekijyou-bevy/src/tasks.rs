use crate::RuntimeGeneration;
use bevy_tasks::TaskPool;
use gekijyou_computation::{EvalError, EvaluationSnapshot, SnapshotId};
use std::{
    collections::{HashMap, VecDeque},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::mpsc::{self, Receiver, Sender},
};

/// Total quotas across every generation, including canceled running workers.
#[derive(Clone, Copy, Debug)]
pub struct TaskLimits {
    pub max_in_flight: usize,
    pub max_pending: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskStatus {
    Queued,
    Running,
    CanceledRunning,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskStats {
    pub dispatched: u64,
    pub terminal: u64,
    pub failed: u64,
    pub obsolete: u64,
    pub canceled_pending: u64,
    pub peak_in_flight: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Session {
    snapshot: SnapshotId,
    generation: RuntimeGeneration,
}
type Job<R> = Box<dyn FnOnce(EvaluationSnapshot) -> Result<R, EvalError> + Send>;
struct Queued<R> {
    id: TaskId,
    session: Session,
    snapshot: EvaluationSnapshot,
    job: Job<R>,
}
struct Running {
    session: Session,
    canceled: bool,
}
struct Completion<R> {
    id: TaskId,
    session: Session,
    result: Result<R, EvalError>,
}
struct TerminalGuard<R> {
    sender: Sender<Completion<R>>,
    completion: Option<Completion<R>>,
}
impl<R> TerminalGuard<R> {
    fn finish(mut self, result: Result<R, EvalError>) {
        if let Some(mut completed) = self.completion.take() {
            completed.result = result;
            let _ = self.sender.send(completed);
        }
    }
}
impl<R> Drop for TerminalGuard<R> {
    fn drop(&mut self) {
        if let Some(completed) = self.completion.take() {
            let _ = self.sender.send(completed);
        }
    }
}
/// A current-session terminal result. Worker failure is explicit and is not retried.
pub struct FinishedTask<R> {
    pub id: TaskId,
    pub snapshot: SnapshotId,
    pub generation: RuntimeGeneration,
    pub result: Result<R, EvalError>,
}

/// Bevy worker lifecycle prototype. Immutable jobs cannot access the coordinator
/// or the main evaluator's caches. Switching sessions cancels queued jobs only;
/// old running workers retain quota until one confirmed terminal message.
///
/// Call `poll` periodically to reclaim completions, including obsolete results.
/// Dropping this coordinator closes delivery; detached jobs release their owned
/// snapshots on natural completion. This is not a force-interruption mechanism.
pub struct TaskCoordinator<R: Send + 'static> {
    snapshot: EvaluationSnapshot,
    generation: RuntimeGeneration,
    limits: TaskLimits,
    next: u64,
    pending: VecDeque<Queued<R>>,
    running: HashMap<TaskId, Running>,
    sender: Sender<Completion<R>>,
    receiver: Receiver<Completion<R>>,
    stats: TaskStats,
}
impl<R: Send + 'static> TaskCoordinator<R> {
    pub fn new(
        snapshot: EvaluationSnapshot,
        generation: RuntimeGeneration,
        limits: TaskLimits,
    ) -> Self {
        assert!(
            limits.max_in_flight > 0,
            "At least one worker slot is required"
        );
        limits
            .max_in_flight
            .checked_add(limits.max_pending)
            .expect("Task quota overflow");
        let (sender, receiver) = mpsc::channel();
        Self {
            snapshot,
            generation,
            limits,
            next: 0,
            pending: VecDeque::new(),
            running: HashMap::new(),
            sender,
            receiver,
            stats: TaskStats::default(),
        }
    }
    fn session(&self) -> Session {
        Session {
            snapshot: self.snapshot.id(),
            generation: self.generation,
        }
    }
    pub fn available(&self) -> usize {
        self.limits.max_in_flight + self.limits.max_pending
            - self.running.len()
            - self.pending.len()
    }
    pub fn in_flight(&self) -> usize {
        self.running.len()
    }
    pub fn pending(&self) -> usize {
        self.pending.len()
    }
    pub fn stats(&self) -> TaskStats {
        self.stats
    }
    pub fn status(&self, id: TaskId) -> Option<TaskStatus> {
        if self.pending.iter().any(|j| j.id == id) {
            return Some(TaskStatus::Queued);
        }
        self.running.get(&id).map(|r| {
            if r.canceled {
                TaskStatus::CanceledRunning
            } else {
                TaskStatus::Running
            }
        })
    }
    /// Returns None at capacity, without accepting or retaining the job.
    pub fn submit(
        &mut self,
        pool: &TaskPool,
        job: impl FnOnce(EvaluationSnapshot) -> R + Send + 'static,
    ) -> Option<TaskId> {
        self.submit_fallible(pool, move |snapshot| Ok(job(snapshot)))
    }
    /// Explicit task failures follow the same one-terminal-state quota protocol.
    pub fn submit_fallible(
        &mut self,
        pool: &TaskPool,
        job: impl FnOnce(EvaluationSnapshot) -> Result<R, EvalError> + Send + 'static,
    ) -> Option<TaskId> {
        if self.running.len() >= self.limits.max_in_flight
            && self.pending.len() >= self.limits.max_pending
        {
            return None;
        }
        let id = TaskId(self.next);
        self.next = self.next.checked_add(1).expect("Task identity exhausted");
        self.pending.push_back(Queued {
            id,
            session: self.session(),
            snapshot: self.snapshot.clone(),
            job: Box::new(job),
        });
        self.dispatch(pool);
        Some(id)
    }
    /// Definitions may advance but cannot roll back. Snapshot or generation changes
    /// discard old waiting jobs; running jobs remain counted and retain their Arcs.
    pub fn set_session(
        &mut self,
        snapshot: EvaluationSnapshot,
        generation: RuntimeGeneration,
    ) -> Result<(), EvalError> {
        if snapshot.graph_id() != self.snapshot.graph_id() {
            return Err(EvalError::ForeignGraph);
        }
        if snapshot.definition_epoch() < self.snapshot.definition_epoch() {
            return Err(EvalError::StaleGraphDefinition {
                requested: snapshot.definition_epoch(),
                active: self.snapshot.definition_epoch(),
            });
        }
        if snapshot.id() == self.snapshot.id() && generation == self.generation {
            return Ok(());
        }
        self.snapshot = snapshot;
        self.generation = generation;
        self.cancel();
        Ok(())
    }
    /// Cancel queued jobs and mark running jobs obsolete without releasing quota.
    pub fn cancel(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        self.stats.canceled_pending += pending.len() as u64;
        for running in self.running.values_mut() {
            running.canceled = true;
        }
        drop(pending);
    }
    pub fn dispatch_pending(&mut self, pool: &TaskPool) {
        self.dispatch(pool);
    }

    fn dispatch(&mut self, pool: &TaskPool) {
        while self.running.len() < self.limits.max_in_flight {
            let Some(queued) = self.pending.pop_front() else {
                break;
            };
            let guard = TerminalGuard {
                sender: self.sender.clone(),
                completion: Some(Completion {
                    id: queued.id,
                    session: queued.session,
                    result: Err(EvalError::WorkerPanicked),
                }),
            };
            self.running.insert(
                queued.id,
                Running {
                    session: queued.session,
                    canceled: false,
                },
            );
            self.stats.dispatched += 1;
            self.stats.peak_in_flight = self.stats.peak_in_flight.max(self.running.len());
            pool.spawn(async move {
                let result = catch_unwind(AssertUnwindSafe(|| (queued.job)(queued.snapshot)))
                    .unwrap_or(Err(EvalError::WorkerPanicked));
                guard.finish(result);
            })
            .detach();
        }
    }
    /// Reclaims quota before filtering or delivering payloads. Duplicate/late
    /// terminal messages cannot release a slot twice. Panicking user destructors
    /// therefore do not strand a running ticket or roll back the active session.
    pub fn poll(&mut self, pool: &TaskPool) -> Vec<FinishedTask<R>> {
        let finished = self.poll_completions();
        self.dispatch(pool);
        finished
    }
    /// Separate terminal processing lets a planner cancel successors on failure
    /// before any queued work is dispatched.
    pub fn poll_completions(&mut self) -> Vec<FinishedTask<R>> {
        let mut finished = Vec::new();
        while let Ok(completed) = self.receiver.try_recv() {
            let Some(running) = self.running.remove(&completed.id) else {
                continue;
            };
            self.stats.terminal += 1;
            if completed.result.is_err() {
                self.stats.failed += 1;
            }
            if running.canceled
                || running.session != self.session()
                || completed.session != running.session
            {
                self.stats.obsolete += 1;
                continue;
            }
            finished.push(FinishedTask {
                id: completed.id,
                snapshot: completed.session.snapshot,
                generation: completed.session.generation,
                result: completed.result,
            });
        }
        finished
    }
}
