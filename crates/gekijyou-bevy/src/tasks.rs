use bevy_tasks::TaskPool;
use gekijyou_computation::{
    ComputeCompletion, ComputeTask, EvalError, EvaluationSnapshot, SnapshotId,
};
use smallvec::SmallVec;
use std::{
    collections::{HashMap, VecDeque},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
};

/// Process-local identity of an explicit request session, independent of snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestGeneration(u64);
impl RequestGeneration {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(
            NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("Request generation exhausted"),
        )
    }
}
impl Default for RequestGeneration {
    fn default() -> Self {
        Self::new()
    }
}

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
    /// Physical TaskPool jobs spawned; logical chunks make this smaller than
    /// `dispatched`. Every successful `pool.spawn` path increments it once.
    pub worker_jobs_spawned: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Session {
    snapshot: SnapshotId,
    generation: RequestGeneration,
}
type Job<R> = Box<dyn FnOnce(EvaluationSnapshot) -> Result<R, EvalError> + Send>;
/// Queued unit of work: a generic snapshot job, or one pure compute task whose
/// result is mapped into the coordinator's result type on the main thread.
enum QueuedWork<R> {
    Job(Job<R>),
    Compute(ComputeTask, fn(ComputeCompletion) -> R),
}
struct Queued<R> {
    id: TaskId,
    session: Session,
    snapshot: EvaluationSnapshot,
    work: QueuedWork<R>,
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
    pub generation: RequestGeneration,
    pub result: Result<R, EvalError>,
}

/// Bevy worker lifecycle coordinator. Immutable jobs cannot access the coordinator
/// or the main evaluator's caches. Switching sessions cancels queued jobs only;
/// old running workers retain quota until one confirmed terminal message.
///
/// Call `poll` periodically to reclaim completions, including obsolete results.
/// Dropping this coordinator closes delivery; detached jobs release their owned
/// snapshots on natural completion. This is not a force-interruption mechanism.
pub struct TaskCoordinator<R: Send + 'static> {
    snapshot: EvaluationSnapshot,
    generation: RequestGeneration,
    limits: TaskLimits,
    chunk_size: usize,
    next: u64,
    pending: VecDeque<Queued<R>>,
    running: HashMap<TaskId, Running>,
    sender: Sender<Completion<R>>,
    receiver: Receiver<Completion<R>>,
    stats: TaskStats,
}
impl<R: Send + 'static> TaskCoordinator<R> {
    /// `chunk_size` only groups logical tasks into physical TaskPool jobs;
    /// quotas, per-task terminals and statistics stay logical, and
    /// `worker_jobs_spawned` is not guaranteed smaller than `dispatched`
    /// (chunk 1 keeps them equal). Zero panics at construction.
    pub fn new(
        snapshot: EvaluationSnapshot,
        generation: RequestGeneration,
        limits: TaskLimits,
        chunk_size: usize,
    ) -> Self {
        assert!(
            limits.max_in_flight > 0,
            "At least one worker slot is required"
        );
        assert!(chunk_size > 0, "Chunk size must be positive");
        limits
            .max_in_flight
            .checked_add(limits.max_pending)
            .expect("Task quota overflow");
        let (sender, receiver) = mpsc::channel();
        Self {
            snapshot,
            generation,
            limits,
            chunk_size,
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
        if self.available() == 0 {
            return None;
        }
        let id = self.enqueue(job);
        self.dispatch(pool);
        Some(id)
    }
    /// Submits a whole batch as one admission decision. The exact length is
    /// taken first: any batch that does not fit the total quota is rejected
    /// whole (None), leaving queue, running, statistics and task identity
    /// untouched; an exact fit is always accepted. Accepted tasks are queued
    /// in iteration order and dispatched once, in chunks. An empty batch is
    /// accepted and still dispatches already-pending work.
    ///
    /// The capacity rejection itself is atomic. If the caller's own iterator
    /// panics while being consumed, an already-accepted prefix can stay queued
    /// and undispatched; `cancel` cleans that up.
    pub fn submit_batch_fallible<I, J>(&mut self, pool: &TaskPool, jobs: I) -> Option<Vec<TaskId>>
    where
        I: IntoIterator<Item = J>,
        I::IntoIter: ExactSizeIterator,
        J: FnOnce(EvaluationSnapshot) -> Result<R, EvalError> + Send + 'static,
    {
        let iter = jobs.into_iter();
        let len = iter.len();
        if len > self.available() {
            return None;
        }
        // Identity overflow is detected before any admission; nothing is queued.
        let len = u64::try_from(len).expect("Task identity exhausted");
        self.next.checked_add(len).expect("Task identity exhausted");
        let mut ids = Vec::with_capacity(len as usize);
        for job in iter {
            let id = self.enqueue(job);
            ids.push(id);
        }
        self.dispatch(pool);
        Some(ids)
    }
    /// Shared enqueue: ID allocation and queue insertion only. Dispatch is
    /// always the caller's explicit decision; callers validate admission.
    fn enqueue<J>(&mut self, job: J) -> TaskId
    where
        J: FnOnce(EvaluationSnapshot) -> Result<R, EvalError> + Send + 'static,
    {
        let id = TaskId(self.next);
        self.next = self.next.checked_add(1).expect("Task identity exhausted");
        self.pending.push_back(Queued {
            id,
            session: self.session(),
            snapshot: self.snapshot.clone(),
            work: QueuedWork::Job(Box::new(job)),
        });
        id
    }
    /// Definitions may advance but cannot roll back. Snapshot or generation changes
    /// discard old waiting jobs; running jobs remain counted and retain their Arcs.
    pub fn set_session(
        &mut self,
        snapshot: EvaluationSnapshot,
        generation: RequestGeneration,
    ) -> Result<(), EvalError> {
        self.validate_snapshot(&snapshot)?;
        if snapshot.id() == self.snapshot.id() && generation == self.generation {
            return Ok(());
        }
        self.snapshot = snapshot;
        self.generation = generation;
        self.cancel();
        Ok(())
    }
    pub(crate) fn validate_snapshot(&self, snapshot: &EvaluationSnapshot) -> Result<(), EvalError> {
        if snapshot.graph_id() != self.snapshot.graph_id() {
            return Err(EvalError::ForeignGraph);
        }
        if snapshot.definition_epoch() < self.snapshot.definition_epoch() {
            return Err(EvalError::StaleGraphDefinition {
                requested: snapshot.definition_epoch(),
                active: self.snapshot.definition_epoch(),
            });
        }
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
            let chunk_len = self
                .chunk_size
                .min(self.pending.len())
                .min(self.limits.max_in_flight - self.running.len());
            if chunk_len == 0 {
                break;
            }
            // One chunk is one physical job. Guards and running tickets are
            // registered for every logical task before a single spawn.
            let mut chunk: smallvec::SmallVec<[(Queued<R>, TerminalGuard<R>); 4]> =
                SmallVec::with_capacity(chunk_len);
            for _ in 0..chunk_len {
                let queued = self.pending.pop_front().unwrap();
                let guard = TerminalGuard {
                    sender: self.sender.clone(),
                    completion: Some(Completion {
                        id: queued.id,
                        session: queued.session,
                        result: Err(EvalError::WorkerPanicked),
                    }),
                };
                chunk.push((queued, guard));
            }
            for (queued, _) in &chunk {
                self.running.insert(
                    queued.id,
                    Running {
                        session: queued.session,
                        canceled: false,
                    },
                );
                self.stats.dispatched += 1;
            }
            self.stats.peak_in_flight = self.stats.peak_in_flight.max(self.running.len());
            // `pool.spawn` runs the future on a worker immediately; the borrow
            // of self ends before the call, so the physical job counter only
            // increments on the successful spawn path.
            pool.spawn(async move {
                // FIFO order inside the chunk; one panicking job never skips
                // the rest, each task still finishes its own terminal guard.
                let mut iter = chunk.into_iter().peekable();
                while let Some((queued, guard)) = iter.next() {
                    match queued.work {
                        QueuedWork::Job(job) => {
                            let result = catch_unwind(AssertUnwindSafe(|| job(queued.snapshot)))
                                .unwrap_or(Err(EvalError::WorkerPanicked));
                            guard.finish(result);
                        }
                        QueuedWork::Compute(task, map) => match task.kernel_id() {
                            // Non-batch tasks deliver their terminal message
                            // immediately, before any gated successor.
                            None => {
                                let completion = task.execute();
                                let result = match completion.failure() {
                                    Some(error) => Err(error),
                                    None => Ok(map(completion)),
                                };
                                guard.finish(result);
                            }
                            Some(kernel_id) => {
                                let snapshot_id = task.snapshot_id();
                                let mut tasks = vec![task];
                                let mut tail: Vec<(TerminalGuard<R>, fn(ComputeCompletion) -> R)> =
                                    vec![(guard, map)];
                                loop {
                                    let consecutive = match iter.peek() {
                                        Some((next, _)) => match &next.work {
                                            QueuedWork::Compute(next_task, _) => {
                                                next_task.kernel_id() == Some(kernel_id)
                                                    && next_task.snapshot_id() == snapshot_id
                                            }
                                            _ => false,
                                        },
                                        None => false,
                                    };
                                    if !consecutive {
                                        break;
                                    }
                                    let (next, next_guard) = iter.next().unwrap();
                                    match next.work {
                                        QueuedWork::Compute(next_task, next_map) => {
                                            tasks.push(next_task);
                                            tail.push((next_guard, next_map));
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                                let completions = ComputeTask::execute_batch(tasks);
                                for ((guard, map), completion) in tail.into_iter().zip(completions)
                                {
                                    let result = match completion.failure() {
                                        Some(error) => Err(error),
                                        None => Ok(map(completion)),
                                    };
                                    guard.finish(result);
                                }
                            }
                        },
                    }
                }
            })
            .detach();
            self.stats.worker_jobs_spawned += 1;
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

impl TaskCoordinator<ComputeCompletion> {
    /// Submits a whole wave of pure compute tasks as one admission decision.
    /// Rejected whole at capacity, accepted in iteration order, and dispatched
    /// once in chunks; returns the allocated task identities.
    pub(crate) fn submit_compute_batch(
        &mut self,
        pool: &TaskPool,
        tasks: Vec<ComputeTask>,
    ) -> Option<Vec<TaskId>> {
        let len = tasks.len();
        if len > self.available() {
            return None;
        }
        let len = u64::try_from(len).expect("Task identity exhausted");
        self.next.checked_add(len).expect("Task identity exhausted");
        let mut ids = Vec::with_capacity(len as usize);
        for task in tasks {
            let id = TaskId(self.next);
            self.next = self.next.checked_add(1).expect("Task identity exhausted");
            self.pending.push_back(Queued {
                id,
                session: self.session(),
                snapshot: self.snapshot.clone(),
                work: QueuedWork::Compute(task, std::convert::identity),
            });
            ids.push(id);
        }
        self.dispatch(pool);
        Some(ids)
    }
}
