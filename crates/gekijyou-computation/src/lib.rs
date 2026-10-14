//! Typed pure computation graphs with transactional editing and immutable snapshots.
//!
//! Values and closure captures must remain semantically immutable. Closures must
//! be deterministic functions of explicit inputs and immutable configuration;
//! interior mutability, system time and previous ECS state cannot bypass this contract.
//! Equivalence comparators must be stable equivalence relations permitting safe
//! substitution in every downstream computation. Approximate float equality usually
//! violates this requirement; bitwise comparison is available through custom semantics.
//! Reference addresses and Arc reference counts are not semantic computation inputs.
//! Model randomness with explicit seeds rather than a mutable global generator.

//!
//! ```
//! use gekijyou_computation::{GraphState, Evaluator, FrameIndex, RequestBatch};
//! let mut state = GraphState::new();
//! let (offset, position) = state.edit(|e| {
//!     let offset = e.input::<u64>();
//!     e.set(offset, 10)?;
//!     let position = e.map((e.frame(), offset.node()), |f: &FrameIndex, x: &u64| f.0 + x)?;
//!     Ok((offset, position))
//! })?;
//! let mut evaluator = Evaluator::new(&state);
//! let mut batch = RequestBatch::new();
//! let a = batch.at(FrameIndex(120)).request(position);
//! let b = batch.at(FrameIndex(30)).request(position);
//! let results = evaluator.bind(state.snapshot())?.evaluate(batch)?;
//! assert_eq!(results.get(a)?, &130);
//! assert_eq!(*results.get_shared(b)?, 40);
//! state.edit(|e| e.set(offset, 20))?;
//! assert_eq!(*evaluator.bind(state.snapshot())?.sample(FrameIndex(30), position)?, 50);
//! # Ok::<(), gekijyou_computation::EvalError>(())
//! ```

mod cache;
mod collections;
mod context_cache;
mod error;
mod eval;
mod graph;
mod inputs;
mod node;
mod request;
mod snapshot;
#[cfg(test)]
mod test_allocations;
mod value;

pub use error::{EvalError, GraphError};
pub use graph::{Graph, GraphEdit, GraphState};
pub use inputs::{ComputationDefinition, ComputationInputs};
pub use node::{
    CachePolicy, FrameIndex, GraphDefinitionEpoch, GraphId, Input, Node, NodeCacheConfig, NodeId,
};
pub use snapshot::{EvaluationSnapshot, SnapshotId};

pub use eval::{CacheStats, EvalStats, EvaluationScope, Evaluator};
pub use request::{EvaluationResults, FrameRequests, RequestBatch, RequestBatchId, RequestId};

pub use value::SharedValue;

mod planner;
pub use planner::{
    ComputeCompletion, ComputeTask, ComputeTaskId, EvaluationPlan, Executor, PlannerConfig,
    PlannerStats, SerialExecutor,
};
