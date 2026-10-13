//! Typed pure computation graphs with transactional editing and immutable snapshots.
//!
//! Values and closure captures must remain semantically immutable. Closures must
//! be deterministic functions of explicit inputs and immutable configuration;
//! interior mutability, system time and previous ECS state cannot bypass this contract.
//! Equivalence comparators must be stable equivalence relations permitting safe
//! substitution in every downstream computation. Approximate float equality usually
//! violates this requirement; bitwise comparison is available through custom semantics.

mod cache;
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

/// Frozen P3.5 implementation for historical tests and benchmarks.
/// New applications use the snapshot-based API at the crate root.
#[cfg(feature = "p35-baseline")]
pub mod p35;

pub use eval::{CacheStats, EvalStats, EvaluationScope, Evaluator};
pub use request::{EvaluationResults, FrameRequests, RequestBatch, RequestId};
