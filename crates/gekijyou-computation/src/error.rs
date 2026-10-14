use super::{FrameIndex, GraphDefinitionEpoch, NodeId};
use thiserror::Error;

/// Validation errors never publish a partial edit or partial batch result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum GraphError {
    #[error("Node {node:?} belongs to another graph")]
    ForeignNode { node: NodeId },
    #[error("Node {node:?} does not exist")]
    MissingNode { node: NodeId },
    #[error("Type mismatch for node {node:?}")]
    TypeMismatch { node: NodeId },
    #[error("Node {node:?} is not an ordinary input")]
    NotInput { node: NodeId },
    #[error("Node {node:?} is not a computation")]
    NotComputation { node: NodeId },
    #[error("The dedicated Frame cannot be edited or deleted")]
    InvalidFrame,
    #[error("Graph contains a dependency cycle")]
    Cycle,
    #[error("Revision exhausted for node {node:?}")]
    RevisionExhausted { node: NodeId },
    #[error("Graph definition epoch exhausted")]
    DefinitionEpochExhausted,
    #[error("LRU capacity must be positive; use Off for no retention")]
    InvalidCacheCapacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum EvalError {
    #[error("Snapshot belongs to another graph")]
    ForeignGraph,
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("Missing input {node:?} at frame {frame:?}")]
    MissingInput { frame: FrameIndex, node: NodeId },
    #[error("Snapshot epoch {requested:?} precedes active epoch {active:?}")]
    StaleGraphDefinition {
        requested: GraphDefinitionEpoch,
        active: GraphDefinitionEpoch,
    },
    #[error("RequestId belongs to another batch")]
    ForeignRequest,
    #[error("Invalid request index {index}")]
    InvalidRequest { index: usize },
    #[error("Output type mismatch for request {index}")]
    ResultTypeMismatch { index: usize },
    #[error("DFS depth exceeds supported maximum {limit}")]
    DepthLimit { limit: usize },
    #[error("Invalid planner configuration")]
    InvalidPlannerConfig,
    #[error("Planner active window exceeds task budget {limit}")]
    PlannerBudgetExceeded { limit: usize },
    #[error("Plan is canceled")]
    PlanCanceled,
    #[error("Plan has not produced all requested outputs")]
    PlanIncomplete,
    #[error("Completion belongs to a different plan")]
    ForeignPlanCompletion,
    #[error("Worker terminated with a panic")]
    WorkerPanicked,
}
