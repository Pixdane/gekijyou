use super::{batch::ContextId, node::NodeId};
use thiserror::Error;

/// A graph ownership, configuration, dependency, or structure validation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum GraphError {
    /// The handle or dependency belongs to another graph.
    #[error("Node {node:?} belongs to another graph")]
    ForeignNode { node: NodeId },
    /// The referenced node does not exist in this graph.
    #[error("Node {node:?} does not exist")]
    MissingNode { node: NodeId },
    /// The declared type does not match the node's stored output type.
    #[error("Type mismatch for node {node:?}")]
    TypeMismatch { node: NodeId },
    /// External input nodes cannot be replaced by computation closures.
    #[error("Input node {node:?} cannot be replaced")]
    InputNodeReplacement { node: NodeId },
    /// External inputs read the current context and cannot retain computation results.
    #[error("Input node {node:?} cannot have a computation cache policy")]
    InputNodeCachePolicy { node: NodeId },
    /// The built-in frame node is no longer an input.
    #[error("Frame node must be an external input")]
    InvalidFrame,
    /// The graph contains a dependency cycle.
    #[error("Graph contains a dependency cycle")]
    Cycle,
    /// The node's revision cannot be incremented without wrapping.
    #[error("Revision counter exhausted for node {node:?}")]
    RevisionExhausted { node: NodeId },
    /// The graph's replacement epoch cannot be incremented without wrapping.
    #[error("Graph replacement epoch exhausted")]
    ReplacementEpochExhausted,
}

/// A context, request, or evaluation failure. Computation panics unwind normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum EvalError {
    /// Context, batch, evaluator, and supplied graph must have the same identity.
    #[error("Context, batch, evaluator, or supplied graph belongs to another graph")]
    ForeignGraph,
    /// A node's ownership, existence, or declared type is invalid.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Only external input nodes can receive context values.
    #[error("Node {node:?} is a computation, not an external input")]
    NotInput { node: NodeId },
    /// An input reached by a requested target has no value in this context.
    #[error("Missing input {node:?} in context {context:?}")]
    MissingInput { context: ContextId, node: NodeId },
    /// A context handle was issued by another batch.
    #[error("ContextId belongs to another batch")]
    ForeignContext,
    /// A context handle does not name a registered context in this batch.
    #[error("Invalid context index {index}")]
    InvalidContext { index: usize },
    /// A request handle was issued by another batch.
    #[error("RequestId belongs to another batch")]
    ForeignRequest,
    /// A request handle does not name a result in this batch.
    #[error("Invalid request index {index}")]
    InvalidRequest { index: usize },
    /// The requested type differs from the stored result's concrete type.
    #[error("Output type mismatch for request {index}")]
    ResultTypeMismatch { index: usize },
}
