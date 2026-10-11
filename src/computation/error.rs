use super::node::NodeId;
use thiserror::Error;

/// A graph ownership, dependency, or structure validation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum GraphError {
    /// The handle or dependency belongs to another graph.
    #[error("Node {node:?} belongs to another graph")]
    ForeignNode { node: NodeId },
    /// The referenced node does not exist in this graph.
    #[error("Node {node:?} does not exist")]
    MissingNode { node: NodeId },
    /// The declared type does not match the node's stored output type.
    #[error("Output type mismatch for node {node:?}")]
    TypeMismatch { node: NodeId },
    /// External input nodes cannot be replaced by computation closures.
    #[error("Input node {node:?} cannot be replaced")]
    InputNodeReplacement { node: NodeId },
    /// The built-in frame node is no longer an input.
    #[error("Frame node must be an input")]
    InvalidFrame,
    /// The graph contains a dependency cycle.
    #[error("Graph contains a dependency cycle")]
    Cycle,
}
