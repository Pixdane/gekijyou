//! Computation graph construction with typed node handles.
//!
//! [`Graph`] stores nodes, explicit dependencies, and computation closures.
//! [`Node<T>`] identifies a node's output type. [`Graph::map`] registers a computation
//! without invoking its closure. This module currently supports graph construction;
//! evaluation is not yet implemented.
//!
//! Each graph starts with a [`Frame`] input node. Use it to build computations
//! with zero or more inputs. [`Graph::constant`] stores a value without requiring
//! it to be cloned; [`Graph::input`] declares an external input:
//!
//! ```
//! use gekijyou::computation::{Frame, Graph};
//!
//! let mut graph = Graph::new();
//! let frame = graph.frame();
//! let position = graph.map(frame, |f: &Frame| f.0 as f64 * 2.0);
//! let opacity = graph.map(frame, |f: &Frame| if f.0 < 30 { 1.0 } else { 0.0 });
//! let state = graph.map((position, opacity), |x: &f64, alpha: &f64| (*x, *alpha));
//!
//! // Combining nodes copies their handles, so position can feed another branch.
//! let offset = graph.map(position, |x: &f64| x + 10.0);
//! ```
//!
//! A handle belongs to the graph that created it. Passing a foreign handle to
//! `map` panics before inserting the node. [`Graph::replace`] preserves a computation
//! handle while changing its closure and dependencies, rejecting cycles before
//! committing. [`Graph::validate`] checks ownership, types, references, and cycles.
//!
//! # Value and computation contracts
//!
//! Computations, including zero-input closures, must be deterministic pure functions
//! of their explicit inputs and immutable captured configuration. Published values
//! must remain semantically immutable, including values with interior mutability.
//! Allocation addresses and reference counts are not stable computation inputs.
//!
//! Values default to shared-object identity comparison. [`Graph::register_eq`]
//! installs content comparison for subsequently created nodes of a type; existing
//! nodes retain their semantics. The built-in [`Frame`] type starts with content
//! comparison. Equality must be a stable equivalence relation and permit substitution
//! in every downstream computation. Approximate comparisons that violate these laws
//! cannot serve as general cache equality. Floating-point bitwise distinctions,
//! including positive and negative zero, require an appropriate wrapper or comparator.
//!
//! Cache policies are metadata only. No evaluation, caching, or scheduling is
//! implemented yet. A constant's value is retained by the graph's closure regardless
//! of its future cache policy.

mod error;
mod graph;
mod inputs;
mod node;
mod value;

pub use error::GraphError;
pub use graph::Graph;
pub use inputs::{ComputationDefinition, ComputationInputs};
pub use node::{Frame, Node, NodeId};
