//! Typed computation graphs for animation sampling.
//!
//! [`Graph`] defines nodes and their explicit dependencies. [`Context`] supplies
//! external inputs for one sample. [`RequestBatch`] groups contexts and targets;
//! [`Evaluator`] evaluates them serially, sharing dependencies within each context.
//! [`BatchResults`] owns the requested outputs independently of the graph.
//!
//! # Build and evaluate
//!
//! Each graph starts with a [`FrameIndex`] input. [`Graph::map`] registers a closure
//! without executing it; [`Node<T>`] is a copyable handle to its typed output.
//! Context assignments freeze when ownership transfers to a batch:
//!
//! ```
//! use gekijyou::computation::{Context, Evaluator, FrameIndex, Graph, RequestBatch};
//!
//! let mut graph = Graph::new();
//! let position = graph.map(graph.frame(), |frame: &FrameIndex| frame.0 * 2);
//! let mut context = Context::new(&graph);
//! context.set(&graph, graph.frame(), FrameIndex(12))?;
//! let mut batch = RequestBatch::new(&graph);
//! let context = batch.add_context(context)?;
//! let request = batch.request(context, position)?;
//! let results = Evaluator::new(&graph).evaluate_batch(&graph, batch)?;
//! drop(graph);
//! assert_eq!(results.get(request)?, &24);
//! # Ok::<(), gekijyou::computation::EvalError>(())
//! ```
//!
//! Inputs can be a single node, a tuple of one to four nodes, or `()` for a
//! zero-input computation. [`Graph::input`] declares another external input;
//! [`Graph::constant`] stores a shared value without requiring `Clone`:
//!
//! ```
//! use gekijyou::computation::{FrameIndex, Graph};
//!
//! let mut graph = Graph::new();
//! let frame = graph.frame();
//! let position = graph.map(frame, |f: &FrameIndex| f.0 as f64 * 2.0);
//! let opacity = graph.map(frame, |f: &FrameIndex| if f.0 < 30 { 1.0 } else { 0.0 });
//! let state = graph.map((position, opacity), |x: &f64, alpha: &f64| (*x, *alpha));
//!
//! // Handles can feed several branches without copying their output values.
//! let offset = graph.map(position, |x: &f64| x + 10.0);
//! ```
//!
//! # Identity, edits and failures
//!
//! A node belongs to the graph that created it. Passing a foreign handle to
//! `map` panics before insertion. [`Graph::replace`] preserves the handle while
//! changing its closure and dependencies, rejecting cycles before committing.
//! [`Graph::validate`] checks ownership, types, references and cycles.
//!
//! Contexts, batches and evaluators bind only graph identity. Batches use the
//! definitions present when evaluation starts; existing results retain their old
//! values across later edits. Each [`RequestId`] reads only its own batch's results.
//!
//! Every target is validated before any closure executes. Missing inputs produce
//! [`EvalError::MissingInput`] only when reached. Evaluation errors return no partial
//! results; computation panics unwind normally. The evaluator can be reused after
//! either kind of failure. Recursive traversal has a finite stack-depth limit.
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
//! nodes retain their semantics. The built-in [`FrameIndex`] starts with content
//! comparison. Equality must be a stable equivalence relation and permit substitution
//! in every downstream computation. Approximate comparisons that violate these laws
//! cannot serve as general cache equality. Floating-point bitwise distinctions,
//! including positive and negative zero, require an appropriate wrapper or comparator.
//!
//! # Current evaluation scope
//!
//! Dependencies are shared within each context in one batch. Equality and cache
//! policies are retained as metadata for future cross-batch reuse; they do not
//! participate in evaluation yet. A constant's closure retains its value regardless
//! of cache policy. Cross-batch caching and parallel scheduling are not implemented.

mod batch;
mod context;
mod error;
mod eval;
mod graph;
mod inputs;
mod node;
mod value;

pub use batch::{BatchResults, ContextId, RequestBatch, RequestId};
pub use context::Context;
pub use error::{EvalError, GraphError};
pub use eval::Evaluator;
pub use graph::Graph;
pub use inputs::{ComputationDefinition, ComputationInputs};
pub use node::{FrameIndex, Node, NodeId};
