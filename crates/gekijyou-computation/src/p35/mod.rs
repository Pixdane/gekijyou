//! Typed computation graphs for animation sampling.
//!
//! [`Graph`] defines nodes and their explicit dependencies. [`Context`] supplies
//! external inputs for one sample. [`RequestBatch`] groups contexts and targets;
//! [`Evaluator`] evaluates them serially, sharing dependencies within each context
//! and retaining node-local memo results across contexts and batches.
//! [`BatchResults`] owns the requested outputs independently of the graph.
//!
//! # Build and evaluate
//!
//! Each graph starts with a [`FrameIndex`] input. [`Graph::map`] registers a closure
//! without executing it; [`Node<T>`] is a copyable handle to its typed output.
//! Context assignments freeze when ownership transfers to a batch:
//!
//! ```
//! use gekijyou_computation::p35::{Context, Evaluator, FrameIndex, Graph, RequestBatch};
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
//! # Ok::<(), gekijyou_computation::p35::EvalError>(())
//! ```
//!
//! Inputs can be a single node, a tuple of one to four nodes, or `()` for a
//! zero-input computation. [`Graph::input`] declares another external input;
//! [`Graph::constant`] stores a shared value without requiring `Clone`:
//!
//! ```
//! use gekijyou_computation::p35::{FrameIndex, Graph};
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
//! Values default to shared-object identity comparison. [`Graph::set_default_eq`]
//! and [`Graph::set_default_equivalence`] configure comparison for subsequently
//! created nodes of a type; existing nodes retain their semantics, including after
//! replacement. The built-in [`FrameIndex`] starts with content comparison.
//! Equality must be a stable equivalence relation and permit substitution
//! in every downstream computation. Approximate comparisons that violate these laws
//! cannot serve as general cache equality. Floating-point bitwise distinctions,
//! including positive and negative zero, can use a comparator based on `to_bits()`.
//!
//! [`Context::set`] creates a fresh stored shared object for each assignment. Equal
//! scalar inputs need content equivalence for reuse across contexts. For an
//! `Arc<Resource>` value, use the caller's resource identity explicitly:
//!
//! ```
//! use gekijyou_computation::p35::{Context, Evaluator, Graph, RequestBatch};
//! use std::sync::Arc;
//!
//! struct Resource { width: u32 }
//! let resource = Arc::new(Resource { width: 1920 });
//! let mut graph = Graph::new();
//! graph.set_default_equivalence::<Arc<Resource>, _>(Arc::ptr_eq);
//! let input = graph.input::<Arc<Resource>>();
//! let width = graph.map(input, |r: &Arc<Resource>| r.width);
//! let mut evaluator = Evaluator::new(&graph);
//! for _ in 0..2 {
//!     let mut context = Context::new(&graph);
//!     context.set(&graph, input, Arc::clone(&resource))?;
//!     let mut batch = RequestBatch::new(&graph);
//!     let context = batch.add_context(context)?;
//!     let request = batch.request(context, width)?;
//!     assert_eq!(evaluator.evaluate_batch(&graph, batch)?.get(request)?, &1920);
//! }
//! assert_eq!(evaluator.stats().computation_calls, 1);
//! assert_eq!(evaluator.stats().memo_hits, 1);
//! # Ok::<(), gekijyou_computation::p35::EvalError>(())
//! ```
//!
//! # Cache policy and statistics
//!
//! Computations default to [`CachePolicy::Last`], retaining one input combination.
//! [`CachePolicy::Lru`] retains bounded history with LRU eviction. [`CachePolicy::Transient`]
//! and `Lru { capacity: 0 }` retain no persistent entries; within-context sharing
//! remains available. [`Graph::set_cache_policy`] applies lazily on the next visit,
//! clearing only that node's history when the applied policy changes.
//! A variant or capacity change clears history, even `Last` to `Lru { capacity: 1 }`.
//! Input nodes always read the current context and reject cache policy configuration.
//!
//! Memo comparison uses each direct input source's value semantics. A memo query
//! still resolves all direct dependencies. Replacing a node invalidates its own
//! memo, preserving unchanged downstream nodes' history for equal actual inputs.
//! Cached inputs and outputs are shared Arcs and need not implement `Clone`.
//! A constant's closure retains its value regardless of cache policy.
//!
//! [`Evaluator::stats`] reads cumulative [`EvalStats`]; [`Evaluator::reset_stats`]
//! resets counters without clearing memo. Successful pure-function entries survive
//! later errors and panics in the same batch.
//!
//! Context caches are independently opt-in through [`Graph::set_context_cache_capacity`].
//! Zero disables them by default; positive capacity retains bounded LRU history.
//! They compare only the transitive external Input ancestors, using each input's
//! frozen semantics. Zero-input computations contribute no external roots. Hits
//! return before dependency traversal or direct Memo queries and do not grow
//! the dense per-context temporary result table. Missing roots skip this query and
//! follow ordinary DFS failure order. Any successful replacement lazily discards
//! Context history and rebuilds projections, preserving other nodes' Direct Memo.
//! Node additions and direct policy changes do not invalidate Context history.
//! Retained entry counts are bounded, but Arcs can retain arbitrarily large values.
//! Parallel scheduling is not implemented.
//!
//! ```
//! use gekijyou_computation::p35::{Context, Evaluator, FrameIndex, Graph, RequestBatch};
//! let mut graph = Graph::new();
//! let upstream = graph.map(graph.frame(), |f: &FrameIndex| f.0 * 2);
//! let scene = graph.map(upstream, |n: &u64| n + 1);
//! graph.set_context_cache_capacity(scene, 8)?;
//! let mut evaluator = Evaluator::new(&graph);
//! for _ in 0..2 {
//!     let mut context = Context::new(&graph);
//!     context.set(&graph, graph.frame(), FrameIndex(12))?;
//!     let mut batch = RequestBatch::new(&graph);
//!     let context = batch.add_context(context)?;
//!     let request = batch.request(context, scene)?;
//!     assert_eq!(evaluator.evaluate_batch(&graph, batch)?.get(request)?, &25);
//! }
//! assert_eq!(evaluator.stats().context_hits, 1);
//! assert_eq!(evaluator.stats().resolution_visits, 4); // First chain: 3, next hit: 1.
//! # Ok::<(), gekijyou_computation::p35::EvalError>(())
//! ```

mod batch;
mod cache;
mod context;
mod context_cache;
mod error;
mod eval;
mod graph;
mod inputs;
mod node;
mod value;

pub use batch::{BatchResults, ContextId, RequestBatch, RequestId};
pub use context::Context;
pub use error::{EvalError, GraphError};
pub use eval::{EvalStats, Evaluator};
pub use graph::Graph;
pub use inputs::{ComputationDefinition, ComputationInputs};
pub use node::{CachePolicy, FrameIndex, Node, NodeId};
