use super::value::{ErasedComputation, ValueSemantics};
use slotmap::new_key_type;
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

/// An integer target frame number.
///
/// Carries no frame rate or time domain. The caller defines conversions to seconds
/// or local time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameIndex(pub u64);

new_key_type! {
    /// A node storage key local to one graph.
    ///
    /// Different graphs can produce equal keys. An internal graph identity and
    /// this key together identify a node. Use typed [`Node<T>`] handles to build
    /// computations through the public API.
    pub struct NodeId;
}

/// A lightweight handle to a node with output type `T` in one graph.
///
/// Copying or cloning a handle does not copy the output value and does not require
/// `T` to implement [`Copy`] or [`Clone`]. Handles do not borrow the graph, so moving
/// the graph preserves their validity. Adding a computation still requires the
/// owning graph. A handle owns neither the graph nor its results and does not keep
/// the graph alive.
///
/// The output type is invariant so a handle always names its exact stored type:
///
/// ```compile_fail
/// use gekijyou::computation::{Graph, Node};
///
/// let mut graph = Graph::new();
/// let node = graph.input::<fn(&str)>();
/// let relabeled: Node<fn(&'static str)> = node;
/// ```
pub struct Node<T> {
    pub(super) graph_id: GraphId,
    pub(super) id: NodeId,
    // Require the exact stored type without owning T or inheriting its Send/Sync bounds.
    _marker: PhantomData<fn(T) -> T>,
}

impl<T> Node<T> {
    // Private construction prevents callers from forging types or graph ownership.
    pub(super) fn new(graph_id: GraphId, id: NodeId) -> Self {
        Self {
            graph_id,
            id,
            _marker: PhantomData,
        }
    }
}

// Manual implementations avoid imposing Copy/Clone bounds on the marker type T.
impl<T> Copy for Node<T> {}

impl<T> Clone for Node<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Cross-batch result retention metadata; no cache is implemented yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum CachePolicy {
    /// Do not retain memo entries across batches; batch sharing is still allowed.
    Transient,
    /// Retain the most recently used input combination and result.
    #[default]
    Last,
    /// Retain at most `capacity` historical input combinations.
    /// Zero retains no cross-batch memo entries, like `Transient`.
    Memo { capacity: usize },
}

/// A process-local graph identity independent of graph-local SlotMap keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GraphId(u64);

impl GraphId {
    pub(super) fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        // Only uniqueness is needed; this counter does not publish graph data.
        // Prevent wraparound so old handles cannot match a later graph.
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("Graph identity space exhausted");
        Self(id)
    }
}

/// An input edge recording source ownership and the type expected by the closure.
pub(super) struct Dependency {
    pub(super) graph_id: GraphId,
    pub(super) source: NodeId,
    pub(super) type_id: TypeId,
}

/// Distinguishes externally supplied inputs from computations over dependencies.
pub(super) enum NodeKind {
    Input,
    Computation {
        inputs: Box<[Dependency]>,
        computation: Box<ErasedComputation>,
    },
}

/// A heterogeneous node record whose output type matches its Node<T> marker.
pub(super) struct NodeRecord {
    pub(super) type_id: TypeId,
    /// Successful replacements of this node's computation definition.
    pub(super) revision: u64,
    pub(super) kind: NodeKind,
    pub(super) semantics: ValueSemantics,
    pub(super) cache_policy: CachePolicy,
}
