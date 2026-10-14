use super::value::{ErasedComputation, ValueSemantics};
use std::{
    any::TypeId,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Integer target frame; time conversion belongs to the animation runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameIndex(pub u64);

pub(crate) fn next_id(counter: &AtomicU64) -> u64 {
    counter
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("Identity space exhausted")
}

/// Process-local identity of one logical graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphId(u64);
impl GraphId {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(next_id(&NEXT))
    }
}

/// Non-reused identity within a graph, allocated even for failed transactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u64);

/// Monotonic published graph definition version; ordinary inputs do not advance it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GraphDefinitionEpoch(pub(crate) u64);
impl GraphDefinitionEpoch {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Copyable invariant typed handle. Copy does not require `T: Copy` or retain values.
///
/// ```compile_fail
/// use gekijyou_computation::Node;
/// fn relabel(x: Node<fn(&str)>) -> Node<fn(&'static str)> { x }
/// ```
pub struct Node<T> {
    pub(crate) graph_id: GraphId,
    pub(crate) id: NodeId,
    marker: PhantomData<fn(T) -> T>,
}
impl<T> Node<T> {
    pub(crate) fn new(graph_id: GraphId, id: NodeId) -> Self {
        Self {
            graph_id,
            id,
            marker: PhantomData,
        }
    }
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn graph_id(self) -> GraphId {
        self.graph_id
    }
}
impl<T> Copy for Node<T> {}
impl<T> Clone for Node<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.graph_id == other.graph_id && self.id == other.id
    }
}
impl<T> Eq for Node<T> {}
impl<T> Hash for Node<T> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.graph_id.hash(h);
        self.id.hash(h);
    }
}
impl<T> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("graph", &self.graph_id)
            .field("id", &self.id)
            .finish()
    }
}

/// Writable ordinary external input. A computation Node cannot be converted into Input.
///
/// ```compile_fail
/// use gekijyou_computation::Input;
/// fn relabel(x: Input<fn(&str)>) -> Input<fn(&'static str)> { x }
/// ```
pub struct Input<T>(pub(crate) Node<T>);
impl<T> Input<T> {
    pub fn node(self) -> Node<T> {
        self.0
    }
}
impl<T> Copy for Input<T> {}
impl<T> Clone for Input<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for Input<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Persistent retention; Off still permits temporary sharing in a single evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    Off,
    #[default]
    Last,
    /// Capacity must be nonzero at the public configuration boundary.
    Lru {
        capacity: usize,
    },
}
impl CachePolicy {
    pub(crate) fn capacity(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Last => 1,
            Self::Lru { capacity } => capacity,
        }
    }
}

/// Independent evaluator-owned policies for direct memo and root-input caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeCacheConfig {
    pub memo: CachePolicy,
    pub context: CachePolicy,
}
impl Default for NodeCacheConfig {
    fn default() -> Self {
        Self {
            memo: CachePolicy::Last,
            context: CachePolicy::Off,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Dependency {
    pub(crate) graph_id: GraphId,
    pub(crate) source: NodeId,
    pub(crate) type_id: TypeId,
}
#[derive(Clone)]
pub(crate) enum NodeKind {
    Frame,
    Input,
    Computation {
        inputs: Arc<[Dependency]>,
        computation: Arc<ErasedComputation>,
    },
}
#[derive(Clone)]
pub(crate) struct NodeRecord {
    pub(crate) type_id: TypeId,
    pub(crate) revision: u64,
    pub(crate) kind: NodeKind,
    pub(crate) semantics: ValueSemantics,
    pub(crate) depends_on_frame: bool,
}
