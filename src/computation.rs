//! Computation graph construction with typed node handles.
//!
//! [`Graph`] stores nodes, explicit dependencies, and computation closures.
//! [`Node<T>`] identifies a node's output type. [`Map::map`] registers a computation
//! without invoking its closure. This module currently supports graph construction;
//! evaluation is not yet implemented.
//!
//! Each graph starts with a [`Frame`] input node. Use it to build computations
//! with one or more inputs:
//!
//! ```
//! use gekijyou::computation::{Frame, Graph, Map};
//!
//! let mut graph = Graph::new();
//! let frame = graph.frame();
//! let position = frame.map(&mut graph, |f: &Frame| f.0 as f64 * 2.0);
//! let opacity = frame.map(&mut graph, |f: &Frame| if f.0 < 30 { 1.0 } else { 0.0 });
//! let state = (position, opacity).map(&mut graph, |x: &f64, alpha: &f64| (*x, *alpha));
//!
//! // Combining nodes copies their handles, so position can feed another branch.
//! let offset = position.map(&mut graph, |x: &f64| x + 10.0);
//! ```
//!
//! A handle belongs to the graph that created it. Passing a foreign handle to
//! `map` panics before inserting the node. Dependencies can only reference existing
//! nodes and cannot be rewritten, so construction preserves an acyclic graph.

use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use slotmap::{SlotMap, new_key_type};

/// An integer target frame number.
///
/// Carries no frame rate or time domain. The caller defines conversions to seconds
/// or local time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Frame(pub u64);

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
pub struct Node<T> {
    graph_id: GraphId,
    id: NodeId,
    // Marks the output type without owning T or inheriting its Send/Sync bounds.
    _marker: PhantomData<fn() -> T>,
}

impl<T> Node<T> {
    // Private construction prevents callers from forging types or graph ownership.
    fn new(graph_id: GraphId, id: NodeId) -> Self {
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

/// A graph storing heterogeneous nodes, dependencies, and computation closures.
///
/// Obtain the initial input with [`Graph::frame`] and add computations with
/// [`Map::map`]. Nodes can only depend on previously inserted nodes. The current
/// API does not support removing nodes or rewriting dependencies.
///
/// Graph identities validate handle ownership at runtime; they are not stable
/// object identifiers across project rebuilds.
pub struct Graph {
    id: GraphId,
    nodes: SlotMap<NodeId, NodeRecord>,
    frame: Node<Frame>,
}

impl Graph {
    /// Creates a graph containing only its frame input node.
    ///
    /// Each call allocates a distinct graph identity. [`Default::default`] has
    /// the same behavior.
    ///
    /// # Panics
    ///
    /// Panics if the process-local graph identity counter is exhausted, rather
    /// than reusing an existing identity.
    pub fn new() -> Self {
        let id = GraphId::new();
        let mut nodes = SlotMap::with_key();
        let frame = Node::new(
            id,
            nodes.insert(NodeRecord {
                type_id: TypeId::of::<Frame>(),
                kind: NodeKind::Input,
            }),
        );

        Self { id, nodes, frame }
    }

    /// Returns a handle to the frame input node.
    ///
    /// Returns the handle by value without retaining a graph borrow, allowing
    /// `graph.frame().map(&mut graph, |f: &Frame| f.0)`.
    /// Does not set or read an actual frame input value.
    pub fn frame(&self) -> Node<Frame> {
        self.frame
    }

    // Validate every dependency before insertion so rejection leaves no partial node.
    fn insert(&mut self, record: NodeRecord) -> NodeId {
        if let NodeKind::Computation { inputs, .. } = &record.kind {
            for input in inputs.iter() {
                // SlotMap keys are graph-local: check ownership before looking up the key.
                assert_eq!(
                    input.graph_id, self.id,
                    "Input node belongs to another graph"
                );
                let source = self
                    .nodes
                    .get(input.source)
                    .expect("Input node does not exist");

                assert_eq!(source.type_id, input.type_id, "Input type mismatch");
            }
        }

        self.nodes.insert(record)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Combines a node or a tuple of nodes into a new computation node.
///
/// Built-in implementations support a single [`Node<T>`] and tuples of one to four
/// nodes. The closure receives shared references to input values in tuple order
/// and returns an output value `O`. A node may appear more than once as an input.
///
/// Built-in implementations require input types to be `'static` and the output
/// and closure to be `Send + Sync + 'static`. Stored closures cannot capture
/// non-`'static` borrows. Use `move` to capture owned data; this does not extend the
/// lifetime of any references it contains. Copying handles does not require input
/// values to be copyable.
///
/// `Fn + Send + Sync` does not guarantee purity or determinism. For deterministic
/// evaluation at a target frame, model external inputs explicitly and avoid
/// closures that depend on untracked mutable state.
pub trait Map<F, O> {
    /// Registers dependencies and a closure in `graph`, returning the output handle.
    ///
    /// Does not invoke `f`. Input handles remain available for other branches.
    ///
    /// # Panics
    ///
    /// Built-in implementations panic if any input belongs to another graph.
    fn map(&self, graph: &mut Graph, f: F) -> Node<O>;
}

/// A process-local graph identity independent of graph-local SlotMap keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphId(u64);

impl GraphId {
    fn new() -> Self {
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
struct NodeInput {
    graph_id: GraphId,
    source: NodeId,
    type_id: TypeId,
}

/// Erased output storage; NodeRecord::type_id records the concrete output type.
type AnyValue = Box<dyn Any + Send + Sync + 'static>;

/// An erased computation whose arguments must match its input edges in order,
/// count, and type.
type AnyComputation = dyn Fn(&[&dyn Any]) -> AnyValue + Send + Sync + 'static;

/// Distinguishes externally supplied inputs from computations over dependencies.
enum NodeKind {
    Input,
    Computation {
        inputs: Box<[NodeInput]>,
        computation: Box<AnyComputation>,
    },
}

/// A heterogeneous node record whose output type matches its Node<T> marker.
struct NodeRecord {
    type_id: TypeId,
    kind: NodeKind,
}

macro_rules! impl_map {
    (@tuple $($T:ident $node:ident $arg:ident),+ $(,)?) => {
        // Record tuple dependencies and adapt the typed closure to erased storage.
        impl<F, O, $($T),+> Map<F, O> for ($(Node<$T>,)+)
        where
            $($T: 'static,)+
            F: Fn($(&$T),+) -> O + Send + Sync + 'static,
            O: Send + Sync + 'static,
        {
            fn map(&self, graph: &mut Graph, f: F) -> Node<O> {
                let ($($node,)+) = self;

                let inputs = Box::new([
                    $(
                        NodeInput {
                            graph_id: $node.graph_id,
                            source: $node.id,
                            type_id: TypeId::of::<$T>(),
                        },
                    )+
                ]);

                // Argument shape must match inputs; failure indicates an internal contract violation.
                let computation: Box<AnyComputation> =
                    Box::new(move |args: &[&dyn Any]| {
                        let [$($arg,)+] = args else {
                            panic!("Invalid input count");
                        };

                        $(
                            let $arg = (*$arg)
                                .downcast_ref::<$T>()
                                .expect("Input type mismatch");
                        )+

                        let output = f($($arg),+);
                        Box::new(output) as AnyValue
                    });

                let node_id = graph.insert(NodeRecord {
                    type_id: TypeId::of::<O>(),
                    kind: NodeKind::Computation {
                        inputs,
                        computation,
                    },
                });

                Node::new(graph.id, node_id)
            }
        }
    };

    ($T:ident $node:ident $arg:ident $(,)?) => {
        // Reuse the one-tuple implementation for identical validation and erasure.
        impl_map!(@tuple $T $node $arg);

        impl<F, O, $T> Map<F, O> for Node<$T>
        where
            $T: 'static,
            F: Fn(&$T) -> O + Send + Sync + 'static,
            O: Send + Sync + 'static,
        {
            fn map(&self, graph: &mut Graph, f: F) -> Node<O> {
                (*self,).map(graph, f)
            }
        }
    };

    // Multi-input entry point; tuple order determines closure argument order.
    ($($T:ident $node:ident $arg:ident),+ $(,)?) => {
        impl_map!(@tuple $($T $node $arg),+);
    };
}

impl_map!(A a va);
impl_map!(A a va, B b vb);
impl_map!(A a va, B b vb, C c vc);
impl_map!(A a va, B b vb, C c vc, D d vd);
