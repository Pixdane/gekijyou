use super::{
    node::{Dependency, Node},
    value::{ErasedComputation, SharedValue},
};
use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    sync::Arc,
};

/// Input handles that can be combined with `F` to produce output `O`.
///
/// Built-in implementations support `()`, a single [`Node<T>`], and tuples of one
/// to four nodes. Custom input wrappers can delegate to these implementations.
/// [`Graph::map`](super::Graph::map) and [`Graph::replace`](super::Graph::replace)
/// share the same construction step.
///
/// Custom adapters wrap the built-in input shapes by delegating definition
/// construction. The erased representation stays private so adapters preserve
/// the declared input and output types:
///
/// ```
/// use gekijyou_computation::{ComputationDefinition, ComputationInputs, Graph, Node};
///
/// struct Position(Node<i64>, Node<i64>);
///
/// impl<F, O> ComputationInputs<F, O> for Position
/// where
///     F: Fn(&i64, &i64) -> O + Send + Sync + 'static,
///     O: Send + Sync + 'static,
/// {
///     fn build_computation(&self, f: F) -> ComputationDefinition<O> {
///         (self.0, self.1).build_computation(f)
///     }
/// }
///
/// let mut graph = Graph::new();
/// let x = graph.constant(10_i64);
/// let y = graph.constant(20_i64);
/// let point = graph.map(Position(x, y), |x: &i64, y: &i64| (*x, *y));
/// ```
pub trait ComputationInputs<F, O> {
    /// Builds a definition without executing the closure or modifying a graph.
    /// Handle ownership and dependencies are checked when the graph inserts or
    /// replaces the definition. Implementations must uphold the module's purity
    /// and value contracts.
    fn build_computation(&self, f: F) -> ComputationDefinition<O>;
}

/// An opaque computation definition with ordered inputs and output type `O`.
///
/// Built through [`ComputationInputs::build_computation`]. Its fields and constructor
/// are private to the computation module, preserving the typed output contract
/// across closure erasure. Creating a definition does not execute the closure.
///
/// The output type is invariant: custom adapters cannot relabel an erased output
/// through function-pointer subtyping.
///
/// ```compile_fail
/// use gekijyou_computation::{ComputationDefinition, ComputationInputs, Graph};
///
/// type Callback = fn(&str);
/// type StaticCallback = fn(&'static str);
/// fn callback(_: &str) {}
///
/// struct RelabeledInputs;
/// impl ComputationInputs<(), StaticCallback> for RelabeledInputs {
///     fn build_computation(&self, _: ()) -> ComputationDefinition<StaticCallback> {
///         let definition: ComputationDefinition<Callback> =
///             ().build_computation(|| callback as Callback);
///         definition
///     }
/// }
///
/// let mut graph = Graph::new();
/// let node = graph.map(RelabeledInputs, ());
/// graph.replace(node, RelabeledInputs, ()).unwrap();
/// graph.validate().unwrap();
/// ```
pub struct ComputationDefinition<O> {
    inputs: Box<[Dependency]>,
    computation: Box<ErasedComputation>,
    // Require the exact erased output type without owning O or inheriting its auto traits.
    _marker: PhantomData<fn(O) -> O>,
}

impl<O> ComputationDefinition<O> {
    // Only internal typed adapters construct definitions; the erased closure returns O.
    pub(super) fn new(inputs: Box<[Dependency]>, computation: Box<ErasedComputation>) -> Self {
        Self {
            inputs,
            computation,
            _marker: PhantomData,
        }
    }

    pub(super) fn into_parts(self) -> (Box<[Dependency]>, Box<ErasedComputation>) {
        (self.inputs, self.computation)
    }
}

impl<F, O> ComputationInputs<F, O> for ()
where
    F: Fn() -> O + Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    fn build_computation(&self, f: F) -> ComputationDefinition<O> {
        let computation: Box<ErasedComputation> = Box::new(move |args| {
            assert!(args.is_empty(), "Invalid input count");
            Arc::new(f()) as SharedValue
        });
        ComputationDefinition::new(Box::new([]), computation)
    }
}

macro_rules! impl_computation_inputs {
    (@tuple $($T:ident $node:ident $arg:ident),+ $(,)?) => {
        impl<F, O, $($T),+> ComputationInputs<F, O> for ($(Node<$T>,)+)
        where
            $($T: 'static,)+
            F: Fn($(&$T),+) -> O + Send + Sync + 'static,
            O: Send + Sync + 'static,
        {
            fn build_computation(&self, f: F) -> ComputationDefinition<O> {
                let ($($node,)+) = self;
                let inputs: Box<[Dependency]> = Box::new([
                    $(Dependency {
                        graph_id: $node.graph_id,
                        source: $node.id,
                        type_id: TypeId::of::<$T>(),
                    },)+
                ]);
                // Preserve argument order and multiplicity at the type erasure boundary.
                let computation: Box<ErasedComputation> = Box::new(move |args: &[&dyn Any]| {
                    let [$($arg,)+] = args else {
                        panic!("Invalid input count");
                    };
                    $(let $arg = (*$arg).downcast_ref::<$T>().expect("Input type mismatch");)+
                    Arc::new(f($($arg),+)) as SharedValue
                });
                ComputationDefinition::new(inputs, computation)
            }
        }
    };

    ($T:ident $node:ident $arg:ident $(,)?) => {
        impl_computation_inputs!(@tuple $T $node $arg);

        impl<F, O, $T> ComputationInputs<F, O> for Node<$T>
        where
            $T: 'static,
            F: Fn(&$T) -> O + Send + Sync + 'static,
            O: Send + Sync + 'static,
        {
            fn build_computation(&self, f: F) -> ComputationDefinition<O> {
                (*self,).build_computation(f)
            }
        }
    };

    ($($T:ident $node:ident $arg:ident),+ $(,)?) => {
        impl_computation_inputs!(@tuple $($T $node $arg),+);
    };
}

impl_computation_inputs!(A a va);
impl_computation_inputs!(A a va, B b vb);
impl_computation_inputs!(A a va, B b vb, C c vc);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd);
