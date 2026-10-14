use super::{
    node::{Dependency, KernelId, Node},
    value::{BatchKernelRecord, ErasedBatchKernel, ErasedComputation, SharedValue},
};
use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    sync::Arc,
};

/// Ordered typed dependencies: `()`, `Node<T>`, or tuples of one to twelve Nodes.
/// Custom adapters delegate to these shapes, preserving the erased output type.
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
/// use gekijyou_computation::{ComputationDefinition, ComputationInputs};
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

    /// Homogeneous slice building: one dependency per input node, in order.
    /// Empty slices are valid and produce a frame-independent zero-input node.
    pub(super) fn from_slice<T, F>(nodes: &[Node<T>], f: F) -> Self
    where
        T: 'static,
        F: Fn(&[&T]) -> O + Send + Sync + 'static,
        O: Send + Sync + 'static,
    {
        let expected_len = nodes.len();
        let inputs: Box<[Dependency]> = nodes
            .iter()
            .map(|node| Dependency {
                graph_id: node.graph_id,
                source: node.id,
                type_id: TypeId::of::<T>(),
            })
            .collect();
        let computation: Box<ErasedComputation> = Box::new(move |args: &[&dyn Any]| {
            assert_eq!(args.len(), expected_len, "Invalid input count");
            let refs: smallvec::SmallVec<[&T; 8]> = args
                .iter()
                .map(|arg| (*arg).downcast_ref::<T>().expect("Input type mismatch"))
                .collect();
            Arc::new(f(&refs)) as SharedValue
        });
        Self::new(inputs, computation)
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
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf, G g vg);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf, G g vg, H h vh);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf, G g vg, H h vh, I i vi);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf, G g vg, H h vh, I i vi, J j vj);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf, G g vg, H h vh, I i vi, J j vj, K k vk);
impl_computation_inputs!(A a va, B b vb, C c vc, D d vd, E e ve, F_ f vf, G g vg, H h vh, I i vi, J j vj, K k vk, L l vl);

/// Reusable vectorized computation kernel with per-node parameter `P`,
/// input shape `I`, and output type `O`.
///
/// The closure receives one `(parameter, inputs)` pair per ready node and must
/// push exactly one output per pair, in order.
///
/// ```compile_fail
/// use gekijyou_computation::{BatchKernel, Node};
/// fn relabel(
///     k: BatchKernel<u64, Node<u64>, fn(&str)>,
/// ) -> BatchKernel<u64, Node<u64>, fn(&'static str)> {
///     k
/// }
/// ```
pub struct BatchKernel<P, I, O> {
    pub(crate) id: KernelId,
    pub(crate) kernel: Arc<ErasedBatchKernel>,
    // Require the exact erased parameter, input shape, and output type.
    marker: PhantomData<fn(P, I, O) -> (P, I, O)>,
}
impl<P, I, O> Clone for BatchKernel<P, I, O> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            kernel: self.kernel.clone(),
            marker: PhantomData,
        }
    }
}
impl<P, I, O> std::fmt::Debug for BatchKernel<P, I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchKernel").field("id", &self.id).finish()
    }
}
impl<P, I, O> BatchKernel<P, I, O> {
    pub fn id(&self) -> KernelId {
        self.id
    }
    /// Compiles a vectorized closure into a reusable kernel for shape `I`.
    pub fn new<F>(f: F) -> Self
    where
        I: BatchInputs<P, F, O>,
    {
        I::build_kernel(f)
    }
}

/// An opaque inserted batch node: ordered inputs, single-item fallback
/// computation, and the reusable kernel record.
///
/// Built through [`BatchInputNodes::build_batch_node`]. Its fields are private to
/// the computation module, preserving the typed output contract across erasure.
pub struct BatchNodeDefinition<O> {
    inputs: Box<[Dependency]>,
    computation: Arc<ErasedComputation>,
    batch: BatchKernelRecord,
    // Require the exact erased output type without owning O or inheriting its auto traits.
    _marker: PhantomData<fn(O) -> O>,
}
impl<O> BatchNodeDefinition<O> {
    pub(super) fn new(
        inputs: Box<[Dependency]>,
        computation: Arc<ErasedComputation>,
        batch: BatchKernelRecord,
    ) -> Self {
        Self {
            inputs,
            computation,
            batch,
            _marker: PhantomData,
        }
    }

    pub(super) fn into_parts(
        self,
    ) -> (Box<[Dependency]>, Arc<ErasedComputation>, BatchKernelRecord) {
        (self.inputs, self.computation, self.batch)
    }
}

/// Input shapes accepted by [`BatchKernel`]: `()`, `Node<A>`, or tuples.
pub trait BatchInputNodes<P, O>: Sized {
    /// Builds a single-node fallback plus the reusable batch record.
    fn build_batch_node(
        &self,
        kernel: &BatchKernel<P, Self, O>,
        param: P,
    ) -> BatchNodeDefinition<O>;
}

/// Input shapes whose vectorized closure can be compiled into a [`BatchKernel`].
pub trait BatchInputs<P, F, O>: BatchInputNodes<P, O> {
    fn build_kernel(f: F) -> BatchKernel<P, Self, O>;
}

impl<P, O> BatchInputNodes<P, O> for ()
where
    P: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    fn build_batch_node(
        &self,
        kernel: &BatchKernel<P, Self, O>,
        param: P,
    ) -> BatchNodeDefinition<O> {
        let param: SharedValue = Arc::new(param);
        let kernel_fn = kernel.kernel.clone();
        let param_for_single = param.clone();
        let computation: Arc<ErasedComputation> = Arc::new(move |args: &[&dyn Any]| {
            let items = [(param_for_single.as_ref() as &dyn Any, args)];
            let mut outputs = (kernel_fn)(&items);
            outputs.pop().expect("Batch kernel single output")
        });
        BatchNodeDefinition::new(
            Box::new([]),
            computation,
            BatchKernelRecord {
                id: kernel.id,
                param,
                kernel: kernel.kernel.clone(),
            },
        )
    }
}

impl<P, F, O> BatchInputs<P, F, O> for ()
where
    P: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(&[&P], &mut Vec<O>) + Send + Sync + 'static,
{
    fn build_kernel(f: F) -> BatchKernel<P, Self, O> {
        let kernel: Arc<ErasedBatchKernel> =
            Arc::new(move |batch: &[(&dyn Any, &[&dyn Any])]| {
                let params: smallvec::SmallVec<[&P; 16]> = batch
                    .iter()
                    .map(|(param, args)| {
                        assert!(args.is_empty(), "Invalid input count");
                        param
                            .downcast_ref::<P>()
                            .expect("Batch parameter type mismatch")
                    })
                    .collect();
                let mut outputs = Vec::with_capacity(batch.len());
                f(&params, &mut outputs);
                assert_eq!(
                    outputs.len(),
                    batch.len(),
                    "Batch kernel output count mismatch"
                );
                outputs
                    .into_iter()
                    .map(|o| Arc::new(o) as SharedValue)
                    .collect()
            });
        BatchKernel {
            id: KernelId::new(),
            kernel,
            marker: PhantomData,
        }
    }
}

macro_rules! impl_batch_inputs {
    (@tuple $($T:ident $node:ident),+ $(,)?) => {
        impl<P, F, O, $($T),+> BatchInputs<P, F, O> for ($(Node<$T>,)+)
        where
            P: Send + Sync + 'static,
            O: Send + Sync + 'static,
            $($T: 'static,)+
            F: Fn(&[(&P, $(&$T),+)], &mut Vec<O>) + Send + Sync + 'static,
        {
            fn build_kernel(f: F) -> BatchKernel<P, Self, O> {
                let kernel: Arc<ErasedBatchKernel> =
                    Arc::new(move |batch: &[(&dyn Any, &[&dyn Any])]| {
                        let items: smallvec::SmallVec<[(&P, $(&$T),+); 16]> = batch
                            .iter()
                            .map(|(param, args)| {
                                let [$($node,)+] = args else {
                                    panic!("Invalid input count");
                                };
                                $(
                                    let $node = (*$node)
                                        .downcast_ref::<$T>()
                                        .expect("Input type mismatch");
                                )+
                                let param = param
                                    .downcast_ref::<P>()
                                    .expect("Batch parameter type mismatch");
                                (param, $($node),+)
                            })
                            .collect();
                        let mut outputs = Vec::with_capacity(batch.len());
                        f(&items, &mut outputs);
                        assert_eq!(
                            outputs.len(),
                            batch.len(),
                            "Batch kernel output count mismatch"
                        );
                        outputs
                            .into_iter()
                            .map(|o| Arc::new(o) as SharedValue)
                            .collect()
                    });
                BatchKernel {
                    id: KernelId::new(),
                    kernel,
                    marker: PhantomData,
                }
            }
        }

        impl<P, O, $($T),+> BatchInputNodes<P, O> for ($(Node<$T>,)+)
        where
            P: Send + Sync + 'static,
            O: Send + Sync + 'static,
            $($T: 'static,)+
        {
            fn build_batch_node(
                &self,
                kernel: &BatchKernel<P, Self, O>,
                param: P,
            ) -> BatchNodeDefinition<O> {
                let ($($node,)+) = self;
                let inputs: Box<[Dependency]> = Box::new([
                    $(Dependency {
                        graph_id: $node.graph_id,
                        source: $node.id,
                        type_id: TypeId::of::<$T>(),
                    },)+
                ]);
                let param: SharedValue = Arc::new(param);
                let kernel_fn = kernel.kernel.clone();
                let param_for_single = param.clone();
                let computation: Arc<ErasedComputation> =
                    Arc::new(move |args: &[&dyn Any]| {
                        let items = [(param_for_single.as_ref() as &dyn Any, args)];
                        let mut outputs = (kernel_fn)(&items);
                        outputs.pop().expect("Batch kernel single output")
                    });
                BatchNodeDefinition::new(
                    inputs,
                    computation,
                    BatchKernelRecord {
                        id: kernel.id,
                        param,
                        kernel: kernel.kernel.clone(),
                    },
                )
            }
        }
    };

    ($T:ident $node:ident $(,)?) => {
        impl_batch_inputs!(@tuple $T $node);

        impl<P, F, O, $T> BatchInputs<P, F, O> for Node<$T>
        where
            P: Send + Sync + 'static,
            O: Send + Sync + 'static,
            $T: 'static,
            F: Fn(&[(&P, &$T)], &mut Vec<O>) + Send + Sync + 'static,
        {
            fn build_kernel(f: F) -> BatchKernel<P, Self, O> {
                let kernel: Arc<ErasedBatchKernel> =
                    Arc::new(move |batch: &[(&dyn Any, &[&dyn Any])]| {
                        let items: smallvec::SmallVec<[(&P, &$T); 16]> = batch
                            .iter()
                            .map(|(param, args)| {
                                let [$node] = args else {
                                    panic!("Invalid input count");
                                };
                                let $node = (*$node)
                                    .downcast_ref::<$T>()
                                    .expect("Input type mismatch");
                                let param = param
                                    .downcast_ref::<P>()
                                    .expect("Batch parameter type mismatch");
                                (param, $node)
                            })
                            .collect();
                        let mut outputs = Vec::with_capacity(batch.len());
                        f(&items, &mut outputs);
                        assert_eq!(
                            outputs.len(),
                            batch.len(),
                            "Batch kernel output count mismatch"
                        );
                        outputs
                            .into_iter()
                            .map(|o| Arc::new(o) as SharedValue)
                            .collect()
                    });
                BatchKernel {
                    id: KernelId::new(),
                    kernel,
                    marker: PhantomData,
                }
            }
        }

        impl<P, O, $T> BatchInputNodes<P, O> for Node<$T>
        where
            P: Send + Sync + 'static,
            O: Send + Sync + 'static,
            $T: 'static,
        {
            fn build_batch_node(
                &self,
                kernel: &BatchKernel<P, Self, O>,
                param: P,
            ) -> BatchNodeDefinition<O> {
                let node = *self;
                let inputs: Box<[Dependency]> = Box::new([Dependency {
                    graph_id: node.graph_id,
                    source: node.id,
                    type_id: TypeId::of::<$T>(),
                }]);
                let param: SharedValue = Arc::new(param);
                let kernel_fn = kernel.kernel.clone();
                let param_for_single = param.clone();
                let computation: Arc<ErasedComputation> =
                    Arc::new(move |args: &[&dyn Any]| {
                        let items = [(param_for_single.as_ref() as &dyn Any, args)];
                        let mut outputs = (kernel_fn)(&items);
                        outputs.pop().expect("Batch kernel single output")
                    });
                BatchNodeDefinition::new(
                    inputs,
                    computation,
                    BatchKernelRecord {
                        id: kernel.id,
                        param,
                        kernel: kernel.kernel.clone(),
                    },
                )
            }
        }
    };

    ($($T:ident $node:ident),+ $(,)?) => {
        impl_batch_inputs!(@tuple $($T $node),+);
    };
}

impl_batch_inputs!(A a);
impl_batch_inputs!(A a, B b);
impl_batch_inputs!(A a, B b, C c);
impl_batch_inputs!(A a, B b, C c, D d);
impl_batch_inputs!(A a, B b, C c, D d, E e);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f, G g);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f, G g, H h);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f, G g, H h, I i);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f, G g, H h, I i, J j);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f, G g, H h, I i, J j, K k);
impl_batch_inputs!(A a, B b, C c, D d, E e, F_ f, G g, H h, I i, J j, K k, L l);
