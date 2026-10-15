use super::{
    node::{Dependency, KernelId, Node},
    value::{
        BatchKernelRecord, ErasedBatchKernel, ErasedComputation, ErasedSingleBatch, SharedValue,
    },
};
use std::{
    any::{Any, TypeId},
    cell::Cell,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Monotonic slot assignment for the preferred scratch slot. Masked to the pool
/// size so more threads than slots simply start their scan elsewhere.
static NEXT_THREAD_SLOT: AtomicUsize = AtomicUsize::new(0);
std::thread_local! {
    /// `usize::MAX` marks a thread that has not picked a slot yet. `const`
    /// initialization is deliberate: a runtime-initialized TLS value registers a
    /// heap-allocated init/destructor node the first time each newly spawned
    /// worker thread touches it.
    static THREAD_SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Resolves this thread's preferred pool slot, assigning one on first use.
#[inline]
fn thread_slot() -> usize {
    THREAD_SLOT.with(|cell| {
        let slot = cell.get();
        if slot != usize::MAX {
            slot
        } else {
            let next = NEXT_THREAD_SLOT.fetch_add(1, Ordering::Relaxed) & 15;
            cell.set(next);
            next
        }
    })
}

/// A cache-line-aligned mutex-guarded reusable scratch buffer.
#[repr(align(64))]
struct ScratchSlot<T>(Mutex<Vec<T>>);

/// Fixed pool of per-thread scratch buffers.
///
/// At most one buffer per thread is in flight, and only a handful of threads
/// ever run a batch kernel concurrently, so a slot scan starting at the
/// thread's preferred slot always finds a free pre-allocated slot in one or two
/// steps. Locks are always tried, never waited on: a contended or reentrant
/// borrow simply continues the scan, and a fully saturated pool falls back to a
/// fresh `Vec`.
struct BatchScratchPool<T>([ScratchSlot<T>; 16]);

impl<T> BatchScratchPool<T> {
    fn new() -> Self {
        Self(std::array::from_fn(|_| {
            ScratchSlot(Mutex::new(Vec::with_capacity(64)))
        }))
    }

    /// Takes a scratch buffer plus the slot it came from. The returned index must
    /// be passed back to [`Self::put`] so the buffer is returned to a pool slot.
    fn take(&self, min_cap: usize) -> (usize, Vec<T>) {
        let start = thread_slot();
        for offset in 0..16 {
            let idx = (start + offset) & 15;
            if let Ok(mut guard) = self.0[idx].0.try_lock() {
                if guard.capacity() >= min_cap || (std::mem::size_of::<T>() == 0 && min_cap == 0) {
                    return (idx, std::mem::take(&mut *guard));
                }
            }
        }
        (start, Vec::with_capacity(min_cap.max(64)))
    }

    /// Returns a scratch buffer to `idx`, or to any free slot if that one is busy.
    /// Oversized buffers are dropped rather than pinned in the pool.
    fn put(&self, idx: usize, mut vec: Vec<T>) {
        vec.clear();
        if vec.capacity() <= 64 || std::mem::size_of::<T>() == 0 {
            if let Ok(mut guard) = self.0[idx].0.try_lock() {
                if guard.capacity() == 0 {
                    *guard = vec;
                    return;
                }
            }
            for slot in &self.0 {
                if let Ok(mut guard) = slot.0.try_lock() {
                    if guard.capacity() == 0 {
                        *guard = vec;
                        return;
                    }
                }
            }
        }
    }
}

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
    pub(crate) single: Arc<ErasedSingleBatch>,
    // Require the exact erased parameter, input shape, and output type.
    marker: PhantomData<fn(P, I, O) -> (P, I, O)>,
}
impl<P, I, O> Clone for BatchKernel<P, I, O> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            kernel: self.kernel.clone(),
            single: self.single.clone(),
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
        let single_fn = kernel.single.clone();
        let param_for_single = param.clone();
        let computation: Arc<ErasedComputation> =
            Arc::new(move |args: &[&dyn Any]| (single_fn)(param_for_single.as_ref(), args));
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
        let f = Arc::new(f);
        let scratch = Arc::new(BatchScratchPool::<O>::new());
        let single_fn: Arc<ErasedSingleBatch> = {
            let f = f.clone();
            let scratch = scratch.clone();
            Arc::new(move |param: &dyn Any, args: &[&dyn Any]| {
                assert!(args.is_empty(), "Invalid input count");
                let param = param
                    .downcast_ref::<P>()
                    .expect("Batch parameter type mismatch");
                let items = [param];
                let (slot, mut outputs) = scratch.take(1);
                f(&items, &mut outputs);
                let actual_len = outputs.len();
                if actual_len != 1 {
                    scratch.put(slot, outputs);
                    panic!(
                        "Batch kernel single output count mismatch: expected 1, got {}",
                        actual_len
                    );
                }
                let value =
                    Arc::new(outputs.pop().expect("Batch kernel single output")) as SharedValue;
                scratch.put(slot, outputs);
                value
            })
        };
        let kernel: Arc<ErasedBatchKernel> = Arc::new(
            move |batch: &[(&dyn Any, &[&dyn Any])],
                  out: &mut smallvec::SmallVec<[SharedValue; 64]>| {
                let params: smallvec::SmallVec<[&P; 64]> = batch
                    .iter()
                    .map(|(param, args)| {
                        assert!(args.is_empty(), "Invalid input count");
                        param
                            .downcast_ref::<P>()
                            .expect("Batch parameter type mismatch")
                    })
                    .collect();
                let (slot, mut outputs) = scratch.take(batch.len());
                f(&params, &mut outputs);
                let actual_len = outputs.len();
                if actual_len != batch.len() {
                    scratch.put(slot, outputs);
                    panic!(
                        "Batch kernel output count mismatch: expected {}, got {}",
                        batch.len(),
                        actual_len
                    );
                }
                out.reserve(batch.len());
                for o in outputs.drain(..) {
                    out.push(Arc::new(o) as SharedValue);
                }
                scratch.put(slot, outputs);
            },
        );
        BatchKernel {
            id: KernelId::new(),
            kernel,
            single: single_fn,
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
                let f = Arc::new(f);
                let scratch = Arc::new(BatchScratchPool::<O>::new());
                let single_fn: Arc<ErasedSingleBatch> = {
                    let f = f.clone();
                    let scratch = scratch.clone();
                    Arc::new(move |param: &dyn Any, args: &[&dyn Any]| {
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
                        let items = [(param, $($node),+)];
                        let (slot, mut outputs) = scratch.take(1);
                        f(&items, &mut outputs);
                        let actual_len = outputs.len();
                        if actual_len != 1 {
                            scratch.put(slot, outputs);
                            panic!(
                                "Batch kernel single output count mismatch: expected 1, got {}",
                                actual_len
                            );
                        }
                        let value = Arc::new(outputs.pop().expect("Batch kernel single output"))
                            as SharedValue;
                        scratch.put(slot, outputs);
                        value
                    })
                };
                let kernel: Arc<ErasedBatchKernel> = Arc::new(
                    move |batch: &[(&dyn Any, &[&dyn Any])],
                          out: &mut smallvec::SmallVec<[SharedValue; 64]>| {
                        let items: smallvec::SmallVec<[(&P, $(&$T),+); 64]> = batch
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
                        let (slot, mut outputs) = scratch.take(batch.len());
                        f(&items, &mut outputs);
                        let actual_len = outputs.len();
                        if actual_len != batch.len() {
                            scratch.put(slot, outputs);
                            panic!(
                                "Batch kernel output count mismatch: expected {}, got {}",
                                batch.len(),
                                actual_len
                            );
                        }
                        out.reserve(batch.len());
                        for o in outputs.drain(..) {
                            out.push(Arc::new(o) as SharedValue);
                        }
                        scratch.put(slot, outputs);
                    },
                );
                BatchKernel {
                    id: KernelId::new(),
                    kernel,
                    single: single_fn,
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
                let single_fn = kernel.single.clone();
                let param_for_single = param.clone();
                let computation: Arc<ErasedComputation> = Arc::new(move |args: &[&dyn Any]| {
                    (single_fn)(param_for_single.as_ref(), args)
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
                let f = Arc::new(f);
                let scratch = Arc::new(BatchScratchPool::<O>::new());
                let single_fn: Arc<ErasedSingleBatch> = {
                    let f = f.clone();
                    let scratch = scratch.clone();
                    Arc::new(move |param: &dyn Any, args: &[&dyn Any]| {
                        let [$node] = args else {
                            panic!("Invalid input count");
                        };
                        let $node = (*$node).downcast_ref::<$T>().expect("Input type mismatch");
                        let param = param
                            .downcast_ref::<P>()
                            .expect("Batch parameter type mismatch");
                        let items = [(param, $node)];
                        let (slot, mut outputs) = scratch.take(1);
                        f(&items, &mut outputs);
                        let actual_len = outputs.len();
                        if actual_len != 1 {
                            scratch.put(slot, outputs);
                            panic!(
                                "Batch kernel single output count mismatch: expected 1, got {}",
                                actual_len
                            );
                        }
                        let value = Arc::new(outputs.pop().expect("Batch kernel single output"))
                            as SharedValue;
                        scratch.put(slot, outputs);
                        value
                    })
                };
                let kernel: Arc<ErasedBatchKernel> = Arc::new(
                    move |batch: &[(&dyn Any, &[&dyn Any])],
                          out: &mut smallvec::SmallVec<[SharedValue; 64]>| {
                        let items: smallvec::SmallVec<[(&P, &$T); 64]> = batch
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
                        let (slot, mut outputs) = scratch.take(batch.len());
                        f(&items, &mut outputs);
                        let actual_len = outputs.len();
                        if actual_len != batch.len() {
                            scratch.put(slot, outputs);
                            panic!(
                                "Batch kernel output count mismatch: expected {}, got {}",
                                batch.len(),
                                actual_len
                            );
                        }
                        out.reserve(batch.len());
                        for o in outputs.drain(..) {
                            out.push(Arc::new(o) as SharedValue);
                        }
                        scratch.put(slot, outputs);
                    },
                );
                BatchKernel {
                    id: KernelId::new(),
                    kernel,
                    single: single_fn,
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
                let single_fn = kernel.single.clone();
                let param_for_single = param.clone();
                let computation: Arc<ErasedComputation> = Arc::new(move |args: &[&dyn Any]| {
                    (single_fn)(param_for_single.as_ref(), args)
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
