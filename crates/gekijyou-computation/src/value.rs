use super::node::KernelId;
use std::{any::Any, sync::Arc};

/// Erased output storage; NodeRecord::type_id records the concrete output type.
pub type SharedValue = Arc<dyn Any + Send + Sync + 'static>;

/// An erased computation whose arguments must match its input edges in order,
/// count, and type.
pub(super) type ErasedComputation = dyn Fn(&[&dyn Any]) -> SharedValue + Send + Sync + 'static;

/// Erased vectorized batch computation: one parameter and ordered argument slice
/// per element, producing one output per element in the same order.
pub(crate) type ErasedBatchKernel = dyn Fn(&[(&dyn Any, &[&dyn Any])], &mut smallvec::SmallVec<[SharedValue; 64]>)
    + Send
    + Sync
    + 'static;

/// Batch kernel identity plus the per-node parameter captured at insertion.
#[derive(Clone)]
pub(crate) struct BatchKernelRecord {
    pub(crate) id: KernelId,
    pub(crate) param: SharedValue,
    pub(crate) kernel: Arc<ErasedBatchKernel>,
}

type Equivalence = dyn Fn(&dyn Any, &dyn Any) -> bool + Send + Sync + 'static;

#[derive(Clone, Default)]
pub(super) struct ValueSemantics {
    equivalence: Option<Arc<Equivalence>>,
}

impl ValueSemantics {
    pub(super) fn by_eq<T: Eq + 'static>() -> Self {
        Self::by_equivalence::<T, _>(|lhs, rhs| lhs == rhs)
    }

    pub(super) fn by_equivalence<T: 'static, F>(equivalent: F) -> Self
    where
        F: Fn(&T, &T) -> bool + Send + Sync + 'static,
    {
        Self {
            equivalence: Some(Arc::new(move |lhs, rhs| {
                let lhs = lhs
                    .downcast_ref::<T>()
                    .expect("Equivalence input type mismatch");
                let rhs = rhs
                    .downcast_ref::<T>()
                    .expect("Equivalence input type mismatch");
                equivalent(lhs, rhs)
            })),
        }
    }

    pub(super) fn equivalent(&self, lhs: &SharedValue, rhs: &SharedValue) -> bool {
        Arc::ptr_eq(lhs, rhs)
            || self
                .equivalence
                .as_ref()
                .is_some_and(|equivalent| equivalent(lhs.as_ref(), rhs.as_ref()))
    }
}
