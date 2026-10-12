use std::{any::Any, sync::Arc};

/// Erased output storage; NodeRecord::type_id records the concrete output type.
pub(super) type SharedValue = Arc<dyn Any + Send + Sync + 'static>;

/// An erased computation whose arguments must match its input edges in order,
/// count, and type.
pub(super) type ErasedComputation = dyn Fn(&[&dyn Any]) -> SharedValue + Send + Sync + 'static;

type Equality = dyn Fn(&dyn Any, &dyn Any) -> bool + Send + Sync + 'static;

#[derive(Clone, Default)]
pub(super) struct ValueSemantics {
    eq: Option<Arc<Equality>>,
}

impl ValueSemantics {
    pub(super) fn by_eq<T: Eq + 'static>() -> Self {
        Self {
            eq: Some(Arc::new(|lhs, rhs| {
                let lhs = lhs
                    .downcast_ref::<T>()
                    .expect("Equality input type mismatch");
                let rhs = rhs
                    .downcast_ref::<T>()
                    .expect("Equality input type mismatch");
                lhs == rhs
            })),
        }
    }

    pub(super) fn equivalent(&self, lhs: &SharedValue, rhs: &SharedValue) -> bool {
        Arc::ptr_eq(lhs, rhs)
            || self
                .eq
                .as_ref()
                .is_some_and(|eq| eq(lhs.as_ref(), rhs.as_ref()))
    }
}
