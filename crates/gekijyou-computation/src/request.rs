use super::{
    EvalError, FrameIndex, Node, SnapshotId,
    node::{GraphId, NodeId, next_id},
    value::SharedValue,
};
use std::{
    any::TypeId,
    fmt,
    marker::PhantomData,
    sync::{Arc, atomic::AtomicU64},
};
/// Opaque identity shared by one request batch and its completed results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestBatchId(u64);
impl RequestBatchId {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(next_id(&NEXT))
    }
}

/// Typed result identity tied to one batch. Copies never copy the output.
///
/// ```compile_fail
/// use gekijyou_computation::RequestId;
/// fn relabel(x: RequestId<fn(&str)>) -> RequestId<fn(&'static str)> { x }
/// ```
pub struct RequestId<T> {
    batch: RequestBatchId,
    index: usize,
    marker: PhantomData<fn(T) -> T>,
}
impl<T> Copy for RequestId<T> {}
impl<T> Clone for RequestId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for RequestId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestId")
            .field("batch", &self.batch)
            .field("index", &self.index)
            .finish()
    }
}
pub(crate) struct Request {
    pub(crate) frame: FrameIndex,
    pub(crate) graph: GraphId,
    pub(crate) node: NodeId,
    pub(crate) ty: TypeId,
}

/// Ordered Frame/Node requests; one bound scope supplies all ordinary inputs.
pub struct RequestBatch {
    id: RequestBatchId,
    pub(crate) requests: Vec<Request>,
}
impl RequestBatch {
    pub fn new() -> Self {
        Self {
            id: RequestBatchId::new(),
            requests: Vec::new(),
        }
    }
    pub fn id(&self) -> RequestBatchId {
        self.id
    }
    pub fn at(&mut self, frame: FrameIndex) -> FrameRequests<'_> {
        FrameRequests { batch: self, frame }
    }
    pub fn len(&self) -> usize {
        self.requests.len()
    }
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
    pub(crate) fn finish(
        self,
        snapshot: SnapshotId,
        outputs: Vec<SharedValue>,
    ) -> EvaluationResults {
        EvaluationResults {
            batch: self.id,
            snapshot,
            outputs,
        }
    }
}
impl Default for RequestBatch {
    fn default() -> Self {
        Self::new()
    }
}
/// Temporary request builder for a single sampling frame.
pub struct FrameRequests<'a> {
    batch: &'a mut RequestBatch,
    frame: FrameIndex,
}
impl FrameRequests<'_> {
    pub fn request<T: 'static>(&mut self, node: Node<T>) -> RequestId<T> {
        let id = RequestId {
            batch: self.batch.id,
            index: self.batch.requests.len(),
            marker: PhantomData,
        };
        self.batch.requests.push(Request {
            frame: self.frame,
            graph: node.graph_id,
            node: node.id,
            ty: TypeId::of::<T>(),
        });
        id
    }
}
/// Owns only requested outputs, independently of evaluator, graph and inputs.
pub struct EvaluationResults {
    batch: RequestBatchId,
    snapshot: SnapshotId,
    outputs: Vec<SharedValue>,
}
impl EvaluationResults {
    pub fn batch_id(&self) -> RequestBatchId {
        self.batch
    }
    pub fn snapshot_id(&self) -> SnapshotId {
        self.snapshot
    }
    fn value<T>(&self, request: RequestId<T>) -> Result<&SharedValue, EvalError> {
        if request.batch != self.batch {
            return Err(EvalError::ForeignRequest);
        }
        self.outputs
            .get(request.index)
            .ok_or(EvalError::InvalidRequest {
                index: request.index,
            })
    }
    pub fn get<T: 'static>(&self, request: RequestId<T>) -> Result<&T, EvalError> {
        self.value(request)?
            .downcast_ref()
            .ok_or(EvalError::ResultTypeMismatch {
                index: request.index,
            })
    }
    pub fn get_shared<T: Send + Sync + 'static>(
        &self,
        request: RequestId<T>,
    ) -> Result<Arc<T>, EvalError> {
        self.value(request)?
            .clone()
            .downcast()
            .map_err(|_| EvalError::ResultTypeMismatch {
                index: request.index,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_reads_validate_batch_index_and_concrete_storage_type() {
        let batch = RequestBatchId::new();
        let outputs = vec![Arc::new(3_u64) as SharedValue];
        let results = EvaluationResults {
            batch,
            snapshot: SnapshotId::new(),
            outputs,
        };
        let invalid = RequestId::<u64> {
            batch,
            index: 2,
            marker: PhantomData,
        };
        let wrong_type = RequestId::<u32> {
            batch,
            index: 0,
            marker: PhantomData,
        };
        assert_eq!(
            results.get(invalid),
            Err(EvalError::InvalidRequest { index: 2 })
        );
        assert_eq!(
            results.get_shared(wrong_type),
            Err(EvalError::ResultTypeMismatch { index: 0 })
        );
        let foreign = RequestId::<u64> {
            batch: RequestBatchId::new(),
            index: 0,
            marker: PhantomData,
        };
        assert_eq!(results.get(foreign), Err(EvalError::ForeignRequest));
    }
}
