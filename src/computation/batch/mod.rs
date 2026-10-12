use super::{
    context::Context,
    error::{EvalError, GraphError},
    graph::Graph,
    node::{GraphId, Node, NodeId},
    value::SharedValue,
};
use std::{
    any::TypeId,
    fmt,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchId(u64);

impl BatchId {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("Batch identity space exhausted");
        Self(id)
    }
}

/// Identifies an owned context within the batch that registered it.
///
/// Handles from another batch are rejected even if their indices match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextId {
    batch_id: BatchId,
    pub(super) index: usize,
}

/// A typed result request issued by one batch. Copies do not copy output values.
///
/// Copy, Clone, Debug and equality impose no bounds on `T`. Fields are private and
/// the output type is invariant, preserving the exact result type:
///
/// ```compile_fail
/// use gekijyou::computation::RequestId;
/// fn relabel(id: RequestId<fn(&str)>) -> RequestId<fn(&'static str)> {
///     id
/// }
/// ```
pub struct RequestId<T> {
    batch_id: BatchId,
    index: usize,
    _marker: PhantomData<fn(T) -> T>,
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
            .field("batch_id", &self.batch_id)
            .field("index", &self.index)
            .finish()
    }
}

impl<T> PartialEq for RequestId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.batch_id == other.batch_id && self.index == other.index
    }
}

impl<T> Eq for RequestId<T> {}

pub(super) struct Request {
    pub(super) context: ContextId,
    pub(super) node: NodeId,
    pub(super) type_id: TypeId,
}

/// Owned contexts and ordered target requests for a single evaluation.
///
/// Binds only the graph identity. Nodes added or replaced after construction are
/// evaluated using the graph definitions supplied to `evaluate_batch`.
pub struct RequestBatch {
    pub(super) id: BatchId,
    pub(super) graph_id: GraphId,
    pub(super) contexts: Vec<Context>,
    pub(super) requests: Vec<Request>,
}

impl RequestBatch {
    /// Creates an empty batch for this graph.
    ///
    /// # Panics
    ///
    /// Panics if the process-local batch identity counter is exhausted.
    pub fn new(graph: &Graph) -> Self {
        Self {
            id: BatchId::new(),
            graph_id: graph.id(),
            contexts: Vec::new(),
            requests: Vec::new(),
        }
    }

    /// Takes ownership of a context from the same graph and freezes its assignments.
    pub fn add_context(&mut self, context: Context) -> Result<ContextId, EvalError> {
        if context.graph_id != self.graph_id {
            return Err(EvalError::ForeignGraph);
        }
        let id = ContextId {
            batch_id: self.id,
            index: self.contexts.len(),
        };
        self.contexts.push(context);
        Ok(id)
    }

    /// Appends a target request, returning an independent typed result handle.
    ///
    /// Checks context and graph ownership immediately. Node existence and concrete
    /// output type are checked against the supplied graph before evaluation starts.
    pub fn request<T: 'static>(
        &mut self,
        context: ContextId,
        node: Node<T>,
    ) -> Result<RequestId<T>, EvalError> {
        self.context(context)?;
        if node.graph_id != self.graph_id {
            return Err(GraphError::ForeignNode { node: node.id }.into());
        }
        let id = RequestId {
            batch_id: self.id,
            index: self.requests.len(),
            _marker: PhantomData,
        };
        self.requests.push(Request {
            context,
            node: node.id,
            type_id: TypeId::of::<T>(),
        });
        Ok(id)
    }

    pub(super) fn context(&self, id: ContextId) -> Result<&Context, EvalError> {
        if id.batch_id != self.id {
            return Err(EvalError::ForeignContext);
        }
        self.contexts
            .get(id.index)
            .ok_or(EvalError::InvalidContext { index: id.index })
    }
}

/// Requested outputs owned independently of the graph, evaluator and input contexts.
///
/// Each output is retained by Arc. Graph replacement or destruction cannot discard
/// these values. As with all graph values, callers must uphold semantic immutability.
/// Duplicate requests may share the same stored value while retaining distinct IDs.
pub struct BatchResults {
    batch_id: BatchId,
    outputs: Vec<SharedValue>,
}

impl BatchResults {
    pub(super) fn new(batch_id: BatchId, outputs: Vec<SharedValue>) -> Self {
        Self { batch_id, outputs }
    }

    /// Reads a typed result after checking batch ownership, index and concrete type.
    ///
    /// The returned reference borrows only these BatchResults, not the graph or evaluator.
    pub fn get<T: 'static>(&self, request: RequestId<T>) -> Result<&T, EvalError> {
        if request.batch_id != self.batch_id {
            return Err(EvalError::ForeignRequest);
        }
        self.outputs
            .get(request.index)
            .ok_or(EvalError::InvalidRequest {
                index: request.index,
            })?
            .downcast_ref::<T>()
            .ok_or(EvalError::ResultTypeMismatch {
                index: request.index,
            })
    }
}

#[cfg(test)]
mod tests;
