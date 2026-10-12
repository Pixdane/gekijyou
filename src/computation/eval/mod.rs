use super::{
    batch::{BatchResults, ContextId, RequestBatch},
    context::Context,
    error::{EvalError, GraphError},
    graph::Graph,
    node::{GraphId, NodeId, NodeKind},
    value::SharedValue,
};
use slotmap::SecondaryMap;
use smallvec::SmallVec;
use std::{any::Any, sync::Arc};

/// Serial, on-demand evaluation of batches bound to one graph identity.
///
/// Holds no persistent values or cache. It can be reused after graph edits and
/// evaluation failures, including computation panics caught by the caller.
pub struct Evaluator {
    graph_id: GraphId,
}

impl Evaluator {
    /// Creates an evaluator bound to this graph's identity.
    pub fn new(graph: &Graph) -> Self {
        Self {
            graph_id: graph.id(),
        }
    }

    /// Consumes a batch and evaluates its requests with the graph's current definitions.
    ///
    /// Validates every target before executing any closure. Missing external inputs
    /// are discovered only when reached. Each node is evaluated at most once per
    /// context in this batch. Empty batches succeed, and unused contexts do no work.
    ///
    /// On error, no partial BatchResults are returned and temporary values are released.
    /// Closures already executed are not rolled back. Computation panics unwind
    /// normally. Traversal is recursive and does not guarantee arbitrary chain depth.
    pub fn evaluate_batch(
        &mut self,
        graph: &Graph,
        batch: RequestBatch,
    ) -> Result<BatchResults, EvalError> {
        if self.graph_id != graph.id() || batch.graph_id != graph.id() {
            return Err(EvalError::ForeignGraph);
        }
        for request in &batch.requests {
            batch.context(request.context)?;
            graph.check_node(batch.graph_id, request.node, request.type_id)?;
        }

        let mut context_values: Vec<SecondaryMap<NodeId, SharedValue>> =
            batch.contexts.iter().map(|_| SecondaryMap::new()).collect();
        let mut outputs = Vec::with_capacity(batch.requests.len());
        for request in &batch.requests {
            let context = batch.context(request.context)?;
            let value = evaluate_node(
                graph,
                context,
                request.context,
                &mut context_values[request.context.index],
                request.node,
            )?;
            outputs.push(value);
        }
        Ok(BatchResults::new(batch.id, outputs))
    }
}

fn evaluate_node(
    graph: &Graph,
    context: &Context,
    context_id: ContextId,
    values: &mut SecondaryMap<NodeId, SharedValue>,
    node: NodeId,
) -> Result<SharedValue, EvalError> {
    if let Some(value) = values.get(node) {
        return Ok(Arc::clone(value));
    }
    let record = graph.node_record(node)?;
    let value = match &record.kind {
        NodeKind::Input => Arc::clone(context.value(node).ok_or(EvalError::MissingInput {
            context: context_id,
            node,
        })?),
        NodeKind::Computation {
            inputs,
            computation,
        } => {
            // Keep dependency Arcs alive independently of the mutable result table.
            // Preserve input order and repeated edges at the erasure boundary.
            // Inline the common 0–4 input case; larger definitions spill to the heap.
            let dependencies: SmallVec<[SharedValue; 4]> = inputs
                .iter()
                .map(|input| evaluate_node(graph, context, context_id, values, input.source))
                .collect::<Result<_, _>>()?;
            let args: SmallVec<[&dyn Any; 4]> = dependencies
                .iter()
                .map(|value| value.as_ref() as &dyn Any)
                .collect();
            computation(&args)
        }
    };
    if value.as_ref().type_id() != record.type_id {
        return Err(GraphError::TypeMismatch { node }.into());
    }
    values.insert(node, Arc::clone(&value));
    Ok(value)
}

#[cfg(test)]
mod tests;
