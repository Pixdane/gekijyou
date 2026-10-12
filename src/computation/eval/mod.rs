use super::{
    batch::{BatchResults, ContextId, RequestBatch},
    cache::NodeCache,
    context::Context,
    error::{EvalError, GraphError},
    graph::Graph,
    node::{GraphId, NodeId, NodeKind},
    value::SharedValue,
};
use slotmap::SecondaryMap;
use smallvec::SmallVec;
use std::{any::Any, sync::Arc};

/// Cumulative evaluation counters for one evaluator, independent of batch results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// Successful persistent memo reuse; excludes per-context temporary sharing.
    pub memo_hits: u64,
    /// Persistent lookups that completed without a match, including empty caches.
    pub memo_misses: u64,
    /// Actual input equivalence checks, including identity checks and short circuits.
    pub memo_comparisons: u64,
    /// Closure invocations, including those that subsequently panic.
    pub computation_calls: u64,
}

/// Serial, on-demand evaluation of batches bound to one graph identity.
///
/// Retains node-local direct-input memo entries according to each node's policy.
/// Revisions and policies synchronize lazily; editing one node preserves other
/// nodes' history. Reusable after errors and computation panics caught by the caller.
/// Retained input/output Arcs live until eviction, invalidation or evaluator drop.
pub struct Evaluator {
    graph_id: GraphId,
    caches: SecondaryMap<NodeId, NodeCache>,
    stats: EvalStats,
}

impl Evaluator {
    /// Creates an evaluator bound to this graph's identity.
    pub fn new(graph: &Graph) -> Self {
        Self {
            graph_id: graph.id(),
            caches: SecondaryMap::new(),
            stats: EvalStats::default(),
        }
    }

    /// Returns cumulative counters without changing the cache or counters.
    pub fn stats(&self) -> EvalStats {
        self.stats
    }

    /// Returns previous counters and resets them; retained memo entries are unchanged.
    pub fn reset_stats(&mut self) -> EvalStats {
        std::mem::take(&mut self.stats)
    }

    /// Consumes a batch and evaluates its requests with the graph's current definitions.
    ///
    /// Validates every target before executing any closure. Missing external inputs
    /// are discovered only when reached. Each node is evaluated at most once per
    /// context in this batch. Persistent memo lookup still resolves all direct
    /// dependencies; it never skips the dependency chain based on root inputs.
    /// Empty batches succeed, and unused contexts do no work.
    ///
    /// On error, no partial BatchResults are returned and temporary values are released.
    /// Successful pure-function memo entries are retained. Computation panics unwind
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
            let value = self.evaluate_node(
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

    fn evaluate_node(
        &mut self,
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
            NodeKind::Input => {
                let value = Arc::clone(context.value(node).ok_or(EvalError::MissingInput {
                    context: context_id,
                    node,
                })?);
                if value.as_ref().type_id() != record.type_id {
                    return Err(GraphError::TypeMismatch { node }.into());
                }
                value
            }
            NodeKind::Computation {
                inputs,
                computation,
            } => {
                // Keep dependency Arcs alive independently of the mutable result table.
                // Preserve input order and repeated edges at the erasure boundary.
                // Inline the common 0–4 input case; larger definitions spill to the heap.
                let dependencies: SmallVec<[SharedValue; 4]> = inputs
                    .iter()
                    .map(|input| {
                        self.evaluate_node(graph, context, context_id, values, input.source)
                    })
                    .collect::<Result<_, _>>()?;
                // Dependency traversal is complete before borrowing any node cache.
                let policy = record.cache_policy;
                let cached = if policy.capacity() == 0 {
                    if let Some(cache) = self.caches.get_mut(node) {
                        cache.synchronize(record.revision, policy);
                    }
                    None
                } else {
                    let cache = self
                        .caches
                        .entry(node)
                        .expect("Valid node key")
                        .or_insert_with(|| NodeCache::new(record.revision, policy));
                    cache.synchronize(record.revision, policy);
                    let cached = cache.lookup(&dependencies, |position, current, previous| {
                        self.stats.memo_comparisons += 1;
                        let source = graph.node_record(inputs[position].source)?;
                        Ok::<_, EvalError>(source.semantics.equivalent(current, previous))
                    })?;
                    if cached.is_some() {
                        self.stats.memo_hits += 1;
                    } else {
                        self.stats.memo_misses += 1;
                    }
                    cached
                };
                if let Some(output) = cached {
                    output
                } else {
                    let args: SmallVec<[&dyn Any; 4]> = dependencies
                        .iter()
                        .map(|value| value.as_ref() as &dyn Any)
                        .collect();
                    self.stats.computation_calls += 1;
                    let output = computation(&args);
                    // Validate before retaining or publishing a computed result.
                    if output.as_ref().type_id() != record.type_id {
                        return Err(GraphError::TypeMismatch { node }.into());
                    }
                    if policy.capacity() != 0 {
                        self.caches
                            .get_mut(node)
                            .expect("Synchronized node cache")
                            .insert(&dependencies, Arc::clone(&output));
                    }
                    output
                }
            }
        };
        values.insert(node, Arc::clone(&value));
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
