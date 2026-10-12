use super::{
    batch::{BatchResults, ContextId, RequestBatch},
    cache::NodeCache,
    context::Context,
    context_cache::NodeContextCache,
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
    /// Successful external-input projection reuse, excluding direct memo hits.
    pub context_hits: u64,
    /// Completed projection queries without a match; missing inputs skip queries.
    pub context_misses: u64,
    /// Actual external-input equivalence checks, including identity comparisons.
    pub context_comparisons: u64,
    /// Node resolution entries, including temporary sharing and immediate hits.
    pub resolution_visits: u64,
}

/// Serial, on-demand evaluation of batches bound to one graph identity.
///
/// Retains direct-input memo and opt-in external-input projection caches separately.
/// Revisions, replacement epochs and configurations synchronize lazily. Replacing
/// a node preserves other nodes' direct memo; context caches conservatively rebuild.
/// Reusable after errors and computation panics caught by the caller.
/// Retained input/output Arcs live until eviction, invalidation or evaluator drop.
pub struct Evaluator {
    graph_id: GraphId,
    caches: SecondaryMap<NodeId, NodeCache>,
    context_caches: SecondaryMap<NodeId, NodeContextCache>,
    stats: EvalStats,
}

impl Evaluator {
    /// Creates an evaluator bound to this graph's identity.
    pub fn new(graph: &Graph) -> Self {
        Self {
            graph_id: graph.id(),
            caches: SecondaryMap::new(),
            context_caches: SecondaryMap::new(),
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
    /// context in this batch. Direct memo lookup resolves dependencies; enabled
    /// context caches can return before visiting any dependencies. Repeated context
    /// hits may compare projections again without invoking any closure.
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
        self.stats.resolution_visits += 1;
        if let Some(value) = values.get(node) {
            return Ok(Arc::clone(value));
        }
        let record = graph.node_record(node)?;
        // This query precedes dependency traversal. Hits deliberately do not
        // populate the dense temporary map, which could grow to a large NodeId.
        if record.context_cache_capacity == 0 {
            self.context_caches.remove(node);
        } else {
            let epoch = graph.replacement_epoch();
            if self
                .context_caches
                .get(node)
                .is_none_or(|cache| cache.replacement_epoch != epoch)
            {
                let cache = NodeContextCache::new(graph, node, record.context_cache_capacity)?;
                self.context_caches.insert(node, cache);
            }
            let cache = self
                .context_caches
                .get_mut(node)
                .expect("Synchronized context cache");
            cache.synchronize_capacity(record.context_cache_capacity);
            // Missing roots must fall back to DFS rather than report an error
            // here: an earlier dependency may panic before the missing input.
            if cache
                .roots
                .iter()
                .all(|&root| context.value(root).is_some())
            {
                let cached = cache.lookup(|root, previous| {
                    self.stats.context_comparisons += 1;
                    let current = context.value(root).expect("Present root input");
                    Ok::<_, EvalError>(
                        graph
                            .node_record(root)?
                            .semantics
                            .equivalent(current, previous),
                    )
                })?;
                if let Some(output) = cached {
                    self.stats.context_hits += 1;
                    return Ok(output);
                }
                self.stats.context_misses += 1;
            }
        }
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
        if record.context_cache_capacity != 0 {
            let cache = self
                .context_caches
                .get_mut(node)
                .expect("Synchronized context cache");
            let roots = cache
                .roots
                .iter()
                .map(|&root| {
                    Arc::clone(
                        context
                            .value(root)
                            .expect("Successfully resolved root input"),
                    )
                })
                .collect();
            cache.insert(roots, Arc::clone(&value));
        }
        values.insert(node, Arc::clone(&value));
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
