use super::{
    CachePolicy, EvalError, EvaluationResults, EvaluationSnapshot, FrameIndex, Graph, GraphError,
    GraphState, Node, NodeCacheConfig, NodeId, RequestBatch,
    cache::NodeCache,
    context_cache::NodeContextCache,
    node::{NodeKind, NodeRecord},
    value::SharedValue,
};
use smallvec::SmallVec;
use std::{any::Any, collections::HashMap, sync::Arc};

/// Cumulative actual work, independent of the lifetime of returned results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalStats {
    pub memo_hits: u64,
    pub memo_misses: u64,
    pub memo_comparisons: u64,
    pub computation_calls: u64,
    pub context_hits: u64,
    pub context_misses: u64,
    pub context_comparisons: u64,
    pub resolution_visits: u64,
    pub definition_syncs: u64,
    pub projection_builds: u64,
}
/// Current retained entry counts, not byte sizes of shared objects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub memo_entries: usize,
    pub context_entries: usize,
}
#[derive(Default)]
struct CacheStore {
    memo: HashMap<NodeId, NodeCache>,
    context: HashMap<NodeId, NodeContextCache>,
    config: HashMap<NodeId, NodeCacheConfig>,
}
#[derive(Default)]
struct EvalScratch {
    values: HashMap<(FrameIndex, NodeId), SharedValue>,
    frames: HashMap<FrameIndex, SharedValue>,
}
impl EvalScratch {
    fn clear(&mut self) {
        self.values.clear();
        self.frames.clear();
    }
}

/// Sole owner of persistent caches. Binding definitions advances monotonically.
/// Input-only snapshots preserve both histories and root projections. Snapshot
/// scopes hold exclusive mutable access; computation closures never see the cache.
pub struct Evaluator {
    pub(crate) graph: Arc<Graph>,
    cache: CacheStore,
    scratch: EvalScratch,
    stats: EvalStats,
}
impl Evaluator {
    pub fn new(state: &GraphState) -> Self {
        Self::from_snapshot(&state.snapshot())
    }
    /// Independent evaluator for offline sampling of an old immutable definition.
    pub fn from_snapshot(snapshot: &EvaluationSnapshot) -> Self {
        Self {
            graph: snapshot.graph.clone(),
            cache: CacheStore::default(),
            scratch: EvalScratch::default(),
            stats: EvalStats::default(),
        }
    }
    pub fn stats(&self) -> EvalStats {
        self.stats
    }
    pub fn reset_stats(&mut self) -> EvalStats {
        std::mem::take(&mut self.stats)
    }
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            memo_entries: self.cache.memo.values().map(NodeCache::entry_count).sum(),
            context_entries: self
                .cache
                .context
                .values()
                .map(NodeContextCache::entry_count)
                .sum(),
        }
    }
    pub fn clear_cache(&mut self) {
        // Detach before dropping user-owned values, so unwinding cannot expose a
        // partially cleared cache as valid on the next call.
        let memo = std::mem::take(&mut self.cache.memo);
        let context = std::mem::take(&mut self.cache.context);
        drop((memo, context));
    }
    pub fn configure_cache<T: 'static>(
        &mut self,
        node: Node<T>,
        config: NodeCacheConfig,
    ) -> Result<(), GraphError> {
        self.graph.check(node)?;
        if !matches!(
            self.graph.node_record(node.id)?.kind,
            NodeKind::Computation { .. }
        ) {
            return Err(GraphError::NotComputation { node: node.id });
        }
        if matches!(config.memo, CachePolicy::Lru { capacity: 0 })
            || matches!(config.context, CachePolicy::Lru { capacity: 0 })
        {
            return Err(GraphError::InvalidCacheCapacity);
        }
        let old = self
            .cache
            .config
            .insert(node.id, config)
            .unwrap_or_default();
        let memo = (old.memo != config.memo).then(|| self.cache.memo.remove(&node.id));
        let context = (old.context != config.context).then(|| self.cache.context.remove(&node.id));
        drop((memo, context));
        Ok(())
    }
    pub fn bind(&mut self, snapshot: EvaluationSnapshot) -> Result<EvaluationScope<'_>, EvalError> {
        if snapshot.graph_id() != self.graph.id {
            return Err(EvalError::ForeignGraph);
        }
        if snapshot.definition_epoch() < self.graph.epoch {
            return Err(EvalError::StaleGraphDefinition {
                requested: snapshot.definition_epoch(),
                active: self.graph.epoch,
            });
        }
        if snapshot.definition_epoch() > self.graph.epoch {
            let old_context = std::mem::take(&mut self.cache.context);
            self.graph = snapshot.graph.clone();
            self.stats.definition_syncs += 1;
            // Revision synchronization is also checked on lookup/commit. Remove
            // dead nodes now to release their retained values and configuration.
            self.cache
                .memo
                .retain(|id, _| self.graph.nodes.contains_key(id));
            self.cache
                .config
                .retain(|id, _| self.graph.nodes.contains_key(id));
            for (&id, cache) in &mut self.cache.memo {
                cache.synchronize(
                    self.graph.node_record(id)?.revision,
                    self.cache.config.get(&id).copied().unwrap_or_default().memo,
                );
            }
            drop(old_context);
        }
        Ok(EvaluationScope {
            snapshot,
            evaluator: self,
        })
    }
    fn evaluate(
        &mut self,
        snapshot: &EvaluationSnapshot,
        batch: RequestBatch,
    ) -> Result<EvaluationResults, EvalError> {
        for request in &batch.requests {
            snapshot
                .graph
                .check_node(request.graph, request.node, request.ty)?;
        }
        // Scratch ownership is local during traversal. Error/unwind drops all
        // temporary Arcs; ordinary success/error restores only empty capacity.
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = (|| {
            let mut outputs = Vec::with_capacity(batch.len());
            for request in &batch.requests {
                let frame = scratch
                    .frames
                    .entry(request.frame)
                    .or_insert_with(|| Arc::new(request.frame))
                    .clone();
                outputs.push(self.resolve(
                    snapshot,
                    request.frame,
                    &frame,
                    request.node,
                    &mut scratch,
                    0,
                )?);
            }
            Ok(batch.finish(snapshot.id(), outputs))
        })();
        scratch.clear();
        self.scratch = scratch;
        result
    }
    fn resolve(
        &mut self,
        snapshot: &EvaluationSnapshot,
        frame: FrameIndex,
        frame_value: &SharedValue,
        node: NodeId,
        scratch: &mut EvalScratch,
        depth: usize,
    ) -> Result<SharedValue, EvalError> {
        const MAX_DEPTH: usize = 256;
        if depth > MAX_DEPTH {
            return Err(EvalError::DepthLimit { limit: MAX_DEPTH });
        }
        self.stats.resolution_visits += 1;
        if let Some(value) = scratch.values.get(&(frame, node)) {
            return Ok(value.clone());
        }
        let record = snapshot.graph.node_record(node)?;
        if let Some(output) = self.context_lookup(snapshot, frame_value, node)? {
            scratch.values.insert((frame, node), output.clone());
            return Ok(output);
        }
        let value = match &record.kind {
            NodeKind::Frame => frame_value.clone(),
            NodeKind::Input => snapshot
                .inputs
                .get(&node)
                .cloned()
                .ok_or(EvalError::MissingInput { frame, node })?,
            NodeKind::Computation {
                inputs,
                computation,
            } => {
                let dependencies: SmallVec<[SharedValue; 4]> = inputs
                    .iter()
                    .map(|i| {
                        self.resolve(snapshot, frame, frame_value, i.source, scratch, depth + 1)
                    })
                    .collect::<Result<_, _>>()?;
                if let Some(output) =
                    self.memo_lookup(&snapshot.graph, node, record, &dependencies)?
                {
                    output
                } else {
                    let args: SmallVec<[&dyn Any; 4]> = dependencies
                        .iter()
                        .map(|v| v.as_ref() as &dyn Any)
                        .collect();
                    self.stats.computation_calls += 1;
                    let output = computation(&args);
                    if output.as_ref().type_id() != record.type_id {
                        return Err(GraphError::TypeMismatch { node }.into());
                    }
                    self.memo_insert(node, record, &dependencies, output.clone());
                    output
                }
            }
        };
        self.context_insert(snapshot, frame_value, node, value.clone());
        scratch.values.insert((frame, node), value.clone());
        Ok(value)
    }
    pub(crate) fn memo_lookup(
        &mut self,
        graph: &Graph,
        node: NodeId,
        record: &NodeRecord,
        inputs: &[SharedValue],
    ) -> Result<Option<SharedValue>, EvalError> {
        let policy = self
            .cache
            .config
            .get(&node)
            .copied()
            .unwrap_or_default()
            .memo;
        if policy == CachePolicy::Off {
            return Ok(None);
        }
        let cache = self
            .cache
            .memo
            .entry(node)
            .or_insert_with(|| NodeCache::new(record.revision, policy));
        cache.synchronize(record.revision, policy);
        let sources = match &record.kind {
            NodeKind::Computation { inputs, .. } => inputs,
            _ => unreachable!(),
        };
        let output = cache.lookup(inputs, |position, current, old| {
            self.stats.memo_comparisons += 1;
            Ok::<_, EvalError>(
                graph
                    .node_record(sources[position].source)?
                    .semantics
                    .equivalent(current, old),
            )
        })?;
        if output.is_some() {
            self.stats.memo_hits += 1;
        } else {
            self.stats.memo_misses += 1;
        }
        Ok(output)
    }
    pub(crate) fn memo_insert(
        &mut self,
        node: NodeId,
        record: &NodeRecord,
        inputs: &[SharedValue],
        output: SharedValue,
    ) {
        let policy = self
            .cache
            .config
            .get(&node)
            .copied()
            .unwrap_or_default()
            .memo;
        if policy.capacity() != 0 {
            let cache = self
                .cache
                .memo
                .entry(node)
                .or_insert_with(|| NodeCache::new(record.revision, policy));
            cache.synchronize(record.revision, policy);
            cache.insert(inputs, output);
        }
    }
    pub(crate) fn context_lookup(
        &mut self,
        snapshot: &EvaluationSnapshot,
        frame: &SharedValue,
        node: NodeId,
    ) -> Result<Option<SharedValue>, EvalError> {
        let capacity = self
            .cache
            .config
            .get(&node)
            .copied()
            .unwrap_or_default()
            .context
            .capacity();
        if capacity == 0 {
            return Ok(None);
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.cache.context.entry(node) {
            entry.insert(NodeContextCache::new(&snapshot.graph, node, capacity)?);
            self.stats.projection_builds += 1;
        }
        let cache = self.cache.context.get_mut(&node).unwrap();
        if cache
            .roots
            .iter()
            .any(|&id| root_value(snapshot, frame, id).is_none())
        {
            return Ok(None);
        }
        let output = cache.lookup(|id, old| {
            self.stats.context_comparisons += 1;
            Ok::<_, EvalError>(
                snapshot
                    .graph
                    .node_record(id)?
                    .semantics
                    .equivalent(root_value(snapshot, frame, id).unwrap(), old),
            )
        })?;
        if output.is_some() {
            self.stats.context_hits += 1;
        } else {
            self.stats.context_misses += 1;
        }
        Ok(output)
    }
    pub(crate) fn context_insert(
        &mut self,
        snapshot: &EvaluationSnapshot,
        frame: &SharedValue,
        node: NodeId,
        output: SharedValue,
    ) {
        if let Some(cache) = self.cache.context.get_mut(&node) {
            let roots = cache
                .roots
                .iter()
                .map(|&id| {
                    root_value(snapshot, frame, id)
                        .expect("Resolved root")
                        .clone()
                })
                .collect();
            cache.insert(roots, output);
        }
    }
}
fn root_value<'a>(
    snapshot: &'a EvaluationSnapshot,
    frame: &'a SharedValue,
    node: NodeId,
) -> Option<&'a SharedValue> {
    if node == snapshot.graph.frame().id {
        Some(frame)
    } else {
        snapshot.inputs.get(&node)
    }
}

/// One snapshot, multiple frame requests, exclusive use of one persistent evaluator.
pub struct EvaluationScope<'a> {
    pub(crate) snapshot: EvaluationSnapshot,
    pub(crate) evaluator: &'a mut Evaluator,
}
impl EvaluationScope<'_> {
    pub fn snapshot(&self) -> &EvaluationSnapshot {
        &self.snapshot
    }
    pub fn evaluate(&mut self, batch: RequestBatch) -> Result<EvaluationResults, EvalError> {
        self.evaluator.evaluate(&self.snapshot, batch)
    }
    /// Owns its Arc so the value outlives this temporary scope and scratch.
    pub fn sample<T: Send + Sync + 'static>(
        &mut self,
        frame: FrameIndex,
        node: Node<T>,
    ) -> Result<Arc<T>, EvalError> {
        let mut batch = RequestBatch::new();
        let id = batch.at(frame).request(node);
        self.evaluate(batch)?.get_shared(id)
    }
}
