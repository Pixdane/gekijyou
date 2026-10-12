use super::*;
use crate::computation::value::SharedValue;
use crate::computation::{Context, Evaluator, RequestBatch};
use slotmap::Key;

fn evaluated(graph: &Graph, node: Node<u64>) -> u64 {
    let mut batch = RequestBatch::new(graph);
    let context = batch.add_context(Context::new(graph)).unwrap();
    let request = batch.request(context, node).unwrap();
    *Evaluator::new(graph)
        .evaluate_batch(graph, batch)
        .unwrap()
        .get(request)
        .unwrap()
}

#[test]
fn versions_count_successful_replacements_only() {
    let mut graph = Graph::new();
    let first = graph.constant(2_u64);
    let second = graph.map(first, |n: &u64| n + 3);
    let input = graph.input::<u64>();
    assert_eq!(graph.replacement_epoch, 0);
    assert!(graph.nodes.values().all(|record| record.revision == 0));
    let id = first.id;
    for (revision, value) in [(1, 7), (2, 11)] {
        graph.replace(first, (), move || value).unwrap();
        assert_eq!(first.id, id);
        assert_eq!(graph.nodes[first.id].revision, revision);
        assert_eq!(graph.replacement_epoch, revision);
        assert_eq!(graph.nodes[second.id].revision, 0);
        assert_eq!(evaluated(&graph, second), value + 3);
    }
    graph.input::<String>();
    graph.constant(99_u64);
    graph.map(input, |n: &u64| n + 1);
    graph.set_default_eq::<String>();
    assert_eq!(graph.replacement_epoch, 2);
    assert_eq!(graph.nodes[first.id].revision, 2);
    assert!(
        graph
            .nodes
            .iter()
            .all(|(id, record)| id == first.id || record.revision == 0)
    );
}

#[test]
fn validation_errors_preserve_versions_and_behavior() {
    let mut graph = Graph::new();
    let target = graph.constant(7_u64);
    let child = graph.map(target, |n: &u64| n + 1);
    graph.replace(target, (), || 11_u64).unwrap();
    let foreign = Graph::new();
    assert_eq!(
        graph.replace(target, child, |n: &u64| *n),
        Err(GraphError::Cycle)
    );
    assert_eq!(
        graph.replace(target, foreign.frame(), |_: &FrameIndex| 99_u64),
        Err(GraphError::ForeignNode {
            node: foreign.frame().id
        })
    );
    let wrong: Node<String> = Node::new(graph.id(), target.id);
    assert_eq!(
        graph.replace(wrong, (), String::new),
        Err(GraphError::TypeMismatch { node: target.id })
    );
    let missing: Node<u64> = Node::new(graph.id(), NodeId::null());
    assert_eq!(
        graph.replace(missing, (), || 99_u64),
        Err(GraphError::MissingNode { node: missing.id })
    );
    assert_eq!(
        graph.replace(graph.frame(), (), || FrameIndex(99)),
        Err(GraphError::InputNodeReplacement {
            node: graph.frame().id
        })
    );
    assert_eq!(graph.nodes[target.id].revision, 1);
    assert_eq!(graph.replacement_epoch, 1);
    assert_eq!(graph.nodes[child.id].revision, 0);
    assert_eq!(graph.nodes[graph.frame().id].revision, 0);
    assert_eq!(evaluated(&graph, child), 12);
}

#[test]
fn either_counter_exhaustion_is_transactional() {
    for (revision, epoch) in [(u64::MAX, 0), (0, u64::MAX), (u64::MAX, u64::MAX)] {
        let mut graph = Graph::new();
        let node = graph.constant(7_u64);
        graph.nodes[node.id].revision = revision;
        graph.replacement_epoch = epoch;
        let expected = if revision == u64::MAX {
            GraphError::RevisionExhausted { node: node.id }
        } else {
            GraphError::ReplacementEpochExhausted
        };
        assert_eq!(graph.replace(node, (), || 99_u64), Err(expected));
        assert_eq!(graph.nodes[node.id].revision, revision);
        assert_eq!(graph.replacement_epoch, epoch);
        assert_eq!(evaluated(&graph, node), 7);
        assert_eq!(graph.validate(), Ok(()));
    }
}

#[test]
fn default_equality_and_replacement_preserve_node_semantics() {
    let mut graph = Graph::new();
    let early = graph.input::<String>();
    graph.set_default_eq::<String>();
    let later = graph.constant(String::from("same"));
    let first: SharedValue = Arc::new(String::from("same"));
    let equal: SharedValue = Arc::new(String::from("same"));
    let different: SharedValue = Arc::new(String::from("different"));
    assert!(graph.nodes[early.id].semantics.equivalent(&first, &first));
    assert!(!graph.nodes[early.id].semantics.equivalent(&first, &equal));
    assert!(graph.nodes[later.id].semantics.equivalent(&first, &equal));
    assert!(
        !graph.nodes[later.id]
            .semantics
            .equivalent(&first, &different)
    );

    // Replacing a node cannot change the equality policy it was created with.
    let mut identity_graph = Graph::new();
    let original = identity_graph.constant(String::from("same"));
    identity_graph.set_default_eq::<String>();
    identity_graph
        .replace(original, (), || String::from("same"))
        .unwrap();
    assert!(
        !identity_graph.nodes[original.id]
            .semantics
            .equivalent(&first, &equal)
    );

    let frame: SharedValue = Arc::new(FrameIndex(12));
    let equal_frame: SharedValue = Arc::new(FrameIndex(12));
    let later_frame: SharedValue = Arc::new(FrameIndex(13));
    assert!(
        graph.nodes[graph.frame.id]
            .semantics
            .equivalent(&frame, &equal_frame)
    );
    assert!(
        !graph.nodes[graph.frame.id]
            .semantics
            .equivalent(&frame, &later_frame)
    );
}

#[test]
fn replacement_updates_roots_and_preserves_policy_and_repeated_edges() {
    let mut graph = Graph::new();
    let input = graph.input::<u64>();
    let target = graph.constant(3_u64);
    graph.nodes[target.id].cache_policy = CachePolicy::Lru { capacity: 7 };
    let root_count = graph.roots().count();
    graph
        .replace(target, (input, input), |a: &u64, b: &u64| a + b)
        .unwrap();
    assert_eq!(graph.roots().count(), root_count - 1);
    assert_eq!(
        graph.nodes[target.id].cache_policy,
        CachePolicy::Lru { capacity: 7 }
    );
    assert_eq!(graph.validate(), Ok(()));
    graph.replace(target, (), || 5_u64).unwrap();
    assert_eq!(graph.roots().count(), root_count);
    assert_eq!(
        graph.nodes[target.id].cache_policy,
        CachePolicy::Lru { capacity: 7 }
    );
}

#[test]
fn foreign_handles_are_rejected_even_when_storage_keys_match() {
    let mut graph = Graph::new();
    let foreign = Graph::new();
    assert_eq!(graph.frame().id, foreign.frame().id);
    assert_ne!(graph.id, foreign.id);
    let count = graph.nodes.len();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        graph.map(foreign.frame(), |f: &FrameIndex| f.0);
    }));
    assert!(result.is_err());
    assert_eq!(graph.nodes.len(), count);
    assert_eq!(graph.validate(), Ok(()));
}

#[test]
fn validation_checks_disconnected_cycles_independently_of_replace() {
    let mut graph = Graph::new();
    let title = graph.constant(String::from("第一幕"));
    let layout = graph.map(title, |title: &String| title.clone());
    let NodeKind::Computation { inputs, .. } = &mut graph.nodes[title.id].kind else {
        unreachable!()
    };
    *inputs = Box::new([Dependency {
        graph_id: graph.id,
        source: layout.id,
        type_id: TypeId::of::<String>(),
    }]);
    assert_eq!(graph.validate(), Err(GraphError::Cycle));
}

#[test]
fn custom_equivalence_is_frozen_at_creation_and_preserved_by_replace() {
    let mut graph = Graph::new();
    let early = graph.constant(0.0_f64);
    graph.set_default_equivalence::<f64, _>(|a, b| a.to_bits() == b.to_bits());
    let later = graph.constant(0.0_f64);
    let positive: SharedValue = Arc::new(0.0_f64);
    let same: SharedValue = Arc::new(0.0_f64);
    let negative: SharedValue = Arc::new(-0.0_f64);
    assert!(!graph.nodes[early.id].semantics.equivalent(&positive, &same));
    assert!(graph.nodes[later.id].semantics.equivalent(&positive, &same));
    assert!(
        !graph.nodes[later.id]
            .semantics
            .equivalent(&positive, &negative)
    );
    graph.set_default_equivalence::<f64, _>(|a, b| a == b);
    graph.replace(later, (), || -0.0_f64).unwrap();
    assert!(
        !graph.nodes[later.id]
            .semantics
            .equivalent(&positive, &negative)
    );
}

#[test]
fn policy_configuration_validates_handles_without_changing_versions() {
    let mut graph = Graph::new();
    let node = graph.constant(1_u64);
    let foreign = Graph::new();
    assert_eq!(
        graph.set_cache_policy(foreign.frame(), CachePolicy::Transient),
        Err(GraphError::ForeignNode {
            node: foreign.frame().id
        })
    );
    let missing: Node<u64> = Node::new(graph.id(), NodeId::null());
    assert_eq!(
        graph.set_cache_policy(missing, CachePolicy::Last),
        Err(GraphError::MissingNode { node: missing.id })
    );
    let wrong: Node<String> = Node::new(graph.id(), node.id);
    assert_eq!(
        graph.set_cache_policy(wrong, CachePolicy::Last),
        Err(GraphError::TypeMismatch { node: node.id })
    );
    for policy in [
        CachePolicy::Transient,
        CachePolicy::Last,
        CachePolicy::Lru { capacity: 4 },
    ] {
        let frame = graph.frame();
        assert_eq!(
            graph.set_cache_policy(frame, policy),
            Err(GraphError::InputNodeCachePolicy { node: frame.id })
        );
        assert_eq!(graph.nodes[frame.id].cache_policy, CachePolicy::default());
        graph.set_cache_policy(node, policy).unwrap();
        assert_eq!(graph.nodes[node.id].cache_policy, policy);
        assert_eq!(graph.nodes[node.id].revision, 0);
        assert_eq!(graph.replacement_epoch, 0);
    }
    assert_eq!(graph.validate(), Ok(()));
}
