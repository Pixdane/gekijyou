use super::*;
use crate::p35::value::ValueSemantics;
use std::{
    convert::Infallible,
    panic::{AssertUnwindSafe, catch_unwind},
};

fn value(n: u64) -> SharedValue {
    Arc::new(n)
}

#[test]
fn projections_are_deduplicated_ordered_and_exclude_constants() {
    let mut graph = Graph::new();
    let roots: Vec<_> = (0..6).map(|_| graph.input::<u64>()).collect();
    let left = graph.map(
        (roots[0], roots[1], roots[2], roots[0]),
        |_: &u64, _: &u64, _: &u64, _: &u64| 0_u64,
    );
    let right = graph.map(
        (roots[3], roots[4], roots[5], roots[1]),
        |_: &u64, _: &u64, _: &u64, _: &u64| 0_u64,
    );
    let constant = graph.constant(9_u64);
    let target = graph.map(
        (left, right, left, constant),
        |_: &u64, _: &u64, _: &u64, _: &u64| 0_u64,
    );
    assert_eq!(
        &*root_inputs(&graph, target.id).unwrap(),
        &roots.iter().map(|r| r.id).collect::<Vec<_>>()
    );
    assert_eq!(&*root_inputs(&graph, roots[0].id).unwrap(), &[roots[0].id]);
    assert!(root_inputs(&graph, constant.id).unwrap().is_empty());
}

#[test]
fn lru_bounds_capacity_sync_and_output_lifetime() {
    let graph = Graph::new();
    let mut cache = NodeContextCache::new(&graph, graph.frame().id, 2).unwrap();
    let first = value(10);
    cache.insert(vec![value(1)].into(), Arc::clone(&first));
    cache.insert(vec![value(2)].into(), value(20));
    let hit = cache
        .lookup(|_, old| {
            Ok::<_, Infallible>(ValueSemantics::by_eq::<u64>().equivalent(&value(1), old))
        })
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&first, &hit));
    cache.insert(vec![value(3)].into(), value(30));
    assert_eq!(cache.entries.len(), 2);
    assert_eq!(
        cache.entries.front().unwrap().root_values[0].downcast_ref::<u64>(),
        Some(&1)
    );
    cache.synchronize_capacity(2);
    assert_eq!(cache.entries.len(), 2);
    cache.synchronize_capacity(1);
    assert!(cache.entries.is_empty());
    cache.insert(vec![value(4)].into(), value(40));
    cache.synchronize_capacity(0);
    cache.insert(vec![value(5)].into(), value(50));
    assert!(cache.entries.is_empty());
    drop(cache);
    assert_eq!(hit.downcast_ref::<u64>(), Some(&10));
}

#[test]
fn comparison_errors_and_panics_preserve_all_entries_and_recency() {
    for panic in [false, true] {
        let mut graph = Graph::new();
        let a = graph.input::<u64>();
        let b = graph.input::<u64>();
        let target = graph.map((a, b), |a: &u64, b: &u64| a + b);
        let mut cache = NodeContextCache::new(&graph, target.id, 2).unwrap();
        cache.insert(vec![value(1), value(10)].into(), value(10));
        cache.insert(vec![value(1), value(20)].into(), value(20));
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            cache.lookup(|root, _| {
                if root == b.id {
                    assert!(!panic, "comparator panic");
                    return Err(());
                }
                Ok(true)
            })
        }));
        if panic {
            assert!(attempt.is_err());
        } else {
            assert!(matches!(attempt, Ok(Err(()))));
        }
        cache.insert(vec![value(1), value(30)].into(), value(30));
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(
            cache.entries.front().unwrap().output.downcast_ref::<u64>(),
            Some(&20)
        );
    }
}

#[test]
fn projection_builder_handles_deep_chains_without_recursive_stack() {
    let mut graph = Graph::new();
    let root = graph.input::<u64>();
    let mut node = root;
    for _ in 0..10_000 {
        node = graph.map(node, |n: &u64| *n);
    }
    assert_eq!(&*root_inputs(&graph, node.id).unwrap(), &[root.id]);
}
