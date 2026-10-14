use gekijyou_computation::{
    CachePolicy, EvalError, Evaluator, FrameIndex, GraphError, GraphState, Node, NodeCacheConfig,
    RequestBatch,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[test]
fn arbitrary_frames_share_dependencies_and_results_own_values() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let (input, node) = state
        .edit(|e| {
            let i = e.input::<u64>();
            e.set(i, 10)?;
            let calls = calls.clone();
            let n = e.map((e.frame(), i.node()), move |f: &FrameIndex, x: &u64| {
                calls.fetch_add(1, Ordering::Relaxed);
                f.0 + x
            })?;
            Ok((i, n))
        })
        .unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        node,
        NodeCacheConfig {
            memo: CachePolicy::Off,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    let mut batch = RequestBatch::new();
    let requests: Vec<_> = [120, 30, 120, 300]
        .map(|f| batch.at(FrameIndex(f)).request(node))
        .into();
    let results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(results.snapshot_id(), snapshot.id());
    assert!(Arc::ptr_eq(
        &results.get_shared(requests[0]).unwrap(),
        &results.get_shared(requests[2]).unwrap()
    ));
    state.edit(|e| e.set(input, 20)).unwrap();
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(120), node)
            .unwrap(),
        140
    );
    drop(eval);
    drop(state);
    for (id, f) in requests.into_iter().zip([120, 30, 120, 300]) {
        assert_eq!(*results.get(id).unwrap(), f + 10);
    }
}

#[test]
fn targets_validate_before_computations_and_handles_do_not_cross_batches() {
    let mut a = GraphState::new();
    let mut b = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let n = a
        .edit(|e| {
            let calls = calls.clone();
            e.map((), move || {
                calls.fetch_add(1, Ordering::Relaxed);
                3_u64
            })
        })
        .unwrap();
    let foreign = b.edit(|e| Ok(e.constant(4_u64))).unwrap();
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(1)).request(n);
    batch.at(FrameIndex(1)).request(foreign);
    let mut eval = Evaluator::new(&a);
    assert!(eval.bind(a.snapshot()).unwrap().evaluate(batch).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let mut one = RequestBatch::new();
    let old = one.at(FrameIndex(1)).request(n);
    let mut two = RequestBatch::new();
    let new = two.at(FrameIndex(1)).request(n);
    let result = eval.bind(a.snapshot()).unwrap().evaluate(two).unwrap();
    assert_eq!(result.get(new).unwrap(), &3);
    assert_eq!(result.get(old), Err(EvalError::ForeignRequest));
}

struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
#[test]
fn error_and_panic_release_scratch_and_allow_reuse() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut state = GraphState::new();
    let (probe, missing, panics) = state
        .edit(|e| {
            let drops = drops.clone();
            let p = e.map((), move || Probe(drops.clone()))?;
            let m = e.input::<u64>();
            let panics = e.map(p, |_: &Probe| -> u64 { panic!("compute") })?;
            Ok((p, m, panics))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        probe,
        NodeCacheConfig {
            memo: CachePolicy::Off,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    let mut batch = RequestBatch::new();
    batch.at(FrameIndex(0)).request(probe);
    batch.at(FrameIndex(0)).request(missing.node());
    assert!(
        eval.bind(state.snapshot())
            .unwrap()
            .evaluate(batch)
            .is_err()
    );
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            eval.bind(state.snapshot())
                .unwrap()
                .sample(FrameIndex(0), panics)
                .unwrap();
        }))
        .is_err()
    );
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    drop(
        eval.bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), probe)
            .unwrap(),
    );
    assert_eq!(drops.load(Ordering::Relaxed), 3);
}

#[test]
fn defining_a_new_graph_never_reuses_incompatible_history() {
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(1_u64))).unwrap();
    let old = state.snapshot();
    let mut eval = Evaluator::new(&state);
    assert_eq!(
        *eval
            .bind(old.clone())
            .unwrap()
            .sample(FrameIndex(0), node)
            .unwrap(),
        1
    );
    state.edit(|e| e.replace(node, (), || 2_u64)).unwrap();
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), node)
            .unwrap(),
        2
    );
    assert!(matches!(
        eval.bind(old),
        Err(EvalError::StaleGraphDefinition { .. })
    ));
}

#[test]
fn twelve_input_map_and_homogeneous_slice_nodes() {
    let mut state = GraphState::new();
    let (sum, sources) = state
        .edit(|e| {
            let a = e.constant(1_u64);
            let b = e.constant(2_u64);
            let c = e.constant(3_u64);
            let d = e.constant(4_u64);
            let e5 = e.constant(5_u64);
            let f = e.constant(6_u64);
            let g = e.constant(7_u64);
            let h = e.constant(8_u64);
            let i = e.constant(9_u64);
            let j = e.constant(10_u64);
            let k = e.constant(11_u64);
            let l = e.constant(12_u64);
            let sum = e.map(
                (a, b, c, d, e5, f, g, h, i, j, k, l),
                |a: &u64,
                 b: &u64,
                 c: &u64,
                 d: &u64,
                 e5: &u64,
                 f: &u64,
                 g: &u64,
                 h: &u64,
                 i: &u64,
                 j: &u64,
                 k: &u64,
                 l: &u64| {
                    *a + *b + *c + *d + *e5 + *f + *g + *h + *i + *j + *k + *l
                },
            )?;
            Ok((sum, [a, b, c, d, e5, f, g, h, i, j, k, l]))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), sum)
            .unwrap(),
        78
    );

    // Homogeneous slice mapping preserves order; replacing rewrites in place.
    let (plain, replaced) = state
        .edit(|e| {
            let plain = e.map_slice(&sources[..3], |xs: &[&u64]| {
                xs.iter().map(|x| **x).sum::<u64>()
            })?;
            let replaced = e.map_slice(&sources[..1], |xs: &[&u64]| *xs[0] * 100)?;
            e.replace_slice(replaced, &sources[..2], |xs: &[&u64]| {
                (*xs[0] + *xs[1]) * 10
            })?;
            Ok((plain, replaced))
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    let snapshot = state.snapshot();
    assert_eq!(
        *eval
            .bind(snapshot.clone())
            .unwrap()
            .sample(FrameIndex(0), plain)
            .unwrap(),
        6
    );
    assert_eq!(
        *eval
            .bind(snapshot)
            .unwrap()
            .sample(FrameIndex(0), replaced)
            .unwrap(),
        30
    );
}

#[test]
fn empty_slice_is_frame_independent_and_foreign_inputs_roll_back() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let empty = state
        .edit(|e| {
            let calls = calls.clone();
            e.map_slice(&[] as &[Node<u64>], move |xs: &[&u64]| {
                calls.fetch_add(1, Ordering::Relaxed);
                xs.len() as u64
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        empty,
        NodeCacheConfig {
            memo: CachePolicy::Off,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    let mut batch = RequestBatch::new();
    let ids: Vec<_> = [10_u64, 20, 30]
        .iter()
        .map(|&f| batch.at(FrameIndex(f)).request(empty))
        .collect();
    let results = eval
        .bind(state.snapshot())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    for id in &ids {
        assert_eq!(*results.get(*id).unwrap(), 0);
    }
    // A zero-input node is frame-independent: one evaluation serves every frame.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(Arc::ptr_eq(
        &results.get_shared(ids[0]).unwrap(),
        &results.get_shared(ids[2]).unwrap()
    ));

    // Foreign inputs are rejected and the staged graph is left untouched.
    let before = state.graph().len();
    let mut other = GraphState::new();
    let foreign = other.edit(|e| Ok(e.constant(1_u64))).unwrap();
    let error = state
        .edit(|e| e.map_slice(&[foreign], |xs: &[&u64]| xs.len() as u64))
        .unwrap_err();
    assert!(matches!(error, GraphError::ForeignNode { .. }));
    assert_eq!(state.graph().len(), before);
    let error = state
        .edit(|e| e.replace_slice(empty, &[foreign], |xs: &[&u64]| xs.len() as u64))
        .unwrap_err();
    assert!(matches!(error, GraphError::ForeignNode { .. }));
    assert_eq!(state.graph().len(), before);
    let mut eval = Evaluator::new(&state);
    assert_eq!(
        *eval
            .bind(state.snapshot())
            .unwrap()
            .sample(FrameIndex(0), empty)
            .unwrap(),
        0
    );
}

#[test]
fn static_node_with_off_caches_is_evaluated_once_across_frames() {
    let mut state = GraphState::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let node = state
        .edit(|e| {
            let calls = calls.clone();
            e.map((), move || {
                calls.fetch_add(1, Ordering::Relaxed);
                5_u64
            })
        })
        .unwrap();
    let mut eval = Evaluator::new(&state);
    eval.configure_cache(
        node,
        NodeCacheConfig {
            memo: CachePolicy::Off,
            context: CachePolicy::Off,
        },
    )
    .unwrap();
    let mut batch = RequestBatch::new();
    let ids: Vec<_> = [10_u64, 20, 30, 40]
        .iter()
        .map(|&f| batch.at(FrameIndex(f)).request(node))
        .collect();
    let results = eval
        .bind(state.snapshot())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    // Scratch normalization shares one frame-independent evaluation across frames.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(eval.stats().computation_calls, 1);
    for id in &ids {
        assert_eq!(*results.get(*id).unwrap(), 5);
    }
    assert!(ids.windows(2).all(|pair| Arc::ptr_eq(
        &results.get_shared(pair[0]).unwrap(),
        &results.get_shared(pair[1]).unwrap()
    )));
}
