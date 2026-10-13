use gekijyou_computation::{
    CachePolicy, EvalError, Evaluator, FrameIndex, GraphState, NodeCacheConfig, RequestBatch,
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
