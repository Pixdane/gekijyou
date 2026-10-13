use gekijyou_computation::{FrameIndex, GraphError, GraphState};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

#[test]
fn snapshots_share_inputs_and_preserve_old_definitions_after_deletion() {
    let mut state = GraphState::new();
    let input = state
        .edit(|e| {
            let i = e.input::<Vec<u8>>();
            e.set_shared(i, Arc::new(vec![1; 1024]))?;
            Ok(i)
        })
        .unwrap();
    let old = state.snapshot();
    assert_eq!(old.id(), state.snapshot().id());
    let value = old.get_shared(input).unwrap().unwrap();
    assert!(Arc::ptr_eq(
        &value,
        &state.get_shared(input).unwrap().unwrap()
    ));
    state.edit(|e| e.remove(input.node())).unwrap();
    assert!(state.get_shared(input).is_err());
    assert_eq!(old.get_shared(input).unwrap().unwrap().len(), 1024);
    assert!(state.snapshot().definition_epoch() > old.definition_epoch());
}

#[test]
fn errors_and_panics_burn_handles_without_publishing_partial_edits() {
    let mut state = GraphState::new();
    let original = state.snapshot().id();
    let mut leaked = None;
    let error = state.edit(|e| {
        leaked = Some(e.input::<u32>());
        Err::<(), _>(GraphError::Cycle)
    });
    assert!(error.is_err());
    assert_eq!(state.snapshot().id(), original);
    let mut panicked = None;
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = state.edit::<()>(|e| {
                panicked = Some(e.input::<u32>());
                panic!("rollback")
            });
        }))
        .is_err()
    );
    let valid = state.edit(|e| Ok(e.input::<u32>())).unwrap();
    assert_ne!(valid.node(), leaked.unwrap().node());
    assert_ne!(valid.node(), panicked.unwrap().node());
    for invalid in [leaked.unwrap(), panicked.unwrap()] {
        assert!(state.edit(|e| e.set(invalid, 12)).is_err());
    }
}

#[test]
fn ordinary_frame_inputs_are_not_the_sampling_dimension() {
    let mut state = GraphState::new();
    let (input, plain, sampled) = state
        .edit(|e| {
            let i = e.input::<FrameIndex>();
            let plain = e.map(i.node(), |f: &FrameIndex| f.0)?;
            let sampled = e.map((e.frame(), plain), |f: &FrameIndex, x: &u64| f.0 + x)?;
            e.set(i, FrameIndex(17))?;
            Ok((i, plain, sampled))
        })
        .unwrap();
    let before = state.snapshot();
    assert!(!before.depends_on_frame(plain).unwrap());
    assert!(before.depends_on_frame(sampled).unwrap());
    state.edit(|e| e.set(input, FrameIndex(19))).unwrap();
    assert_eq!(
        before.definition_epoch(),
        state.snapshot().definition_epoch()
    );
    assert_ne!(before.id(), state.snapshot().id());
    assert!(state.edit(|e| e.remove(e.frame())).is_err());
    assert!(state.edit(|e| e.remove(plain)).is_err());
    state
        .edit(|e| {
            e.remove(plain)?;
            e.remove(sampled)
        })
        .unwrap();
}

#[test]
fn foreign_handles_and_cycles_rollback_all_changes() {
    let mut a = GraphState::new();
    let mut b = GraphState::new();
    let other = b.edit(|e| Ok(e.input::<u32>())).unwrap();
    assert!(a.edit(|e| e.set(other, 2)).is_err());
    let (x, y) = a
        .edit(|e| {
            let x = e.constant(2_u32);
            let y = e.map(x, |x: &u32| x + 1)?;
            Ok((x, y))
        })
        .unwrap();
    let before = a.snapshot();
    assert!(a.edit(|e| e.replace(x, y, |y: &u32| *y)).is_err());
    assert_eq!(a.snapshot().id(), before.id());
}
