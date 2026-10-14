use bevy_ecs::{prelude::*, world::World};
use bevy_math::{Quat, Vec3};
use bevy_transform::components::Transform;
use gekijyou_bevy::{BindingError, Bindings, InputField, OutputField};
use gekijyou_computation::{EvalError, Evaluator, FrameIndex, GraphState, RequestBatch};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Component, Clone)]
struct Offset(u64);
#[derive(Component)]
struct Tint(bevy_color::Color);

#[test]
fn arbitrary_frames_share_nodes_and_owned_results_remain_reusable_after_input_edits() {
    let mut world = World::new();
    let source = world.spawn(Offset(7)).id();
    let a = world
        .spawn((Transform::default(), Tint(bevy_color::Color::WHITE)))
        .id();
    let b = world.spawn(Transform::default()).id();
    let mut state = GraphState::new();
    let (input, position, color) = state
        .edit(|e| {
            let i = e.input::<u64>();
            let p = e.map((e.frame(), i.node()), |f: &FrameIndex, x: &u64| {
                Vec3::new((f.0 + x) as f32, 0., 0.)
            })?;
            let c = e.map(e.frame(), |f: &FrameIndex| {
                if f.0 == 20 {
                    bevy_color::Color::BLACK
                } else {
                    bevy_color::Color::WHITE
                }
            })?;
            Ok((i, p, c))
        })
        .unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_input(
            &state.snapshot(),
            source,
            input,
            InputField::new(|c: &Offset| c.0),
        )
        .unwrap();
    for entity in [a, b] {
        bindings
            .bind_output(
                &state.snapshot(),
                entity,
                position,
                OutputField::transform_translation(),
            )
            .unwrap();
    }
    bindings
        .bind_output(
            &state.snapshot(),
            a,
            color,
            OutputField::new(|c: &mut Tint, v: &bevy_color::Color| c.0 = *v),
        )
        .unwrap();
    bindings.sync_inputs(&world, &mut state).unwrap();
    let snapshot = state.snapshot();
    let mut bound = bindings
        .request(
            &snapshot,
            &[
                FrameIndex(100),
                FrameIndex(20),
                FrameIndex(100),
                FrameIndex(300),
            ],
        )
        .unwrap();
    let batch = bound.take_batch().unwrap();
    assert_eq!(batch.len(), 6);
    assert!(matches!(bound.take_batch(), Err(BindingError::BatchTaken)));
    // Editing ECS and GraphState leaves the previously selected basis immutable.
    world.get_mut::<Offset>(source).unwrap().0 = 99;
    bindings.sync_inputs(&world, &mut state).unwrap();
    let mut eval = Evaluator::from_snapshot(&snapshot);
    let results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    for f in [100, 20, 300, 20] {
        bindings
            .apply(&mut world, &snapshot, &bound, &results, FrameIndex(f))
            .unwrap();
        assert_eq!(
            world.get::<Transform>(a).unwrap().translation.x,
            (f + 7) as f32
        );
        assert_eq!(
            world.get::<Transform>(b).unwrap().translation.x,
            (f + 7) as f32
        );
    }
    assert_eq!(world.get::<Tint>(a).unwrap().0, bevy_color::Color::BLACK);
    assert_eq!(
        bindings.apply(
            &mut world,
            &state.snapshot(),
            &bound,
            &results,
            FrameIndex(20)
        ),
        Err(BindingError::SnapshotMismatch)
    );
    assert_eq!(
        bindings.apply(&mut world, &snapshot, &bound, &results, FrameIndex(1)),
        Err(BindingError::MissingFrame)
    );
}
#[test]
fn multiple_input_writers_and_overlapping_output_ranges_fail() {
    let mut state = GraphState::new();
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let (input, pos, rot, all) = state
        .edit(|e| {
            let i = e.input::<Vec3>();
            Ok((
                i,
                e.constant(Vec3::ONE),
                e.constant(Quat::IDENTITY),
                e.constant(Transform::default()),
            ))
        })
        .unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_input(
            &state.snapshot(),
            entity,
            input,
            InputField::transform_translation(),
        )
        .unwrap();
    assert!(matches!(
        bindings.bind_input(
            &state.snapshot(),
            entity,
            input,
            InputField::transform_translation()
        ),
        Err(BindingError::InputConflict)
    ));
    // Using this same ECS field as output would create hidden feedback.
    assert!(matches!(
        bindings.bind_output(
            &state.snapshot(),
            entity,
            pos,
            OutputField::transform_translation()
        ),
        Err(BindingError::FeedbackConflict)
    ));
    let other = world.spawn(Transform::default()).id();
    bindings
        .bind_output(
            &state.snapshot(),
            other,
            pos,
            OutputField::transform_translation(),
        )
        .unwrap();
    bindings
        .bind_output(
            &state.snapshot(),
            other,
            rot,
            OutputField::transform_rotation(),
        )
        .unwrap();
    assert!(matches!(
        bindings.bind_output(
            &state.snapshot(),
            other,
            pos,
            OutputField::transform_translation()
        ),
        Err(BindingError::OutputConflict)
    ));
    assert!(matches!(
        bindings.bind_output(&state.snapshot(), other, all, OutputField::component()),
        Err(BindingError::OutputConflict)
    ));
}
#[test]
fn despawn_pruning_binding_revisions_and_deleted_nodes_are_checked() {
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(Vec3::ONE))).unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    let mut bindings = Bindings::new();
    let handle = bindings
        .bind_output(
            &snapshot,
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    let mut bound = bindings
        .request(&snapshot, &[FrameIndex(1), FrameIndex(2)])
        .unwrap();
    let results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(bound.take_batch().unwrap())
        .unwrap();
    world.despawn(entity);
    let replacement = world.spawn(Transform::default()).id();
    bindings
        .apply(&mut world, &snapshot, &bound, &results, FrameIndex(1))
        .unwrap();
    assert_eq!(
        world.get::<Transform>(replacement).unwrap().translation,
        Vec3::ZERO
    );
    assert_eq!(bindings.prune(&world, &snapshot), 1);
    assert_eq!(
        bindings.apply(&mut world, &snapshot, &bound, &results, FrameIndex(2)),
        Err(BindingError::StaleBindings)
    );
    assert!(matches!(
        bindings.request_selected(&snapshot, &[FrameIndex(2)], &[handle]),
        Err(BindingError::UnknownOutput)
    ));
    bindings
        .bind_output(
            &snapshot,
            replacement,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    state.edit(|e| e.remove(node)).unwrap();
    assert!(
        bindings
            .request(&state.snapshot(), &[FrameIndex(2)])
            .is_err()
    );
    assert_eq!(bindings.prune(&world, &state.snapshot()), 1);
}

#[test]
fn completeness_is_the_selected_output_set_not_the_whole_registry() {
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let mut state = GraphState::new();
    let (position, rotation) = state
        .edit(|e| {
            let p = e.constant(Vec3::ONE);
            let missing = e.input::<Quat>();
            let r = e.map(missing.node(), |q: &Quat| *q)?;
            Ok((p, r))
        })
        .unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    let mut bindings = Bindings::new();
    let pos = bindings
        .bind_output(
            &snapshot,
            entity,
            position,
            OutputField::transform_translation(),
        )
        .unwrap();
    bindings
        .bind_output(
            &snapshot,
            entity,
            rotation,
            OutputField::transform_rotation(),
        )
        .unwrap();
    let mut all = bindings.request(&snapshot, &[FrameIndex(0)]).unwrap();
    assert!(
        eval.bind(snapshot.clone())
            .unwrap()
            .evaluate(all.take_batch().unwrap())
            .is_err()
    );
    assert_eq!(
        world.get::<Transform>(entity).unwrap().translation,
        Vec3::ZERO
    );
    let mut selected = bindings
        .request_selected(&snapshot, &[FrameIndex(0)], &[pos, pos])
        .unwrap();
    let batch = selected.take_batch().unwrap();
    assert_eq!(batch.len(), 1);
    let results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    bindings
        .apply(&mut world, &snapshot, &selected, &results, FrameIndex(0))
        .unwrap();
    assert_eq!(
        world.get::<Transform>(entity).unwrap().translation,
        Vec3::ONE
    );
    assert_eq!(
        world.get::<Transform>(entity).unwrap().rotation,
        Quat::IDENTITY
    );
}

#[test]
fn missing_components_and_foreign_batches_are_rejected_before_any_write() {
    let mut world = World::new();
    let a = world.spawn(Transform::default()).id();
    let b = world.spawn(Transform::default()).id();
    let mut state = GraphState::new();
    let node = state.edit(|e| Ok(e.constant(Vec3::ONE))).unwrap();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    let mut bindings = Bindings::new();
    for entity in [a, b] {
        bindings
            .bind_output(
                &snapshot,
                entity,
                node,
                OutputField::transform_translation(),
            )
            .unwrap();
    }
    let mut bound = bindings.request(&snapshot, &[FrameIndex(0)]).unwrap();
    let results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(bound.take_batch().unwrap())
        .unwrap();
    world.entity_mut(b).remove::<Transform>();
    assert_eq!(
        bindings.apply(&mut world, &snapshot, &bound, &results, FrameIndex(0)),
        Err(BindingError::MissingComponent)
    );
    assert_eq!(world.get::<Transform>(a).unwrap().translation, Vec3::ZERO);
    world.entity_mut(b).insert(Transform::default());
    let mut other = RequestBatch::new();
    other.at(FrameIndex(0)).request(node);
    let other_results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(other)
        .unwrap();
    assert_eq!(
        bindings.apply(&mut world, &snapshot, &bound, &other_results, FrameIndex(0)),
        Err(BindingError::Evaluation(EvalError::ForeignRequest))
    );
    assert_eq!(world.get::<Transform>(a).unwrap().translation, Vec3::ZERO);
    let mut other_registry = Bindings::new();
    other_registry
        .bind_output(&snapshot, a, node, OutputField::transform_translation())
        .unwrap();
    other_registry
        .bind_output(&snapshot, b, node, OutputField::transform_translation())
        .unwrap();
    assert_eq!(other_registry.revision(), bindings.revision());
    assert_eq!(
        other_registry.apply(&mut world, &snapshot, &bound, &results, FrameIndex(0)),
        Err(BindingError::ForeignBindings)
    );
    let foreign = GraphState::new().snapshot();
    assert_eq!(
        bindings.apply(&mut world, &foreign, &bound, &results, FrameIndex(0)),
        Err(BindingError::Evaluation(EvalError::ForeignGraph))
    );
    state
        .edit(|e| e.replace(node, (), || Vec3::splat(2.)))
        .unwrap();
    let newer_results = eval
        .bind(state.snapshot())
        .unwrap()
        .evaluate(RequestBatch::new())
        .unwrap();
    assert_eq!(
        bindings.apply(&mut world, &snapshot, &bound, &newer_results, FrameIndex(0)),
        Err(BindingError::SnapshotMismatch)
    );
}

#[test]
fn failed_or_panicking_input_sync_is_one_atomic_transaction() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut world = World::new();
    let good = world.spawn(Offset(9)).id();
    let missing = world.spawn_empty().id();
    let mut state = GraphState::new();
    let (a, b) = state
        .edit(|e| {
            let a = e.input::<u64>();
            let b = e.input::<u64>();
            e.set(a, 1)?;
            e.set(b, 2)?;
            Ok((a, b))
        })
        .unwrap();
    let snapshot = state.snapshot();
    let mut bindings = Bindings::new();
    bindings
        .bind_input(&snapshot, good, a, InputField::new(|c: &Offset| c.0))
        .unwrap();
    bindings
        .bind_input(&snapshot, missing, b, InputField::new(|c: &Offset| c.0))
        .unwrap();
    assert_eq!(
        bindings.sync_inputs(&world, &mut state),
        Err(BindingError::MissingComponent)
    );
    assert_eq!(state.snapshot().id(), snapshot.id());
    world.entity_mut(missing).insert(Offset(10));
    bindings.sync_inputs(&world, &mut state).unwrap();
    assert_eq!(state.get(a).unwrap(), Some(&9));
    let mut panics = Bindings::new();
    panics
        .bind_input(
            &state.snapshot(),
            good,
            a,
            InputField::new(|c: &Offset| c.0 + 1),
        )
        .unwrap();
    panics
        .bind_input(
            &state.snapshot(),
            missing,
            b,
            InputField::new(|_: &Offset| -> u64 { panic!("extractor") }),
        )
        .unwrap();
    let before = state.snapshot().id();
    assert!(catch_unwind(AssertUnwindSafe(|| panics.sync_inputs(&world, &mut state))).is_err());
    assert_eq!(state.snapshot().id(), before);
    assert_eq!(state.get(a).unwrap(), Some(&9));
    world.despawn(missing);
    assert_eq!(
        bindings.sync_inputs(&world, &mut state),
        Err(BindingError::MissingEntity)
    );
}

#[test]
fn infrastructure_plugin_does_no_work_and_explicit_apply_precedes_propagation() {
    use bevy_app::{App, Update};
    use bevy_transform::{TransformPlugin, components::GlobalTransform};
    use gekijyou_bevy::{BindingPlugin, BindingSet};
    let mut app = App::new();
    app.add_plugins((TransformPlugin, BindingPlugin));
    let entity = app
        .world_mut()
        .spawn((Transform::default(), GlobalTransform::default()))
        .id();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut state = GraphState::new();
    let node = state
        .edit(|e| {
            let calls = calls.clone();
            e.map((), move || {
                calls.fetch_add(1, Ordering::Relaxed);
                Vec3::ONE
            })
        })
        .unwrap();
    let snapshot = state.snapshot();
    app.world_mut()
        .resource_mut::<Bindings>()
        .bind_output(
            &snapshot,
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        app.world().get::<Transform>(entity).unwrap().translation,
        Vec3::ZERO
    );
    let mut bound = app
        .world()
        .resource::<Bindings>()
        .request(&snapshot, &[FrameIndex(100)])
        .unwrap();
    let results = Evaluator::new(&state)
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(bound.take_batch().unwrap())
        .unwrap();
    app.add_systems(
        Update,
        (move |world: &mut World| {
            world.resource_scope(|world, bindings: Mut<Bindings>| {
                bindings
                    .apply(world, &snapshot, &bound, &results, FrameIndex(100))
                    .unwrap();
            });
        })
        .in_set(BindingSet::Apply),
    );
    app.update();
    app.update();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        app.world()
            .get::<GlobalTransform>(entity)
            .unwrap()
            .translation(),
        Vec3::ONE
    );
}

#[test]
fn shared_ecs_inputs_freeze_large_non_clone_objects_without_new_snapshots_on_repeat() {
    use std::sync::Arc;
    struct Large(Vec<u8>);
    #[derive(Component)]
    struct Source(Arc<Large>);
    let payload = Arc::new(Large(vec![1; 1024 * 1024]));
    let mut world = World::new();
    let entity = world.spawn(Source(payload.clone())).id();
    let mut state = GraphState::new();
    let input = state.edit(|e| Ok(e.input::<Large>())).unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind_input(
            &state.snapshot(),
            entity,
            input,
            InputField::shared(|s: &Source| s.0.clone()),
        )
        .unwrap();
    bindings.sync_inputs(&world, &mut state).unwrap();
    let snapshot = state.snapshot();
    assert!(Arc::ptr_eq(
        &payload,
        &snapshot.get_shared(input).unwrap().unwrap()
    ));
    assert_eq!(snapshot.get(input).unwrap().unwrap().0.len(), 1024 * 1024);
    bindings.sync_inputs(&world, &mut state).unwrap();
    assert_eq!(state.snapshot().id(), snapshot.id());
}

#[test]
fn empty_output_selection_still_requires_its_own_batch_identity() {
    let state = GraphState::new();
    let snapshot = state.snapshot();
    let mut eval = Evaluator::new(&state);
    let bindings = Bindings::new();
    let mut bound = bindings
        .request_selected(&snapshot, &[FrameIndex(20)], &[])
        .unwrap();
    let results = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(bound.take_batch().unwrap())
        .unwrap();
    let foreign = eval
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(RequestBatch::new())
        .unwrap();
    let mut world = World::new();
    assert_eq!(
        bindings.apply(&mut world, &snapshot, &bound, &foreign, FrameIndex(20)),
        Err(BindingError::Evaluation(EvalError::ForeignRequest))
    );
    bindings
        .apply(&mut world, &snapshot, &bound, &results, FrameIndex(20))
        .unwrap();
}
