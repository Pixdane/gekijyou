use bevy_ecs::{prelude::*, world::World};
use bevy_math::{Quat, Vec3};
use bevy_transform::components::Transform;
use gekijyou_bevy::{
    AnimationRuntime, BindingError, Bindings, FrameStatus, InputField, OutputField,
};
use gekijyou_computation::FrameIndex;

#[derive(Component, Clone)]
struct Offset(u64);
#[derive(Component)]
struct Tint(bevy_color::Color);

#[test]
fn one_graph_drives_multiple_entities_from_a_frozen_input() {
    let mut world = World::new();
    let source = world.spawn(Offset(7)).id();
    let a = world
        .spawn((Transform::default(), Tint(bevy_color::Color::WHITE)))
        .id();
    let b = world.spawn(Transform::default()).id();
    let mut runtime = AnimationRuntime::new(4);
    let mut bindings = Bindings::new();
    let (input, position, color) = runtime
        .edit(|e| {
            let i = e.input::<u64>();
            let p = e.map((e.frame(), i.node()), |f: &FrameIndex, x: &u64| {
                Vec3::new((f.0 + x) as f32, 0., 0.)
            })?;
            let c = e.map(e.frame(), |f: &FrameIndex| {
                if f.0 == 1 {
                    bevy_color::Color::BLACK
                } else {
                    bevy_color::Color::WHITE
                }
            })?;
            Ok((i, p, c))
        })
        .unwrap();
    bindings
        .bind_input(
            runtime.snapshot(),
            source,
            input,
            InputField::new(|c: &Offset| c.0),
        )
        .unwrap();
    for entity in [a, b] {
        bindings
            .bind_output(
                runtime.snapshot(),
                entity,
                position,
                OutputField::transform_translation(),
            )
            .unwrap();
    }
    bindings
        .bind_output(
            runtime.snapshot(),
            a,
            color,
            OutputField::new(|c: &mut Tint, v: &bevy_color::Color| c.0 = *v),
        )
        .unwrap();
    runtime.sync_inputs(&world, &bindings).unwrap();
    runtime.play();
    world.get_mut::<Offset>(source).unwrap().0 = 99;
    assert!(matches!(
        runtime.sync_inputs(&world, &bindings),
        Err(BindingError::Playing)
    ));
    runtime.prepare(&bindings, &[FrameIndex(1)]).unwrap();
    assert_eq!(runtime.prepared().status(FrameIndex(1)), FrameStatus::Ready);
    bindings.apply(&mut world, &runtime, FrameIndex(1)).unwrap();
    assert_eq!(world.get::<Transform>(a).unwrap().translation.x, 8.);
    assert_eq!(world.get::<Transform>(b).unwrap().translation.x, 8.);
    assert_eq!(world.get::<Tint>(a).unwrap().0, bevy_color::Color::BLACK);
}
#[test]
fn multiple_input_writers_and_overlapping_output_ranges_fail() {
    let mut runtime = AnimationRuntime::new(2);
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let (input, pos, rot, all) = runtime
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
            runtime.snapshot(),
            entity,
            input,
            InputField::transform_translation(),
        )
        .unwrap();
    assert!(matches!(
        bindings.bind_input(
            runtime.snapshot(),
            entity,
            input,
            InputField::transform_translation()
        ),
        Err(BindingError::InputConflict)
    ));
    // Using this same ECS field as output would create hidden playback feedback.
    assert!(matches!(
        bindings.bind_output(
            runtime.snapshot(),
            entity,
            pos,
            OutputField::transform_translation()
        ),
        Err(BindingError::FeedbackConflict)
    ));
    let other = world.spawn(Transform::default()).id();
    bindings
        .bind_output(
            runtime.snapshot(),
            other,
            pos,
            OutputField::transform_translation(),
        )
        .unwrap();
    bindings
        .bind_output(
            runtime.snapshot(),
            other,
            rot,
            OutputField::transform_rotation(),
        )
        .unwrap();
    assert!(matches!(
        bindings.bind_output(
            runtime.snapshot(),
            other,
            pos,
            OutputField::transform_translation()
        ),
        Err(BindingError::OutputConflict)
    ));
    assert!(matches!(
        bindings.bind_output(runtime.snapshot(), other, all, OutputField::component()),
        Err(BindingError::OutputConflict)
    ));
}
#[test]
fn despawn_deleted_nodes_binding_changes_and_eviction_are_safe() {
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let mut runtime = AnimationRuntime::new(1);
    let mut bindings = Bindings::new();
    let position = runtime.edit(|e| Ok(e.constant(Vec3::ONE))).unwrap();
    bindings
        .bind_output(
            runtime.snapshot(),
            entity,
            position,
            OutputField::transform_translation(),
        )
        .unwrap();
    runtime
        .prepare(&bindings, &[FrameIndex(1), FrameIndex(2)])
        .unwrap();
    assert_eq!(
        runtime.prepared().status(FrameIndex(1)),
        FrameStatus::Missing
    );
    assert!(bindings.apply(&mut world, &runtime, FrameIndex(1)).is_err());
    world.despawn(entity);
    bindings.apply(&mut world, &runtime, FrameIndex(2)).unwrap();
    assert_eq!(bindings.prune(&world, runtime.snapshot()), 1);
    let other = world.spawn(Transform::default()).id();
    bindings
        .bind_output(
            runtime.snapshot(),
            other,
            position,
            OutputField::transform_translation(),
        )
        .unwrap();
    assert!(bindings.apply(&mut world, &runtime, FrameIndex(2)).is_err());
    runtime.edit(|e| e.remove(position)).unwrap();
    assert!(runtime.prepare(&bindings, &[FrameIndex(2)]).is_err());
    assert_eq!(bindings.prune(&world, runtime.snapshot()), 1);
}
#[test]
fn required_outputs_are_published_only_as_a_complete_frame() {
    let mut world = World::new();
    let entity = world.spawn(Transform::default()).id();
    let mut runtime = AnimationRuntime::new(2);
    let mut bindings = Bindings::new();
    let (position, missing) = runtime
        .edit(|e| {
            let p = e.constant(Vec3::ONE);
            let i = e.input::<Quat>();
            let n = e.map(i.node(), |q: &Quat| *q)?;
            Ok((p, n))
        })
        .unwrap();
    bindings
        .bind_output(
            runtime.snapshot(),
            entity,
            position,
            OutputField::transform_translation(),
        )
        .unwrap();
    bindings
        .bind_output(
            runtime.snapshot(),
            entity,
            missing,
            OutputField::transform_rotation(),
        )
        .unwrap();
    assert!(runtime.prepare(&bindings, &[FrameIndex(0)]).is_err());
    assert_eq!(
        runtime.prepared().status(FrameIndex(0)),
        FrameStatus::Failed
    );
    assert!(bindings.apply(&mut world, &runtime, FrameIndex(0)).is_err());
    assert_eq!(
        world.get::<Transform>(entity).unwrap().translation,
        Vec3::ZERO
    );
}

#[test]
fn plugin_applies_before_transform_propagation() {
    use bevy_app::App;
    use bevy_transform::{TransformPlugin, components::GlobalTransform};
    use gekijyou_bevy::AnimationPlugin;
    let mut app = App::new();
    app.add_plugins((TransformPlugin, AnimationPlugin));
    let entity = app
        .world_mut()
        .spawn((Transform::default(), GlobalTransform::default()))
        .id();
    let node = app
        .world_mut()
        .resource_mut::<AnimationRuntime>()
        .edit(|e| Ok(e.constant(Vec3::ONE)))
        .unwrap();
    let snapshot = app
        .world()
        .resource::<AnimationRuntime>()
        .snapshot()
        .clone();
    app.world_mut()
        .resource_mut::<Bindings>()
        .bind_output(
            &snapshot,
            entity,
            node,
            OutputField::transform_translation(),
        )
        .unwrap();
    app.update();
    assert_eq!(
        app.world()
            .get::<GlobalTransform>(entity)
            .unwrap()
            .translation(),
        Vec3::ONE
    );
    assert_eq!(
        app.world().resource::<AnimationRuntime>().last_error(),
        None
    );
}
