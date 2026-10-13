//! Actual ECS scenario: report input extraction, evaluation and application
//! separately. Each iteration verifies every entity against an independent oracle.
use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_transform::components::Transform;
use gekijyou_bevy::{AnimationRuntime, Bindings, InputField, OutputField};
use gekijyou_computation::FrameIndex;
use std::time::Instant;
#[derive(Component)]
struct Offset(u64);
fn main() {
    let mut world = World::new();
    let source = world.spawn(Offset(10)).id();
    let mut runtime = AnimationRuntime::new(2);
    let mut bindings = Bindings::new();
    let (input, nodes) = runtime
        .edit(|e| {
            let i = e.input::<u64>();
            let mut nodes = Vec::new();
            for actor in 0..512 {
                nodes.push(
                    e.map((e.frame(), i.node()), move |f: &FrameIndex, x: &u64| {
                        Vec3::new((f.0 + x + actor) as f32, actor as f32, 0.)
                    })?,
                );
            }
            Ok((i, nodes))
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
    let mut entities = Vec::new();
    for node in nodes {
        let entity = world.spawn(Transform::default()).id();
        entities.push(entity);
        bindings
            .bind_output(
                runtime.snapshot(),
                entity,
                node,
                OutputField::transform_translation(),
            )
            .unwrap();
    }
    println!("sample\tentities\tsync_ns\tevaluate_ns\tapply_ns");
    for sample in 0..9 {
        let mut sync = 0;
        let mut evaluate = 0;
        let mut apply = 0;
        for f in 0..128 {
            runtime.pause();
            let start = Instant::now();
            runtime.sync_inputs(&world, &bindings).unwrap();
            sync += start.elapsed().as_nanos();
            runtime.play();
            let start = Instant::now();
            runtime.prepare(&bindings, &[FrameIndex(f)]).unwrap();
            evaluate += start.elapsed().as_nanos();
            let start = Instant::now();
            bindings.apply(&mut world, &runtime, FrameIndex(f)).unwrap();
            apply += start.elapsed().as_nanos();
            for (actor, &entity) in entities.iter().enumerate() {
                assert_eq!(
                    world.get::<Transform>(entity).unwrap().translation,
                    Vec3::new((f + 10 + actor as u64) as f32, actor as f32, 0.)
                );
            }
        }
        println!(
            "{sample}\t512\t{}\t{}\t{}",
            sync / 128,
            evaluate / 128,
            apply / 128
        );
    }
}
