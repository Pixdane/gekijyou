//! Actual ECS scenario: report input extraction, evaluation and application
//! separately. Each iteration verifies every entity against an independent oracle.
//! sync_ns includes the atomic GraphState input commit; snapshot_ns measures only
//! access to the resulting immutable snapshot. No automatic schedule is installed.
use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_transform::components::Transform;
use gekijyou_bevy::{Bindings, InputField, OutputField};
use gekijyou_computation::{Evaluator, FrameIndex, GraphState};
use std::time::Instant;
#[derive(Component)]
struct Offset(u64);
fn main() {
    let mut world = World::new();
    let source = world.spawn(Offset(10)).id();
    let mut state = GraphState::new();
    let mut bindings = Bindings::new();
    let (input, nodes) = state
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
            &state.snapshot(),
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
                &state.snapshot(),
                entity,
                node,
                OutputField::transform_translation(),
            )
            .unwrap();
    }
    let mut evaluator = Evaluator::new(&state);
    println!("sample\tentities\tsync_ns\tsnapshot_ns\trequest_ns\tevaluate_ns\tapply_ns");
    for sample in 0..9 {
        let mut sync = 0;
        let mut snapshot_time = 0;
        let mut request_time = 0;
        let mut evaluate = 0;
        let mut apply = 0;
        for f in 0..128 {
            let start = Instant::now();
            bindings.sync_inputs(&world, &mut state).unwrap();
            sync += start.elapsed().as_nanos();
            let start = Instant::now();
            let snapshot = state.snapshot();
            snapshot_time += start.elapsed().as_nanos();
            let start = Instant::now();
            let mut bound = bindings.request(&snapshot, &[FrameIndex(f)]).unwrap();
            let batch = bound.take_batch().unwrap();
            request_time += start.elapsed().as_nanos();
            let start = Instant::now();
            let results = evaluator
                .bind(snapshot.clone())
                .unwrap()
                .evaluate(batch)
                .unwrap();
            evaluate += start.elapsed().as_nanos();
            let start = Instant::now();
            bindings
                .apply(&mut world, &snapshot, &bound, &results, FrameIndex(f))
                .unwrap();
            apply += start.elapsed().as_nanos();
            for (actor, &entity) in entities.iter().enumerate() {
                assert_eq!(
                    world.get::<Transform>(entity).unwrap().translation,
                    Vec3::new((f + 10 + actor as u64) as f32, actor as f32, 0.)
                );
            }
        }
        println!(
            "{sample}\t512\t{}\t{}\t{}\t{}\t{}",
            sync / 128,
            snapshot_time / 128,
            request_time / 128,
            evaluate / 128,
            apply / 128
        );
    }
}
