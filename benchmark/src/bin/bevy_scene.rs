//! Actual ECS scenario: report input extraction, evaluation and application
//! separately. Each sample builds a fresh world, graph, evaluator and binding set,
//! so no cache warm-up leaks across samples. Every entity/frame pair is verified
//! against an independent oracle outside phase timing.
//!
//! `repeated_sync` synchronizes ECS inputs and takes a snapshot every frame (128/128).
//! `frozen_snapshot` synchronizes and snapshots once, then reuses that immutable
//! snapshot for all 128 frames (1/1); it shows the setup/snapshot cost boundary, not
//! a reduction in the evaluation algorithm's own work. All phase columns are
//! whole-sample nanosecond totals, so the cases are directly comparable.
//! No automatic schedule is installed; the caller drives every phase.

use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_transform::components::Transform;
use gekijyou_bevy::{Bindings, InputField, OutputField};
use gekijyou_computation::{EvaluationSnapshot, Evaluator, FrameIndex, GraphState};
use std::time::Instant;

const ENTITIES: usize = 512;
const FRAMES: u64 = 128;
const SAMPLES: usize = 9;

#[derive(Component)]
struct Offset(u64);

/// Fresh world, graph, binding registry and output entities for one sample.
fn scene() -> (World, GraphState, Bindings, Vec<Entity>) {
    let mut world = World::new();
    let source = world.spawn(Offset(10)).id();
    let mut state = GraphState::new();
    let (input, nodes) = state
        .edit(|e| {
            let i = e.input::<u64>();
            let mut nodes = Vec::new();
            for actor in 0..ENTITIES {
                nodes.push(
                    e.map((e.frame(), i.node()), move |f: &FrameIndex, x: &u64| {
                        Vec3::new((f.0 + x + actor as u64) as f32, actor as f32, 0.)
                    })?,
                );
            }
            Ok((i, nodes))
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
    let mut entities = Vec::with_capacity(nodes.len());
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
    (world, state, bindings, entities)
}

/// One request/evaluate/apply pass; `timers` accumulates request/evaluate/apply.
/// Oracle checks after `apply` stay outside all phase timing.
fn pass(
    world: &mut World,
    bindings: &Bindings,
    evaluator: &mut Evaluator,
    entities: &[Entity],
    snapshot: &EvaluationSnapshot,
    frame: u64,
    timers: &mut [u128; 3],
) {
    let start = Instant::now();
    let mut bound = bindings.request(snapshot, &[FrameIndex(frame)]).unwrap();
    let batch = bound.take_batch().unwrap();
    timers[0] += start.elapsed().as_nanos();

    let start = Instant::now();
    let results = evaluator
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    timers[1] += start.elapsed().as_nanos();

    let start = Instant::now();
    bindings
        .apply(world, snapshot, &bound, &results, FrameIndex(frame))
        .unwrap();
    timers[2] += start.elapsed().as_nanos();

    for (actor, &entity) in entities.iter().enumerate() {
        // Full Transform comparison, so rotation and scale must remain identity
        // and cannot be silently dropped by the application path.
        assert_eq!(
            world.get::<Transform>(entity).unwrap(),
            &Transform::from_translation(Vec3::new(
                (frame + 10 + actor as u64) as f32,
                actor as f32,
                0.,
            )),
        );
    }
}

fn main() {
    println!(
        "sample\tcase\tentities\tframes\tsync_count\tsnapshot_count\tsync_ns\tsnapshot_ns\trequest_ns\tevaluate_ns\tapply_ns"
    );
    for case in ["repeated_sync", "frozen_snapshot"] {
        for sample in 0..SAMPLES {
            let (mut world, mut state, bindings, entities) = scene();
            let mut evaluator = Evaluator::new(&state);
            let mut timers = [0_u128; 3];
            let mut sync_ns = 0_u128;
            let mut snapshot_ns = 0_u128;
            let mut sync_count = 0_usize;
            let mut snapshot_count = 0_usize;

            if case == "frozen_snapshot" {
                let start = Instant::now();
                bindings.sync_inputs(&world, &mut state).unwrap();
                sync_ns += start.elapsed().as_nanos();
                sync_count += 1;
                let start = Instant::now();
                let snapshot = state.snapshot();
                snapshot_ns += start.elapsed().as_nanos();
                snapshot_count += 1;
                for frame in 0..FRAMES {
                    pass(
                        &mut world,
                        &bindings,
                        &mut evaluator,
                        &entities,
                        &snapshot,
                        frame,
                        &mut timers,
                    );
                }
            } else {
                for frame in 0..FRAMES {
                    let start = Instant::now();
                    bindings.sync_inputs(&world, &mut state).unwrap();
                    sync_ns += start.elapsed().as_nanos();
                    sync_count += 1;
                    let start = Instant::now();
                    let snapshot = state.snapshot();
                    snapshot_ns += start.elapsed().as_nanos();
                    snapshot_count += 1;
                    pass(
                        &mut world,
                        &bindings,
                        &mut evaluator,
                        &entities,
                        &snapshot,
                        frame,
                        &mut timers,
                    );
                }
            }

            println!(
                "{sample}\t{case}\t{ENTITIES}\t{FRAMES}\t{sync_count}\t{snapshot_count}\t{sync_ns}\t{snapshot_ns}\t{}\t{}\t{}",
                timers[0], timers[1], timers[2]
            );
        }
    }
}
