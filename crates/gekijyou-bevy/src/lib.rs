//! Explicit ECS adaptation for deterministic computation graphs.
//! Callers own GraphState, Evaluator, request timing and completed results. The
//! optional BindingPlugin installs infrastructure only. Input synchronization,
//! output selection, evaluation and Apply are independently scheduled operations.
//! Asset-dependent computations need immutable content snapshots or explicit
//! content versions; a Handle alone does not freeze the asset contents.
//!
//! ```
//! use bevy_ecs::prelude::*;
//! use bevy_math::Vec3;
//! use bevy_transform::components::Transform;
//! use gekijyou_computation::{GraphState, Evaluator, FrameIndex};
//! use gekijyou_bevy::{Bindings, OutputField};
//! let mut world = World::new();
//! let entity = world.spawn(Transform::default()).id();
//! let mut state = GraphState::new();
//! let node = state.edit(|e| e.map(e.frame(), |f: &FrameIndex| Vec3::new(f.0 as f32, 0., 0.)))?;
//! let mut evaluator = Evaluator::new(&state);
//! let snapshot = state.snapshot();
//! let mut bindings = Bindings::new();
//! bindings.bind_output(&snapshot, entity, node, OutputField::transform_translation())?;
//! let mut request = bindings.request(&snapshot, &[FrameIndex(100), FrameIndex(20)])?;
//! let results = evaluator.bind(snapshot.clone())?.evaluate(request.take_batch()?)?;
//! bindings.apply(&mut world, &snapshot, &request, &results, FrameIndex(20))?;
//! assert_eq!(world.get::<Transform>(entity).unwrap().translation.x, 20.);
//! // The caller may retain and consume the same completed result again.
//! bindings.apply(&mut world, &snapshot, &request, &results, FrameIndex(100))?;
//! # Ok::<(), gekijyou_bevy::BindingError>(())
//! ```

mod bindings;
mod executor;
mod plugin;
mod tasks;
pub use bindings::{
    BindingError, Bindings, BoundRequest, InputField, OutputBinding, OutputField, WriteRange,
};
pub use executor::{AsyncDriver, BevyExecutor};
pub use plugin::{BindingPlugin, BindingSet};
pub use tasks::{
    FinishedTask, RequestGeneration, TaskCoordinator, TaskId, TaskLimits, TaskStats, TaskStatus,
};
