//! Typed ECS bindings around one global computation graph. Input synchronization
//! is an explicit edit phase; playing never samples previously applied ECS output
//! as a new input. Apply reads only complete prepared frames from the runtime.
//! Asset-dependent computations need immutable content snapshots or explicit content
//! versions; an asset Handle identity alone does not guarantee unchanged contents.
//!
//! ```
//! use bevy_ecs::prelude::*;
//! use bevy_math::Vec3;
//! use bevy_transform::components::Transform;
//! use gekijyou_computation::FrameIndex;
//! use gekijyou_bevy::{AnimationRuntime, Bindings, OutputField};
//! let mut world = World::new();
//! let entity = world.spawn(Transform::default()).id();
//! let mut runtime = AnimationRuntime::new(8);
//! let node = runtime.edit(|e| e.map(e.frame(), |f: &FrameIndex| Vec3::new(f.0 as f32, 0., 0.)))?;
//! let mut bindings = Bindings::new();
//! bindings.bind_output(runtime.snapshot(), entity, node, OutputField::transform_translation())?;
//! runtime.play();
//! runtime.prepare(&bindings, &[FrameIndex(12)])?;
//! bindings.apply(&mut world, &runtime, FrameIndex(12))?;
//! assert_eq!(world.get::<Transform>(entity).unwrap().translation.x, 12.);
//! # Ok::<(), gekijyou_bevy::BindingError>(())
//! ```

mod bindings;
mod runtime;
pub use bindings::{BindingError, Bindings, InputField, OutputField, WriteRange};
pub use runtime::{
    AnimationRuntime, FrameStatus, PlaybackState, PreparedResults, RuntimeGeneration,
};
mod plugin;
pub use plugin::{AnimationPlugin, AnimationSet};
mod tasks;
pub use tasks::{FinishedTask, TaskCoordinator, TaskId, TaskLimits, TaskStats, TaskStatus};
mod executor;
pub use executor::{AsyncDriver, BevyExecutor};
