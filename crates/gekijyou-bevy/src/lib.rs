//! Typed ECS bindings around one global computation graph. Input synchronization
//! is an explicit edit phase; playing never samples previously applied ECS output
//! as a new input. Apply reads only complete prepared frames from the runtime.
mod bindings;
mod runtime;
pub use bindings::{BindingError, Bindings, InputField, OutputField, WriteRange};
pub use runtime::{
    AnimationRuntime, FrameStatus, PlaybackState, PreparedResults, RuntimeGeneration,
};
mod plugin;
pub use plugin::{AnimationPlugin, AnimationSet};
