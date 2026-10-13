use crate::Bindings;
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;

/// Optional ordering for caller-installed Update systems. Transform propagation
/// runs in PostUpdate; callers can also use these sets in their own schedules.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingSet {
    InputSync,
    Evaluate,
    Apply,
}

/// Installs only the binding registry and Update set ordering. No input extraction,
/// request construction, evaluation or result application runs automatically.
pub struct BindingPlugin;
impl Plugin for BindingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Bindings>().configure_sets(
            Update,
            (
                BindingSet::InputSync,
                BindingSet::Evaluate,
                BindingSet::Apply,
            )
                .chain(),
        );
    }
}
