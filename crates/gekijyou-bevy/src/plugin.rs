use crate::{AnimationRuntime, Bindings, PlaybackState};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;

/// Explicit phase order in Update, before Transform propagation in PostUpdate.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnimationSet {
    InputSync,
    Prepare,
    Apply,
}
/// Installs a minimal synchronous pipeline using one global graph resource.
/// Playback systems read prepared outputs only. Pause gates all ECS input commits.
pub struct AnimationPlugin;
impl Plugin for AnimationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AnimationRuntime>()
            .init_resource::<Bindings>()
            .configure_sets(
                Update,
                (
                    AnimationSet::InputSync,
                    AnimationSet::Prepare,
                    AnimationSet::Apply,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    sync.in_set(AnimationSet::InputSync),
                    prepare.in_set(AnimationSet::Prepare),
                    apply.in_set(AnimationSet::Apply),
                ),
            );
    }
}
fn sync(world: &mut World) {
    world.resource_scope(|world, mut bindings: Mut<Bindings>| {
        world.resource_scope(|world, mut runtime: Mut<AnimationRuntime>| {
            bindings.prune(world, runtime.snapshot());
            if runtime.playback() == PlaybackState::Paused {
                runtime.last_error = runtime.sync_inputs(world, &bindings).err();
            }
        });
    });
}
fn prepare(world: &mut World) {
    world.resource_scope(|world, mut runtime: Mut<AnimationRuntime>| {
        if runtime.last_error.is_some() {
            return;
        }
        let frame = runtime.frame();
        runtime.last_error = runtime
            .prepare(world.resource::<Bindings>(), &[frame])
            .err();
    });
}
fn apply(world: &mut World) {
    world.resource_scope(|world, bindings: Mut<Bindings>| {
        world.resource_scope(|world, mut runtime: Mut<AnimationRuntime>| {
            if runtime.last_error.is_none() {
                runtime.last_error = bindings.apply(world, &runtime, runtime.frame()).err();
            }
        });
    });
}
