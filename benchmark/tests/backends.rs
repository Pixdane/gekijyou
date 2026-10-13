use benchmark::backends::{Backend, BackendScene};
use bevy_tasks::TaskPoolBuilder;
#[test]
fn every_backend_matches_full_oracle_with_explicit_initialization_and_sharing_costs() {
    let pool = TaskPoolBuilder::new().num_threads(2).build();
    for periodic in [false, true] {
        for rounds in [0, 64] {
            let scene = BackendScene::new(4, 16, rounds, periodic);
            for backend in Backend::ALL {
                let result = scene.run(backend, &pool);
                scene.verify(&result);
                match backend {
                    Backend::EntityParallelDfs => {
                        assert_eq!(result.initialized_evaluators, 4);
                        assert_eq!(result.calls, 128);
                    }
                    Backend::Dfs => {
                        assert_eq!(result.initialized_evaluators, 1);
                        assert_eq!(result.calls, 80);
                    }
                    _ => assert_eq!(result.calls, if periodic { 48 } else { 80 }),
                }
            }
        }
    }
}
