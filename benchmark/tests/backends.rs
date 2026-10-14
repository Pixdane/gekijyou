use benchmark::backends::{
    Backend, BackendScene, CacheLifetime, CacheMode, FramePattern, SceneConfig, Topology,
    benchmark_pool,
};
use gekijyou_bevy::TaskStats;
use gekijyou_computation::PlannerStats;

/// Configured logical in-flight quota for the planner and asynchronous backends.
const QUOTA: usize = 64;

/// The original comparison load is kept explicitly (SharedPhase/Ordered/Off caches)
/// so its stable initialization and sharing costs remain asserted.
#[test]
fn original_load_preserves_explicit_initialization_and_sharing_costs() {
    let pool = benchmark_pool();
    for periodic in [false, true] {
        for rounds in [0, 64] {
            let scene = BackendScene::new(SceneConfig {
                actors: 4,
                frames: 16,
                rounds,
                periodic,
                topology: Topology::SharedPhase,
                frame_pattern: FramePattern::Ordered,
            });
            for backend in Backend::ALL {
                let mut prepared =
                    scene.prepare(backend, &pool, CacheMode::OffOff, CacheLifetime::Cold);
                let result = prepared.run(&pool);
                scene.verify(&result);
                assert_eq!(result.eval.computation_calls, result.calls);
                match backend {
                    Backend::EntityParallelDfs => {
                        assert_eq!(result.initialized_evaluators, 4);
                        assert_eq!(result.calls, 128);
                    }
                    Backend::Dfs => {
                        assert_eq!(result.initialized_evaluators, 1);
                        assert_eq!(result.calls, 80);
                    }
                    Backend::Async(_) => {
                        // Chunking groups physical jobs; it never changes the logical
                        // evaluator or the logical quota contract.
                        assert_eq!(result.initialized_evaluators, 1);
                        assert!(result.planner.peak_in_flight <= QUOTA);
                        assert!(result.tasks.peak_in_flight <= QUOTA);
                        assert_eq!(result.tasks.terminal, result.tasks.dispatched);
                    }
                    _ => assert_eq!(result.calls, if periodic { 48 } else { 80 }),
                }
            }
        }
    }
}

/// Every topology/pattern/cache/lifetime combination must match the independent
/// oracle for every backend, including the asynchronous chunks 1/4/16, and must
/// stay inside the configured logical quota without reporting asynchronous task
/// counters for synchronous backends. Incidental call counts of the new
/// topologies are deliberately not hardcoded.
#[test]
fn every_topology_pattern_cache_and_lifetime_matches_full_oracle_within_quota() {
    let pool = benchmark_pool();
    for topology in Topology::ALL {
        for frame_pattern in FramePattern::ALL {
            for periodic in [false, true] {
                let scene = BackendScene::new(SceneConfig {
                    actors: 4,
                    frames: 16,
                    rounds: 4,
                    periodic,
                    topology,
                    frame_pattern,
                });
                for cache in CacheMode::ALL {
                    for lifetime in CacheLifetime::ALL {
                        for backend in Backend::ALL {
                            let mut prepared = scene.prepare(backend, &pool, cache, lifetime);
                            let result = prepared.run(&pool);
                            scene.verify(&result);
                            assert_eq!(result.eval.computation_calls, result.calls);
                            match lifetime {
                                // Warm reuses preheated evaluators created outside
                                // observation; cold creates them inside it.
                                CacheLifetime::Warm => assert_eq!(result.initialized_evaluators, 0),
                                CacheLifetime::Cold => assert_eq!(
                                    result.initialized_evaluators,
                                    if matches!(backend, Backend::EntityParallelDfs) {
                                        scene.config().actors
                                    } else {
                                        1
                                    }
                                ),
                            }
                            match backend {
                                Backend::Async(_) => {
                                    // Full outputs are already verified above; every
                                    // dispatched logical task must be reclaimed.
                                    assert_eq!(result.tasks.terminal, result.tasks.dispatched);
                                    assert!(result.tasks.peak_in_flight <= QUOTA);
                                    assert!(result.planner.peak_in_flight <= QUOTA);
                                }
                                Backend::Dfs | Backend::EntityParallelDfs => {
                                    assert_eq!(result.planner, PlannerStats::default());
                                    assert_eq!(result.tasks, TaskStats::default());
                                }
                                _ => {
                                    assert!(result.planner.peak_in_flight <= QUOTA);
                                    assert_eq!(result.tasks, TaskStats::default());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
