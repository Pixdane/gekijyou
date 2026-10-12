use benchmark::scene::{Config, ValueComparison, Workload, direct_checksum, inputs};

#[test]
fn scene_workload_matches_direct_calculation_for_all_requested_values() {
    for (actors, stages, rounds) in [(1, 0, 0), (7, 3, 5), (32, 16, 0), (32, 16, 128)] {
        let workload = Workload::new(Config {
            actors,
            stages,
            rounds,
        });
        let mut evaluator = workload.evaluator();
        assert!(workload.node_count() > actors);
        for variant in [0, 1, 19, u64::MAX] {
            let input = inputs(16, variant);
            let direct = workload.direct(&input);
            let graph = workload.evaluate(&mut evaluator, workload.prepare(&input));
            let expected: Vec<_> = direct.iter().map(|output| output.snapshot()).collect();
            assert_eq!(
                graph.snapshots(),
                expected,
                "actors={actors}, stages={stages}, rounds={rounds}, variant={variant}"
            );
            assert_eq!(graph.checksum(), direct_checksum(&direct));
            assert!(graph.duplicates_share_storage());
        }
    }
}

use benchmark::scenario::{Case, Suite, evaluate};

#[test]
fn memo_scene_loads_preserve_complete_direct_results_and_expected_reuse() {
    for &case in Suite::Memo.cases() {
        let data = case.datasets(1, 32);
        for &configuration in Suite::Memo.configurations() {
            let mut workload = Workload::configured(
                Config {
                    actors: 7,
                    stages: 3,
                    rounds: 5,
                },
                configuration.policy,
                ValueComparison::Content,
            );
            case.verify(&mut workload, &data);
            let mut evaluator = case.warm(&workload, &data);
            if case == Case::GraphEdit {
                workload.replace_phase();
            }
            evaluate(&workload, &mut evaluator, &data[0]);
            let stats = evaluator.stats();
            if configuration.policy == gekijyou_computation::CachePolicy::Transient {
                assert_eq!(stats.memo_hits + stats.memo_misses, 0);
                assert!(stats.computation_calls > 1);
            } else {
                match case {
                    Case::HighHit | Case::GraphEdit => {
                        assert!(stats.computation_calls <= 2);
                        assert!(
                            stats.memo_hits as f64 / (stats.memo_hits + stats.memo_misses) as f64
                                > 0.95
                        );
                    }
                    Case::AllMiss => {
                        assert!(
                            (stats.memo_hits as f64 / (stats.memo_hits + stats.memo_misses) as f64)
                                < 0.02
                        )
                    }
                    Case::History if matches!(configuration.policy, gekijyou_computation::CachePolicy::Lru { capacity } if capacity>=4) =>
                    {
                        assert_eq!(stats.computation_calls, 0);
                        assert_eq!(
                            stats.memo_hits as f64 / (stats.memo_hits + stats.memo_misses) as f64,
                            1.0
                        );
                    }
                    Case::History => assert!(stats.computation_calls > 1),
                    _ => unreachable!("Memo suite case"),
                }
            }
            assert!(!case.name().is_empty());
        }
    }
}
