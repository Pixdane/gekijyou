#[path = "../examples/support/scene_workload.rs"]
mod fixture;

use fixture::{Config, ValueComparison, Workload, direct_checksum, inputs};

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

#[path = "../examples/support/memo_workload.rs"]
mod memo;

#[test]
fn memo_scene_loads_preserve_complete_direct_results_and_expected_reuse() {
    for case in memo::CASES {
        let data = case.datasets(1);
        for (_, policy) in memo::POLICIES {
            let mut workload = Workload::configured(
                Config {
                    actors: 7,
                    stages: 3,
                    rounds: 5,
                },
                policy,
                ValueComparison::Content,
            );
            memo::verify(&mut workload, case, &data);
            let mut evaluator = memo::warm(&workload, case, &data);
            if case == memo::Case::GraphEdit {
                workload.replace_phase();
            }
            memo::evaluate(&workload, &mut evaluator, &data[0]);
            let stats = evaluator.stats();
            if policy == gekijyou::computation::CachePolicy::Transient {
                assert_eq!(stats.memo_hits + stats.memo_misses, 0);
                assert!(stats.computation_calls > 1);
            } else {
                match case {
                    memo::Case::HighHit | memo::Case::GraphEdit => {
                        assert!(stats.computation_calls <= 2);
                        assert!(memo::hit_rate(stats) > 0.95);
                    }
                    memo::Case::AllMiss => assert!(memo::hit_rate(stats) < 0.02),
                    memo::Case::History
                        if matches!(policy, gekijyou::computation::CachePolicy::Lru { .. }) =>
                    {
                        assert_eq!(stats.computation_calls, 0);
                        assert_eq!(memo::hit_rate(stats), 1.0);
                    }
                    memo::Case::History => assert!(stats.computation_calls > 1),
                }
            }
            assert!(!case.name().is_empty());
        }
    }
}
