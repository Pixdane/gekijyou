#[path = "../examples/support/scene_workload.rs"]
mod fixture;

use fixture::{Config, Workload, direct_checksum, inputs};

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
