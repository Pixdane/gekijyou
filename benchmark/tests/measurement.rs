use benchmark::{
    options::Options,
    runner::{Observation, Observer, measure_sample, run},
    scenario::{Case, Suite, workload},
    scene::Config,
};
use std::time::Duration;
#[derive(Default)]
struct Recorder {
    checksums: Vec<u64>,
}
impl Observer for Recorder {
    fn name(&self) -> &'static str {
        "test"
    }
    fn observe(&mut self, operation: impl FnOnce() -> u64) -> Observation {
        self.checksums.push(operation());
        Observation {
            elapsed: Some(Duration::from_micros(1)),
            allocations: None,
        }
    }
}
#[test]
fn direct_evaluate_and_end_to_end_consume_the_same_outputs() {
    let mut checksums = Vec::new();
    for &selected in Suite::Scene.configurations() {
        let mut workload = workload(
            Config {
                actors: 7,
                stages: 3,
                rounds: 5,
            },
            selected,
        );
        let data = Case::AllMiss.datasets(2, 32);
        Case::AllMiss.verify(&mut workload, &data);
        let mut recorder = Recorder::default();
        let result = measure_sample(
            &mut recorder,
            &mut workload,
            Case::AllMiss,
            selected,
            &data,
            32,
        );
        assert_eq!(result.batches, 32);
        assert_eq!(recorder.checksums.len(), 1);
        if selected.name == "direct" {
            assert_eq!(result.stats.computation_calls, 0);
        } else {
            assert!(result.stats.computation_calls > 0);
        }
        checksums.push(recorder.checksums[0]);
    }
    assert!(checksums.iter().all(|&sum| sum == checksums[0]));
}
#[test]
fn context_workloads_match_oracle_and_second_edit_scopes_exclude_first_evaluation() {
    for &case in Suite::Context.cases() {
        for &selected in Suite::Context.configurations() {
            let mut workload = workload(
                Config {
                    actors: 7,
                    stages: 3,
                    rounds: 5,
                },
                selected,
            );
            let data = case.datasets(2, 4);
            case.verify(&mut workload, &data);
            let mut recorder = Recorder::default();
            let result = measure_sample(&mut recorder, &mut workload, case, selected, &data, 4);
            if case == Case::Cold {
                assert_eq!(result.batches, 1);
                assert!(result.stats.computation_calls > 0);
            }
            if case.edits() {
                assert_eq!(recorder.checksums.len(), 8);
                assert_eq!(result.peak_scope, "batch");
            } else {
                assert_eq!(recorder.checksums.len(), 1);
            }
            if case == Case::EditSecond || case == Case::Repeat {
                assert_eq!(result.stats.computation_calls, 0);
            }
            if selected.context_capacity > 0 && (case == Case::EditSecond || case == Case::Repeat) {
                assert_eq!(result.stats.resolution_visits, 4 * 2 * 6);
                assert_eq!(result.stats.context_hits, 4 * 2 * 6);
            }
        }
    }
}
#[test]
fn continuous_data_never_wraps_within_a_sample() {
    let data = Case::Continuous.datasets(64, 80);
    let frames: Vec<_> = data.iter().flatten().map(|i| i.frame.0).collect();
    assert_eq!(frames.len(), 64 * 80);
    assert!(frames.windows(2).all(|p| p[1] == p[0] + 1));
}
#[test]
fn invalid_cli_selections_fail_instead_of_producing_empty_results() {
    for args in [
        vec!["--samples", "0"],
        vec!["--iterations", "0"],
        vec!["--suite", "memo", "--case", "repeat"],
        vec!["--suite", "scene", "--config", "last"],
        vec!["--contexts", "2"],
        vec!["--rounds", "7"],
        vec!["--unknown", "1"],
        vec!["--samples"],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
    assert!(Options::parse(["--help".to_owned()]).unwrap().is_none());
    assert!(Options::parse(["--iterations", "128"].into_iter().map(str::to_owned)).is_err());
    assert!(
        Options::parse(
            ["--case", "repeat", "--iterations", "128"]
                .into_iter()
                .map(str::to_owned)
        )
        .is_ok()
    );
}
#[test]
fn raw_rows_preserve_samples_and_configuration_rotation() {
    let options = Options::parse(
        [
            "--suite",
            "memo",
            "--case",
            "high_hit",
            "--rounds",
            "0",
            "--contexts",
            "1",
            "--samples",
            "3",
            "--iterations",
            "2",
        ]
        .into_iter()
        .map(str::to_owned),
    )
    .unwrap()
    .unwrap();
    let mut recorder = Recorder::default();
    let mut output = Vec::new();
    run(&options, &mut recorder, &mut output).unwrap();
    let text = String::from_utf8(output).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 1 + 3 * 5);
    let header: Vec<_> = lines[0].split('\t').collect();
    let column = |name| header.iter().position(|&s| s == name).unwrap();
    let configurations: Vec<_> = Suite::Memo
        .configurations()
        .iter()
        .map(|c| c.name)
        .collect();
    for (row, line) in lines[1..].iter().enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), header.len());
        let sample = row / 5;
        let order = row % 5;
        assert_eq!(fields[column("sample")], sample.to_string());
        assert_eq!(fields[column("order")], order.to_string());
        assert_eq!(
            fields[column("config")],
            configurations[(sample + order) % 5]
        );
        assert_eq!(fields[column("allocs_per_batch")], "NA");
    }
}
