//! Shared CLI options for the scene timing examples.
use std::time::Duration;

#[derive(Clone, Copy)]
pub struct TimingOptions {
    pub samples: usize,
    pub target: Duration,
}

impl TimingOptions {
    pub fn from_env() -> Self {
        if cfg!(debug_assertions) {
            eprintln!("Run this benchmark with --release");
            std::process::exit(2);
        }
        let mut samples = 9;
        let mut sample_ms = 40;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("Missing value for {arg}"));
            match arg.as_str() {
                "--samples" => samples = value.parse().expect("Invalid --samples"),
                "--sample-ms" => sample_ms = value.parse().expect("Invalid --sample-ms"),
                _ => panic!("Unknown option {arg}; use --samples N or --sample-ms N"),
            }
        }
        assert!(samples >= 3, "--samples must be at least 3");
        assert!(sample_ms > 0, "--sample-ms must be positive");
        Self {
            samples,
            target: Duration::from_millis(sample_ms),
        }
    }
}
