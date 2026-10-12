//! Validated selections shared by the two measurement programs.
use crate::scenario::{Case, Configuration, Mode, Suite};

pub const HELP: &str = "Usage: cargo run --release -p benchmark --bin <timing|allocations> -- [OPTIONS]\n\
  --suite scene|memo|context|all  default: all\n\
  --case NAME                   select one case from the chosen suites\n\
  --config NAME                 select one configuration\n\
  --samples N                   positive sample count; default: 9\n\
  --iterations N                Batch count per sample; default: 32\n\
  --rounds 0|128|all             default: all\n\
  --contexts 1|64|all            default: all\n\
  --help                        print this help\n\
Raw TSV goes to stdout. Cold-first always measures one Batch per sample.\n\
Increase iterations for short operations; use at least 9 samples for comparisons.";

#[derive(Clone, Debug)]
pub struct Options {
    pub samples: usize,
    pub iterations: usize,
    pub suites: Vec<Suite>,
    pub rounds: Vec<usize>,
    pub contexts: Vec<usize>,
    pub case: Option<String>,
    pub configuration: Option<String>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            samples: 9,
            iterations: 32,
            suites: Suite::ALL.to_vec(),
            rounds: vec![0, 128],
            contexts: vec![1, 64],
            case: None,
            configuration: None,
        }
    }
}
impl Options {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, String> {
        let mut options = Self::default();
        let mut args = args.into_iter();
        while let Some(key) = args.next() {
            if key == "--help" {
                return Ok(None);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("Missing value for {key}"))?;
            match key.as_str() {
                "--samples" => options.samples = positive(&value, &key)?,
                "--iterations" => options.iterations = positive(&value, &key)?,
                "--suite" => {
                    options.suites = match value.as_str() {
                        "all" => Suite::ALL.to_vec(),
                        "scene" => vec![Suite::Scene],
                        "memo" => vec![Suite::Memo],
                        "context" => vec![Suite::Context],
                        _ => return Err(format!("Unknown suite: {value}")),
                    }
                }
                "--rounds" => options.rounds = selection(&value, "--rounds", 0, 128)?,
                "--contexts" => options.contexts = selection(&value, "--contexts", 1, 64)?,
                "--case" => options.case = Some(value),
                "--config" => options.configuration = Some(value),
                _ => return Err(format!("Unknown option: {key}")),
            }
        }
        options.validate()?;
        Ok(Some(options))
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.samples == 0 || self.iterations == 0 {
            return Err("Samples and iterations must be positive".into());
        }
        if self.suites.is_empty() || self.rounds.is_empty() || self.contexts.is_empty() {
            return Err("Selections must be nonempty".into());
        }
        if self.rounds.iter().any(|n| ![0, 128].contains(n))
            || self.contexts.iter().any(|n| ![1, 64].contains(n))
        {
            return Err("Use rounds 0/128 and Context count 1/64".into());
        }
        if !self.suites.iter().any(|s| {
            s.cases().iter().any(|&c| self.selected_case(c))
                && s.configurations()
                    .iter()
                    .any(|&c| self.selected_configuration(c))
        }) {
            return Err("No matching suite/case/configuration".into());
        }
        for &suite in &self.suites {
            if !suite.cases().iter().any(|&case| self.selected_case(case)) {
                continue;
            }
            if suite
                .configurations()
                .iter()
                .any(|&c| self.selected_configuration(c) && c.mode == Mode::Evaluate)
            {
                for &contexts in &self.contexts {
                    if self
                        .iterations
                        .checked_mul(contexts)
                        .is_none_or(|n| n > 4096)
                    {
                        return Err("Evaluate-only prebuild is limited to 4096 Contexts; select fewer iterations/Contexts or another configuration".into());
                    }
                }
            }
            for &contexts in &self.contexts {
                if self.iterations.checked_mul(contexts).is_none() {
                    return Err("Input sequence length overflows".into());
                }
            }
        }
        Ok(())
    }
    pub fn selected_case(&self, case: Case) -> bool {
        self.case.as_ref().is_none_or(|name| name == case.name())
    }
    pub fn selected_configuration(&self, configuration: Configuration) -> bool {
        self.configuration
            .as_ref()
            .is_none_or(|name| name == configuration.name)
    }
}
fn positive(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{name} must be a positive integer"))
}
fn selection(value: &str, name: &str, first: usize, second: usize) -> Result<Vec<usize>, String> {
    if value == "all" {
        return Ok(vec![first, second]);
    }
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be {first}, {second} or all"))?;
    if ![first, second].contains(&parsed) {
        return Err(format!("{name} must be {first}, {second} or all"));
    }
    Ok(vec![parsed])
}
