mod counting;
use benchmark::{
    options::{HELP, Options},
    runner,
};
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(options) = Options::parse(std::env::args().skip(1))? else {
        println!("{HELP}");
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err("Use --release or --profile perf".into());
    }
    runner::run(
        &options,
        &mut counting::Counter,
        &mut std::io::stdout().lock(),
    )?;
    Ok(())
}
