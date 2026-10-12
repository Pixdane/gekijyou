use benchmark::{
    options::{HELP, Options},
    runner,
    timing::Timer,
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
    runner::run(&options, &mut Timer, &mut std::io::stdout().lock())?;
    Ok(())
}
