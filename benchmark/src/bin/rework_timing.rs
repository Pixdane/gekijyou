//! Timing only; run in release. Raw rotated backend samples retain full oracle checks.
fn main() {
    if cfg!(debug_assertions) {
        eprintln!("Use --release");
        std::process::exit(2);
    }
    benchmark::rework::measure(
        &mut benchmark::timing::Timer,
        &mut std::io::stdout().lock(),
        9,
    )
    .unwrap();
}
