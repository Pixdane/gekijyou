//! Requested-layout allocations and live peak, including every Bevy worker thread.
//! Output retention is identical across backends; measurements do not represent RSS.
mod counting;
fn main() {
    if cfg!(debug_assertions) {
        eprintln!("Use --release");
        std::process::exit(2);
    }
    benchmark::backends::measure(&mut counting::Counter, &mut std::io::stdout().lock(), 3).unwrap();
}
