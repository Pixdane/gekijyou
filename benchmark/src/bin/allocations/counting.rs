//! Requested-layout accounting, scoped to this allocation-only binary.
use benchmark::runner::{Allocations, Observation, Observer};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
};
static TRACK: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static DEALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
struct Meter;
fn change(delta: i64) {
    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    if TRACK.load(Ordering::Relaxed) {
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}
fn allocation(size: usize) {
    change(size as i64);
    if TRACK.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}
// System's contracts are preserved. Counter operations neither allocate nor unwind.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            allocation(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            allocation(l.size());
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            change(n as i64 - l.size() as i64);
            if TRACK.load(Ordering::Relaxed) {
                REALLOCS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(n as u64, Ordering::Relaxed);
            }
        }
        q
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        change(-(l.size() as i64));
        if TRACK.load(Ordering::Relaxed) {
            DEALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        TRACK.store(false, Ordering::Relaxed);
    }
}
pub struct Counter;
impl Observer for Counter {
    fn name(&self) -> &'static str {
        "alloc"
    }
    fn observe(&mut self, operation: impl FnOnce() -> u64) -> Observation {
        assert!(
            !TRACK.load(Ordering::Relaxed),
            "Nested allocation scopes are unsupported"
        );
        let baseline = LIVE.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        for n in [&ALLOCS, &REALLOCS, &DEALLOCS, &BYTES] {
            n.store(0, Ordering::Relaxed);
        }
        TRACK.store(true, Ordering::Relaxed);
        let guard = Guard;
        std::hint::black_box(operation());
        drop(guard);
        Observation {
            elapsed: None,
            allocations: Some(Allocations {
                allocs: ALLOCS.load(Ordering::Relaxed),
                reallocs: REALLOCS.load(Ordering::Relaxed),
                deallocs: DEALLOCS.load(Ordering::Relaxed),
                requested_bytes: BYTES.load(Ordering::Relaxed),
                live_delta: LIVE.load(Ordering::Relaxed) - baseline,
                peak_extra: PEAK.load(Ordering::Relaxed) - baseline,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn preexisting_payload_and_unwinding_do_not_corrupt_later_scopes() {
        let existing = std::hint::black_box(Vec::<u8>::with_capacity(256));
        let mut retained = None;
        let observed = Counter
            .observe(|| {
                drop(existing);
                retained = Some(std::hint::black_box(Vec::<u8>::with_capacity(128)));
                0
            })
            .allocations
            .unwrap();
        assert_eq!(observed.allocs, 1);
        assert_eq!(observed.deallocs, 1);
        assert_eq!(observed.requested_bytes, 128);
        assert_eq!(observed.live_delta, -128);
        assert_eq!(observed.peak_extra, 0);
        drop(retained);
        assert!(
            catch_unwind(AssertUnwindSafe(
                || Counter.observe(|| panic!("injected scope panic"))
            ))
            .is_err()
        );
        assert!(!TRACK.load(Ordering::Relaxed));
        let observed = Counter.observe(|| 0).allocations.unwrap();
        assert_eq!(observed, Allocations::default());
    }
}
