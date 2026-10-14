use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
std::thread_local! {
    static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}
struct Meter;
fn record() {
    let _ = COUNT.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            record();
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            record();
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(p, layout, size) };
        if !next.is_null() {
            record();
        }
        next
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        COUNT.with(|count| count.set(None));
    }
}
pub fn count(f: impl FnOnce()) -> usize {
    COUNT.with(|count| {
        assert!(count.get().is_none());
        count.set(Some(0));
    });
    let guard = Guard;
    f();
    let n = COUNT.with(|count| count.get().unwrap());
    drop(guard);
    n
}
