use super::{node::CachePolicy, value::SharedValue};
use std::{collections::VecDeque, sync::Arc};

struct MemoEntry {
    inputs: Box<[SharedValue]>,
    output: SharedValue,
}

/// Entries belong to one node definition; front is LRU, back is MRU.
pub(super) struct NodeCache {
    revision: u64,
    policy: CachePolicy,
    entries: VecDeque<MemoEntry>,
}

impl NodeCache {
    pub(super) fn new(revision: u64, policy: CachePolicy) -> Self {
        Self {
            revision,
            policy,
            entries: VecDeque::new(),
        }
    }

    /// Lazily discard only this node's entries after definition or policy edits.
    pub(super) fn synchronize(&mut self, revision: u64, policy: CachePolicy) {
        if self.revision != revision || self.policy != policy {
            self.entries.clear();
            self.revision = revision;
            self.policy = policy;
        }
    }

    /// The caller supplies each input source's frozen value semantics.
    /// A hit promotes the entry to most recently used before returning its output.
    /// Failed comparisons leave recency unchanged, including unwinding comparators.
    pub(super) fn lookup<E>(
        &mut self,
        inputs: &[SharedValue],
        mut equivalent: impl FnMut(usize, &SharedValue, &SharedValue) -> Result<bool, E>,
    ) -> Result<Option<SharedValue>, E> {
        // Search recent entries first, without losing older input combinations.
        for index in (0..self.entries.len()).rev() {
            let entry = &self.entries[index];
            if entry.inputs.len() != inputs.len() {
                continue;
            }
            let mut matches = true;
            for (position, (current, cached)) in inputs.iter().zip(&entry.inputs).enumerate() {
                if !equivalent(position, current, cached)? {
                    matches = false;
                    break;
                }
            }
            if matches {
                let entry = self.entries.remove(index).expect("Existing cache entry");
                let output = Arc::clone(&entry.output);
                self.entries.push_back(entry);
                return Ok(Some(output));
            }
        }
        Ok(None)
    }

    /// Called only after successful computation and output type validation.
    pub(super) fn insert(&mut self, inputs: &[SharedValue], output: SharedValue) {
        let capacity = self.policy.capacity();
        if capacity == 0 {
            return;
        }
        if self.entries.len() >= capacity {
            // Keep the victim outside the queue until both key and output are ready.
            // A panicking value destructor must not leave a partially updated key.
            let mut victim = self.entries.pop_front().expect("Existing cache entry");
            if victim.inputs.len() == inputs.len() {
                victim.inputs.clone_from_slice(inputs);
            } else {
                victim.inputs = inputs.into();
            }
            victim.output = output;
            self.entries.push_back(victim);
            return;
        }
        self.entries.push_back(MemoEntry {
            inputs: inputs.into(),
            output,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ValueSemantics;
    use std::convert::Infallible;

    fn value(n: u64) -> SharedValue {
        Arc::new(n)
    }

    fn lookup(cache: &mut NodeCache, n: u64) -> Option<SharedValue> {
        cache
            .lookup(&[value(n)], |_, a, b| {
                Ok::<_, Infallible>(ValueSemantics::by_eq::<u64>().equivalent(a, b))
            })
            .unwrap()
    }

    #[test]
    fn policies_bound_entries_and_hits_promote_lru() {
        for policy in [CachePolicy::Transient, CachePolicy::Lru { capacity: 0 }] {
            let mut cache = NodeCache::new(0, policy);
            cache.insert(&[value(1)], value(10));
            assert!(cache.entries.is_empty());
        }
        let mut last = NodeCache::new(0, CachePolicy::Last);
        last.insert(&[value(1)], value(10));
        last.insert(&[value(2)], value(20));
        assert_eq!(last.entries.len(), 1);
        assert!(lookup(&mut last, 1).is_none());
        assert!(lookup(&mut last, 2).is_some());

        let mut memo = NodeCache::new(0, CachePolicy::Lru { capacity: 2 });
        let output = value(10);
        memo.insert(&[value(1)], Arc::clone(&output));
        memo.insert(&[value(2)], value(20));
        assert!(Arc::ptr_eq(&lookup(&mut memo, 1).unwrap(), &output));
        memo.insert(&[value(3)], value(30));
        assert_eq!(memo.entries.len(), 2);
        assert!(lookup(&mut memo, 2).is_none());
        assert!(lookup(&mut memo, 1).is_some());
        assert!(lookup(&mut memo, 3).is_some());
    }

    #[test]
    fn matching_preserves_order_multiplicity_and_short_circuit() {
        let mut cache = NodeCache::new(0, CachePolicy::Last);
        cache.insert(&[value(1), value(2), value(1)], value(9));
        let mut comparisons = 0;
        let mut lookup = |inputs: &[SharedValue]| {
            cache.lookup(inputs, |_, a, b| {
                comparisons += 1;
                Ok::<_, Infallible>(ValueSemantics::by_eq::<u64>().equivalent(a, b))
            })
        };
        assert!(lookup(&[value(1), value(2)]).unwrap().is_none());
        assert!(lookup(&[value(2), value(1), value(1)]).unwrap().is_none());
        assert!(lookup(&[value(1), value(2), value(1)]).unwrap().is_some());
        assert_eq!(comparisons, 4);
    }

    #[test]
    fn revision_and_policy_sync_clear_only_on_change() {
        let mut cache = NodeCache::new(7, CachePolicy::Last);
        cache.insert(&[], value(10));
        cache.synchronize(7, CachePolicy::Last);
        assert_eq!(cache.entries.len(), 1);
        cache.synchronize(8, CachePolicy::Last);
        assert!(cache.entries.is_empty());
        cache.insert(&[], value(20));
        cache.synchronize(8, CachePolicy::Lru { capacity: 4 });
        assert!(cache.entries.is_empty());
        cache.insert(&[], value(30));
        cache.synchronize(8, CachePolicy::Transient);
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn failed_comparison_preserves_lru_order() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        for panic_comparator in [false, true] {
            let mut cache = NodeCache::new(0, CachePolicy::Lru { capacity: 2 });
            cache.insert(&[value(1), value(10)], value(10));
            cache.insert(&[value(1), value(20)], value(20));
            let attempt = catch_unwind(AssertUnwindSafe(|| {
                cache.lookup(&[value(1), value(10)], |position, a, b| {
                    if position == 1 {
                        assert!(!panic_comparator, "injected comparator panic");
                        return Err(());
                    }
                    Ok(ValueSemantics::by_eq::<u64>().equivalent(a, b))
                })
            }));
            if panic_comparator {
                assert!(attempt.is_err());
            } else {
                assert!(matches!(attempt, Ok(Err(()))));
            }
            cache.insert(&[value(1), value(30)], value(30));
            let mut lookup_pair = |n| {
                cache
                    .lookup(&[value(1), value(n)], |_, a, b| {
                        Ok::<_, Infallible>(ValueSemantics::by_eq::<u64>().equivalent(a, b))
                    })
                    .unwrap()
            };
            assert!(lookup_pair(10).is_none());
            assert!(lookup_pair(20).is_some());
            assert!(lookup_pair(30).is_some());
        }
    }
    // Test-only allocation accounting is thread-local: parallel unit tests cannot
    // contribute to this scope, and release/library builds do not contain it.
    mod allocations {
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
    }

    #[test]
    fn full_last_reuses_input_allocation_through_continuous_misses() {
        let mut cache = NodeCache::new(0, CachePolicy::Last);
        cache.insert(&[value(0), value(0)], value(0));
        let address = cache.entries[0].inputs.as_ptr();
        for n in 1..20 {
            let inputs = [value(n), value(n + 1)];
            let output = value(n * 10);
            assert!(
                cache
                    .lookup(&inputs, |_, a, b| Ok::<_, Infallible>(Arc::ptr_eq(a, b)))
                    .unwrap()
                    .is_none()
            );
            assert_eq!(allocations::count(|| cache.insert(&inputs, output)), 0);
            assert_eq!(cache.entries.len(), 1);
            assert_eq!(cache.entries[0].inputs.as_ptr(), address);
            assert!(
                cache
                    .lookup(&inputs, |_, a, b| Ok::<_, Infallible>(Arc::ptr_eq(a, b)))
                    .unwrap()
                    .is_some()
            );
        }
    }

    #[test]
    fn unfilled_lru_allocates_then_reuses_only_the_promoted_victim() {
        let mut cache = NodeCache::new(0, CachePolicy::Lru { capacity: 3 });
        let a = [value(1)];
        let b = [value(2)];
        let c = [value(3)];
        let d = [value(4)];
        cache.insert(&a, value(10));
        let output = value(20);
        assert_eq!(allocations::count(|| cache.insert(&b, output)), 1);
        cache.insert(&c, value(30));
        let victim_buffer = cache.entries[1].inputs.as_ptr();
        assert!(lookup(&mut cache, 1).is_some());
        let output = value(40);
        assert_eq!(allocations::count(|| cache.insert(&d, output)), 0);
        assert_eq!(cache.entries.back().unwrap().inputs.as_ptr(), victim_buffer);
        assert!(lookup(&mut cache, 2).is_none());
        for n in [1, 3, 4] {
            assert!(lookup(&mut cache, n).is_some());
        }
    }

    #[test]
    fn full_cache_falls_back_for_changed_arity_without_retaining_old_values() {
        let mut cache = NodeCache::new(0, CachePolicy::Last);
        for arity in [1, 4, 2, 0, 5, 1] {
            let inputs: Vec<_> = (0..arity).map(|n| value(n as u64)).collect();
            let weak: Vec<_> = cache
                .entries
                .iter()
                .flat_map(|e| e.inputs.iter().map(Arc::downgrade))
                .collect();
            cache.insert(&inputs, value(arity as u64));
            assert!(weak.iter().all(|w| w.upgrade().is_none()));
            assert_eq!(cache.entries[0].inputs.len(), arity);
            assert!(
                cache
                    .lookup(&inputs, |_, a, b| Ok::<_, Infallible>(Arc::ptr_eq(a, b)))
                    .unwrap()
                    .is_some()
            );
        }
    }

    struct DropBomb;
    impl Drop for DropBomb {
        fn drop(&mut self) {
            panic!("injected cached value drop panic");
        }
    }

    #[test]
    fn victim_input_drop_panic_never_publishes_partial_keys_and_cache_recovers() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        for new_arity in [1, 2] {
            let mut cache = NodeCache::new(0, CachePolicy::Lru { capacity: 2 });
            cache.insert(&[Arc::new(DropBomb)], value(0));
            cache.insert(&[value(9)], value(90));
            let next: Vec<_> = (0..new_arity).map(|n| value(n + 1)).collect();
            assert!(catch_unwind(AssertUnwindSafe(|| cache.insert(&next, value(10)))).is_err());
            assert_eq!(cache.entries.len(), 1);
            assert!(lookup(&mut cache, 9).is_some());
            assert!(
                cache
                    .lookup(&next, |_, a, b| Ok::<_, Infallible>(Arc::ptr_eq(a, b)))
                    .unwrap()
                    .is_none()
            );
            cache.insert(&next, value(20));
            assert!(
                cache
                    .lookup(&next, |_, a, b| Ok::<_, Infallible>(Arc::ptr_eq(a, b)))
                    .unwrap()
                    .is_some()
            );
        }
    }

    #[test]
    fn victim_output_drop_panic_never_publishes_new_keys_and_cache_recovers() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let mut cache = NodeCache::new(0, CachePolicy::Last);
        cache.insert(&[value(1)], Arc::new(DropBomb));
        let next = [value(2)];
        assert!(catch_unwind(AssertUnwindSafe(|| cache.insert(&next, value(20)))).is_err());
        assert!(cache.entries.is_empty());
        assert!(lookup(&mut cache, 2).is_none());
        cache.insert(&next, value(30));
        assert!(lookup(&mut cache, 2).is_some());
    }
}
