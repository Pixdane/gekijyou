// Keys are internally allocated identities, frame indices and Arc pointers.
// Fx hashing avoids the generic hasher cost on hot graph/cache lookup paths.
// Exact Eq still determines identity; this does not hash ValueSemantics content.
pub(crate) type Map<K, V> = rustc_hash::FxHashMap<K, V>;
pub(crate) type Set<T> = rustc_hash::FxHashSet<T>;
