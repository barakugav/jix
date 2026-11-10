use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::{self, MaybeUninit};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// ID of an array used to uniquely identify its data in the read cache map
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct ArrayCacheId(NonZeroU64);

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

impl ArrayCacheId {
    #[inline(never)]
    pub(crate) fn new() -> Self {
        Self(NonZeroU64::new(NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)).unwrap())
    }
}

const INLINE_LEN: usize = 2;

pub(crate) enum ArrayCacheIdMap<V> {
    Inline {
        keys: [Option<ArrayCacheId>; INLINE_LEN],
        /// `values[i]` is initialized exactly when `keys[i]` is `Some`.
        values: [MaybeUninit<V>; INLINE_LEN],
    },
    Heap(HashMap<ArrayCacheId, V, BuildHasherDefault<FxHasher>>),
}
impl<V> Default for ArrayCacheIdMap<V> {
    fn default() -> Self {
        Self::Inline {
            keys: [None; INLINE_LEN],
            values: const { unsafe { MaybeUninit::uninit().assume_init() } },
        }
    }
}
impl<V> ArrayCacheIdMap<V> {
    #[inline]
    pub(crate) fn contains_key(&self, id: ArrayCacheId) -> bool {
        match self {
            Self::Inline { keys, .. } => keys
                .iter()
                .fold(false, |found, &key| found | (key == Some(id))), // branchless
            Self::Heap(map) => map.contains_key(&id),
        }
    }

    #[inline]
    pub(crate) fn get(&self, id: ArrayCacheId) -> Option<&V> {
        match self {
            Self::Inline { keys, values } => {
                let i = keys.iter().position(|&key| key == Some(id))?;
                // SAFETY: `keys[i]` is `Some`, so `values[i]` is initialized.
                Some(unsafe { values[i].assume_init_ref() })
            }
            Self::Heap(map) => map.get(&id),
        }
    }

    pub(crate) fn insert(&mut self, id: ArrayCacheId, value: V) -> Option<V> {
        match self {
            Self::Inline { keys, values } => {
                if let Some(i) = keys.iter().position(|&key| key == Some(id)) {
                    // SAFETY: `keys[i]` is `Some`, so `values[i]` is initialized.
                    return Some(mem::replace(unsafe { values[i].assume_init_mut() }, value));
                }
                if let Some(i) = keys.iter().position(Option::is_none) {
                    values[i].write(value);
                    keys[i] = Some(id);
                    return None;
                }
                let mut map = HashMap::with_capacity_and_hasher(2 * INLINE_LEN, Default::default());
                // The keys are cleared first, so a panic below leaks the values rather than
                // dropping them twice.
                let keys = mem::replace(keys, [None; INLINE_LEN]);
                for (key, value) in keys.into_iter().zip(values) {
                    // SAFETY: the inline map is full, so every key was `Some` and every value is
                    // initialized. With the keys cleared, the values are not read again.
                    map.insert(key.unwrap(), unsafe { value.assume_init_read() });
                }
                map.insert(id, value);
                *self = Self::Heap(map);
                None
            }
            Self::Heap(map) => map.insert(id, value),
        }
    }

    pub(crate) fn remove(&mut self, id: ArrayCacheId) -> Option<V> {
        match self {
            Self::Inline { keys, values } => {
                let i = keys.iter().position(|&key| key == Some(id))?;
                keys[i] = None;
                // SAFETY: `keys[i]` was `Some`, so `values[i]` is initialized. With the key cleared,
                // the value is not read again.
                Some(unsafe { values[i].assume_init_read() })
            }
            Self::Heap(map) => map.remove(&id),
        }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Inline { keys, .. } => keys.iter().all(Option::is_none),
            Self::Heap(map) => map.is_empty(),
        }
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = ArrayCacheId> + '_ {
        self.iter().map(|(key, _)| key)
    }

    fn iter(&self) -> impl Iterator<Item = (ArrayCacheId, &V)> {
        let (inline, heap) = match self {
            Self::Inline { keys, values } => {
                let entries = keys.iter().zip(values).filter_map(|(key, value)| {
                    // SAFETY: `key` is `Some`, so `value` is initialized.
                    key.map(|key| (key, unsafe { value.assume_init_ref() }))
                });
                (Some(entries), None)
            }
            Self::Heap(map) => (None, Some(map.iter().map(|(&key, value)| (key, value)))),
        };
        inline
            .into_iter()
            .flatten()
            .chain(heap.into_iter().flatten())
    }

    /// Removes all the entries, passing their values to `f`
    pub(crate) fn clear_with(&mut self, mut f: impl FnMut(V)) {
        match self {
            Self::Inline { keys, values } => {
                // The keys are cleared first, so a panic in `f` leaks the values left rather than
                // dropping them twice.
                for (key, value) in mem::replace(keys, [None; INLINE_LEN]).iter().zip(values) {
                    if key.is_some() {
                        // SAFETY: `key` was `Some`, so `value` is initialized. With the keys
                        // cleared, the value is not read again.
                        f(unsafe { value.assume_init_read() });
                    }
                }
            }
            Self::Heap(map) => map.drain().map(|(_, value)| value).for_each(f),
        }
    }
}
impl<V> Drop for ArrayCacheIdMap<V> {
    fn drop(&mut self) {
        if let Self::Inline { keys, values } = self {
            for (key, value) in keys.iter().zip(values) {
                if key.is_some() {
                    // SAFETY: `key` is `Some`, so `value` is initialized, and the map is not used again.
                    unsafe { value.assume_init_drop() };
                }
            }
        }
    }
}
impl<V: Clone> Clone for ArrayCacheIdMap<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Inline { keys, values } => Self::Inline {
                keys: *keys,
                values: std::array::from_fn(|i| match keys[i] {
                    // SAFETY: `keys[i]` is `Some`, so `values[i]` is initialized.
                    Some(_) => MaybeUninit::new(unsafe { values[i].assume_init_ref() }.clone()),
                    None => MaybeUninit::uninit(),
                }),
            },
            Self::Heap(map) => Self::Heap(map.clone()),
        }
    }
}
impl<V: fmt::Debug> fmt::Debug for ArrayCacheIdMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ArrayCacheIdSet(ArrayCacheIdMap<()>);
impl ArrayCacheIdSet {
    #[inline]
    pub(crate) fn contains(&self, id: ArrayCacheId) -> bool {
        self.0.contains_key(id)
    }

    pub(crate) fn insert(&mut self, id: ArrayCacheId) -> bool {
        self.0.insert(id, ()).is_none()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = ArrayCacheId> + '_ {
        self.0.keys()
    }
}

#[derive(Default)]
pub(crate) struct FxHasher(u64);
impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    #[inline]
    fn write_usize(&mut self, word: usize) {
        self.write_u64(word as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::rc::Rc;

    use super::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet, INLINE_LEN};

    #[test]
    fn option_array_id_is_a_u64() {
        assert_eq!(size_of::<Option<ArrayCacheId>>(), size_of::<u64>());
    }

    #[test]
    fn array_id_map_spills_to_the_heap() {
        let ids = (0..6).map(|_| ArrayCacheId::new()).collect::<Vec<_>>();
        let mut map = ArrayCacheIdMap::default();
        assert!(map.is_empty());
        for (n, &id) in ids.iter().enumerate() {
            assert_eq!(map.insert(id, n), None);
            assert!(!map.is_empty());
            assert_eq!(map.insert(id, n * 10), Some(n));
            assert_eq!(
                matches!(map, ArrayCacheIdMap::Inline { .. }),
                n < INLINE_LEN
            );
            for (m, &id) in ids.iter().enumerate() {
                assert_eq!(map.contains_key(id), m <= n);
                assert_eq!(map.get(id), (m <= n).then_some(&(m * 10)));
            }
            let keys = map.keys().collect::<HashSet<_>>();
            assert_eq!(keys, ids[..=n].iter().copied().collect());
        }
        assert_eq!(map.remove(ids[2]), Some(20));
        assert_eq!(map.remove(ids[2]), None);
        let mut values = Vec::new();
        map.clear_with(|value| values.push(value));
        values.sort();
        assert_eq!(values, [0, 10, 30, 40, 50]);
        assert_eq!(map.keys().count(), 0);
        assert!(map.is_empty());

        // Removing from the inline map frees a slot for the next insert.
        let mut map = ArrayCacheIdMap::default();
        for &id in &ids[..INLINE_LEN] {
            map.insert(id, ());
        }
        map.remove(ids[0]);
        map.insert(ids[4], ());
        assert!(matches!(map, ArrayCacheIdMap::Inline { .. }));
        assert!(map.contains_key(ids[4]) && !map.contains_key(ids[0]));
        map.remove(ids[1]);
        assert!(!map.is_empty());
        map.remove(ids[4]);
        assert!(map.is_empty());
    }

    #[test]
    fn array_id_map_drops_each_value_once() {
        let value = Rc::new(());
        let count = || Rc::strong_count(&value) - 1;
        let ids = (0..4).map(|_| ArrayCacheId::new()).collect::<Vec<_>>();
        // A map holding a clone of `value` under each of the first `n` ids, with the one under
        // `ids[0]` removed if `hole`: empty, partly or fully inline, and on the heap.
        let make = |n: usize, hole: bool| {
            let mut map = ArrayCacheIdMap::default();
            for &id in &ids[..n] {
                assert!(map.insert(id, value.clone()).is_none());
            }
            if hole {
                assert!(map.remove(ids[0]).is_some());
            }
            map
        };
        for n in 0..=ids.len() {
            for hole in [false, true] {
                if hole && n == 0 {
                    continue;
                }
                let len = n - hole as usize;
                assert_eq!(
                    n > INLINE_LEN,
                    matches!(make(n, hole), ArrayCacheIdMap::Heap(_))
                );

                let mut map = make(n, hole);
                assert_eq!(count(), len);
                if len > 0 {
                    // Replacing a value returns the old one.
                    assert!(map.insert(ids[n - 1], value.clone()).is_some());
                    assert_eq!(count(), len);
                }
                let clone = map.clone();
                assert_eq!(count(), 2 * len);
                drop(clone);
                assert_eq!(count(), len);
                drop(map);
                assert_eq!(count(), 0);

                let mut map = make(n, hole);
                let mut cleared = 0;
                map.clear_with(|_| cleared += 1);
                assert_eq!((cleared, count()), (len, 0));
                // The cleared map is reused, as the read cache does.
                map.insert(ids[0], value.clone());
                assert_eq!(count(), 1);
                drop(map);
                assert_eq!(count(), 0);
            }
        }
    }

    #[test]
    fn array_id_set() {
        let ids = (0..6).map(|_| ArrayCacheId::new()).collect::<Vec<_>>();
        let mut set = ArrayCacheIdSet::default();
        for (n, &id) in ids.iter().enumerate() {
            assert!(set.insert(id));
            assert!(!set.insert(id));
            assert!(ids
                .iter()
                .enumerate()
                .all(|(m, &id)| set.contains(id) == (m <= n)));
            let all = set.iter().collect::<HashSet<_>>();
            assert_eq!(all, ids[..=n].iter().copied().collect());
        }
    }
}
