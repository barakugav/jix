use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A process-wide unique identifier of an array's data, used as a cache key.
///
/// Views of the same data (e.g. a borrowed or re-typed storage) share the id; new data gets a new
/// one. Ids are never reused, so a cached entry can never be mistaken for another array's data.
///
/// `repr(transparent)` over a [`NonZeroUsize`] guarantees `Option<ArrayId>` is a plain `usize`,
/// with `None` as zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct ArrayId(NonZeroUsize);

static NEXT_ARRAY_ID: AtomicUsize = AtomicUsize::new(1);

impl ArrayId {
    pub(crate) fn new() -> Self {
        Self(NonZeroUsize::new(NEXT_ARRAY_ID.fetch_add(1, Ordering::Relaxed)).unwrap())
    }
}

/// A set of [`ArrayId`]s. Up to 4 ids are stored inline, more on the heap.
#[derive(Clone, Debug)]
pub(crate) enum ArrayIdSet {
    /// Up to 4 ids, in any slots, with `None` in the unused ones.
    Inline([Option<ArrayId>; 4]),
    /// More than 4 ids.
    Heap(HashSet<ArrayId>),
}
impl Default for ArrayIdSet {
    fn default() -> Self {
        Self::Inline([None; 4])
    }
}
impl ArrayIdSet {
    #[inline]
    pub(crate) fn contains(&self, id: ArrayId) -> bool {
        match self {
            Self::Inline(ids) => Self::inline_contains(ids, id),
            Self::Heap(ids) => ids.contains(&id),
        }
    }

    /// Compare all the slots without short-circuiting, so the comparisons are branchless (and a
    /// single vector compare on targets with 64-bit lane compares).
    #[inline(always)]
    fn inline_contains(ids: &[Option<ArrayId>; 4], id: ArrayId) -> bool {
        ids.iter()
            .fold(false, |found, &slot| found | (slot == Some(id)))
    }

    /// Adds `id`, returning whether it was not in the set yet.
    pub(crate) fn insert(&mut self, id: ArrayId) -> bool {
        match self {
            Self::Inline(ids) => {
                if Self::inline_contains(ids, id) {
                    return false;
                }
                match ids.iter_mut().find(|slot| slot.is_none()) {
                    Some(slot) => *slot = Some(id),
                    None => *self = Self::Heap(ids.iter().flatten().copied().chain([id]).collect()),
                }
                true
            }
            Self::Heap(ids) => ids.insert(id),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = ArrayId> + '_ {
        let (inline, heap) = match self {
            Self::Inline(ids) => (Some(ids.iter().flatten()), None),
            Self::Heap(ids) => (None, Some(ids.iter())),
        };
        inline
            .into_iter()
            .flatten()
            .chain(heap.into_iter().flatten())
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{ArrayId, ArrayIdSet};

    #[test]
    fn option_array_id_is_a_usize() {
        assert_eq!(size_of::<Option<ArrayId>>(), size_of::<usize>());
    }

    #[test]
    fn array_id_set_spills_to_the_heap() {
        let ids = (0..6).map(|_| ArrayId::new()).collect::<Vec<_>>();
        let mut set = ArrayIdSet::default();
        for (n, &id) in ids.iter().enumerate() {
            assert!(set.insert(id));
            assert!(!set.insert(id));
            assert_eq!(matches!(set, ArrayIdSet::Inline(_)), n < 4);
            assert!(ids[..=n].iter().all(|&id| set.contains(id)));
            assert!(ids[n + 1..].iter().all(|&id| !set.contains(id)));
            let all = set.iter().collect::<HashSet<_>>();
            assert_eq!(all, ids[..=n].iter().copied().collect());
        }
    }
}
