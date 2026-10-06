use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A process-wide unique identifier of an array's data, used as a cache key.
///
/// Views of the same data (e.g. a borrowed or re-typed storage) share the id; new data gets a new
/// one. Ids are never reused, so a cached entry can never be mistaken for another array's data.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ArrayId(NonZeroUsize);

static NEXT_ARRAY_ID: AtomicUsize = AtomicUsize::new(1);

impl ArrayId {
    pub(crate) fn new() -> Self {
        Self(NonZeroUsize::new(NEXT_ARRAY_ID.fetch_add(1, Ordering::Relaxed)).unwrap())
    }
}

/// A shared set of [`ArrayId`]s.
pub(crate) type ArrayIdSet = Arc<HashSet<ArrayId>>;
