use std::sync::atomic::{AtomicUsize, Ordering};

/// A process-wide unique identifier of an array's data, used as a cache key.
///
/// Views of the same data (e.g. a borrowed or re-typed storage) share the id; new data gets a new
/// one. Ids are never reused, so a cached entry can never be mistaken for another array's data.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ArrayId(usize);

static NEXT_ARRAY_ID: AtomicUsize = AtomicUsize::new(0);

impl ArrayId {
    pub(crate) fn new() -> Self {
        Self(NEXT_ARRAY_ID.fetch_add(1, Ordering::Relaxed))
    }
}
