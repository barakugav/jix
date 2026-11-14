use std::hint::assert_unchecked;
use std::ops::{Deref, Range};

use crate::error::{check_ndim, Result};
use crate::NDIM_MAX;

/// The region [`read_data`](crate::ArrayStorage::read_data) reads: one half-open range
/// (`start..end`) per dimension, at most [`NDIM_MAX`] of them.
///
/// Dereferences to the `&[Range<u64>]` slice it wraps.
#[derive(Clone, Copy, Debug)]
pub struct Index<'a>(&'a [Range<u64>]);

impl<'a> Index<'a> {
    /// Wraps `ranges`, failing with
    /// [`TooManyDimensions`](crate::error::ErrorKind::TooManyDimensions) if there are more than
    /// [`NDIM_MAX`].
    #[inline(always)]
    pub fn new(ranges: &'a [Range<u64>]) -> Result<Self> {
        Self::check(ranges)?;
        // SAFETY: just checked.
        Ok(unsafe { Self::new_unchecked(ranges) })
    }

    /// Wraps `ranges` without checking them.
    ///
    /// # Safety
    ///
    /// `ranges` must pass the checks of [`Index::new`], as an index derived from a valid one does.
    #[inline(always)]
    pub(crate) unsafe fn new_unchecked(ranges: &'a [Range<u64>]) -> Self {
        debug_assert!(Self::check(ranges).is_ok());
        Self(ranges)
    }

    #[inline(always)]
    fn check(ranges: &[Range<u64>]) -> Result<()> {
        check_ndim(ranges.len())
    }
}

impl Deref for Index<'_> {
    type Target = [Range<u64>];

    #[inline(always)]
    fn deref(&self) -> &[Range<u64>] {
        unsafe { assert_unchecked(self.0.len() <= NDIM_MAX) };
        self.0
    }
}
