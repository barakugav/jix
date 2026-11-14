use std::hint::assert_unchecked;
use std::mem::transmute;
use std::num::NonZeroUsize;
use std::ops::Range;

use crate::error::{bail, check_ndim, Result};
use crate::util::{dim_arr, DimArray};
use crate::NDIM_MAX;

/// The region [`read_data`](crate::ArrayStorage::read_data) reads: a start and a size per
/// dimension, covering `start[d]..start[d] + size[d]`.
///
/// There are as many sizes as starts, at most [`NDIM_MAX`] of each, no size is zero (the region is
/// not empty), and no `start + size` overflows.
#[derive(Clone, Copy, Debug)]
pub struct Index<'a> {
    start: &'a [u64],
    size: &'a [NonZeroUsize],
}

impl<'a> Index<'a> {
    /// Create a new index from the given `start` and `size` slices.
    ///
    /// # Errors
    ///
    /// - [`TooManyDimensions`](crate::error::ErrorKind::TooManyDimensions) if there are more than
    ///   [`NDIM_MAX`] dims.
    /// - [`InvalidIndex`](crate::error::ErrorKind::InvalidIndex) if the lengths of `start` and `size`
    ///   differ, a size is zero, or a `start + size` overflows.
    #[inline(always)]
    pub fn new(start: &'a [u64], size: &'a [usize]) -> Result<Self> {
        Self::check(start, size)?;
        // SAFETY: just checked.
        Ok(unsafe { Self::new_unchecked(start, size) })
    }

    /// Create a new index from the given `start` and `size` slices without checking them.
    ///
    /// # Safety
    ///
    /// `start` and `size` must pass the checks of [`Index::new`], as an index derived from a valid
    /// one does.
    #[inline(always)]
    pub(crate) unsafe fn new_unchecked(start: &'a [u64], size: &'a [usize]) -> Self {
        debug_assert!(Self::check(start, size).is_ok());
        // SAFETY: the caller guarantees that no size is zero.
        let size = unsafe { transmute::<&[usize], &[NonZeroUsize]>(size) };
        Self { start, size }
    }

    #[inline(always)]
    fn check(start: &[u64], size: &[usize]) -> Result<()> {
        check_ndim(start.len())?;
        if start.len() != size.len()
            || start
                .iter()
                .zip(size)
                .any(|(&s, &n)| n == 0 || s.checked_add(n as u64).is_none())
        {
            return invalid_index_fail(start, size);
        }
        Ok(())
    }

    /// The start of the read range in each dimension.
    #[inline(always)]
    pub fn start(&self) -> &'a [u64] {
        self.assume_invariants();
        self.start
    }

    /// The size of the read range in each dimension.
    #[inline(always)]
    pub fn size(&self) -> &'a [usize] {
        self.assume_invariants();
        // SAFETY: `NonZeroUsize` has the layout of `usize`.
        unsafe { transmute::<&[NonZeroUsize], &[usize]>(self.size) }
    }

    #[inline(always)]
    pub(crate) fn ndim(&self) -> usize {
        self.size().len()
    }

    #[inline(always)]
    pub(crate) fn end(&self, dim: usize) -> u64 {
        self.start()[dim] + self.size()[dim] as u64
    }

    #[inline(always)]
    fn assume_invariants(&self) {
        unsafe {
            assert_unchecked(self.start.len() == self.size.len() && self.size.len() <= NDIM_MAX)
        };
    }
}

#[inline(always)]
pub(crate) fn split_ranges(ranges: &[Range<u64>]) -> (DimArray<u64>, DimArray<usize>) {
    let start = dim_arr(ranges.len(), |d| ranges[d].start);
    let size = dim_arr(ranges.len(), |d| (ranges[d].end - ranges[d].start) as usize);
    (start, size)
}

#[cold]
#[inline(never)]
fn invalid_index_fail<T>(start: &[u64], size: &[usize]) -> Result<T> {
    bail!(
        InvalidIndex,
        "Invalid index start {start:?} and size {size:?}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn invalid_indices_are_rejected() {
        assert!(Index::new(&[0, 2], &[2, 1]).is_ok());
        let kind = |start: &[u64], size: &[usize]| Index::new(start, size).unwrap_err().kind();
        assert_eq!(kind(&[0, 2], &[2]), ErrorKind::InvalidIndex);
        assert_eq!(kind(&[0, 2], &[2, 0]), ErrorKind::InvalidIndex);
        assert_eq!(kind(&[u64::MAX], &[1]), ErrorKind::InvalidIndex);
        assert_eq!(
            kind(&[0; NDIM_MAX + 1], &[1; NDIM_MAX + 1]),
            ErrorKind::TooManyDimensions
        );
    }
}
