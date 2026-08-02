use std::hint::assert_unchecked;

use crate::NDIM_MAX;

/// A compact, fixed-length set of per-dimension boolean flags.
///
/// Bit `d` (the low bit is dimension 0) stores one boolean for dimension `d`; callers decide what a
/// set bit means. The bitmap tracks its own dimension count ([`len`](Self::len)), so it builds (via
/// [`FromIterator`]), iterates (via [`IntoIterator`]), and compares as a fixed-length sequence of
/// booleans. Since the maximum number of dimensions is [`NDIM_MAX`] (which is 8), all flags fit in a
/// single `u8`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DimBitmap {
    bits: u8,
    len: u8,
}

impl DimBitmap {
    /// The low `n` bits set (and nothing above), used to mask off unused high bits.
    #[inline]
    fn low_mask(n: usize) -> u8 {
        debug_assert!(n <= NDIM_MAX);
        if n >= NDIM_MAX {
            u8::MAX
        } else {
            (1u8 << n) - 1
        }
    }

    /// A bitmap of `len` dimensions, every flag set to `value`.
    #[inline]
    pub(crate) fn filled(len: usize, value: bool) -> Self {
        assert!(len <= NDIM_MAX);
        Self {
            bits: if value { Self::low_mask(len) } else { 0 },
            len: len as u8,
        }
    }

    /// The number of dimensions the bitmap covers.
    #[inline]
    pub(crate) fn len(self) -> usize {
        let len = self.len as usize;
        unsafe { assert_unchecked(len <= NDIM_MAX) };
        len
    }

    /// Returns the flag for dimension `dim`.
    #[inline]
    pub(crate) fn get(self, dim: usize) -> bool {
        assert!(dim < self.len());
        self.bits & (1u8 << dim) != 0
    }

    /// Sets the flag for dimension `dim` to `value`.
    #[inline]
    pub(crate) fn set(&mut self, dim: usize, value: bool) {
        assert!(dim < self.len());
        let bit = 1u8 << dim;
        if value {
            self.bits |= bit;
        } else {
            self.bits &= !bit;
        }
    }

    /// Returns whether every dimension's flag is set.
    #[inline]
    pub(crate) fn all(self) -> bool {
        let mask = Self::low_mask(self.len());
        self.bits & mask == mask
    }

    /// Inserts a new dimension at position `pos`, shifting higher dimensions up by one, and
    /// grows the length by one. The inserted dimension takes `value`.
    #[inline]
    pub(crate) fn insert(&mut self, pos: usize, value: bool) {
        assert!(pos <= self.len() && self.len() < NDIM_MAX);
        let low_mask = Self::low_mask(pos);
        self.bits = (self.bits & low_mask) | ((self.bits & !low_mask) << 1);
        self.len += 1;
        self.set(pos, value);
    }
}

impl FromIterator<bool> for DimBitmap {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut bitmap = Self { bits: 0, len: 0 };
        for value in iter {
            assert!(bitmap.len() < NDIM_MAX);
            let dim = bitmap.len();
            bitmap.len += 1;
            bitmap.set(dim, value);
        }
        bitmap
    }
}

/// Iterator over a [`DimBitmap`], yielding the flag of each dimension (dimension 0 first).
pub(crate) struct DimBitmapIter {
    bitmap: DimBitmap,
    pos: usize,
}
impl Iterator for DimBitmapIter {
    type Item = bool;
    #[inline]
    fn next(&mut self) -> Option<bool> {
        (self.pos < self.bitmap.len()).then(|| {
            let value = self.bitmap.get(self.pos);
            self.pos += 1;
            value
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bitmap.len() - self.pos;
        (remaining, Some(remaining))
    }
}
impl ExactSizeIterator for DimBitmapIter {}
impl IntoIterator for DimBitmap {
    type Item = bool;
    type IntoIter = DimBitmapIter;
    #[inline]
    fn into_iter(self) -> DimBitmapIter {
        DimBitmapIter {
            bitmap: self,
            pos: 0,
        }
    }
}
