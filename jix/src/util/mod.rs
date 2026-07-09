mod aligned_vec;
pub(crate) use aligned_vec::AlignedBytes;

mod arr_sequence;
pub use arr_sequence::*;

pub(crate) mod arrayvec;
pub(crate) mod cpu_cache;

pub(crate) mod iter;

mod nd_copy;
pub(crate) use nd_copy::*;

use std::mem::MaybeUninit;

pub(crate) use crate::dimension::{dim_arr, try_dim_arr, DimArray};
use crate::{DimVec, Dimension};

pub(crate) trait Idx:
    Clone
    + Copy
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Rem<Output = Self>
    + core::fmt::Display
    + core::fmt::Debug
    + core::iter::Sum
{
    const ZERO: Self;
    const ONE: Self;

    fn usize(self) -> usize;
    fn from_usize(n: usize) -> Self;

    fn div_ceil(self, rhs: Self) -> Self;
    fn checked_mul(self, rhs: Self) -> Option<Self>;

    #[inline(always)]
    fn ceil_to_multiple(self, m: Self) -> Self {
        debug_assert!(m > Self::ZERO);
        self.div_ceil(m) * m
    }

    #[inline(always)]
    fn floor_to_multiple(self, m: Self) -> Self {
        debug_assert!(m > Self::ZERO);
        (self / m) * m
    }
}
macro_rules! impl_idx_for_primitive {
    ($t:ty) => {
        impl Idx for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;

            #[inline(always)]
            fn usize(self) -> usize {
                self as usize
            }
            #[inline(always)]
            fn from_usize(n: usize) -> Self {
                n as $t
            }

            #[inline(always)]
            fn div_ceil(self, rhs: Self) -> Self {
                self.div_ceil(rhs)
            }

            #[inline(always)]
            fn checked_mul(self, rhs: Self) -> Option<Self> {
                self.checked_mul(rhs)
            }
        }
    };
}
impl_idx_for_primitive!(usize);
impl_idx_for_primitive!(u16);
impl_idx_for_primitive!(u32);
impl_idx_for_primitive!(u64);

#[inline(always)]
pub(crate) fn default_strides<V, Ix: Idx>(shape: &V, itemsize: Ix) -> V
where
    V: DimVec<Ix>,
{
    let shape = shape.as_ref();
    default_strides_from_iter::<V::Dimension, Ix>(shape.len(), shape.iter().copied(), itemsize)
}
#[inline(always)]
pub(crate) fn default_strides_cast<V, IxIn: Idx, IxOut: Idx>(
    shape: &V,
    itemsize: IxOut,
) -> <V::Dimension as Dimension>::Vec<IxOut>
where
    V: DimVec<IxIn>,
{
    let shape = shape.as_ref();
    default_strides_from_iter::<V::Dimension, IxOut>(
        shape.len(),
        shape.iter().copied().map(|s| IxOut::from_usize(s.usize())),
        itemsize,
    )
}
#[inline(always)]
pub(crate) fn default_strides_from_iter<D: Dimension, Ix: Idx>(
    ndim: usize,
    shape: impl DoubleEndedIterator<Item = Ix>,
    itemsize: Ix,
) -> D::Vec<Ix> {
    let mut strides = D::vec(ndim, |_| itemsize);
    if ndim > 1 {
        for (i, s) in shape.rev().take(ndim - 1).enumerate() {
            let dim = ndim - i - 1;
            strides[dim - 1] = strides[dim] * s;
        }
    }
    strides
}
#[inline(always)]
pub(crate) fn default_logical_strides<V, Ix: Idx>(shape: &V) -> V
where
    V: DimVec<Ix>,
{
    default_strides(shape, Ix::ONE)
}
#[inline(always)]
pub(crate) fn default_strides_slice<Ix: Idx>(shape: &[Ix], itemsize: Ix) -> DimArray<Ix> {
    let ndim = shape.len();
    let mut strides = dim_arr(ndim, |_| itemsize);
    if ndim > 0 {
        for dim in (0..ndim - 1).rev() {
            strides[dim] = strides[dim + 1] * shape[dim + 1];
        }
    }
    strides
}
#[inline(always)]
pub(crate) fn default_logical_strides_slice<Ix: Idx>(shape: &[Ix]) -> DimArray<Ix> {
    default_strides_slice(shape, Ix::ONE)
}

#[inline(always)]
pub(crate) unsafe fn cast_slice<T, U>(slice: &[T]) -> &[U]
where
    T: Copy + Sized,
    U: Copy + Sized,
{
    let (ptr, len) = (slice.as_ptr().cast::<U>(), slice.len());
    let len_bytes = len * size_of::<T>();
    assert!(ptr.is_aligned());
    assert!(size_of::<U>() > 0 && len_bytes.is_multiple_of(size_of::<U>()));
    unsafe { std::slice::from_raw_parts(ptr.cast::<U>(), len_bytes / size_of::<U>()) }
}
#[inline(always)]
pub(crate) unsafe fn cast_slice_mut<T, U>(slice: &mut [T]) -> &mut [U]
where
    T: Sized,
    U: Sized,
{
    let (ptr, len) = (slice.as_mut_ptr().cast::<U>(), slice.len());
    let len_bytes = len * size_of::<T>();
    assert!(ptr.is_aligned());
    assert!(size_of::<U>() > 0 && len_bytes.is_multiple_of(size_of::<U>()));
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<U>(), len_bytes / size_of::<U>()) }
}

pub(crate) trait IterExt: Iterator {
    #[inline]
    fn try_product(mut self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Idx,
    {
        self.try_fold(Self::Item::ONE, |acc, x| acc.checked_mul(x))
    }

    #[inline]
    fn collect_dim_vec<D>(mut self, size: usize) -> D::Vec<Self::Item>
    where
        Self: Sized,
        D: Dimension,
    {
        let v = D::vec(size, |_| self.next().unwrap());
        assert!(self.next().is_none());
        v
    }
}
impl<Iter: Iterator> IterExt for Iter {}

// pub(crate) enum MaybeOwned<'a, T> {
//     Owned(T),
//     Borrowed(&'a T),
// }
// impl<'a, T> AsRef<T> for MaybeOwned<'a, T> {
//     fn as_ref(&self) -> &T {
//         match self {
//             MaybeOwned::Owned(t) => t,
//             MaybeOwned::Borrowed(t) => t,
//         }
//     }
// }

// pub(crate) enum CowMut<'a, T> {
//     Owned(T),
//     Borrowed(&'a mut T),
// }
// impl<'a, T> AsRef<T> for CowMut<'a, T> {
//     fn as_ref(&self) -> &T {
//         match self {
//             CowMut::Owned(t) => t,
//             CowMut::Borrowed(t) => t,
//         }
//     }
// }
// impl<'a, T> AsMut<T> for CowMut<'a, T> {
//     fn as_mut(&mut self) -> &mut T {
//         match self {
//             CowMut::Owned(t) => t,
//             CowMut::Borrowed(t) => t,
//         }
//     }
// }

/// A state machine for applying a pipeline of in-place byte transformations using two
/// pre-allocated scratch buffers, alternating between them at each step.
///
/// This avoids per-step heap allocations. The two variants are:
///
/// - [`Init`](Self::Init): no transformation applied yet; holds a reference to the
///   original source data alongside the two reusable scratch buffers.
/// - [`Alternating`](Self::Alternating): at least one transformation has been applied;
///   `main_buf` holds the most recently written output and `secondary_buf` is the next
///   write target.
///
/// # Usage
///
/// Call [`edit`](Self::edit) to obtain `(src, dst)` for each step and write the
/// transformed bytes into `dst`. After all steps, call [`data`](Self::data) to read
/// the final result.
pub(crate) enum AlternatingBuffers<'a> {
    Init {
        original_data: &'a [u8],
        tmp_buf1: &'a mut [u8],
        tmp_buf2: &'a mut [u8],
    },
    Alternating {
        main_buf: &'a mut [u8],
        secondary_buf: &'a mut [u8],
    },
}
impl<'a> AlternatingBuffers<'a> {
    /// Creates a new `AlternatingBuffers` in the [`Init`](Self::Init) state.
    ///
    /// `data` is the original source slice. `tmp_buf1` and `tmp_buf2` are the two
    /// scratch buffers that will be alternated between during transformation steps.
    #[inline]
    pub(crate) fn with_const_src(
        data: &'a [u8],
        tmp_buf1: &'a mut [u8],
        tmp_buf2: &'a mut [u8],
    ) -> Self {
        Self::Init {
            original_data: data,
            tmp_buf1,
            tmp_buf2,
        }
    }

    #[inline]
    pub(crate) fn new(main_buf: &'a mut [u8], secondary_buf: &'a mut [u8]) -> Self {
        Self::Alternating {
            main_buf,
            secondary_buf,
        }
    }

    /// Returns the current data.
    ///
    /// - In [`Init`](Self::Init): the original source slice (no transformation applied yet).
    /// - In [`Alternating`](Self::Alternating): the contents of `main_buf`, i.e. the output
    ///   of the most recently completed transformation step.
    #[allow(dead_code)]
    pub(crate) fn data(&self) -> &[u8] {
        match self {
            Self::Init {
                original_data,
                tmp_buf1: _,
                tmp_buf2: _,
            } => original_data,
            Self::Alternating {
                main_buf,
                secondary_buf: _,
            } => main_buf,
        }
    }

    pub(crate) fn into_data(self) -> &'a [u8] {
        match self {
            Self::Init {
                original_data,
                tmp_buf1: _,
                tmp_buf2: _,
            } => original_data,
            Self::Alternating {
                main_buf,
                secondary_buf: _,
            } => main_buf,
        }
    }

    /// Returns `(src, dst)` for the next transformation step, advancing internal state.
    ///
    /// - In [`Init`](Self::Init): transitions `self` to [`Alternating`](Self::Alternating)
    ///   with `tmp_buf1` as `main_buf`, then returns `(original_data, tmp_buf1)`. After
    ///   writing the transformed output into `dst`, [`data`](Self::data) will return its
    ///   contents with no further state change.
    /// - In [`Alternating`](Self::Alternating): swaps `main_buf` <-> `secondary_buf`, then
    ///   returns `(old_main_data, old_secondary_buf)`. The caller writes transformed output
    ///   into `dst` (which is the new `main_buf`), and [`data`](Self::data) will immediately
    ///   reflect the new contents.
    pub(crate) fn edit(&mut self) -> (&[u8], &mut [u8]) {
        match self {
            Self::Init {
                original_data,
                tmp_buf1,
                tmp_buf2,
            } => {
                let data = *original_data;
                let main_buf = *tmp_buf1 as *mut [u8];
                let secondary_buf = *tmp_buf2 as *mut [u8];
                *self = Self::Alternating {
                    main_buf: unsafe { &mut *main_buf },
                    secondary_buf: unsafe { &mut *secondary_buf },
                };
                let buf = match self {
                    Self::Alternating { main_buf, .. } => main_buf,
                    _ => unreachable!(),
                };
                (data, buf)
            }
            Self::Alternating {
                main_buf,
                secondary_buf,
            } => {
                std::mem::swap(main_buf, secondary_buf);
                let prev_main_buf = secondary_buf;
                let prev_secondary_buf = main_buf;
                (prev_main_buf, prev_secondary_buf)
            }
        }
    }
}

pub(crate) fn scale_read_shape(
    read_shape: &mut impl Dimension, // TODO: Vec<u64>
    total_read_shape: &[u64],
    array_shape: &[u64],
    target_nitems: (u64, u64),
    scale_order: impl Iterator<Item = usize>,
) {
    let ndim = total_read_shape.len();
    assert_eq!(array_shape.len(), ndim);
    let (min_nitems, max_nitems) = target_nitems;

    // Scale down
    {
        let mut max_dim_size = (1u64 << 30).min(max_nitems.next_power_of_two());
        loop {
            for dim in 0..ndim {
                read_shape[dim] = read_shape[dim]
                    .min(max_dim_size)
                    .min(total_read_shape[dim])
                    .max(1);
            }
            if let Some(read_size) = read_shape.as_slice().iter().copied().try_product()
                && (read_size / 2 <= max_nitems || max_dim_size <= 1)
            {
                break;
            }
            max_dim_size = (max_dim_size / 2).max(1);
        }
    };

    // Scale up
    let mut current_volume = read_shape.as_slice().iter().product::<u64>();
    for dim in scale_order {
        let dim_len = total_read_shape[dim];
        let mult_by_budget = min_nitems / current_volume.max(1);
        let mult_by_range = dim_len.div_ceil(read_shape[dim]);
        let multiplier = mult_by_budget.min(mult_by_range).max(1);
        let new_read_size = (read_shape[dim] * multiplier).min(dim_len);
        current_volume = current_volume / read_shape[dim] * new_read_size;
        read_shape[dim] = new_read_size;
    }

    // Snap any dim already covering its full requested range to `shape[d]` so
    // the read boundary doesn't accidentally split the range along an unaligned start.
    for dim in 0..ndim {
        if read_shape[dim] == total_read_shape[dim] {
            read_shape[dim] = array_shape[dim].max(1);
        }
    }
}

#[derive(Clone)]
pub(crate) struct SendSyncPtr<T>(*const T);
unsafe impl<T> Send for SendSyncPtr<T> {}
unsafe impl<T> Sync for SendSyncPtr<T> {}
impl<T> SendSyncPtr<T> {
    #[inline]
    pub unsafe fn new(ptr: *const T) -> Self {
        Self(ptr)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.0
    }
}

macro_rules! assert_unchecked_eq {
    ($a:expr, $b:expr) => {{
        debug_assert_eq!($a, $b);
        std::hint::assert_unchecked($a == $b);
    }};
}
pub(crate) use assert_unchecked_eq;

#[inline(always)]
pub(crate) unsafe fn value_as_bytes<T>(val: &T) -> &[u8]
where
    T: Sized + Send + Sync + Copy + 'static,
{
    let val: *const T = val;
    unsafe { std::slice::from_raw_parts(val.cast::<u8>(), size_of::<T>()) }
}

// pub(crate) unsafe fn value_from_bytes<T>(bytes: &[u8]) -> T
// where
//     T: Sized + Send + Sync + Copy + 'static,
// {
//     assert!(bytes.len() == size_of::<T>());
//     let ptr = bytes.as_ptr().cast::<T>();
//     unsafe { *ptr }
// }

#[inline]
pub(crate) unsafe fn value_from_io<T>(mut src: impl std::io::Read) -> std::io::Result<T>
where
    T: Sized + Send + Sync + Copy + 'static,
{
    let mut val = MaybeUninit::<T>::uninit();
    {
        let buf = unsafe {
            std::slice::from_raw_parts_mut(val.as_mut_ptr().cast::<u8>(), size_of::<T>())
        };
        src.read_exact(buf)?;
    }
    Ok(unsafe { val.assume_init() })
}

macro_rules! or_else {
    ($( { $($optional:tt)+ } )? or { $($else:tt)+ }) => {
        crate::util::or_else!(@impl_ $( { $($optional)+ } )? or { $($else)* })
    };
    (@impl_ { $($optional:tt)+ } or { $($else:tt)* }) => {
        $($optional)*
    };
    (@impl_ or { $($else:tt)* }) => {
        $($else)*
    };
}
// macro_rules! if_none {
//     ($( { $($optional:tt)+ } )? than { $($else:tt)+ }) => {
//         crate::util::if_none!(@impl_ $( { $($optional)+ } )? than { $($else)* });
//     };
//     (@impl_ { $($optional:tt)+ } than { $($else:tt)* }) => {
//     };
//     (@impl_ than { $($else:tt)* }) => {
//         $($else)*
//     };
// }
pub(crate) use or_else;

#[inline]
pub(crate) fn calc_block_end(begin: u64, end: u64, block_size: u64) -> u64 {
    debug_assert!(begin <= end);
    // div_ceil always works for all cases, except for empty ranges where begin is not aligned with block_size
    if begin == end {
        end / block_size
    } else {
        end.div_ceil(block_size)
    }
}

pub(crate) trait ArrayExt<T, const N: usize> {
    fn try_map_<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<[U; N], E>
    where
        Self: Sized;
}
impl<T, const N: usize> ArrayExt<T, N> for [T; N] {
    #[inline]
    fn try_map_<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<[U; N], E>
    where
        Self: Sized,
    {
        let res = self.map(f);
        if res.iter().all(|r| r.is_ok()) {
            Ok(res.map(|items| unsafe { items.unwrap_unchecked() }))
        } else {
            Err(res.into_iter().filter_map(|r| r.err()).next().unwrap())
        }
    }
}

pub(crate) trait SliceExt<T> {
    fn to_dim_vec<D>(&self) -> D::Vec<T>
    where
        D: Dimension,
        T: Clone;
}
impl<T> SliceExt<T> for [T] {
    #[inline]
    fn to_dim_vec<D>(&self) -> D::Vec<T>
    where
        D: Dimension,
        T: Clone,
    {
        D::vec(self.len(), |i| self[i].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{calc_block_end, default_strides_slice, scale_read_shape, AlternatingBuffers};
    use crate::DimDyn;

    #[test]
    fn calc_block_end_non_empty_ranges() {
        // Non-empty ranges behave like div_ceil of the end.
        assert_eq!(calc_block_end(0, 4, 4), 1); // exactly one block
        assert_eq!(calc_block_end(0, 3, 4), 1); // partial block
        assert_eq!(calc_block_end(1, 5, 4), 2); // spans two blocks
        assert_eq!(calc_block_end(0, 8, 4), 2); // two full blocks
    }

    #[test]
    fn calc_block_end_empty_aligned_ranges() {
        // An empty range starting on a block boundary touches zero blocks.
        assert_eq!(calc_block_end(0, 0, 4), 0); // 0 blocks, begin = 0 / 4 = 0
        assert_eq!(calc_block_end(4, 4, 4), 1); // 0 blocks, begin = 4 / 4 = 1
        assert_eq!(calc_block_end(8, 8, 4), 2); // 0 blocks, begin = 8 / 4 = 2
    }

    #[test]
    fn calc_block_end_empty_unaligned_ranges() {
        // Regression: an empty range whose start is NOT block-aligned must still
        // touch zero blocks. The buggy `end.div_ceil(block_size)` returned one
        // block too many here (e.g. ceil(3 / 4) = 1 while begin / 4 = 0).
        for block_size in 1..=8u64 {
            for begin in 0..=16u64 {
                // begin == end -> empty range -> zero blocks -> end == begin block.
                assert_eq!(
                    calc_block_end(begin, begin, block_size),
                    begin / block_size,
                    "empty range [{begin}, {begin}) with block_size {block_size}",
                );
            }
        }
    }

    #[test]
    fn test_default_strides() {
        let s = |shape, itemsize| default_strides_slice(shape, itemsize).to_vec();
        assert_eq!(s(&[], 4), &[] as &[usize]); // scalar
        assert_eq!(s(&[5], 2), &[2]); // 1-d
        assert_eq!(s(&[3, 4], 4), &[16, 4]); // 2-d, itemsize 4
        assert_eq!(s(&[2, 3, 4], 1), &[12, 4, 1]); // 3-d, itemsize 1
        assert_eq!(s(&[2, 3, 4], 8), &[96, 32, 8]); // 3-d, itemsize 8
    }

    // --- AlternatingBuffers ---

    /// data() on a freshly created Init state returns the original slice.
    #[test]
    fn alternating_buffers_init_data() {
        let data = [1u8, 2, 3, 4];
        let mut buf1 = [0u8; 4];
        let mut buf2 = [0u8; 4];
        let ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);
        assert_eq!(ab.data(), &[1u8, 2, 3, 4]);
    }

    /// The first edit() call transitions from Init to Alternating, returning
    /// (original_data, tmp_buf1). After writing to dst, data() reflects the result.
    #[test]
    fn alternating_buffers_first_edit_transitions() {
        let data = [1u8, 2, 3, 4];
        let mut buf1 = [0u8; 4];
        let mut buf2 = [0u8; 4];
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        {
            let (src, dst) = ab.edit();
            assert_eq!(src, &[1u8, 2, 3, 4]);
            dst.copy_from_slice(&[10, 20, 30, 40]);
        }

        assert!(matches!(ab, AlternatingBuffers::Alternating { .. }));
        assert_eq!(ab.data(), &[10u8, 20, 30, 40]);
    }

    /// A second edit() call reads the output of the first and writes a new result.
    #[test]
    fn alternating_buffers_two_edits() {
        let data = [1u8, 2, 3];
        let mut buf1 = [0u8; 3];
        let mut buf2 = [0u8; 3];
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        {
            let (src, dst) = ab.edit();
            assert_eq!(src, &[1u8, 2, 3]);
            for (d, &b) in dst.iter_mut().zip(src) {
                *d = b + 10;
            }
        }
        assert_eq!(ab.data(), &[11u8, 12, 13]);

        {
            let (src, dst) = ab.edit();
            assert_eq!(src, &[11u8, 12, 13]);
            for (d, &b) in dst.iter_mut().zip(src) {
                *d = b + 10;
            }
        }
        assert_eq!(ab.data(), &[21u8, 22, 23]);
    }

    /// Three or more edits continue to alternate correctly.
    #[test]
    fn alternating_buffers_pipeline() {
        let data = [0u8];
        let mut buf1 = [0u8; 1];
        let mut buf2 = [0u8; 1];
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        for step in 1u8..=5 {
            let (src, dst) = ab.edit();
            let prev = src[0];
            dst[0] = prev + step;
        }
        // 0 + 1 + 2 + 3 + 4 + 5 = 15
        assert_eq!(ab.data(), &[15u8]);
    }

    /// edit() src correctly mirrors data() from the previous step.
    #[test]
    fn alternating_buffers_src_matches_previous_data() {
        let data = [42u8];
        let mut buf1 = [0u8; 1];
        let mut buf2 = [0u8; 1];
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        for _ in 0..4 {
            let current = ab.data().to_vec();
            let (src, dst) = ab.edit();
            assert_eq!(src, current.as_slice());
            dst[0] = src[0].wrapping_add(1);
        }
    }

    #[test]
    fn scale_read_shape_uses_max_to_cap_and_min_to_grow() {
        use crate::Dimension;
        // 1-D: total range 1000 elems. Window items: min=16, max=256.
        // Seed read_shape from a small block (8); scale-up should grow toward min (16+),
        // and the volume must never exceed ~max (256).
        let total = [1000u64];
        let mut read_shape = DimDyn::from_fn(1, |_| 8);
        scale_read_shape(&mut read_shape, &total, &total, (16, 256), (0..1).rev());
        let v = read_shape[0];
        assert!(v >= 16, "expected scale-up to reach the min floor, got {v}");
        assert!(
            v <= 256,
            "expected scale-down to respect the max ceiling, got {v}"
        );

        // Large seed (full range) must be CAPPED by the `max` ceiling, not collapsed to `min`.
        // If scale-down mistakenly used `min` (16), this would shrink to ~16 instead of ~max.
        let mut read_shape = DimDyn::from_fn(1, |_| 1000);
        scale_read_shape(&mut read_shape, &total, &total, (16, 256), (0..1).rev());
        let v = read_shape[0];
        assert!(v <= 256, "expected scale-down to cap at max, got {v}");
        assert!(
            v >= 128,
            "expected the cap to stay near max, not collapse to min, got {v}"
        );
    }
}

#[cfg(test)]
mod test_util;
#[cfg(test)]
pub(crate) use test_util::*;
