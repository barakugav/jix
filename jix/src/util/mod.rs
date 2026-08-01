mod aligned_vec;
pub(crate) use aligned_vec::AlignedBytes;

mod arr_sequence;
pub use arr_sequence::*;

pub(crate) mod arrayvec;
pub(crate) mod cpu_cache;

pub(crate) mod iter;

mod arr_ext;
pub(crate) use arr_ext::*;

mod nd_copy;
pub(crate) use nd_copy::*;

mod nd_iter_unordered;
pub(crate) use nd_iter_unordered::*;

mod bitmap;
pub(crate) use bitmap::*;

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
    + 'static
{
    const ZERO: Self;
    const ONE: Self;

    fn usize(self) -> usize;
    fn from_usize(n: usize) -> Self;

    #[allow(unused)]
    fn u64(self) -> u64;
    fn from_u64(n: u64) -> Self;

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
            fn u64(self) -> u64 {
                self as u64
            }
            #[inline(always)]
            fn from_u64(n: u64) -> Self {
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
/// Compute the largest byte offset of a region accessed by `shape` and `strides`.
#[inline]
pub(crate) fn strided_span_bytes(shape: &[usize], strides: &[usize], itemsize: usize) -> usize {
    let mut span = itemsize;
    for (&len, &stride) in shape.iter().zip(strides) {
        if len == 0 {
            return 0;
        }
        span += stride * (len - 1);
    }
    span
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
    assert!(ptr.is_aligned() && size_of::<U>() > 0 && len_bytes.is_multiple_of(size_of::<U>()));
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
    assert!(ptr.is_aligned() && size_of::<U>() > 0 && len_bytes.is_multiple_of(size_of::<U>()));
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
        let (sz_low, sz_high) = self.size_hint();
        assert!(sz_low <= size && sz_high.is_none_or(|h| size <= h));

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

/// Master compile-time toggle for the read-shape-hint feature on the *consumption* side.
///
/// When `true`, read tiles are steered by the propagated `read_shape_scale_order`: [`scale_read_shape`]
/// runs the priority strategy (fully cover the highest-priority broadcast/reduction dims first), and
/// every consumer - the read heuristic / subset scaling (`read_shape_scale_dims`), the compaction
/// read path, and reductions - consults that order.
///
/// When `false` (default) the order is ignored everywhere: [`scale_read_shape`] runs the balanced
/// strategy (cap all dims to a common shrinking bound -> near-square tiles) and every consumer scales
/// in fixed C-order (inner dim first), reproducing the pre-hint behavior. The priority strategy wins
/// only when the covered axis is contiguous; it regresses the common row-major / block-compressed
/// case (block-orthogonal reads), so it is parked until the heuristic is made stride/block-aware.
///
/// `element_cost` and `read_shape_scale_order` are still *propagated* through the ops regardless;
/// this flag only controls whether the scaling functions *consume* them.
pub(crate) const USE_NEW_READ_SCALING: bool = false;

/// Choose a read tile from a block-shape seed, steered by a per-dim coverage priority.
///
/// Which strategy runs is gated by [`USE_NEW_READ_SCALING`]. The priority strategy (below):
/// `read_shape` enters holding the block-shape seed (the minimum non-wasteful read shape) and
/// leaves holding the chosen tile. `scale_order` is the coverage priority, **highest first**: the
/// down-scan shrinks from the low-priority end and the up-scan grows from the high-priority end, so
/// the dim we would grow first is the last we would shrink. Concretely:
///
/// 1. clamp each scaled dim into `[1, max_shape]`,
/// 2. scale down to `<= max_nitems` by shrinking the lowest-priority dims first, each only as much
///    as needed so higher-priority dims stay fully covered,
/// 3. scale up to `>= min_nitems` by growing the highest-priority dims first, by an integer multiple
///    of the (block-aligned) seed toward each dim's extent,
/// 4. snap any scaled dim that reached `max_shape[d]` to the full `array_shape[d]`, so the read
///    boundary doesn't split the requested range along an unaligned start.
///
/// Only the dims listed in `scale_order` are touched; any dim absent from it is left exactly as
/// seeded. A caller can therefore scale a *subset* of the dims by passing a partial order and
/// seeding the remaining dims to their final value (typically 1, so they don't consume the budget).
pub(crate) fn scale_read_shape(
    read_shape: &mut [u64],
    max_shape: &[u64],
    array_shape: &[u64],
    target_nitems: (u64, u64),
    scale_order: impl Iterator<Item = usize>,
) {
    let ndim = max_shape.len();
    assert_eq!(array_shape.len(), ndim);
    assert_eq!(read_shape.len(), ndim);
    let (min_nitems, max_nitems) = target_nitems;

    // The dims to scale, in coverage priority (highest first); dims absent from it are left as seeded.
    let order = scale_order.collect::<DimArray<_>>();
    debug_assert!(order.len() <= ndim);

    // Clamp the seed of each scaled dim into [1, max_shape].
    for &dim in order.iter() {
        read_shape[dim] = read_shape[dim].clamp(1, max_shape[dim].max(1));
    }

    if USE_NEW_READ_SCALING {
        // PRIORITY strategy (parked): shrink lowest-priority dims first so high-priority
        // (broadcast/reduction) dims stay fully covered - anisotropic tiles. O(ndim): the running
        // volume is maintained across both scans instead of recomputing the product each step.
        let mut current_volume = read_shape.iter().product::<u64>();
        for &dim in order.iter().rev() {
            if current_volume <= max_nitems {
                break;
            }
            let others = current_volume / read_shape[dim];
            let new_len = (max_nitems / others.max(1)).clamp(1, read_shape[dim]);
            current_volume = others * new_len;
            read_shape[dim] = new_len;
        }
        for &dim in order.iter() {
            let dim_len = max_shape[dim].max(1);
            let mult_by_budget = min_nitems / current_volume.max(1);
            let mult_by_range = dim_len.div_ceil(read_shape[dim]);
            let multiplier = mult_by_budget.min(mult_by_range).max(1);
            let new_read_size = (read_shape[dim] * multiplier).min(dim_len);
            current_volume = current_volume / read_shape[dim] * new_read_size;
            read_shape[dim] = new_read_size;
        }
    } else {
        // BALANCED strategy (default): cap every scaled dim to a common bound that halves until the
        // volume fits `max_nitems` - order-independent, so it yields near-square tiles that stay
        // aligned with storage blocks / contiguous runs. Then grow in priority order toward
        // `min_nitems`.
        let mut max_dim_size = (1u64 << 30).min(max_nitems.next_power_of_two());
        loop {
            for &dim in order.iter() {
                read_shape[dim] = read_shape[dim]
                    .min(max_dim_size)
                    .min(max_shape[dim].max(1))
                    .max(1);
            }
            let read_size = read_shape.iter().product::<u64>();
            if read_size / 2 <= max_nitems || max_dim_size <= 1 {
                break;
            }
            max_dim_size = (max_dim_size / 2).max(1);
        }
        let mut current_volume = read_shape.iter().product::<u64>();
        for &dim in order.iter() {
            let dim_len = max_shape[dim].max(1);
            let mult_by_budget = min_nitems / current_volume.max(1);
            let mult_by_range = dim_len.div_ceil(read_shape[dim]);
            let multiplier = mult_by_budget.min(mult_by_range).max(1);
            let new_read_size = (read_shape[dim] * multiplier).min(dim_len);
            current_volume = current_volume / read_shape[dim] * new_read_size;
            read_shape[dim] = new_read_size;
        }
    }

    // Snap any scaled dim already covering its full requested range to `array_shape[d]` so the read
    // boundary doesn't accidentally split the range along an unaligned start.
    for &dim in order.iter() {
        if read_shape[dim] == max_shape[dim] {
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

pub(crate) trait PtrExt<T> {
    fn read_maybe_aligned<const ALIGNED: bool>(self) -> T;
}
impl<T> PtrExt<T> for *const T {
    #[inline(always)]
    fn read_maybe_aligned<const ALIGNED: bool>(self) -> T {
        if ALIGNED {
            unsafe { self.read() }
        } else {
            unsafe { self.read_unaligned() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        calc_block_end, default_strides_slice, scale_read_shape, AlternatingBuffers,
        USE_NEW_READ_SCALING,
    };
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
        assert_eq!(s(&[], 4), &[] as &[usize]);
        assert_eq!(s(&[5], 2), &[2]);
        assert_eq!(s(&[3, 4], 4), &[16, 4]);
        assert_eq!(s(&[2, 3, 4], 1), &[12, 4, 1]);
        assert_eq!(s(&[2, 3, 4], 8), &[96, 32, 8]);
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
        scale_read_shape(
            read_shape.as_mut_slice(),
            &total,
            &total,
            (16, 256),
            (0..1).rev(),
        );
        let v = read_shape[0];
        assert!(v >= 16, "expected scale-up to reach the min floor, got {v}");
        assert!(
            v <= 256,
            "expected scale-down to respect the max ceiling, got {v}"
        );

        // Large seed (full range) must be CAPPED by the `max` ceiling, not collapsed to `min`.
        // If scale-down mistakenly used `min` (16), this would shrink to ~16 instead of ~max.
        let mut read_shape = DimDyn::from_fn(1, |_| 1000);
        scale_read_shape(
            read_shape.as_mut_slice(),
            &total,
            &total,
            (16, 256),
            (0..1).rev(),
        );
        let v = read_shape[0];
        assert!(v <= 256, "expected scale-down to cap at max, got {v}");
        assert!(
            v >= 128,
            "expected the cap to stay near max, not collapse to min, got {v}"
        );
    }

    #[test]
    fn scale_read_shape_prioritizes_high_order_dim() {
        use crate::Dimension;
        // 2-D [100, 100], budget max=200 items. Seed the whole thing (over budget -> scale down).
        // Priority order = [dim1, dim0] (dim1 highest).
        let shape = [100u64, 100];
        let mut read_shape = DimDyn::from_fn(2, |_| 100);
        scale_read_shape(
            read_shape.as_mut_slice(),
            &shape,
            &shape,
            (1, 200),
            [1usize, 0].into_iter(),
        );
        if USE_NEW_READ_SCALING {
            // Priority: dim0 (low) is shrunk first, dim1 (high) stays fully covered.
            assert_eq!(read_shape[1], 100, "high-priority dim stays fully covered");
            assert!(
                read_shape[0] <= 2,
                "low-priority dim absorbs the shrink, got {}",
                read_shape[0]
            );
        } else {
            // Balanced: both dims capped to a common bound (near-square), volume within ~max.
            assert_eq!(
                read_shape[0], read_shape[1],
                "balanced strategy yields a near-square tile, got {read_shape:?}"
            );
            assert!(
                read_shape[0] * read_shape[1] <= 2 * 200,
                "volume stays within ~max budget, got {read_shape:?}"
            );
        }
    }

    #[test]
    fn scale_read_shape_scales_only_ordered_dims() {
        use crate::Dimension;
        // The order lists only dim 1, so dim 0 is left exactly as seeded and only dim 1 grows -
        // this is how reduction scales the reduced and non-reduced dim groups separately.
        let shape = [100u64, 100];
        let mut read_shape = DimDyn::from_fn(2, |_| 7);
        scale_read_shape(
            read_shape.as_mut_slice(),
            &shape,
            &shape,
            (200, 200),
            std::iter::once(1usize),
        );
        assert_eq!(read_shape[0], 7, "unlisted dim is left exactly as seeded");
        // dim 1 grows in seed-multiples toward the budget: 7 * floor(200 / 7^2) = 7 * 4 = 28.
        assert_eq!(read_shape[1], 28, "listed dim scales toward the budget");
    }
}

#[cfg(test)]
mod test_util;
#[cfg(test)]
pub(crate) use test_util::*;

#[doc(hidden)]
pub mod bench_util;
