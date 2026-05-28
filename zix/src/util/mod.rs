mod aligned_vec;
pub(crate) use aligned_vec::AlignedBytes;

mod arr_sequence;
pub use arr_sequence::ArraySequence;

pub(crate) mod cache_size;
pub(crate) mod iter;

use iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use iter::NdIter;

pub(crate) use crate::dimension::{dim_arr, try_dim_arr, DimArray};

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
    + TryInto<usize, Error: core::fmt::Debug>
    + TryFrom<usize, Error: core::fmt::Debug>
    + core::fmt::Display
    + core::fmt::Debug
    + core::iter::Sum
{
    const ZERO: Self;
    const ONE: Self;

    fn div_ceil(self, rhs: Self) -> Self;
    fn checked_mul(self, rhs: Self) -> Option<Self>;

    fn ceil_to_multiple(self, m: Self) -> Self {
        assert!(m > Self::ZERO);
        self.div_ceil(m) * m
    }

    fn floor_to_multiple(self, m: Self) -> Self {
        assert!(m > Self::ZERO);
        (self / m) * m
    }
}
macro_rules! impl_idx_for_primitive {
    ($t:ty) => {
        impl Idx for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;

            fn div_ceil(self, rhs: Self) -> Self {
                self.div_ceil(rhs)
            }

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

pub(crate) fn default_strides<Ix: Idx>(shape: &[Ix], itemsize: Ix) -> DimArray<Ix> {
    let ndim = shape.len();
    let mut strides = dim_arr(ndim, |_| itemsize);
    if ndim > 0 {
        for dim in (0..ndim - 1).rev() {
            strides[dim] = strides[dim + 1] * shape[dim + 1];
        }
    }
    strides
}

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
pub(crate) unsafe fn cast_slice_mut<T, U>(slice: &mut [T]) -> &mut [U]
where
    T: Copy + Sized,
    U: Copy + Sized,
{
    let (ptr, len) = (slice.as_mut_ptr().cast::<U>(), slice.len());
    let len_bytes = len * size_of::<T>();
    assert!(ptr.is_aligned());
    assert!(size_of::<U>() > 0 && len_bytes.is_multiple_of(size_of::<U>()));
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<U>(), len_bytes / size_of::<U>()) }
}

pub(crate) trait IxIterExt: Iterator {
    fn try_product(self) -> Option<Self::Item>;
}
impl<Ix, Iter> IxIterExt for Iter
where
    Ix: Idx,
    Iter: Iterator<Item = Ix>,
{
    fn try_product(mut self) -> Option<Self::Item> {
        self.try_fold(Ix::ONE, |acc, x| acc.checked_mul(x))
    }
}

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
        tmp_buf1: &'a mut AlignedBytes,
        tmp_buf2: &'a mut AlignedBytes,
    },
    Alternating {
        main_buf: &'a mut AlignedBytes,
        secondary_buf: &'a mut AlignedBytes,
    },
}
impl<'a> AlternatingBuffers<'a> {
    /// Creates a new `AlternatingBuffers` in the [`Init`](Self::Init) state.
    ///
    /// `data` is the original source slice. `tmp_buf1` and `tmp_buf2` are the two
    /// scratch buffers that will be alternated between during transformation steps.
    pub(crate) fn with_const_src(
        data: &'a [u8],
        tmp_buf1: &'a mut AlignedBytes,
        tmp_buf2: &'a mut AlignedBytes,
    ) -> Self {
        Self::Init {
            original_data: data,
            tmp_buf1,
            tmp_buf2,
        }
    }

    pub(crate) fn new(main_buf: &'a mut AlignedBytes, secondary_buf: &'a mut AlignedBytes) -> Self {
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
            } => main_buf.as_slice(),
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
    pub(crate) fn edit(&mut self) -> (&[u8], &mut AlignedBytes) {
        match self {
            Self::Init {
                original_data,
                tmp_buf1,
                tmp_buf2,
            } => {
                let data = *original_data;
                let main_buf = *tmp_buf1 as *mut AlignedBytes;
                let secondary_buf = *tmp_buf2 as *mut AlignedBytes;
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
                (prev_main_buf.as_slice(), prev_secondary_buf)
            }
        }
    }
}

pub(crate) unsafe fn nd_copy<S1, S2, S3>(
    src: *const u8,
    dst: *mut u8,
    shape: &[S1],
    src_strides: &[S2],
    dst_strides: &[S3],
    itemsize: usize,
) where
    S1: Idx + 'static,
    S2: Idx + 'static,
    S3: Idx + 'static,
{
    let ndim = shape.len();
    assert_eq!(ndim, src_strides.len());
    assert_eq!(ndim, dst_strides.len());

    // copy more then itemsize if the last dim(s) is contiguous
    let n_continuous_dims = (0..ndim)
        .rev()
        .scan(itemsize, |expected_stride, dim| {
            let src_stride: usize = src_strides[dim].try_into().unwrap();
            let dst_stride: usize = dst_strides[dim].try_into().unwrap();
            let is_contiguous = src_stride == *expected_stride && dst_stride == *expected_stride;
            *expected_stride *= shape[dim].try_into().unwrap();
            Some(is_contiguous)
        })
        .take_while(|&is_contiguous| is_contiguous)
        .count();
    let itemsize = itemsize
        * shape[ndim - n_continuous_dims..]
            .iter()
            .map(|&d_len| {
                let d_len: usize = d_len.try_into().unwrap();
                d_len
            })
            .product::<usize>();
    let shape = &shape[..ndim - n_continuous_dims];
    let src_strides = &src_strides[..ndim - n_continuous_dims];
    let dst_strides = &dst_strides[..ndim - n_continuous_dims];

    let mut iter = NdIter::new(
        shape,
        (
            NdIterExtStridesPtr::new(src_strides, src),
            NdIterExtStridesPtrMut::new(dst_strides, dst),
        ),
    );
    while let Some((_, (src_ptr, dst_ptr))) = iter.next() {
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
        }
    }
}

pub(crate) struct SendSyncPtr<T>(*const T);
unsafe impl<T> Send for SendSyncPtr<T> {}
unsafe impl<T> Sync for SendSyncPtr<T> {}
impl<T> SendSyncPtr<T> {
    pub unsafe fn new(ptr: *const T) -> Self {
        Self(ptr)
    }

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

#[cfg(test)]
mod tests {
    use super::{default_strides, AlignedBytes, AlternatingBuffers};

    #[test]
    fn test_default_strides() {
        let s = |shape, itemsize| default_strides(shape, itemsize).to_vec();
        assert_eq!(s(&[], 4), &[] as &[usize]); // scalar
        assert_eq!(s(&[5], 2), &[2]); // 1-d
        assert_eq!(s(&[3, 4], 4), &[16, 4]); // 2-d, itemsize 4
        assert_eq!(s(&[2, 3, 4], 1), &[12, 4, 1]); // 3-d, itemsize 1
        assert_eq!(s(&[2, 3, 4], 8), &[96, 32, 8]); // 3-d, itemsize 8
    }

    // --- AlternatingBuffers ---

    fn empty_buf() -> AlignedBytes {
        AlignedBytes::new(1)
    }

    /// data() on a freshly created Init state returns the original slice.
    #[test]
    fn alternating_buffers_init_data() {
        let data = [1u8, 2, 3, 4];
        let mut buf1 = empty_buf();
        let mut buf2 = empty_buf();
        let ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);
        assert_eq!(ab.data(), &[1u8, 2, 3, 4]);
    }

    /// The first edit() call transitions from Init to Alternating, returning
    /// (original_data, tmp_buf1). After writing to dst, data() reflects the result.
    #[test]
    fn alternating_buffers_first_edit_transitions() {
        let data = [1u8, 2, 3, 4];
        let mut buf1 = empty_buf();
        let mut buf2 = empty_buf();
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        {
            let (src, dst) = ab.edit();
            assert_eq!(src, &[1u8, 2, 3, 4]);
            dst.extend_from_slice(&[10, 20, 30, 40]);
        }

        assert!(matches!(ab, AlternatingBuffers::Alternating { .. }));
        assert_eq!(ab.data(), &[10u8, 20, 30, 40]);
    }

    /// A second edit() call reads the output of the first and writes a new result.
    #[test]
    fn alternating_buffers_two_edits() {
        let data = [1u8, 2, 3];
        let mut buf1 = empty_buf();
        let mut buf2 = empty_buf();
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        {
            let (src, dst) = ab.edit();
            assert_eq!(src, &[1u8, 2, 3]);
            for &b in src {
                dst.push(b + 10);
            }
        }
        assert_eq!(ab.data(), &[11u8, 12, 13]);

        {
            let (src, dst) = ab.edit();
            assert_eq!(src, &[11u8, 12, 13]);
            for &b in src {
                dst.push(b + 10);
            }
        }
        assert_eq!(ab.data(), &[21u8, 22, 23]);
    }

    /// Three or more edits continue to alternate correctly.
    #[test]
    fn alternating_buffers_pipeline() {
        let data = [0u8];
        let mut buf1 = empty_buf();
        let mut buf2 = empty_buf();
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        for step in 1u8..=5 {
            let (src, dst) = ab.edit();
            let prev = src[0];
            dst.clear();
            dst.push(prev + step);
        }
        // 0 + 1 + 2 + 3 + 4 + 5 = 15
        assert_eq!(ab.data(), &[15u8]);
    }

    /// edit() src correctly mirrors data() from the previous step.
    #[test]
    fn alternating_buffers_src_matches_previous_data() {
        let data = [42u8];
        let mut buf1 = empty_buf();
        let mut buf2 = empty_buf();
        let mut ab = AlternatingBuffers::with_const_src(&data, &mut buf1, &mut buf2);

        for _ in 0..4 {
            let current = ab.data().to_vec();
            let (src, dst) = ab.edit();
            assert_eq!(src, current.as_slice());
            dst.clear();
            dst.push(src[0].wrapping_add(1));
        }
    }
}

#[cfg(test)]
mod test_util;
#[cfg(test)]
pub(crate) use test_util::*;
