use std::ptr::{NonNull, null_mut};

use crate::NDIM_MAX;

mod aligned_vec;

pub(crate) type DimArray<T> = arrayvec::ArrayVec<T, NDIM_MAX>;

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
    let mut strides = full_dim_array(itemsize, ndim);
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
    let (ptr, len) = (slice.as_ptr(), slice.len());
    let len_bytes = len * size_of::<T>();
    assert!(ptr as usize % align_of::<U>() == 0);
    assert!(size_of::<U>() > 0 && len_bytes % size_of::<U>() == 0);
    unsafe { std::slice::from_raw_parts(ptr.cast::<U>(), len_bytes / size_of::<U>()) }
}
pub(crate) unsafe fn cast_slice_mut<T, U>(slice: &mut [T]) -> &mut [U]
where
    T: Copy + Sized,
    U: Copy + Sized,
{
    let (ptr, len) = (slice.as_mut_ptr(), slice.len());
    let len_bytes = len * size_of::<T>();
    assert!(ptr as usize % align_of::<U>() == 0);
    assert!(size_of::<U>() > 0 && len_bytes % size_of::<U>() == 0);
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<U>(), len_bytes / size_of::<U>()) }
}

pub(crate) fn full_dim_array<T: Clone>(value: T, len: usize) -> DimArray<T> {
    (0..len).map(|_| value.clone()).collect()
}

pub(crate) fn ceil_to_multiple<Ix: Idx>(x: Ix, m: Ix) -> Ix {
    assert!(m > Ix::ZERO);
    x.div_ceil(m) * m
}

pub(crate) trait IxIterExt: Iterator {
    fn try_product(self) -> Option<Self::Item>;
}
impl<Ix, Iter> IxIterExt for Iter
where
    Ix: Idx,
    Iter: Iterator<Item = Ix>,
{
    fn try_product(self) -> Option<Self::Item> {
        self.fold(Some(Ix::ONE), |acc, x| {
            acc.and_then(|acc| acc.checked_mul(x))
        })
    }
}

pub(crate) enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}
impl<'a, T> AsRef<T> for MaybeOwned<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            MaybeOwned::Owned(t) => t,
            MaybeOwned::Borrowed(t) => t,
        }
    }
}

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

// struct AlignedVecU8 {
//     ptr: NonNull<u8>,
//     capacity: usize,
//     len: usize,
//     alignment: usize,
// }
// impl AlignedVecU8 {
//     fn new(align: usize) -> Self {
//         Self {
//             ptr: unsafe { NonNull::new_unchecked(null_mut::<u8>().wrapping_add(align)) },
//             capacity: 0,
//             len: 0,
//             alignment: align,
//         }
//     }
//     fn reserve(&mut self, additional: usize) {
//         if additional > self.capacity().wrapping_sub(self.len) {
//             unsafe { self.buf.grow_amortized(self.len, additional) };
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::default_strides;

    #[test]
    fn test_default_strides() {
        let s = |shape, itemsize| default_strides(shape, itemsize).to_vec();
        assert_eq!(s(&[], 4), &[] as &[usize]); // scalar
        assert_eq!(s(&[5], 2), &[2]); // 1-d
        assert_eq!(s(&[3, 4], 4), &[16, 4]); // 2-d, itemsize 4
        assert_eq!(s(&[2, 3, 4], 1), &[12, 4, 1]); // 3-d, itemsize 1
        assert_eq!(s(&[2, 3, 4], 8), &[96, 32, 8]); // 3-d, itemsize 8
    }
}
