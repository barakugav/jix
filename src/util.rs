use crate::NDIM_MAX;

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
