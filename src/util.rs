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
    + TryInto<usize, Error: core::fmt::Debug>
    + TryFrom<usize, Error: core::fmt::Debug>
{
    const ZERO: Self;
    const ONE: Self;
}
macro_rules! impl_idx_for_primitive {
    ($t:ty) => {
        impl Idx for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
        }
    };
}
impl_idx_for_primitive!(usize);
impl_idx_for_primitive!(u32);
impl_idx_for_primitive!(u64);
