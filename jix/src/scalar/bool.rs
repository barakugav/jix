use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor, Not};

/// A one-byte boolean, the boolean element type of jix arrays. `0` is `false`, any other value is `true`.
///
/// An std rust `bool` must be exactly `0` or `1`, so we cant interpret bytes from a file or
/// other external sources as `bool` as it is undefined behavior for any other value.
/// For `bool8`, any byte value is valid. `0` is `false`, any other value is `true`.
///
/// ```
/// use jix::scalar::bool8;
///
/// assert_eq!(bool8::from(true), bool8::TRUE);
/// assert!(bool::from(bool8::TRUE));
/// assert_eq!(!bool8::TRUE, bool8::FALSE);
/// ```
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct bool8(pub u8);

impl bool8 {
    /// The `false` value, byte `0`.
    pub const FALSE: Self = Self(0);
    /// The canonical `true` value, byte `1`.
    pub const TRUE: Self = Self(1);
}

impl From<bool> for bool8 {
    #[inline(always)]
    fn from(b: bool) -> Self {
        Self(b as u8)
    }
}
impl From<bool8> for bool {
    #[inline(always)]
    fn from(b: bool8) -> Self {
        b.0 != 0
    }
}

impl PartialEq for bool8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        bool::from(*self) == bool::from(*other)
    }
}
impl Eq for bool8 {}
impl PartialOrd for bool8 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for bool8 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        bool::from(*self).cmp(&bool::from(*other))
    }
}
impl Hash for bool8 {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        bool::from(*self).hash(state);
    }
}
impl fmt::Debug for bool8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&bool::from(*self), f)
    }
}
impl fmt::Display for bool8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&bool::from(*self), f)
    }
}

impl Not for bool8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        (!bool::from(self)).into()
    }
}
macro_rules! impl_bool8_op2 {
    ($Trait:ident::$f:ident, $op:tt) => {
        impl $Trait for bool8 {
            type Output = Self;
            #[inline(always)]
            fn $f(self, rhs: Self) -> Self {
                (bool::from(self) $op bool::from(rhs)).into()
            }
        }
    };
}
impl_bool8_op2!(BitAnd::bitand, &);
impl_bool8_op2!(BitOr::bitor, |);
impl_bool8_op2!(BitXor::bitxor, ^);

#[cfg(test)]
mod tests {
    use ndarray::{array, Ix0, Ix1};

    use crate::scalar::bool8;
    use crate::Array;

    #[test]
    fn any_non_zero_byte_is_true() {
        // Bytes other than 0 and 1, as may be read from a file or a NumPy buffer.
        let bytes = Array::compact_ndarray(&array![0u8, 1, 2, 255]).unwrap();
        let a = unsafe { bytes.transmute_elements::<bool8>() };
        assert_eq!(
            a.to_ndarray::<Ix1>().unwrap(),
            array![bool8::FALSE, bool8::TRUE, bool8::TRUE, bool8::TRUE]
        );

        let not = unsafe { (!a.view()).transmute_elements::<u8>() };
        assert_eq!(not.to_ndarray::<Ix1>().unwrap(), array![1, 0, 0, 0]);
        let trues =
            Array::compact_ndarray(&array![bool8::TRUE, bool8::TRUE, bool8::TRUE, bool8::TRUE])
                .unwrap();
        let eq = unsafe { a.view().equal(trues).transmute_elements::<u8>() };
        assert_eq!(eq.to_ndarray::<Ix1>().unwrap(), array![0, 1, 1, 1]);
        assert_eq!(
            a.view().cast::<u8>().to_ndarray::<Ix1>().unwrap(),
            array![0, 1, 1, 1]
        );
        assert_eq!(a.view().sum(0).to_ndarray::<Ix0>().unwrap()[[]], 3);
    }
}
