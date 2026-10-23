use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor, Not};

/// A one-byte boolean, the boolean element type of jix arrays: `0` is false, any other byte is true.
///
/// A Rust `bool` must be exactly `0` or `1`, so viewing bytes from a file or a NumPy buffer as `bool`
/// is undefined behavior for any other value. Every byte is a valid `Bool8`, so such bytes are safe
/// to read. Comparisons, hashing and formatting use the logical value, so all non-zero bytes are
/// equal, and the logical operators return the canonical [`TRUE`] or [`FALSE`].
///
/// ```
/// use jix::{Bool8, FALSE, TRUE};
///
/// assert_eq!(Bool8::from(true), TRUE);
/// assert!(bool::from(TRUE));
/// assert_eq!(!TRUE, FALSE);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct Bool8(u8);

/// The canonical false [`Bool8`], the byte `0`.
pub const FALSE: Bool8 = Bool8(0);
/// The canonical true [`Bool8`], the byte `1`.
pub const TRUE: Bool8 = Bool8(1);

impl From<bool> for Bool8 {
    #[inline(always)]
    fn from(b: bool) -> Self {
        Self(b as u8)
    }
}
impl From<Bool8> for bool {
    #[inline(always)]
    fn from(b: Bool8) -> Self {
        b.0 != 0
    }
}

// Compare, hash and format by the logical value, not the raw byte.
impl PartialEq for Bool8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        bool::from(*self) == bool::from(*other)
    }
}
impl Eq for Bool8 {}
impl PartialOrd for Bool8 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bool8 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        bool::from(*self).cmp(&bool::from(*other))
    }
}
impl Hash for Bool8 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        bool::from(*self).hash(state);
    }
}
impl fmt::Debug for Bool8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&bool::from(*self), f)
    }
}
impl fmt::Display for Bool8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&bool::from(*self), f)
    }
}

// Logical operators, always producing a canonical value.
impl Not for Bool8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        (!bool::from(self)).into()
    }
}
macro_rules! impl_bool8_op2 {
    ($Trait:ident::$f:ident, $op:tt) => {
        impl $Trait for Bool8 {
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

    use crate::{Array, Bool8, FALSE, TRUE};

    #[test]
    fn any_non_zero_byte_is_true() {
        // Bytes other than 0 and 1, as may be read from a file or a NumPy buffer.
        let bytes = Array::compact_ndarray(&array![0u8, 1, 2, 255]).unwrap();
        let a = unsafe { bytes.transmute_elements::<Bool8>() };
        assert_eq!(
            a.to_ndarray::<Ix1>().unwrap(),
            array![FALSE, TRUE, TRUE, TRUE]
        );

        let not = unsafe { (!a.view()).transmute_elements::<u8>() };
        assert_eq!(not.to_ndarray::<Ix1>().unwrap(), array![1, 0, 0, 0]);
        let trues = Array::compact_ndarray(&array![TRUE, TRUE, TRUE, TRUE]).unwrap();
        let eq = unsafe { a.view().equal(trues).transmute_elements::<u8>() };
        assert_eq!(eq.to_ndarray::<Ix1>().unwrap(), array![0, 1, 1, 1]);
        assert_eq!(
            a.view().cast::<u8>().to_ndarray::<Ix1>().unwrap(),
            array![0, 1, 1, 1]
        );
        assert_eq!(a.view().sum(0).to_ndarray::<Ix0>().unwrap()[[]], 3);
    }
}
