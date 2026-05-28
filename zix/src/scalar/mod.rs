cfg_if::cfg_if! { if #[cfg(feature = "half")] {
    pub use half::f16;
} else {
    /// A 16-bit floating point type implementing the IEEE 754-2008 standard `binary16` a.k.a "half"
    /// format.
    ///
    /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
    /// Enable the `half` feature to get a fully functional `f16` type.
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    #[allow(non_camel_case_types)]
    pub struct f16(u16);
    impl f16 {
        #[doc = concat!("Creates a new `f16` from its raw bit representation.")]
        pub const fn from_bits(bits: u16) -> Self {
            Self(bits)
        }
        #[doc = concat!("Get the raw bit representation of the `f16`.")]
        pub const fn to_bits(&self) -> u16 {
            self.0
        }
    }
} }

cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
    pub use num_complex::Complex;
} else {
    /// A complex number in Cartesian form.
    ///
    /// Doesn't provide any arithmetic operations, but expose the real and imaginary parts.
    /// Enable the `num-complex` feature to get a fully functional `Complex` type.
    ///
    /// `Complex<T>` is memory layout compatible with an array `[T; 2]`, which is compatible with
    /// libc, numpy, etc.
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct Complex<T> {
        /// Real portion of the complex number
        pub re: T,
        /// Imaginary portion of the complex number
        pub im: T,
    }
} }

pub(crate) mod traits_util;
pub use crate::ops::_traits::*;
