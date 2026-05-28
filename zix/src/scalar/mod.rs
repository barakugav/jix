//! Scalar element types and associated arithmetic traits used by the operation layer.
//!
//! ## `f16` and `Complex<T>` coverage
//!
//! The [`struct@f16`] and [`Complex<T>`] types are available
//! as minimal stubs by default, and can be upgraded to full-featured types with the **`half`** and
//! **`num-complex`** crate features, respectively.
//! See their documentation for details.
//!
//! ## Operation traits
//!
//! Most of the element-wise [`Array`](crate::Array) operation are bounded by a *scalar-level* trait
//! implemented for each supported element type, for example [`crate::ops::Add`] require [`core::ops::Add`].
//! Scalar traits come from three sources:
//!
//! - **[`core::ops`]** — standard Rust operator traits (`Neg`, `Add`, `Sub`, `Mul`, `Div`,
//!   `BitAnd`, `BitOr`, `BitXor`, `Not`, `Shl`, `Shr`) plus [`PartialEq`] and [`PartialOrd`]
//!   for comparisons.
//! - **[`num_traits`]** — extended numeric traits: [`num_traits::Float`] for transcendental and
//!   classification ops (`floor`, `exp`, `sin`, `is_nan`, ...), [`num_traits::Pow`] for
//!   exponentiation, and [`num_traits::PrimInt`] for integer bit-manipulation ops (`rotate_left`,
//!   `count_ones`, `swap_bytes`, ...).
//! - **This module** — zix-specific traits re-exported from [`crate::ops`] for cases not covered
//!   by the above: [`Abs`] (handles `Complex<T>`), [`Maximum`]/[`Minimum`] (NaN-propagating, unlike
//!   `f32::max`/`f32::min`), [`LogicalAnd`]/[`LogicalOr`]/[`LogicalXor`]/[`LogicalNot`] (cast
//!   to `bool` before operating), [`Cast<D>`] for type conversion, and the `Reduce*` family
//!   ([`ReduceSum`], [`ReduceMax`], [`ReduceMean`], [`ArgMax`], ...) for reductions.
//!

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
