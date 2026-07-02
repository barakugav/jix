//! Scalar element types and associated arithmetic traits used by the operation layer.
//!
//! ## `f16` and `Complex<T>` coverage
//!
//! The [`struct@f16`] and [`Complex<T>`] types are available under the  **`half`** and
//! **`num-complex`** crate features, respectively.
//!
//! ## Operation traits
//!
//! Most of the element-wise [`Array`](crate::Array) operation are bounded by a *scalar-level* trait
//! implemented for each supported element type, for example [`crate::ops::Add`] require [`core::ops::Add`].
//! Scalar traits come from three sources:
//!
//! - **[`core::ops`]** - standard Rust operator traits (`Neg`, `Add`, `Sub`, `Mul`, `Div`,
//!   `BitAnd`, `BitOr`, `BitXor`, `Not`, `Shl`, `Shr`) plus [`PartialEq`] and [`PartialOrd`]
//!   for comparisons.
//! - **[`num_traits`]** - extended numeric traits: [`num_traits::Float`] for transcendental and
//!   classification ops (`floor`, `exp`, `sin`, `is_nan`, ...), [`num_traits::Pow`] for
//!   exponentiation, and [`num_traits::PrimInt`] for integer bit-manipulation ops (`rotate_left`,
//!   `count_ones`, `swap_bytes`, ...).
//! - **This module** - jix-specific traits for cases not covered
//!   by the above: [`Abs`] (handles `Complex<T>`), [`Maximum`]/[`Minimum`] (NaN-propagating, unlike
//!   `f32::max`/`f32::min`), [`Cast<D>`] for type conversion, and the `Reduce*` family
//!   ([`Sum`], [`Mean`], ...) for reductions.

#[cfg(feature = "half")]
pub use half::f16;

#[cfg(feature = "num-complex")]
pub use num_complex::Complex;

pub(crate) mod traits_util;
pub use crate::ops::_traits::*;
