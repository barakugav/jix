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
//! implemented for each supported element type, for example [`crate::ops::Add`] requires
//! [`Add`]. Scalar traits come from three sources:
//!
//! - **This module** - jix traits. The ones of the ops with SIMD support have a `*_bulk`
//!   function on `N` elements, vectorized for the types with SIMD support: [`Add`], [`Sub`],
//!   [`Mul`], [`Div`], [`Neg`], [`BitAnd`], [`BitOr`], [`BitXor`], [`Not`], [`Shr`] (as their
//!   [`core::ops`] namesakes), [`Floor`], [`Ceil`], [`Sqrt`], [`CountZeros`], [`Abs`] (handles
//!   `Complex<T>`), [`Maximum`]/[`Minimum`] (NaN-propagating, unlike `f32::max`/`f32::min`) and
//!   [`Cast<D>`] for type conversion. And without: [`Sign`] and the `Reduce*` family ([`Sum`],
//!   [`Mean`], ...) for reductions.
//! - **[`core::ops`]** - [`Shl`](core::ops::Shl), plus [`PartialEq`] and [`PartialOrd`] for
//!   comparisons.
//! - **[`num_traits`]** - extended numeric traits: [`num_traits::Float`] for transcendental and
//!   classification ops (`exp`, `sin`, `is_nan`, ...), [`num_traits::Pow`] for
//!   exponentiation, and [`num_traits::PrimInt`] for integer bit-manipulation ops (`rotate_left`,
//!   `count_ones`, `swap_bytes`, ...).

#[cfg(feature = "half")]
pub use half::f16;

#[cfg(feature = "num-complex")]
pub use num_complex::Complex;

pub(crate) mod simd;
pub(crate) mod traits_util;
pub use crate::ops::_traits::*;
