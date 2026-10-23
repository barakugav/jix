//! Scalar element types and associated arithmetic traits used by the operation layer.
//!
//! ## Operation traits
//!
//! Each element-wise [`Array`](crate::Array) operation is bounded by a *scalar-level* trait of
//! this module, implemented for each supported element type: for example [`crate::ops::Add`]
//! requires [`Add`], and [`crate::ops::Exp`] requires [`Exp`].
//! In addition to element-wise trait, this module also contains reduction traits such as [`Sum`]
//! and [`Mean`].
//! Each such trait has the scalar
//! function (`add`) and a bulk one on `N` elements (`add_bulk`), which the ops call, vectorized
//! with SIMD for the types with SIMD support (the default computes element by element).
//!
//! ## `f16` and `Complex<T>` coverage
//!
//! The [`struct@f16`] and [`Complex<T>`] types are available under the  **`half`** and
//! **`num-complex`** crate features, respectively.

mod bool;
pub use bool::bool8;

#[cfg(feature = "half")]
pub use half::f16;

#[cfg(feature = "num-complex")]
pub use num_complex::Complex;

pub(crate) mod simd;
pub(crate) mod traits_util;
pub use crate::ops::_traits::*;
