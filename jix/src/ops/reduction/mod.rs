//! Reductions: the `ReductionOp` engine and the op families built on it.

mod common;

mod min_max;
pub use min_max::*;

mod arith;
pub use arith::*;

mod variance;
pub use variance::*;

mod bools;
pub use bools::*;

mod custom;
pub use custom::*;

/// Public scalar-level traits implemented by primitive element types and consumed by the
/// reduction op storage wrappers ([`Sum`], [`Product`], [`Mean`], [`Variance`], ...).
///
/// Each trait describes how to fold a stream of `Self` values into a result. Some are simple
/// (e.g. `Sum` only needs an `Output` accumulator); others (`Mean`, `Variance`)
/// expose a richer state machine (`type State`, `init`, `update`, `finalize`) because the
/// final result is not just the accumulator.
///
/// Max/Min do **not** appear here - those ops are bounded directly by the element-wise
/// [`crate::scalar::Maximum`] / [`crate::scalar::Minimum`].
pub(crate) mod traits {
    pub use super::arith::traits::*;
    pub use super::min_max::traits::*;
    pub use super::variance::traits::*;
}
