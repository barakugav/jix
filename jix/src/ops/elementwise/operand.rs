//! EXPERIMENT (api-simplify): scalar operands for binary ops.
//!
//! A binary op takes its right-hand side as `impl IntoOperand<T>`: either another array, or a
//! plain scalar of a `Dtyped` type, which becomes a zero-copy constant array (a broadcast
//! [`Scalar`](crate::storage::Scalar) storage) of the left-hand side's shape.

use crate::dtype::Dtyped;
use crate::error::Result;
use crate::storage::{ArrayStorageTyped, Scalar};
use crate::{Array, ArrayParams};

/// The other operand of a binary op: an [`Array`] with element type `T`, or a scalar `T`.
///
/// A scalar is broadcast to the shape of the array it is combined with, so `a.maximum(0.0)`,
/// `a.greater(0.5)` and `a.clamp(0.0, 1.0)` work without building constant arrays by hand.
pub trait IntoOperand<T> {
    /// The storage of the operand once it is an array.
    type Storage: ArrayStorageTyped<Item = T>;

    /// Convert into an array; a scalar takes the shape `shape` of the other operand.
    fn into_operand(self, shape: &[u64]) -> Result<Array<Self::Storage>>;
}

impl<T: Dtyped> IntoOperand<T> for T {
    type Storage = Scalar<T>;

    #[inline]
    fn into_operand(self, shape: &[u64]) -> Result<Array<Scalar<T>>> {
        Scalar::new(self, shape, ArrayParams::default()).map(Array::from_storage)
    }
}

impl<S: ArrayStorageTyped> IntoOperand<S::Item> for Array<S> {
    type Storage = S;

    #[inline(always)]
    fn into_operand(self, _shape: &[u64]) -> Result<Array<S>> {
        Ok(self)
    }
}

impl<T: Dtyped> Array<Scalar<T>> {
    /// EXPERIMENT: a constant array of `shape`, every element `value`. Zero-copy: nothing is
    /// allocated per element.
    pub fn full(value: T, shape: &[u64]) -> Result<Self> {
        value.into_operand(shape)
    }
}
