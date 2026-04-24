mod broadcast;
pub use broadcast::*;

mod slice;
pub use slice::*;

mod insert_axes;
pub use insert_axes::*;

mod remove_axes;
pub use remove_axes::*;

mod permute_axes;
pub use permute_axes::*;

mod reshape;
pub use reshape::*;

mod concatenate;
pub use concatenate::*;

mod stack;
pub use stack::*;

use crate::storage::{ArrayStorage, Compact};
use crate::Array;

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Returns a copy of the array with a new shape. See [`Reshape`] for details and examples.
    ///
    /// Preferred over [`reshape_view`](Self::reshape_view) when the result will be read more than
    /// once: the copy realigns blocks to the new shape, avoiding read-amplification on future
    /// reads.
    #[track_caller]
    pub fn reshape(self, new_shape: &[u64]) -> Array<Compact> {
        self.reshape_view(new_shape).copy().unwrap()
    }

    /// Returns a lazy view of the array with a new shape. See [`Reshape`] for details and
    /// examples.
    ///
    /// No data is copied at construction time, but reads may be slow when the new shape crosses
    /// block boundaries of the original layout. Call [`.copy()`](Array::copy) to realign blocks
    /// before repeated reads, or prefer [`reshape`](Self::reshape) directly.
    #[track_caller]
    pub fn reshape_view(self, new_shape: &[u64]) -> Array<Reshape<S>> {
        Array::from_storage(Reshape::new(self, new_shape).unwrap())
    }

    /// Returns a lazy view of a sub-region of the array. See [`Slice`] for details and examples.
    ///
    /// Accepts a tuple of Rust ranges or [`SliceItem`]s, one per dimension. Negative integer
    /// range bounds are supported (Python-style end-relative indexing). No data is copied.
    ///
    /// # Panics
    ///
    /// Panics if the number of items != `self.ndim()` or any `step < 1`.
    #[track_caller]
    pub fn slice(self, slice: impl Into<SliceSpec>) -> Array<Slice<S>> {
        Array::from_storage(Slice::new(self, slice.into()).unwrap())
    }

    /// Returns a lazy view of the array with its axes reordered. See [`PermuteAxes`] for details
    /// and examples.
    ///
    /// `axes[i]` names the input axis that maps to output axis `i`. No data is copied.
    ///
    /// # Panics
    ///
    /// Panics if `axes` is not a valid permutation of `0..ndim`.
    #[track_caller]
    pub fn permute_axes(self, axes: &[usize]) -> Array<PermuteAxes<S>> {
        Array::from_storage(PermuteAxes::new(self, axes).unwrap())
    }

    /// Expands the array to `new_shape` by repeating length-1 dimensions and returns a
    /// materialized copy. See [`Broadcast`] for details and examples.
    ///
    /// Preferred over [`broadcast_view`](Self::broadcast_view) when the result will be read more
    /// than once: the copy stores each element once at its expanded position, avoiding repeated
    /// reads of the same source blocks on future accesses.
    ///
    /// # Panics
    ///
    /// See [`broadcast_view`](Self::broadcast_view) for validity rules.
    #[track_caller]
    pub fn broadcast(self, new_shape: &[u64]) -> Array<Compact> {
        self.broadcast_view(new_shape).copy().unwrap()
    }

    /// Returns a lazy view of the array expanded to `new_shape` by repeating length-1 dimensions.
    /// See [`Broadcast`] for details and examples.
    ///
    /// No data is copied, but reads may be slow — the same source blocks are decompressed
    /// repeatedly. Call [`.copy()`](Array::copy) to materialise, or prefer
    /// [`broadcast`](Self::broadcast) directly.
    ///
    /// # Panics
    ///
    /// Panics if `new_shape.len() != self.ndim()` or any dimension with size > 1 is expanded.
    #[track_caller]
    pub fn broadcast_view(self, new_shape: &[u64]) -> Array<Broadcast<S>> {
        Array::from_storage(Broadcast::new(self, new_shape).unwrap())
    }

    /// Returns a lazy view of the array with the specified length-1 dimensions removed.
    /// See [`RemoveAxes`] for details and examples.
    ///
    /// Each axis in `axes` must have length 1. No data is copied.
    ///
    /// # Panics
    ///
    /// Panics if any axis is out of bounds, duplicated, or has length != 1.
    #[track_caller]
    pub fn remove_axes(self, axes: &[usize]) -> Array<RemoveAxes<S>> {
        Array::from_storage(RemoveAxes::new(self, axes).unwrap())
    }

    /// Returns a lazy view of the array with new length-1 dimensions inserted. See [`InsertAxes`]
    /// for details and examples.
    ///
    /// Each value in `axes` is a gap index in the output shape: `0` inserts before dim 0, `ndim`
    /// appends after the last dim. Duplicates are allowed and each inserts one dimension. No data
    /// is copied.
    ///
    /// # Panics
    ///
    /// Panics if any value in `axes` is > `self.ndim()` or the resulting ndim exceeds the maximum.
    #[track_caller]
    pub fn insert_axes(self, axes: &[usize]) -> Array<InsertAxes<S>> {
        Array::from_storage(InsertAxes::new(self, axes).unwrap())
    }
}
