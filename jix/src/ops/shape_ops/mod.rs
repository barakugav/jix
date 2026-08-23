mod broadcast;
pub use broadcast::*;

mod slice;
pub use slice::*;

mod insert_axis;
pub use insert_axis::*;

mod remove_axis;
pub use remove_axis::*;

mod permute_axes;
pub use permute_axes::*;

mod reshape;
pub use reshape::*;

mod concatenate;
pub use concatenate::*;

mod stack;
pub use stack::*;

mod repeat;
pub use repeat::*;

mod flip;
pub use flip::*;

mod roll;
pub use roll::*;

mod tile;
pub use tile::*;

use crate::ops::AxesArg;
use crate::{Array, ArrayStorage, Dimension, IntoDimension};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Returns a lazy view of the array with a new shape. See [`Reshape`] for details and
    /// examples.
    ///
    /// Like the other shape operations, reshape is lazy: no data is copied at construction time.
    /// Reshape is uniquely prone to read-amplification, though - when the new shape crosses block
    /// boundaries of the original layout, a single read may decompress many more blocks than it
    /// appears to touch. When the result will be read more than once, call
    /// [`.compact()`](Array::compact) to materialize it with a block layout matched to the new
    /// shape.
    ///
    /// # Panics
    ///
    /// Panics if the total number of elements differs or the new ndim exceeds [`NDIM_MAX`](crate::NDIM_MAX).
    #[track_caller]
    pub fn reshape<Sh>(self, shape: Sh) -> Array<Reshape<S, Sh::Dimension>>
    where
        Sh: IntoDimension,
    {
        Reshape::new_array(self, shape).unwrap()
    }

    /// Returns a lazy view of a sub-region of the array. See [`Slice`] for details and examples.
    ///
    /// Accepts a tuple of Rust ranges or [`SliceItem`]s, one per dimension. Negative integer
    /// range bounds are supported (Python-style end-relative indexing).
    ///
    /// # Panics
    ///
    /// Panics if the number of items != `self.ndim()` or any `step < 1`.
    #[track_caller]
    pub fn slice(self, slice: impl Into<SliceSpec>) -> Array<Slice<S>> {
        Slice::new_array(self, slice.into()).unwrap()
    }

    /// Returns a lazy view of the array with its axes reordered. See [`PermuteAxes`] for details
    /// and examples.
    ///
    /// `axes[i]` names the input axis that maps to output axis `i`.
    ///
    /// # Panics
    ///
    /// Panics if `axes` is not a valid permutation of `0..ndim`.
    #[track_caller]
    pub fn permute_axes(self, axes: &[usize]) -> Array<PermuteAxes<S>> {
        PermuteAxes::new_array(self, axes).unwrap()
    }

    /// Returns a lazy view of the array with its axes reversed. See [`PermuteAxes`] for details
    /// and examples.
    #[track_caller]
    pub fn transpose(self) -> Array<PermuteAxes<S>> {
        let ndim = self.ndim();
        let axes = S::Dimension::vec(ndim, |i| ndim - 1 - i);
        PermuteAxes::new_array(self, axes.as_ref()).unwrap()
    }

    /// Returns a lazy view with each element repeated `repeats` times along `axis`.
    /// See [`Repeat`] for details and examples.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`, if `self.ndim() == NDIM_MAX` (one extra
    /// internal axis is required), or if `self.shape()[axis] * repeats` overflows `u64`.
    #[track_caller]
    pub fn repeat(self, repeats: u64, axis: usize) -> Array<Repeat<S>> {
        Repeat::new_array(self, repeats, axis).unwrap()
    }

    /// Returns a lazy view of the array with the order of elements reversed along the
    /// specified axes. See [`Flip`] for details and examples.
    ///
    /// `axis` accepts any [`AxesArg`]: a single `usize`, an array `[usize; N]`, a tuple
    /// `(usize, ...)`, a `Vec<usize>`, or a slice `&[usize]`.
    ///
    /// # Panics
    ///
    /// Panics if any axis is out of bounds or duplicated.
    #[track_caller]
    pub fn flip(self, axis: impl AxesArg) -> Array<Flip<S>> {
        Flip::new_array(self, axis).unwrap()
    }

    /// Returns a lazy view of the array with elements rolled along the given axis.
    /// See [`Roll`] for details and examples.
    ///
    /// `shift` is reduced modulo `shape[axis]`. Positive shifts move elements toward
    /// larger indices (wrapping around at the end); negative shifts move them the other
    /// way.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    #[track_caller]
    pub fn roll(self, shift: i64, axis: usize) -> Array<Roll<S>> {
        Roll::new_array(self, shift, axis).unwrap()
    }

    /// Returns a lazy view of the array replicated `repeats` times along `axis`.
    /// See [`Tile`] for details and examples.
    ///
    /// Unlike NumPy's `tile`, `axis` must satisfy `axis < self.ndim()`; the array is
    /// not extended with new leading dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`, if `self.ndim() == NDIM_MAX` (one extra
    /// internal axis is required), or if `self.shape()[axis] * repeats` overflows `u64`.
    #[track_caller]
    pub fn tile(self, repeats: u64, axis: usize) -> Array<Tile<S>> {
        Tile::new_array(self, repeats, axis).unwrap()
    }

    /// Returns a lazy view of the array expanded to `shape` by repeating length-1 dimensions.
    /// See [`Broadcast`] for details and examples.
    ///
    /// # Panics
    ///
    /// Panics if `shape.len() != self.ndim()` or any dimension with size > 1 is expanded.
    #[track_caller]
    pub fn broadcast(self, shape: &[u64]) -> Array<Broadcast<S>> {
        Broadcast::new_array(self, shape).unwrap()
    }

    /// Returns a lazy view of the array with the specified length-1 dimensions removed.
    /// See [`RemoveAxis`] for details and examples.
    ///
    /// Each axis in `axis` must have length 1.
    ///
    /// # Panics
    ///
    /// Panics if any axis is out of bounds, duplicated, or has length != 1.
    #[track_caller]
    pub fn remove_axis<Ax>(
        self,
        axis: Ax,
    ) -> Array<RemoveAxis<S, Ax::ReducedDimension<S::Dimension>>>
    where
        Ax: AxesArg,
    {
        RemoveAxis::new_array(self, axis).unwrap()
    }

    /// Returns a lazy view of the array with new length-1 dimensions inserted. See [`InsertAxis`]
    /// for details and examples.
    ///
    /// Each value in `axis` is a gap index: `0` inserts before dim 0, `ndim` appends after the
    /// last dim. Duplicates are allowed and each inserts one dimension.
    ///
    /// # Panics
    ///
    /// Panics if any value in `axis` is > `self.ndim()` or the resulting ndim exceeds [`NDIM_MAX`](crate::NDIM_MAX).
    #[track_caller]
    pub fn insert_axis<Ax>(
        self,
        axis: Ax,
    ) -> Array<InsertAxis<S, Ax::ExpandedDimension<S::Dimension>>>
    where
        Ax: AxesArg,
    {
        InsertAxis::new_array(self, axis).unwrap()
    }
}
