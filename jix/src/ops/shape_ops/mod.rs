mod broadcast;
pub use broadcast::*;

mod slice;
pub use slice::*;

mod insert_dim;
pub use insert_dim::*;

mod remove_dim;
pub use remove_dim::*;

mod permute_dims;
pub use permute_dims::*;

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

use crate::ops::DimsArg;
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

    /// Returns a lazy view of the array with its dims reordered. See [`PermuteDims`] for details
    /// and examples.
    ///
    /// `dims[i]` names the input dim that maps to output dim `i`.
    ///
    /// # Panics
    ///
    /// Panics if `dims` is not a valid permutation of `0..ndim`.
    #[track_caller]
    pub fn permute_dims(self, dims: &[usize]) -> Array<PermuteDims<S>> {
        PermuteDims::new_array(self, dims).unwrap()
    }

    /// Returns a lazy view of the array with its dims reversed. See [`PermuteDims`] for details
    /// and examples.
    #[track_caller]
    pub fn transpose(self) -> Array<PermuteDims<S>> {
        let ndim = self.ndim();
        let dims = S::Dimension::vec(ndim, |i| ndim - 1 - i);
        PermuteDims::new_array(self, dims.as_ref()).unwrap()
    }

    /// Returns a lazy view with each element repeated `repeats` times along `dim`.
    /// See [`Repeat`] for details and examples.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= self.ndim()`, if `self.ndim() == NDIM_MAX` (one extra
    /// internal dim is required), or if `self.shape()[dim] * repeats` overflows `u64`.
    #[track_caller]
    pub fn repeat(self, repeats: u64, dim: usize) -> Array<Repeat<S>> {
        Repeat::new_array(self, repeats, dim).unwrap()
    }

    /// Returns a lazy view of the array with the order of elements reversed along the
    /// specified dims. See [`Flip`] for details and examples.
    ///
    /// `dim` accepts any [`DimsArg`]: a single `usize`, an array `[usize; N]`, a tuple
    /// `(usize, ...)`, a `Vec<usize>`, or a slice `&[usize]`.
    ///
    /// # Panics
    ///
    /// Panics if any dim is out of bounds or duplicated.
    #[track_caller]
    pub fn flip(self, dim: impl DimsArg) -> Array<Flip<S>> {
        Flip::new_array(self, dim).unwrap()
    }

    /// Returns a lazy view of the array with elements rolled along the given dim.
    /// See [`Roll`] for details and examples.
    ///
    /// `shift` is reduced modulo `shape[dim]`. Positive shifts move elements toward
    /// larger indices (wrapping around at the end); negative shifts move them the other
    /// way.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= self.ndim()`.
    #[track_caller]
    pub fn roll(self, shift: i64, dim: usize) -> Array<Roll<S>> {
        Roll::new_array(self, shift, dim).unwrap()
    }

    /// Returns a lazy view of the array replicated `repeats` times along `dim`.
    /// See [`Tile`] for details and examples.
    ///
    /// Unlike NumPy's `tile`, `dim` must satisfy `dim < self.ndim()`; the array is
    /// not extended with new leading dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= self.ndim()`, if `self.ndim() == NDIM_MAX` (one extra
    /// internal dim is required), or if `self.shape()[dim] * repeats` overflows `u64`.
    #[track_caller]
    pub fn tile(self, repeats: u64, dim: usize) -> Array<Tile<S>> {
        Tile::new_array(self, repeats, dim).unwrap()
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
    /// See [`RemoveDim`] for details and examples.
    ///
    /// Each dim in `dim` must have length 1.
    ///
    /// # Panics
    ///
    /// Panics if any dim is out of bounds, duplicated, or has length != 1.
    #[track_caller]
    pub fn remove_dim<Ax>(self, dim: Ax) -> Array<RemoveDim<S, Ax::ReducedDimension<S::Dimension>>>
    where
        Ax: DimsArg,
    {
        RemoveDim::new_array(self, dim).unwrap()
    }

    /// Returns a lazy view of the array with new length-1 dimensions inserted. See [`InsertDim`]
    /// for details and examples.
    ///
    /// Each value in `dim` is a gap index: `0` inserts before dim 0, `ndim` appends after the
    /// last dim. Duplicates are allowed and each inserts one dimension.
    ///
    /// # Panics
    ///
    /// Panics if any value in `dim` is > `self.ndim()` or the resulting ndim exceeds [`NDIM_MAX`](crate::NDIM_MAX).
    #[track_caller]
    pub fn insert_dim<Ax>(self, dim: Ax) -> Array<InsertDim<S, Ax::ExpandedDimension<S::Dimension>>>
    where
        Ax: DimsArg,
    {
        InsertDim::new_array(self, dim).unwrap()
    }
}
