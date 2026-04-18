mod broadcast;
pub use broadcast::*;

mod slice;
pub use slice::*;

mod insert_axes;
pub use insert_axes::*;

mod remove_axes;
pub use remove_axes::*;

mod permute_dims;
pub use permute_dims::*;

mod reshape;
pub use reshape::*;

use crate::Array;
use crate::storage::{ArrayStorage, Owned};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    #[track_caller]
    pub fn reshape(self, new_shape: &[u64]) -> Array<Owned> {
        self.reshape_view(new_shape).data().copy().unwrap()
    }

    #[track_caller]
    pub fn reshape_view(self, new_shape: &[u64]) -> Array<Reshape<S>> {
        Array::from_storage(Reshape::new(self, new_shape).unwrap())
    }

    /// Return a lazy view of a sub-region of the array.
    ///
    /// `slice` accepts anything that converts to [`SliceSpec`].  The most ergonomic form is a
    /// tuple — one item per dimension — where each item can be a standard Rust range or a
    /// [`SliceItem`]:
    ///
    /// ```text
    /// array.slice((.., 1..4))              // axis 0: all, axis 1: indices 1, 2, 3
    /// array.slice((2.., ..3))              // axis 0: from 2, axis 1: up to (not including) 3
    /// array.slice((1..=3, ..))             // axis 0: indices 1, 2, 3 (inclusive end)
    /// array.slice((.., SliceItem::new(None, None, 2)))   // axis 1: every other element
    /// ```
    ///
    /// **Negative indices** (Python-style) — use negative integer range literals:
    ///
    /// ```text
    /// array.slice(((-2..), ..))            // axis 0: last 2 elements  (start = len - 2)
    /// array.slice((.., ..-1))              // axis 1: all but the last (end   = len - 1)
    /// array.slice((.., -4..-1))            // axis 1: four-from-end up to one-from-end
    /// ```
    ///
    /// When a step is also needed, use [`SliceItem`] directly (range syntax has no step):
    ///
    /// ```text
    /// array.slice((SliceItem::new(Some(-6), None, 2), ..))  // last-6 to end, every 2nd
    /// ```
    ///
    /// No data is copied at construction time.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is invalid:
    ///
    /// * number of items != `self.ndim()`
    /// * any `step < 1` (for anow)
    #[track_caller]
    pub fn slice(self, slice: impl Into<SliceSpec>) -> Array<Slice<S>> {
        Array::from_storage(Slice::new(self, slice.into()).unwrap())
    }

    /// Return a lazy view of the array with its axes reordered.
    ///
    /// The i-th axis of the returned array corresponds to the axis numbered `axes[i]` of the
    /// input — identical to the convention used by NumPy's `numpy.permute_dims` (also known as
    /// `numpy.transpose`).  No data is copied at construction time; elements are read and
    /// rearranged on demand when the result is materialized.
    ///
    /// # Arguments
    ///
    /// * `axes` — a permutation of `0..ndim`.  Must satisfy:
    ///   - `axes.len() == self.ndim()`
    ///   - every value is in `0..self.ndim()`
    ///   - no value appears more than once
    ///
    /// # Panics
    ///
    /// Panics if `axes` is not a valid permutation of `0..ndim`:
    ///
    /// * `axes.len() != self.ndim()`
    /// * any axis value is ≥ `self.ndim()`
    /// * any axis value is repeated
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    ///
    /// // 2-D transpose — equivalent to np.permute_dims(a, [1, 0])
    /// let a = Array::from_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]].view().into_dyn(), ArrayParams::default()).unwrap();
    /// let t = a.permute_dims(&[1, 0]);
    /// assert_eq!(t.shape(), &[3, 2]);
    /// // t[i, j] == a[j, i]
    ///
    /// // 3-D cyclic permutation — equivalent to np.permute_dims(a, [2, 0, 1])
    /// // output axis 0 ← input axis 2
    /// // output axis 1 ← input axis 0
    /// // output axis 2 ← input axis 1
    /// let a = Array::from_ndarray(&ndarray::Array::from_shape_fn((2,3,4), |(i,j,k)| (i*12+j*4+k) as i32).view().into_dyn(), ArrayParams::default()).unwrap();
    /// let p = a.permute_dims(&[2, 0, 1]);
    /// assert_eq!(p.shape(), &[4, 2, 3]);
    /// ```
    #[track_caller]
    pub fn permute_dims(self, axes: &[usize]) -> Array<PermuteDims<S>> {
        Array::from_storage(PermuteDims::new(self, axes).unwrap())
    }

    /// Expand the array to `new_shape` and return a fully materialized copy.
    ///
    /// Equivalent to calling [`broadcast_view`](Self::broadcast_view) and then copying the result
    /// into a new owned array.  Use `broadcast_view` instead when you only need a lazy view
    /// without allocating.
    ///
    /// # Panics
    ///
    /// See [`broadcast_view`](Self::broadcast_view) for the validity rules on `new_shape`.
    #[track_caller]
    pub fn broadcast(self, new_shape: &[u64]) -> Array<Owned> {
        self.broadcast_view(new_shape).data().copy().unwrap()
    }

    /// Return a lazy view of the array expanded to `new_shape` by repeating elements along
    /// dimensions that have length 1.
    ///
    /// `new_shape` must have the same number of dimensions as the array.  For each dimension:
    /// either it stays the same (`new_shape[d] == self.shape()[d]`), or it is broadcast from
    /// length 1 to `new_shape[d]`.  Attempting to change the size of a dimension that is not
    /// length 1 panics.  No data is copied at construction time.
    ///
    /// # Panics
    ///
    /// Panics if `new_shape` is invalid:
    ///
    /// * `new_shape.len() != self.ndim()`
    /// * any dimension with `input_shape[d] != new_shape[d]` has `input_shape[d] != 1`
    #[track_caller]
    pub fn broadcast_view(self, new_shape: &[u64]) -> Array<Broadcast<S>> {
        Array::from_storage(Broadcast::new(self, new_shape).unwrap())
    }

    /// Return a lazy view of the array with the specified dimensions removed.
    ///
    /// Each value in `axes` names an axis of the input array (0-based index). That axis must
    /// have length exactly 1; attempting to remove a dimension with length > 1 panics. Duplicate
    /// axis indices are not allowed.
    ///
    /// # Panics
    ///
    /// Panics if `axes` is invalid:
    ///
    /// * any axis value is ≥ `self.ndim()`
    /// * any axis value is duplicated
    /// * any named axis has length != 1
    #[track_caller]
    pub fn remove_axes(self, axes: &[usize]) -> Array<RemoveAxes<S>> {
        Array::from_storage(RemoveAxes::new(self, axes).unwrap())
    }

    /// Return a lazy view of the array with new length-1 dimensions inserted at the given gap
    /// positions.
    ///
    /// Each value in `axes` is a **gap index in the input shape**: `0` means "before input dim 0",
    /// `1` means "between input dims 0 and 1", ..., `ndim` means "after the last input dim".
    /// Duplicate values are allowed — each occurrence inserts one additional dimension at that gap.
    /// The order of values in `axes` does not matter; only the multiset of gap indices matters.
    /// No data is copied at construction time or at read time.
    ///
    /// # Panics
    ///
    /// Panics if `axes` is invalid:
    ///
    /// * any axis value is > `self.ndim()` (valid gap indices are `0..=self.ndim()`)
    /// * the resulting ndim would exceed the maximum allowed ndim
    #[track_caller]
    pub fn insert_axes(self, axes: &[usize]) -> Array<InsertAxes<S>> {
        Array::from_storage(InsertAxes::new(self, axes).unwrap())
    }
}
