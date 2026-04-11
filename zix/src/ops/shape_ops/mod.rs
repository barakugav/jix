mod permute_dims;
pub use permute_dims::*;

mod reshape;
pub use reshape::*;

use crate::Array;
use crate::storage::ArrayStorage;

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Return a lazy view of the array with its axes reordered.
    ///
    /// The i-th axis of the returned array corresponds to the axis numbered `axes[i]` of the
    /// input — identical to the convention used by NumPy's `numpy.permute_dims` (also known as
    /// `numpy.transpose`).  No data is copied at construction time; elements are read and
    /// rearranged on demand when the result is materialised.
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
        Array::from_storage(PermuteDims::new(self.storage, axes).unwrap())
    }

    #[track_caller]
    pub fn reshape_view(self, new_shape: &[u64]) -> Array<Reshape<S>> {
        let a = Array::from_storage(self.storage);
        Array::from_storage(Reshape::new(a, new_shape).unwrap())
    }
}
