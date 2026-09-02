use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, ensure, Result};
use crate::ops::AxesArg;
use crate::storage::{check_out_buf, materialize_out_buf, ArraySpec, ArrayStorageInfo, StridedBuf};
use crate::util::iter::NdIter;
use crate::{Array, ArrayStorage, Dimension, NdCopier};

/// Reverses the order of elements along one or more axes, returned by
/// [`Array::flip`](crate::Array::flip).
///
/// Output shape and dtype equal the input. Axes not listed in `axes` are passed through
/// unchanged. `axes` may be empty (no-op), and axes of size 0 or 1 do not require any
/// data motion.
///
/// See also [`Roll`](crate::ops::Roll), which cyclically shifts elements along an axis
/// without reversing them.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// `axis` accepts any [`AxesArg`]: a single `usize`, a fixed-size array `[usize; N]`,
/// a tuple `(usize, ...)`, a `Vec<usize>`, or a slice `&[usize]`.
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]])?;
/// // Flip along axis 0 reverses the row order (single axis: pass a usize)
/// assert_eq!(a.view().flip(0).to_ndarray()?, array![[4, 5, 6], [1, 2, 3]]);
/// // Flip along axis 1 reverses each row
/// assert_eq!(a.view().flip(1).to_ndarray()?, array![[3, 2, 1], [6, 5, 4]]);
/// // Flip along both axes (multiple axes: pass an array or tuple)
/// assert_eq!(a.flip([0, 1]).to_ndarray()?, array![[6, 5, 4], [3, 2, 1]]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Flip<S: ArrayStorage> {
    array: S,
    is_flipped: <S::Dimension as Dimension>::Vec<bool>,
}

impl<S: ArrayStorage> Flip<S> {
    /// Constructs a [`Flip`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, axis: impl AxesArg) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();

        let mut is_flipped = S::Dimension::vec(ndim, |_| false);
        for i in 0..axis.len() {
            let ax = axis.get(i);
            ensure!(
                ax < ndim,
                InvalidShapeOperation,
                "flip axis {ax} is out of bounds for array with ndim {ndim}"
            );
            ensure!(
                !is_flipped[ax],
                InvalidShapeOperation,
                "duplicate axis {ax} in flip"
            );
            is_flipped[ax] = true;
        }
        Ok(Self { array, is_flipped })
    }

    /// Constructs an array with [`Flip`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, axis: impl AxesArg) -> Result<Array<Self>> {
        Self::new(array.into_storage(), axis).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Flip<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_get_range(self.shape(), index)?;
        check_out_buf(out.as_deref(), self.shape())?;

        let ndim = index.len();
        let dtype = self.dtype();
        let out_shape = S::Dimension::vec(ndim, |d| (index[d].end - index[d].start) as usize);

        let shape = self.shape();
        // For each flipped axis d with requested output range [s, e), the inner range is
        // [shape[d]-e, shape[d]-s) (same length, reversed position). Non-flipped axes pass through.
        let inner_index = S::Dimension::vec(ndim, |d| {
            if self.is_flipped[d] {
                (shape[d] - index[d].end)..(shape[d] - index[d].start)
            } else {
                index[d].clone()
            }
        });

        // Read the (unreversed) inner region as a view, then mirror-copy slabs into the destination.
        let tmp = self.array.read_data(inner_index.as_ref(), context, None)?;
        let (tmp_buf, src_strides) = tmp.data();
        let mut out = materialize_out_buf(out, context, out_shape.as_ref(), dtype);
        if out_shape.as_ref().contains(&0) {
            return Ok(out);
        }
        let (out_buf, out_strides) = out.data_mut();

        // Iterate one slab at a time. Each slab is a single combination of indices on the
        // flipped axes; non-flipped axes are copied contiguously via nd_copy per slab.
        let iter_shape = S::Dimension::vec(ndim, |d| {
            if self.is_flipped[d] {
                out_shape[d] as u64
            } else {
                1
            }
        });
        let slab_shape =
            S::Dimension::vec(ndim, |d| if self.is_flipped[d] { 1 } else { out_shape[d] });

        // Step src through the inner view on flipped axes; 0 on non-flipped (iter_shape=1 there).
        let src_ptr_strides = S::Dimension::vec(ndim, |d| {
            if self.is_flipped[d] {
                src_strides[d]
            } else {
                0
            }
        });

        let iter = NdIter::builder(iter_shape)
            .with_strides_offset_ext(src_ptr_strides, 0)
            .build();
        let nd_copy = NdCopier::new(dtype);
        for (idx, src_off) in iter {
            // The output position on a flipped axis mirrors the source position:
            // out_idx = L-1 - src_idx, placed at the destination's own strides.
            let dst_off = (0..ndim)
                .filter(|&d| self.is_flipped[d])
                .map(|d| (out_shape[d] - 1 - idx[d] as usize) * out_strides[d])
                .sum::<usize>();

            unsafe {
                nd_copy.copy(
                    tmp_buf.get_unchecked(src_off..),
                    out_buf.get_unchecked_mut(dst_off..),
                    slab_shape.as_ref(),
                    src_strides,
                    out_strides,
                    dtype,
                )
            };
        }
        Ok(out)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array.spec().with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Flip", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Flip<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let ndim = self.shape().len();
        let array = self.array.dimension_change()?;
        let is_flipped = NewD::vec(ndim, |d| self.is_flipped[d]);
        Ok(Flip { array, is_flipped })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Flip<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Flip {
            array: self.array.element_type_change()?,
            is_flipped: self.is_flipped,
        })
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::array;

    use crate::codec::ReadContext;
    use crate::storage::Compact;
    use crate::{Array, IntoDimension, Ty};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make<Sh>(vals: Vec<i32>, shape: Sh) -> Array<Compact<Ty<i32>, Sh::Dimension>>
    where
        Sh: IntoDimension,
    {
        let shape = shape.into_dimension().unwrap();
        let nd = ndarray::Array::from_shape_vec(shape, vals).unwrap();
        Array::compact_ndarray(&nd).unwrap().into_dim().unwrap()
    }

    fn arange(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata (flip preserves shape)
    // -----------------------------------------------------------------------

    #[test]
    fn shape_preserved_single_axis() {
        assert_eq!(make(arange(12), &[3u64, 4]).flip(0).shape(), &[3, 4]);
    }

    #[test]
    fn shape_preserved_all_axes() {
        assert_eq!(
            make(arange(24), &[2u64, 3, 4]).flip([0, 1, 2]).shape(),
            &[2, 3, 4]
        );
    }

    #[test]
    fn shape_preserved_empty_axes() {
        assert_eq!(make(arange(12), &[3u64, 4]).flip([]).shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // AxesArg input forms
    // -----------------------------------------------------------------------

    #[test]
    fn input_form_single_usize() {
        let got = make(arange(6), &[3u64, 2]).flip(0).to_ndarray().unwrap();
        assert_eq!(got, array![[4, 5], [2, 3], [0, 1]]);
    }

    #[test]
    fn input_form_array_two_axes() {
        let got = make(arange(6), &[3u64, 2])
            .flip([0, 1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[5, 4], [3, 2], [1, 0]]);
    }

    #[test]
    fn input_form_tuple_two_axes() {
        let got = make(arange(6), &[3u64, 2])
            .flip((0, 1))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[5, 4], [3, 2], [1, 0]]);
    }

    #[test]
    fn input_form_slice_dynamic() {
        let axes = vec![0, 1];
        let got = make(arange(6), &[3u64, 2])
            .flip(axes.as_slice())
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[5, 4], [3, 2], [1, 0]]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make(arange(12), &[3u64, 4]);
        let err = super::Flip::new_array(a.view(), 2).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn error_duplicate_axis() {
        let a = make(arange(12), &[3u64, 4]);
        let err = super::Flip::new_array(a.view(), [1, 1]).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    // -----------------------------------------------------------------------
    // Fast paths: identity (no axis with size > 1)
    // -----------------------------------------------------------------------

    #[test]
    fn identity_empty_axes_full_read() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4]).flip([]).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_size_one_flipped_axis() {
        // Flipping a size-1 axis is a no-op.
        let nd = ndarray::Array::from_shape_vec((1, 4), arange(4)).unwrap();
        let got = make(arange(4), &[1u64, 4]).flip([0]).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_empty_array() {
        // Flipping an axis of size 0 is a no-op (no data to move).
        let got = make(vec![], &[0u64, 4]).flip([0]).to_ndarray().unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_single_axis() {
        let got = make(vec![10, 20, 30, 40], &[4u64])
            .flip([0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![40, 30, 20, 10]);
    }

    #[test]
    fn full_read_2d_axis0() {
        let got = make(arange(8), &[2u64, 4]).flip([0]).to_ndarray().unwrap();
        assert_eq!(got, array![[4, 5, 6, 7], [0, 1, 2, 3]]);
    }

    #[test]
    fn full_read_2d_axis1() {
        let got = make(arange(8), &[2u64, 4]).flip([1]).to_ndarray().unwrap();
        assert_eq!(got, array![[3, 2, 1, 0], [7, 6, 5, 4]]);
    }

    #[test]
    fn full_read_2d_both_axes() {
        let got = make(arange(8), &[2u64, 4])
            .flip([0, 1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[7, 6, 5, 4], [3, 2, 1, 0]]);
    }

    #[test]
    fn full_read_3d_middle_axis() {
        let arr = ndarray::Array::from_shape_vec((2, 3, 2), arange(12)).unwrap();
        // Expected: per (i, k), values along j are reversed.
        let mut expected = arr.clone();
        for i in 0..2 {
            for k in 0..2 {
                let col: Vec<i32> = (0..3).map(|j| arr[(i, j, k)]).collect();
                for (j, v) in col.iter().rev().enumerate() {
                    expected[(i, j, k)] = *v;
                }
            }
        }
        let got = make(arange(12), &[2u64, 3, 2])
            .flip([1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_1d_partial() {
        let got = make(vec![10, 20, 30, 40, 50], &[5u64])
            .flip([0])
            .to_ndarray_sub(&[1..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![40, 30, 20]);
    }

    #[test]
    fn sub_read_2d_axis0_partial_rows() {
        let got = make(arange(9), &[3u64, 3])
            .flip([0])
            .to_ndarray_sub(&[0..2, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[6, 7, 8], [3, 4, 5]]);
    }

    #[test]
    fn sub_read_2d_both_axes_corner() {
        let got = make(arange(12), &[3u64, 4])
            .flip([0, 1])
            .to_ndarray_sub(&[0..2, 1..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[10, 9], [6, 5]]);
    }

    #[test]
    fn sub_read_empty_range() {
        let got = make(arange(8), &[2u64, 4])
            .flip([0])
            .to_ndarray_sub(&[1..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    #[test]
    fn sub_read_single_element() {
        // Single-element read still goes through the flip path.
        let got = make(arange(5), &[5u64])
            .flip([0])
            .to_ndarray_sub(&[0..1], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![4]);
    }

    // -----------------------------------------------------------------------
    // Composition with other ops
    // -----------------------------------------------------------------------

    #[test]
    fn compose_flip_then_compact() {
        let got = make(arange(6), &[2u64, 3])
            .flip([0])
            .compact()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_flip_then_slice() {
        let got = make(arange(9), &[3u64, 3])
            .flip([0])
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_flip_then_cast() {
        let got = make(vec![1, 2, 3], &[3u64])
            .flip([0])
            .cast::<f32>()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![3.0f32, 2.0, 1.0]);
    }

    #[test]
    fn compose_permute_then_flip() {
        let got = make(arange(6), &[2u64, 3])
            .permute_axes(&[1, 0])
            .flip([0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[2, 5], [1, 4], [0, 3]]);
    }

    #[test]
    fn compose_flip_then_flip_same_axis_is_identity() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .flip([0])
            .flip([0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    // -----------------------------------------------------------------------
    // Dimension and element type preservation
    // -----------------------------------------------------------------------

    #[test]
    fn dim_change_into_dim_dyn() {
        let a = make(arange(12), &[3u64, 4]); // Compact<Ty<i32>, Dim<2>>
        let f = a.flip([0]); // Array<Flip<...Dim<2>>>
        let dyn_arr = f.into_dim_dyn();
        assert_eq!(dyn_arr.shape(), &[3, 4]);
    }

    #[test]
    fn element_type_change_into_type_dyn() {
        let a = make(arange(6), &[2u64, 3]);
        let f = a.flip([1]);
        let dyn_et = f.into_type_dyn();
        assert_eq!(dyn_et.dtype(), &<i32 as crate::dtype::Dtyped>::DTYPE);
    }

    // -----------------------------------------------------------------------
    // Proptest: random shape + axes vs hand-rolled reference
    // -----------------------------------------------------------------------

    fn flip_reference<T: Clone + Default>(
        nd: &ndarray::ArrayD<T>,
        axes: &[usize],
    ) -> ndarray::ArrayD<T> {
        let mut out = nd.clone();
        for &ax in axes {
            out.invert_axis(ndarray::Axis(ax));
        }
        // Materialize so the result has standard strides.
        ndarray::ArrayD::from_shape_vec(out.shape().to_vec(), out.iter().cloned().collect())
            .unwrap()
    }

    #[allow(clippy::type_complexity)]
    fn flip_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            crate::util::TestArray<i32>,
            Vec<usize>,
        ),
    > {
        use proptest::prelude::*;

        let shape = crate::util::shape_strategy().prop_filter("non-empty ndim", |s| !s.is_empty());

        shape.prop_flat_map(|shape| {
            let ndim = shape.len();
            let array_strat = crate::util::array_strategy_from_shape::<i32>(
                Just(shape.clone()),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            );
            // Random subset of axes 0..ndim
            let axes_mask = proptest::collection::vec(any::<bool>(), ndim);
            (array_strat, axes_mask).prop_map(|((nd, za), mask)| {
                let axes = mask
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, b)| if b { Some(i) } else { None })
                    .collect();
                (nd, za, axes)
            })
        })
    }

    proptest::proptest! {
        #[test]
        fn proptest_flip_generic(
            (nd, za, axes) in flip_strategy()
        ) {
            let expected = flip_reference(&nd, &axes);
            let actual = za.flip(&axes);
            crate::util::assert_array_matches(&actual, &expected);
        }
    }
}
