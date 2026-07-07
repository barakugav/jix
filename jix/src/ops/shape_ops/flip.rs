use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::ops::AxesArg;
use crate::storage::{ArraySpec, ArrayStorageInfo, OutBuf};
use crate::util::iter::strides::NdIterExtStridesPtr;
use crate::util::iter::NdIter;
use crate::util::{default_strides, nd_copy, DimArray};
use crate::{Array, ArrayStorage, Dimension};

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
/// assert_eq!(a.as_ref().flip(0).to_ndarray()?, array![[4, 5, 6], [1, 2, 3]]);
/// // Flip along axis 1 reverses each row
/// assert_eq!(a.as_ref().flip(1).to_ndarray()?, array![[3, 2, 1], [6, 5, 4]]);
/// // Flip along both axes (multiple axes: pass an array or tuple)
/// assert_eq!(a.flip([0, 1]).to_ndarray()?, array![[6, 5, 4], [3, 2, 1]]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Flip<S: ArrayStorage> {
    array: S,
    /// User-provided axes after dedup + sort + bounds check. May include size-1 axes
    /// (preserved as-is for introspection; they do not affect `read_data`).
    axes: DimArray<usize>,
}

impl<S: ArrayStorage> Flip<S> {
    /// Constructs a [`Flip`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, axis: impl AxesArg) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();

        let mut seen = S::Dimension::vec(ndim, |_| false);
        for i in 0..axis.len() {
            let ax = axis.get(i);
            ensure!(
                ax < ndim,
                InvalidShapeOperation,
                "flip axis {ax} is out of bounds for array with ndim {ndim}"
            );
            ensure!(
                !seen[ax],
                InvalidShapeOperation,
                "duplicate axis {ax} in flip"
            );
            seen[ax] = true;
        }
        let sorted_axes = (0..ndim).filter(|d| seen[*d]).collect::<DimArray<_>>();

        Ok(Self {
            array,
            axes: sorted_axes,
        })
    }

    /// Constructs an array with [`Flip`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, axis: impl AxesArg) -> Result<Array<Self>> {
        Self::new(array.into_storage(), axis).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Flip<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        check_get_range(self.shape(), index)?;

        if index.iter().any(|r| r.start >= r.end) {
            buf.get_mut(index, self.dtype()); // ensure buffer is allocated for empty read
            return Ok(());
        }

        let ndim = index.len();
        let dtype = self.dtype();
        let itemsize = dtype.itemsize() as usize;
        let shape = self.shape();

        let mut is_flipped = S::Dimension::vec(ndim, |_| false);
        for &ax in self.axes.iter() {
            is_flipped[ax] = true;
        }

        // For each flipped axis d with requested output range [s, e), the inner range is
        // [shape[d]-e, shape[d]-s) (same length, reversed position). Non-flipped axes pass through.
        let inner_index = S::Dimension::vec(ndim, |d| {
            if is_flipped[d] {
                (shape[d] - index[d].end)..(shape[d] - index[d].start)
            } else {
                index[d].clone()
            }
        });

        let out_shape = S::Dimension::vec(ndim, |d| index[d].end - index[d].start);
        let mut tmp_buf = OutBuf::new_lazy(context);
        self.array
            .read_data(inner_index.as_ref(), &mut tmp_buf, context)?;
        let tmp_buf = tmp_buf.as_slice().unwrap();
        let buf = buf.get_mut(index, dtype);
        check_get_buffer_size(index, dtype, buf)?;

        // tmp_buf is C-contiguous over out_shape (sub_shape_in == out_shape).
        let strides = default_strides(out_shape.as_ref(), itemsize as u64);

        // Iterate one slab at a time. Each slab is a single combination of indices on the
        // flipped axes; non-flipped axes are copied contiguously via nd_copy per slab.
        let iter_shape =
            S::Dimension::from_fn(ndim, |d| if is_flipped[d] { out_shape[d] } else { 1 });
        let slab_shape =
            S::Dimension::from_fn(ndim, |d| if is_flipped[d] { 1 } else { out_shape[d] });

        // src strides ext: forward strides on flipped axes; 0 elsewhere (non-flipped axes
        // are iter_shape=1 so they don't step regardless, but 0 keeps it explicit).
        let src_ptr_strides =
            S::Dimension::vec(ndim, |d| if is_flipped[d] { strides[d] } else { 0 });

        // dst pointer base = position where every flipped axis is at its MAX index.
        // As src advances forward by some byte offset along flipped axes, dst moves the
        // same offset BACKWARD from this base (since dst_idx = L-1 - src_idx on flipped axes).
        let dst_base_offset = (0..ndim)
            .filter(|&d| is_flipped[d])
            .map(|d| (out_shape[d] - 1) * strides[d])
            .sum::<u64>();
        let tmp_base = tmp_buf.as_ptr();
        let dst_base = unsafe { buf.as_mut_ptr().add(dst_base_offset as usize) };

        let iter = NdIter::new(
            iter_shape,
            NdIterExtStridesPtr::new(src_ptr_strides.as_ref(), tmp_base),
        );
        for (_idx, src_ptr) in iter {
            let off = unsafe { src_ptr.offset_from(tmp_base) } as usize;
            let dst_ptr = unsafe { dst_base.sub(off) };

            unsafe {
                nd_copy(
                    src_ptr,
                    dst_ptr,
                    slab_shape.clone(),
                    &strides,
                    &strides,
                    itemsize,
                )
            };
        }
        Ok(())
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
    #[inline]
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Flip", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Flip<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Flip {
            array: self.array.dimension_change()?,
            axes: self.axes,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Flip<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Flip {
            array: self.array.element_type_change()?,
            axes: self.axes,
        })
    }
}

#[cfg(test)]
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
        // Single axis: pass a usize.
        assert_eq!(make(arange(12), &[3u64, 4]).flip(0).shape(), &[3, 4]);
    }

    #[test]
    fn shape_preserved_all_axes() {
        // Three axes: pass an array.
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
        // .flip(usize) - single axis.
        let got = make(arange(6), &[3u64, 2]).flip(0).to_ndarray().unwrap();
        assert_eq!(got, array![[4, 5], [2, 3], [0, 1]]);
    }

    #[test]
    fn input_form_array_two_axes() {
        // .flip([usize; 2]) - two axes via fixed-size array.
        let got = make(arange(6), &[3u64, 2])
            .flip([0, 1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[5, 4], [3, 2], [1, 0]]);
    }

    #[test]
    fn input_form_tuple_two_axes() {
        // .flip((usize, usize)) - two axes via tuple.
        let got = make(arange(6), &[3u64, 2])
            .flip((0, 1))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[5, 4], [3, 2], [1, 0]]);
    }

    #[test]
    fn input_form_slice_dynamic() {
        // .flip(&[usize]) - dynamic axis count via slice.
        let axes: Vec<usize> = vec![0, 1];
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
        let err = super::Flip::new_array(a.as_ref(), 2).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn error_duplicate_axis() {
        let a = make(arange(12), &[3u64, 4]);
        let err = super::Flip::new_array(a.as_ref(), [1, 1]).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    // -----------------------------------------------------------------------
    // Fast paths: identity (no axis with size > 1)
    // -----------------------------------------------------------------------

    #[test]
    fn identity_empty_axes_full_read() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4]).flip(&[]).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_size_one_flipped_axis() {
        // Flipping a size-1 axis is a no-op.
        let nd = ndarray::Array::from_shape_vec((1, 4), arange(4)).unwrap();
        let got = make(arange(4), &[1u64, 4]).flip(&[0]).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_empty_array() {
        // Flipping an axis of size 0 is a no-op (no data to move).
        let got = make(vec![], &[0u64, 4]).flip(&[0]).to_ndarray().unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_single_axis() {
        // [10, 20, 30, 40] flip axis 0 -> [40, 30, 20, 10]
        let got = make(vec![10, 20, 30, 40], &[4u64])
            .flip(&[0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![40, 30, 20, 10]);
    }

    #[test]
    fn full_read_2d_axis0() {
        // [[0,1,2,3],[4,5,6,7]] flip axis 0 -> rows reversed
        let got = make(arange(8), &[2u64, 4]).flip(&[0]).to_ndarray().unwrap();
        assert_eq!(got, array![[4, 5, 6, 7], [0, 1, 2, 3]]);
    }

    #[test]
    fn full_read_2d_axis1() {
        // [[0,1,2,3],[4,5,6,7]] flip axis 1 -> columns reversed within each row
        let got = make(arange(8), &[2u64, 4]).flip(&[1]).to_ndarray().unwrap();
        assert_eq!(got, array![[3, 2, 1, 0], [7, 6, 5, 4]]);
    }

    #[test]
    fn full_read_2d_both_axes() {
        // [[0,1,2,3],[4,5,6,7]] flip axes [0, 1] -> all reversed
        let got = make(arange(8), &[2u64, 4])
            .flip(&[0, 1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[7, 6, 5, 4], [3, 2, 1, 0]]);
    }

    #[test]
    fn full_read_3d_middle_axis() {
        // Flip axis 1 of a 3-D array reverses only the middle axis.
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
            .flip(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_1d_partial() {
        // [10, 20, 30, 40, 50] flip axis 0 -> [50, 40, 30, 20, 10]
        // Read output positions [1..4) -> [40, 30, 20]
        let got = make(vec![10, 20, 30, 40, 50], &[5u64])
            .flip(&[0])
            .to_ndarray_sub(&[1..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![40, 30, 20]);
    }

    #[test]
    fn sub_read_2d_axis0_partial_rows() {
        // [[0,1,2],[3,4,5],[6,7,8]] flip axis 0 -> [[6,7,8],[3,4,5],[0,1,2]]
        // Read rows [0..2) -> [[6,7,8],[3,4,5]]
        let got = make(arange(9), &[3u64, 3])
            .flip(&[0])
            .to_ndarray_sub(&[0..2, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[6, 7, 8], [3, 4, 5]]);
    }

    #[test]
    fn sub_read_2d_both_axes_corner() {
        // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] flip [0,1] -> [[11,10,9,8],[7,6,5,4],[3,2,1,0]]
        // Read sub [0..2, 1..3) -> [[10, 9], [6, 5]]
        let got = make(arange(12), &[3u64, 4])
            .flip(&[0, 1])
            .to_ndarray_sub(&[0..2, 1..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[10, 9], [6, 5]]);
    }

    #[test]
    fn sub_read_empty_range() {
        let got = make(arange(8), &[2u64, 4])
            .flip(&[0])
            .to_ndarray_sub(&[1..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    #[test]
    fn sub_read_single_element() {
        // Single-element read still goes through the flip path.
        let got = make(arange(5), &[5u64])
            .flip(&[0])
            .to_ndarray_sub(&[0..1], &ReadContext::default())
            .unwrap();
        // output[0] is what was input[4]
        assert_eq!(got, array![4]);
    }

    // -----------------------------------------------------------------------
    // Composition with other ops
    // -----------------------------------------------------------------------

    #[test]
    fn compose_flip_then_compact() {
        let got = make(arange(6), &[2u64, 3])
            .flip(&[0])
            .compact()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_flip_then_slice() {
        // Flip axis 0, then slice the bottom two rows.
        let got = make(arange(9), &[3u64, 3])
            .flip(&[0])
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        // Flipped: [[6,7,8],[3,4,5],[0,1,2]]; rows 1..3 = [[3,4,5],[0,1,2]]
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_flip_then_cast() {
        let got = make(vec![1, 2, 3], &[3u64])
            .flip(&[0])
            .cast::<f32>()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![3.0f32, 2.0, 1.0]);
    }

    #[test]
    fn compose_permute_then_flip() {
        // [[0,1,2],[3,4,5]] permute -> [[0,3],[1,4],[2,5]]; flip axis 0 -> [[2,5],[1,4],[0,3]]
        let got = make(arange(6), &[2u64, 3])
            .permute_axes(&[1, 0])
            .flip(&[0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[2, 5], [1, 4], [0, 3]]);
    }

    #[test]
    fn compose_flip_then_flip_same_axis_is_identity() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .flip(&[0])
            .flip(&[0])
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
        let f = a.flip(&[0]); // Array<Flip<...Dim<2>>>
        let dyn_arr = f.into_dim_dyn();
        assert_eq!(dyn_arr.shape(), &[3, 4]);
    }

    #[test]
    fn element_type_change_into_type_dyn() {
        let a = make(arange(6), &[2u64, 3]);
        let f = a.flip(&[1]);
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

    fn flip_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            Array<Compact<Ty<i32>, crate::DimDyn>>,
            Vec<usize>,
        ),
    > {
        use proptest::prelude::*;

        let shape = crate::util::shape_strategy().prop_filter("non-empty ndim", |s| !s.is_empty());

        shape.prop_flat_map(|shape| {
            let ndim = shape.len();
            let array_strat = crate::util::carray_strategy_from_shape::<i32>(
                Just(shape.clone()),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            );
            // Random subset of axes 0..ndim
            let axes_mask = proptest::collection::vec(any::<bool>(), ndim);
            (array_strat, axes_mask).prop_map(|((nd, za), mask)| {
                let axes: Vec<usize> = mask
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
