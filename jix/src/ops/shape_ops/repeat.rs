use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, check_shape_overflow, ensure, error, Result};
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{
    check_out_buf, materialize_out_buf, ArraySpec, ArrayStorageInfo, BlockSize, StridedBuf,
};
use crate::util::calc_block_end;
use crate::{Array, ArrayStorage, DimArray, DimIdx, Dimension, NdCopier, NDIM_MAX};

/// Replicates each element along an axis by a scalar count, returned by
/// [`Array::repeat`](crate::Array::repeat).
///
/// This differs from [`Tile`](crate::ops::Tile): `repeat` replicates each element in place
/// `(a, b, c, ...) -> (a, a, b, b, c, c, ...)` whereas `tile` repeats the whole sequence
/// `(a, b, c, ...) -> (a, b, c, a, b, c, ...)`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a1 = Array::compact_ndarray(&array![17])?;
/// let r1 = a1.repeat(4, 0).to_ndarray()?;
/// assert_eq!(r1, array![17, 17, 17, 17]);
///
/// let a2 = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]])?;
/// let r2 = a2.repeat(2, 1).to_ndarray()?;
/// assert_eq!(r2, array![[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Repeat<S: ArrayStorage> {
    array: S,
    axis: DimIdx,
    repeats: u64,
    new_shape: S::Dimension,
    spec: ArraySpecDynamic,
}

impl<S: ArrayStorage> Repeat<S> {
    /// Constructs a [`Repeat`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, repeats: u64, axis: usize) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();

        ensure!(
            ndim < NDIM_MAX,
            InvalidShapeOperation,
            "repeat requires ndim < NDIM_MAX ({NDIM_MAX}); got {ndim} \
             (one extra axis is used internally during reads)"
        );
        ensure!(
            axis < ndim,
            InvalidShapeOperation,
            "repeat axis {axis} is out of bounds for array with ndim {ndim}"
        );

        let new_len = input_shape[axis].checked_mul(repeats).ok_or_else(|| {
            error!(
                InvalidShapeOperation,
                "repeat overflow: shape[{axis}] ({}) * repeats ({}) exceeds u64",
                input_shape[axis],
                repeats,
            )
        })?;

        let new_shape =
            S::Dimension::from_fn(ndim, |d| if d == axis { new_len } else { input_shape[d] });
        check_shape_overflow(new_shape.as_slice(), array.dtype().itemsize() as _)?;

        let inner_spec = array.spec();
        let mut block_shape = inner_spec.block_shape().clone();
        // A repeat preserves whether the repeated dimension is fixed: a fixed block length is
        // scaled with the repeat count and stays fixed, a non-fixed one stays non-fixed.
        let block_shape_fixed_dims = inner_spec.block_shape_fixed_dims();
        block_shape[axis] = block_shape[axis]
            .saturating_mul(repeats.min(BlockSize::MAX as u64) as BlockSize)
            .min(new_len.min(BlockSize::MAX as u64) as BlockSize)
            .max(1);
        // Repeating re-reads each inner element `repeats` times along `axis`; covering it in full
        // with one read avoids that, so give `axis` the highest scaling priority (front of order),
        // keeping the inner relative order among the rest.
        let in_order = inner_spec.read_shape_scale_order();
        let read_shape_scale_order = std::iter::once(axis as DimIdx)
            .chain(in_order.iter().copied().filter(|&d| d as usize != axis))
            .collect::<DimArray<_>>();
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims,
            element_cost: inner_spec.element_cost(),
            read_shape_scale_order,
        };

        Ok(Self {
            array,
            axis: axis as DimIdx,
            repeats,
            new_shape,
            spec,
        })
    }

    /// Constructs an array with [`Repeat`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, repeats: u64, axis: usize) -> Result<Array<Self>> {
        Self::new(array.into_storage(), repeats, axis).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Repeat<S> {
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

        let ndim = self.new_shape.ndim();
        let dtype = self.dtype();
        let out_shape = S::Dimension::vec(ndim, |d| (index[d].end - index[d].start) as usize);

        // Empty output (any zero-length range, including repeats == 0) is a no-op.
        if out_shape.as_ref().contains(&0) {
            return Ok(materialize_out_buf(out, context, out_shape.as_ref(), dtype));
        }

        let k = self.axis as usize;
        let n = self.repeats; // n > 0
        let s = index[k].start;
        let e = index[k].end;
        let g_start = s / n;
        let g_end = calc_block_end(s, e, n);

        // Read the inner sub-region with axis k collapsed to [g_start..g_end) as a view.
        let inner_index = S::Dimension::vec(ndim, |d| {
            if d == k {
                g_start..g_end
            } else {
                index[d].clone()
            }
        });
        let inner_buf = self.array.read_data(inner_index.as_ref(), context, None)?;
        let (inner_buf, inner_strides) = inner_buf.data();

        let mut out = materialize_out_buf(out, context, out_shape.as_ref(), dtype);
        let (out_buf, out_strides) = out.data_mut();
        let copier = NdCopier::new(dtype);

        // Issue one nd_copy for a (g_range, p_range) region.
        //   `g_range` indexes groups relative to the inner read (0..g_count).
        //   `p_range` is the within-group output range [0..n).
        let mut copy_region = |g_range: Range<u64>, p_range: Range<u64>| {
            let g_len = g_range.end - g_range.start;
            let p_len = p_range.end - p_range.start;
            if g_len == 0 || p_len == 0 {
                return;
            }

            // (ndim + 1)-D shape: original axes, but axis k is split into
            // (g_len, p_len) at positions k and k+1.
            let mut copy_shape = <S::Dimension as Dimension>::Larger::vec(ndim + 1, |_| 0);
            copy_shape[..k].copy_from_slice(&out_shape[..k]);
            copy_shape[k] = g_len as usize;
            copy_shape[k + 1] = p_len as usize;
            copy_shape[k + 2..].copy_from_slice(&out_shape[k + 1..]);

            // src strides: itemsize-strides over inner_shape, with the within-group
            // (p) axis stride = 0 (the repeat trick).
            let mut src_strides = <S::Dimension as Dimension>::Larger::vec(ndim + 1, |_| 0);
            src_strides[..k + 1].copy_from_slice(&inner_strides[..k + 1]);
            src_strides[k + 1] = 0;
            src_strides[k + 2..].copy_from_slice(&inner_strides[k + 1..]);

            // dst strides: from `dst_strides`, with axis k split into
            // (n * dst_strides[k], dst_strides[k]).
            let mut dst_strides_split = <S::Dimension as Dimension>::Larger::vec(ndim + 1, |_| 0);
            dst_strides_split[..k].copy_from_slice(&out_strides[..k]);
            dst_strides_split[k] = out_strides[k] * n as usize;
            dst_strides_split[k + 1..].copy_from_slice(&out_strides[k..]);

            // src slice: tmp_buf from (g_range.start) along the inner k axis to its end.
            let src_byte_offset = (g_range.start as usize) * inner_strides[k];
            let src = unsafe { inner_buf.get_unchecked(src_byte_offset..) };

            // dst slice: buf from the first output position of this region to its end.
            // First output k-position = g_start*n + g_range.start*n + p_range.start.
            // Output-relative k-position = that minus s.
            let first_out_k = g_start * n + g_range.start * n + p_range.start;
            debug_assert!(first_out_k >= s);
            let dst_k_offset_units = first_out_k - s;
            let dst_byte_offset = (dst_k_offset_units as usize) * out_strides[k];
            let dst = unsafe { out_buf.get_unchecked_mut(dst_byte_offset..) };

            unsafe {
                copier.copy(
                    src,
                    dst,
                    copy_shape.as_ref(),
                    src_strides.as_ref(),
                    dst_strides_split.as_ref(),
                    dtype,
                )
            };
        };

        let g_count = g_end - g_start;
        let p_start = s - g_start * n; // = s mod n, in 0..n
        let p_end_tail = e - (g_end - 1) * n; // = ((e-1) mod n) + 1, in 1..=n

        if g_count == 1 {
            // Single group: head and tail merge into one region.
            copy_region(0..1, p_start..p_end_tail);
        } else {
            // Head: partial first group (omit if p_start == 0; that case folds into middle).
            if p_start > 0 {
                copy_region(0..1, p_start..n);
            }

            // Middle: full groups between any head and any tail.
            let middle_start: u64 = if p_start > 0 { 1 } else { 0 };
            let middle_end: u64 = if p_end_tail < n { g_count - 1 } else { g_count };
            if middle_end > middle_start {
                copy_region(middle_start..middle_end, 0..n);
            }

            // Tail: partial last group (omit if p_end_tail == n; folded into middle).
            if p_end_tail < n {
                copy_region((g_count - 1)..g_count, 0..p_end_tail);
            }
        }

        Ok(out)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.new_shape.as_slice()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Repeat", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Repeat<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        check_ndim::<NewD>(self.shape().len())?;
        let new_shape = NewD::from_slice(self.shape());

        Ok(Repeat {
            array: self.array.dimension_change()?,
            axis: self.axis,
            repeats: self.repeats,
            new_shape,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Repeat<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Repeat {
            array: self.array.element_type_change()?,
            axis: self.axis,
            repeats: self.repeats,
            new_shape: self.new_shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::array;

    use crate::codec::ReadContext;
    use crate::storage::Compact;
    use crate::{Array, ArrayStorage, IntoDimension, Ty, NDIM_MAX};

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
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_repeat_axis0() {
        assert_eq!(make(arange(12), &[3u64, 4]).repeat(2, 0).shape(), &[6, 4]);
    }

    #[test]
    fn shape_repeat_axis_last() {
        assert_eq!(make(arange(12), &[3u64, 4]).repeat(3, 1).shape(), &[3, 12]);
    }

    #[test]
    fn shape_repeat_3d_middle() {
        assert_eq!(
            make(arange(24), &[2u64, 3, 4]).repeat(5, 1).shape(),
            &[2, 15, 4]
        );
    }

    #[test]
    fn shape_repeats_zero() {
        assert_eq!(make(arange(12), &[3u64, 4]).repeat(0, 0).shape(), &[0, 4]);
    }

    #[test]
    fn shape_repeats_one_is_identity() {
        let a = make(arange(12), &[3u64, 4]);
        let r = super::Repeat::new_array(a.as_ref(), 1, 0)
            .unwrap()
            .into_storage();
        assert_eq!(r.shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make(arange(12), &[3u64, 4]);
        let err = super::Repeat::new_array(a.as_ref(), 2, 2).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn error_ndim_equals_ndim_max() {
        // Build an ndim == NDIM_MAX array of all-1s shape so it stays tiny.
        // NDIM_MAX == 8, so use a fixed-size array of length 8.
        let shape: [u64; 8] = [1; 8];
        let a = make(arange(1), &shape);
        assert_eq!(a.shape().len(), NDIM_MAX);
        let err = super::Repeat::new_array(a.as_ref(), 2, 0).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn error_repeats_overflow() {
        // shape[0] * repeats overflows u64
        let a = make(arange(2), &[2u64]);
        let err = super::Repeat::new_array(a.as_ref(), u64::MAX, 0).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    // -----------------------------------------------------------------------
    // Fast paths: identity (repeats == 1) and empty (repeats == 0)
    // -----------------------------------------------------------------------

    #[test]
    fn identity_full_read_returns_input() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .repeat(1, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_sub_read_correct() {
        let got = make(arange(12), &[3u64, 4])
            .repeat(1, 1)
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        // rows 1..3, cols 1..3 of [[0..3],[4..7],[8..11]] = [[5,6],[9,10]]
        assert_eq!(got, array![[5, 6], [9, 10]]);
    }

    #[test]
    fn zero_repeats_full_read_is_empty() {
        let got = make(arange(12), &[3u64, 4])
            .repeat(0, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
        assert_eq!(got.len(), 0);
    }

    #[test]
    fn empty_subrange_returns_ok() {
        // Even with repeats > 1, an empty output sub-range short-circuits.
        let got = make(arange(12), &[3u64, 4])
            .repeat(2, 0)
            .to_ndarray_sub(&[2..2, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads (aligned by definition)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d() {
        let got = make(arange(3), &[3u64]).repeat(2, 0).to_ndarray().unwrap();
        assert_eq!(got, array![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn full_read_2d_axis0() {
        let got = make(arange(8), &[2u64, 4])
            .repeat(2, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            array![[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]
        );
    }

    #[test]
    fn full_read_2d_axis1() {
        let got = make(arange(6), &[2u64, 3])
            .repeat(3, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            array![[0, 0, 0, 1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4, 5, 5, 5]]
        );
    }

    #[test]
    fn full_read_3d_middle_axis() {
        let got = make(arange(8), &[2u64, 2, 2])
            .repeat(2, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            array![
                [[0, 1], [0, 1], [2, 3], [2, 3]],
                [[4, 5], [4, 5], [6, 7], [6, 7]]
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Sub-region reads: head / middle / tail trimming
    // -----------------------------------------------------------------------

    // Reference layout for these tests:
    //   input = [10, 20, 30, 40], repeats=3, axis=0
    //   output = [10,10,10, 20,20,20, 30,30,30, 40,40,40] (length 12)
    fn make_1d_repeat3() -> Array<Compact<Ty<i32>, crate::Dim<1>>> {
        make(vec![10, 20, 30, 40], &[4u64])
    }

    #[test]
    fn sub_read_aligned_middle_only() {
        // s=3, e=9 -> [20,20,20, 30,30,30]; both ends aligned to 3.
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[3..9], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![20, 20, 20, 30, 30, 30]);
    }

    #[test]
    fn sub_read_head_only_single_group() {
        // s=1, e=3 -> within group 0 (10,10,10), positions 1..3 -> [10, 10].
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[1..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10, 10]);
    }

    #[test]
    fn sub_read_tail_only_single_group() {
        // s=9, e=11 -> within group 3 (40,40,40), positions 0..2 -> [40, 40].
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[9..11], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![40, 40]);
    }

    #[test]
    fn sub_read_single_group_both_ends_partial() {
        // s=4, e=5 -> within group 1, single element [20].
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[4..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![20]);
    }

    #[test]
    fn sub_read_head_plus_middle() {
        // s=1, e=6 -> [10,10, 20,20,20]: head (g=0, p=1..3) + middle (g=1, p=0..3).
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[1..6], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10, 10, 20, 20, 20]);
    }

    #[test]
    fn sub_read_middle_plus_tail() {
        // s=6, e=11 -> [30,30,30, 40,40]: middle (g=2 full) + tail (g=3, p=0..2).
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[6..11], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![30, 30, 30, 40, 40]);
    }

    #[test]
    fn sub_read_head_middle_tail() {
        // s=1, e=11 -> [10,10, 20,20,20, 30,30,30, 40,40]: head + middle + tail.
        let got = make_1d_repeat3()
            .repeat(3, 0)
            .to_ndarray_sub(&[1..11], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10, 10, 20, 20, 20, 30, 30, 30, 40, 40]);
    }

    #[test]
    fn sub_read_2d_axis1_head_only() {
        // [[1,2],[3,4]] repeat 4 axis 1 -> [[1,1,1,1, 2,2,2,2],[3,3,3,3, 4,4,4,4]]
        // sub-region cols 2..3 -> still within group 0 (the "1" or "3" group), single column.
        let got = make(vec![1, 2, 3, 4], &[2u64, 2])
            .repeat(4, 1)
            .to_ndarray_sub(&[0..2, 2..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[1], [3]]);
    }

    // -----------------------------------------------------------------------
    // Composition with other ops
    // -----------------------------------------------------------------------

    #[test]
    fn compose_repeat_then_compact() {
        let got = make(arange(6), &[2u64, 3])
            .repeat(2, 0)
            .compact()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]]);
    }

    #[test]
    fn compose_repeat_then_slice() {
        let got = make(arange(6), &[2u64, 3])
            .repeat(2, 0)
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        // Output before slice: [[0,1,2],[0,1,2],[3,4,5],[3,4,5]]; rows 1..3 = [[0,1,2],[3,4,5]]
        assert_eq!(got, array![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn compose_repeat_then_cast() {
        let got = make(arange(3), &[3u64])
            .repeat(2, 0)
            .cast::<f32>()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn compose_permute_then_repeat() {
        // [[0,1,2],[3,4,5]] (shape [2,3]) permute axes -> shape [3,2] with values
        //   [[0,3],[1,4],[2,5]]
        // then repeat axis 1 by 2 -> [[0,0,3,3],[1,1,4,4],[2,2,5,5]] (shape [3,4])
        let got = make(arange(6), &[2u64, 3])
            .permute_axes(&[1, 0])
            .repeat(2, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 0, 3, 3], [1, 1, 4, 4], [2, 2, 5, 5]]);
    }

    // -----------------------------------------------------------------------
    // Dimension and element type preservation
    // -----------------------------------------------------------------------

    #[test]
    fn dim_change_into_dim_dyn() {
        // Static Dim<2> -> Repeat<...> -> into_dim_dyn -> DimDyn-typed.
        let a = make(arange(12), &[3u64, 4]); // Compact<Ty<i32>, Dim<2>>
        let rep = a.repeat(2, 0); // Array<Repeat<Compact<Ty<i32>, Dim<2>>>>
        let dyn_arr = rep.into_dim_dyn(); // shape (6, 4) under DimDyn
        assert_eq!(dyn_arr.shape(), &[6, 4]);
    }

    #[test]
    fn element_type_change_into_type_dyn() {
        let a = make(arange(6), &[2u64, 3]);
        let rep = a.repeat(3, 1);
        let dyn_et = rep.into_type_dyn();
        // dtype preserved at runtime.
        assert_eq!(dyn_et.dtype(), &<i32 as crate::dtype::Dtyped>::DTYPE);
    }

    // -----------------------------------------------------------------------
    // Proptest: random shape + axis + repeats vs hand-rolled reference
    // -----------------------------------------------------------------------

    /// Reference implementation: produces the expected ndarray result of repeating
    /// `nd` `repeats` times along `axis`. ndarray has no built-in `repeat`, so we
    /// build the output by iterating per-slice along `axis`.
    fn repeat_reference<T: Clone + Default>(
        nd: &ndarray::ArrayD<T>,
        repeats: u64,
        axis: usize,
    ) -> ndarray::ArrayD<T> {
        let mut out_shape = nd.shape().to_vec();
        out_shape[axis] *= repeats as usize;
        let mut out = ndarray::ArrayD::<T>::from_elem(out_shape.as_slice(), T::default());
        if repeats == 0 || nd.is_empty() {
            return out;
        }
        for (i, src_slice) in nd.axis_iter(ndarray::Axis(axis)).enumerate() {
            for r in 0..repeats as usize {
                let out_idx = i * repeats as usize + r;
                let mut out_subview = out.index_axis_mut(ndarray::Axis(axis), out_idx);
                out_subview.assign(&src_slice);
            }
        }
        out
    }

    #[allow(clippy::type_complexity)]
    fn repeat_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            Array<Compact<Ty<i32>, crate::DimDyn>>,
            usize,
            u64,
        ),
    > {
        use proptest::prelude::*;

        // Cap ndim to NDIM_MAX - 1 because Repeat needs room for one extra
        // synthetic axis at read time.
        let shape = crate::util::shape_strategy()
            .prop_filter("ndim < NDIM_MAX", |s| s.len() < NDIM_MAX)
            .prop_filter("non-empty ndim", |s| !s.is_empty());

        shape.prop_flat_map(|shape| {
            let ndim = shape.len();
            let array_strat = crate::util::carray_strategy_from_shape::<i32>(
                Just(shape.clone()),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            );
            let axis = 0..ndim;
            let repeats = 0u64..=4u64;
            (array_strat, axis, repeats)
                .prop_map(|((nd, za), axis, repeats)| (nd, za, axis, repeats))
        })
    }

    proptest::proptest! {
        #[test]
        fn proptest_repeat_generic(
            (nd, za, axis, repeats) in repeat_strategy()
        ) {
            let expected = repeat_reference(&nd, repeats, axis);
            let actual = za.repeat(repeats, axis);
            crate::util::assert_array_matches(&actual, &expected);
        }
    }
}
