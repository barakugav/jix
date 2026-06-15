use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, ensure, Error, ErrorKind, Result};
use crate::storage::{ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::{default_strides, dim_arr, nd_copy, DimArray};
use crate::{Array, ArrayStorage, DimDyn, Dimension, NDIM_MAX};

/// Replicates the array along one axis by a scalar count, returned by
/// [`Array::tile`](crate::Array::tile).
///
/// Output shape equals the input except `shape[axis]` becomes `shape[axis] * reps`. The
/// rolled axis is *not* extended: `axis` must satisfy `axis < ndim`. Element `i` along the
/// output axis comes from input element `i mod L`, where `L = input.shape()[axis]`.
///
/// This differs from [`Repeat`](crate::ops::Repeat): `tile` repeats the whole sequence
/// `(a, b, c, ...) -> (a, b, c, a, b, c, ...)` whereas `repeat` repeats each element in
/// place `(a, b, c, ...) -> (a, a, b, b, c, c, ...)`. When the input axis already has
/// length 1, [`Broadcast`](crate::ops::Broadcast) is a zero-cost alternative.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // 1-D: tile the whole sequence twice along axis 0.
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// assert_eq!(a.as_ref().tile(2, 0).to_ndarray()?, array![1, 2, 3, 1, 2, 3]);
///
/// // 2-D axis 0: stack the matrix on top of itself.
/// let m = Array::compact_ndarray(&array![[1i32, 2], [3, 4]])?;
/// assert_eq!(
///     m.as_ref().tile(2, 0).to_ndarray()?,
///     array![[1, 2], [3, 4], [1, 2], [3, 4]],
/// );
///
/// // 2-D axis 1: each row is repeated horizontally.
/// assert_eq!(
///     m.tile(2, 1).to_ndarray()?,
///     array![[1, 2, 1, 2], [3, 4, 3, 4]],
/// );
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Tile<S: ArrayStorage> {
    array: S,
    axis: usize,
    reps: u64,
    new_shape: S::Dimension,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> Tile<S> {
    /// Constructs a [`Tile`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, reps: u64, axis: usize) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();

        ensure!(
            ndim < NDIM_MAX,
            InvalidShapeOperation,
            "tile requires ndim < NDIM_MAX ({NDIM_MAX}); got {ndim} \
             (one extra axis is used internally during reads)"
        );
        ensure!(
            axis < ndim,
            InvalidShapeOperation,
            "tile axis {axis} is out of bounds for array with ndim {ndim}"
        );

        let new_len = input_shape[axis].checked_mul(reps).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidShapeOperation,
                format!(
                    "tile overflow: shape[{axis}] ({}) * reps ({}) exceeds u64",
                    input_shape[axis], reps,
                ),
            )
        })?;

        let new_shape =
            S::Dimension::from_fn(ndim, |d| if d == axis { new_len } else { input_shape[d] })
                .unwrap();

        let mut b_layout = array.spec().blocks_layout.clone();
        let reps_u32 = reps.min(u32::MAX as u64) as u32;
        b_layout.block_shape_hint[axis] = b_layout.block_shape_hint[axis]
            .saturating_mul(reps_u32)
            .max(1);
        b_layout.preferred_read_shape[axis] = b_layout.preferred_read_shape[axis]
            .saturating_mul(reps_u32)
            .max(1);
        b_layout.block_shape_tag[axis] = BlockShapeTag::Any;

        Ok(Self {
            array,
            axis,
            reps,
            new_shape,
            blocks_layout: b_layout,
        })
    }

    /// Constructs an array with [`Tile`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, reps: u64, axis: usize) -> Result<Array<Self>> {
        Self::new(array.into_storage(), reps, axis).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Tile<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, self.dtype(), buf)?;

        if index.iter().any(|r| r.start >= r.end) {
            return Ok(());
        }

        let ndim = index.len();
        let k = self.axis;
        let dtype = self.dtype();
        let itemsize = dtype.itemsize() as usize;
        let l = self.array.shape()[k];
        let s = index[k].start;
        let e = index[k].end;
        let total = e - s;
        // L > 0 here because output axis k has length L * reps > 0 (any zero-length range
        // was already short-circuited above).
        let s_in = s % l;

        // Case A - single read, no wrap: the requested output range maps to one contiguous
        // input range along axis k. Read directly into `buf` (no tmp_buf, no nd_copy).
        if s_in + total <= l {
            let inner_index = dim_arr(ndim, |d| {
                if d == k {
                    s_in..s_in + total
                } else {
                    index[d].clone()
                }
            });
            return self.array.read_data(&inner_index, buf, context);
        }

        let out_shape = dim_arr(ndim, |d| index[d].end - index[d].start);
        let dst_strides = default_strides(&out_shape, itemsize as u64);

        // Case B - two reads, single wrap (total <= L): split the request into two
        // contiguous input ranges along axis k and read each into a separate tmp_buf, then
        // place them at the right axis-k offset in `buf`.
        if total <= l {
            let len1 = l - s_in;
            let len2 = total - len1;

            let mut read_region = |inner_index: &[Range<u64>],
                                   region_shape: &[u64],
                                   dst_axis_k_offset: u64|
             -> Result<()> {
                let region_size = region_shape.iter().product::<u64>() as usize * itemsize;
                let mut tmp = context.tmp_buf(region_size, dtype.alignment());
                let tmp = tmp.as_mut_slice();
                self.array.read_data(inner_index, tmp, context)?;
                let src_strides = default_strides(region_shape, itemsize as u64);
                let dst_byte_offset = (dst_axis_k_offset * dst_strides[k]) as usize;
                unsafe {
                    nd_copy(
                        tmp.as_ptr(),
                        buf.as_mut_ptr().add(dst_byte_offset),
                        S::Dimension::from_slice(region_shape).unwrap(),
                        &src_strides,
                        &dst_strides,
                        itemsize,
                    )
                };
                Ok(())
            };

            let inner_r1 = dim_arr(ndim, |d| if d == k { s_in..l } else { index[d].clone() });
            let r1_shape = dim_arr(ndim, |d| if d == k { len1 } else { out_shape[d] });
            read_region(&inner_r1, &r1_shape, 0)?;

            let inner_r2 = dim_arr(ndim, |d| if d == k { 0..len2 } else { index[d].clone() });
            let r2_shape = dim_arr(ndim, |d| if d == k { len2 } else { out_shape[d] });
            read_region(&inner_r2, &r2_shape, len1)?;

            return Ok(());
        }

        // Case C - span > L: read the full period [0, L) along axis k into tmp_buf once,
        // then memcpy chunks into `buf`. Layout along axis k:
        //   head:   tmp[s_in..L]      -> buf[0..head_len)
        //   middle: tmp[0..L] x F      -> buf[head_len..head_len + F*L)   (F = num_full)
        //   tail:   tmp[0..tail_len]   -> buf[head_len + F*L..total)
        let inner_full = dim_arr(ndim, |d| if d == k { 0..l } else { index[d].clone() });
        let period_shape = dim_arr(ndim, |d| if d == k { l } else { out_shape[d] });
        let period_size = period_shape.iter().product::<u64>() as usize * itemsize;
        let mut tmp = context.tmp_buf(period_size, dtype.alignment());
        let tmp = tmp.as_mut_slice();
        self.array.read_data(&inner_full, tmp, context)?;

        let src_strides = default_strides(&period_shape, itemsize as u64);

        let head_len = l - s_in; // 0 < head_len <= L
        let remaining = total - head_len; // > 0 since total > L
        let num_full = remaining / l;
        let tail_len = remaining % l;

        // Head: tmp[s_in..L] -> buf[0..head_len)
        {
            let copy_shape =
                S::Dimension::from_fn(ndim, |d| if d == k { head_len } else { out_shape[d] })
                    .unwrap();
            let src_off = (s_in * src_strides[k]) as usize;
            unsafe {
                nd_copy(
                    tmp.as_ptr().add(src_off),
                    buf.as_mut_ptr(),
                    copy_shape,
                    &src_strides,
                    &dst_strides,
                    itemsize,
                )
            };
        }

        // Middle: replicate tmp[0..L] num_full times using the stride-0 trick on a
        // synthesized tile axis inserted at position k.
        if num_full > 0 {
            let mut copy_shape = DimArray::new();
            let mut src_strides_split = DimArray::new();
            let mut dst_strides_split = DimArray::new();
            for d in 0..ndim {
                if d == k {
                    copy_shape.push(num_full);
                    copy_shape.push(l);
                    src_strides_split.push(0);
                    src_strides_split.push(src_strides[d]);
                    dst_strides_split.push(l * dst_strides[d]);
                    dst_strides_split.push(dst_strides[d]);
                } else {
                    copy_shape.push(out_shape[d]);
                    src_strides_split.push(src_strides[d]);
                    dst_strides_split.push(dst_strides[d]);
                }
            }
            let dst_off = (head_len * dst_strides[k]) as usize;
            unsafe {
                nd_copy(
                    tmp.as_ptr(),
                    buf.as_mut_ptr().add(dst_off),
                    DimDyn::from_slice(&copy_shape).unwrap(),
                    &src_strides_split,
                    &dst_strides_split,
                    itemsize,
                )
            };
        }

        // Tail: tmp[0..tail_len] -> buf[head_len + num_full * L..total)
        if tail_len > 0 {
            let copy_shape =
                S::Dimension::from_fn(ndim, |d| if d == k { tail_len } else { out_shape[d] })
                    .unwrap();
            let dst_off = ((head_len + num_full * l) * dst_strides[k]) as usize;
            unsafe {
                nd_copy(
                    tmp.as_ptr(),
                    buf.as_mut_ptr().add(dst_off),
                    copy_shape,
                    &src_strides,
                    &dst_strides,
                    itemsize,
                )
            };
        }

        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.new_shape.as_slice()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }

    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.spec()
        }
    }

    type DimensionChange<NewD: crate::Dimension> = Tile<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let new_shape = NewD::from_slice(self.new_shape.as_slice())?;
        Ok(Tile {
            array: self.array.dimension_change()?,
            axis: self.axis,
            reps: self.reps,
            new_shape,
            blocks_layout: self.blocks_layout,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Tile<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Tile {
            array: self.array.element_type_change()?,
            axis: self.axis,
            reps: self.reps,
            new_shape: self.new_shape,
            blocks_layout: self.blocks_layout,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::codec::ReadContext;
    use crate::storage::Compact;
    use crate::{Array, ArrayStorage, IntoDimension, Ty, NDIM_MAX};
    use ndarray::array;

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
    fn shape_tile_axis0() {
        // [3, 4] tile 2 along axis 0 -> [6, 4]
        assert_eq!(make(arange(12), &[3u64, 4]).tile(2, 0).shape(), &[6, 4]);
    }

    #[test]
    fn shape_tile_axis_last() {
        // [3, 4] tile 3 along axis 1 -> [3, 12]
        assert_eq!(make(arange(12), &[3u64, 4]).tile(3, 1).shape(), &[3, 12]);
    }

    #[test]
    fn shape_tile_3d_middle() {
        // [2, 3, 4] tile 5 along axis 1 -> [2, 15, 4]
        assert_eq!(
            make(arange(24), &[2u64, 3, 4]).tile(5, 1).shape(),
            &[2, 15, 4]
        );
    }

    #[test]
    fn shape_reps_zero() {
        // [3, 4] tile 0 along axis 0 -> [0, 4]
        assert_eq!(make(arange(12), &[3u64, 4]).tile(0, 0).shape(), &[0, 4]);
    }

    #[test]
    fn shape_reps_one_is_identity() {
        let a = make(arange(12), &[3u64, 4]);
        let r = super::Tile::new_array(a.as_ref(), 1, 0)
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
        let err = super::Tile::new_array(a.as_ref(), 2, 2).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn error_ndim_equals_ndim_max() {
        // ndim == NDIM_MAX leaves no room for the synthetic tile axis used during reads.
        let shape: [u64; 8] = [1; 8];
        let a = make(arange(1), &shape);
        assert_eq!(a.shape().len(), NDIM_MAX);
        let err = super::Tile::new_array(a.as_ref(), 2, 0).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn error_reps_overflow() {
        let a = make(arange(2), &[2u64]);
        let err = super::Tile::new_array(a.as_ref(), u64::MAX, 0).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    // -----------------------------------------------------------------------
    // Fast paths: identity (reps == 1) and empty (reps == 0)
    // -----------------------------------------------------------------------

    #[test]
    fn identity_full_read_returns_input() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .tile(1, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_sub_read_correct() {
        let got = make(arange(12), &[3u64, 4])
            .tile(1, 1)
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[5, 6], [9, 10]]);
    }

    #[test]
    fn zero_reps_full_read_is_empty() {
        let got = make(arange(12), &[3u64, 4])
            .tile(0, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
        assert_eq!(got.len(), 0);
    }

    #[test]
    fn empty_subrange_returns_ok() {
        let got = make(arange(12), &[3u64, 4])
            .tile(2, 0)
            .to_ndarray_sub(&[2..2, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads (aligned by definition)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d() {
        // [0, 1, 2] tile 2 axis 0 -> [0, 1, 2, 0, 1, 2]
        let got = make(arange(3), &[3u64]).tile(2, 0).to_ndarray().unwrap();
        assert_eq!(got, array![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn full_read_2d_axis0() {
        // [[0,1,2,3],[4,5,6,7]] tile 2 axis 0 -> matrix stacked twice
        let got = make(arange(8), &[2u64, 4]).tile(2, 0).to_ndarray().unwrap();
        assert_eq!(
            got,
            array![[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
        );
    }

    #[test]
    fn full_read_2d_axis1() {
        // [[0,1,2],[3,4,5]] tile 3 axis 1 -> each row repeated three times horizontally
        let got = make(arange(6), &[2u64, 3]).tile(3, 1).to_ndarray().unwrap();
        assert_eq!(
            got,
            array![[0, 1, 2, 0, 1, 2, 0, 1, 2], [3, 4, 5, 3, 4, 5, 3, 4, 5]]
        );
    }

    #[test]
    fn full_read_3d_middle_axis() {
        // [2, 2, 2] tile 2 axis 1 -> [2, 4, 2]
        let got = make(arange(8), &[2u64, 2, 2])
            .tile(2, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            array![
                [[0, 1], [2, 3], [0, 1], [2, 3]],
                [[4, 5], [6, 7], [4, 5], [6, 7]]
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------
    //
    // Reference layout for these tests:
    //   input  = [10, 20, 30, 40] (L = 4), reps = 3, axis = 0
    //   output = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]  (length 12)
    //   s_in   = s mod L

    fn make_1d_tile3() -> Array<Compact<Ty<i32>, crate::Dim<1>>> {
        make(vec![10, 20, 30, 40], &[4u64])
    }

    #[test]
    fn sub_read_case_a_within_period_no_wrap() {
        // Case A: s=1, e=4 -> s_in=1, total=3, s_in+total=4 <= L. Single inner read.
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[1..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![20, 30, 40]);
    }

    #[test]
    fn sub_read_case_a_period_aligned() {
        // Case A: s=4, e=8 -> s_in=0, total=4 == L. Single read at offset 0.
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[4..8], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10, 20, 30, 40]);
    }

    #[test]
    fn sub_read_case_b_wrap_once_within_period() {
        // Case B: s=2, e=5 -> s_in=2, total=3, s_in+total=5 > L=4, total<=L. Two reads.
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![30, 40, 10]);
    }

    #[test]
    fn sub_read_case_b_wrap_spans_period_boundary() {
        // Case B: s=3, e=6 -> s_in=3, total=3, s_in+total=6 > L. Two reads.
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[3..6], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![40, 10, 20]);
    }

    #[test]
    fn sub_read_case_c_spans_multiple_periods_aligned() {
        // Case C: s=0, e=8 -> total=8 > L. s_in=0, head_len=L, num_full=1, tail=0.
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[0..8], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10, 20, 30, 40, 10, 20, 30, 40]);
    }

    #[test]
    fn sub_read_case_c_unaligned_with_tail() {
        // Case C: s=1, e=10 -> s_in=1, total=9 > L. head_len=3, num_full=1, tail=2.
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[1..10], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![20, 30, 40, 10, 20, 30, 40, 10, 20]);
    }

    #[test]
    fn sub_read_case_c_full_output() {
        // Case C: full read of the tiled output (s=0, e=12 = reps*L).
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[0..12], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]);
    }

    #[test]
    fn sub_read_single_element_at_wrap_point() {
        // s=4 -> s_in=0, total=1. Single read of input[0..1].
        let got = make_1d_tile3()
            .tile(3, 0)
            .to_ndarray_sub(&[4..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![10]);
    }

    #[test]
    fn sub_read_2d_axis0_case_c() {
        // [[0,1,2],[3,4,5]] tile 3 axis 0 -> 6 rows; sub-read rows [1..5)
        // = [[3,4,5],[0,1,2],[3,4,5],[0,1,2]]
        let got = make(arange(6), &[2u64, 3])
            .tile(3, 0)
            .to_ndarray_sub(&[1..5, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn sub_read_2d_axis1_wrap() {
        // [[0,1,2,3],[4,5,6,7]] tile 2 axis 1 -> 8 cols, output:
        //   [[0,1,2,3,0,1,2,3],[4,5,6,7,4,5,6,7]]
        // sub-read cols [3..6) -> [[3,0,1],[7,4,5]]  (case B: wraps once)
        let got = make(arange(8), &[2u64, 4])
            .tile(2, 1)
            .to_ndarray_sub(&[0..2, 3..6], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[3, 0, 1], [7, 4, 5]]);
    }

    #[test]
    fn sub_read_2d_axis1_case_c() {
        // [[0,1,2,3],[4,5,6,7]] tile 3 axis 1 -> cols [1..11) hits all three cases.
        // Full: [[0,1,2,3,0,1,2,3,0,1,2,3],[4,5,6,7,4,5,6,7,4,5,6,7]]
        // cols 1..11: [[1,2,3,0,1,2,3,0,1,2],[5,6,7,4,5,6,7,4,5,6]]
        let got = make(arange(8), &[2u64, 4])
            .tile(3, 1)
            .to_ndarray_sub(&[0..2, 1..11], &ReadContext::default())
            .unwrap();
        assert_eq!(
            got,
            array![
                [1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
                [5, 6, 7, 4, 5, 6, 7, 4, 5, 6]
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Composition with other ops
    // -----------------------------------------------------------------------

    #[test]
    fn compose_tile_then_compact() {
        let got = make(arange(6), &[2u64, 3])
            .tile(2, 0)
            .compact()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn compose_tile_then_slice() {
        // tile axis 0 by 2 then slice rows 1..3 of (4, 3).
        let got = make(arange(6), &[2u64, 3])
            .tile(2, 0)
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        // Tiled: [[0,1,2],[3,4,5],[0,1,2],[3,4,5]]; rows 1..3 = [[3,4,5],[0,1,2]]
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_tile_then_cast() {
        let got = make(arange(3), &[3u64])
            .tile(2, 0)
            .cast::<f32>()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn compose_permute_then_tile() {
        // [[0,1,2],[3,4,5]] (shape [2,3]) permute axes -> shape [3,2] values [[0,3],[1,4],[2,5]]
        // then tile axis 1 by 2 -> [[0,3,0,3],[1,4,1,4],[2,5,2,5]]
        let got = make(arange(6), &[2u64, 3])
            .permute_axes(&[1, 0])
            .tile(2, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 3, 0, 3], [1, 4, 1, 4], [2, 5, 2, 5]]);
    }

    #[test]
    fn compose_tile_then_flip_same_axis() {
        // [0,1,2] tile 2 axis 0 -> [0,1,2,0,1,2]; flip axis 0 -> [2,1,0,2,1,0]
        let got = make(arange(3), &[3u64])
            .tile(2, 0)
            .flip(0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![2, 1, 0, 2, 1, 0]);
    }

    // -----------------------------------------------------------------------
    // Dimension and element type preservation
    // -----------------------------------------------------------------------

    #[test]
    fn dim_change_into_dim_dyn() {
        let a = make(arange(12), &[3u64, 4]); // Compact<Ty<i32>, Dim<2>>
        let t = a.tile(2, 0); // Array<Tile<Compact<Ty<i32>, Dim<2>>>>
        let dyn_arr = t.into_dim_dyn();
        assert_eq!(dyn_arr.shape(), &[6, 4]);
    }

    #[test]
    fn element_type_change_into_type_dyn() {
        let a = make(arange(6), &[2u64, 3]);
        let t = a.tile(3, 1);
        let dyn_et = t.into_type_dyn();
        assert_eq!(dyn_et.dtype(), &<i32 as crate::dtype::Dtyped>::DTYPE);
    }

    // -----------------------------------------------------------------------
    // Proptest: random shape + axis + reps vs hand-rolled reference
    // -----------------------------------------------------------------------

    fn tile_reference<T: Clone + Default>(
        nd: &ndarray::ArrayD<T>,
        reps: u64,
        axis: usize,
    ) -> ndarray::ArrayD<T> {
        let mut out_shape: Vec<usize> = nd.shape().to_vec();
        out_shape[axis] *= reps as usize;
        let mut out = ndarray::ArrayD::<T>::from_elem(out_shape.as_slice(), T::default());
        let l = nd.shape()[axis];
        if reps == 0 || l == 0 {
            return out;
        }
        for r in 0..reps as usize {
            for i in 0..l {
                let src_slice = nd.index_axis(ndarray::Axis(axis), i);
                let out_idx = r * l + i;
                let mut dst = out.index_axis_mut(ndarray::Axis(axis), out_idx);
                dst.assign(&src_slice);
            }
        }
        out
    }

    fn tile_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            Array<Compact<Ty<i32>, crate::DimDyn>>,
            usize,
            u64,
        ),
    > {
        use proptest::prelude::*;

        // Cap ndim to NDIM_MAX - 1: Tile needs room for one extra synthetic axis at read time.
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
            let reps = 0u64..=4u64;
            (array_strat, axis, reps).prop_map(|((nd, za), axis, reps)| (nd, za, axis, reps))
        })
    }

    proptest::proptest! {
        #[test]
        fn proptest_tile_generic(
            (nd, za, axis, reps) in tile_strategy()
        ) {
            let expected = tile_reference(&nd, reps, axis);
            let actual = za.tile(reps, axis);
            crate::util::assert_array_matches(&actual, &expected);
        }
    }
}
