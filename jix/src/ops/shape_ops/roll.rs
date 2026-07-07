use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArraySpec, ArrayStorageInfo, OutBuf};
use crate::util::{default_strides, nd_copy};
use crate::{Array, ArrayStorage, Dimension};

/// Rolls elements along an axis, wrapping around at the boundary, returned by
/// [`Array::roll`](crate::Array::roll).
///
/// `output[..., i, ...] = input[..., (i - shift) mod L, ...]` on the rolled axis, where
/// `L = shape[axis]`. A positive `shift` moves elements toward larger indices (elements
/// that fall off the end re-enter at the beginning); a negative `shift` moves them toward
/// smaller indices. `shift` is reduced modulo `L`, so any signed integer is accepted.
///
/// See also [`Flip`](crate::ops::Flip), which reverses element order along an axis
/// without wrapping.
///
/// Output shape and dtype equal the input. The result is a lazy view; no computation
/// occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // 1-D positive shift wraps tail to head.
/// let a = Array::compact_ndarray(&array![0i32, 1, 2, 3, 4])?;
/// assert_eq!(a.as_ref().roll(2, 0).to_ndarray()?, array![3, 4, 0, 1, 2]);
/// // Negative shift wraps head to tail.
/// assert_eq!(a.as_ref().roll(-1, 0).to_ndarray()?, array![1, 2, 3, 4, 0]);
/// // Shift mod L; shift == L is identity.
/// assert_eq!(a.roll(5, 0).to_ndarray()?, array![0, 1, 2, 3, 4]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Roll<S: ArrayStorage> {
    array: S,
    axis: usize,
    /// `shift` normalized to `[0, shape[axis])`. Zero means the op is a pass-through.
    shift: u64,
}

impl<S: ArrayStorage> Roll<S> {
    /// Constructs a [`Roll`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, shift: i64, axis: usize) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();
        ensure!(
            axis < ndim,
            InvalidShapeOperation,
            "roll axis {axis} is out of bounds for array with ndim {ndim}"
        );

        let len = input_shape[axis];
        let shift = if len == 0 {
            0
        } else {
            shift.rem_euclid(len as i64) as u64
        };

        Ok(Self { array, axis, shift })
    }

    /// Constructs an array with [`Roll`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, shift: i64, axis: usize) -> Result<Array<Self>> {
        Self::new(array.into_storage(), shift, axis).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Roll<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        check_get_range(self.shape(), index)?;

        let k = self.axis;
        let shift = self.shift;

        let ndim = index.len();
        let dtype = self.dtype();
        let itemsize = dtype.itemsize() as usize;
        let l = self.shape()[k];
        let s = index[k].start;
        let e = index[k].end;

        // Non-wrap: the rolled output sub-range on axis k maps to a single contiguous input
        // range. The shape of that read is exactly `out_shape`, so we read straight into buf.
        if s >= shift || e <= shift {
            let j_start = if s >= shift { s - shift } else { s + l - shift };
            let inner_index = S::Dimension::vec(ndim, |d| {
                if d == k {
                    j_start..(j_start + (e - s))
                } else {
                    index[d].clone()
                }
            });
            return self.array.read_data(inner_index.as_ref(), buf, context);
        }

        // Wrap: split the output along axis k into two regions and read each separately.
        //   Region 1 (output axis-k [0, len1)):   input axis-k [s + L - shift, L), length len1.
        //   Region 2 (output axis-k [len1, end)): input axis-k [0, e - shift),     length len2.
        let len1 = shift - s;
        let len2 = e - shift;
        let buf = buf.get_mut(index, dtype);
        check_get_buffer_size(index, dtype, buf)?;
        let out_shape = S::Dimension::vec(ndim, |d| index[d].end - index[d].start);
        let dst_strides = default_strides(out_shape.as_ref(), itemsize as u64);

        let mut read_region = |inner_index: &[Range<u64>],
                               region_shape: &[u64],
                               dst_axis_k_offset: u64|
         -> Result<()> {
            let region_size = region_shape.iter().product::<u64>() as usize * itemsize;
            let mut tmp = context.tmp_buf(region_size, dtype.alignment());
            let tmp = tmp.as_mut_slice();
            self.array
                .read_data(inner_index, &mut OutBuf::new(tmp), context)?;

            let src_strides = default_strides(region_shape, itemsize as u64);
            let dst_byte_offset = (dst_axis_k_offset * dst_strides[k]) as usize;
            unsafe {
                nd_copy(
                    tmp.as_ptr(),
                    buf.as_mut_ptr().add(dst_byte_offset),
                    S::Dimension::from_slice(region_shape),
                    &src_strides,
                    &dst_strides,
                    itemsize,
                )
            };
            Ok(())
        };

        let inner_index_r1 = S::Dimension::vec(ndim, |d| {
            if d == k {
                (s + l - shift)..l
            } else {
                index[d].clone()
            }
        });
        let r1_shape = S::Dimension::vec(ndim, |d| if d == k { len1 } else { out_shape[d] });
        read_region(inner_index_r1.as_ref(), r1_shape.as_ref(), 0)?;

        let inner_index_r2 =
            S::Dimension::vec(ndim, |d| if d == k { 0..len2 } else { index[d].clone() });
        let r2_shape = S::Dimension::vec(ndim, |d| if d == k { len2 } else { out_shape[d] });
        read_region(inner_index_r2.as_ref(), r2_shape.as_ref(), len1)?;

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
        ArrayStorageInfo::new_deps("Roll", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Roll<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Roll {
            array: self.array.dimension_change()?,
            axis: self.axis,
            shift: self.shift,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Roll<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Roll {
            array: self.array.element_type_change()?,
            axis: self.axis,
            shift: self.shift,
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
    // Shape metadata (roll preserves shape)
    // -----------------------------------------------------------------------

    #[test]
    fn shape_preserved_positive_shift() {
        assert_eq!(make(arange(12), &[3u64, 4]).roll(1, 0).shape(), &[3, 4]);
    }

    #[test]
    fn shape_preserved_negative_shift() {
        assert_eq!(make(arange(12), &[3u64, 4]).roll(-2, 1).shape(), &[3, 4]);
    }

    #[test]
    fn shape_preserved_large_shift() {
        // shift larger than the axis length is reduced mod L.
        assert_eq!(make(arange(12), &[3u64, 4]).roll(100, 0).shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make(arange(12), &[3u64, 4]);
        let err = super::Roll::new_array(a.as_ref(), 1, 2).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidShapeOperation);
    }

    // -----------------------------------------------------------------------
    // Fast paths: identity (shift mod L == 0) and empty
    // -----------------------------------------------------------------------

    #[test]
    fn identity_zero_shift_full_read() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .roll(0, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_shift_equals_axis_len() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .roll(3, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_shift_negative_multiple_of_len() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), &[3u64, 4])
            .roll(-6, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_size_one_axis() {
        // Rolling a size-1 axis is a no-op for any shift.
        let nd = ndarray::Array::from_shape_vec((1, 4), arange(4)).unwrap();
        let got = make(arange(4), &[1u64, 4]).roll(7, 0).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_empty_array() {
        // Rolling an axis of size 0 is a no-op (no data to move).
        let got = make(vec![], &[0u64, 4]).roll(3, 0).to_ndarray().unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_positive_shift() {
        // [0,1,2,3,4] roll +2 axis 0 -> [3, 4, 0, 1, 2]
        let got = make(arange(5), &[5u64]).roll(2, 0).to_ndarray().unwrap();
        assert_eq!(got, array![3, 4, 0, 1, 2]);
    }

    #[test]
    fn full_read_1d_negative_shift() {
        // [0,1,2,3,4] roll -1 axis 0 -> [1, 2, 3, 4, 0]
        let got = make(arange(5), &[5u64]).roll(-1, 0).to_ndarray().unwrap();
        assert_eq!(got, array![1, 2, 3, 4, 0]);
    }

    #[test]
    fn full_read_2d_axis0() {
        // [[0,1,2,3],[4,5,6,7]] roll +1 axis 0 -> [[4,5,6,7],[0,1,2,3]]
        let got = make(arange(8), &[2u64, 4]).roll(1, 0).to_ndarray().unwrap();
        assert_eq!(got, array![[4, 5, 6, 7], [0, 1, 2, 3]]);
    }

    #[test]
    fn full_read_2d_axis1() {
        // [[0,1,2],[3,4,5]] roll +1 axis 1 -> [[2,0,1],[5,3,4]]
        let got = make(arange(6), &[2u64, 3]).roll(1, 1).to_ndarray().unwrap();
        assert_eq!(got, array![[2, 0, 1], [5, 3, 4]]);
    }

    #[test]
    fn full_read_3d_middle_axis() {
        // Compare against a hand-rolled reference along axis 1 by +1.
        let arr = ndarray::Array::from_shape_vec((2, 3, 2), arange(12)).unwrap();
        let mut expected = arr.clone();
        for i in 0..2 {
            for j in 0..3 {
                for k_dim in 0..2 {
                    let src_j = (j + 3 - 1) % 3;
                    expected[(i, j, k_dim)] = arr[(i, src_j, k_dim)];
                }
            }
        }
        let got = make(arange(12), &[2u64, 3, 2])
            .roll(1, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Sub-region reads: non-wrap and wrap variants
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_1d_nonwrap_after_shift() {
        // [0,1,2,3,4] roll +2 -> [3, 4, 0, 1, 2]. Output [2..5) = [0, 1, 2] (no wrap).
        let got = make(arange(5), &[5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![0, 1, 2]);
    }

    #[test]
    fn sub_read_1d_nonwrap_before_shift() {
        // [0,1,2,3,4] roll +2 -> [3, 4, 0, 1, 2]. Output [0..2) = [3, 4] (no wrap).
        let got = make(arange(5), &[5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[0..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![3, 4]);
    }

    #[test]
    fn sub_read_1d_wrap_across_shift() {
        // [0,1,2,3,4] roll +2 -> [3, 4, 0, 1, 2]. Output [1..4) = [4, 0, 1] (wraps at i=2).
        let got = make(arange(5), &[5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[1..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![4, 0, 1]);
    }

    #[test]
    fn sub_read_2d_axis0_wrap() {
        // [[0,1,2],[3,4,5],[6,7,8]] roll +1 axis 0 -> [[6,7,8],[0,1,2],[3,4,5]]
        // Sub rows [0..2), cols [0..3) -> [[6,7,8],[0,1,2]]. Wraps at i=1.
        let got = make(arange(9), &[3u64, 3])
            .roll(1, 0)
            .to_ndarray_sub(&[0..2, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[6, 7, 8], [0, 1, 2]]);
    }

    #[test]
    fn sub_read_2d_axis1_wrap() {
        // [[0,1,2,3],[4,5,6,7]] roll +1 axis 1 -> [[3,0,1,2],[7,4,5,6]]
        // Sub rows [0..2), cols [0..3) -> [[3,0,1],[7,4,5]]. Wraps at i=1 on axis 1.
        let got = make(arange(8), &[2u64, 4])
            .roll(1, 1)
            .to_ndarray_sub(&[0..2, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[3, 0, 1], [7, 4, 5]]);
    }

    #[test]
    fn sub_read_empty_range() {
        let got = make(arange(8), &[2u64, 4])
            .roll(1, 0)
            .to_ndarray_sub(&[1..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    #[test]
    fn sub_read_single_element_wrap_point() {
        // [10,20,30,40,50] roll +2 -> [40, 50, 10, 20, 30]. Position 1 == 50.
        let got = make(vec![10, 20, 30, 40, 50], &[5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[1..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![50]);
    }

    // -----------------------------------------------------------------------
    // Composition with other ops
    // -----------------------------------------------------------------------

    #[test]
    fn compose_roll_then_compact() {
        let got = make(arange(6), &[2u64, 3])
            .roll(1, 0)
            .compact()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_roll_then_slice() {
        // Roll axis 0, then slice rows 1..3.
        let got = make(arange(9), &[3u64, 3])
            .roll(1, 0)
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        // Rolled: [[6,7,8],[0,1,2],[3,4,5]]; rows 1..3 = [[0,1,2],[3,4,5]]
        assert_eq!(got, array![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn compose_roll_then_cast() {
        let got = make(arange(3), &[3u64])
            .roll(1, 0)
            .cast::<f32>()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![2.0f32, 0.0, 1.0]);
    }

    #[test]
    fn compose_permute_then_roll() {
        // [[0,1,2],[3,4,5]] permute -> [[0,3],[1,4],[2,5]]; roll +1 axis 0 -> [[2,5],[0,3],[1,4]]
        let got = make(arange(6), &[2u64, 3])
            .permute_axes(&[1, 0])
            .roll(1, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[2, 5], [0, 3], [1, 4]]);
    }

    #[test]
    fn compose_roll_then_roll_same_axis_combines() {
        // Two rolls on the same axis are equivalent to one roll with the sum of shifts.
        let nd1 = make(arange(5), &[5u64])
            .roll(2, 0)
            .roll(1, 0)
            .to_ndarray()
            .unwrap();
        let nd2 = make(arange(5), &[5u64]).roll(3, 0).to_ndarray().unwrap();
        assert_eq!(nd1, nd2);
    }

    #[test]
    fn compose_roll_inverse_is_identity() {
        let nd = ndarray::Array::from_shape_vec((5,), arange(5)).unwrap();
        let got = make(arange(5), &[5u64])
            .roll(3, 0)
            .roll(-3, 0)
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
        let r = a.roll(1, 0);
        let dyn_arr = r.into_dim_dyn();
        assert_eq!(dyn_arr.shape(), &[3, 4]);
    }

    #[test]
    fn element_type_change_into_type_dyn() {
        let a = make(arange(6), &[2u64, 3]);
        let r = a.roll(1, 1);
        let dyn_et = r.into_type_dyn();
        assert_eq!(dyn_et.dtype(), &<i32 as crate::dtype::Dtyped>::DTYPE);
    }

    // -----------------------------------------------------------------------
    // Proptest: random shape + axis + shift vs hand-rolled reference
    // -----------------------------------------------------------------------

    fn roll_reference<T: Clone + Default>(
        nd: &ndarray::ArrayD<T>,
        shift: i64,
        axis: usize,
    ) -> ndarray::ArrayD<T> {
        let l = nd.shape()[axis];
        let mut out = ndarray::ArrayD::<T>::from_elem(nd.shape(), T::default());
        if l == 0 {
            return out;
        }
        let l_i128 = l as i128;
        let s_prime = ((shift as i128).rem_euclid(l_i128)) as usize;
        for (i, src_slice) in nd.axis_iter(ndarray::Axis(axis)).enumerate() {
            let dst_i = (i + s_prime) % l;
            let mut dst = out.index_axis_mut(ndarray::Axis(axis), dst_i);
            dst.assign(&src_slice);
        }
        out
    }

    fn roll_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            Array<Compact<Ty<i32>, crate::DimDyn>>,
            usize,
            i64,
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
            let axis = 0..ndim;
            // Mix of small positive, small negative, zero, and large-magnitude shifts.
            let shift = -20i64..=20i64;
            (array_strat, axis, shift).prop_map(|((nd, za), axis, shift)| (nd, za, axis, shift))
        })
    }

    proptest::proptest! {
        #[test]
        fn proptest_roll_generic(
            (nd, za, axis, shift) in roll_strategy()
        ) {
            let expected = roll_reference(&nd, shift, axis);
            let actual = za.roll(shift, axis);
            crate::util::assert_array_matches(&actual, &expected);
        }
    }
}
