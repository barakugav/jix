use std::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, ensure, Result};
use crate::storage::block::BlockSize;
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{ArraySpec, ArrayStorageInfo, OutBuf};
use crate::util::iter::NdIter;
use crate::util::{try_dim_arr, DimArray};
use crate::{Array, ArrayStorage, Dimension};

/// Selects a sub-region of an array along each dimension, returned by [`Array::slice`].
///
/// The most ergonomic form is a tuple - one Rust range (or [`SliceItem`]) per dimension.
/// Standard range types and negative integer ranges are all accepted:
///
/// ```text
/// array.slice((.., 1..4))                            // axis 0: all, axis 1: indices 1, 2, 3
/// array.slice((2.., ..3))                            // axis 0: from 2, axis 1: up to (excl.) 3
/// array.slice((1..=3, ..))                           // axis 0: indices 1-3 (inclusive end)
/// array.slice(((-2..), ..))                          // axis 0: last 2 elements
/// array.slice((.., ..-1))                            // axis 1: all but the last
/// array.slice((.., SliceItem::new(None, None, 2)))   // axis 1: every other element (step=2)
/// ```
///
/// # Slice convention
///
/// The slice is specified via [`SliceSpec`], which wraps one [`SliceItem`] per dimension.
/// Each [`SliceItem`] has three fields:
///
/// * `start` - first element to include (negative: counted from the end; `None`: beginning).
/// * `end`   - first element to exclude (negative: counted from the end; `None`: past the end).
/// * `step`  - step between selected elements (must be >= 1).
///
/// Standard Rust ranges convert to [`SliceItem`] automatically; negative-integer range
/// literals work for Python-style end-relative indexing.
///
/// `Slice<S>` carries `type Dimension = S::Dimension` - slicing does not change the number of
/// axes so the dimension type is preserved unchanged.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
///
/// // First two rows, last two columns
/// let result = a.slice((0..2, 1..)).to_ndarray()?;
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 2);
/// assert_eq!(result[[1, 1]], 6);
///
/// // Negative index: last row only
/// let b = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
/// let result = b.slice(((-1i64..), ..)).to_ndarray()?;
/// assert_eq!(result.shape(), &[1, 3]);
/// assert_eq!(result[[0, 1]], 8);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Slice<S: ArrayStorage> {
    array: S,
    /// Resolved slice for each dimension.
    slice: <S::Dimension as Dimension>::Vec<DimSlice>,
    /// `true` when every dimension has `step == 1`. Enables a cheaper read path.
    no_steps: bool,

    shape: S::Dimension,
    spec: ArraySpecDynamic,
}

impl<S: ArrayStorage> Slice<S> {
    /// Constructs a [`Slice`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, slice: SliceSpec) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();

        ensure!(
            slice.slice.len() == ndim,
            InvalidIndex,
            "slice has {} items but array has {ndim} dims",
            slice.slice.len()
        );

        let slice = try_dim_arr(ndim, |dim| {
            DimSlice::resolve(&slice.slice[dim], input_shape[dim])
        })?;
        let no_steps = slice.iter().all(|ds| ds.is_contiguous());

        let slice = S::Dimension::vec(ndim, |dim| slice[dim].clone());
        let shape = S::Dimension::from_fn(ndim, |dim| slice[dim].len());

        let inner_spec = array.spec();
        let mut block_shape = inner_spec.block_shape().clone();
        for dim in 0..ndim {
            if shape[dim] == input_shape[dim] {
                continue; // dim is unchanged
            } else if shape[dim] >= block_shape[dim] as u64 {
                continue; // dim is sliced, but still larger than the block size - no change
            } else {
                block_shape[dim] = (shape[dim] as BlockSize).max(1);
                // block_shape_tag is unchanged
            }
        }
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_tag: inner_spec.block_shape_tag().clone(),
        };

        Ok(Self {
            array,
            slice,
            no_steps,
            shape,
            spec,
        })
    }

    /// Constructs an array with [`Slice`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, slice: SliceSpec) -> Result<Array<Self>> {
        Self::new(array.into_storage(), slice).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Slice<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        // # Read behaviour
        //
        // When all dimensions have `step == 1` (`no_steps` fast path), each read translates the
        // requested index ranges by the per-dimension `start` offsets and forwards directly to the inner
        // storage - no temporary buffer is needed.
        //
        // When any dimension has `step > 1`, [`NdIter`] iterates over every combination of strided-dim
        // output indices. For each step:
        // * Strided dims use a single-element inner range for that step's position.
        // * Non-strided dims use the full translated range.
        // Each inner read goes straight into its strided sub-region of `buf` - no temporary buffer.

        check_get_range(self.shape(), index)?;

        // -----------------------------------------------------------------------
        // Fast path: all dims have step == 1.
        //
        // Each requested output range [a, b) for dim maps to inner range
        // [start + a, start + b). A single forwarded call suffices.
        // -----------------------------------------------------------------------
        let ndim = self.slice.as_ref().len();
        if self.no_steps {
            let inner_index = S::Dimension::vec(ndim, |dim| {
                let off = self.slice[dim].start;
                (index[dim].start + off)..(index[dim].end + off)
            });
            return self.array.read_data(inner_index.as_ref(), buf, context);
        }

        // -----------------------------------------------------------------------
        // General path: one or more dims have step > 1.
        //
        // We iterate over all combinations of strided-dim output indices with
        // NdIter. On each step we read from the inner storage (strided dims
        // collapsed to a single-element range; non-strided dims as full ranges)
        // straight into the matching strided sub-region of `buf`.
        //
        // Let:
        //   strided dim     - dims[d].step > 1
        //   non-strided dim - dims[d].step == 1
        //
        // For NdIter we define `iter_shape`:
        //   iter_shape[d] = out_shape[d]   if strided       (iterate over each step)
        //   iter_shape[d] = 1              if non-strided   (treated as a single block)
        //
        // For each NdIter step `idx`:
        //   inner_index[d]:
        //     strided:     let pos = dims[d].start + (index[d].start + idx[d]) * step[d]
        //                  pos..(pos + 1)
        //     non-strided: (dims[d].start + index[d].start)..(dims[d].start + index[d].end)
        //
        //   inner_read_shape[d]:
        //     strided:     1
        //     non-strided: index[d].end - index[d].start   (full range for this dim)
        //
        //   dst_byte_offset = sum_{strided d} idx[d] * dst_strides[d]
        //   (non-strided dims contribute 0 since idx[d] == 0 for them in iter_shape)
        //
        // The inner read targets a strided OutBuf over `buf[dst_byte_offset..]` with
        // `dst_strides` (the destination's own strides), so each non-strided dim's full
        // range lands at its correct position in `buf` directly - no temporary buffer or
        // copy. `dst_byte_offset` places the single strided-dim step at the right row/column.
        // -----------------------------------------------------------------------
        let dtype = self.dtype();
        let out_shape = S::Dimension::vec(ndim, |dim| index[dim].end - index[dim].start);
        if out_shape.as_ref().contains(&0) {
            buf.materialize(0, dtype);
            return Ok(());
        }
        // Forward the (possibly strided) destination's own strides so each inner read scatters
        // directly into `buf`.
        let (dst, dst_strides) = buf.get_strided_mut::<S::Dimension>(index, dtype);

        // iter_shape: out_shape for strided dims, 1 for non-strided dims.
        let iter_shape = S::Dimension::vec(ndim, |dim| {
            if self.slice[dim].is_contiguous() {
                1
            } else {
                out_shape[dim]
            }
        });
        let iter = NdIter::new(iter_shape, ());
        for (idx, ()) in iter {
            let inner_index = S::Dimension::vec(ndim, |dim| {
                let ds = &self.slice[dim];
                if ds.is_contiguous() {
                    (ds.start + index[dim].start)..(ds.start + index[dim].end)
                } else {
                    let pos = ds.start + (index[dim].start + idx[dim]) * ds.step;
                    pos..(pos + 1)
                }
            });
            let dst_byte_offset = (0..ndim)
                .filter(|&dim| !self.slice[dim].is_contiguous())
                .map(|dim| idx[dim] as usize * dst_strides[dim])
                .sum::<usize>();
            let mut out =
                unsafe { OutBuf::new_strided(&mut dst[dst_byte_offset..], dst_strides.as_ref()) };
            self.array
                .read_data(inner_index.as_ref(), &mut out, context)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
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
        ArrayStorageInfo::new_deps("Slice", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Slice<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let ndim = self.shape().len();
        check_ndim::<NewD>(ndim)?;
        let shape = NewD::from_slice(self.shape());
        let slice = NewD::vec(ndim, |dim| self.slice[dim].clone());
        Ok(Slice {
            array: self.array.dimension_change()?,
            slice,
            no_steps: self.no_steps,
            shape,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Slice<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Slice {
            array: self.array.element_type_change()?,
            slice: self.slice,
            no_steps: self.no_steps,
            shape: self.shape,
            spec: self.spec,
        })
    }
}

/// A complete slice specification: one [`SliceItem`] per dimension.
pub struct SliceSpec {
    slice: DimArray<SliceItem>,
}

impl SliceSpec {
    /// Constructs a `SliceSpec` from a slice of [`SliceItem`]s, one per dimension.
    pub fn new(slice: &[SliceItem]) -> Self {
        Self {
            slice: slice.iter().cloned().collect(),
        }
    }
}

impl<I> From<&[I]> for SliceSpec
where
    I: Into<SliceItem> + Clone,
{
    fn from(slice: &[I]) -> Self {
        Self {
            slice: slice.iter().map(|item| item.clone().into()).collect(),
        }
    }
}

macro_rules! impl_from_tuple_for_slice_spec {
    ($($ty:ident),+ ; $n:expr ; $($idx:tt),+) => {
        impl<$($ty),+> From<($($ty,)+)> for SliceSpec
        where
            $($ty: Into<SliceItem>,)+
        {
            fn from(slice: ($($ty,)+)) -> Self {
                let items: [SliceItem; $n] = [$(slice.$idx.into()),+];
                Self { slice: items.into_iter().collect() }
            }
        }
    };
}

impl_from_tuple_for_slice_spec!(I1 ; 1 ; 0);
impl_from_tuple_for_slice_spec!(I1, I2 ; 2 ; 0, 1);
impl_from_tuple_for_slice_spec!(I1, I2, I3 ; 3 ; 0, 1, 2);
impl_from_tuple_for_slice_spec!(I1, I2, I3, I4 ; 4 ; 0, 1, 2, 3);
impl_from_tuple_for_slice_spec!(I1, I2, I3, I4, I5 ; 5 ; 0, 1, 2, 3, 4);
impl_from_tuple_for_slice_spec!(I1, I2, I3, I4, I5, I6 ; 6 ; 0, 1, 2, 3, 4, 5);
impl_from_tuple_for_slice_spec!(I1, I2, I3, I4, I5, I6, I7 ; 7 ; 0, 1, 2, 3, 4, 5, 6);
impl_from_tuple_for_slice_spec!(I1, I2, I3, I4, I5, I6, I7, I8 ; 8 ; 0, 1, 2, 3, 4, 5, 6, 7);

/// A single-dimension slice descriptor: start, end, and step.
///
/// `start` and `end` may be negative (Python-style: `-1` means last element).
/// `None` means "use the natural boundary" (0 for start, dim length for end).
/// `step` must be >= 1 (negative steps are currently not supported).
#[derive(Debug, Clone, Copy)]
pub struct SliceItem {
    /// First element to include. Negative values count from the end; `None` means the start of the dimension.
    pub start: Option<i64>,
    /// First element to exclude. Negative values count from the end; `None` means past the end of the dimension.
    pub end: Option<i64>,
    /// Step between selected elements. Must be >= 1.
    pub step: i64,
}

impl SliceItem {
    /// Constructs a [`SliceItem`] from explicit `start`, `end`, and `step`. See [`Slice`] for conventions.
    pub fn new(start: Option<i64>, end: Option<i64>, step: i64) -> Self {
        Self { start, end, step }
    }
}

macro_rules! impl_from_range {
    ($range_ty:ty) => {
        impl From<$range_ty> for SliceItem {
            fn from(r: $range_ty) -> Self {
                use std::ops::RangeBounds;
                let start = match r.start_bound() {
                    Bound::Included(&s) => Some(s as i64),
                    Bound::Excluded(&s) => Some(s as i64 + 1),
                    Bound::Unbounded => None,
                };
                let end = match r.end_bound() {
                    Bound::Included(&e) => Some(e as i64 + 1),
                    Bound::Excluded(&e) => Some(e as i64),
                    Bound::Unbounded => None,
                };
                Self::new(start, end, 1)
            }
        }
    };
}
macro_rules! impl_from_range_all_types {
    ($range_ty:ident) => {
        impl_from_range!($range_ty<u32>);
        impl_from_range!($range_ty<i32>);
        impl_from_range!($range_ty<u64>);
        impl_from_range!($range_ty<i64>);
        impl_from_range!($range_ty<usize>);
        impl_from_range!($range_ty<isize>);
    };
}

impl_from_range_all_types!(Range);
impl_from_range_all_types!(RangeInclusive);
impl_from_range_all_types!(RangeFrom);
impl_from_range_all_types!(RangeTo);
impl_from_range_all_types!(RangeToInclusive);
impl From<RangeFull> for SliceItem {
    fn from(_: RangeFull) -> Self {
        Self::new(None, None, 1)
    }
}

/// Resolved, concrete slice parameters for one dimension.
#[derive(Clone, Debug)]
struct DimSlice {
    /// First element to include (absolute index into the inner array).
    start: u64,
    /// One past the last element to include (absolute, exclusive).
    end: u64,
    /// Step size (always >= 1 for now).
    step: u64,
}

impl DimSlice {
    fn resolve(item: &SliceItem, dim_len: u64) -> Result<Self> {
        let resolve_endpoint = |idx: i64, label: &str| -> Result<u64> {
            let norm = if idx < 0 { idx + dim_len as i64 } else { idx };
            ensure!(
                (0..=dim_len as i64).contains(&norm),
                InvalidIndex,
                "slice {label} {idx} is out of bounds for axis with size {dim_len}"
            );
            Ok(norm as u64)
        };

        let start = match item.start {
            None => 0,
            Some(s) => resolve_endpoint(s, "start")?,
        };
        let end = match item.end {
            None => dim_len,
            Some(e) => resolve_endpoint(e, "end")?,
        };

        ensure!(
            start <= end,
            InvalidIndex,
            "slice start {start} must be <= end {end}"
        );

        ensure!(
            item.step >= 1,
            InvalidIndex,
            "slice step {} is not supported (must be >= 1)",
            item.step
        );
        let step = item.step as u64;

        Ok(Self { start, end, step })
    }

    #[inline]
    fn len(&self) -> u64 {
        debug_assert!(self.start <= self.end);
        (self.end - self.start).div_ceil(self.step)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        self.step == 1
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use super::SliceItem;
    use crate::array::Array;
    use crate::codec::ReadContext;
    use crate::storage::Compact;
    use crate::util::{arr_params, shape_strategy, ScalarStrategy};
    use crate::{Dim, DimDyn, Ty};

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<Compact<Ty<i32>, Dim<2>>> {
        let nd = ndarray::Array::from_shape_vec([rows, cols], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<Compact<Ty<i32>, Dim<3>>> {
        let nd = ndarray::Array::from_shape_vec([d0, d1, d2], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[d0, d1, d2])).unwrap()
    }

    fn arange(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata - contiguous (no_steps)
    // -----------------------------------------------------------------------

    #[test]
    fn shape_slice_rows() {
        // tuple syntax: (1..3, ..) keeps rows 1 and 2
        assert_eq!(make2d(arange(12), 3, 4).slice((1..3, ..)).shape(), &[2, 4]);
    }

    #[test]
    fn shape_slice_cols() {
        assert_eq!(make2d(arange(12), 3, 4).slice((.., 1..3)).shape(), &[3, 2]);
    }

    #[test]
    fn shape_slice_identity_rangefull() {
        assert_eq!(make2d(arange(12), 3, 4).slice((.., ..)).shape(), &[3, 4]);
    }

    #[test]
    fn shape_slice_empty_dim() {
        // empty range on axis 0
        assert_eq!(make2d(arange(12), 3, 4).slice((1..1, ..)).shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Shape metadata - strided
    // -----------------------------------------------------------------------

    #[test]
    fn shape_strided_step2_both_axes() {
        // [3, 8], step 2 on both -> ceil(3/2)=2, ceil(8/2)=4
        assert_eq!(
            make2d(arange(24), 3, 8)
                .slice((SliceItem::new(None, None, 2), SliceItem::new(None, None, 2)))
                .shape(),
            &[2, 4]
        );
    }

    // -----------------------------------------------------------------------
    // Full reads - contiguous (tuple syntax)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_slice_rows() {
        let got = make2d(arange(12), 3, 4)
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[4, 5, 6, 7], [8, 9, 10, 11]]);
    }

    #[test]
    fn full_read_slice_cols() {
        let got = make2d(arange(12), 3, 4)
            .slice((.., 1..3))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[1, 2], [5, 6], [9, 10]]);
    }

    #[test]
    fn full_read_slice_subblock() {
        let got = make2d(arange(12), 3, 4)
            .slice((1..3, 1..3))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[5, 6], [9, 10]]);
    }

    #[test]
    fn full_read_3d_slice() {
        // [2,3,4] -> (0..2, 1..3, 1..3)
        let got = make3d(arange(24), 2, 3, 4)
            .slice((0..2, 1..3, 1..3))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[[5, 6], [9, 10]], [[17, 18], [21, 22]]]);
    }

    // -----------------------------------------------------------------------
    // Full reads - strided (tuple syntax)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_strided_axis1_step2() {
        // [3, 8], step 2 on axis 1 -> cols 0,2,4,6
        let got = make2d(arange(24), 3, 8)
            .slice((.., SliceItem::new(None, None, 2)))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22]]);
    }

    #[test]
    fn full_read_strided_axis0_step2() {
        // [6, 4], step 2 on axis 0 -> rows 0, 2, 4
        let got = make2d(arange(24), 6, 4)
            .slice((SliceItem::new(None, None, 2), ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]]);
    }

    #[test]
    fn full_read_strided_both_axes() {
        // [4, 6], step 2 on both -> rows 0,2; cols 0,2,4
        let got = make2d(arange(24), 4, 6)
            .slice((SliceItem::new(None, None, 2), SliceItem::new(None, None, 2)))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 2, 4], [12, 14, 16]]);
    }

    #[test]
    fn full_read_strided_with_start_offset() {
        // [6, 8]: axis 1 from index 1, step 2 -> indices 1,3,5,7
        let got = make2d(arange(48), 6, 8)
            .slice((.., SliceItem::new(Some(1), None, 2)))
            .to_ndarray()
            .unwrap();
        let expected_row = |r: i32| vec![r * 8 + 1, r * 8 + 3, r * 8 + 5, r * 8 + 7];
        let vals: Vec<i32> = (0..6).flat_map(expected_row).collect();
        assert_eq!(got, ndarray::Array::from_shape_vec([6, 4], vals).unwrap());
    }

    #[test]
    fn full_read_strided_3d() {
        // [4, 4, 4], step 2 on middle axis
        let got = make3d(arange(64), 4, 4, 4)
            .slice((.., SliceItem::new(None, None, 2), ..))
            .to_ndarray()
            .unwrap();
        // inner[i,j,k] = i*16 + j*4 + k; axis 1 keeps j=0,2
        let mut expected = vec![];
        for i in 0..4i32 {
            for j in [0i32, 2] {
                for k in 0..4i32 {
                    expected.push(i * 16 + j * 4 + k);
                }
            }
        }
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([4, 2, 4], expected).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Negative indices
    // -----------------------------------------------------------------------

    #[test]
    fn negative_start_last_two_rows() {
        // (-2..) on axis 0 of [5, 4] -> rows 3 and 4
        let got = make2d(arange(20), 5, 4)
            .slice((-2.., ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[12, 13, 14, 15], [16, 17, 18, 19]]);
    }

    #[test]
    fn negative_end_all_but_last_col() {
        // (..-1) on axis 1 of [3, 4] -> cols 0,1,2
        let got = make2d(arange(12), 3, 4)
            .slice((.., ..-1))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2], [4, 5, 6], [8, 9, 10]]);
    }

    #[test]
    fn negative_start_and_end() {
        // [3, 6]: axis 1 with (-4..-1) -> indices 2,3,4
        let got = make2d(arange(18), 3, 6)
            .slice((.., -4..-1))
            .to_ndarray()
            .unwrap();
        // row 0: [2,3,4]; row 1: [8,9,10]; row 2: [14,15,16]
        assert_eq!(got, array![[2, 3, 4], [8, 9, 10], [14, 15, 16]]);
    }

    #[test]
    fn negative_start_strided() {
        // [6, 4]: axis 0 from -6 (= 0) step 2 -> rows 0, 2, 4
        // negative start + step requires SliceItem since range syntax has no step
        let got = make2d(arange(24), 6, 4)
            .slice((SliceItem::new(Some(-6), None, 2), ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]]);
    }

    // -----------------------------------------------------------------------
    // Sub-region reads (slice of a slice)
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_within_contiguous_slice() {
        let got = make2d(arange(12), 3, 4)
            .slice((1..3, ..))
            .to_ndarray_sub(&[0..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[4, 5, 6, 7]]);
    }

    #[test]
    fn sub_read_within_strided_slice() {
        // [6, 8] step 2 on axis 1 -> shape [6, 4]; then read only row 0
        let got = make2d(arange(48), 6, 8)
            .slice((.., SliceItem::new(None, None, 2)))
            .to_ndarray_sub(&[0..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[0, 2, 4, 6]]);
    }

    // -----------------------------------------------------------------------
    // no_steps flag (uses Slice::new to inspect internal state)
    // -----------------------------------------------------------------------

    #[test]
    fn no_steps_flag_set_for_contiguous() {
        let a = make2d(arange(12), 3, 4);
        let s = super::Slice::new_array(a.as_ref(), (1..3, ..).into())
            .unwrap()
            .into_storage();
        assert!(s.no_steps);
    }

    #[test]
    fn no_steps_flag_unset_for_strided() {
        let a = make2d(arange(12), 3, 4);
        let s = super::Slice::new_array(a.as_ref(), (.., SliceItem::new(None, None, 2)).into())
            .unwrap()
            .into_storage();
        assert!(!s.no_steps);
    }

    // -----------------------------------------------------------------------
    // Error cases (uses Slice::new to trigger errors)
    // -----------------------------------------------------------------------

    #[test]
    fn error_wrong_number_of_items() {
        let a = make2d(arange(12), 3, 4);
        assert!(super::Slice::new_array(a.as_ref(), (0..3,).into()).is_err());
    }

    #[test]
    fn error_negative_step() {
        let a = make2d(arange(12), 3, 4);
        assert!(
            super::Slice::new_array(a.as_ref(), (SliceItem::new(None, None, -1), ..).into())
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary shape, per-dim start/end/step, ndarray oracle
    // -----------------------------------------------------------------------

    fn slice_strategy<T>() -> impl Strategy<
        Value = (
            ndarray::ArrayD<T>,
            Array<Compact<Ty<T>, DimDyn>>,
            Vec<SliceItem>,
        ),
    >
    where
        T: ScalarStrategy,
    {
        shape_strategy()
            .prop_flat_map(|shape| {
                let ndim = shape.len();
                // Generate (a, b, step) per dim using a fixed range; clamped to dim size below.
                let raw_slices =
                    prop::collection::vec((0usize..=100, 0usize..=100, 1i64..=5), ndim);
                let array_strat = crate::util::carray_strategy_from_shape::<T>(
                    Just(shape.clone()),
                    T::any_strategy(),
                );
                (array_strat, raw_slices, Just(shape))
            })
            .prop_map(|((nd, za), raw_slices, shape)| {
                let items: Vec<SliceItem> = shape
                    .iter()
                    .zip(raw_slices.iter())
                    .map(|(&dim_size, &(a, b, step))| {
                        let start = a.min(b).min(dim_size) as i64;
                        let end = a.max(b).min(dim_size) as i64;
                        SliceItem::new(Some(start), Some(end), step)
                    })
                    .collect();
                (nd, za, items)
            })
    }

    // -----------------------------------------------------------------------
    // Bound checks in DimSlice::resolve.
    // -----------------------------------------------------------------------

    fn try_slice_one_dim(
        arr: Array<Compact<Ty<i32>, Dim<2>>>,
        item: SliceItem,
    ) -> super::Result<()> {
        super::Slice::new_array(
            arr,
            super::SliceSpec::new(&[item, SliceItem::new(None, None, 1)]),
        )
        .map(|_| ())
    }

    #[test]
    fn resolve_rejects_start_positive_out_of_range() {
        // axis 0 has size 3; start = 4 is out of [-3, 3].
        let arr = make2d(arange(12), 3, 4);
        let err = try_slice_one_dim(arr, SliceItem::new(Some(4), None, 1)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidIndex);
    }

    #[test]
    fn resolve_rejects_start_negative_out_of_range() {
        // axis 0 has size 3; start = -4 normalizes to -1 (out of [0, 3]).
        let arr = make2d(arange(12), 3, 4);
        let err = try_slice_one_dim(arr, SliceItem::new(Some(-4), None, 1)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidIndex);
    }

    #[test]
    fn resolve_rejects_end_positive_out_of_range() {
        let arr = make2d(arange(12), 3, 4);
        let err = try_slice_one_dim(arr, SliceItem::new(None, Some(4), 1)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidIndex);
    }

    #[test]
    fn resolve_rejects_end_negative_out_of_range() {
        let arr = make2d(arange(12), 3, 4);
        let err = try_slice_one_dim(arr, SliceItem::new(None, Some(-4), 1)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidIndex);
    }

    #[test]
    fn resolve_rejects_start_greater_than_end() {
        let arr = make2d(arange(12), 3, 4);
        let err = try_slice_one_dim(arr, SliceItem::new(Some(2), Some(1), 1)).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::InvalidIndex);
    }

    #[test]
    fn resolve_accepts_start_equal_to_end_empty_slice() {
        // start == end (anywhere inside [0, dim_len]) is a valid empty slice.
        assert_eq!(make2d(arange(12), 3, 4).slice((2..2, ..)).shape(), &[0, 4]);
    }

    #[test]
    fn resolve_accepts_start_equal_to_dim_len_empty_slice() {
        // start == dim_len (with end == dim_len) is allowed: empty slice at the end.
        assert_eq!(make2d(arange(12), 3, 4).slice((3..3, ..)).shape(), &[0, 4]);
    }

    #[test]
    fn resolve_accepts_end_equal_to_dim_len() {
        assert_eq!(make2d(arange(12), 3, 4).slice((..3, ..)).shape(), &[3, 4]);
    }

    #[test]
    fn resolve_accepts_negative_endpoints_within_range() {
        // start = -3 -> 0; end = -1 -> 2 on axis 0 of size 3.
        assert_eq!(
            make2d(arange(12), 3, 4).slice(((-3i64..-1i64), ..)).shape(),
            &[2, 4]
        );
    }

    proptest::proptest! {
        #[test]
        fn proptest_slice((nd, za, items) in slice_strategy::<i32>()) {
            // Oracle: apply each SliceItem via ndarray's slice_axis_inplace.
            let mut expected = nd.clone();
            for (axis, item) in items.iter().enumerate() {
                expected.slice_axis_inplace(
                    ndarray::Axis(axis),
                    ndarray::Slice {
                        start: item.start.unwrap() as isize,
                        end: item.end.map(|e| e as isize),
                        step: item.step as isize,
                    },
                );
            }
            let expected = expected.as_standard_layout().into_owned();
            crate::util::assert_array_matches(
                &za.slice(super::SliceSpec::new(&items)),
                &expected,
            );
        }
    }
}
