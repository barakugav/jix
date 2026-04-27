use std::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::block::BlockSize;
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::iter::NdIter;
use crate::util::{default_strides, dim_arr, nd_copy, try_dim_arr, DimArray};

/// Selects a sub-region of an array along each dimension, returned by [`Array::slice`].
///
/// The most ergonomic form is a tuple — one Rust range (or [`SliceItem`]) per dimension.
/// Standard range types and negative integer ranges are all accepted:
///
/// ```text
/// array.slice((.., 1..4))                            // axis 0: all, axis 1: indices 1, 2, 3
/// array.slice((2.., ..3))                            // axis 0: from 2, axis 1: up to (excl.) 3
/// array.slice((1..=3, ..))                           // axis 0: indices 1–3 (inclusive end)
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
/// * `start` — first element to include (negative: counted from the end; `None`: beginning).
/// * `end`   — first element to exclude (negative: counted from the end; `None`: past the end).
/// * `step`  — step between selected elements (must be ≥ 1).
///
/// Standard Rust ranges convert to [`SliceItem`] automatically; negative-integer range
/// literals work for Python-style end-relative indexing.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
///
/// // First two rows, last two columns
/// let result = a.slice((0..2, 1..)).to_ndarray::<i32>()?;
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 2);
/// assert_eq!(result[[1, 1]], 6);
///
/// // Negative index: last row only
/// let b = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
/// let result = b.slice(((-1i64..), ..)).to_ndarray::<i32>()?;
/// assert_eq!(result.shape(), &[1, 3]);
/// assert_eq!(result[[0, 1]], 8);
/// # Ok::<(), zix::error::Error>(())
/// ```
pub struct Slice<S> {
    array: Array<S>,
    /// Resolved slice for each dimension.
    slice: DimArray<DimSlice>,
    /// `true` when every dimension has `step == 1`.  Enables a cheaper read path.
    no_steps: bool,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> Slice<S> {
    pub fn new(array: Array<S>, slice: SliceSpec) -> Result<Self> {
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

        let shape = dim_arr(ndim, |d| slice[d].len());
        let dtype = array.dtype().clone();

        let mut b_layout = array.blocks_layout().clone();
        for dim in 0..ndim {
            if shape[dim] == input_shape[dim] {
                continue;
            }
            b_layout.block_shape_hint[dim] =
                b_layout.block_shape_hint[dim].min(shape[dim] as BlockSize);
            b_layout.block_shape_tag[dim] = BlockShapeTag::Any;
            b_layout.preferred_read_shape[dim] =
                b_layout.preferred_read_shape[dim].min(shape[dim] as BlockSize);
        }

        Ok(Self {
            array,
            slice,
            no_steps,
            dtype,
            shape,
            blocks_layout: b_layout,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for Slice<S> {
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        // # Read behaviour
        //
        // When all dimensions have `step == 1` (`no_steps` fast path), each read translates the
        // requested index ranges by the per-dimension `start` offsets and forwards directly to the inner
        // storage — no temporary buffer is needed.
        //
        // When any dimension has `step > 1`, [`NdIter`] iterates over every combination of strided-dim
        // output indices.  For each step:
        // * Strided dims use a single-element inner range for that step's position.
        // * Non-strided dims use the full translated range.
        // The inner read goes into a temporary buffer which is then scattered into `buf` via [`nd_copy`].

        check_get_range(self.shape(), index)?;

        // -----------------------------------------------------------------------
        // Fast path: all dims have step == 1.
        //
        // Each requested output range [a, b) for dim d maps to inner range
        // [start + a, start + b).  A single forwarded call suffices.
        // -----------------------------------------------------------------------
        if self.no_steps {
            let ndim = self.slice.len();
            let inner_index = dim_arr(ndim, |d| {
                let off = self.slice[d].start;
                (index[d].start + off)..(index[d].end + off)
            });
            return self.array.storage.read_data(&inner_index, buf, context);
        }

        // -----------------------------------------------------------------------
        // General path: one or more dims have step > 1.
        //
        // We iterate over all combinations of strided-dim output indices with
        // NdIter.  On each step we read from the inner storage (strided dims
        // collapsed to a single-element range; non-strided dims as full ranges)
        // and scatter the result into `buf` using nd_copy.
        //
        // Let:
        //   strided dim     — dims[d].step > 1
        //   non-strided dim — dims[d].step == 1
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
        //   dst_byte_offset = Σ_{strided d} idx[d] * dst_strides[d]
        //   (non-strided dims contribute 0 since idx[d] == 0 for them in iter_shape)
        //
        //   nd_copy(tmp_buf → buf + dst_byte_offset, shape = inner_read_shape,
        //           src_strides = C-order over inner_read_shape,
        //           dst_strides = C-order over out_shape)
        //
        // nd_copy iterates over inner_read_shape (1 for strided dims, full for non-
        // strided).  The single step on strided dims is handled by dst_byte_offset
        // already placing us at the right row/column; nd_copy takes care of the rest.
        // -----------------------------------------------------------------------
        check_get_buffer_size(index, &self.dtype, buf)?;
        let ndim = self.slice.len();
        let itemsize = self.dtype.itemsize() as usize;
        let out_shape = dim_arr(ndim, |d| (index[d].end - index[d].start) as usize);
        let dst_strides = default_strides(&out_shape, itemsize);

        // inner_read_shape: 1 for strided dims, full range for non-strided dims.
        let inner_read_shape = dim_arr(ndim, |d| {
            if self.slice[d].is_contiguous() {
                out_shape[d]
            } else {
                1
            }
        });
        let src_strides = default_strides(&inner_read_shape, itemsize);
        let tmp_buf_bytes = inner_read_shape.iter().product::<usize>() * itemsize;
        let mut tmp_buf = context.tmp_buf(tmp_buf_bytes, self.dtype.alignment());

        // iter_shape: out_shape for strided dims, 1 for non-strided dims.
        let iter_shape = dim_arr(ndim, |d| {
            if self.slice[d].is_contiguous() {
                1
            } else {
                out_shape[d] as u64
            }
        });
        let mut iter = NdIter::new(&iter_shape, ());
        while let Some((idx, ())) = iter.next() {
            let inner_index = dim_arr(ndim, |d| {
                let ds = &self.slice[d];
                if ds.is_contiguous() {
                    (ds.start + index[d].start)..(ds.start + index[d].end)
                } else {
                    let pos = ds.start + (index[d].start + idx[d]) * ds.step;
                    pos..(pos + 1)
                }
            });
            let tmp = tmp_buf.as_mut_slice();
            self.array.storage.read_data(&inner_index, tmp, context)?;

            let dst_byte_offset = (0..ndim)
                .filter(|&d| !self.slice[d].is_contiguous())
                .map(|d| idx[d] as usize * dst_strides[d])
                .sum::<usize>();
            let dst_ptr = unsafe { buf.as_mut_ptr().add(dst_byte_offset) };
            unsafe {
                nd_copy(
                    tmp.as_ptr(),
                    dst_ptr,
                    &inner_read_shape,
                    &src_strides,
                    &dst_strides,
                    itemsize,
                )
            };
        }
        Ok(())
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage._spec()
        }
    }
}

/// A complete slice specification: one [`SliceItem`] per dimension.
pub struct SliceSpec {
    slice: DimArray<SliceItem>,
}

impl SliceSpec {
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
/// `step` must be ≥ 1 (negative steps are not supported).
#[derive(Debug, Clone, Copy)]
pub struct SliceItem {
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub step: i64,
}

impl SliceItem {
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
        ensure!(
            item.step >= 1,
            InvalidIndex,
            "slice step {} is not supported (must be >= 1)",
            item.step
        );
        let step = item.step as u64;

        let resolve_index = |idx: i64| -> u64 {
            if idx < 0 {
                (dim_len as i64 + idx).max(0) as u64
            } else {
                (idx as u64).min(dim_len)
            }
        };

        let start = item.start.map(resolve_index).unwrap_or(0);
        let end = item.end.map(resolve_index).unwrap_or(dim_len);

        Ok(Self { start, end, step })
    }

    fn len(&self) -> u64 {
        if self.start >= self.end {
            0
        } else {
            (self.end - self.start).div_ceil(self.step)
        }
    }

    fn is_contiguous(&self) -> bool {
        self.step == 1
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use super::SliceItem;
    use crate::array::Array;
    use crate::codec::ReadContext;
    use crate::util::arr_params;

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<crate::storage::Compact> {
        let nd = ndarray::ArrayD::from_shape_vec(vec![rows, cols], vals).unwrap();
        Array::compact_array_with(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<crate::storage::Compact> {
        let nd = ndarray::ArrayD::from_shape_vec(vec![d0, d1, d2], vals).unwrap();
        Array::compact_array_with(&nd, arr_params(&[d0, d1, d2])).unwrap()
    }

    fn seq(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata — contiguous (no_steps)
    // -----------------------------------------------------------------------

    #[test]
    fn shape_slice_rows() {
        // tuple syntax: (1..3, ..) keeps rows 1 and 2
        assert_eq!(make2d(seq(12), 3, 4).slice((1..3, ..)).shape(), &[2, 4]);
    }

    #[test]
    fn shape_slice_cols() {
        assert_eq!(make2d(seq(12), 3, 4).slice((.., 1..3)).shape(), &[3, 2]);
    }

    #[test]
    fn shape_slice_identity_rangefull() {
        assert_eq!(make2d(seq(12), 3, 4).slice((.., ..)).shape(), &[3, 4]);
    }

    #[test]
    fn shape_slice_empty_dim() {
        // empty range on axis 0
        assert_eq!(make2d(seq(12), 3, 4).slice((1..1, ..)).shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Shape metadata — strided
    // -----------------------------------------------------------------------

    #[test]
    fn shape_strided_step2_both_axes() {
        // [3, 8], step 2 on both → ceil(3/2)=2, ceil(8/2)=4
        assert_eq!(
            make2d(seq(24), 3, 8)
                .slice((SliceItem::new(None, None, 2), SliceItem::new(None, None, 2)))
                .shape(),
            &[2, 4]
        );
    }

    // -----------------------------------------------------------------------
    // Full reads — contiguous (tuple syntax)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_slice_rows() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 4], vec![4, 5, 6, 7, 8, 9, 10, 11]).unwrap()
        );
    }

    #[test]
    fn full_read_slice_cols() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .slice((.., 1..3))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 2], vec![1, 2, 5, 6, 9, 10]).unwrap()
        );
    }

    #[test]
    fn full_read_slice_subblock() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .slice((1..3, 1..3))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![5, 6, 9, 10]).unwrap()
        );
    }

    #[test]
    fn full_read_3d_slice() {
        // [2,3,4] → (0..2, 1..3, 1..3)
        let got: ArrayD<i32> = make3d(seq(24), 2, 3, 4)
            .slice((0..2, 1..3, 1..3))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2, 2], vec![5, 6, 9, 10, 17, 18, 21, 22]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Full reads — strided (tuple syntax)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_strided_axis1_step2() {
        // [3, 8], step 2 on axis 1 → cols 0,2,4,6
        let got: ArrayD<i32> = make2d(seq(24), 3, 8)
            .slice((.., SliceItem::new(None, None, 2)))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
                .unwrap()
        );
    }

    #[test]
    fn full_read_strided_axis0_step2() {
        // [6, 4], step 2 on axis 0 → rows 0, 2, 4
        let got: ArrayD<i32> = make2d(seq(24), 6, 4)
            .slice((SliceItem::new(None, None, 2), ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19])
                .unwrap()
        );
    }

    #[test]
    fn full_read_strided_both_axes() {
        // [4, 6], step 2 on both → rows 0,2; cols 0,2,4
        let got: ArrayD<i32> = make2d(seq(24), 4, 6)
            .slice((SliceItem::new(None, None, 2), SliceItem::new(None, None, 2)))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3], vec![0, 2, 4, 12, 14, 16]).unwrap()
        );
    }

    #[test]
    fn full_read_strided_with_start_offset() {
        // [6, 8]: axis 1 from index 1, step 2 → indices 1,3,5,7
        let got: ArrayD<i32> = make2d(seq(48), 6, 8)
            .slice((.., SliceItem::new(Some(1), None, 2)))
            .to_ndarray()
            .unwrap();
        let expected_row = |r: i32| vec![r * 8 + 1, r * 8 + 3, r * 8 + 5, r * 8 + 7];
        let vals: Vec<i32> = (0..6).flat_map(expected_row).collect();
        assert_eq!(got, ArrayD::from_shape_vec(vec![6, 4], vals).unwrap());
    }

    #[test]
    fn full_read_strided_3d() {
        // [4, 4, 4], step 2 on middle axis
        let got: ArrayD<i32> = make3d(seq(64), 4, 4, 4)
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
            ArrayD::from_shape_vec(vec![4, 2, 4], expected).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Negative indices
    // -----------------------------------------------------------------------

    #[test]
    fn negative_start_last_two_rows() {
        // (-2..) on axis 0 of [5, 4] → rows 3 and 4
        let got: ArrayD<i32> = make2d(seq(20), 5, 4)
            .slice((-2.., ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 4], vec![12, 13, 14, 15, 16, 17, 18, 19]).unwrap()
        );
    }

    #[test]
    fn negative_end_all_but_last_col() {
        // (..-1) on axis 1 of [3, 4] → cols 0,1,2
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .slice((.., ..-1))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 3], vec![0, 1, 2, 4, 5, 6, 8, 9, 10]).unwrap()
        );
    }

    #[test]
    fn negative_start_and_end() {
        // [3, 6]: axis 1 with (-4..-1) → indices 2,3,4
        let got: ArrayD<i32> = make2d(seq(18), 3, 6)
            .slice((.., -4..-1))
            .to_ndarray()
            .unwrap();
        // row 0: [2,3,4]; row 1: [8,9,10]; row 2: [14,15,16]
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 3], vec![2, 3, 4, 8, 9, 10, 14, 15, 16]).unwrap()
        );
    }

    #[test]
    fn negative_start_strided() {
        // [6, 4]: axis 0 from -6 (= 0) step 2 → rows 0, 2, 4
        // negative start + step requires SliceItem since range syntax has no step
        let got: ArrayD<i32> = make2d(seq(24), 6, 4)
            .slice((SliceItem::new(Some(-6), None, 2), ..))
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19])
                .unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Sub-region reads (slice of a slice)
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_within_contiguous_slice() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .slice((1..3, ..))
            .to_ndarray_sub(&[0..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 4], vec![4, 5, 6, 7]).unwrap()
        );
    }

    #[test]
    fn sub_read_within_strided_slice() {
        // [6, 8] step 2 on axis 1 → shape [6, 4]; then read only row 0
        let got: ArrayD<i32> = make2d(seq(48), 6, 8)
            .slice((.., SliceItem::new(None, None, 2)))
            .to_ndarray_sub(&[0..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 4], vec![0, 2, 4, 6]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // no_steps flag (uses Slice::new to inspect internal state)
    // -----------------------------------------------------------------------

    #[test]
    fn no_steps_flag_set_for_contiguous() {
        let a = make2d(seq(12), 3, 4);
        let s = super::Slice::new(a.as_ref(), (1..3, ..).into()).unwrap();
        assert!(s.no_steps);
    }

    #[test]
    fn no_steps_flag_unset_for_strided() {
        let a = make2d(seq(12), 3, 4);
        let s = super::Slice::new(a.as_ref(), (.., SliceItem::new(None, None, 2)).into()).unwrap();
        assert!(!s.no_steps);
    }

    // -----------------------------------------------------------------------
    // Error cases (uses Slice::new to trigger errors)
    // -----------------------------------------------------------------------

    #[test]
    fn error_wrong_number_of_items() {
        let a = make2d(seq(12), 3, 4);
        assert!(super::Slice::new(a.as_ref(), (0..3,).into()).is_err());
    }

    #[test]
    fn error_negative_step() {
        let a = make2d(seq(12), 3, 4);
        assert!(
            super::Slice::new(a.as_ref(), (SliceItem::new(None, None, -1), ..).into()).is_err()
        );
    }
}
