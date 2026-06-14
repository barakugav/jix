use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::iter::NdIter;
use crate::util::{default_strides, dim_arr, nd_copy, DimArray, IterExt};
use crate::{ArrayStorage, Dimension, IntoDimension};

/// Reinterprets an array with a different shape, returned by [`Array::reshape_view`].
///
/// The total number of elements must be the same: the product of the new shape must equal the
/// product of the original shape. Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Dimension tracking
///
/// `Reshape<S, D>` is generic over `D: Dimension`, determined by the shape argument type.
/// Statically-sized arguments encode the output ndim in the type; slice arguments yield
/// [`DimDyn`](crate::DimDyn):
///
/// | Argument type | Output `D` |
/// |---|---|
/// | `u64` | `Dim<1>` |
/// | `[u64; N]` / `&[u64; N]` | `Dim<N>` |
/// | `(u64, u64, ...)` N-tuple | `Dim<N>` |
/// | `&[u64]` / `&Vec<u64>` | `DimDyn` |
///
/// # Performance
///
/// When the new shape is not aligned with the underlying block layout, a single read request may
/// span many blocks that were not contiguous in the original array, causing significant
/// read-amplification. In the worst case every element access decompresses a different block.
///
/// Prefer [`Array::reshape`] over `reshape_view` unless you intend to chain further lazy
/// operations before materializing. If you do use `reshape_view`, call [`.compact()`](Array::compact)
/// as soon as possible to produce a compactly-stored array with a block layout matched to the
/// new shape.
///
/// # Examples
///
/// Different argument types select both the new shape and the output dimension type:
///
/// ```
/// use jix::{Array, Dim};
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]])?; // shape [2, 3]
///
/// // [u64; 1] -> output D = Dim<1>: compiler knows the result is 1-D
/// assert_eq!(a.as_ref().reshape_view([6u64]).shape(), &[6]);
///
/// // (u64, u64) -> output D = Dim<2>: compiler knows the result is 2-D
/// assert_eq!(a.as_ref().reshape_view((3u64, 2u64)).shape(), &[3, 2]);
///
/// // &[u64] -> output D = DimDyn: ndim only known at runtime
/// let new_shape = vec![6u64];
/// assert_eq!(a.as_ref().reshape_view(new_shape.as_slice()).shape(), &[6]);
///
/// // Elements are the same regardless of argument style
/// assert_eq!(
///     a.reshape_view([6u64]).to_ndarray()?.as_slice().unwrap(),
///     &[1, 2, 3, 4, 5, 6]
/// );
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Reshape<S, D> {
    array: S,

    new_shape: D,
    blocks_layout: BlocksLayout,
}
impl<S, D> Reshape<S, D> {
    /// Constructs a [`Reshape`] storage. See the struct docs for semantics and examples.
    pub fn new<Sh>(array: S, shape: Sh) -> Result<Self>
    where
        S: ArrayStorage,
        D: Dimension,
        Sh: IntoDimension<Dimension = D>,
    {
        let new_shape_raw = shape.into_dimension()?;
        let new_shape = DimArray::from_slice(new_shape_raw.as_slice()).unwrap();
        let orig_shape = DimArray::from_slice(array.shape()).unwrap();
        let nitems = orig_shape.iter().cloned().try_product().unwrap();
        let new_nitems = new_shape.iter().cloned().try_product().unwrap();
        ensure!(
            nitems == new_nitems,
            InvalidShapeOperation,
            "cannot reshape array of shape {orig_shape:?} into shape {new_shape:?}"
        );

        let orig_logical_strides = default_strides(&orig_shape, 1);
        let new_logical_strides = default_strides(&new_shape, 1);
        let same_logical_stride = (0..new_shape.len())
            .scan(0, |orig_dim_idx, new_dim_idx| {
                Some(loop {
                    if *orig_dim_idx >= orig_shape.len() {
                        break None; // cant really happen, last dims always match, unless orig_shape.len()==0
                    }
                    if orig_logical_strides[*orig_dim_idx] == new_logical_strides[new_dim_idx] {
                        break Some(*orig_dim_idx as u8);
                    }
                    *orig_dim_idx += 1;
                })
            })
            .collect::<DimArray<_>>();

        let mut b_layout = array.spec().blocks_layout.clone();
        let mut block_shape_hint = DimArray::new();
        let mut block_shape_tag = DimArray::new();
        let mut preferred_read_shape = DimArray::new();
        // TODO: finalize the logic here, we can find a good heuristic
        for dim in 0..new_shape.len() {
            if let Some(orig_dim) = same_logical_stride[dim] {
                let orig_dim = orig_dim as usize;
                let same_dim_len = orig_shape[orig_dim] == new_shape[dim];
                block_shape_hint.push(b_layout.block_shape_hint[orig_dim]);
                preferred_read_shape.push(b_layout.preferred_read_shape[orig_dim]);
                block_shape_tag.push(match b_layout.block_shape_tag[orig_dim] {
                    BlockShapeTag::Fixed => {
                        if same_dim_len {
                            BlockShapeTag::Fixed
                        } else {
                            BlockShapeTag::MultipleOf
                        }
                    }
                    BlockShapeTag::MultipleOf => BlockShapeTag::MultipleOf,
                    BlockShapeTag::Any => BlockShapeTag::Any,
                });
            } else {
                block_shape_hint.push(1);
                preferred_read_shape.push(1);
                block_shape_tag.push(BlockShapeTag::Any);
            }
        }
        b_layout.block_shape_hint = block_shape_hint;
        b_layout.block_shape_tag = block_shape_tag;
        b_layout.preferred_read_shape = preferred_read_shape;

        Ok(Self {
            new_shape: new_shape_raw,
            blocks_layout: b_layout,
            array,
        })
    }

    /// Constructs an array with [`Reshape`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<Sh>(array: Array<S>, shape: Sh) -> Result<Array<Self>>
    where
        S: ArrayStorage,
        D: Dimension,
        Sh: IntoDimension<Dimension = D>,
    {
        Self::new(array.into_storage(), shape).map(Array::from_storage)
    }
}
impl<S, D> ArrayStorage for Reshape<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    type ElementType = S::ElementType;
    type Dimension = D;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        // -----------------------------------------------------------------------
        // Core concept
        // -----------------------------------------------------------------------
        // A reshape does not move any data - it only reinterprets the flat,
        // C-order (row-major) element sequence under a new shape. Element `k`
        // in the flattened array is the same physical byte regardless of whether
        // the array is shaped [A, B] or [C, D] (as long as A*B == C*D).
        //
        // When a caller asks for a sub-region of the *new* shape we must figure
        // out which ranges in the *original* shape cover exactly those elements,
        // then forward one or more reads to the underlying storage and assemble
        // the results into `buf`.
        //
        // -----------------------------------------------------------------------
        // Dimension matching: "same logical stride"
        // -----------------------------------------------------------------------
        // Two shapes share a "dimension boundary" when a particular stride value
        // appears in both stride arrays. "Logical stride" here means the number
        // of *elements* between successive steps along a dimension (i.e. strides
        // computed with itemsize = 1).
        //
        // For example:
        //   orig [6, 4]  -> logical strides [4, 1]
        //   new  [2, 3, 4] -> logical strides [12, 4, 1]
        //
        // New dim 1 (stride 4) == orig dim 0 (stride 4), so they are "matched".
        // New dim 2 (stride 1) == orig dim 1 (stride 1), so they are matched too.
        // New dim 0 (stride 12) has no counterpart in orig -> unmatched.
        //
        // When a new dim is matched to an orig dim it means that consecutive
        // steps along the new dim correspond to exactly the same memory layout as
        // consecutive steps along the orig dim. The requested index range for
        // that new dim can therefore be forwarded verbatim as the read range for
        // the corresponding orig dim.
        //
        // The matching scan (`same_logical_stride`) advances through orig dims
        // monotonically: for each new dim (left to right) it looks for the next
        // orig dim with the same stride. This preserves the ordering invariant
        // that matched pairs always respect the nesting of C-order dimensions.
        //
        // -----------------------------------------------------------------------
        // Reading strategy
        // -----------------------------------------------------------------------
        // We split all new dims into two groups:
        //
        //   MATCHED dims   - new dim j is paired with orig dim i.
        //                    Their index ranges are forwarded directly. A single
        //                    call to the underlying storage can cover the full
        //                    requested range along all matched dims at once.
        //
        //   UNMATCHED dims - new dim j crosses an original dimension boundary
        //                    (e.g. in [6]->[2,3] neither new dim aligns with the
        //                    single orig dim). We cannot express an arbitrary
        //                    sub-region of an unmatched dim as a contiguous range
        //                    in the original shape, so we handle them by iterating
        //                    over every index along those dims one step at a time.
        //
        // `iteration_shape` is shaped like the new dims but with size 1 for every
        // matched dim and the actual requested size for every unmatched dim.
        // `NdIter` over this shape visits every combination of unmatched-dim
        // positions exactly once.
        //
        // For each iteration step `idx`:
        //
        //   1. BUILD `read_range` for the underlying storage (length orig_ndim):
        //      - Matched orig dim i  -> forward index[matched_new_dim].
        //      - Unmatched orig dim i -> convert the current unmatched-dim position
        //        to a flat element index and decompose it back into orig coords:
        //
        //          flat = sum_{unmatched new dim d} (index[d].start + idx[d])
        //                                         * new_logical_strides[d]
        //
        //          orig_coord[i] = (flat / orig_logical_strides[i]) % orig_shape[i]
        //          read_range[i] = orig_coord[i]..(orig_coord[i] + 1)
        //
        //        This is safe because the set of unmatched new dims covers the set
        //        of unmatched orig dims in the flat index space, so `flat`
        //        decomposes cleanly using the original strides.
        //
        //   2. READ into `tmp_buf` from the underlying storage using `read_range`.
        //      `tmp_buf` is sized for the matched-dims block only (one element per
        //      unmatched orig dim, full range for matched dims).
        //
        //   3. COPY from `tmp_buf` into the correct position in `buf` using
        //      `nd_copy`. The source shape is `new_read_shape` (the matched dims'
        //      requested sizes, 1 elsewhere). The destination pointer is offset
        //      by the byte contribution of the unmatched dims' current position:
        //
        //          dst_byte_offset = sum_{unmatched new dim d} idx[d] * dst_strides[d]
        //
        //      `nd_copy` then iterates over the matched dims internally, so each
        //      element ends up exactly where it belongs in `buf`.
        // -----------------------------------------------------------------------
        check_get_range(self.shape(), index)?;
        let dtype = self.dtype();
        check_get_buffer_size(index, dtype, buf)?;

        let orig_shape = self.array.shape();
        let new_shape = self.new_shape.as_slice();
        let ndim = new_shape.len();
        let orig_ndim = orig_shape.len();
        if index.iter().any(|r| r.start >= r.end) {
            return Ok(());
        }

        let orig_logical_strides = default_strides(orig_shape, 1);
        let new_logical_strides = default_strides(new_shape, 1);
        let same_logical_stride = (0..new_shape.len())
            .scan(0, |orig_dim_idx, new_dim_idx| {
                Some(loop {
                    if *orig_dim_idx >= orig_shape.len() {
                        break None; // cant really happen, last dims always match, unless orig_shape.len()==0
                    }
                    if orig_logical_strides[*orig_dim_idx] == new_logical_strides[new_dim_idx]
                        && new_shape[new_dim_idx] >= 1
                        && orig_shape[*orig_dim_idx] >= new_shape[new_dim_idx]
                    // TODO: its possible to remove the dim length conditions
                    {
                        let matched = *orig_dim_idx as u8;
                        *orig_dim_idx += 1;
                        break Some(matched);
                    }
                    *orig_dim_idx += 1;
                })
            })
            .collect::<DimArray<_>>();
        let same_logical_stride_inv = {
            let mut inv = dim_arr(orig_ndim, |_| None);
            for (new_dim, &orig_dim) in same_logical_stride.iter().enumerate() {
                if let Some(orig_dim) = orig_dim {
                    inv[orig_dim as usize] = Some(new_dim as u8);
                }
            }
            inv
        };
        debug_assert_eq!(
            same_logical_stride
                .iter()
                .filter(|dim| dim.is_some())
                .count(),
            same_logical_stride_inv
                .iter()
                .filter(|dim| dim.is_some())
                .count()
        );

        // dims that have the same logical stride in the original and new shape can be read directly,
        // the rest we need to read one entry at a time and copy into the output buffer.
        let orig_read_shape = dim_arr(orig_ndim, |dim| {
            if let Some(new_dim) = same_logical_stride_inv[dim] {
                index[new_dim as usize].end - index[new_dim as usize].start
            } else {
                1
            }
        });
        let new_read_shape = dim_arr(ndim, |dim| {
            if same_logical_stride[dim].is_some() {
                index[dim].end - index[dim].start
            } else {
                1
            }
        });

        let mut tmp_buf = context.tmp_buf(
            orig_read_shape.iter().product::<u64>() as usize * dtype.itemsize() as usize,
            dtype.alignment(),
        );
        let tmp_buf_strides = default_strides(&new_read_shape, dtype.itemsize() as _);
        let out_buf_shape = dim_arr(ndim, |dim| index[dim].end - index[dim].start);
        let dst_strides = default_strides(&out_buf_shape, dtype.itemsize() as _);

        // We use an nd-iter over the dims that DO NOT match any original dim.
        let iteration_shape = dim_arr(ndim, |dim| {
            if same_logical_stride[dim].is_some() {
                1
            } else {
                index[dim].end - index[dim].start
            }
        });
        let mut iter = NdIter::new(D::from_slice(&iteration_shape).unwrap(), ());
        while let Some((idx, ())) = iter.next() {
            let read_range = dim_arr(orig_ndim, |dim| {
                if let Some(new_dim) = same_logical_stride_inv[dim] {
                    debug_assert_eq!(idx[new_dim as usize], 0);
                    index[new_dim as usize].clone()
                } else {
                    let flat: u64 = (0..ndim)
                        .filter(|&dim| same_logical_stride[dim].is_none())
                        .map(|dim| (index[dim].start + idx[dim]) * new_logical_strides[dim])
                        .sum();
                    let orig_coord = (flat / orig_logical_strides[dim]) % orig_shape[dim];
                    orig_coord..(orig_coord + 1)
                }
            });

            let tmp_buf = tmp_buf.as_mut_slice();
            self.array.read_data(&read_range, tmp_buf, context)?;

            let dst_byte_offset: usize = (0..ndim)
                .filter(|&dim| same_logical_stride[dim].is_none())
                .map(|dim| idx[dim] as usize * dst_strides[dim] as usize)
                .sum();
            let dst_ptr = unsafe { buf.as_mut_ptr().add(dst_byte_offset) };
            unsafe {
                nd_copy(
                    tmp_buf.as_ptr(),
                    dst_ptr,
                    D::from_slice(&new_read_shape).unwrap(),
                    &tmp_buf_strides,
                    &dst_strides,
                    dtype.itemsize() as _,
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

    type DimensionChange<NewD: crate::Dimension> = Reshape<S, NewD>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Reshape {
            array: self.array,
            new_shape: NewD::from_slice(self.new_shape.as_slice())?,
            blocks_layout: self.blocks_layout,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Reshape<S::ElementTypeChange<NewET>, D>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Reshape {
            array: self.array.element_type_change()?,
            new_shape: self.new_shape,
            blocks_layout: self.blocks_layout,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::array::Array;
    use crate::storage::Compact;
    use crate::util::{arr_params, shape_strategy, ScalarStrategy};
    use crate::{DimDyn, Ty};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Create a 1-D Array<Compact> from `vals` with the given block size.
    fn make1d<T: crate::dtype::Dtyped>(
        vals: Vec<T>,
        block_size: usize,
    ) -> Array<Compact<Ty<T>, DimDyn>> {
        let nd = ndarray::Array::from_shape_vec(vec![vals.len()], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[block_size])).unwrap()
    }

    /// Create a 2-D Array<Compact>.
    fn make2d<T: crate::dtype::Dtyped>(
        vals: Vec<T>,
        rows: usize,
        cols: usize,
        block_shape: &[usize],
    ) -> Array<Compact<Ty<T>, DimDyn>> {
        let nd = ndarray::Array::from_shape_vec(vec![rows, cols], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(block_shape)).unwrap()
    }

    /// Create a 3-D Array<Compact>.
    fn make3d<T: crate::dtype::Dtyped>(
        vals: Vec<T>,
        d0: usize,
        d1: usize,
        d2: usize,
        block_shape: &[usize],
    ) -> Array<Compact<Ty<T>, DimDyn>> {
        let nd = ndarray::Array::from_shape_vec(vec![d0, d1, d2], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(block_shape)).unwrap()
    }

    fn u8s(n: usize) -> Vec<u8> {
        (0..n).map(|i| i as u8).collect()
    }
    fn i32s(n: usize) -> Vec<i32> {
        (0..n).map(|i| i as i32).collect()
    }
    fn f32s(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }
    fn f64s(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_after_reshape_1d_to_2d() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        assert_eq!(r.shape(), &[3, 4]);
    }

    #[test]
    fn shape_after_reshape_2d_to_1d() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        assert_eq!(r.shape(), &[12]);
    }

    #[test]
    fn shape_after_reshape_2d_to_3d() {
        let a = make2d(u8s(24), 4, 6, &[4, 6]);
        let r = a.reshape_view(&[2, 3, 4]);
        assert_eq!(r.shape(), &[2, 3, 4]);
    }

    #[test]
    fn dtype_preserved_after_reshape() {
        use crate::dtype::Dtyped;
        let a = make1d(i32s(6), 6);
        let r = a.reshape_view(&[2, 3]);
        assert_eq!(r.dtype(), &i32::DTYPE);
    }

    #[test]
    fn reshape_wrong_size_errors() {
        let a = make1d(u8s(12), 12);
        assert!(super::Reshape::new_array(a, &[3, 5]).is_err());
    }

    // -----------------------------------------------------------------------
    // Full read: 1-D -> 2-D
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_to_2d_3x4_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_1d_to_2d_4x3_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[4, 3]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([4, 3], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_1d_to_2d_2x6_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 6], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_1d_to_2d_6x2_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[6, 2]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([6, 2], u8s(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Full read: 2-D -> 1-D (flatten)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_2d_to_1d_flatten() {
        let a = make2d(i32s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([12], i32s(12)).unwrap());
    }

    #[test]
    fn full_read_2d_to_1d_flatten_non_square() {
        let a = make2d(u8s(20), 4, 5, &[4, 5]);
        let r = a.reshape_view(&[20]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([20], u8s(20)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: 2-D -> 2-D (re-partition)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_2d_to_2d_repartition() {
        // [3, 4] -> [2, 6]: rows of 3 in orig map to interleaved rows of 2 in new
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 6], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_2d_to_2d_repartition_asymmetric() {
        // [4, 3] -> [3, 4]
        let a = make2d(u8s(12), 4, 3, &[4, 3]);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], u8s(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Full read: higher dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_to_3d() {
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], u8s(24)).unwrap()
        );
    }

    #[test]
    fn full_read_3d_to_1d_flatten() {
        let a = make3d(i32s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape_view(&[24]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([24], i32s(24)).unwrap());
    }

    #[test]
    fn full_read_3d_to_2d() {
        // [2, 3, 4] -> [6, 4]
        let a = make3d(u8s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape_view(&[6, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([6, 4], u8s(24)).unwrap()
        );
    }

    #[test]
    fn full_read_2d_to_3d() {
        // [6, 4] -> [2, 3, 4]
        let a = make2d(u8s(24), 6, 4, &[6, 4]);
        let r = a.reshape_view(&[2, 3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], u8s(24)).unwrap()
        );
    }

    #[test]
    fn full_read_3d_repartition() {
        // [2, 3, 4] -> [2, 12]
        let a = make3d(u8s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape_view(&[2, 12]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 12], u8s(24)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Full read: identity reshape (same shape)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_same_shape_1d() {
        let a = make1d(i32s(8), 8);
        let r = a.reshape_view(&[8]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([8], i32s(8)).unwrap());
    }

    #[test]
    fn full_read_same_shape_2d() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], u8s(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Full read: single-element array
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_single_element_1d_to_1d() {
        let a = make1d(vec![42u8], 1);
        let r = a.reshape_view(&[1]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, array![42u8]);
    }

    // -----------------------------------------------------------------------
    // Full read: various dtypes
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_dtype_i32() {
        let a = make1d(i32s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], i32s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_dtype_f32() {
        let a = make1d(f32s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], f32s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_dtype_f64() {
        let a = make1d(f64s(12), 12);
        let r = a.reshape_view(&[4, 3]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([4, 3], f64s(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Subregion reads: 1-D -> 2-D [3, 4], values 0..12
    // -----------------------------------------------------------------------
    //
    // Layout after reshape to [3, 4]:
    //   row 0: [0,  1,  2,  3]
    //   row 1: [4,  5,  6,  7]
    //   row 2: [8,  9, 10, 11]

    #[test]
    fn sub_read_first_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[0..1, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3]]);
    }

    #[test]
    fn sub_read_middle_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[1..2, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[4, 5, 6, 7]]);
    }

    #[test]
    fn sub_read_last_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[2..3, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[8, 9, 10, 11]]);
    }

    #[test]
    fn sub_read_first_two_rows() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[0..2, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3], [4, 5, 6, 7]]);
    }

    #[test]
    fn sub_read_first_two_columns() {
        // [0..3, 0..2] -> rows 0-2, cols 0-1
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[0..3, 0..2], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1], [4, 5], [8, 9]]);
    }

    #[test]
    fn sub_read_last_two_columns() {
        // [0..3, 2..4] -> rows 0-2, cols 2-3
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[0..3, 2..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[2, 3], [6, 7], [10, 11]]);
    }

    #[test]
    fn sub_read_inner_2x2() {
        // [1..3, 1..3]
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[1..3, 1..3], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[5, 6], [9, 10]]);
    }

    #[test]
    fn sub_read_single_element_center() {
        // [1..2, 2..3] -> element at (1,2) = 6
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[1..2, 2..3], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[6]]);
    }

    #[test]
    fn sub_read_single_element_corner() {
        // [2..3, 3..4] -> element at (2,3) = 11
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[2..3, 3..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[11]]);
    }

    // -----------------------------------------------------------------------
    // Subregion reads: 2-D -> 1-D
    // reshape [3, 4] -> [12], sub-read [3..9]
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_flatten_middle_range() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        let got = r.to_ndarray_sub(&[3..9], &r.read_ctx()).unwrap();
        assert_eq!(got, array![3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sub_read_flatten_partial_first_row() {
        // Flat [2..6) spans the last 2 of row-0 and first 2 of row-1 (in orig [3,4])
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        let got = r.to_ndarray_sub(&[2..6], &r.read_ctx()).unwrap();
        assert_eq!(got, array![2, 3, 4, 5]);
    }

    // -----------------------------------------------------------------------
    // Subregion reads: [2, 6] <- reshape of 1-D [12]
    //
    // Layout:
    //   row 0: [0,  1,  2,  3,  4,  5]
    //   row 1: [6,  7,  8,  9, 10, 11]
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_2x6_row0_partial() {
        // [0..1, 1..4] -> [1, 3] = [1, 2, 3]
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray_sub(&[0..1, 1..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[1, 2, 3]]);
    }

    #[test]
    fn sub_read_2x6_both_rows_partial_cols() {
        // [0..2, 2..5] -> rows 0-1, cols 2-4
        // row 0: [2, 3, 4]; row 1: [8, 9, 10]
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray_sub(&[0..2, 2..5], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[2, 3, 4], [8, 9, 10]]);
    }

    // -----------------------------------------------------------------------
    // Subregion reads: 3-D [2, 3, 4]
    //
    // Layout (flat indices):
    //   (0,0,*): 0-3    (0,1,*): 4-7    (0,2,*): 8-11
    //   (1,0,*): 12-15  (1,1,*): 16-19  (1,2,*): 20-23
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_3d_single_row() {
        // [0..1, 1..2, 0..4] -> (0,1,*) = [4, 5, 6, 7]
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got = r
            .to_ndarray_sub(&[0..1, 1..2, 0..4], &r.read_ctx())
            .unwrap();
        assert_eq!(got, array![[[4, 5, 6, 7]]]);
    }

    #[test]
    fn sub_read_3d_inner_block() {
        // [0..2, 1..3, 1..3]:
        //   (0,1,1)=5  (0,1,2)=6
        //   (0,2,1)=9  (0,2,2)=10
        //   (1,1,1)=17 (1,1,2)=18
        //   (1,2,1)=21 (1,2,2)=22
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got = r
            .to_ndarray_sub(&[0..2, 1..3, 1..3], &r.read_ctx())
            .unwrap();
        assert_eq!(got, array![[[5, 6], [9, 10]], [[17, 18], [21, 22]]]);
    }

    #[test]
    fn sub_read_3d_second_slab() {
        // [1..2, 0..3, 0..4] -> all of the second "slab" = [12..24]
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got = r
            .to_ndarray_sub(&[1..2, 0..3, 0..4], &r.read_ctx())
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([1, 3, 4], (12u8..24).collect()).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Multi-block: block boundaries in the original array
    // -----------------------------------------------------------------------

    #[test]
    fn multiblock_1d_full_read_reshape_to_2d() {
        // 12 elements, block_size=4 -> 3 blocks; reshape to [3, 4]
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], u8s(12)).unwrap()
        );
    }

    #[test]
    fn multiblock_1d_sub_read_crosses_block_boundary() {
        // block_size=4, reshape to [3, 4]; read row 0 of new shape
        // flat [0..4) = one full original block -> [0, 1, 2, 3]
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[3, 4]);
        let got = r.to_ndarray_sub(&[0..1, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3]]);
    }

    #[test]
    fn multiblock_1d_to_2x6_sub_read_crosses_block_boundary() {
        // block_size=4, reshape to [2, 6]:
        //   row 0: [0..6) spans block0 (0-3) and part of block1 (4-5)
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray_sub(&[0..1, 0..6], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn multiblock_1d_to_2x6_full_read() {
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 6], u8s(12)).unwrap()
        );
    }

    #[test]
    fn multiblock_2d_orig_reshape_to_1d() {
        // orig [3, 4] with block_shape [2, 2], flatten to [12]
        let a = make2d(u8s(12), 3, 4, &[2, 2]);
        let r = a.reshape_view(&[12]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([12], u8s(12)).unwrap());
    }

    #[test]
    fn multiblock_2d_orig_reshape_sub_read() {
        // orig [3, 4] with block_shape [2, 2], reshape to [2, 6]
        // sub-read row 1: flat [6..12) -> [6, 7, 8, 9, 10, 11]
        let a = make2d(u8s(12), 3, 4, &[2, 2]);
        let r = a.reshape_view(&[2, 6]);
        let got = r.to_ndarray_sub(&[1..2, 0..6], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[6, 7, 8, 9, 10, 11]]);
    }

    #[test]
    fn multiblock_small_blocks_reshape_3d() {
        // 24 elements, block_size=3 -> 8 blocks; reshape to [2, 3, 4]
        let a = make1d(u8s(24), 3);
        let r = a.reshape_view(&[2, 3, 4]);
        let got = r
            .to_ndarray_sub(&[0..2, 0..3, 0..4], &r.read_ctx())
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], u8s(24)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Chained: reshape -> reshape
    // -----------------------------------------------------------------------

    #[test]
    fn chained_reshape_1d_to_2d_to_3d() {
        let a = make1d(u8s(24), 24);
        let r1 = a.reshape_view(&[4, 6]);
        let r2 = r1.reshape_view(&[2, 3, 4]);
        let got = r2.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], u8s(24)).unwrap()
        );
    }

    #[test]
    fn chained_reshape_then_flatten() {
        let a = make1d(i32s(12), 12);
        let r1 = a.reshape_view(&[3, 4]);
        let r2 = r1.reshape_view(&[12]);
        let got = r2.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([12], i32s(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Verify flat element order is preserved
    // (reshape cannot change the value at flat index i)
    // -----------------------------------------------------------------------

    #[test]
    fn flat_order_preserved_4x3_vs_3x4() {
        // Both reshape [12] -> [4,3] and [3,4] must yield same flat sequence
        let a12 = make1d(u8s(12), 12);
        let r43 = a12.as_ref().reshape_view(&[4, 3]);
        let r34 = a12.as_ref().reshape_view(&[3, 4]);

        let flat_43 = r43.reshape_view(&[12]).to_ndarray().unwrap();
        let flat_34 = r34.reshape_view(&[12]).to_ndarray().unwrap();
        assert_eq!(flat_43, flat_34);
        assert_eq!(
            flat_43,
            ndarray::Array::from_shape_vec([12], u8s(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary input shape, arbitrary output factorization
    // -----------------------------------------------------------------------

    fn divisors(n: usize) -> Vec<usize> {
        let mut divs = Vec::new();
        let mut d = 1;
        while d * d <= n {
            if n % d == 0 {
                divs.push(d);
                if d != n / d {
                    divs.push(n / d);
                }
            }
            d += 1;
        }
        divs.sort_unstable();
        divs
    }

    fn reshape_strategy<T>(
    ) -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact<Ty<T>, DimDyn>>, Vec<u64>)>
    where
        T: ScalarStrategy,
    {
        shape_strategy()
            .prop_flat_map(|input_shape| {
                let n: usize = input_shape.iter().product();
                let n_u64 = n as u64;
                let array_strat = crate::util::carray_strategy_from_shape::<T>(
                    Just(input_shape),
                    T::any_strategy(),
                );
                // Candidate output shapes: [n] (flatten) and [d, n/d] for every divisor d.
                let out_options: Vec<Vec<u64>> = if n == 0 {
                    vec![vec![0u64]]
                } else {
                    std::iter::once(vec![n_u64])
                        .chain(
                            divisors(n)
                                .iter()
                                .map(|&d| vec![d as u64, n_u64 / d as u64]),
                        )
                        .collect()
                };
                (array_strat, proptest::sample::select(out_options))
            })
            .prop_map(|((nd, za), out_shape)| (nd, za, out_shape))
    }

    proptest::proptest! {
        #[test]
        fn proptest_reshape((nd, za, out_shape) in reshape_strategy::<i32>()) {
            // Oracle: reshape preserves flat element order.
            let expected = ndarray::Array::from_shape_vec(
                out_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                nd.iter().cloned().collect::<Vec<_>>(),
            )
            .unwrap();
            crate::util::assert_array_matches(&za.reshape_view(&out_shape), &expected);
        }
    }
}
