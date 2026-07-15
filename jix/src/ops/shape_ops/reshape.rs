use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, ensure, Result};
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{ArraySpec, ArrayStorageInfo, BlockShapeTag, BlockSize, OutBuf};
use crate::util::iter::NdIter;
use crate::util::{DimArray, IterExt};
use crate::{
    default_logical_strides, default_strides_from_iter, ArrayStorage, Dimension, IntoDimension,
};

/// Reinterprets an array with a different shape, returned by [`Array::reshape`].
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
/// Reshape is a lazy view like every other shape operation, so a single read before discarding
/// the result pays this cost only once. When the result will be read more than once, call
/// [`.compact()`](Array::compact) as soon as possible to materialize a compactly-stored array
/// with a block layout matched to the new shape.
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
/// assert_eq!(a.as_ref().reshape([6u64]).shape(), &[6]);
///
/// // (u64, u64) -> output D = Dim<2>: compiler knows the result is 2-D
/// assert_eq!(a.as_ref().reshape((3u64, 2u64)).shape(), &[3, 2]);
///
/// // &[u64] -> output D = DimDyn: ndim only known at runtime
/// let new_shape = vec![6u64];
/// assert_eq!(a.as_ref().reshape(new_shape.as_slice()).shape(), &[6]);
///
/// // Elements are the same regardless of argument style
/// assert_eq!(
///     a.reshape([6u64]).to_ndarray()?.as_slice().unwrap(),
///     &[1, 2, 3, 4, 5, 6]
/// );
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Reshape<S, D> {
    array: S,

    new_shape: D,
    spec: ArraySpecDynamic,
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
        let new_nitems = new_shape.iter().cloned().try_product();
        ensure!(
            Some(nitems) == new_nitems,
            InvalidShapeOperation,
            "cannot reshape array of shape {orig_shape:?} into shape {new_shape:?}"
        );

        let orig_logical_strides = default_logical_strides(&orig_shape);
        let new_logical_strides = default_logical_strides(&new_shape);
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

        let inner_spec = array.spec();
        let inner_block_shape = inner_spec.block_shape();
        let inner_block_shape_tag = inner_spec.block_shape_tag();
        let mut block_shape = DimArray::new();
        let mut block_shape_tag = DimArray::new();
        for dim in 0..new_shape.len() {
            if let Some(orig_dim) = same_logical_stride[dim] {
                let orig_dim = orig_dim as usize;
                let same_dim_len = orig_shape[orig_dim] == new_shape[dim];
                block_shape.push(
                    inner_block_shape[orig_dim]
                        .min(new_shape[dim].min(BlockSize::MAX as u64) as BlockSize)
                        .max(1),
                );
                block_shape_tag.push(match inner_block_shape_tag[orig_dim] {
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
                block_shape.push(1);
                block_shape_tag.push(BlockShapeTag::Any);
            }
        }
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_tag,
        };

        Ok(Self {
            new_shape: new_shape_raw,
            spec,
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

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
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
        //   2. READ the matched-dims block (using `read_range`) straight into `buf`
        //      at the unmatched dims' byte offset:
        //
        //          dst_byte_offset = sum_{unmatched new dim d} idx[d] * dst_strides[d]
        //
        //      The read targets a strided `OutBuf` over `buf[dst_byte_offset..]` whose
        //      strides (expressed in original axis order) place each matched element at
        //      its C-order position in `buf`, so no temporary buffer or extra copy is
        //      needed.
        // -----------------------------------------------------------------------
        check_get_range(self.shape(), index)?;
        let dtype = self.dtype();
        if index.iter().any(|r| r.start >= r.end) {
            buf.materialize(0, dtype);
            return Ok(());
        }
        // Write straight into the (possibly strided) destination, using its own strides: each inner
        // read scatters directly into `buf` at the unmatched dims' byte offset.
        let (dst, dst_strides) = buf.get_strided_mut::<D>(index, dtype);

        let orig_shape = S::Dimension::from_slice(self.array.shape());
        let new_shape = self.new_shape.as_slice();
        let ndim = new_shape.len();
        let orig_ndim = orig_shape.ndim();

        let orig_logical_strides = default_strides_from_iter::<S::Dimension, _>(
            orig_ndim,
            orig_shape.as_slice().iter().cloned(),
            1,
        );
        let new_logical_strides =
            default_strides_from_iter::<D, _>(ndim, new_shape.iter().cloned(), 1);
        let same_logical_stride = (0..new_shape.len())
            .scan(0, |orig_dim_idx, new_dim_idx| {
                Some(loop {
                    if *orig_dim_idx >= orig_shape.ndim() {
                        break None; // cant really happen, last dims always match, unless orig_shape.ndim()==0
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
            let mut inv = S::Dimension::vec(orig_ndim, |_| None);
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
                .as_ref()
                .iter()
                .filter(|dim| dim.is_some())
                .count()
        );

        // dims that have the same logical stride in the original and new shape can be read
        // directly; the rest we read one entry at a time and place into the output buffer.
        // Byte-strides for the inner read, in *original* axis order (so they match the read's
        // shape): a matched orig dim reuses its new dim's output stride; an unmatched orig dim has
        // extent 1 and is never stepped, so its stride is a dummy 0.
        let orig_strides = S::Dimension::vec(orig_ndim, |dim| match same_logical_stride_inv[dim] {
            Some(new_dim) => dst_strides[new_dim as usize],
            None => 0,
        });

        // We use an nd-iter over the dims that DO NOT match any original dim.
        let iteration_shape = D::vec(ndim, |dim| {
            if same_logical_stride[dim].is_some() {
                1
            } else {
                index[dim].end - index[dim].start
            }
        });
        let iter = NdIter::builder(iteration_shape).build();
        for (idx, ()) in iter {
            let read_range = S::Dimension::vec(orig_ndim, |dim| {
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

            // Read this matched-dims block straight into `buf` at the unmatched dims' byte offset;
            // the strided OutBuf places each element at its C-order position in `buf`.
            let dst_byte_offset: usize = (0..ndim)
                .filter(|&dim| same_logical_stride[dim].is_none())
                .map(|dim| idx[dim] as usize * dst_strides[dim])
                .sum();
            let mut out =
                unsafe { OutBuf::new_strided(&mut dst[dst_byte_offset..], orig_strides.as_ref()) };
            self.array
                .read_data(read_range.as_ref(), &mut out, context)?;
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
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Reshape", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Reshape<S, NewD>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        check_ndim::<NewD>(self.shape().len())?;
        let new_shape = NewD::from_slice(self.shape());

        Ok(Reshape {
            array: self.array,
            new_shape,
            spec: self.spec,
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
            spec: self.spec,
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
        let r = a.reshape([3, 4]);
        assert_eq!(r.shape(), &[3, 4]);
    }

    #[test]
    fn shape_after_reshape_2d_to_1d() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape(12);
        assert_eq!(r.shape(), &[12]);
    }

    #[test]
    fn shape_after_reshape_2d_to_3d() {
        let a = make2d(u8s(24), 4, 6, &[4, 6]);
        let r = a.reshape([2, 3, 4]);
        assert_eq!(r.shape(), &[2, 3, 4]);
    }

    #[test]
    fn dtype_preserved_after_reshape() {
        use crate::dtype::Dtyped;
        let a = make1d(i32s(6), 6);
        let r = a.reshape([2, 3]);
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
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_1d_to_2d_4x3_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape([4, 3]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([4, 3], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_1d_to_2d_2x6_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape([2, 6]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 6], u8s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_1d_to_2d_6x2_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape([6, 2]);
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
        let r = a.reshape(12);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([12], i32s(12)).unwrap());
    }

    #[test]
    fn full_read_2d_to_1d_flatten_non_square() {
        let a = make2d(u8s(20), 4, 5, &[4, 5]);
        let r = a.reshape(20);
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
        let r = a.reshape([2, 6]);
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
        let r = a.reshape([3, 4]);
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
        let r = a.reshape([2, 3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], u8s(24)).unwrap()
        );
    }

    #[test]
    fn full_read_3d_to_1d_flatten() {
        let a = make3d(i32s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape(24);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([24], i32s(24)).unwrap());
    }

    #[test]
    fn full_read_3d_to_2d() {
        // [2, 3, 4] -> [6, 4]
        let a = make3d(u8s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape([6, 4]);
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
        let r = a.reshape([2, 3, 4]);
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
        let r = a.reshape([2, 12]);
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
        let r = a.reshape(8);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([8], i32s(8)).unwrap());
    }

    #[test]
    fn full_read_same_shape_2d() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape([3, 4]);
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
        let r = a.reshape(1);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, array![42u8]);
    }

    // -----------------------------------------------------------------------
    // Full read: various dtypes
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_dtype_i32() {
        let a = make1d(i32s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], i32s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_dtype_f32() {
        let a = make1d(f32s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], f32s(12)).unwrap()
        );
    }

    #[test]
    fn full_read_dtype_f64() {
        let a = make1d(f64s(12), 12);
        let r = a.reshape([4, 3]);
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
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[0..1, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3]]);
    }

    #[test]
    fn sub_read_middle_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[1..2, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[4, 5, 6, 7]]);
    }

    #[test]
    fn sub_read_last_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[2..3, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[8, 9, 10, 11]]);
    }

    #[test]
    fn sub_read_first_two_rows() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[0..2, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3], [4, 5, 6, 7]]);
    }

    #[test]
    fn sub_read_first_two_columns() {
        // [0..3, 0..2] -> rows 0-2, cols 0-1
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[0..3, 0..2], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1], [4, 5], [8, 9]]);
    }

    #[test]
    fn sub_read_last_two_columns() {
        // [0..3, 2..4] -> rows 0-2, cols 2-3
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[0..3, 2..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[2, 3], [6, 7], [10, 11]]);
    }

    #[test]
    fn sub_read_inner_2x2() {
        // [1..3, 1..3]
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[1..3, 1..3], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[5, 6], [9, 10]]);
    }

    #[test]
    fn sub_read_single_element_center() {
        // [1..2, 2..3] -> element at (1,2) = 6
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[1..2, 2..3], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[6]]);
    }

    #[test]
    fn sub_read_single_element_corner() {
        // [2..3, 3..4] -> element at (2,3) = 11
        let a = make1d(u8s(12), 12);
        let r = a.reshape([3, 4]);
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
        let r = a.reshape(12);
        let got = r.to_ndarray_sub(&[3..9], &r.read_ctx()).unwrap();
        assert_eq!(got, array![3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sub_read_flatten_partial_first_row() {
        // Flat [2..6) spans the last 2 of row-0 and first 2 of row-1 (in orig [3,4])
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape(12);
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
        let r = a.reshape([2, 6]);
        let got = r.to_ndarray_sub(&[0..1, 1..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[1, 2, 3]]);
    }

    #[test]
    fn sub_read_2x6_both_rows_partial_cols() {
        // [0..2, 2..5] -> rows 0-1, cols 2-4
        // row 0: [2, 3, 4]; row 1: [8, 9, 10]
        let a = make1d(u8s(12), 12);
        let r = a.reshape([2, 6]);
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
        let r = a.reshape([2, 3, 4]);
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
        let r = a.reshape([2, 3, 4]);
        let got = r
            .to_ndarray_sub(&[0..2, 1..3, 1..3], &r.read_ctx())
            .unwrap();
        assert_eq!(got, array![[[5, 6], [9, 10]], [[17, 18], [21, 22]]]);
    }

    #[test]
    fn sub_read_3d_second_slab() {
        // [1..2, 0..3, 0..4] -> all of the second "slab" = [12..24]
        let a = make1d(u8s(24), 24);
        let r = a.reshape([2, 3, 4]);
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
        let r = a.reshape([3, 4]);
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
        let r = a.reshape([3, 4]);
        let got = r.to_ndarray_sub(&[0..1, 0..4], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3]]);
    }

    #[test]
    fn multiblock_1d_to_2x6_sub_read_crosses_block_boundary() {
        // block_size=4, reshape to [2, 6]:
        //   row 0: [0..6) spans block0 (0-3) and part of block1 (4-5)
        let a = make1d(u8s(12), 4);
        let r = a.reshape([2, 6]);
        let got = r.to_ndarray_sub(&[0..1, 0..6], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn multiblock_1d_to_2x6_full_read() {
        let a = make1d(u8s(12), 4);
        let r = a.reshape([2, 6]);
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
        let r = a.reshape(12);
        let got = r.to_ndarray().unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([12], u8s(12)).unwrap());
    }

    #[test]
    fn multiblock_2d_orig_reshape_sub_read() {
        // orig [3, 4] with block_shape [2, 2], reshape to [2, 6]
        // sub-read row 1: flat [6..12) -> [6, 7, 8, 9, 10, 11]
        let a = make2d(u8s(12), 3, 4, &[2, 2]);
        let r = a.reshape([2, 6]);
        let got = r.to_ndarray_sub(&[1..2, 0..6], &r.read_ctx()).unwrap();
        assert_eq!(got, array![[6, 7, 8, 9, 10, 11]]);
    }

    #[test]
    fn multiblock_small_blocks_reshape_3d() {
        // 24 elements, block_size=3 -> 8 blocks; reshape to [2, 3, 4]
        let a = make1d(u8s(24), 3);
        let r = a.reshape([2, 3, 4]);
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
        let r1 = a.reshape([4, 6]);
        let r2 = r1.reshape([2, 3, 4]);
        let got = r2.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], u8s(24)).unwrap()
        );
    }

    #[test]
    fn chained_reshape_then_flatten() {
        let a = make1d(i32s(12), 12);
        let r1 = a.reshape([3, 4]);
        let r2 = r1.reshape(12);
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
        let r43 = a12.as_ref().reshape([4, 3]);
        let r34 = a12.as_ref().reshape([3, 4]);

        let flat_43 = r43.reshape(12).to_ndarray().unwrap();
        let flat_34 = r34.reshape(12).to_ndarray().unwrap();
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
            crate::util::assert_array_matches(&za.reshape(&out_shape), &expected);
        }
    }
}
