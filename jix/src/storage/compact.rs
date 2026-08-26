//! Block-compressed nd-array storage backends.
//!
//! This module provides the three concrete [`ArrayStorage`] implementations that store array
//! data as independently compressed nd-blocks:
//!
//! - [`Compact`] - heap-allocated; the standard in-memory storage.
//! - [`CompactBorrowed`] - borrows its data from an existing byte buffer.
//! - [`CompactMmap`] - memory-mapped; the OS pages data from disk on demand.
//!
//! All three are thin wrappers around [`ArrayBlockTableStorageBase`], which contains the
//! actual nd-array logic and delegates 1D block I/O to [`BlockTable`](crate::storage::block::BlockTable).

use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage};
use crate::storage::params::{ArraySpecFlags, ArraySpecOwned};
use crate::storage::{check_out_buf, materialize_out_buf, ArraySpec, ElementType, StridedBuf};
use crate::util::iter::NdIter;
use crate::util::{calc_block_end, NdCopier};
use crate::{default_strides, ArrayParams, ArrayStorage, Dim, DimDyn, DimVec, Dimension};

/// Heap-allocated, block-compressed nd-array storage.
///
/// The array data is divided into fixed-size nd-blocks and each block is independently
/// compressed. All compressed bytes are held in a heap-allocated buffer owned by
/// this struct.
///
/// `Compact<ET, D>` has two type parameters:
///
/// - **`ET: ElementType`** - compile-time element type, either [`Ty<T>`](crate::Ty)
///   (element type known at compile time) or [`TypeDyn`](crate::TypeDyn) (runtime only).
///   Arrays constructed from typed sources carry `Ty<_>` automatically; arrays loaded from disk
///   carry `TypeDyn`.
///
/// - **`D: Dimension`** - compile-time dimension, either [`Dim<N>`](crate::Dim) (statically known
///   ndim) or [`DimDyn`](crate::DimDyn) (runtime only).
///
/// Use [`Array::into_dim`](crate::Array::into_dim) to convert between `D` variants in-place, or
/// [`Array::into_typed`](crate::Array::into_typed) to assert a concrete element type and go from
/// `TypeDyn` to `Ty<T>`.
///
/// Created by [`Array::compact_ndarray`](crate::Array::compact_ndarray), [`Array::compact`](crate::Array::compact)
/// and their variants or by deserializing an archive file. The memory-mapped equivalent is [`CompactMmap`].
///
/// # Block codec pipeline
///
/// Each block of array data is encoded through a two-stage pipeline before being stored:
///
/// ```text
/// raw block bytes
///     |
///     v
/// [ Filter 0 ] -> [ Filter 1 ] -> ...  (optional pre-compression transforms)
///     |
///     v
/// [ Codec (e.g. Zstd) ]                (lossless compression)
///     |
///     v
/// stored block bytes
/// ```
///
/// Decoding reverses the pipeline exactly: decompress first, then apply the filters in reverse
/// order.
///
/// ## Configuration
///
/// The codec, compression level, and filter pipeline are chosen at write time through
/// [`ArrayParams`](crate::ArrayParams): [`codec`](crate::ArrayParams::codec),
/// [`level`](crate::ArrayParams::level), and [`filters`](crate::ArrayParams::filters). They are
/// stored alongside the data so the correct decoder is reconstructed automatically on read - readers
/// never need to know the settings in advance.
///
/// ## Filters
///
/// [`Filter`](crate::Filter)s are byte-level transforms that rearrange element data into a layout
/// that compresses more efficiently, then reverse the transform after decompression. For most
/// numeric workloads [`Filter::ByteShuffle`](crate::Filter::ByteShuffle) is the right default.
/// [`Filter::BitShuffle`](crate::Filter::BitShuffle) can squeeze out more compression for
/// low-entropy data at higher CPU cost.
///
/// ## Read context
///
/// [`ReadContext`](crate::ReadContext) holds a long-lived decompressor instance and reusable scratch
/// buffers. Create one per thread and pass it to every read call to amortize initialization overhead
/// across many block reads. The preferred way to obtain one is
/// [`Array::read_ctx()`](crate::Array::read_ctx).
pub struct Compact<ET, D>(
    pub(crate) ArrayBlockTableStorageBase<crate::storage::block::Owned, ET, D>,
);

/// Borrowed, block-compressed nd-array storage.
///
/// Same layout as [`Compact`] but borrows its compressed bytes from an existing byte slice
/// instead of owning them. Used internally when constructing temporary views into a
/// pre-encoded buffer.
pub struct CompactBorrowed<'a, ET, D>(
    pub(crate) ArrayBlockTableStorageBase<crate::storage::block::Borrowed<'a>, ET, D>,
);

/// Memory-mapped, block-compressed nd-array storage.
///
/// Same layout as [`Compact`] but the compressed bytes are served from a memory-mapped file.
/// The OS pages data into memory on demand, avoiding a full file read at open time.
/// Arrays loaded via mmap always start as `CompactMmap<TypeDyn, DimDyn>` because both the
/// element type and ndim are read from the file header at runtime. The `T` and `D` parameters
/// carry the same semantics as in [`Compact<ET, D>`](Compact).
///
/// Created by [`Array::read_from_file_mmap`](crate::Array::read_from_file_mmap).
///
/// See [`Compact`] for more details.
pub struct CompactMmap<ET, D>(
    pub(crate) ArrayBlockTableStorageBase<crate::storage::block::Mmap, ET, D>,
);
macro_rules! impl_array_storage {
    ($ty:ident < $($lt:lifetime,)? ET, D >) => {
        impl<$($lt,)? ET, D> ArrayStorage for $ty<$($lt,)? ET, D>
        where
            ET: crate::ElementType,
            D: crate::Dimension,
        {
            type ElementType = ET;
            type Dimension = D;

            #[inline(always)]
            fn read_data<'rd>(
                &'rd self,
                index: &[Range<u64>],
                context: &'rd ReadContext,
                out: Option<&'rd mut crate::storage::StridedBuf<'_>>,
            ) -> Result<crate::storage::StridedBuf<'rd>> {
                self.0.read_data(index, context, out)
            }

            #[inline(always)]
            fn shape(&self) -> &[u64] {
                self.0.shape()
            }
            #[inline(always)]
            fn dtype(&self) -> &Dtype {
                self.0.blocks.dtype()
            }

            #[inline(always)]
            fn spec(&self) -> ArraySpec<'_> {
                self.0.spec.as_ref()
            }

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new("Compact")
            }

            fn as_compact(&self) -> Option<CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
                Some(CompactBorrowed(ArrayBlockTableStorageBase {
                    blocks: self.0.blocks.as_ref(),
                    shape: self.0.shape.clone(),
                    spec: self.0.spec.clone(),
                }))
            }

            type DimensionChange<NewD: Dimension> = $ty<$($lt,)? ET, NewD>;
            #[inline]
            fn dimension_change<NewD: Dimension>(self) -> Result<Self::DimensionChange<NewD>> {
                Ok($ty(self.0.dimension_change()?))
            }

            type ElementTypeChange<NewET: ElementType> = $ty<$($lt,)? NewET, D>;
            #[inline]
            fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
                Ok($ty(self.0.element_type_change()?))
            }
        }
    };
}
impl_array_storage!(Compact<ET, D>);
impl_array_storage!(CompactBorrowed<'a, ET, D>);
impl_array_storage!(CompactMmap<ET, D>);

/// Nd-array layer on top of [`BlockTable<S>`](crate::storage::block::BlockTable).
///
/// Bridges the nd-array world of [`ArrayStorage`] and the 1D block world of [`BlockTable`].
/// The array's nd-blocks are stored in row-major order in the `BlockTable`: an nd-block at
/// grid position `(b0, b1, ..., bn)` has 1D index
/// `b0 * block_grid_shape[1] * ... * block_grid_shape[n-1] * block_grid_shape[n] + ... + bn`.
///
/// The block shape is stored in `blocks_layout.block_shape_hint` (items per dimension, not
/// bytes). The number of blocks per dimension is `block_grid_shape[d] = ceil(shape[d] / block_shape[d])`.
/// All blocks in the `BlockTable` are full, so `shape[d]` must be a multiple of `block_shape[d]`
/// for all `d`.
///
/// `encoder_params` and `decoder_params` are kept here - not in `BlockTable` - so that
/// `ArrayStorage::spec` can propagate them through lazy view operations and `compact_with`.
pub(crate) struct ArrayBlockTableStorageBase<S, ET, D>
where
    S: BlockTableStorage,
{
    pub(crate) blocks: BlockTable<S, ET>,
    shape: D,

    spec: ArraySpecOwned,
}
impl<S, ET, D> ArrayBlockTableStorageBase<S, ET, D>
where
    S: BlockTableStorage,
{
    /// Construct the nd-array layer from a pre-built `BlockTable`.
    ///
    /// `blocks_layout.block_shape_hint` must match the block geometry already encoded in
    /// `blocks`. `block_grid_shape` is derived from `shape` and the block shape.
    pub(crate) fn new(blocks: BlockTable<S, ET>, shape: D, params: ArrayParams) -> Result<Self>
    where
        ET: ElementType,
        D: Dimension,
    {
        let shape_slice = shape.as_slice();
        let mut spec = params.into_spec(
            shape_slice,
            blocks.dtype(),
            ArraySpecFlags::new().set_compact(),
        )?;
        // Reading a compact element is more expensive than reading a plain element (1).
        spec.dynamic_mut().element_cost = 8.0;
        Ok(Self {
            blocks,
            shape,
            spec,
        })
    }

    /// Returns the nd-block shape (items per dimension) used by this storage.
    #[inline(always)]
    pub(crate) fn block_shape(&self) -> &[BlockSize] {
        self.spec.as_ref().block_shape()
    }

    #[inline(always)]
    pub(crate) fn shape(&self) -> &[u64]
    where
        D: Dimension,
    {
        self.shape.as_slice()
    }

    /// Read a rectangular sub-region of the nd-array into `buf`.
    ///
    /// Identifies which nd-blocks overlap `index`, decompresses each in turn, and copies the
    /// relevant portion of each block into the correct position in `buf`.
    ///
    /// ## Algorithm
    ///
    /// 1. For each dimension, compute the range of block indices that overlap `index`, and track
    ///    two flags across all dimensions:
    ///    `b_begin[d] = index[d].start / block_shape[d]`,
    ///    `b_end[d]   = ceil(index[d].end / block_shape[d])`;
    ///    `is_single_block` - every dimension touches exactly one block (`b_begin[d] + 1 == b_end[d]`);
    ///    `is_aligned` - `index[d]` starts and ends on block boundaries in every dimension.
    ///
    /// 2. **Fast path (aligned single block)** - if `is_single_block && is_aligned`, the requested
    ///    region *is* one whole block, so compute the 1D block index via row-major flattening and
    ///    call [`BlockTable::read_block`](crate::storage::block::BlockTable::read_block) directly
    ///    into `buf`. No temporary buffer and no copy needed.
    ///
    /// 3. **Fast path (single block)** - if `is_single_block` but the region is not block-aligned,
    ///    only one block is touched. Decompress it once into `tmp_buf`, then do a single strided
    ///    `nd_copy` of the requested sub-region into `buf`, skipping the `NdIter` setup below.
    ///
    /// 4. **General path** - iterate over every nd-block in the touched range using `NdIter`,
    ///    extended with two side-cars:
    ///    - `nd_iter_ext_logical_global_index` - yields the 1D `BlockTable` index for each
    ///      block (row-major flattening of the nd block index within `block_grid_shape`).
    ///    - `NdIterExtBlockOffsetSize` - yields `(block_inner_offset, block_size)`: the
    ///      element-space start offset within the block and the active nd extent, both clipped
    ///      to the requested `index` at the array boundaries.
    ///
    ///    \
    ///    For each block:
    ///    - Decompress the full block into `tmp_buf` (a scratch buffer from `context`,
    ///      sized for one full block, reused across iterations).
    ///    - Compute `active_start`: byte offset into `tmp_buf` of the active region's first
    ///      element, using the block's row-major strides and `block_inner_offset`.
    ///    - Compute `out_start`: byte offset into `buf` where this region's first element
    ///      belongs, using the output array's row-major strides and the element-space
    ///      position of the active region relative to `index`.
    ///    - Call `nd_copy` to scatter the active sub-region from `tmp_buf` into `buf`,
    ///      respecting both strides.
    #[inline]
    fn read_data<'rd>(
        &'rd self,
        index: &[Range<u64>],
        context: &'rd ReadContext,
        out: Option<&'rd mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'rd>>
    where
        ET: ElementType,
        D: Dimension,
    {
        let shape = self.shape();
        check_get_range(shape, index)?;
        check_out_buf(out.as_deref(), shape)?;
        let dtype = self.blocks.dtype();
        let out_shape = D::vec(shape.len(), |d| (index[d].end - index[d].start) as usize);
        let nitems = out_shape.as_ref().iter().product::<usize>();
        let mut out = materialize_out_buf(out, context, out_shape.as_ref(), dtype);
        if nitems == 0 {
            return Ok(out);
        }
        let is_contiguous = out.is_contiguous(out_shape.as_ref(), dtype);
        let (out_buf, out_strides) = out.data_mut();

        let ndim = shape.len();
        let block_shape = self.block_shape();
        assert_eq!(ndim, block_shape.len());

        let mut b_range = D::vec(ndim, |_| 0..0);
        let mut is_single_block = true; // every dimension touches exactly one block.
        let mut is_block_aligned = true; // the requested region starts and ends on block boundaries in every dimension.
        for dim in 0..ndim {
            let b = block_shape[dim] as u64;
            let (i_start, i_end) = (index[dim].start, index[dim].end);
            let (b_begin, b_end) = (i_start / b, calc_block_end(i_start, i_end, b));
            b_range[dim] = b_begin..b_end;
            is_single_block &= b_begin + 1 == b_end;
            is_block_aligned &= i_start.is_multiple_of(b) && i_end.is_multiple_of(b);
        }

        // Row-major-flattened 1D block index
        let single_block_idx = is_single_block.then(|| {
            (0..ndim).fold(0u64, |blk_idx, dim| {
                let block_grid_len = shape[dim].div_ceil(block_shape[dim] as u64);
                blk_idx * block_grid_len + b_range[dim].start
            })
        });

        // Fast path for an aligned single-block read into a contiguous destination
        if let Some(single_block_idx) = single_block_idx
            && is_block_aligned
            && is_contiguous
        {
            let buf = &mut out_buf[..nitems * dtype.itemsize() as usize];
            self.blocks.read_block(single_block_idx, buf, context)?;
        } else {
            self.read_data_slow(index, out_buf, out_strides, context, single_block_idx)?;
        }
        Ok(out)
    }

    fn read_data_slow(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        out_strides: &[usize],
        context: &ReadContext,
        single_block_idx: Option<u64>,
    ) -> Result<()>
    where
        ET: ElementType,
        D: Dimension,
    {
        let read_fn = if D::NDIM.is_some() {
            Self::read_data_slow_impl::<D>
        } else {
            match self.shape().len() {
                1 => Self::read_data_slow_impl::<Dim<1>>,
                2 => Self::read_data_slow_impl::<Dim<2>>,
                3 => Self::read_data_slow_impl::<Dim<3>>,
                4 => Self::read_data_slow_impl::<Dim<4>>,
                _ => Self::read_data_slow_impl::<DimDyn>,
            }
        };
        read_fn(self, index, buf, out_strides, context, single_block_idx)
    }

    #[inline(never)]
    fn read_data_slow_impl<ActualD: Dimension>(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        out_strides: &[usize],
        context: &ReadContext,
        single_block_idx: Option<u64>,
    ) -> Result<()>
    where
        ET: ElementType,
        D: Dimension,
    {
        let shape = self.shape();
        let ndim = shape.len();
        let block_shape = self.block_shape();
        assert_eq!(ndim, block_shape.len());

        let dtype = self.blocks.dtype();
        let itemsize = dtype.itemsize() as usize;
        let out_shape = ActualD::vec(ndim, |dim| (index[dim].end - index[dim].start) as usize);
        let block_shape_u64 = ActualD::vec(ndim, |dim| block_shape[dim] as u64);
        let block_strides = default_strides(&block_shape_u64, itemsize);
        let copier = NdCopier::new(dtype);

        // Pre-allocate a buffer large enough for a full block.
        let full_buf_len =
            block_shape_u64.as_ref().iter().copied().product::<u64>() as usize * itemsize;
        let mut tmp_buf = context.allocate_buf(full_buf_len, dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();

        // Fast path for (unaligned) single-block read
        if let Some(single_block_idx) = single_block_idx {
            self.blocks.read_block(single_block_idx, tmp_buf, context)?;

            // Byte offset into `tmp_buf` of the requested region's first element.
            let active_start = (0..ndim)
                .map(|dim| {
                    let inner_offset = index[dim].start % block_shape_u64[dim];
                    inner_offset as usize * block_strides[dim]
                })
                .sum::<usize>();
            let src = unsafe { tmp_buf.get_unchecked(active_start..) };

            unsafe {
                copier.copy(
                    src,
                    buf,
                    out_shape.as_ref(),
                    block_strides.as_ref(),
                    out_strides,
                    dtype,
                )
            };
            return Ok(());
        }

        // A block can be decoded straight into `buf` - skipping `tmp_buf` and the
        // `nd_copy` scatter - when the destination is C-contiguous and the block is a single
        // contiguous run
        let read_into_out = {
            let c_strides = default_strides(&out_shape, itemsize);
            let out_buf_contiguous =
                (0..ndim).all(|d| out_shape[d] == 1 || out_strides[d] == c_strides[d]);
            let lead_dim = (0..ndim).find(|&d| block_shape[d] > 1);
            let inner_full_width = lead_dim
                .is_none_or(|k| (k + 1..ndim).all(|d| out_shape[d] == block_shape[d] as usize));
            out_buf_contiguous && inner_full_width
        };

        // Block-space begin/end for NdIter.
        let block_begin = ActualD::vec(ndim, |dim| index[dim].start / block_shape_u64[dim]);
        let block_end = ActualD::vec(ndim, |dim| {
            calc_block_end(index[dim].start, index[dim].end, block_shape_u64[dim])
        });
        let block_grid_shape =
            ActualD::vec(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));

        let block_iter = NdIter::builder_with_begin(block_begin, block_end)
            .with_logical_global_index_ext(block_grid_shape.as_ref())
            .with_block_offset_size_ext(
                &ActualD::vec(ndim, |dim| index[dim].start),
                &ActualD::vec(ndim, |dim| index[dim].end),
                block_shape_u64.clone(),
            )
            .build();
        for (block_idx, (block_global_id, (block_inner_offset, block_size))) in block_iter {
            // Map the active region's start to its position in the output array.
            let out_start = (0..ndim)
                .map(|dim| {
                    let full_idx = block_idx[dim] * block_shape_u64[dim] + block_inner_offset[dim];
                    let out_idx = full_idx - index[dim].start;
                    out_idx as usize * out_strides[dim]
                })
                .sum::<usize>();

            let direct = read_into_out && (0..ndim).all(|d| block_size[d] == block_shape_u64[d]);
            let read_dst = if direct {
                &mut buf[out_start..out_start + full_buf_len]
            } else {
                &mut tmp_buf[..]
            };

            self.blocks.read_block(block_global_id, read_dst, context)?;

            if !direct {
                let active_start = (0..ndim)
                    .map(|dim| block_inner_offset[dim] as usize * block_strides[dim])
                    .sum::<usize>();
                let src = unsafe { tmp_buf.get_unchecked(active_start..) };
                let dst = unsafe { buf.get_unchecked_mut(out_start..) };

                unsafe {
                    copier.copy(
                        src,
                        dst,
                        ActualD::vec(ndim, |dim| block_size[dim] as usize).as_ref(),
                        block_strides.as_ref(),
                        out_strides,
                        dtype,
                    )
                };
            }
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn element_type_change<NewET: ElementType>(
        self,
    ) -> Result<ArrayBlockTableStorageBase<S, NewET, D>>
    where
        ET: ElementType,
    {
        Ok(ArrayBlockTableStorageBase {
            blocks: self.blocks.element_type_change()?,
            shape: self.shape,
            spec: self.spec,
        })
    }

    #[inline]
    pub(crate) fn dimension_change<NewD: Dimension>(
        self,
    ) -> Result<ArrayBlockTableStorageBase<S, ET, NewD>>
    where
        D: Dimension,
    {
        check_ndim::<NewD>(self.shape().len())?;
        let shape = NewD::from_slice(self.shape());
        Ok(ArrayBlockTableStorageBase {
            blocks: self.blocks,
            shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::storage::StridedBuf;
    use crate::{Array, ArrayParams, ArrayStorage};

    /// Read `index` from a compact `i32` array both into a plain contiguous buffer and into a
    /// *strided* destination whose inner spacing is doubled (a one-element gap between consecutive
    /// slots, propagated outward). Assert the strided slots match the contiguous read exactly and
    /// that the gap elements are left untouched - i.e. `read_data` scatters straight into the
    /// caller's byte-strides instead of staging through a contiguous scratch and scattering at the
    /// end.
    fn check_strided_matches_contiguous(
        shape: &[usize],
        block_shape: &[u32],
        index: &[Range<u64>],
    ) {
        let n: usize = shape.iter().product();
        let nd = ndarray::ArrayD::from_shape_vec(
            shape.to_vec(),
            (0..n as i32).map(|x| x * 7 - 11).collect(),
        )
        .unwrap();
        let mut params = ArrayParams::new();
        params.block_shape(block_shape);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let ctx = za.read_ctx();
        let storage = za.into_storage();

        let ndim = index.len();
        let out_shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<Vec<_>>();
        let nitems: usize = out_shape.iter().product();

        // Reference: a plain contiguous read.
        let mut expected = vec![0i32; nitems.max(1)];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(expected.as_mut_ptr().cast::<u8>(), nitems * 4)
            };
            let c = crate::util::default_strides_slice(&out_shape, 4);
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, c.as_ref()) };
            storage.read_data(index, &ctx, Some(&mut out)).unwrap();
        }

        // Strided destination: element strides with a doubled inner spacing so every other element
        // is an untouched gap. Backed by a `Vec<i32>` for 4-alignment.
        let mut el_strides = vec![0usize; ndim];
        if ndim > 0 {
            el_strides[ndim - 1] = 2;
            for d in (0..ndim - 1).rev() {
                el_strides[d] = el_strides[d + 1] * out_shape[d + 1];
            }
        }
        let span = (0..ndim)
            .map(|d| out_shape[d].saturating_sub(1) * el_strides[d])
            .sum::<usize>()
            + 1;
        const SENTINEL: i32 = i32::MIN;
        let mut backing = vec![SENTINEL; span.max(1)];
        let byte_strides = el_strides.iter().map(|&s| s * 4).collect::<Vec<_>>();
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), backing.len() * 4)
            };
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, &byte_strides) };
            storage.read_data(index, &ctx, Some(&mut out)).unwrap();
        }

        // Compare every logical output element to its strided slot, tracking which backing
        // elements are slots so the rest can be asserted untouched.
        let mut is_slot = vec![false; backing.len()];
        let mut coord = vec![0usize; ndim];
        for flat in 0..nitems {
            let mut rem = flat;
            for d in (0..ndim).rev() {
                coord[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            let slot = (0..ndim).map(|d| coord[d] * el_strides[d]).sum::<usize>();
            assert_eq!(backing[slot], expected[flat], "coord {coord:?}");
            is_slot[slot] = true;
        }
        for (i, &v) in backing.iter().enumerate() {
            if !is_slot[i] {
                assert_eq!(v, SENTINEL, "gap at element {i} was overwritten");
            }
        }
    }

    #[test]
    fn read_into_strided_output_multi_block() {
        // A full read spanning many blocks -> the general (multi-block) path scatters each block
        // into the strided destination at its own `out_start`.
        check_strided_matches_contiguous(&[6, 8], &[3, 2], &[0..6, 0..8]);
    }

    #[test]
    fn read_into_strided_output_unaligned_single_block() {
        // A sub-region inside a single block -> the single-block slow branch scatters once.
        check_strided_matches_contiguous(&[4, 4], &[4, 4], &[1..3, 1..3]);
    }

    #[test]
    fn read_into_strided_output_aligned_single_block() {
        // An aligned single block whose destination is strided: the direct `read_block` fast path
        // (which needs a contiguous buffer) is skipped and the slow branch scatters instead.
        check_strided_matches_contiguous(&[4, 4], &[2, 2], &[0..2, 0..2]);
    }

    #[test]
    fn read_into_strided_output_3d_multi_block() {
        check_strided_matches_contiguous(&[4, 4, 4], &[2, 2, 2], &[0..4, 0..4, 0..4]);
    }

    /// Read `index` from a compact `i32` array into a plain *contiguous* destination and compare to
    /// the source values over that range. This drives `read_data_slow`'s direct-into-`buf` path
    /// (for full blocks spanning the full output width in the inner dims) and its `tmp_buf` +
    /// `nd_copy` fallback (for clipped/boundary blocks) - the result must be identical either way.
    fn check_contiguous_read(shape: &[usize], block_shape: &[u32], index: &[Range<u64>]) {
        let n: usize = shape.iter().product();
        let nd = ndarray::ArrayD::from_shape_vec(
            shape.to_vec(),
            (0..n as i32).map(|x| x * 13 - 7).collect(),
        )
        .unwrap();
        let mut params = ArrayParams::new();
        params.block_shape(block_shape);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let ctx = za.read_ctx();
        let storage = za.into_storage();

        let ndim = index.len();
        let out_shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<Vec<_>>();
        let nitems: usize = out_shape.iter().product();

        let mut got = vec![0i32; nitems.max(1)];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(got.as_mut_ptr().cast::<u8>(), nitems * 4)
            };
            let c = crate::util::default_strides_slice(&out_shape, 4);
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, c.as_ref()) };
            storage.read_data(index, &ctx, Some(&mut out)).unwrap();
        }

        // Expected: the source values over `index`, in row-major order.
        let mut coord = vec![0usize; ndim];
        for (flat, &g) in got[..nitems].iter().enumerate() {
            let mut rem = flat;
            for d in (0..ndim).rev() {
                coord[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            let nd_idx = (0..ndim)
                .map(|d| coord[d] + index[d].start as usize)
                .collect::<Vec<_>>();
            assert_eq!(g, nd[ndarray::IxDyn(&nd_idx)], "coord {coord:?}");
        }
    }

    #[test]
    fn contiguous_read_multi_block_dim0_full_inner() {
        // Blocks stack only along dim 0 and each spans the full inner width -> every block is read
        // straight into `buf` (the direct path), no scratch/copy.
        check_contiguous_read(&[8, 4], &[2, 4], &[0..8, 0..4]);
    }

    #[test]
    fn contiguous_read_clipped_dim0_boundaries() {
        // `index[0]` is not block-aligned, so the first/last dim-0 blocks are clipped (partial) and
        // must use the `tmp_buf` fallback, while the full interior blocks are read directly. Mixed.
        check_contiguous_read(&[8, 4], &[2, 4], &[1..7, 0..4]);
    }

    #[test]
    fn contiguous_read_multi_inner_blocks_uses_tmp() {
        // The output is several blocks wide in the inner dim, so no block spans the full inner
        // width -> the direct path is disabled entirely and every block goes through `tmp_buf`.
        check_contiguous_read(&[4, 8], &[4, 2], &[0..4, 0..8]);
    }

    #[test]
    fn contiguous_read_unaligned_inner_dim_uses_tmp() {
        // The output is exactly one block wide in the inner dim (`out_shape[1] == block_shape[1]`),
        // yet `index[1]` is block-*unaligned*, so the inner dim is split across two clipped blocks.
        // The per-block full check (all dims, not just the leading one) must reject them and fall
        // back to `tmp_buf`; a leading-dim-only check would corrupt here.
        check_contiguous_read(&[4, 8], &[4, 4], &[0..4, 1..5]);
    }

    #[test]
    fn contiguous_read_3d_full_blocks_direct() {
        // 3D: dim 0 stacks full blocks, inner dims span the full output width -> direct path.
        check_contiguous_read(&[4, 3, 2], &[2, 3, 2], &[0..4, 0..3, 0..2]);
    }

    #[test]
    fn contiguous_read_leading_size1_dim_direct() {
        // A leading dim of extent 1 (block extent 1 there too): the first dim with block extent > 1
        // is dim 1, so dim 0 is ignored and dim 1 becomes the stacking dim. Each full block is a
        // contiguous run -> the direct path applies even though `out_shape[0] != ...` for dim 0.
        check_contiguous_read(&[1, 8, 4], &[1, 2, 4], &[0..1, 0..8, 0..4]);
    }

    #[test]
    fn contiguous_read_inner_size1_wider_output_uses_tmp() {
        // A block that is extent-1 in an *inner* dim whose output is wider (`block_shape[1] == 1`
        // but `out_shape[1] == 3`) is NOT a contiguous run - blindly ignoring the size-1 dim would
        // corrupt. The inner-full-width gate rejects it, so every block uses `tmp_buf`.
        check_contiguous_read(&[8, 3, 4], &[2, 1, 4], &[0..8, 0..3, 0..4]);
    }

    #[test]
    fn read_into_c_contiguous_strided_ignoring_size1_dim() {
        // A strided destination whose only non-C-contiguous stride is on a length-1 dim (dim 0)
        // still counts as contiguous - that stride never steps - so the direct-into-`buf` path
        // applies. Result must match a plain read.
        use crate::ArrayStorage;

        let nd = ndarray::ArrayD::from_shape_vec(
            vec![1usize, 8, 4],
            (0..32i32).map(|x| x * 3 - 5).collect(),
        )
        .unwrap();
        let mut params = ArrayParams::new();
        params.block_shape(&[1, 2, 4]);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let ctx = za.read_ctx();
        let storage = za.into_storage();

        // C-order byte strides for [1, 8, 4] i32 are [128, 16, 4]; give dim 0 (length 1) a bogus
        // large stride that must be ignored. The footprint is unaffected (dim 0 has one element).
        let byte_strides = [4000usize, 16, 4];
        let mut backing = vec![i32::MIN; 32];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), 32 * 4)
            };
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, &byte_strides) };
            storage
                .read_data(&[0..1, 0..8, 0..4], &ctx, Some(&mut out))
                .unwrap();
        }
        let expected: Vec<i32> = (0..32).map(|x| x * 3 - 5).collect();
        assert_eq!(backing, expected);
    }

    #[test]
    fn single_block_read_into_misaligned_contiguous_buf() {
        // Block decoding is byte-level, so a packed destination that is *not* aligned to the dtype
        // still takes the direct `read_block` fast path. Miri checks the accesses.
        use crate::ArrayStorage;

        let nd =
            ndarray::ArrayD::from_shape_vec(vec![2usize, 3], (0..6i32).map(|x| x + 1).collect())
                .unwrap();
        let mut params = ArrayParams::new();
        params.block_shape(&[2, 3]); // single block covering the whole array
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let ctx = za.read_ctx();
        let storage = za.into_storage();

        // Byte buffer whose window starts one byte past a 4-aligned address.
        let mut backing = [0u8; 6 * 4 + 4];
        let off = backing.as_ptr().align_offset(4) + 1;
        {
            let bytes = &mut backing[off..off + 6 * 4];
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, &[12, 4]) };
            storage
                .read_data(&[0..2, 0..3], &ctx, Some(&mut out))
                .unwrap();
        }
        let got = (0..6)
            .map(|i| unsafe {
                backing
                    .as_ptr()
                    .add(off + i * 4)
                    .cast::<i32>()
                    .read_unaligned()
            })
            .collect::<Vec<_>>();
        assert_eq!(got, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn aligned_single_block_c_contiguous_strided_ok() {
        // An aligned single block with a *strided* destination whose strides happen to be exactly
        // C-contiguous. `read_data` routes it to `read_data_slow` (strided); the direct-into-`buf`
        // path is nd-loop-only, so this takes the single-block `tmp_buf` + scatter path. The
        // backing buffer has trailing slack that must be left untouched.
        use crate::ArrayStorage;

        let nd =
            ndarray::ArrayD::from_shape_vec(vec![2usize, 3], (0..6i32).map(|x| x + 1).collect())
                .unwrap();
        let mut params = ArrayParams::new();
        params.block_shape(&[2, 3]); // single block covering the whole array
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let ctx = za.read_ctx();
        let storage = za.into_storage();

        // C-contiguous byte strides for [2, 3] i32: [3*4, 4] = [12, 4]. Backing has trailing slack.
        const SENTINEL: i32 = i32::MIN;
        let mut backing = vec![SENTINEL; 6 + 2];
        let byte_strides = [12usize, 4];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), backing.len() * 4)
            };
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, &byte_strides) };
            storage
                .read_data(&[0..2, 0..3], &ctx, Some(&mut out))
                .unwrap();
        }
        assert_eq!(&backing[..6], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&backing[6..], &[SENTINEL, SENTINEL], "wrote past the block");
    }
}
