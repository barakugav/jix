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
use crate::error::{check_get_buffer_size, check_get_range, check_ndim, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage};
use crate::storage::params::ArraySpecOwned;
use crate::storage::{ArraySpec, ElementType};
use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::nd_iter_ext_logical_global_index;
use crate::util::iter::NdIter;
use crate::util::{calc_block_end, default_strides, dim_arr, nd_copy, DimArray};
use crate::{ArrayParams, ArrayStorage, Dimension};

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
            fn read_data(
                &self,
                index: &[Range<u64>],
                buf: &mut [u8],
                context: &ReadContext,
            ) -> Result<()> {
                self.0.read_data(index, buf, context)
            }

            #[inline(always)]
            fn shape(&self) -> &[u64] {
                self.0.shape()
            }
            #[inline(always)]
            fn dtype(&self) -> &Dtype {
                self.0.blocks.dtype()
            }

            #[inline]
            fn spec(&self) -> ArraySpec<'_> {
                self.0.spec.as_ref()
            }

            fn as_compact(&self) -> Option<CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
                Some(CompactBorrowed(ArrayBlockTableStorageBase {
                    blocks: self.0.blocks.as_ref(),
                    shape: self.0.shape.clone(),

                    block_grid_shape: self.0.block_grid_shape.clone(),

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

    /// Number of blocks per dimension: `ceil(shape[d] / block_shape[d])`.
    block_grid_shape: DimArray<u64>,

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
        let ndim = shape_slice.len();
        let spec = params.into_spec(shape_slice, blocks.dtype())?;
        let block_shape = spec.as_ref().block_shape();
        let block_grid_shape = dim_arr(ndim, |dim| {
            shape_slice[dim].div_ceil(block_shape[dim] as u64)
        });
        Ok(Self {
            blocks,
            shape,
            block_grid_shape,
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
    /// 1. For each dimension, compute the range of block indices that overlap `index`:
    ///    `b_begin[d] = index[d].start / block_shape[d]`,
    ///    `b_end[d]   = ceil(index[d].end / block_shape[d])`.
    ///
    /// 2. **Fast path** - if the index falls exactly on a single aligned block (start and end
    ///    are both block-aligned, one block per dimension), compute the 1D block index via
    ///    row-major flattening and call [`BlockTable::read_block`](crate::storage::block::BlockTable::read_block)
    ///    directly into `buf`. No temporary buffer needed.
    ///
    /// 3. **General path** - iterate over every nd-block in the touched range using `NdIter`,
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
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()>
    where
        ET: ElementType,
        D: Dimension,
    {
        let shape = self.shape();
        check_get_range(shape, index)?;
        let _nitems = check_get_buffer_size(index, self.blocks.dtype(), buf)?;

        let ndim = shape.len();
        let block_shape = self.block_shape();

        let mut b_range = DimArray::default();
        let mut single_full_block = true; // TODO: use two flags: is_single_block, is_aligned
        for dim in 0..ndim {
            let b = block_shape[dim] as u64;
            let (i_start, i_end) = (index[dim].start, index[dim].end);
            let (b_begin, b_end) = (i_start / b, calc_block_end(i_start, i_end, b));
            b_range.push(b_begin..b_end);
            single_full_block &=
                b_begin + 1 == b_end && i_start.is_multiple_of(b) && i_end.is_multiple_of(b);
        }

        // Fast path for aligned single-block read
        if single_full_block {
            let block_idx = (0..ndim).fold(0u64, |blk_idx, dim| {
                blk_idx * self.block_grid_shape[dim] + b_range[dim].start
            });
            return self.blocks.read_block(block_idx, buf, context);
        }

        let dtype = self.blocks.dtype();
        let itemsize = dtype.itemsize() as usize;
        let out_shape = dim_arr(ndim, |dim| (index[dim].end - index[dim].start) as usize);
        let out_strides = default_strides(&out_shape, itemsize);
        let block_strides = default_strides(block_shape, itemsize as BlockSize); // TODO: precomute me

        // Block-space begin/end for NdIter.
        let block_begin = D::from_fn(ndim, |dim| index[dim].start / block_shape[dim] as u64);
        let block_end = D::from_fn(ndim, |dim| {
            calc_block_end(index[dim].start, index[dim].end, block_shape[dim] as u64)
        });
        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = D::from_fn(ndim, |dim| index[dim].start);
        let elem_end = D::from_fn(ndim, |dim| index[dim].end);
        let block_global_idx_ext =
            nd_iter_ext_logical_global_index(&self.block_grid_shape, block_begin.as_slice());

        let block_iter = NdIter::new_with_begin(
            block_begin,
            block_end,
            (
                block_global_idx_ext,
                NdIterExtBlockOffsetSize::new(
                    elem_begin,
                    elem_end,
                    D::from_fn(ndim, |dim| block_shape[dim] as u64),
                ),
            ),
        );

        // Pre-allocate a buffer large enough for a full block.
        let full_buf_len = block_shape.iter().map(|s| *s as usize).product::<usize>() * itemsize;
        let mut tmp_buf = context.tmp_buf(full_buf_len, dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();

        for (block_idx, (block_global_id, (block_inner_offset, block_size))) in block_iter {
            self.blocks.read_block(block_global_id, tmp_buf, context)?;

            // Navigate to the active region within the block buffer (block-local strides).
            let active_start = (0..ndim)
                .map(|dim| block_inner_offset[dim] as usize * block_strides[dim] as usize)
                .sum::<usize>();
            let src_ptr = unsafe { tmp_buf.as_ptr().add(active_start) };

            // Map the active region's start to its position in the output array.
            let out_start = (0..ndim)
                .map(|dim| {
                    let full_idx =
                        block_idx[dim] * block_shape[dim] as u64 + block_inner_offset[dim];
                    let out_idx = full_idx - index[dim].start;
                    out_idx as usize * out_strides[dim]
                })
                .sum::<usize>();
            let dst_ptr = unsafe { buf.as_mut_ptr().add(out_start) };

            unsafe {
                nd_copy(
                    src_ptr,
                    dst_ptr,
                    block_size.clone(),
                    &block_strides,
                    &out_strides,
                    itemsize,
                )
            };
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
            block_grid_shape: self.block_grid_shape,
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
            block_grid_shape: self.block_grid_shape,
            spec: self.spec,
        })
    }
}
