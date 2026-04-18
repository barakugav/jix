use std::ops::Range;

use crate::codec::{DecoderParams, EncoderParams, ReadContext};
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::nd_iter_ext_logical_global_index;
use crate::util::iter::NdIter;
use crate::util::{default_strides, dim_arr, nd_copy, DimArray};

pub struct Owned(pub(crate) ArrayBlockTableStorageBase<crate::storage::block::Owned>);
pub struct Borrowed<'a>(pub(crate) ArrayBlockTableStorageBase<crate::storage::block::Borrowed<'a>>);
pub struct Mmap(pub(crate) ArrayBlockTableStorageBase<crate::storage::block::Mmap>);
macro_rules! impl_array_storage {
    ($ty:ty) => {
        impl ArrayStorage for $ty {
            fn read_data(
                &self,
                index: &[Range<u64>],
                buf: &mut [u8],
                context: &ReadContext,
            ) -> Result<()> {
                self.0.read_data(index, buf, context)
            }
            fn shape(&self) -> &[u64] {
                &self.0.shape
            }
            fn dtype(&self) -> &Dtype {
                self.0.blocks.dtype()
            }
            fn spec(&self) -> ArrayStorageSpec<'_> {
                ArrayStorageSpec {
                    blocks_layout: &self.0.blocks_layout,
                    encoder_params: Some(&self.0.encoder_params),
                    decoder_params: Some(&self.0.decoder_params),
                    // decoder_config: Some(&self.0.blocks.decoder_config),
                }
            }
        }
    };
}
impl_array_storage!(Owned);
impl_array_storage!(Borrowed<'_>);
impl_array_storage!(Mmap);

pub(crate) struct ArrayBlockTableStorageBase<S> {
    pub(crate) blocks: BlockTable<S>,
    shape: DimArray<u64>,

    blocks_layout: BlocksLayout,
    block_grid_shape: DimArray<u64>, // shape.div_ceil(block_shape)

    encoder_params: EncoderParams,
    decoder_params: DecoderParams,
}
impl<S> ArrayBlockTableStorageBase<S> {
    pub(crate) fn new(
        blocks: BlockTable<S>,
        shape: DimArray<u64>,
        blocks_layout: BlocksLayout,
        encoder_params: EncoderParams,
        decoder_params: DecoderParams,
    ) -> Self {
        let ndim = shape.len();
        let block_grid_shape = dim_arr(ndim, |dim| {
            shape[dim].div_ceil(blocks_layout.block_shape_hint[dim] as u64)
        });
        Self {
            blocks,
            shape,

            blocks_layout,
            block_grid_shape,

            encoder_params,
            decoder_params,
        }
    }

    pub(crate) fn block_shape(&self) -> &[BlockSize] {
        &self.blocks_layout.block_shape_hint
    }

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()>
    where
        S: BlockTableStorage,
    {
        check_get_range(&self.shape, index)?;
        let _nitems = check_get_buffer_size(index, self.blocks.dtype(), buf)?;

        let ndim = self.shape.len();
        let block_shape = self.block_shape();

        let mut b_range = DimArray::default();
        let mut single_full_block = true; // TODO: use two flags: is_single_block, is_aligned
        for dim in 0..ndim {
            let b = block_shape[dim] as u64;
            let (i_start, i_end) = (index[dim].start, index[dim].end);
            let (b_begin, b_end) = (i_start / b, i_end.div_ceil(b));
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

        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = dim_arr(ndim, |dim| index[dim].start);
        let elem_end = dim_arr(ndim, |dim| index[dim].end);

        // Block-space begin/end for NdIter.
        let block_begin = dim_arr(ndim, |dim| index[dim].start / block_shape[dim] as u64);
        let block_end = dim_arr(ndim, |dim| index[dim].end.div_ceil(block_shape[dim] as u64));

        let mut block_iter = NdIter::new_with_begin(
            &block_begin,
            &block_end,
            (
                nd_iter_ext_logical_global_index(&self.block_grid_shape, &block_begin),
                NdIterExtBlockOffsetSize::new(
                    &self.shape,
                    &elem_begin,
                    &elem_end,
                    &dim_arr(ndim, |dim| block_shape[dim] as u64),
                ),
            ),
        );

        // Pre-allocate a buffer large enough for a full block.
        let full_buf_len = block_shape.iter().map(|s| *s as usize).product::<usize>() * itemsize;
        let mut tmp_buf = context.tmp_buf(full_buf_len, dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();

        while let Some((block_idx, (block_global_id, (block_inner_offset, block_size)))) =
            block_iter.next()
        {
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
                    block_size,
                    &block_strides,
                    &out_strides,
                    itemsize,
                )
            };
        }

        Ok(())
    }
}
