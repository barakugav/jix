use std::io;
use std::ops::Range;

use crate::array::BlocksLayout;
use crate::dtype::Dtype;
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{
    NdIterExtStridesPtr, NdIterExtStridesPtrMut, nd_iter_ext_logical_global_index,
};
use crate::util::default_strides;
use crate::util::{DimArray, dim_arr};

use crate::block::{BlockTable, BlockTableStorage};
use crate::codec::ReadContext;

pub trait ArrayStorage {
    fn dtype(&self) -> &Dtype;
    fn shape(&self) -> &[usize];
    fn blocks_layout(&self) -> &BlocksLayout;

    /// Read the specified slice of the array into the provided buffer.
    ///
    /// # Arguments
    ///
    /// - `index`: A slice of ranges, one per dimension, specifying the slice of the array to read.
    ///   Each range is half-open: `start..end`, where `start` is inclusive and `end` is exclusive.
    /// - `buf`: A mutable byte slice to store the read data.
    ///   The size of the buffer must be exactly equal to the number of elements in the specified
    ///   slice multiplied by the item size of the array's dtype.
    ///   The buffer base pointer must be suitably aligned for the array's dtype.
    ///   Elements should be laid out in row-major order (C-style contiguous) in the buffer.
    /// - `context`: A context object that may be used for caching or other purposes during the
    ///   read operation. See `ReadContext` for more details.
    fn read_data(
        &self,
        index: &[Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()>;
}
pub struct Owned(pub(crate) ArrayBlockTableStorageBase<crate::block::Owned>);
pub struct Borrowed<'a>(pub(crate) ArrayBlockTableStorageBase<crate::block::Borrowed<'a>>);
pub struct Mmap(pub(crate) ArrayBlockTableStorageBase<crate::block::Mmap>);
macro_rules! impl_array_storage {
    ($ty:ty) => {
        impl ArrayStorage for $ty {
            fn dtype(&self) -> &Dtype {
                self.0.blocks.dtype()
            }
            fn shape(&self) -> &[usize] {
                &self.0.shape
            }
            fn blocks_layout(&self) -> &BlocksLayout {
                &self.0.blocks_layout
            }
            fn read_data(
                &self,
                index: &[Range<usize>],
                buf: &mut [u8],
                context: &ReadContext,
            ) -> io::Result<()> {
                self.0.read_data(index, buf, context)
            }
        }
    };
}
impl_array_storage!(Owned);
impl_array_storage!(Borrowed<'_>);
impl_array_storage!(Mmap);

pub(crate) struct ArrayBlockTableStorageBase<S> {
    pub(crate) blocks: BlockTable<S>,
    shape: DimArray<usize>,
    blocks_layout: BlocksLayout,
}
impl<S> ArrayBlockTableStorageBase<S> {
    pub(crate) fn new(
        blocks: BlockTable<S>,
        shape: DimArray<usize>,
        blocks_layout: BlocksLayout,
    ) -> Self {
        Self {
            blocks,
            shape,
            blocks_layout,
        }
    }

    fn read_data(
        &self,
        index: &[Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()>
    where
        S: BlockTableStorage,
    {
        let ndim = self.shape.len();
        assert_eq!(index.len(), ndim);
        let block_shape = &self.blocks_layout.block_shape;
        let mut b_range = DimArray::default();
        let mut single_full_block = true;
        for dim in 0..ndim {
            let i_range = &index[dim];
            let b = block_shape[dim];
            let b_begin = i_range.start / b;
            let b_end = i_range.end.div_ceil(b);
            b_range.push(b_begin..b_end);
            single_full_block &=
                b_begin + 1 == b_end && i_range.start % b == 0 && i_range.end % b == 0;
        }

        let shape = self.shape.as_slice();
        let grid_shape = dim_arr(shape.len(), |dim| shape[dim].div_ceil(block_shape[dim]));

        // Fast path for aligned single-block read
        if single_full_block {
            let block_idx = (0..ndim).fold(0, |blk_idx, dim| {
                blk_idx * grid_shape[dim] + b_range[dim].start
            });
            return self.blocks.read_block(block_idx, buf, context);
        }

        let dtype = self.blocks.dtype();
        let itemsize = dtype.itemsize() as usize;
        let out_shape = dim_arr(ndim, |dim| index[dim].end - index[dim].start);
        let out_size = out_shape.iter().product::<usize>();
        if buf.len() != out_size * itemsize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "output buffer has incorrect size: expected {} bytes, actual {} bytes",
                    out_size * itemsize,
                    buf.len()
                ),
            ));
        }
        let out_strides = default_strides(&out_shape, itemsize);
        let block_strides = default_strides(&block_shape, itemsize);

        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = dim_arr(ndim, |dim| index[dim].start);
        let elem_end = dim_arr(ndim, |dim| index[dim].end);

        // Block-space begin/end for NdIter.
        let block_begin = dim_arr(ndim, |dim| index[dim].start / block_shape[dim]);
        let block_end = dim_arr(ndim, |dim| index[dim].end.div_ceil(block_shape[dim]));

        let mut block_iter = NdIter::new_with_begin(
            &block_begin,
            &block_end,
            (
                nd_iter_ext_logical_global_index(&grid_shape, &block_begin),
                NdIterExtBlockOffsetSize::new(shape, &elem_begin, &elem_end, block_shape),
            ),
        );

        // Pre-allocate a buffer large enough for a full block.
        let full_buf_len = block_shape.iter().product::<usize>() * itemsize;
        let mut tmp_buf = Vec::with_capacity(full_buf_len); // TODO: move to ReadContext
        unsafe { tmp_buf.set_len(full_buf_len) };
        while let Some((block_idx, (block_global_id, (block_inner_offset, block_size)))) =
            block_iter.next()
        {
            self.blocks
                .read_block(block_global_id, &mut tmp_buf, context)?;

            // Navigate to the active region within the block buffer (block-local strides).
            let active_start = (0..ndim)
                .map(|dim| block_inner_offset[dim] * block_strides[dim])
                .sum::<usize>();
            let src_ptr = unsafe { tmp_buf.as_ptr().add(active_start) };

            // Map the active region's start to its position in the output array.
            let out_start = (0..ndim)
                .map(|dim| {
                    let full_idx = block_idx[dim] * block_shape[dim] + block_inner_offset[dim];
                    let out_idx = full_idx - index[dim].start;
                    out_idx * out_strides[dim]
                })
                .sum::<usize>();
            let dst_ptr = unsafe { buf.as_mut_ptr().add(out_start) };

            let mut iter = NdIter::new(
                &block_size,
                (
                    NdIterExtStridesPtr::new(&block_strides, src_ptr),
                    NdIterExtStridesPtrMut::new(&out_strides, dst_ptr),
                ),
            );
            while let Some((_idx, (src, dst))) = iter.next() {
                unsafe { std::ptr::copy_nonoverlapping(src, dst, itemsize) };
            }
        }

        Ok(())
    }
}

pub struct Ref<'a, S>(pub(crate) &'a S);
impl_array_storage_forward!(Ref<'a, S> where S: ArrayStorage);

macro_rules! impl_array_storage_forward {
    ($wrapper:ident $(<$($gen:tt),*>)? $(where $($wh:tt)*)?) => {
        impl $(<$($gen),*>)? ArrayStorage for $wrapper $(<$($gen),*>)?
        where
            $($($wh)*)?
        {
            fn dtype(&self) -> &crate::dtype::Dtype {
                self.0.dtype()
            }
            fn shape(&self) -> &[usize] {
                self.0.shape()
            }
            fn blocks_layout(&self) -> &crate::array::BlocksLayout {
                self.0.blocks_layout()
            }
            fn read_data(
                &self,
                index: &[core::ops::Range<usize>],
                buf: &mut [u8],
                context: &crate::codec::ReadContext,
            ) -> io::Result<()> {
                self.0.read_data(index, buf, context)
            }
        }
    };
}
pub(crate) use impl_array_storage_forward;
