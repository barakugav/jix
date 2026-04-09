use std::io::{self, Read, Seek};
use std::ops::Range;

use crate::NDIM_MAX;
use crate::archive::{ArchiveReader, Section};
use crate::array::{BlockShapeTag, BlocksLayout};
use crate::dtype::Dtype;
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{
    NdIterExtStridesPtr, NdIterExtStridesPtrMut, nd_iter_ext_logical_global_index,
};
use crate::schema::{self, ArchiveType};
use crate::util::{DimArray, dim_arr};
use crate::util::{Idx, default_strides};

use crate::block::{BlockSize, BlockTable, BlockTableStorage};
use crate::codec::{DecoderParams, EncoderParams, ReadContext};

pub trait ArrayStorage {
    fn shape(&self) -> &[u64];
    fn dtype(&self) -> &Dtype;

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
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()>;

    fn blocks_layout(&self) -> &BlocksLayout;

    fn codec_params(&self) -> (&EncoderParams, &DecoderParams);
}
pub struct Owned(pub(crate) ArrayBlockTableStorageBase<crate::block::Owned>);
pub struct Borrowed<'a>(pub(crate) ArrayBlockTableStorageBase<crate::block::Borrowed<'a>>);
pub struct Mmap(pub(crate) ArrayBlockTableStorageBase<crate::block::Mmap>);
macro_rules! impl_array_storage {
    ($ty:ty) => {
        impl ArrayStorage for $ty {
            fn shape(&self) -> &[u64] {
                &self.0.shape
            }
            fn dtype(&self) -> &Dtype {
                self.0.blocks.dtype()
            }
            fn read_data(
                &self,
                index: &[Range<u64>],
                buf: &mut [u8],
                context: &ReadContext,
            ) -> io::Result<()> {
                self.0.read_data(index, buf, context)
            }
            fn blocks_layout(&self) -> &BlocksLayout {
                &self.0.blocks_layout
            }
            fn codec_params(&self) -> (&EncoderParams, &DecoderParams) {
                (&self.0.encoder_params, &self.0.decoder_params)
            }
        }
    };
}
impl_array_storage!(Owned);
impl_array_storage!(Borrowed<'_>);
impl_array_storage!(Mmap);

#[derive(Clone, Default, Debug)]
pub struct ArrayParams {
    pub(crate) block_shape: Option<DimArray<BlockSize>>,
    pub(crate) block_shape_tag: Option<DimArray<BlockShapeTag>>,
    pub(crate) block_size_hint: Option<u64>,
    pub(crate) preferred_read_block_shape: Option<DimArray<BlockSize>>,
    pub(crate) preferred_read_block_size_hint: Option<u64>,
    pub(crate) encoder_params: Option<EncoderParams>,
    pub(crate) decoder_params: Option<DecoderParams>,
}
impl ArrayParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn override_from_storage(&mut self, storage: &impl ArrayStorage) {
        let (s_encoder_params, s_decoder_params) = storage.codec_params();
        self.encoder_params
            .get_or_insert_with(|| s_encoder_params.clone());
        self.decoder_params
            .get_or_insert_with(|| s_decoder_params.clone());

        let blocks_layout = storage.blocks_layout();
        self.block_shape
            .get_or_insert_with(|| blocks_layout.block_shape_hint.clone());
        self.block_shape_tag
            .get_or_insert_with(|| blocks_layout.block_shape_tag.clone());
        self.block_size_hint
            .get_or_insert(blocks_layout.block_size_hint);
        self.preferred_read_block_shape
            .get_or_insert_with(|| blocks_layout.preferred_read_block_shape.clone());
        self.preferred_read_block_size_hint
            .get_or_insert(blocks_layout.preferred_read_block_size_hint);
    }
}

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

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()>
    where
        S: BlockTableStorage,
    {
        let ndim = self.shape.len();
        assert_eq!(index.len(), ndim);
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

            // TODO: fast path for full blocks, where we can copy the entire block buffer in a single memcpy
            let mut iter = NdIter::new(
                block_size,
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

    pub(crate) fn read_from<R>(
        reader: R,
        len: u64,
        read_block_storage: impl FnOnce(&mut ArchiveReader<R>, Section, Section) -> io::Result<S>,
        params: ArrayParams,
    ) -> io::Result<Self>
    where
        R: Read + Seek,
        S: BlockTableStorage,
    {
        let mut reader = ArchiveReader::new(reader, len)?;
        let f_meta = reader.read_file_meta()?;
        if f_meta.archive_type != schema::ArchiveType::ArrayV1 as i32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unexpected zix file type: expected {:?}, actual {:?}",
                    schema::ArchiveType::ArrayV1,
                    ArchiveType::try_from(f_meta.archive_type)
                ),
            ));
        }

        let header = reader.read_message::<schema::ArrayHeader>()?;
        let ndim = header.shape.len();
        if ndim > NDIM_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("array ndim {ndim} exceeds maximum supported ndim {NDIM_MAX}"),
            ));
        }
        let shape: DimArray<_> = header.shape.as_slice().try_into().unwrap();
        if header.block_shape.len() != ndim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "array block_shape has different ndim {} than shape {ndim}",
                    header.block_shape.len(),
                ),
            ));
        }
        let block_shape = dim_arr(ndim, |dim| header.block_shape[dim] as BlockSize);
        // Compute padded shape in usize for nitems validation.
        let expected_nitems = (0..ndim)
            .map(|dim| {
                let s = shape[dim];
                let b = block_shape[dim] as u64;
                if s == 0 { 0 } else { s.ceil_to_multiple(b) }
            })
            .product::<u64>();

        let blocks = BlockTable::read_content(&mut reader, read_block_storage)?;
        if blocks.nitems() != expected_nitems {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "array blocks nitems {} does not match shape product {}",
                    blocks.nitems(),
                    expected_nitems
                ),
            ));
        }

        let b_layout = BlocksLayout::new2(
            Some(block_shape),
            params.block_shape_tag,
            params.block_size_hint,
            params.preferred_read_block_shape,
            params.preferred_read_block_size_hint,
            &shape,
            blocks.dtype().itemsize(),
        );

        Ok(Self::new(
            blocks,
            shape,
            b_layout,
            params.encoder_params.unwrap_or_default(),
            params.decoder_params.unwrap_or_default(),
        ))
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
            fn shape(&self) -> &[u64] {
                self.0.shape()
            }
            fn dtype(&self) -> &crate::dtype::Dtype {
                self.0.dtype()
            }
            fn read_data(
                &self,
                index: &[core::ops::Range<u64>],
                buf: &mut [u8],
                context: &crate::codec::ReadContext,
            ) -> io::Result<()> {
                self.0.read_data(index, buf, context)
            }
            fn blocks_layout(&self) -> &crate::array::BlocksLayout {
                self.0.blocks_layout()
            }
            fn codec_params(&self) -> (&crate::codec::EncoderParams, &crate::codec::DecoderParams) {
                self.0.codec_params()
            }
        }
    };
}
pub(crate) use impl_array_storage_forward;
