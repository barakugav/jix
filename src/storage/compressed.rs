use std::borrow::Cow;
use std::cell::RefCell;
use std::io;

use crate::NDIM_MAX;
use crate::dtype::{Dtype, Dtyped, Itemsize};
use crate::iter::NdIter;
use crate::iter::chunk::NdIterExtChunkOffsetSize;
use crate::iter::strides::{NdIterExtensionStridesPtr, NdIterExtensionStridesPtrMut};
use crate::storage::block::{BlockSize, BlockTable};
use crate::storage::codec::{Decoder, Encoder};
use crate::storage::{ArrayStorage, ChunksLayout};
use crate::util::{DimArray, ceil_to_multiple, default_strides, full_dim_array};

struct CompressedStorage {
    blocks: BlockTable<'static>,

    dtype: Dtype,
    shape: DimArray<usize>,
    padded_shape: DimArray<usize>,
    /// # Invariants:
    /// - len(chunk_shape) == len(shape)`
    /// - all(0 <= chunk_shape[i] <= shape[i])`
    /// - all(shape[i] == 0 or chunk_shape[i] > 0)` (chunk_shape[i] cant be zero unless shape[i] is also zero)
    chunks_layout: ChunksLayout,

    decoder: RefCell<Decoder>,
}
impl CompressedStorage {
    pub fn from_ndarray<T, D>(
        array: &ndarray::ArrayView<T, D>,
        block_shape: &[usize],
    ) -> io::Result<Self>
    where
        T: Dtyped,
        D: ndarray::Dimension,
    {
        let ndim = array.ndim();
        assert!(ndim < NDIM_MAX);
        assert_eq!(ndim, block_shape.len());
        let dtype = T::dtype();
        let itemsize = dtype.itemsize() as usize;
        let shape = array.shape().iter().cloned().collect::<DimArray<_>>();

        let block_shape = block_shape
            .iter()
            .zip(&shape)
            .map(|(&b, &s)| b.min(s))
            .collect::<DimArray<_>>();
        let padded_shape = block_shape
            .iter()
            .zip(&shape)
            .map(|(&b, &s)| if s == 0 { 0 } else { ceil_to_multiple(s, b) })
            .collect::<DimArray<_>>();
        let c_layout = ChunksLayout::new(&block_shape, &shape);
        let nblocks = c_layout.chunk_space_shape.iter().product::<usize>();

        let mut chunk_iter = NdIter::new(
            &c_layout.chunk_space_shape,
            NdIterExtChunkOffsetSize::new(
                &shape,
                &full_dim_array(0, ndim),
                &c_layout.chunk_space_shape,
                &c_layout,
            ),
        );

        let mut encoder = Encoder::new(3)?;
        let mut cdata = Vec::<u8>::new();
        let mut block_offsets =
            Vec::<u64>::with_capacity(if nblocks == 0 { 0 } else { nblocks + 1 });
        if nblocks > 0 {
            block_offsets.push(0);
        }
        let chunk_capacity_bytes = c_layout.chunk_size * itemsize;
        let max_blk_cdata_len = encoder.encode_bound(chunk_capacity_bytes);
        let mut tmp_chunk_data = Vec::<u8>::with_capacity(chunk_capacity_bytes);
        let tmp_chunk_strides = default_strides(&c_layout.chunk_shape, itemsize);
        let strides = array
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).unwrap() * size_of::<T>())
            .collect::<DimArray<_>>();
        while let Some((chunk_idx, (chunk_inner_offset, chunk_size))) = chunk_iter.next() {
            debug_assert!(chunk_inner_offset.iter().all(|&o| o == 0));

            // Init chunk data to zeros.
            // The padding elements (if any) will not be written by the iter below, so they will stay zeros.
            tmp_chunk_data.clear();
            tmp_chunk_data.resize(chunk_capacity_bytes, 0);

            // TODO: fast path for contiguous data
            let initial_arr_offset = (0..ndim)
                .map(|dim| {
                    let idx = chunk_idx[dim] * c_layout.chunk_shape[dim] + chunk_inner_offset[dim];
                    idx * strides[dim]
                })
                .sum::<usize>();
            let initial_arr_ptr = unsafe { array.as_ptr().cast::<u8>().add(initial_arr_offset) };
            let initial_chunk_offset = (0..ndim)
                .map(|dim| chunk_inner_offset[dim] * tmp_chunk_strides[dim])
                .sum::<usize>();
            let initial_chunk_ptr =
                unsafe { tmp_chunk_data.as_mut_ptr().add(initial_chunk_offset) };
            let mut iter = NdIter::new(
                chunk_size,
                (
                    NdIterExtensionStridesPtr::new(&strides, initial_arr_ptr),
                    NdIterExtensionStridesPtrMut::new(&tmp_chunk_strides, initial_chunk_ptr),
                ),
            );
            while let Some((_idx, (src, dst))) = iter.next() {
                unsafe { std::ptr::copy_nonoverlapping(src, dst, itemsize) };
            }

            let cdata_len = cdata.len();
            cdata.reserve(max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + max_blk_cdata_len) };
            let blk_buf = &mut cdata[cdata_len..];
            let blk_cdata_len = encoder.encode(&tmp_chunk_data, blk_buf)?;
            debug_assert!(blk_cdata_len <= max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + blk_cdata_len) };
            block_offsets.push(cdata.len() as u64);
        }

        let blocks = BlockTable::new(
            Cow::Owned(cdata),
            Cow::Owned(block_offsets),
            dtype.itemsize(),
            padded_shape.iter().product::<usize>(),
            chunk_capacity_bytes as BlockSize,
        );

        Ok(Self {
            blocks,
            dtype,
            shape,
            padded_shape,
            chunks_layout: c_layout,
            decoder: RefCell::new(Decoder::new()?),
        })
    }
}
impl ArrayStorage for CompressedStorage {
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn chunks_layout(&self) -> &ChunksLayout {
        &self.chunks_layout
    }
    fn get_chunk_data(
        &self,
        chunk_global_id: usize,
        _chunk_idx: &[usize],
        buf: &mut [u8],
    ) -> io::Result<()> {
        let itemsize = self.dtype.itemsize() as usize;
        let b_size_bytes = self.chunks_layout.chunk_size * itemsize;
        if buf.len() < b_size_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer too small",
            ));
        }

        let block = self.blocks.get_block(chunk_global_id);
        let nbytes = self.decoder.borrow_mut().decode(&block, buf)?;
        debug_assert_eq!(nbytes, b_size_bytes);
        Ok(())
    }
}
