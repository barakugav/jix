use std::borrow::Cow;
use std::cell::RefCell;
use std::io;

use crate::NDIM_MAX;
use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{NdIterExtensionStridesPtr, NdIterExtensionStridesPtrMut};
use crate::storage::Storage;
use crate::storage::block::{BlockSize, BlockTable};
use crate::storage::codec::{Decoder, Encoder};
use crate::array::BlocksLayout;
use crate::util::{DimArray, ceil_to_multiple, default_strides, full_dim_array};

pub(crate) struct CompressedStorage {
    blocks: BlockTable<'static>,
    dtype: Dtype,
    decoder: RefCell<Decoder>,
}

impl CompressedStorage {
    pub(crate) fn from_block_table(blocks: BlockTable<'static>, dtype: Dtype) -> io::Result<Self> {
        Ok(Self {
            blocks,
            dtype,
            decoder: RefCell::new(Decoder::new()?),
        })
    }

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

        let block_shape_clamped = block_shape
            .iter()
            .zip(&shape)
            .map(|(&b, &s)| b.min(s))
            .collect::<DimArray<_>>();
        let padded_shape = block_shape_clamped
            .iter()
            .zip(&shape)
            .map(|(&b, &s)| if s == 0 { 0 } else { ceil_to_multiple(s, b) })
            .collect::<DimArray<_>>();
        let blocks_layout = BlocksLayout::new(&block_shape_clamped, &shape);
        let nblocks = blocks_layout.grid_shape.iter().product::<usize>();

        let mut block_iter = NdIter::new(
            &blocks_layout.grid_shape,
            NdIterExtBlockOffsetSize::new(
                &shape,
                &full_dim_array(0, ndim),
                &shape,
                &blocks_layout,
            ),
        );

        let mut encoder = Encoder::new(3)?;
        let mut cdata = Vec::<u8>::new();
        let mut block_offsets =
            Vec::<u64>::with_capacity(if nblocks == 0 { 0 } else { nblocks + 1 });
        if nblocks > 0 {
            block_offsets.push(0);
        }
        let block_capacity_bytes = blocks_layout.block_size * itemsize;
        let max_blk_cdata_len = encoder.encode_bound(block_capacity_bytes);
        let mut tmp_block_data = Vec::<u8>::with_capacity(block_capacity_bytes);
        let tmp_block_strides = default_strides(&block_shape_clamped, itemsize);
        let strides = array
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).unwrap() * size_of::<T>())
            .collect::<DimArray<_>>();
        while let Some((block_idx, (block_inner_offset, block_size))) = block_iter.next() {
            debug_assert!(block_inner_offset.iter().all(|&o| o == 0));

            // Init block data to zeros (padding elements stay zero).
            tmp_block_data.clear();
            tmp_block_data.resize(block_capacity_bytes, 0);

            let initial_arr_offset = (0..ndim)
                .map(|dim| {
                    let idx =
                        block_idx[dim] * blocks_layout.block_shape[dim] + block_inner_offset[dim];
                    idx * strides[dim]
                })
                .sum::<usize>();
            let initial_arr_ptr = unsafe { array.as_ptr().cast::<u8>().add(initial_arr_offset) };
            let initial_block_offset = (0..ndim)
                .map(|dim| block_inner_offset[dim] * tmp_block_strides[dim])
                .sum::<usize>();
            let initial_block_ptr =
                unsafe { tmp_block_data.as_mut_ptr().add(initial_block_offset) };
            let mut iter = NdIter::new(
                block_size,
                (
                    NdIterExtensionStridesPtr::new(&strides, initial_arr_ptr),
                    NdIterExtensionStridesPtrMut::new(&tmp_block_strides, initial_block_ptr),
                ),
            );
            while let Some((_idx, (src, dst))) = iter.next() {
                unsafe { std::ptr::copy_nonoverlapping(src, dst, itemsize) };
            }

            let cdata_len = cdata.len();
            cdata.reserve(max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + max_blk_cdata_len) };
            let blk_buf = &mut cdata[cdata_len..];
            let blk_cdata_len = encoder.encode(&tmp_block_data, blk_buf)?;
            debug_assert!(blk_cdata_len <= max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + blk_cdata_len) };
            block_offsets.push(cdata.len() as u64);
        }

        let blocks = BlockTable::new(
            Cow::Owned(cdata),
            Cow::Owned(block_offsets),
            dtype.itemsize(),
            padded_shape.iter().product::<usize>(),
            blocks_layout.block_size as BlockSize,
        );

        Ok(Self {
            blocks,
            dtype,
            decoder: RefCell::new(Decoder::new()?),
        })
    }
}

impl Storage for CompressedStorage {
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn nitems(&self) -> usize {
        self.blocks.nitems
    }

    fn block_len(&self) -> BlockSize {
        self.blocks.block_size
    }

    fn read_block(&self, block_idx: usize, buf: &mut [u8]) -> io::Result<()> {
        let block = self.blocks.get_block(block_idx);
        let b_size_bytes = self.block_len() as usize * self.blocks.itemsize as usize;
        if buf.len() < b_size_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Buffer too small"));
        }
        let nbytes = self.decoder.borrow_mut().decode(&block, buf)?;
        debug_assert_eq!(nbytes, b_size_bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CompressedStorage;
    use crate::dtype::Dtyped;
    use crate::storage::Storage;
    use crate::storage::block::{BlockSize, BlockTable};
    use crate::storage::codec::Encoder;
    use crate::util::cast_slice;

    fn make_storage<T: Dtyped>(items: &[T], block_len: BlockSize) -> CompressedStorage {
        let mut encoder = Encoder::new(3).unwrap();
        let blocks = BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            size_of::<T>() as _,
            block_len,
            &mut encoder,
        )
        .unwrap();
        CompressedStorage::from_block_table(blocks, T::dtype()).unwrap()
    }

    fn read_block_items<T: Dtyped>(storage: &CompressedStorage, idx: usize) -> Vec<T> {
        let block_bytes = storage.block_len() as usize * storage.dtype().itemsize() as usize;
        let mut buf = vec![0u8; block_bytes];
        storage.read_block(idx, &mut buf).unwrap();
        unsafe { cast_slice::<u8, T>(&buf) }.to_vec()
    }

    #[test]
    fn single_block_u8_round_trips() {
        let items: Vec<u8> = (0..8).collect();
        let s = make_storage(&items, 8);
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 8);
        assert_eq!(s.dtype(), &u8::dtype());
        assert_eq!(read_block_items::<u8>(&s, 0), items);
    }

    #[test]
    fn two_blocks_i32_round_trips() {
        let items: Vec<i32> = (0..8).collect();
        let s = make_storage(&items, 4);
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 4);
        assert_eq!(read_block_items::<i32>(&s, 0), items[..4]);
        assert_eq!(read_block_items::<i32>(&s, 1), items[4..]);
    }

    #[test]
    fn multiple_blocks_f32_round_trips() {
        let items: Vec<f32> = (0..12).map(|x| x as f32 * 0.5).collect();
        let s = make_storage(&items, 4);
        assert_eq!(s.nitems(), 12);
        assert_eq!(s.block_len(), 4);
        for b in 0..3 {
            assert_eq!(read_block_items::<f32>(&s, b), items[b * 4..(b + 1) * 4]);
        }
    }

    #[test]
    fn buffer_too_small_returns_error() {
        let items: Vec<u8> = (0..4).collect();
        let s = make_storage(&items, 4);
        let mut buf = vec![0u8; 3]; // one byte short
        assert!(s.read_block(0, &mut buf).is_err());
    }
}
