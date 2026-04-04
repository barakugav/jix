use std::borrow::Cow;
use std::cell::RefCell;
use std::io;

use crate::NDIM_MAX;
use crate::array::BlocksLayout;
use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{NdIterExtensionStridesPtr, NdIterExtensionStridesPtrMut};
use crate::storage::block::BlockTable;
use crate::storage::codec::{Decoder, Encoder};
use crate::storage::{BlockSize, Storage};
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
        let b_size_bytes = self.block_len() as usize * self.blocks.itemsize as usize;
        if buf.len() < b_size_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer too small",
            ));
        }
        let block = self.blocks.get_block(block_idx);
        let nbytes = self.decoder.borrow_mut().decode(&block, buf)?;
        debug_assert_eq!(nbytes, b_size_bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CompressedStorage;
    use crate::dtype::Dtyped;
    use crate::storage::block::BlockTable;
    use crate::storage::codec::Encoder;
    use crate::storage::{BlockSize, Storage};
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
