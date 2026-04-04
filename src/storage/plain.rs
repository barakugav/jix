use std::io;

use crate::dtype::Dtype;
use crate::storage::{BlockSize, Storage};

pub(crate) struct PlainStorage {
    data: Vec<u8>,
    dtype: Dtype,
    nitems: usize,
    block_len: BlockSize,
}

impl PlainStorage {
    pub(crate) fn new(data: Vec<u8>, dtype: Dtype, nitems: usize, block_len: BlockSize) -> Self {
        assert!(block_len > 0);
        assert!(nitems % block_len as usize == 0);
        assert_eq!(data.len(), nitems * dtype.itemsize() as usize);
        Self {
            data,
            dtype,
            nitems,
            block_len,
        }
    }
}

impl Storage for PlainStorage {
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn nitems(&self) -> usize {
        self.nitems
    }

    fn block_len(&self) -> BlockSize {
        self.block_len
    }

    fn read_block(&self, block_idx: usize, buf: &mut [u8]) -> io::Result<()> {
        let block_bytes = self.block_len as usize * self.dtype.itemsize() as usize;
        let start = block_idx * block_bytes;
        buf.copy_from_slice(&self.data[start..start + block_bytes]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::PlainStorage;
    use crate::dtype::Dtyped;
    use crate::storage::Storage;
    use crate::util::{cast_slice, cast_slice_mut};

    fn make_storage<T: Dtyped>(items: &[T], block_len: u32) -> PlainStorage {
        let data = unsafe { cast_slice::<T, u8>(items) }.to_vec();
        PlainStorage::new(data, T::dtype(), items.len(), block_len)
    }

    fn read_block_items<T: Dtyped>(storage: &PlainStorage, idx: usize) -> Vec<T> {
        let mut buf = vec![MaybeUninit::<T>::uninit(); storage.block_len() as usize];
        let buf_bytes = unsafe { cast_slice_mut::<_, u8>(buf.as_mut_slice()) };
        storage.read_block(idx, buf_bytes).unwrap();
        unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(buf) }
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
    fn non_divisible_panics() {
        let result = std::panic::catch_unwind(|| {
            make_storage(&[0u8; 7], 4);
        });
        assert!(result.is_err());
    }
}
