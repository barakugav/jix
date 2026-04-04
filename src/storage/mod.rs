use std::io;

use crate::dtype::Dtype;
use crate::util::DimArray;

mod block;
mod codec;
mod common;
mod compressed;
mod plain;

pub(crate) trait ArrayStorage {
    fn dtype(&self) -> &Dtype;
    fn shape(&self) -> &[usize];
    fn chunks_layout(&self) -> &ChunksLayout;
    fn get_chunk_data(
        &self,
        chunk_global_id: usize,
        chunk_idx: &[usize],
        buf: &mut [u8],
    ) -> io::Result<()>;
}
pub(crate) struct ChunksLayout {
    pub(crate) chunk_shape: DimArray<usize>,
    pub(crate) chunk_space_shape: DimArray<usize>,
    pub(crate) chunk_size: usize, // chunk_shape.iter().product()
}
impl ChunksLayout {
    pub(crate) fn new(chunk_shape: &[usize], shape: &[usize]) -> Self {
        let chunk_shape = chunk_shape.iter().cloned().collect::<DimArray<_>>();
        let chunk_space_shape = shape
            .iter()
            .zip(&chunk_shape)
            .map(|(&s, &c)| s.div_ceil(c))
            .collect();
        let chunk_size = chunk_shape.iter().product();
        Self {
            chunk_shape,
            chunk_space_shape,
            chunk_size,
        }
    }
}

pub(crate) struct ChunkDesc {
    // global_index: usize,
    chunk_idx: DimArray<usize>,
}

// pub(crate) struct DynStorage {
//     storage: Box<dyn ArrayStorage>,
//     dtype: Dtype,
//     shape: DimVec<usize>,
//     chunks_layout: ChunksLayoutInfo,
// }
// impl ArrayStorage for DynStorage {
//     fn dtype(&self) -> &Dtype {
//         &self.dtype
//     }
//     fn shape(&self) -> &[usize] {
//         &self.shape
//     }
//     fn chunks_layout(&self) -> &ChunksLayoutInfo {
//         &self.chunks_layout
//     }
//     fn get_chunk(&self, chunk_idx: &[usize], chunk_buf: &mut ChunkBuf) -> Result<(), Error> {
//         self.storage.get_chunk(chunk_idx, chunk_buf)
//     }
// }

#[cfg(test)]
pub(crate) mod tests {
    use ndarray::ArrayD;

    use crate::dtype::Dtyped;
    use crate::iter::IdxIter;
    use crate::util::{DimArray, cast_slice_mut, default_strides};

    use super::ArrayStorage;

    /// Verifies that every in-bounds element returned by `storage` matches `reference`.
    /// Padding elements (where chunk extends beyond array bounds) are ignored.
    pub(crate) fn check_storage_matches_array<S, T>(storage: &S, reference: &ArrayD<T>)
    where
        S: ArrayStorage,
        T: Dtyped + PartialEq + std::fmt::Debug,
    {
        assert_eq!(storage.dtype(), &T::dtype(), "dtype mismatch");

        let shape = storage.shape();
        let ndim = shape.len();
        assert_eq!(shape, reference.shape(), "shape mismatch");
        let itemsize = storage.dtype().itemsize() as usize;

        let c_layout = storage.chunks_layout();
        let chunk_shape = c_layout.chunk_shape.as_slice();
        let chunk_space_shape = c_layout.chunk_space_shape.as_slice();
        let chunk_size = c_layout.chunk_size;

        let mut chunk_buf: Vec<T> = Vec::with_capacity(chunk_size);
        // Safety: T: Dtyped: Copy, uninitialized bytes are overwritten by get_chunk_data.
        unsafe { chunk_buf.set_len(chunk_size) };
        let chunk_buf_strides = default_strides(chunk_shape, itemsize);

        let mut chunk_iter = IdxIter::new(chunk_space_shape);
        let mut chunk_global_id = 0;
        while let Some(chunk_idx) = chunk_iter.next() {
            let byte_buf = unsafe { cast_slice_mut::<T, u8>(chunk_buf.as_mut_slice()) };
            storage
                .get_chunk_data(chunk_global_id, &chunk_idx, byte_buf)
                .unwrap();

            let mut iter = IdxIter::new(&chunk_shape);
            while let Some(c_inner_idx) = iter.next() {
                let c_offset = (0..ndim)
                    .map(|d| c_inner_idx[d] * chunk_buf_strides[d])
                    .sum::<usize>();
                let value = &byte_buf[c_offset..c_offset + itemsize];

                let idx = (0..ndim)
                    .map(|d| chunk_idx[d] * chunk_shape[d] + c_inner_idx[d])
                    .collect::<DimArray<_>>();
                let in_bounds = idx.iter().zip(shape).all(|(&i, &s)| i < s);
                if !in_bounds {
                    assert!(value.iter().all(|&b| b == 0)); // padding element
                    continue;
                }
                let value = unsafe { value.as_ptr().cast::<T>().read_unaligned() };
                let expected = &reference[idx.as_slice()];
                assert_eq!(value, *expected,);
            }
            chunk_global_id += 1;
        }
    }
}
