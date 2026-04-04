use std::io;

use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::strides::NdIterExtensionStridesPtr;
use crate::storage::{ArrayStorage, ChunksLayout};
use crate::util::DimArray;

struct PlainStorage<A> {
    /// Arbitrary object that keeps the data pointer alive.
    allocation: A,
    /// # Invariants:
    /// - `data as usize % alignment == 0`
    data: *const u8,
    dtype: Dtype,
    shape: DimArray<usize>,
    /// strides in bytes units.
    /// # Invariants:
    /// - all(strides[i] % dtype.alignment() == 0)
    /// - len(strides) == len(shape)
    strides: DimArray<usize>,
    /// # Invariants:
    /// - len(chunk_shape) == len(shape)`
    /// - all(0 <= chunk_shape[i] <= shape[i])`
    /// - all(shape[i] == 0 or chunk_shape[i] > 0)` (chunk_shape[i] cant be zero unless shape[i] is also zero)
    chunks_layout: ChunksLayout,
}
impl<T> PlainStorage<Vec<T>> {
    pub fn from_ndarray<D>(array: ndarray::Array<T, D>) -> Self
    where
        T: Dtyped,
        D: ndarray::Dimension,
    {
        let dtype = T::dtype();
        let shape = array.shape().iter().cloned().collect::<DimArray<_>>();
        let strides = array
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).unwrap() * core::mem::size_of::<T>())
            .collect::<DimArray<_>>();
        let data = array.as_ptr() as *const u8;
        let allocation = array.into_raw_vec_and_offset().0;

        // no need for chunks in plain storage
        // TODO: use chunks for both performance and testing
        let chunk_shape = &shape;
        let chunks_layout = ChunksLayout::new(chunk_shape, &shape);

        Self {
            allocation,
            data,
            dtype,
            shape,
            strides,
            chunks_layout,
        }
    }
}
impl<A> ArrayStorage for PlainStorage<A> {
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
        _chunk_global_id: usize,
        chunk_idx: &[usize],
        buf: &mut [u8],
    ) -> io::Result<()> {
        let itemsize = self.dtype.itemsize() as usize;
        if buf.len() < self.chunks_layout.chunk_size * itemsize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer too small",
            ));
        }

        let ndim = self.shape.len();
        assert_eq!(chunk_idx.len(), ndim);

        let chunk_shape = &self.chunks_layout.chunk_shape;
        let base_src_offset = (0..ndim)
            .map(|dim| {
                let idx_elements = chunk_idx[dim] * chunk_shape[dim];
                assert!(idx_elements < self.shape[dim]);
                idx_elements * self.strides[dim]
            })
            .sum::<usize>();
        let src_data = unsafe { self.data.add(base_src_offset) };

        // TODO: fast path for contiguous storage
        let mut iter = NdIter::new(
            &self.chunks_layout.chunk_shape,
            NdIterExtensionStridesPtr::new(&self.strides, src_data),
        );
        let mut offset = 0;
        while let Some((_idx, src_ptr)) = iter.next() {
            let dst_ptr = unsafe { buf.as_mut_ptr().add(offset * itemsize) };
            unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize) };
            offset += 1;
        }
        Ok(())
    }
}
