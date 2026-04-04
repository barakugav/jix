use std::io;
use std::ops::Range;

use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::chunk::NdIterExtChunkOffsetSize;
use crate::iter::strides::{NdIterExtensionStridesPtrMut, nd_iter_ext_logical_global_index};
use crate::storage::ArrayStorage;
use crate::util::DimArray;

pub struct Array<S> {
    pub(crate) storage: S,
}
impl<S> Array<S>
where
    S: ArrayStorage,
{
    pub(crate) fn new(storage: S) -> Self {
        Self { storage }
    }

    pub fn dtype(&self) -> &Dtype {
        self.storage.dtype()
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    pub fn to_ndarray<T>(&self) -> io::Result<ndarray::ArrayD<T>>
    where
        T: Dtyped,
    {
        let full_range = self
            .shape()
            .iter()
            .map(|&dim| 0..dim)
            .collect::<DimArray<_>>();
        self.sub_ndarray(&full_range)
    }

    pub fn sub_ndarray<T>(&self, range: &[Range<usize>]) -> io::Result<ndarray::ArrayD<T>>
    where
        T: Dtyped,
    {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();
        let itemsize = dtype.itemsize() as usize;
        assert_eq!(dtype, &T::dtype());
        let mut array = ndarray::ArrayD::uninit(shape);
        let strides = array
            .strides()
            .iter()
            .map(|&s| s as usize)
            .collect::<DimArray<_>>();
        // let array_ptr = array.as_mut_ptr() as *mut u8;

        let c_layout = self.storage.chunks_layout();
        let chunk_begin = range
            .iter()
            .zip(&c_layout.chunk_shape)
            .map(|(r, &c)| r.start / c)
            .collect::<DimArray<_>>();
        let chunk_end = range
            .iter()
            .zip(&c_layout.chunk_shape)
            .map(|(r, &c)| r.end.div_ceil(c))
            .collect::<DimArray<_>>();
        let mut chunk_iter = NdIter::new_with_begin(
            &chunk_begin,
            &chunk_end,
            (
                nd_iter_ext_logical_global_index(&c_layout.chunk_space_shape, &chunk_begin),
                NdIterExtChunkOffsetSize::new(shape, &chunk_begin, &chunk_end, c_layout),
            ),
        );

        let mut tmp_buf = Vec::new();
        while let Some((chunk_idx, (chunk_global_id, (chunk_inner_offset, chunk_size)))) =
            chunk_iter.next()
        {
            let buf_len = chunk_size.iter().product::<usize>() * itemsize;
            tmp_buf.clear();
            tmp_buf.reserve(buf_len);
            unsafe { tmp_buf.set_len(buf_len) };
            self.storage
                .get_chunk_data(chunk_global_id, &chunk_idx, tmp_buf.as_mut_slice())?;

            let array_initial_offset = (0..ndim)
                .map(|dim| {
                    let idx = chunk_idx[dim] * c_layout.chunk_shape[dim] + chunk_inner_offset[dim];
                    assert!(idx < shape[dim]);
                    idx * strides[dim]
                })
                .sum::<usize>();
            let array_ptr = unsafe { array.as_mut_ptr().cast::<u8>().add(array_initial_offset) };
            let mut iter = NdIter::new(
                &chunk_size,
                NdIterExtensionStridesPtrMut::new(&strides, array_ptr),
            );
            let mut buf_offset = 0;
            while let Some((_idx, dst_ptr)) = iter.next() {
                let src_ptr = unsafe { tmp_buf.as_ptr().add(buf_offset) };
                unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize) };
                buf_offset += itemsize;
            }
        }

        Ok(unsafe { array.assume_init() })
    }
}
