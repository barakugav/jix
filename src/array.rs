use std::io;
use std::ops::Range;

use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{NdIterExtensionStridesPtr, NdIterExtensionStridesPtrMut, nd_iter_ext_logical_global_index};
use crate::util::default_strides;
use crate::storage::Storage;
use crate::util::DimArray;

pub(crate) struct BlocksLayout {
    pub(crate) block_shape: DimArray<usize>,
    /// Number of blocks in each dimension.
    pub(crate) grid_shape: DimArray<usize>,
    /// Total items per block: `block_shape.iter().product()`.
    pub(crate) block_size: usize,
}

impl BlocksLayout {
    pub(crate) fn new(block_shape: &[usize], shape: &[usize]) -> Self {
        let block_shape = block_shape.iter().cloned().collect::<DimArray<_>>();
        let grid_shape = shape
            .iter()
            .zip(&block_shape)
            .map(|(&s, &b)| s.div_ceil(b))
            .collect();
        let block_size = block_shape.iter().product();
        Self {
            block_shape,
            grid_shape,
            block_size,
        }
    }
}

pub struct Array<S> {
    pub(crate) storage: S,
    pub(crate) shape: DimArray<usize>,
    pub(crate) blocks_layout: BlocksLayout,
}

impl<S> Array<S>
where
    S: Storage,
{
    pub(crate) fn new(storage: S, shape: &[usize], block_shape: &[usize]) -> Self {
        let blocks_layout = BlocksLayout::new(block_shape, shape);
        Self {
            storage,
            shape: shape.iter().cloned().collect(),
            blocks_layout,
        }
    }

    pub fn dtype(&self) -> &Dtype {
        self.storage.dtype()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
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
        // Output is sized to the requested range, not the full array shape.
        let out_shape = range.iter().map(|r| r.end - r.start).collect::<DimArray<_>>();
        let mut array = ndarray::ArrayD::uninit(&out_shape[..]);
        let out_strides = array
            .strides()
            .iter()
            .map(|&s| s as usize * itemsize)
            .collect::<DimArray<_>>();

        let bl = &self.blocks_layout;
        let block_strides = default_strides(&bl.block_shape, itemsize);

        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = range.iter().map(|r| r.start).collect::<DimArray<_>>();
        let elem_end = range.iter().map(|r| r.end).collect::<DimArray<_>>();

        // Block-space begin/end for NdIter.
        let block_begin = range
            .iter()
            .zip(&bl.block_shape)
            .map(|(r, &b)| r.start / b)
            .collect::<DimArray<_>>();
        let block_end = range
            .iter()
            .zip(&bl.block_shape)
            .map(|(r, &b)| r.end.div_ceil(b))
            .collect::<DimArray<_>>();

        let mut block_iter = NdIter::new_with_begin(
            &block_begin,
            &block_end,
            (
                nd_iter_ext_logical_global_index(&bl.grid_shape, &block_begin),
                NdIterExtBlockOffsetSize::new(shape, &elem_begin, &elem_end, bl),
            ),
        );

        // Pre-allocate a buffer large enough for a full block.
        let full_buf_len = bl.block_size * itemsize;
        let mut tmp_buf = Vec::with_capacity(full_buf_len);
        unsafe { tmp_buf.set_len(full_buf_len) };

        while let Some((block_idx, (block_global_id, (block_inner_offset, block_size)))) =
            block_iter.next()
        {
            self.storage.read_block(block_global_id, &mut tmp_buf)?;

            // Navigate to the active region within the block buffer.
            let active_start = (0..ndim)
                .map(|dim| block_inner_offset[dim] * block_strides[dim])
                .sum::<usize>();
            let src_ptr = unsafe { tmp_buf.as_ptr().add(active_start) };

            // Map the active region's start to its position in the output array.
            let out_start = (0..ndim)
                .map(|dim| {
                    let full_idx =
                        block_idx[dim] * bl.block_shape[dim] + block_inner_offset[dim];
                    let out_idx = full_idx - range[dim].start;
                    out_idx * out_strides[dim]
                })
                .sum::<usize>();
            let dst_ptr = unsafe { array.as_mut_ptr().cast::<u8>().add(out_start) };

            let mut iter = NdIter::new(
                &block_size,
                (
                    NdIterExtensionStridesPtr::new(&block_strides, src_ptr),
                    NdIterExtensionStridesPtrMut::new(&out_strides, dst_ptr),
                ),
            );
            while let Some((_idx, (src, dst))) = iter.next() {
                unsafe { std::ptr::copy_nonoverlapping(src, dst, itemsize) };
            }
        }

        Ok(unsafe { array.assume_init() })
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use ndarray::ArrayD;

    use crate::dtype::{Dtype, Dtyped};
    use crate::storage::{BlockSize, Storage};
    use crate::util::cast_slice;

    use super::Array;

    // -----------------------------------------------------------------------
    // MockStorage: serves pre-built typed blocks
    // -----------------------------------------------------------------------

    struct MockStorage {
        blocks: Vec<Vec<u8>>,
        dtype: Dtype,
        block_len: BlockSize,
    }

    impl Storage for MockStorage {
        fn dtype(&self) -> &Dtype {
            &self.dtype
        }
        fn nitems(&self) -> usize {
            self.blocks.len() * self.block_len as usize
        }
        fn block_len(&self) -> BlockSize {
            self.block_len
        }
        fn read_block(&self, idx: usize, buf: &mut [u8]) -> io::Result<()> {
            let src = &self.blocks[idx];
            buf[..src.len()].copy_from_slice(src);
            Ok(())
        }
    }

    fn mock<T: Dtyped>(blocks: &[&[T]]) -> MockStorage {
        let block_len = blocks[0].len() as BlockSize;
        let dtype = T::dtype();
        let byte_blocks = blocks
            .iter()
            .map(|b| unsafe { cast_slice::<T, u8>(b) }.to_vec())
            .collect();
        MockStorage { blocks: byte_blocks, dtype, block_len }
    }

    fn array<T: Dtyped>(
        blocks: &[&[T]],
        shape: &[usize],
        block_shape: &[usize],
    ) -> Array<MockStorage> {
        Array::new(mock(blocks), shape, block_shape)
    }

    // -----------------------------------------------------------------------
    // Accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn dtype_shape_ndim() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        assert_eq!(a.dtype(), &u8::dtype());
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.ndim(), 1);
    }

    // -----------------------------------------------------------------------
    // to_ndarray — 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4], vec![0, 1, 2, 3]).unwrap());
    }

    #[test]
    fn to_ndarray_1d_two_blocks() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn to_ndarray_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let got: ArrayD<i32> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![8], vec![10, 20, 30, 40, 50, 60, 70, 80]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // to_ndarray — 2D
    // Block-major order: block [r,c] = row-major grid index r*ncols_blocks+c.
    // shape=[4,6], block_shape=[2,3] → grid 2×2:
    //   block0=[0,0]: rows 0-1, cols 0-2 → 0,1,2,6,7,8
    //   block1=[0,1]: rows 0-1, cols 3-5 → 3,4,5,9,10,11
    //   block2=[1,0]: rows 2-3, cols 0-2 → 12,13,14,18,19,20
    //   block3=[1,1]: rows 2-3, cols 3-5 → 15,16,17,21,22,23
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_2d() {
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            &[4, 6],
            &[2, 3],
        );
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4, 6], (0u8..24).collect()).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // sub_ndarray — 1D
    // -----------------------------------------------------------------------

    #[test]
    fn sub_ndarray_1d_full_range() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.sub_ndarray(&[0..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn sub_ndarray_1d_aligned_second_block() {
        // range [3..6) → output shape [3], values [3,4,5]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.sub_ndarray(&[3..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3], vec![3, 4, 5]).unwrap()
        );
    }

    #[test]
    fn sub_ndarray_1d_cross_block_boundary() {
        // range [1..5) → output shape [4], values [1,2,3,4]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.sub_ndarray(&[1..5]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![1, 2, 3, 4]).unwrap()
        );
    }

    #[test]
    fn sub_ndarray_1d_within_single_block() {
        // range [1..2) → output shape [1], value [1]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.sub_ndarray(&[1..2]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1], vec![1]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // sub_ndarray — 2D
    // shape=[4,6], block_shape=[2,3], data as in to_ndarray_2d test.
    // range=[1..3, 2..5] → output shape [2,3]:
    //   [8,  9,  10]
    //   [14, 15, 16]
    // -----------------------------------------------------------------------

    #[test]
    fn sub_ndarray_2d() {
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            &[4, 6],
            &[2, 3],
        );
        let got: ArrayD<u8> = a.sub_ndarray(&[1..3, 2..5]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3], vec![8, 9, 10, 14, 15, 16]).unwrap()
        );
    }
}
