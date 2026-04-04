use std::io;
use std::ops::Range;

use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{
    NdIterExtensionStridesPtr, NdIterExtensionStridesPtrMut, nd_iter_ext_logical_global_index,
};
use crate::storage::Storage;
use crate::util::DimArray;
use crate::util::default_strides;
use std::borrow::Cow;

use crate::NDIM_MAX;
use crate::storage::BlockSize;
use crate::storage::block::BlockTable;
use crate::storage::codec::Encoder;
use crate::util::{ceil_to_multiple, full_dim_array};

pub(crate) struct BlocksLayout {
    pub(crate) block_shape: DimArray<usize>,
    /// Number of blocks in each dimension.
    pub(crate) grid_shape: DimArray<usize>,
    /// Total items per block (`block_shape.iter().product()`).
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
}

impl Array<BlockTable<'static>> {
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
        let b_layout = BlocksLayout::new(&block_shape, &shape);
        let nblocks = b_layout.grid_shape.iter().product::<usize>();

        let mut block_iter = NdIter::new(
            &b_layout.grid_shape,
            NdIterExtBlockOffsetSize::new(&shape, &full_dim_array(0, ndim), &shape, &b_layout),
        );

        let mut encoder = Encoder::new(3)?;
        let mut cdata = Vec::<u8>::new();
        let mut block_offsets =
            Vec::<u64>::with_capacity(if nblocks == 0 { 0 } else { nblocks + 1 });
        if nblocks > 0 {
            block_offsets.push(0);
        }
        let block_capacity_bytes = b_layout.block_size * itemsize;
        let max_blk_cdata_len = encoder.encode_bound(block_capacity_bytes);
        let mut tmp_block_data = Vec::<u8>::with_capacity(block_capacity_bytes);
        let tmp_block_strides = default_strides(&block_shape, itemsize);
        let strides = array
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).unwrap() * size_of::<T>())
            .collect::<DimArray<_>>();
        while let Some((block_idx, (block_inner_offset, block_size))) = block_iter.next() {
            debug_assert!(block_inner_offset.iter().all(|&o| o == 0));

            // Init chunk data to zeros.
            // The padding elements (if any) will not be written by the iter below, so they will stay zeros.
            tmp_block_data.clear();
            tmp_block_data.resize(block_capacity_bytes, 0);

            // TODO: fast path for contiguous data
            let initial_arr_offset = (0..ndim)
                .map(|dim| {
                    let idx = block_idx[dim] * b_layout.block_shape[dim] + block_inner_offset[dim];
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
            dtype,
            padded_shape.iter().product::<usize>(),
            b_layout.block_size as BlockSize,
        );

        Ok(Self {
            storage: blocks,
            shape,
            blocks_layout: b_layout,
        })
    }
}

impl<S> Array<S>
where
    S: Storage,
{
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
        let out_shape = range
            .iter()
            .map(|r| r.end - r.start)
            .collect::<DimArray<_>>();
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
                    let full_idx = block_idx[dim] * bl.block_shape[dim] + block_inner_offset[dim];
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
    // from_ndarray roundtrip helper
    // -----------------------------------------------------------------------

    fn roundtrip<T, S, D>(src: &ndarray::ArrayBase<S, D>, block_shape: &[usize]) -> ArrayD<T>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::from_ndarray(&src.view(), block_shape).unwrap();
        a.to_ndarray().unwrap()
    }

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
        MockStorage {
            blocks: byte_blocks,
            dtype,
            block_len,
        }
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
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![0, 1, 2, 3]).unwrap()
        );
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
        assert_eq!(got, ArrayD::from_shape_vec(vec![3], vec![3, 4, 5]).unwrap());
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
        assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![1]).unwrap());
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

    // -----------------------------------------------------------------------
    // from_ndarray — 1D
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_1d_single_block() {
        let src = ndarray::array![0u8, 1, 2, 3];
        assert_eq!(roundtrip(&src, &[4]), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_1d_multi_block() {
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        assert_eq!(roundtrip(&src, &[3]), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_1d_with_padding() {
        // size 5, block 3 → padded to 6; shape reported as 5
        let src = ndarray::array![0u8, 1, 2, 3, 4];
        let a = Array::from_ndarray(&src.view(), &[3]).unwrap();
        assert_eq!(a.shape(), &[5]);
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(got, src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_1d_i32() {
        let src = ndarray::array![0i32, 10, 20, 30, 40, 50, 60, 70];
        assert_eq!(roundtrip(&src, &[4]), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_1d_f32() {
        let src = ndarray::array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        assert_eq!(roundtrip(&src, &[4]), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_block_larger_than_shape_is_clamped() {
        // block_shape [10] > array size [4]; should clamp to [4]
        let src = ndarray::array![0u8, 1, 2, 3];
        let a = Array::from_ndarray(&src.view(), &[10]).unwrap();
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.to_ndarray::<u8>().unwrap(), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_1d_noncontiguous() {
        // Step-2 slice of [0..10] → [0, 2, 4, 6, 8]
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = src.slice(ndarray::s![..;2]);
        let a = Array::from_ndarray(&view, &[3]).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert_eq!(
            a.to_ndarray::<u8>().unwrap(),
            ndarray::array![0u8, 2, 4, 6, 8].view().into_dyn()
        );
    }

    // -----------------------------------------------------------------------
    // from_ndarray — metadata
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_metadata() {
        let src = ndarray::array![0i32, 1, 2, 3, 4, 5];
        let a = Array::from_ndarray(&src.view(), &[3]).unwrap();
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.dtype(), &i32::dtype());
    }

    // -----------------------------------------------------------------------
    // from_ndarray — 2D
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_2d() {
        #[rustfmt::skip]
        let src = ndarray::array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        assert_eq!(roundtrip(&src, &[2, 3]), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_2d_with_padding() {
        // shape [3,5], block [2,3] → padded to [4,6]; shape reported as [3,5]
        #[rustfmt::skip]
        let src = ndarray::array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::from_ndarray(&src.view(), &[2, 3]).unwrap();
        assert_eq!(a.shape(), &[3, 5]);
        assert_eq!(a.to_ndarray::<i32>().unwrap(), src.view().into_dyn());
    }

    #[test]
    fn from_ndarray_2d_noncontiguous() {
        // Fortran-order (column-major) array
        let src = ndarray::Array2::<u8>::from_shape_vec(
            ndarray::ShapeBuilder::f((3, 4)),
            (0..12).collect(),
        )
        .unwrap();
        assert_eq!(roundtrip(&src, &[2, 2]), src.view().into_dyn());
    }

    // -----------------------------------------------------------------------
    // from_ndarray + sub_ndarray integration
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_then_sub_ndarray_1d() {
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        let a = Array::from_ndarray(&src.view(), &[3]).unwrap();
        let got: ArrayD<u8> = a.sub_ndarray(&[1..5]).unwrap();
        assert_eq!(got, ndarray::array![1u8, 2, 3, 4].view().into_dyn());
    }

    #[test]
    fn from_ndarray_then_sub_ndarray_2d() {
        #[rustfmt::skip]
        let src = ndarray::array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a = Array::from_ndarray(&src.view(), &[2, 3]).unwrap();
        let got: ArrayD<u8> = a.sub_ndarray(&[1..3, 2..5]).unwrap();
        assert_eq!(
            got,
            ndarray::array![[8u8, 9, 10], [14, 15, 16]].view().into_dyn()
        );
    }
}
