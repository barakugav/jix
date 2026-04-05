use std::any::TypeId;
use std::cell::Cell;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, Write};
use std::ops::Range;
use std::path::Path;

use crate::archive::{ArchiveReader, ArchiveWriter, Section};
use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::iter::strides::{
    NdIterExtensionStridesPtr, NdIterExtensionStridesPtrMut, nd_iter_ext_logical_global_index,
};
use crate::schema::ArchiveType;
use crate::util::DimArray;
use crate::util::{MaybeOwned, default_strides};

use crate::block::{BlockSize, BlockTable, BlockTableAllocation};
use crate::codec::{Encoder, ReadContext};
use crate::util::{ceil_to_multiple, full_dim_array};
use crate::{NDIM_MAX, schema};

pub(crate) struct BlocksLayout {
    pub(crate) block_shape: DimArray<usize>,
    /// Number of blocks in each dimension.
    pub(crate) grid_shape: DimArray<usize>,
    /// Total items per block (`block_shape.iter().product()`).
    pub(crate) block_size: usize,
}

impl BlocksLayout {
    pub(crate) fn new(block_shape: &[usize], shape: &[usize]) -> Self {
        let block_shape: DimArray<_> = block_shape.try_into().unwrap();
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

pub struct Array<A> {
    pub(crate) storage: BlockTable<A>,
    pub(crate) shape: DimArray<usize>,
    pub(crate) blocks_layout: BlocksLayout,
}

impl<A> Array<A> {
    pub(crate) fn new(storage: BlockTable<A>, shape: &[usize], block_shape: &[usize]) -> Self {
        let blocks_layout = BlocksLayout::new(block_shape, shape);
        Self {
            storage,
            shape: shape.try_into().unwrap(),
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

    pub fn data(&self) -> ArrayData<'_, A> {
        let context = ReadContext::new().expect("failed to create read context");
        ArrayData::new(self, MaybeOwned::Owned(context))
    }

    pub fn data_ctx<'a>(&'a self, context: &'a ReadContext) -> ArrayData<'a, A> {
        ArrayData::new(self, MaybeOwned::Borrowed(context))
    }
}

impl Array<Owned> {
    pub fn from_ndarray<S, T, D>(
        array: &ndarray::ArrayBase<S, D, T>,
        block_shape: &[usize],
    ) -> io::Result<Self>
    where
        T: Dtyped,
        D: ndarray::Dimension,
        S: ndarray::RawData<Elem = T>,
    {
        let ndim = array.ndim();
        assert!(ndim < NDIM_MAX);
        assert_eq!(ndim, block_shape.len());
        let dtype = T::dtype();
        let itemsize = dtype.itemsize() as usize;
        let shape: DimArray<_> = array.shape().try_into().unwrap();

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
            Owned(crate::block::Owned {
                cdata,
                block_offsets,
            }),
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

pub struct ArrayData<'a, A> {
    array: &'a Array<A>,
    context: MaybeOwned<'a, ReadContext>,
    type_id: Cell<Option<TypeId>>,
}

impl<'a, A> ArrayData<'a, A> {
    fn new(array: &'a Array<A>, context: MaybeOwned<'a, ReadContext>) -> Self {
        Self {
            array,
            context,
            type_id: Cell::new(None),
        }
    }

    pub fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }

    pub fn ndim(&self) -> usize {
        self.array.ndim()
    }

    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn check_type<T: Dtyped>(&self) -> io::Result<()> {
        let type_id = TypeId::of::<T>();
        if self.type_id.get() == Some(type_id) {
            return Ok(());
        }

        let dtype = T::dtype();
        if self.dtype() != &dtype {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested type {:?} does not match array dtype {:?}",
                    dtype,
                    self.dtype()
                ),
            ));
        }

        self.type_id.set(Some(type_id));
        Ok(())
    }

    pub fn to_ndarray<T>(&self) -> io::Result<ndarray::ArrayD<T>>
    where
        T: Dtyped,
        A: ArrayAllocation,
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
        A: ArrayAllocation,
    {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();
        let itemsize = dtype.itemsize() as usize;
        self.check_type::<T>()?;
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

        let b_layout = &self.array.blocks_layout;
        let block_strides = default_strides(&b_layout.block_shape, itemsize);

        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = range.iter().map(|r| r.start).collect::<DimArray<_>>();
        let elem_end = range.iter().map(|r| r.end).collect::<DimArray<_>>();

        // Block-space begin/end for NdIter.
        let block_begin = range
            .iter()
            .zip(&b_layout.block_shape)
            .map(|(r, &b)| r.start / b)
            .collect::<DimArray<_>>();
        let block_end = range
            .iter()
            .zip(&b_layout.block_shape)
            .map(|(r, &b)| r.end.div_ceil(b))
            .collect::<DimArray<_>>();

        let mut block_iter = NdIter::new_with_begin(
            &block_begin,
            &block_end,
            (
                nd_iter_ext_logical_global_index(&b_layout.grid_shape, &block_begin),
                NdIterExtBlockOffsetSize::new(shape, &elem_begin, &elem_end, b_layout),
            ),
        );

        // Pre-allocate a buffer large enough for a full block.
        let full_buf_len = b_layout.block_size * itemsize;
        let mut tmp_buf = Vec::with_capacity(full_buf_len);
        unsafe { tmp_buf.set_len(full_buf_len) };
        let context = self.context.as_ref();
        while let Some((block_idx, (block_global_id, (block_inner_offset, block_size)))) =
            block_iter.next()
        {
            self.array
                .storage
                .read_block(block_global_id, &mut tmp_buf, context)?;

            // Navigate to the active region within the block buffer.
            let active_start = (0..ndim)
                .map(|dim| block_inner_offset[dim] * block_strides[dim])
                .sum::<usize>();
            let src_ptr = unsafe { tmp_buf.as_ptr().add(active_start) };

            // Map the active region's start to its position in the output array.
            let out_start = (0..ndim)
                .map(|dim| {
                    let full_idx =
                        block_idx[dim] * b_layout.block_shape[dim] + block_inner_offset[dim];
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

impl<A> Array<A> {
    pub fn write_to<W>(&self, writer: W) -> io::Result<()>
    where
        W: Write + Seek,
        A: ArrayAllocation,
    {
        let mut writer = ArchiveWriter::new(writer, schema::ArchiveType::ArrayV1)?;

        let header = schema::ArrayHeader {
            shape: self.shape.iter().cloned().map(|s| s as u64).collect(),
            block_shape: self
                .blocks_layout
                .block_shape
                .iter()
                .cloned()
                .map(|s| s as u64)
                .collect(),
        };
        writer.write_message(&header)?;

        self.storage.write_content(&mut writer)
    }
}

impl Array<Owned> {
    pub fn read_from_reader<R>(reader: R, len: u64) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_from_impl(
            reader,
            len,
            |reader, cdata_section, block_offsets_section| {
                Ok(Owned(crate::block::Owned::read_from(
                    reader,
                    cdata_section,
                    block_offsets_section,
                )?))
            },
        )
    }
}
impl Array<Mmap> {
    pub unsafe fn read_from_file_mmap(path: &Path, offset: u64, len: u64) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut reader = BufReader::new(file);
        reader.seek(io::SeekFrom::Start(offset))?;
        Self::read_from_impl(
            reader,
            len,
            |reader, cdata_section, block_offsets_section| {
                Ok(Mmap(crate::block::Mmap::read_from(
                    reader,
                    cdata_section,
                    block_offsets_section,
                    mmap,
                )?))
            },
        )
    }
}
impl<A> Array<A> {
    fn read_from_impl<R>(
        reader: R,
        len: u64,
        read_sections: impl FnOnce(&mut ArchiveReader<R>, Section, Section) -> io::Result<A>,
    ) -> io::Result<Self>
    where
        R: Read + Seek,
        A: ArrayAllocation,
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
        let shape = header
            .shape
            .iter()
            .cloned()
            .map(|s| s as usize)
            .collect::<DimArray<_>>();
        if header.block_shape.len() != ndim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "array block_shape has different ndim {} than shape {ndim}",
                    header.block_shape.len(),
                ),
            ));
        }
        let block_shape = header
            .block_shape
            .iter()
            .cloned()
            .map(|s| s as usize)
            .collect::<DimArray<_>>();
        let padded_shape = block_shape
            .iter()
            .zip(&shape)
            .map(|(&b, &s)| if s == 0 { 0 } else { ceil_to_multiple(s, b) })
            .collect::<DimArray<_>>();

        let blocks = BlockTable::read_content(&mut reader, read_sections)?;
        let expected_nitems = padded_shape.iter().product::<usize>();
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

        Ok(Self::new(blocks, &shape, &block_shape))
    }
}

pub trait ArrayAllocation {
    #[doc(hidden)]
    type __BlockTableAllocation: BlockTableAllocation;
    #[doc(hidden)]
    fn __block_table_allocation(&self) -> &Self::__BlockTableAllocation;
}
impl<A> BlockTableAllocation for A
where
    A: ArrayAllocation,
{
    fn cdata(&self) -> &[u8] {
        self.__block_table_allocation().cdata()
    }

    fn block_offsets(&self) -> &[u64] {
        self.__block_table_allocation().block_offsets()
    }
}
pub struct Owned(crate::block::Owned);
impl ArrayAllocation for Owned {
    type __BlockTableAllocation = crate::block::Owned;
    fn __block_table_allocation(&self) -> &Self::__BlockTableAllocation {
        &self.0
    }
}
pub struct Borrowed<'a>(crate::block::Borrowed<'a>);
impl<'a> ArrayAllocation for Borrowed<'a> {
    type __BlockTableAllocation = crate::block::Borrowed<'a>;
    fn __block_table_allocation(&self) -> &Self::__BlockTableAllocation {
        &self.0
    }
}
pub struct Mmap(crate::block::Mmap);
impl ArrayAllocation for Mmap {
    type __BlockTableAllocation = crate::block::Mmap;
    fn __block_table_allocation(&self) -> &Self::__BlockTableAllocation {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, Write};

    use ndarray::ArrayD;

    use crate::array::Owned;
    use crate::block::{BlockSize, BlockTable};
    use crate::codec::Encoder;
    use crate::dtype::Dtyped;
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
        let a = Array::from_ndarray(&src, block_shape).unwrap();
        a.data().to_ndarray().unwrap()
    }

    // -----------------------------------------------------------------------
    // Helper: build a BlockTable from pre-arranged typed blocks
    // -----------------------------------------------------------------------

    fn make_block_table<T: Dtyped>(blocks: &[&[T]]) -> BlockTable<Owned> {
        let block_len = blocks[0].len() as BlockSize;
        let data: Vec<u8> = blocks
            .iter()
            .flat_map(|b| unsafe { cast_slice::<T, u8>(b) }.iter().copied())
            .collect();
        let mut encoder = Encoder::new(3).unwrap();
        let blocks =
            BlockTable::build_from_data(&data, T::dtype(), block_len, &mut encoder).unwrap();
        let blocks = unsafe { blocks.swap_allocation(|alloc| Owned(alloc)) };
        blocks
    }

    fn array<T: Dtyped>(blocks: &[&[T]], shape: &[usize], block_shape: &[usize]) -> Array<Owned> {
        Array::new(make_block_table(blocks), shape, block_shape)
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
        let got: ArrayD<u8> = a.data().to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![0, 1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn to_ndarray_1d_two_blocks() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn to_ndarray_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let got: ArrayD<i32> = a.data().to_ndarray().unwrap();
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
        let got: ArrayD<u8> = a.data().to_ndarray().unwrap();
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
        let got: ArrayD<u8> = a.data().sub_ndarray(&[0..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn sub_ndarray_1d_aligned_second_block() {
        // range [3..6) → output shape [3], values [3,4,5]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().sub_ndarray(&[3..6]).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3], vec![3, 4, 5]).unwrap());
    }

    #[test]
    fn sub_ndarray_1d_cross_block_boundary() {
        // range [1..5) → output shape [4], values [1,2,3,4]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().sub_ndarray(&[1..5]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![1, 2, 3, 4]).unwrap()
        );
    }

    #[test]
    fn sub_ndarray_1d_within_single_block() {
        // range [1..2) → output shape [1], value [1]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().sub_ndarray(&[1..2]).unwrap();
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
        let got: ArrayD<u8> = a.data().sub_ndarray(&[1..3, 2..5]).unwrap();
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
        assert_eq!(roundtrip(&src, &[4]), src.into_dyn());
    }

    #[test]
    fn from_ndarray_1d_multi_block() {
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        assert_eq!(roundtrip(&src, &[3]), src.into_dyn());
    }

    #[test]
    fn from_ndarray_1d_with_padding() {
        // size 5, block 3 → padded to 6; shape reported as 5
        let src = ndarray::array![0u8, 1, 2, 3, 4];
        let a = Array::from_ndarray(&src, &[3]).unwrap();
        assert_eq!(a.shape(), &[5]);
        let got: ArrayD<u8> = a.data().to_ndarray().unwrap();
        assert_eq!(got, src.into_dyn());
    }

    #[test]
    fn from_ndarray_1d_i32() {
        let src = ndarray::array![0i32, 10, 20, 30, 40, 50, 60, 70];
        assert_eq!(roundtrip(&src, &[4]), src.into_dyn());
    }

    #[test]
    fn from_ndarray_1d_f32() {
        let src = ndarray::array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        assert_eq!(roundtrip(&src, &[4]), src.into_dyn());
    }

    #[test]
    fn from_ndarray_block_larger_than_shape_is_clamped() {
        // block_shape [10] > array size [4]; should clamp to [4]
        let src = ndarray::array![0u8, 1, 2, 3];
        let a = Array::from_ndarray(&src, &[10]).unwrap();
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn from_ndarray_1d_noncontiguous() {
        // Step-2 slice of [0..10] → [0, 2, 4, 6, 8]
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = src.slice(ndarray::s![..;2]);
        let a = Array::from_ndarray(&view, &[3]).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert_eq!(
            a.data().to_ndarray::<u8>().unwrap(),
            ndarray::array![0u8, 2, 4, 6, 8].into_dyn()
        );
    }

    // -----------------------------------------------------------------------
    // from_ndarray — metadata
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_metadata() {
        let src = ndarray::array![0i32, 1, 2, 3, 4, 5];
        let a = Array::from_ndarray(&src, &[3]).unwrap();
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
        assert_eq!(roundtrip(&src, &[2, 3]), src.into_dyn());
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
        let a = Array::from_ndarray(&src, &[2, 3]).unwrap();
        assert_eq!(a.shape(), &[3, 5]);
        assert_eq!(a.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn from_ndarray_2d_noncontiguous() {
        // Fortran-order (column-major) array
        let src = ndarray::Array2::<u8>::from_shape_vec(
            ndarray::ShapeBuilder::f((3, 4)),
            (0..12).collect(),
        )
        .unwrap();
        assert_eq!(roundtrip(&src, &[2, 2]), src.into_dyn());
    }

    // -----------------------------------------------------------------------
    // from_ndarray + sub_ndarray integration
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_then_sub_ndarray_1d() {
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        let a = Array::from_ndarray(&src, &[3]).unwrap();
        let got: ArrayD<u8> = a.data().sub_ndarray(&[1..5]).unwrap();
        assert_eq!(got, ndarray::array![1u8, 2, 3, 4].into_dyn());
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
        let a = Array::from_ndarray(&src, &[2, 3]).unwrap();
        let got: ArrayD<u8> = a.data().sub_ndarray(&[1..3, 2..5]).unwrap();
        assert_eq!(got, ndarray::array![[8u8, 9, 10], [14, 15, 16]].into_dyn());
    }

    // -----------------------------------------------------------------------
    // write_to / read_from round-trip
    // -----------------------------------------------------------------------

    fn array_round_trip<T, S, D>(
        src: &ndarray::ArrayBase<S, D>,
        block_shape: &[usize],
    ) -> Array<Owned>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::from_ndarray(&src, block_shape).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        a.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let len = bytes.len() as u64;
        Array::read_from_reader(Cursor::new(bytes), len).unwrap()
    }

    #[test]
    fn write_read_1d_single_block() {
        let src = ndarray::array![0u8, 1, 2, 3];
        let a2 = array_round_trip::<u8, _, _>(&src, &[4]);
        assert_eq!(a2.shape(), &[4]);
        assert_eq!(a2.ndim(), 1);
        assert_eq!(a2.dtype(), &u8::dtype());
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_multi_block() {
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        let a2 = array_round_trip::<u8, _, _>(&src, &[3]);
        assert_eq!(a2.shape(), &[6]);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_with_padding() {
        // size 5, block 3 → padded to 6; shape is preserved as 5
        let src = ndarray::array![0u8, 1, 2, 3, 4];
        let a2 = array_round_trip::<u8, _, _>(&src, &[3]);
        assert_eq!(a2.shape(), &[5]);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_i32() {
        let src = ndarray::array![0i32, 10, 20, 30, 40, 50, 60, 70];
        let a2 = array_round_trip::<i32, _, _>(&src, &[4]);
        assert_eq!(a2.dtype(), &i32::dtype());
        assert_eq!(a2.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_f32() {
        let src = ndarray::array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5];
        let a2 = array_round_trip::<f32, _, _>(&src, &[3]);
        assert_eq!(a2.dtype(), &f32::dtype());
        assert_eq!(a2.data().to_ndarray::<f32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_2d() {
        #[rustfmt::skip]
        let src = ndarray::array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a2 = array_round_trip::<u8, _, _>(&src, &[2, 3]);
        assert_eq!(a2.shape(), &[4, 6]);
        assert_eq!(a2.ndim(), 2);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_2d_with_padding() {
        // shape [3,5], block [2,3] → padded to [4,6]; shape preserved as [3,5]
        #[rustfmt::skip]
        let src = ndarray::array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a2 = array_round_trip::<i32, _, _>(&src, &[2, 3]);
        assert_eq!(a2.shape(), &[3, 5]);
        assert_eq!(a2.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_file() {
        let src = ndarray::array![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::from_ndarray(&src, &[4]).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        a.write_to(std::fs::File::create(&path).unwrap()).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        let a2 = Array::read_from_reader(file, len).unwrap();

        assert_eq!(a2.shape(), &[12]);
        assert_eq!(a2.dtype(), &u32::dtype());
        assert_eq!(a2.data().to_ndarray::<u32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_nonzero_offset() {
        // Write three arrays with padding between them; read each back by seeking to its recorded offset.
        const PAD: usize = 177;
        // src0: 1D u8
        let src0 = ndarray::array![0u8, 1, 2, 3];
        // src1: 2D i32, shape [3, 4]
        #[rustfmt::skip]
        let src1 = ndarray::array![
            [10i32, 20, 30, 40],
            [50,    60, 70, 80],
            [90,   100, 110, 120],
        ];
        // src2: 3D f32, shape [2, 2, 3]
        let src2 = ndarray::Array3::<f32>::from_shape_vec(
            [2, 2, 3],
            (0..12).map(|i| i as f32 * 0.5).collect(),
        )
        .unwrap();

        let mut buf = Cursor::new(Vec::<u8>::new());

        // arr 0
        let a0 = Array::from_ndarray(&src0, &[4]).unwrap();
        a0.write_to(&mut buf).unwrap();
        let len0 = buf.stream_position().unwrap();
        // pad
        buf.write_all(vec![0u8; PAD].as_slice()).unwrap();
        let off1 = buf.stream_position().unwrap();

        // arr 1
        let a1 = Array::from_ndarray(&src1, &[2, 2]).unwrap();
        a1.write_to(&mut buf).unwrap();
        let len1 = buf.stream_position().unwrap() - off1;
        // pad
        buf.write_all(vec![0u8; PAD].as_slice()).unwrap();
        let off2 = buf.stream_position().unwrap();

        // arr 2
        let a2 = Array::from_ndarray(&src2, &[1, 2, 3]).unwrap();
        a2.write_to(&mut buf).unwrap();
        let len2 = buf.stream_position().unwrap() - off2;

        let bytes = buf.into_inner();

        // Read array 0 (at offset 0).
        let r0 = Array::read_from_reader(Cursor::new(&bytes), len0).unwrap();
        assert_eq!(r0.shape(), &[4]);
        assert_eq!(r0.ndim(), 1);
        assert_eq!(r0.dtype(), &u8::dtype());
        assert_eq!(r0.data().to_ndarray::<u8>().unwrap(), src0.into_dyn());

        // Read array 1 (padded offset, 2D).
        let r1 = Array::read_from_reader(Cursor::new(&bytes[off1 as usize..]), len1).unwrap();
        assert_eq!(r1.shape(), &[3, 4]);
        assert_eq!(r1.ndim(), 2);
        assert_eq!(r1.dtype(), &i32::dtype());
        assert_eq!(r1.data().to_ndarray::<i32>().unwrap(), src1.into_dyn());

        // Read array 2 (padded offset, 3D).
        let r2 = Array::read_from_reader(Cursor::new(&bytes[off2 as usize..]), len2).unwrap();
        assert_eq!(r2.shape(), &[2, 2, 3]);
        assert_eq!(r2.ndim(), 3);
        assert_eq!(r2.dtype(), &f32::dtype());
        assert_eq!(r2.data().to_ndarray::<f32>().unwrap(), src2.into_dyn());
    }

    // -----------------------------------------------------------------------
    // read_from_file_mmap round-trip
    // -----------------------------------------------------------------------

    fn array_mmap_round_trip<T, S, D>(
        src: &ndarray::ArrayBase<S, D>,
        block_shape: &[usize],
        tmp_file: &tempfile::NamedTempFile,
    ) -> super::Array<super::Mmap>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::from_ndarray(&src, block_shape).unwrap();
        let path = tmp_file.path().to_path_buf();
        a.write_to(std::fs::File::create(&path).unwrap()).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        unsafe { super::Array::<super::Mmap>::read_from_file_mmap(&path, 0, len) }.unwrap()
    }

    #[test]
    fn mmap_read_1d_single_block() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = ndarray::array![0u8, 1, 2, 3];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[4], &tmp);
        assert_eq!(a2.shape(), &[4]);
        assert_eq!(a2.ndim(), 1);
        assert_eq!(a2.dtype(), &u8::dtype());
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn mmap_read_1d_multi_block() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[3], &tmp);
        assert_eq!(a2.shape(), &[6]);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn mmap_read_1d_i32() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = ndarray::array![0i32, 10, 20, 30, 40, 50, 60, 70];
        let a2 = array_mmap_round_trip::<i32, _, _>(&src, &[4], &tmp);
        assert_eq!(a2.dtype(), &i32::dtype());
        assert_eq!(a2.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn mmap_read_2d() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        #[rustfmt::skip]
        let src = ndarray::array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[2, 3], &tmp);
        assert_eq!(a2.shape(), &[4, 6]);
        assert_eq!(a2.ndim(), 2);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn mmap_read_nonzero_offset() {
        // Write two arrays back-to-back; read the second via its offset.
        let src1 = ndarray::array![0u8, 1, 2, 3];
        let src2 = ndarray::array![10u8, 11, 12, 13, 14, 15];
        let a1 = Array::from_ndarray(&src1, &[4]).unwrap();
        let a2_arr = Array::from_ndarray(&src2, &[3]).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let mut f = std::fs::File::create(&path).unwrap();
        a1.write_to(&mut f).unwrap();
        let offset = f.metadata().unwrap().len();
        a2_arr.write_to(&mut f).unwrap();
        let total_len = f.metadata().unwrap().len();
        drop(f);

        let len2 = total_len - offset;
        let read = unsafe { super::Array::<super::Mmap>::read_from_file_mmap(&path, offset, len2) }
            .unwrap();
        assert_eq!(read.shape(), &[6]);
        assert_eq!(read.data().to_ndarray::<u8>().unwrap(), src2.into_dyn());
        drop(tmp_file);
    }

    // -----------------------------------------------------------------------
    // type_id cache tests
    // -----------------------------------------------------------------------

    #[test]
    fn check_type_cached_correct_dtype_multiple_reads() {
        // Repeated reads with the correct dtype should all succeed (the cached
        // TypeId path is exercised from the second call onward).
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let data = a.data();
        let expected = ArrayD::from_shape_vec(vec![4], vec![0u8, 1, 2, 3]).unwrap();
        for _ in 0..4 {
            assert_eq!(data.to_ndarray::<u8>().unwrap(), expected);
        }
    }

    #[test]
    fn check_type_interleaved_correct_and_incorrect_dtype() {
        // Reads with the wrong dtype should always return an error, even after
        // a successful read has primed the TypeId cache.
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let data = a.data();
        let expected = ArrayD::from_shape_vec(vec![4], vec![0u8, 1, 2, 3]).unwrap();

        // First two reads: wrong types — must error before cache is primed.
        assert!(data.to_ndarray::<u32>().is_err());
        assert!(data.to_ndarray::<i8>().is_err());
        // Third read: correct — primes the cache.
        assert_eq!(data.to_ndarray::<u8>().unwrap(), expected);
        // Fourth read: wrong type — must error even after cache is primed.
        assert!(data.to_ndarray::<u32>().is_err());
        // Fifth read: correct — cache still valid.
        assert_eq!(data.to_ndarray::<u8>().unwrap(), expected);
    }
}
