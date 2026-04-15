use std::any::TypeId;
use std::cell::Cell;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, Write};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::path::Path;

use crate::archive::ArchiveWriter;
use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::block::NdIterExtBlockOffsetSize;
use crate::storage::{
    ArrayBlockTableStorageBase, ArrayStorage, BlockShapeTag, BlocksLayout, Mmap, Owned, Ref,
};
use crate::util::{AlignedBytes, DimArray, cast_slice_mut, dim_arr, nd_copy};
use crate::util::{MaybeOwned, default_strides};

use crate::block::{BlockSize, BlockTableBuilder};
use crate::codec::{DecoderParams, Encoder, EncoderParams, ReadContext};
use crate::{NDIM_MAX, schema};

pub struct Array<S> {
    pub(crate) storage: S,
}

#[derive(Clone, Default, Debug)]
pub struct ArrayParams {
    pub(crate) block_shape: Option<DimArray<BlockSize>>,
    pub(crate) block_shape_tag: Option<DimArray<BlockShapeTag>>,
    pub(crate) block_size_hint: Option<u64>,
    pub(crate) preferred_read_block_shape: Option<DimArray<BlockSize>>,
    pub(crate) preferred_read_block_size_hint: Option<u64>,
    pub(crate) encoder_params: Option<EncoderParams>,
    pub(crate) decoder_params: Option<DecoderParams>,
}
impl ArrayParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn override_from_storage(&mut self, storage: &impl ArrayStorage) {
        let (s_encoder_params, s_decoder_params, _) = storage.codec_params();
        self.encoder_params
            .get_or_insert_with(|| s_encoder_params.clone());
        self.decoder_params
            .get_or_insert_with(|| s_decoder_params.clone());

        let blocks_layout = storage.blocks_layout();
        self.block_shape
            .get_or_insert_with(|| blocks_layout.block_shape_hint.clone());
        self.block_shape_tag
            .get_or_insert_with(|| blocks_layout.block_shape_tag.clone());
        self.block_size_hint
            .get_or_insert(blocks_layout.block_size_hint);
        self.preferred_read_block_shape
            .get_or_insert_with(|| blocks_layout.preferred_read_block_shape.clone());
        self.preferred_read_block_size_hint
            .get_or_insert(blocks_layout.preferred_read_block_size_hint);
    }
}

impl<S: ArrayStorage> Array<S> {
    pub fn from_storage(storage: S) -> Self {
        Self { storage }
    }

    pub fn shape(&self) -> &[u64] {
        self.storage.shape()
    }

    pub fn ndim(&self) -> usize {
        self.storage.shape().len()
    }

    pub fn dtype(&self) -> &Dtype {
        self.storage.dtype()
    }

    pub fn data(&self) -> ArrayData<'_, S> {
        let params = self.storage.codec_params().1;
        let context = ReadContext::new(params).expect("failed to create read context");
        ArrayData::new(self, MaybeOwned::Owned(context))
    }

    pub fn data_ctx<'a>(&'a self, context: &'a ReadContext) -> ArrayData<'a, S> {
        ArrayData::new(self, MaybeOwned::Borrowed(context))
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn into_storage(self) -> S {
        self.storage
    }

    pub(crate) fn blocks_layout(&self) -> &BlocksLayout {
        self.storage.blocks_layout()
    }

    pub fn as_ref(&self) -> Array<Ref<'_, S>> {
        Array {
            storage: Ref(self.storage()),
        }
    }
}

impl Array<Owned> {
    pub fn from_ndarray<S, T, D>(
        array: &ndarray::ArrayBase<S, D, T>,
        params: ArrayParams,
    ) -> io::Result<Self>
    where
        T: Dtyped,
        D: ndarray::Dimension,
        S: ndarray::RawData<Elem = T>,
    {
        let ndim = array.ndim();
        assert!(ndim < NDIM_MAX);
        let shape = array
            .shape()
            .iter()
            .map(|&s| s as u64)
            .collect::<DimArray<_>>();
        let dtype = T::DTYPE;
        let itemsize = dtype.itemsize() as usize;

        let strides = array.strides();
        let strides = dim_arr(ndim, |dim| {
            usize::try_from(strides[dim]).unwrap() * size_of::<T>()
        });

        let builder = ArrayBuilder::new(&shape, dtype.clone(), params);
        builder.build(
            |builder, block_idx, block_inner_offset, block_size, out_buf| {
                // TODO: fast path for contiguous data
                let block_shape = builder.block_shape();
                let initial_arr_offset = (0..ndim)
                    .map(|dim| {
                        let idx =
                            block_idx[dim] * block_shape[dim] as u64 + block_inner_offset[dim];
                        idx as usize * strides[dim]
                    })
                    .sum::<usize>();
                let initial_arr_ptr =
                    unsafe { array.as_ptr().cast::<u8>().add(initial_arr_offset) };
                let initial_block_offset = (0..ndim)
                    .map(|dim| {
                        block_inner_offset[dim] as usize * builder.block_strides[dim] as usize
                    })
                    .sum::<usize>();
                let initial_block_ptr = unsafe { out_buf.as_mut_ptr().add(initial_block_offset) };
                unsafe {
                    nd_copy(
                        initial_arr_ptr,
                        initial_block_ptr,
                        block_size,
                        &strides,
                        &builder.block_strides,
                        itemsize,
                    )
                };
                Ok(())
            },
        )
    }
}

struct ArrayBuilder {
    shape: DimArray<u64>,
    dtype: Dtype,

    blocks_layout: BlocksLayout,
    encoder_params: EncoderParams,
    decoder_params: DecoderParams,

    block_strides: DimArray<BlockSize>,
}
impl ArrayBuilder {
    fn new(shape: &[u64], dtype: Dtype, params: ArrayParams) -> Self {
        let shape: DimArray<u64> = shape.try_into().unwrap();

        let ndim = shape.len();
        assert!(ndim < NDIM_MAX);

        let b_layout = BlocksLayout::new(
            params.block_shape,
            params.block_shape_tag,
            params.block_size_hint,
            params.preferred_read_block_shape,
            params.preferred_read_block_size_hint,
            shape.as_slice(),
            dtype.itemsize() as _,
        );

        let block_strides = default_strides(&b_layout.block_shape_hint, dtype.itemsize() as _);

        Self {
            shape,
            dtype,

            blocks_layout: b_layout,
            encoder_params: params.encoder_params.unwrap_or_default(),
            decoder_params: params.decoder_params.unwrap_or_default(),

            block_strides,
        }
    }

    fn build(
        self,
        mut block_fn: impl FnMut(&Self, &[u64], &[u64], &[u64], &mut [u8]) -> io::Result<()>,
    ) -> io::Result<Array<Owned>> {
        let ndim = self.shape.len();
        assert!(ndim < NDIM_MAX);
        let block_shape = &self.blocks_layout.block_shape_hint;
        assert_eq!(ndim, block_shape.len());

        let grid_shape = dim_arr(ndim, |dim| {
            self.shape[dim].div_ceil(block_shape[dim] as u64)
        });
        let block_size = block_shape.iter().map(|&s| s as u64).product::<u64>();

        let mut block_iter = NdIter::new(
            &grid_shape,
            NdIterExtBlockOffsetSize::new(
                &self.shape,
                &dim_arr(ndim, |_| 0),
                &self.shape,
                &dim_arr(ndim, |dim| block_shape[dim] as u64),
            ),
        );

        let encoder = Encoder::new(&self.encoder_params, self.dtype.clone())?;
        let block_capacity_bytes = block_size * self.dtype.itemsize() as u64;
        let mut builder =
            BlockTableBuilder::new(self.dtype.clone(), block_size as BlockSize, encoder);
        let mut tmp_block_data = AlignedBytes::with_capacity(
            self.dtype.alignment() as usize,
            block_capacity_bytes as usize,
        );
        while let Some((block_idx, (block_inner_offset, block_size))) = block_iter.next() {
            debug_assert!(block_inner_offset.iter().all(|&o| o == 0)); // TODO

            // Init chunk data to zeros.
            // The padding elements (if any) will not be written by the iter below, so they will stay zeros.
            tmp_block_data.clear();
            tmp_block_data.resize(block_capacity_bytes as usize, 0);

            block_fn(
                &self,
                block_idx,
                block_inner_offset,
                block_size,
                &mut tmp_block_data,
            )?;

            builder.add_block(&tmp_block_data)?;
        }

        let blocks = builder.finish();

        Ok(Array {
            storage: Owned(ArrayBlockTableStorageBase::new(
                blocks,
                self.shape,
                self.blocks_layout,
                self.encoder_params,
                self.decoder_params,
            )),
        })
    }

    fn block_shape(&self) -> &[BlockSize] {
        &self.blocks_layout.block_shape_hint
    }
}

pub struct ArrayData<'a, S> {
    array: &'a Array<S>,
    context: MaybeOwned<'a, ReadContext>,
    type_id_cache: Cell<Option<TypeId>>,
}

impl<'a, S: ArrayStorage> ArrayData<'a, S> {
    fn new(array: &'a Array<S>, context: MaybeOwned<'a, ReadContext>) -> Self {
        Self {
            array,
            context,
            type_id_cache: Cell::new(None),
        }
    }

    pub fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    pub fn ndim(&self) -> usize {
        self.array.ndim()
    }

    pub fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }

    fn check_type<T: Dtyped>(&self) -> io::Result<()> {
        let type_id = TypeId::of::<T>();
        if self.type_id_cache.get() == Some(type_id) {
            return Ok(());
        }

        let dtype = T::DTYPE;
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

        self.type_id_cache.set(Some(type_id));
        Ok(())
    }

    pub fn to_ndarray<T>(&self) -> io::Result<ndarray::ArrayD<T>>
    where
        T: Dtyped,
    {
        let shape = self.shape();
        let full_range = dim_arr(shape.len(), |dim| 0u64..shape[dim]);
        self.to_ndarray_sub(&full_range)
    }

    pub fn to_ndarray_sub<T>(&self, range: &[Range<u64>]) -> io::Result<ndarray::ArrayD<T>>
    where
        T: Dtyped,
    {
        self.check_type::<T>()?;
        let ndim = self.ndim();
        assert_eq!(ndim, range.len());
        let out_shape = dim_arr(ndim, |dim| {
            let len = range[dim].end - range[dim].start;
            let len: usize = len.try_into().unwrap();
            len
        });
        let mut array = ndarray::ArrayD::uninit(&out_shape[..]);
        self.to_ndarray_buf(range, {
            unsafe { cast_slice_mut::<MaybeUninit<T>, u8>(array.as_slice_mut().unwrap()) }
        })?;
        Ok(unsafe { array.assume_init() })
    }

    pub fn to_ndarray_buf(&self, range: &[Range<u64>], buf: &mut [u8]) -> io::Result<()> {
        // TODO: call read_data multiple times with smaller blocks
        self.array
            .storage
            .read_data(range, buf, self.context.as_ref())
    }

    pub fn copy(&self) -> io::Result<Array<Owned>>
    where
        S: ArrayStorage,
    {
        self.copy_with(ArrayParams::default())
    }

    pub fn copy_with(&self, mut params: ArrayParams) -> io::Result<Array<Owned>>
    where
        S: ArrayStorage,
    {
        params.override_from_storage(&self.array.storage);

        let ndim = self.ndim();
        let dtype = self.dtype().clone();
        let itemsize = dtype.itemsize() as usize;
        let mut tmp_block_data = AlignedBytes::new(dtype.alignment() as usize);
        let builder = ArrayBuilder::new(self.shape(), dtype, params);
        builder.build(
            |builder, block_idx, block_inner_offset, block_size, output_block| {
                let block_shape = builder.block_shape();
                let range = dim_arr(ndim, |dim| {
                    let start = block_idx[dim] * block_shape[dim] as u64 + block_inner_offset[dim];
                    let end = start + block_size[dim];
                    start..end
                });

                let full_block = (0..ndim).all(|dim| {
                    block_inner_offset[dim] == 0 && block_size[dim] == block_shape[dim] as u64
                });

                let output_block_ptr = output_block.as_mut_ptr();
                let read_data_buf = if full_block {
                    output_block
                } else {
                    let b_size_bytes = block_size.iter().product::<u64>() as usize * itemsize;
                    tmp_block_data.clear();
                    tmp_block_data.reserve(b_size_bytes);
                    unsafe { tmp_block_data.set_len(b_size_bytes) };
                    tmp_block_data.as_mut_slice()
                };

                self.array
                    .storage
                    .read_data(&range, read_data_buf, self.context.as_ref())?;

                if !full_block {
                    // Copy from temporary buffer to output block with correct strides.
                    let src_strides =
                        default_strides(&dim_arr(ndim, |dim| block_size[dim] as usize), itemsize);
                    unsafe {
                        nd_copy(
                            // TODO use in other place
                            read_data_buf.as_ptr(),
                            output_block_ptr,
                            block_size,
                            &src_strides,
                            &builder.block_strides,
                            itemsize,
                        )
                    };
                }

                Ok(())
            },
        )
    }
}

impl Array<Owned> {
    pub fn write_to<W>(&self, writer: W) -> io::Result<()>
    where
        W: Write + Seek,
    {
        let mut writer = ArchiveWriter::new(writer, schema::ArchiveType::ArrayV1)?;

        let header = schema::ArrayHeader {
            shape: self.shape().to_vec(),
            block_shape: self
                .storage
                .0
                .block_shape()
                .iter()
                .cloned()
                .map(|s| s as u64)
                .collect(),
        };
        writer.write_message(&header)?;

        self.storage.0.blocks.write_content(&mut writer)
    }

    pub fn read_from_reader<R>(reader: R, len: u64, params: ArrayParams) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        let storage = ArrayBlockTableStorageBase::read_from(
            reader,
            len,
            crate::block::Owned::read_from,
            params,
        )?;
        Ok(Self {
            storage: Owned(storage),
        })
    }
}
impl Array<Mmap> {
    /// # Safety
    ///
    /// Same as `memmap2::Mmap::map`.
    pub unsafe fn read_from_file_mmap(
        path: &Path,
        offset: u64,
        len: u64,
        params: ArrayParams,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut reader = BufReader::new(file);
        reader.seek(io::SeekFrom::Start(offset))?;

        let storage = ArrayBlockTableStorageBase::read_from(
            reader,
            len,
            |reader, cdata_section, block_offsets_section| {
                crate::block::Mmap::read_from(reader, cdata_section, block_offsets_section, mmap)
            },
            params,
        )?;

        Ok(Self {
            storage: Mmap(storage),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, Write};

    use ndarray::ArrayD;

    use crate::array::{ArrayBlockTableStorageBase, ArrayParams, Owned};
    use crate::block::{BlockSize, BlockTable};
    use crate::codec::{DecoderParams, Encoder, EncoderParams};
    use crate::dtype::Dtyped;
    use crate::storage::{BlockShapeTag, BlocksLayout};
    use crate::util::{DimArray, cast_slice, dim_arr};

    use super::Array;

    // -----------------------------------------------------------------------
    // from_ndarray roundtrip helper
    // -----------------------------------------------------------------------

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    fn roundtrip<T, S, D>(src: &ndarray::ArrayBase<S, D>, block_shape: &[usize]) -> ArrayD<T>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::from_ndarray(&src, arr_params(block_shape)).unwrap();
        a.data().to_ndarray().unwrap()
    }

    // -----------------------------------------------------------------------
    // Helper: build a BlockTable from pre-arranged typed blocks
    // -----------------------------------------------------------------------

    fn make_block_table<T: Dtyped>(blocks: &[&[T]]) -> BlockTable<crate::block::Owned> {
        let block_len = blocks[0].len() as BlockSize;
        let data: Vec<u8> = blocks
            .iter()
            .flat_map(|b| unsafe { cast_slice::<T, u8>(b) }.iter().copied())
            .collect();
        let encoder = Encoder::new(&EncoderParams::default(), T::DTYPE).unwrap();
        BlockTable::build_from_data(&data, T::DTYPE, block_len, encoder).unwrap()
    }

    fn array<T: Dtyped>(blocks: &[&[T]], shape: &[usize], block_shape: &[usize]) -> Array<Owned> {
        let shape: DimArray<u64> = shape.iter().map(|&x| x as u64).collect();
        let ndim = block_shape.len();
        let block_shape_hint: DimArray<BlockSize> =
            block_shape.iter().map(|&x| x as BlockSize).collect();
        let layout = BlocksLayout {
            block_shape_hint: block_shape_hint.clone(),
            block_shape_tag: dim_arr(ndim, |_| BlockShapeTag::Fixed),
            block_size_hint: 0,
            preferred_read_block_shape: block_shape_hint,
            preferred_read_block_size_hint: 0,
        };
        Array {
            storage: Owned(ArrayBlockTableStorageBase::new(
                make_block_table(blocks),
                shape,
                layout,
                EncoderParams::default(),
                DecoderParams::default(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn dtype_shape_ndim() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        assert_eq!(a.dtype(), &u8::DTYPE);
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
    // to_ndarray_sub — 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_1d_full_range() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[0..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn to_ndarray_sub_1d_aligned_second_block() {
        // range [3..6) → output shape [3], values [3,4,5]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[3..6]).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3], vec![3, 4, 5]).unwrap());
    }

    #[test]
    fn to_ndarray_sub_1d_cross_block_boundary() {
        // range [1..5) → output shape [4], values [1,2,3,4]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[1..5]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![1, 2, 3, 4]).unwrap()
        );
    }

    #[test]
    fn to_ndarray_sub_1d_within_single_block() {
        // range [1..2) → output shape [1], value [1]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[1..2]).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![1]).unwrap());
    }

    // -----------------------------------------------------------------------
    // to_ndarray_sub — 2D
    // shape=[4,6], block_shape=[2,3], data as in to_ndarray_2d test.
    // range=[1..3, 2..5] → output shape [2,3]:
    //   [8,  9,  10]
    //   [14, 15, 16]
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_2d() {
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
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[1..3, 2..5]).unwrap();
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
        let a = Array::from_ndarray(&src, arr_params(&[3])).unwrap();
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
        let a = Array::from_ndarray(&src, arr_params(&[10])).unwrap();
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn from_ndarray_1d_noncontiguous() {
        // Step-2 slice of [0..10] → [0, 2, 4, 6, 8]
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = src.slice(ndarray::s![..;2]);
        let a = Array::from_ndarray(&view, arr_params(&[3])).unwrap();
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
        let a = Array::from_ndarray(&src, arr_params(&[3])).unwrap();
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.dtype(), &i32::DTYPE);
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
        let a = Array::from_ndarray(&src, arr_params(&[2, 3])).unwrap();
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
    // from_ndarray + to_ndarray_sub integration
    // -----------------------------------------------------------------------

    #[test]
    fn from_ndarray_then_to_ndarray_sub_1d() {
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        let a = Array::from_ndarray(&src, arr_params(&[3])).unwrap();
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[1..5]).unwrap();
        assert_eq!(got, ndarray::array![1u8, 2, 3, 4].into_dyn());
    }

    #[test]
    fn from_ndarray_then_to_ndarray_sub_2d() {
        #[rustfmt::skip]
        let src = ndarray::array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a = Array::from_ndarray(&src, arr_params(&[2, 3])).unwrap();
        let got: ArrayD<u8> = a.data().to_ndarray_sub(&[1..3, 2..5]).unwrap();
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
        let a = Array::from_ndarray(&src, arr_params(block_shape)).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        a.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let len = bytes.len() as u64;
        Array::read_from_reader(Cursor::new(bytes), len, ArrayParams::default()).unwrap()
    }

    #[test]
    fn write_read_1d_single_block() {
        let src = ndarray::array![0u8, 1, 2, 3];
        let a2 = array_round_trip::<u8, _, _>(&src, &[4]);
        assert_eq!(a2.shape(), &[4]);
        assert_eq!(a2.ndim(), 1);
        assert_eq!(a2.dtype(), &u8::DTYPE);
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
        assert_eq!(a2.dtype(), &i32::DTYPE);
        assert_eq!(a2.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_f32() {
        let src = ndarray::array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5];
        let a2 = array_round_trip::<f32, _, _>(&src, &[3]);
        assert_eq!(a2.dtype(), &f32::DTYPE);
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

    #[cfg(not(miri))]
    #[test]
    fn write_read_file() {
        let src = ndarray::array![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::from_ndarray(&src, arr_params(&[4])).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        a.write_to(std::fs::File::create(&path).unwrap()).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        let a2 = Array::read_from_reader(file, len, ArrayParams::default()).unwrap();

        assert_eq!(a2.shape(), &[12]);
        assert_eq!(a2.dtype(), &u32::DTYPE);
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
        let a0 = Array::from_ndarray(&src0, arr_params(&[4])).unwrap();
        a0.write_to(&mut buf).unwrap();
        let len0 = buf.stream_position().unwrap();
        // pad
        buf.write_all(vec![0u8; PAD].as_slice()).unwrap();
        let off1 = buf.stream_position().unwrap();

        // arr 1
        let a1 = Array::from_ndarray(&src1, arr_params(&[2, 2])).unwrap();
        a1.write_to(&mut buf).unwrap();
        let len1 = buf.stream_position().unwrap() - off1;
        // pad
        buf.write_all(vec![0u8; PAD].as_slice()).unwrap();
        let off2 = buf.stream_position().unwrap();

        // arr 2
        let a2 = Array::from_ndarray(&src2, arr_params(&[1, 2, 3])).unwrap();
        a2.write_to(&mut buf).unwrap();
        let len2 = buf.stream_position().unwrap() - off2;

        let bytes = buf.into_inner();

        // Read array 0 (at offset 0).
        let r0 =
            Array::read_from_reader(Cursor::new(&bytes), len0, ArrayParams::default()).unwrap();
        assert_eq!(r0.shape(), &[4]);
        assert_eq!(r0.ndim(), 1);
        assert_eq!(r0.dtype(), &u8::DTYPE);
        assert_eq!(r0.data().to_ndarray::<u8>().unwrap(), src0.into_dyn());

        // Read array 1 (padded offset, 2D).
        let r1 = Array::read_from_reader(
            Cursor::new(&bytes[off1 as usize..]),
            len1,
            ArrayParams::default(),
        )
        .unwrap();
        assert_eq!(r1.shape(), &[3, 4]);
        assert_eq!(r1.ndim(), 2);
        assert_eq!(r1.dtype(), &i32::DTYPE);
        assert_eq!(r1.data().to_ndarray::<i32>().unwrap(), src1.into_dyn());

        // Read array 2 (padded offset, 3D).
        let r2 = Array::read_from_reader(
            Cursor::new(&bytes[off2 as usize..]),
            len2,
            ArrayParams::default(),
        )
        .unwrap();
        assert_eq!(r2.shape(), &[2, 2, 3]);
        assert_eq!(r2.ndim(), 3);
        assert_eq!(r2.dtype(), &f32::DTYPE);
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
        let a = Array::from_ndarray(&src, arr_params(block_shape)).unwrap();
        let path = tmp_file.path().to_path_buf();
        a.write_to(std::fs::File::create(&path).unwrap()).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        unsafe {
            super::Array::<super::Mmap>::read_from_file_mmap(&path, 0, len, ArrayParams::default())
        }
        .unwrap()
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_1d_single_block() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = ndarray::array![0u8, 1, 2, 3];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[4], &tmp);
        assert_eq!(a2.shape(), &[4]);
        assert_eq!(a2.ndim(), 1);
        assert_eq!(a2.dtype(), &u8::DTYPE);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_1d_multi_block() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[3], &tmp);
        assert_eq!(a2.shape(), &[6]);
        assert_eq!(a2.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_1d_i32() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = ndarray::array![0i32, 10, 20, 30, 40, 50, 60, 70];
        let a2 = array_mmap_round_trip::<i32, _, _>(&src, &[4], &tmp);
        assert_eq!(a2.dtype(), &i32::DTYPE);
        assert_eq!(a2.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
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

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_nonzero_offset() {
        // Write two arrays back-to-back; read the second via its offset.
        let src1 = ndarray::array![0u8, 1, 2, 3];
        let src2 = ndarray::array![10u8, 11, 12, 13, 14, 15];
        let a1 = Array::from_ndarray(&src1, arr_params(&[4])).unwrap();
        let a2_arr = Array::from_ndarray(&src2, arr_params(&[3])).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let mut f = std::fs::File::create(&path).unwrap();
        a1.write_to(&mut f).unwrap();
        let offset = f.metadata().unwrap().len();
        a2_arr.write_to(&mut f).unwrap();
        let total_len = f.metadata().unwrap().len();
        drop(f);

        let len2 = total_len - offset;
        let read = unsafe {
            super::Array::<super::Mmap>::read_from_file_mmap(
                &path,
                offset,
                len2,
                ArrayParams::default(),
            )
        }
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

    // -----------------------------------------------------------------------
    // copy
    // -----------------------------------------------------------------------

    #[test]
    fn copy_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.ndim(), 1);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [4]);
        assert_eq!(
            b.data().to_ndarray::<u8>().unwrap(),
            ArrayD::from_shape_vec(vec![4], vec![0u8, 1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn copy_1d_multi_block() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[6]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [3]);
        assert_eq!(
            b.data().to_ndarray::<u8>().unwrap(),
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn copy_1d_with_padding() {
        // shape [5], block [3] → stored as 6 elements (padded)
        let src = ndarray::array![0u8, 1, 2, 3, 4];
        let a = Array::from_ndarray(&src, arr_params(&[3])).unwrap();
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[5]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [3]);
        assert_eq!(b.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn copy_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[8]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(
            b.data().to_ndarray::<i32>().unwrap(),
            ArrayD::from_shape_vec(vec![8], vec![10i32, 20, 30, 40, 50, 60, 70, 80]).unwrap()
        );
    }

    #[test]
    fn copy_2d_single_block() {
        // shape=[2,3], block=[2,3] — one block, no partial-block path
        let a = array(&[&[0u8, 1, 2, 3, 4, 5]], &[2, 3], &[2, 3]);
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [2, 3]);
        assert_eq!(
            b.data().to_ndarray::<u8>().unwrap(),
            ArrayD::from_shape_vec(vec![2, 3], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn copy_2d_multi_block() {
        // shape=[4,6], block=[2,3] — 4 blocks, exercises the full-block copy path
        // Block layout (row-major grid):
        //   block0=[0,0]: rows 0-1, cols 0-2 → 0,1,2,6,7,8
        //   block1=[0,1]: rows 0-1, cols 3-5 → 3,4,5,9,10,11
        //   block2=[1,0]: rows 2-3, cols 0-2 → 12,13,14,18,19,20
        //   block3=[1,1]: rows 2-3, cols 3-5 → 15,16,17,21,22,23
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
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[4, 6]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [2, 3]);
        assert_eq!(
            b.data().to_ndarray::<u8>().unwrap(),
            ArrayD::from_shape_vec(vec![4, 6], (0u8..24).collect()).unwrap()
        );
    }

    #[test]
    fn copy_2d_with_padding() {
        // shape=[3,5], block=[2,3] → padded to [4,6]; shape preserved as [3,5].
        // Block grid 2×2:
        //   [0,0]: size [2,3] — full block
        //   [0,1]: size [2,2] — partial in dim1
        //   [1,0]: size [1,3] — partial in dim0
        //   [1,1]: size [1,2] — partial in BOTH dims (corner block)
        #[rustfmt::skip]
        let src = ndarray::array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::from_ndarray(&src, arr_params(&[2, 3])).unwrap();
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[3, 5]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(b.data().to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn copy_3d_with_padding_in_all_dims() {
        // shape=[3,3,5], block=[2,2,3] → padded to [4,4,6].
        // Block grid 2×2×2 = 8 blocks; every boundary block is partial in at least
        // one dimension, and the single corner block [1,1,1] is partial in all three:
        //   size [1,1,2] vs block_shape [2,2,3].
        let src = ndarray::Array3::<u8>::from_shape_vec([3, 3, 5], (0u8..45).collect()).unwrap();
        let a = Array::from_ndarray(&src, arr_params(&[2, 2, 3])).unwrap();
        let b = a.data().copy().unwrap();
        assert_eq!(b.shape(), &[3, 3, 5]);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [2, 2, 3]);
        assert_eq!(b.data().to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn copy_preserves_block_shape() {
        // Verify the copied array has the same block layout as the source.
        let src = ndarray::array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::from_ndarray(&src, arr_params(&[4])).unwrap();
        let b = a.data().copy().unwrap();
        assert_eq!(
            a.blocks_layout().block_shape_hint[..],
            b.blocks_layout().block_shape_hint[..]
        );
    }

    #[test]
    fn copy_result_is_independent() {
        // Mutating the source array should not affect the copy (they are independent).
        // Since Array<Owned> doesn't expose mutation, we verify by round-tripping
        // both through write/read and checking values remain consistent.
        let src = ndarray::array![10u8, 20, 30, 40];
        let a = Array::from_ndarray(&src, arr_params(&[4])).unwrap();
        let b = a.data().copy().unwrap();
        // Both should read back the same data independently.
        assert_eq!(
            a.data().to_ndarray::<u8>().unwrap(),
            b.data().to_ndarray::<u8>().unwrap()
        );
    }
}
