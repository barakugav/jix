use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::archive::common::{ArchiveReader, ArchiveWriter, Section};
use crate::archive::schema;
use crate::error::{check_ndim, ensure, Error, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage};
use crate::storage::{ArrayBlockTableStorageBase, BlocksLayout, Mmap, Owned};
use crate::util::{dim_arr, DimArray, Idx};
use crate::{Array, ArrayParams};

impl Array<Owned> {
    pub fn write_to<W>(&self, writer: W) -> Result<()>
    where
        W: Write + Seek,
    {
        let mut writer =
            ArchiveWriter::new(writer, schema::ArchiveType::ArrayV1).map_err(Error::io)?;

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
        writer.write_message(&header).map_err(Error::io)?;

        self.storage.0.blocks.write_content(&mut writer)
    }

    pub fn read_from_reader<R>(reader: R, len: u64, params: ArrayParams) -> Result<Self>
    where
        R: Read + Seek,
    {
        let storage = ArrayBlockTableStorageBase::read_from(
            reader,
            len,
            crate::storage::block::Owned::read_from,
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
    ) -> Result<Self> {
        let file = File::open(path).map_err(Error::io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(Error::io)? };
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(offset)).map_err(Error::io)?;

        let storage = ArrayBlockTableStorageBase::read_from(
            reader,
            len,
            |reader, cdata_section, block_offsets_section| {
                crate::storage::block::Mmap::read_from(
                    reader,
                    cdata_section,
                    block_offsets_section,
                    mmap,
                )
            },
            params,
        )?;

        Ok(Self {
            storage: Mmap(storage),
        })
    }
}

impl<S> ArrayBlockTableStorageBase<S> {
    pub(crate) fn read_from<R>(
        reader: R,
        len: u64,
        read_block_storage: impl FnOnce(&mut ArchiveReader<R>, Section, Section) -> Result<S>,
        params: ArrayParams,
    ) -> Result<Self>
    where
        R: Read + Seek,
        S: BlockTableStorage,
    {
        let mut reader = ArchiveReader::new(reader, len)?;
        let f_meta = reader.read_file_meta().map_err(Error::io)?;
        ensure!(
            f_meta.archive_type == schema::ArchiveType::ArrayV1 as i32,
            InvalidArchive,
            "unexpected zix file type: expected {:?}, actual {:?}",
            schema::ArchiveType::ArrayV1,
            schema::ArchiveType::try_from(f_meta.archive_type)
        );

        let header = reader
            .read_message::<schema::ArrayHeader>()
            .map_err(Error::io)?;
        let ndim = header.shape.len();
        check_ndim(ndim)?;
        let shape: DimArray<_> = header.shape.as_slice().try_into().unwrap();
        ensure!(
            header.block_shape.len() == ndim,
            InvalidArchive,
            "array block_shape has different ndim {} than shape {ndim}",
            header.block_shape.len(),
        );
        let block_shape = dim_arr(ndim, |dim| header.block_shape[dim] as BlockSize);
        // Compute padded shape in usize for nitems validation.
        let expected_nitems = (0..ndim)
            .map(|dim| {
                let s = shape[dim];
                let b = block_shape[dim] as u64;
                if s == 0 {
                    0
                } else {
                    s.ceil_to_multiple(b)
                }
            })
            .product::<u64>();

        let blocks = BlockTable::read_content(&mut reader, read_block_storage)?;
        ensure!(
            blocks.nitems() == expected_nitems,
            InvalidArchive,
            "array blocks nitems {} does not match shape product {}",
            blocks.nitems(),
            expected_nitems
        );

        let b_layout = BlocksLayout::new(
            Some(block_shape),
            params.block_shape_tag,
            params.block_size_hint,
            params.preferred_read_block_shape,
            params.preferred_read_block_size_hint,
            &shape,
            blocks.dtype().itemsize(),
        )?;

        Ok(Self::new(
            blocks,
            shape,
            b_layout,
            params.encoder_params.unwrap_or_default(),
            params.decoder_params.unwrap_or_default(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, Write};

    use crate::dtype::Dtyped;
    use crate::storage::block::BlockSize;
    use crate::storage::Owned;
    use crate::{Array, ArrayParams};

    // -----------------------------------------------------------------------
    // from_ndarray roundtrip helper
    // -----------------------------------------------------------------------

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
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
}
