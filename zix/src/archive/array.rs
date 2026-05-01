use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use crate::archive::block::BlockTableStorageRead;
use crate::archive::common::{ArchiveReader, ArchiveWriter};
use crate::archive::schema;
use crate::codec::{DecoderCodecConfig, ReadContext};
use crate::error::{check_ndim, ensure, Error, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage};
use crate::storage::{
    ArrayBlockTableStorageBase, ArrayStorage, BlocksLayout, Compact, CompactMmap,
};
use crate::util::{dim_arr, DimArray, Idx, IxIterExt};
use crate::{Array, ArrayParams};

impl Array<Compact> {
    /// Load a compressed array from a `.zix` file, allocating storage on the heap.
    ///
    /// This is the most common way to read an array that was previously saved with
    /// [`write_to_file`](Array::write_to_file) or [`write_to`](Array::write_to).
    ///
    /// Use [`read_from_file_section`](Array::read_from_file_section) if the array occupies only
    /// part of a larger file, or [`read_from_file_mmap`](Array::read_from_file_mmap) for
    /// memory-mapped (zero-copy) loading.
    ///
    /// # Arguments
    ///
    /// - `path`: path to the `.zix` file containing the array.
    /// - `params`: parameters controlling how the array is read and decoded. See
    ///   [`ArrayParams`] for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let tmp_dir = tempfile::tempdir()?;
    /// let path = tmp_dir.path().join("data.zix");
    ///
    /// Array::compact_array(&array![[1.0f32, 2.0], [3.0, 4.0]])?.write_to_file(&path)?;
    /// let array = Array::read_from_file(&path, ArrayParams::default())?;
    /// assert_eq!(array.shape(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn read_from_file(path: &Path, params: ArrayParams) -> Result<Self> {
        let len = path.metadata().map_err(Error::io)?.len();
        Self::read_from_file_section(path, 0, len, params)
    }

    /// Load a compressed array from a byte range within a file.
    ///
    /// Use this when multiple arrays are packed into a single file and you know the byte `offset`
    /// and `len` of the array you want to read. This avoids opening separate files per array and
    /// lets a container format embed arrays alongside other data.
    ///
    /// # Arguments
    ///
    /// - `path`: path to the `.zix` file containing the array.
    /// - `offset`: byte offset of the start of the array's archive section within the file.
    /// - `len`: byte length of the array's archive section.
    /// - `params`: parameters controlling how the array is read and decoded. See
    ///   [`ArrayParams`] for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let tmp_dir = tempfile::tempdir()?;
    /// let path = tmp_dir.path().join("packed.zix");
    ///
    /// // Write two arrays back-to-back into a single file and record their positions.
    /// let a = Array::compact_array(&array![0u8, 1, 2, 3])?;
    /// let b = Array::compact_array(&array![10u8, 20, 30, 40, 50, 60])?;
    /// let mut f = std::fs::File::create(&path)?;
    /// a.write_to(&mut f)?;
    /// let offset = f.metadata()?.len();
    /// b.write_to(&mut f)?;
    /// let total = f.metadata()?.len();
    ///
    /// // Read the second array back using its offset.
    /// let b2 = Array::read_from_file_section(&path, offset, total - offset, ArrayParams::default())?;
    /// assert_eq!(b2.shape(), &[6]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn read_from_file_section(
        path: &Path,
        offset: u64,
        len: u64,
        params: ArrayParams,
    ) -> Result<Self> {
        let file = File::open(path).map_err(Error::io)?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(offset)).map_err(Error::io)?;
        Self::read_from_reader(reader, Some(len), params)
    }

    /// Load a compressed array from a generic reader.
    ///
    /// `len` is the byte length of the archive within the reader. When provided, it enables
    /// bounds checking on section offsets; pass `None` to skip bounds checking.
    ///
    /// # Arguments
    ///
    /// - `reader`: any source implementing `Read` + `Seek` containing the array's archive section.
    /// - `len`: byte length of the archive section within the reader, used for bounds checking.
    ///    Pass `None` to skip bounds checking.
    /// - `params`: parameters controlling how the array is read and decoded. See
    ///   [`ArrayParams`] for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// // Serialize to an in-memory buffer and read it back.
    /// let original = Array::compact_array(&array![1i32, 2, 3, 4])?;
    /// let mut buf = Cursor::new(Vec::new());
    /// original.write_to(&mut buf)?;
    ///
    /// let bytes = buf.into_inner();
    /// let loaded = Array::read_from_reader(Cursor::new(bytes), None, ArrayParams::default())?;
    /// assert_eq!(loaded.to_ndarray::<i32>()?, array![1i32, 2, 3, 4].into_dyn());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn read_from_reader(
        reader: impl Read + Seek,
        len: Option<u64>,
        params: ArrayParams,
    ) -> Result<Self> {
        let storage = ArrayBlockTableStorageBase::read_from(
            reader,
            len,
            crate::storage::block::Owned(PhantomData),
            params,
        )?;
        Ok(Self {
            storage: Compact(storage),
        })
    }
}

impl Array<CompactMmap> {
    /// Load a compressed array from a file using memory mapping (zero-copy).
    ///
    /// Instead of copying the file's bytes into heap memory, this maps the file into virtual
    /// address space. The OS pages data in on demand, so startup is fast and only the blocks you
    /// actually read are loaded into physical memory. The mapping stays alive for as long as the
    /// returned array (or any clone/view of it) exists.
    ///
    /// This is particularly useful for large arrays when you only access a subset of blocks, or
    /// when you want to build a lazy pipeline over the raw data without first copying it to heap.
    /// See [`write_to_with`](Array::write_to_with) for how to use a memory-mapped array as the
    /// source of a streaming write pipeline.
    ///
    /// # Arguments
    ///
    /// - `path`: path to the `.zix` file containing the array.
    /// - `offset`: byte offset of the start of the array's archive section within the file.
    /// - `len`: byte length of the array's archive section.
    /// - `params`: parameters controlling how the array is read and decoded. See
    ///   [`ArrayParams`] for details.
    ///
    /// # Safety
    ///
    /// This function is marked `unsafe` because of the potential for *Undefined Behavior* (UB)
    /// using the mmap array if the underlying file is subsequently modified, in or
    /// out of process. Applications must consider the risk and take appropriate precautions when using
    /// file-backed array. Solutions such as file permissions, locks or process-private (e.g. unlinked)
    /// files exist but are platform specific and limited.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let tmp_dir = tempfile::tempdir()?;
    /// let path = tmp_dir.path().join("data.zix");
    /// Array::compact_array(&array![[1.0f32, 2.0], [3.0, 4.0]])?.write_to_file(&path)?;
    ///
    /// let len = std::fs::metadata(&path)?.len();
    /// // Safety: the file is not modified after this point.
    /// let array = unsafe { Array::read_from_file_mmap(&path, 0, len, ArrayParams::default())? };
    /// assert_eq!(array.shape(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
            Some(len),
            crate::storage::block::Mmap {
                mmap: Arc::new(mmap),
                base_offset: offset,
            },
            params,
        )?;

        Ok(Self {
            storage: CompactMmap(storage),
        })
    }
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Save the array to a new file at `path`.
    ///
    /// This is the most common way to persist an array. Works for any array type: a
    /// `Array<Compact>` streams its already-compressed blocks directly to disk without
    /// decompressing; a lazy view (slice, op chain, etc.) compresses on the fly, so the full
    /// decompressed data is never held in memory.
    ///
    /// # Arguments
    ///
    /// `path`: path to the new `.zix` file to create. Must not already exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let array = Array::compact_array(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
    ///
    /// let tmp_dir = tempfile::tempdir()?;
    /// let path = tmp_dir.path().join("output.zix");
    /// array.write_to_file(&path)?;
    ///
    /// let loaded = Array::read_from_file(&path, ArrayParams::default())?;
    /// assert_eq!(loaded.to_ndarray::<f32>()?, array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let writer = BufWriter::new(std::fs::File::create_new(path).map_err(Error::io)?);
        self.write_to(writer)
    }

    /// Write the array to generic writer.
    ///
    /// Like [`write_to_file`](Array::write_to_file) but accepts an arbitrary writer — useful for
    /// writing into an in-memory buffer, writing multiple arrays into a single open file handle,
    /// or integrating into a custom container format.
    ///
    /// Encoding parameters are chosen automatically. Use
    /// [`write_to_with`](Array::write_to_with) for explicit control.
    ///
    /// # Arguments
    ///
    /// - `writer`: any destination implementing `Write` + `Seek` to which the array's archive section
    ///   will be written.
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// // Write two arrays into a single buffer and record their byte positions.
    /// let a = Array::compact_array(&array![1u8, 2, 3, 4])?;
    /// let b = Array::compact_array(&array![10u8, 20, 30])?;
    ///
    /// let mut buf = Cursor::new(Vec::new());
    /// a.write_to(&mut buf)?;
    /// let offset = buf.position();
    /// b.write_to(&mut buf)?;
    ///
    /// // Read back the second array by seeking to its offset.
    /// let bytes = buf.into_inner();
    /// let b2 = Array::read_from_reader(
    ///     Cursor::new(&bytes[offset as usize..]),
    ///     None,
    ///     ArrayParams::default(),
    /// )?;
    /// assert_eq!(b2.to_ndarray::<u8>()?, array![10u8, 20, 30].into_dyn());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to(&self, writer: impl Write + Seek) -> Result<()> {
        self.write_to_with(writer, ArrayParams::default(), &self.read_ctx())
    }

    /// Write the array to a writer with explicit encoding parameters and read context.
    ///
    /// This is the right choice when you need control over the block shape or codec, or when you
    /// are building a streaming pipeline that reads compressed data from one place and writes it
    /// elsewhere without materializing the full array in memory.
    ///
    /// **When the array is already compact** (`Array<Compact>` or `Array<CompactMmap>`), its
    /// existing compressed blocks are streamed directly to the writer — `params` is ignored
    /// entirely. No decompression or re-compression takes place.
    ///
    /// **For any other array** (lazy views, op chains, plain buffers), each block is read from
    /// the source and compressed according to `params` before being written.
    ///
    /// # Streaming pipeline example
    ///
    /// The following reads a large array from disk via mmap, applies a lazy slice and a
    /// negation, and writes the result directly to a new file — without ever holding the full
    /// array (compressed or decompressed) in memory:
    ///
    /// ```
    /// use std::io::BufWriter;
    /// use std::fs::File;
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let tmp_dir = tempfile::tempdir()?;
    /// let path = tmp_dir.path().join("large.zix");
    /// Array::compact_array(&array![[2.3_f32, 6.99], [-99.1, 0.0]])?.write_to_file(&path)?;
    /// let len = std::fs::metadata(&path)?.len();
    ///
    /// // Map the file — compressed blocks are paged in on demand, no heap copy.
    /// // Safety: the file is not modified while `src` is live.
    /// let src = unsafe { Array::read_from_file_mmap(&path, 0, len, ArrayParams::default())? };
    /// let context = src.read_ctx();
    ///
    /// // Build a lazy view — no data is read yet.
    /// let view = src.exp() + 1.0f32;
    ///
    /// // Write to a new file: blocks are decompressed, modified by ops, and re-compressed one at
    /// // a time.
    /// view.write_to_with(
    ///     BufWriter::new(File::create(tmp_dir.path().join("modified.zix"))?),
    ///     ArrayParams::default(),
    ///     &context,
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_with(
        &self,
        writer: impl Write + Seek,
        mut params: ArrayParams,
        context: &ReadContext,
    ) -> Result<()> {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();
        params.override_from_storage(&self.storage);
        params.tune(shape, dtype)?;
        let block_shape: DimArray<_> = if let Some(storage) = self.storage.as_compact() {
            storage.0.block_shape().try_into().unwrap()
        } else {
            params.block_shape.clone().unwrap()
        };

        let mut writer =
            ArchiveWriter::new(writer, schema::ArchiveType::ArrayV1).map_err(Error::io)?;
        let header = schema::ArrayHeader {
            shape: shape.to_vec(),
            block_shape: block_shape.iter().cloned().map(|s| s as u64).collect(),
        };
        writer.write_message(&header).map_err(Error::io)?;

        if let Some(storage) = self.storage.as_compact() {
            storage.0.blocks.write_content(&mut writer)?;
            return writer.flush().map_err(Error::io);
        }

        let block_shape = params.block_shape.as_ref().unwrap();
        let block_size = block_shape.iter().cloned().try_product().unwrap();
        let grid_shape = dim_arr(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));
        let nblocks = grid_shape.iter().cloned().product::<u64>();

        let encoder_cfg = params.encoder_params.as_ref().unwrap();
        let decoder_cfg = DecoderCodecConfig {
            codec: encoder_cfg.codec.clone(),
            filters: encoder_cfg.filters.clone(),
            dtype: dtype.clone(),
        };

        let (mut block_fn, block_compressed_bound) = self.to_block_fn(&params, context)?;
        crate::archive::block::write_content_impl(
            nblocks,
            block_size,
            &decoder_cfg,
            &mut writer,
            block_compressed_bound,
            &mut block_fn,
        )?;

        writer.flush().map_err(Error::io)
    }
}

impl<S> ArrayBlockTableStorageBase<S>
where
    S: BlockTableStorage,
{
    pub(crate) fn read_from(
        reader: impl Read + Seek,
        len: Option<u64>,
        storage: S,
        params: ArrayParams,
    ) -> Result<Self>
    where
        S: BlockTableStorageRead,
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

        let blocks = BlockTable::read_content(&mut reader, storage)?;
        ensure!(
            blocks.nitems() == expected_nitems,
            InvalidArchive,
            "array blocks nitems {} does not match shape product {expected_nitems}",
            blocks.nitems()
        );

        let b_layout = BlocksLayout::new(
            Some(block_shape),
            params.block_shape_tag,
            params.block_size_hint,
            params.preferred_read_shape,
            params.preferred_read_size_hint,
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

    use ndarray::array;

    use crate::dtype::Dtyped;
    use crate::storage::Compact;
    use crate::util::arr_params;
    use crate::{Array, ArrayParams};

    // -----------------------------------------------------------------------
    // compact_array roundtrip helper
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // write_to / read_from round-trip
    // -----------------------------------------------------------------------

    fn array_round_trip<T, S, D>(
        src: &ndarray::ArrayBase<S, D>,
        block_shape: &[usize],
    ) -> Array<Compact>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::compact_array_with(&src, arr_params(block_shape)).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        a.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        Array::read_from_reader(Cursor::new(bytes), None, ArrayParams::default()).unwrap()
    }

    #[test]
    fn write_read_1d_single_block() {
        let src = array![0u8, 1, 2, 3];
        let a2 = array_round_trip::<u8, _, _>(&src, &[4]);
        assert_eq!(a2.shape(), &[4]);
        assert_eq!(a2.ndim(), 1);
        assert_eq!(a2.dtype(), &u8::DTYPE);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_multi_block() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        let a2 = array_round_trip::<u8, _, _>(&src, &[3]);
        assert_eq!(a2.shape(), &[6]);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_with_padding() {
        // size 5, block 3 → padded to 6; shape is preserved as 5
        let src = array![0u8, 1, 2, 3, 4];
        let a2 = array_round_trip::<u8, _, _>(&src, &[3]);
        assert_eq!(a2.shape(), &[5]);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_i32() {
        let src = array![0i32, 10, 20, 30, 40, 50, 60, 70];
        let a2 = array_round_trip::<i32, _, _>(&src, &[4]);
        assert_eq!(a2.dtype(), &i32::DTYPE);
        assert_eq!(a2.to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_1d_f32() {
        let src = array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5];
        let a2 = array_round_trip::<f32, _, _>(&src, &[3]);
        assert_eq!(a2.dtype(), &f32::DTYPE);
        assert_eq!(a2.to_ndarray::<f32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a2 = array_round_trip::<u8, _, _>(&src, &[2, 3]);
        assert_eq!(a2.shape(), &[4, 6]);
        assert_eq!(a2.ndim(), 2);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_2d_with_padding() {
        // shape [3,5], block [2,3] → padded to [4,6]; shape preserved as [3,5]
        #[rustfmt::skip]
        let src = array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a2 = array_round_trip::<i32, _, _>(&src, &[2, 3]);
        assert_eq!(a2.shape(), &[3, 5]);
        assert_eq!(a2.to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn write_read_file() {
        let src = array![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::compact_array_with(&src, arr_params(&[4])).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        a.write_to(std::fs::File::create(&path).unwrap()).unwrap();

        let a2 = Array::read_from_file(&path, ArrayParams::default()).unwrap();

        assert_eq!(a2.shape(), &[12]);
        assert_eq!(a2.dtype(), &u32::DTYPE);
        assert_eq!(a2.to_ndarray::<u32>().unwrap(), src.into_dyn());
    }

    #[test]
    fn write_read_nonzero_offset() {
        // Write three arrays with padding between them; read each back by seeking to its recorded offset.
        const PAD: usize = 177;
        // src0: 1D u8
        let src0 = array![0u8, 1, 2, 3];
        // src1: 2D i32, shape [3, 4]
        #[rustfmt::skip]
        let src1 = array![
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
        let a0 = Array::compact_array_with(&src0, arr_params(&[4])).unwrap();
        a0.write_to(&mut buf).unwrap();
        let len0 = buf.stream_position().unwrap();
        // pad
        buf.write_all(vec![0u8; PAD].as_slice()).unwrap();
        let off1 = buf.stream_position().unwrap();

        // arr 1
        let a1 = Array::compact_array_with(&src1, arr_params(&[2, 2])).unwrap();
        a1.write_to(&mut buf).unwrap();
        let len1 = buf.stream_position().unwrap() - off1;
        // pad
        buf.write_all(vec![0u8; PAD].as_slice()).unwrap();
        let off2 = buf.stream_position().unwrap();

        // arr 2
        let a2 = Array::compact_array_with(&src2, arr_params(&[1, 2, 3])).unwrap();
        a2.write_to(&mut buf).unwrap();
        let len2 = buf.stream_position().unwrap() - off2;

        let bytes = buf.into_inner();

        // Read array 0 (at offset 0).
        let r0 = Array::read_from_reader(Cursor::new(&bytes), Some(len0), ArrayParams::default())
            .unwrap();
        assert_eq!(r0.shape(), &[4]);
        assert_eq!(r0.ndim(), 1);
        assert_eq!(r0.dtype(), &u8::DTYPE);
        assert_eq!(r0.to_ndarray::<u8>().unwrap(), src0.into_dyn());

        // Read array 1 (padded offset, 2D).
        let r1 = Array::read_from_reader(
            Cursor::new(&bytes[off1 as usize..]),
            Some(len1),
            ArrayParams::default(),
        )
        .unwrap();
        assert_eq!(r1.shape(), &[3, 4]);
        assert_eq!(r1.ndim(), 2);
        assert_eq!(r1.dtype(), &i32::DTYPE);
        assert_eq!(r1.to_ndarray::<i32>().unwrap(), src1.into_dyn());

        // Read array 2 (padded offset, 3D).
        let r2 = Array::read_from_reader(
            Cursor::new(&bytes[off2 as usize..]),
            Some(len2),
            ArrayParams::default(),
        )
        .unwrap();
        assert_eq!(r2.shape(), &[2, 2, 3]);
        assert_eq!(r2.ndim(), 3);
        assert_eq!(r2.dtype(), &f32::DTYPE);
        assert_eq!(r2.to_ndarray::<f32>().unwrap(), src2.into_dyn());
    }

    // -----------------------------------------------------------------------
    // read_from_file_mmap round-trip
    // -----------------------------------------------------------------------

    fn array_mmap_round_trip<T, S, D>(
        src: &ndarray::ArrayBase<S, D>,
        block_shape: &[usize],
        tmp_file: &tempfile::NamedTempFile,
    ) -> super::Array<super::CompactMmap>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::compact_array_with(&src, arr_params(block_shape)).unwrap();
        let path = tmp_file.path().to_path_buf();
        a.write_to(std::fs::File::create(&path).unwrap()).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        unsafe {
            super::Array::<super::CompactMmap>::read_from_file_mmap(
                &path,
                0,
                len,
                ArrayParams::default(),
            )
        }
        .unwrap()
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_1d_single_block() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = array![0u8, 1, 2, 3];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[4], &tmp);
        assert_eq!(a2.shape(), &[4]);
        assert_eq!(a2.ndim(), 1);
        assert_eq!(a2.dtype(), &u8::DTYPE);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_1d_multi_block() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = array![0u8, 1, 2, 3, 4, 5];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[3], &tmp);
        assert_eq!(a2.shape(), &[6]);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_1d_i32() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let src = array![0i32, 10, 20, 30, 40, 50, 60, 70];
        let a2 = array_mmap_round_trip::<i32, _, _>(&src, &[4], &tmp);
        assert_eq!(a2.dtype(), &i32::DTYPE);
        assert_eq!(a2.to_ndarray::<i32>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_2d() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a2 = array_mmap_round_trip::<u8, _, _>(&src, &[2, 3], &tmp);
        assert_eq!(a2.shape(), &[4, 6]);
        assert_eq!(a2.ndim(), 2);
        assert_eq!(a2.to_ndarray::<u8>().unwrap(), src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_nonzero_offset() {
        // Write two arrays back-to-back; read the second via its offset.
        let src1 = array![0u8, 1, 2, 3];
        let src2 = array![10u8, 11, 12, 13, 14, 15];
        let a1 = Array::compact_array_with(&src1, arr_params(&[4])).unwrap();
        let a2_arr = Array::compact_array_with(&src2, arr_params(&[3])).unwrap();

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
            super::Array::<super::CompactMmap>::read_from_file_mmap(
                &path,
                offset,
                len2,
                ArrayParams::default(),
            )
        }
        .unwrap();
        assert_eq!(read.shape(), &[6]);
        assert_eq!(read.to_ndarray::<u8>().unwrap(), src2.into_dyn());
        drop(tmp_file);
    }
}
