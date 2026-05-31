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
use crate::storage::{ArrayBlockTableStorageBase, BlocksLayout, Compact, CompactMmap};
use crate::util::{dim_arr, DimArray, Idx, IxIterExt};
use crate::{Array, ArrayParams, ArrayStorage, DimDyn, Dimension, ErrorKind, TypeDyn};

impl Array<Compact<TypeDyn, DimDyn>> {
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
    ///   Pass `None` to skip bounds checking.
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
    /// let loaded = loaded.to_typed::<i32>()?;
    /// assert_eq!(loaded.to_ndarray()?, array![1i32, 2, 3, 4].into_dyn());
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

impl Array<CompactMmap<TypeDyn, DimDyn>> {
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
    /// let loaded = loaded.to_typed::<f32>()?;
    /// assert_eq!(loaded.to_ndarray()?, array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let writer = BufWriter::new(std::fs::File::create_new(path).map_err(Error::io)?);
        self.write_to(writer)
    }

    /// Write the array to generic writer.
    ///
    /// Like [`write_to_file`](Array::write_to_file) but accepts an arbitrary writer - useful for
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
    /// let b2 = b2.to_typed::<u8>()?;
    /// assert_eq!(b2.to_ndarray()?, array![10u8, 20, 30].into_dyn());
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
    /// existing compressed blocks are streamed directly to the writer - `params` is ignored
    /// entirely. No decompression or re-compression takes place.
    ///
    /// **For any other array** (lazy views, op chains, plain buffers), each block is read from
    /// the source and compressed according to `params` before being written.
    ///
    /// # Streaming pipeline example
    ///
    /// The following reads a large array from disk via mmap, applies a lazy slice and a
    /// negation, and writes the result directly to a new file - without ever holding the full
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
    /// // Map the file - compressed blocks are paged in on demand, no heap copy.
    /// // Safety: the file is not modified while `src` is live.
    /// let src = unsafe { Array::read_from_file_mmap(&path, 0, len, ArrayParams::default())? };
    /// let context = src.read_ctx();
    ///
    /// // Build a lazy view - no data is read yet.
    /// let view = src.to_typed::<f32>()?.exp() + 1.0f32;
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
        let block_shape = if let Some(storage) = self.storage.as_compact() {
            DimArray::from_slice(storage.0.block_shape()).unwrap()
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

impl<S> ArrayBlockTableStorageBase<S, TypeDyn, DimDyn>
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
        let shape = DimArray::from_slice(header.shape.as_slice()).unwrap();
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
            .try_product()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidArchive,
                    format!(
                        "array shape {shape:?} with block shape {block_shape:?} has too many items"
                    ),
                )
            })?;

        let blocks = BlockTable::read_content(&mut reader, storage)?;
        ensure!(
            blocks.nitems() == expected_nitems,
            InvalidArchive,
            "array blocks nitems {} does not match shape product {expected_nitems}",
            blocks.nitems()
        );

        let b_layout = BlocksLayout::tune(
            Some(block_shape),
            params.block_shape_tag,
            params.block_size_hint,
            params.preferred_read_shape,
            params.preferred_read_size_hint,
            &shape,
            blocks.dtype().itemsize(),
        )?;

        let shape = DimDyn::from_slice(&shape).unwrap();
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

    use ndarray::ArrayD;

    use crate::dtype::Dtyped;
    use crate::storage::Compact;
    use crate::util::{arr_params, carray_strategy_any};
    use crate::{Array, ArrayParams, Dimension, IntoDimension, Ty};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn compact<T, Sh>(
        vals: Vec<T>,
        shape: Sh,
        block_shape: &[usize],
    ) -> Array<Compact<Ty<T>, Sh::Dimension>>
    where
        T: Dtyped,
        Sh: IntoDimension,
    {
        let shape = shape.into_dimension().unwrap();
        let shape_usize = shape
            .as_slice()
            .iter()
            .map(|d| *d as usize)
            .collect::<Vec<_>>();
        let src = ArrayD::from_shape_vec(shape_usize, vals)
            .unwrap()
            .into_dimensionality::<<Sh::Dimension as ndarray::IntoDimension>::Dim>()
            .unwrap();
        let array = Array::compact_array_with(&src, arr_params(block_shape)).unwrap();
        array.into_dim().unwrap()
    }

    fn write_read<T: Dtyped, D: Dimension>(
        a: &Array<Compact<Ty<T>, D>>,
    ) -> ndarray::Array<T, <D as ndarray::IntoDimension>::Dim> {
        let mut buf = Cursor::new(Vec::new());
        a.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        Array::read_from_reader(Cursor::new(bytes), None, ArrayParams::default())
            .unwrap()
            .into_typed::<T>()
            .unwrap()
            .into_dim::<D>()
            .unwrap()
            .to_ndarray()
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // Proptest: roundtrip for many dtypes
    //
    // Tests that write_to + read_from_reader recovers the original values for
    // arbitrary inputs, covering:
    //   - 1D single block (no padding)
    //   - 1D multi-block with a padded last block
    //   - 2D multi-block with padding in both dimensions
    // -----------------------------------------------------------------------

    macro_rules! test_archive_roundtrip_dtype {
        ($dtype:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<roundtrip_ $dtype>](
                        (src, a) in carray_strategy_any::<$dtype>()
                    ) {
                        proptest::prop_assert_eq!(write_read::<$dtype, _>(&a), src);
                    }
                }
            }
        };
    }

    test_archive_roundtrip_dtype!(u8);
    test_archive_roundtrip_dtype!(i32);
    test_archive_roundtrip_dtype!(i64);
    test_archive_roundtrip_dtype!(f32);
    test_archive_roundtrip_dtype!(f64);

    // -----------------------------------------------------------------------
    // Many blocks (>100)
    // -----------------------------------------------------------------------

    #[test]
    fn many_blocks_1d_i32() {
        // 630 items, block 5 -> 126 blocks
        let vals = (0..630i32).collect::<Vec<_>>();
        let src = ndarray::Array::from_shape_vec([630], vals.clone()).unwrap();
        let a = compact(vals, &[630], &[5]);
        assert_eq!(write_read(&a), src);
    }

    #[test]
    fn many_blocks_2d_f64() {
        // shape [60, 24], block [4, 3] -> 15*8 = 120 blocks
        let vals: Vec<f64> = (0..60 * 24).map(|x: i32| x as f64).collect();
        let src = ndarray::Array::from_shape_vec([60, 24], vals.clone()).unwrap();
        let a = compact(vals, &[60, 24], &[4, 3]);
        assert_eq!(write_read(&a), src);
    }

    // -----------------------------------------------------------------------
    // 3-D arrays
    // -----------------------------------------------------------------------

    #[test]
    fn array_3d_multiblock_f32() {
        // shape [10, 11, 12], block [2, 3, 4] -> 5*4*3 = 60 blocks
        let vals: Vec<f32> = (0..10 * 11 * 12).map(|x: i32| x as f32).collect();
        let src = ndarray::Array::from_shape_vec([10, 11, 12], vals.clone()).unwrap();
        let a = compact(vals, &[10, 11, 12], &[2, 3, 4]);
        let got = write_read(&a);
        assert_eq!(got.shape(), &[10, 11, 12]);
        assert_eq!(got, src);
    }

    #[test]
    fn array_3d_all_dims_padded_i64() {
        // shape [5, 7, 11], block [3, 4, 5] - every dimension needs padding
        let vals: Vec<i64> = (0..5 * 7 * 11 as i64).collect();
        let src = ndarray::Array::from_shape_vec([5, 7, 11], vals.clone()).unwrap();
        let a = compact(vals, &[5, 7, 11], &[3, 4, 5]);
        let got = write_read(&a);
        assert_eq!(got.shape(), &[5, 7, 11]);
        assert_eq!(got, src);
    }

    // -----------------------------------------------------------------------
    // Non-zero offsets and trailing padding between packed arrays
    //
    // Three arrays of different dtypes and shapes are written back-to-back into
    // a single buffer with 177-byte gaps between them.  Each is read back
    // independently by passing the correct (offset, len) pair to
    // read_from_reader.
    // -----------------------------------------------------------------------

    #[test]
    fn packed_arrays_nonzero_offsets_with_trailing_padding() {
        const PAD: usize = 177;

        let src0 = ndarray::Array::from_shape_vec([6], (0..6u8).collect()).unwrap();
        let src1 = ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap();
        let src2 = ndarray::Array::from_shape_vec(
            [2, 3, 5],
            (0..30).map(|x: i32| x as f64 * 0.5).collect(),
        )
        .unwrap();

        let mut buf = Cursor::new(Vec::<u8>::new());

        let a0 = compact(src0.iter().cloned().collect(), &[6], &[3]);
        a0.write_to(&mut buf).unwrap();
        let end0 = buf.stream_position().unwrap();
        buf.write_all(&vec![0u8; PAD]).unwrap();
        let start1 = buf.stream_position().unwrap();

        let a1 = compact(src1.iter().cloned().collect(), &[3, 4], &[2, 2]);
        a1.write_to(&mut buf).unwrap();
        let end1 = buf.stream_position().unwrap();
        let len1 = end1 - start1;
        buf.write_all(&vec![0u8; PAD]).unwrap();
        let start2 = buf.stream_position().unwrap();

        let a2 = compact(src2.iter().cloned().collect(), &[2, 3, 5], &[1, 2, 3]);
        a2.write_to(&mut buf).unwrap();
        let end2 = buf.stream_position().unwrap();
        let len2 = end2 - start2;

        let bytes = buf.into_inner();

        let r0 = Array::read_from_reader(Cursor::new(&bytes), Some(end0), ArrayParams::default())
            .unwrap();
        assert_eq!(r0.shape(), &[6]);
        assert_eq!(
            r0.to_typed::<u8>().unwrap().to_ndarray().unwrap(),
            src0.into_dyn()
        );

        let r1 = Array::read_from_reader(
            Cursor::new(&bytes[start1 as usize..]),
            Some(len1),
            ArrayParams::default(),
        )
        .unwrap();
        assert_eq!(r1.shape(), &[3, 4]);
        assert_eq!(
            r1.to_typed::<i32>().unwrap().to_ndarray().unwrap(),
            src1.into_dyn()
        );

        let r2 = Array::read_from_reader(
            Cursor::new(&bytes[start2 as usize..]),
            Some(len2),
            ArrayParams::default(),
        )
        .unwrap();
        assert_eq!(r2.shape(), &[2, 3, 5]);
        assert_eq!(
            r2.to_typed::<f64>().unwrap().to_ndarray().unwrap(),
            src2.into_dyn()
        );
    }

    // -----------------------------------------------------------------------
    // write_to_with: lazy views and compact-streams-directly
    // -----------------------------------------------------------------------

    #[test]
    fn write_to_with_lazy_neg_view_i32() {
        // Negation is applied on the fly during write; no full decompressed array
        // is materialized.
        let vals: Vec<i32> = (1..=12i32).collect();
        let src = ndarray::Array::from_shape_vec([3, 4], vals.clone()).unwrap();
        let expected = -&src;
        let a = compact(vals, &[3, 4], &[2, 2]);
        let ctx = a.read_ctx();

        let view = -a.as_ref();
        let mut buf = Cursor::new(Vec::new());
        view.write_to_with(&mut buf, arr_params(&[2, 2]), &ctx)
            .unwrap();

        let got =
            Array::read_from_reader(Cursor::new(buf.into_inner()), None, ArrayParams::default())
                .unwrap()
                .into_typed::<i32>()
                .unwrap()
                .to_ndarray()
                .unwrap();
        assert_eq!(got, expected.into_dyn());
    }

    #[test]
    fn write_to_with_op_chain_3d_i32() {
        // 3-D array, double negation (identity), written via a lazy op chain.
        let vals: Vec<i32> = (1..=3 * 4 * 5i32).collect();
        let src = ndarray::Array::from_shape_vec([3, 4, 5], vals.clone()).unwrap();
        let a = compact(vals, &[3, 4, 5], &[2, 2, 3]);
        let ctx = a.read_ctx();

        let view = -(-a.as_ref()); // neg * neg = identity
        let mut buf = Cursor::new(Vec::new());
        view.write_to_with(&mut buf, arr_params(&[2, 2, 3]), &ctx)
            .unwrap();

        let got =
            Array::read_from_reader(Cursor::new(buf.into_inner()), None, ArrayParams::default())
                .unwrap()
                .into_typed::<i32>()
                .unwrap()
                .to_ndarray()
                .unwrap();
        assert_eq!(got, src.into_dyn());
    }

    #[test]
    fn write_to_with_add_chain_f32() {
        // (a + a) over a compact array - params control the output block shape;
        // the source is read block-by-block, never fully materialized.
        let vals: Vec<f32> = (0..24).map(|x: i32| x as f32).collect();
        let src = ndarray::Array::from_shape_vec([4, 6], vals.clone()).unwrap();
        let expected = &src + &src;
        let a = compact(vals, &[4, 6], &[2, 3]);
        let ctx = a.read_ctx();

        let view = a.as_ref() + a.as_ref();
        let mut buf = Cursor::new(Vec::new());
        view.write_to_with(&mut buf, arr_params(&[2, 3]), &ctx)
            .unwrap();

        let got =
            Array::read_from_reader(Cursor::new(buf.into_inner()), None, ArrayParams::default())
                .unwrap()
                .into_typed::<f32>()
                .unwrap()
                .to_ndarray()
                .unwrap();
        assert_eq!(got, expected.into_dyn());
    }

    #[test]
    fn write_to_with_compact_ignores_params() {
        // For a compact source, write_to and write_to_with must produce identical
        // bytes regardless of the params passed - the compressed blocks are
        // streamed directly.
        let vals: Vec<i32> = (0..16i32).collect();
        let a = compact(vals, &[16], &[4]);
        let ctx = a.read_ctx();

        let mut plain = Cursor::new(Vec::new());
        a.write_to(&mut plain).unwrap();

        let mut with_different_params = Cursor::new(Vec::new());
        a.write_to_with(&mut with_different_params, arr_params(&[8]), &ctx)
            .unwrap();

        assert_eq!(plain.into_inner(), with_different_params.into_inner());
    }

    // -----------------------------------------------------------------------
    // File I/O  (skipped under Miri, which cannot perform real file syscalls)
    // -----------------------------------------------------------------------

    #[cfg(not(miri))]
    #[test]
    fn write_to_file_and_read_from_file_u8() {
        let vals: Vec<u8> = (0..24u8).collect();
        let src = ndarray::Array::from_shape_vec([4, 6], vals.clone()).unwrap();
        let a = compact(vals, &[4, 6], &[2, 3]);

        // Use a path that does not yet exist so create_new succeeds.
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("array.zix");
        a.write_to_file(&path).unwrap();

        let got = Array::read_from_file(&path, ArrayParams::default())
            .unwrap()
            .into_typed::<u8>()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got.shape(), &[4, 6]);
        assert_eq!(got, src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn write_to_file_fails_if_already_exists() {
        let vals: Vec<i32> = (0..4i32).collect();
        let a = compact(vals, &[4], &[4]);
        // NamedTempFile creates the file; write_to_file (create_new) must fail.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        assert!(a.write_to_file(tmp.path()).is_err());
    }

    #[cfg(not(miri))]
    #[test]
    fn read_from_file_section_two_dtypes_packed() {
        // Two arrays of different dtypes written consecutively; each is read back
        // via read_from_file_section using the recorded (offset, len).
        let src0 = ArrayD::<u8>::from_shape_vec(vec![6], (0..6u8).collect()).unwrap();
        let src1 = ndarray::Array::from_shape_vec([4, 5], (0..20).map(|x: i32| x as f32).collect())
            .unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut f = std::fs::File::create(&path).unwrap();
        compact(src0.iter().cloned().collect(), &[6], &[3])
            .write_to(&mut f)
            .unwrap();
        let off1 = f.metadata().unwrap().len();
        compact(src1.iter().cloned().collect(), &[4, 5], &[2, 3])
            .write_to(&mut f)
            .unwrap();
        let total = f.metadata().unwrap().len();
        drop(f);

        let r0 = Array::read_from_file_section(&path, 0, off1, ArrayParams::default()).unwrap();
        assert_eq!(r0.into_typed::<u8>().unwrap().to_ndarray().unwrap(), src0);

        let r1 = Array::read_from_file_section(&path, off1, total - off1, ArrayParams::default())
            .unwrap();
        assert_eq!(
            r1.into_typed::<f32>().unwrap().to_ndarray().unwrap(),
            src1.into_dyn()
        );
    }

    // -----------------------------------------------------------------------
    // mmap round-trips
    // -----------------------------------------------------------------------

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_basic_i64() {
        let vals: Vec<i64> = (0..24i64).collect();
        let src = ndarray::Array::from_shape_vec([4, 6], vals.clone()).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        compact(vals, &[4, 6], &[2, 3])
            .write_to(std::fs::File::create(tmp.path()).unwrap())
            .unwrap();
        let len = std::fs::metadata(tmp.path()).unwrap().len();

        let got = unsafe {
            Array::read_from_file_mmap(tmp.path(), 0, len, ArrayParams::default()).unwrap()
        }
        .into_typed::<i64>()
        .unwrap()
        .to_ndarray()
        .unwrap();
        assert_eq!(got, src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_nonzero_offset() {
        // Two arrays in one file; read only the second via mmap with its offset.
        let src1 =
            ndarray::Array::from_shape_vec([2, 3, 4], (0..24).map(|x: i32| x as f32).collect())
                .unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut f = std::fs::File::create(&path).unwrap();
        // Pad with an unrelated array first.
        compact((0..4u8).collect(), &[4], &[4])
            .write_to(&mut f)
            .unwrap();
        let offset = f.metadata().unwrap().len();
        compact(src1.iter().cloned().collect(), &[2, 3, 4], &[1, 2, 2])
            .write_to(&mut f)
            .unwrap();
        let len = f.metadata().unwrap().len() - offset;
        drop(f);

        let got = unsafe {
            Array::read_from_file_mmap(&path, offset, len, ArrayParams::default()).unwrap()
        }
        .into_typed::<f32>()
        .unwrap()
        .to_ndarray()
        .unwrap();
        assert_eq!(got, src1.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_read_many_blocks_3d_i32() {
        // [10, 11, 12] with blocks [2, 3, 4] -> 5*4*3 = 60 blocks; all dims padded.
        let vals: Vec<i32> = (0..10 * 11 * 12i32).collect();
        let src = ndarray::Array::from_shape_vec([10, 11, 12], vals.clone()).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        compact(vals, &[10, 11, 12], &[2, 3, 4])
            .write_to(std::fs::File::create(tmp.path()).unwrap())
            .unwrap();
        let len = std::fs::metadata(tmp.path()).unwrap().len();

        let got = unsafe {
            Array::read_from_file_mmap(tmp.path(), 0, len, ArrayParams::default()).unwrap()
        }
        .into_typed::<i32>()
        .unwrap()
        .to_ndarray()
        .unwrap();
        assert_eq!(got, src.into_dyn());
    }

    #[cfg(not(miri))]
    #[test]
    fn mmap_source_pipeline_neg_write_to_with() {
        // Full streaming pipeline: mmap source -> lazy neg view -> write_to_with.
        // The full array is never held decompressed in memory.
        let vals: Vec<i32> = (1..=4 * 5 * 6i32).collect();
        let src = ndarray::Array::from_shape_vec([4, 5, 6], vals.clone()).unwrap();
        let expected = -&src;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        compact(vals, &[4, 5, 6], &[2, 2, 3])
            .write_to(std::fs::File::create(tmp.path()).unwrap())
            .unwrap();
        let len = std::fs::metadata(tmp.path()).unwrap().len();

        let mmap_arr = unsafe {
            Array::read_from_file_mmap(tmp.path(), 0, len, ArrayParams::default()).unwrap()
        };
        let mmap_arr = mmap_arr.into_typed::<i32>().unwrap();
        let ctx = mmap_arr.read_ctx();
        let view = -mmap_arr.as_ref();

        let mut out_buf = Cursor::new(Vec::new());
        view.write_to_with(&mut out_buf, arr_params(&[2, 2, 3]), &ctx)
            .unwrap();

        let got = Array::read_from_reader(
            Cursor::new(out_buf.into_inner()),
            None,
            ArrayParams::default(),
        )
        .unwrap()
        .into_typed::<i32>()
        .unwrap()
        .to_ndarray()
        .unwrap();
        assert_eq!(got, expected.into_dyn());
    }
}
