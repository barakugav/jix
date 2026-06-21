use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::archive::common::{ArchiveReader, ArchiveWriter, Section};
use crate::archive::schema;
use crate::codec::{Codec, DecoderCodecConfig, Filter};
use crate::dtype::Dtype;
use crate::error::{bail, ensure, Error, ErrorKind, Result};
use crate::storage::block::{
    BlockFn, BlockSize, BlockTable, BlockTableStorage, Mmap, MmapData, Owned,
};
use crate::util::arrayvec::ArrayVec;
use crate::util::{cast_slice, cast_slice_mut, value_as_bytes, value_from_io, Idx, SendSyncPtr};
use crate::{ArchiveValidation, ElementType, TypeDyn};

/// Extension of [`BlockTableStorage`] that can populate its `Data<T>` arrays from an archive.
///
/// [`BlockTable::read_content`] calls `read_section` twice - once for `block_data` (`T = u8`)
/// and once for `block_offsets` (`T = u64`) - delegating all I/O and lifetime management to the
/// storage-specific implementation:
///
/// - [`Owned`] - reads the section bytes into a freshly allocated `Vec<T>`.
/// - [`Mmap`] - returns a zero-copy [`MmapData<T>`] pointer into the already-mapped region.
pub(crate) trait BlockTableStorageRead: BlockTableStorage {
    /// Read one archive section into a typed array appropriate for this storage backend.
    ///
    /// # Arguments
    ///
    /// - `reader` - the open archive reader; the section is located via `section.offset`.
    /// - `section` - byte offset and length of the section within the archive.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArchive` if the section size is not a multiple of `size_of::<T>()` or if
    /// the data pointer is not properly aligned (mmap only). Propagates I/O errors.
    fn read_section<T, R>(
        &self,
        reader: &mut ArchiveReader<R>,
        section: Section,
    ) -> Result<Self::Data<T>>
    where
        T: Copy + 'static,
        R: Read + Seek;
}

impl<S, ET> BlockTable<S, ET>
where
    ET: ElementType,
    S: BlockTableStorage,
{
    /// Write this `BlockTable` as a self-contained archive to `writer`.
    ///
    /// Wraps `writer` in a fresh [`ArchiveWriter`] with type `BlockTable`, then delegates to
    /// [`write_content`](Self::write_content). Use this when writing a standalone `.jix` file;
    /// use `write_content` when embedding into a larger archive.
    #[allow(unused)]
    pub(crate) fn write_to<W>(&self, writer: W) -> Result<()>
    where
        W: Write + Seek,
    {
        let mut writer =
            ArchiveWriter::new(writer, schema::ArchiveType::BlockTable).map_err(Error::io)?;
        self.write_content(&mut writer)
    }

    /// Write this `BlockTable`'s content into an already-opened [`ArchiveWriter`].
    ///
    /// Obtains a zero-copy [`BlockFn`] via [`BlockTable::to_block_fn`] and drives a
    /// [`BlockArchiveWriter`] with an explicit batch loop. Use this when embedding into a larger
    /// archive that already has an open writer; use [`write_to`](Self::write_to) for standalone
    /// files.
    pub(crate) fn write_content<W>(&self, writer: &mut ArchiveWriter<W>) -> Result<()>
    where
        W: Write + Seek,
    {
        assert!(self.nitems.is_multiple_of(self.block_size as u64));
        let nblocks = self.nitems / self.block_size as u64;

        let (mut block_fn, compressed_block_size_bound) = self.to_block_fn();
        let mut block_writer =
            BlockArchiveWriter::start(writer, nblocks, self.block_size, self.decoder_config())?;
        let chunk = chunk_for(compressed_block_size_bound);
        for block_index in (0..nblocks).step_by(chunk as usize) {
            let blocks = block_index..(block_index + chunk).min(nblocks);
            let (data, offsets) =
                block_fn.get_compressed_blocks(blocks, block_writer.data_len())?;
            block_writer.write_compressed_blocks(data)?;
            block_writer.write_offsets(offsets)?;
        }
        block_writer.finalize()
    }
}

/// Choose how many blocks to process per batch so that roughly 64 KB of compressed data is
/// produced per call. `compressed_block_size_bound` is the largest single block's compressed size.
pub(crate) fn chunk_for(compressed_block_size_bound: usize) -> u64 {
    (64 * 1024 / compressed_block_size_bound.max(1)).max(1) as u64
}

/// Stateful writer that serializes a block table's content into an already-opened
/// [`ArchiveWriter`], owning all the section-offset bookkeeping.
///
/// The caller drives an explicit loop: [`start`](Self::start) writes the header and TOC
/// placeholder and positions the stream at the data section; then for each batch of blocks the
/// caller calls [`write_compressed_blocks`](Self::write_compressed_blocks)
/// and [`write_offsets`](Self::write_offsets); finally[`finalize`](Self::finalize) close it out.
///
/// # Wire layout (within the writer's section)
///
/// ```text
/// [ protobuf BlockTableHeader ]
/// [ TOC placeholder: 2 * Section (overwritten by finalize) ]
/// [ alignment padding to u64 boundary ]
/// [ block_offsets: (nblocks + 1) * u64, or 0 entries when nblocks == 0 ]
/// [ block_data:    concatenated compressed block bytes ]
/// ```
pub(crate) struct BlockArchiveWriter<'w, W> {
    writer: &'w mut ArchiveWriter<W>,
    /// Absolute offset of the TOC placeholder, overwritten by `finalize`.
    toc_offset: u64,
    /// Absolute offset where the `block_offsets` section begins (u64-aligned).
    block_offsets_offset: u64,
    /// Absolute offset where the `block_data` section begins.
    block_data_offset: u64,
    /// Number of u64 entries in the offsets section: `nblocks + 1`, or 0 when `nblocks == 0`.
    block_offsets_num: u64,
    /// Offsets accumulated since the last flush.
    offsets_write_buf: Vec<u64>,
    /// Number of offset entries already persisted to the offsets section.
    written_offsets_num: u64,
    /// Total bytes of compressed block data written so far (also the next batch's base offset).
    block_data_total_len: u64,
    /// Whether the leading `0` offset has been pushed yet.
    first_batch: bool,
}

impl<'w, W> BlockArchiveWriter<'w, W>
where
    W: Write + Seek,
{
    /// Write the header and TOC placeholder, reserve the offsets section, and seek to the data
    /// section. On return the stream is positioned at `block_data_offset`, ready for the loop.
    pub(crate) fn start(
        writer: &'w mut ArchiveWriter<W>,
        nblocks: u64,
        block_size: BlockSize,
        decoder_config: &DecoderCodecConfig,
    ) -> Result<Self> {
        let nitems = nblocks * block_size as u64;

        // Write header
        let header = schema::BlockTableHeader {
            dtype: Some(decoder_config.dtype.to_proto()),
            nitems,
            block_size: block_size as u64,
            codec: Some(schema::Codec {
                kind: Some(match decoder_config.codec {
                    Codec::Zstd => schema::codec::Kind::Zstd(()),
                }),
            }),
            filters: decoder_config
                .filters
                .iter()
                .map(|f| schema::Filter {
                    kind: Some(match f {
                        Filter::ByteShuffle => schema::filter::Kind::ByteShuffle(()),
                        Filter::BitShuffle => schema::filter::Kind::BitShuffle(()),
                    }),
                })
                .collect(),
            body_description: Some(schema::block_table_header::BodyDescription::ContinuousV1(())),
        };
        writer.write_message(&header).map_err(Error::io)?;

        // Write table of contents (placeholder for now, overwritten by finalize)
        let table_of_contents = [Section::default(); 2];
        let toc_offset = writer.stream_position().map_err(Error::io)?;
        writer
            .write_all(unsafe { value_as_bytes(&table_of_contents) })
            .map_err(Error::io)?;

        let block_offsets_offset = {
            let current_offset = writer.stream_position().map_err(Error::io)?;
            let block_offsets_offset = current_offset.ceil_to_multiple(align_of::<u64>() as u64);
            let padding = (block_offsets_offset - current_offset) as usize;
            if padding > 0 {
                let padding_buf = [0u8; size_of::<u64>()];
                writer
                    .write_all(&padding_buf[..padding])
                    .map_err(Error::io)?;
            }
            block_offsets_offset
        };
        let block_offsets_num = if nblocks == 0 { 0 } else { nblocks + 1 };
        let block_data_offset = block_offsets_offset + block_offsets_num * size_of::<u64>() as u64;

        // Seek to data section; the loop writes there without seeking.
        writer
            .seek(SeekFrom::Start(block_data_offset))
            .map_err(Error::io)?;

        Ok(Self {
            writer,
            toc_offset,
            block_offsets_offset,
            block_data_offset,
            block_offsets_num,
            offsets_write_buf: Vec::new(),
            written_offsets_num: 0,
            block_data_total_len: 0,
            first_batch: true,
        })
    }

    /// The total compressed bytes written so far; pass this as `base_offset` to
    /// [`BlockFn::get_compressed_blocks`](crate::storage::block::BlockFn::get_compressed_blocks).
    pub(crate) fn data_len(&self) -> u64 {
        self.block_data_total_len
    }

    /// Append one batch of compressed block bytes at the current data-section position (no seek).
    pub(crate) fn write_compressed_blocks(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data).map_err(Error::io)
    }

    /// Record one batch's cumulative end-offsets. On the first call, pushes the leading `0`.
    /// Advances `data_len` to the last offset in the batch.
    pub(crate) fn write_offsets(&mut self, offsets: &[u64]) -> Result<()> {
        debug_assert!(self.block_data_total_len <= *offsets.first().unwrap());
        debug_assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
        self.block_data_total_len = *offsets.last().unwrap();

        if self.first_batch {
            self.offsets_write_buf.push(0);
            self.first_batch = false;
        }
        self.offsets_write_buf.extend_from_slice(offsets);
        if self.offsets_write_buf.len() > 8192 {
            self.flush_offsets()?;
            self.writer
                .seek(SeekFrom::Start(
                    self.block_data_offset + self.block_data_total_len,
                ))
                .map_err(Error::io)?;
        }
        Ok(())
    }

    /// Write any buffered offsets to the offsets section. Leaves the stream positioned in the
    /// offsets section (callers either seek back to the data section or proceed to finalize).
    fn flush_offsets(&mut self) -> Result<()> {
        let offsets_offset =
            self.block_offsets_offset + self.written_offsets_num * size_of::<u64>() as u64;
        self.writer
            .seek(SeekFrom::Start(offsets_offset))
            .map_err(Error::io)?;
        self.writer
            .write_all(unsafe { cast_slice::<u64, u8>(self.offsets_write_buf.as_slice()) })
            .map_err(Error::io)?;
        self.written_offsets_num += self.offsets_write_buf.len() as u64;
        self.offsets_write_buf.clear();
        Ok(())
    }

    /// Overwrite the placeholder TOC with the real section positions/sizes, then restore the
    /// stream to the end of the data section. Consumes `self`, releasing the writer borrow.
    pub(crate) fn finalize(mut self) -> Result<()> {
        self.flush_offsets()?;

        let table_of_contents = [
            Section {
                offset: self.block_offsets_offset as i64 - self.writer.base_offset as i64,
                size: self.block_offsets_num * size_of::<u64>() as u64,
            },
            Section {
                offset: self.block_data_offset as i64 - self.writer.base_offset as i64,
                size: self.block_data_total_len,
            },
        ];
        self.writer
            .seek(SeekFrom::Start(self.toc_offset))
            .map_err(Error::io)?;
        self.writer
            .write_all(unsafe { value_as_bytes(&table_of_contents) })
            .map_err(Error::io)?;
        self.writer
            .seek(SeekFrom::Start(
                self.block_data_offset + self.block_data_total_len,
            ))
            .map_err(Error::io)?;
        Ok(())
    }
}

impl BlockTable<Owned, TypeDyn> {
    /// Read a `BlockTable` from a self-contained archive, allocating storage on the heap.
    ///
    /// Wraps `reader` in an [`ArchiveReader`], validates the archive type, then delegates to
    /// [`read_content`](BlockTable::read_content) with [`Owned`] storage. Use this for standalone
    /// `.jix` files; use `read_content` when reading from a larger archive.
    ///
    /// # Arguments
    ///
    /// - `reader` - the source; must be positioned at the start of the archive.
    /// - `len` - total byte length of the archive section passed to `ArchiveReader` for bounds
    ///   checking.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArchive` if the archive type header does not match `BlockTable`.
    /// Propagates any I/O or parse error from `read_content`.
    #[allow(unused)]
    pub(crate) fn read_from<R>(reader: R, len: u64) -> Result<Self>
    where
        R: Read + Seek,
    {
        let mut reader = ArchiveReader::new(reader, Some(len))?;
        let f_meta = reader.read_file_meta().map_err(Error::io)?;
        ensure!(
            f_meta.archive_type == schema::ArchiveType::BlockTable as i32,
            InvalidArchive,
            "unexpected jix file type: expected {:?}, actual {:?}",
            schema::ArchiveType::BlockTable,
            schema::ArchiveType::try_from(f_meta.archive_type)
        );
        Self::read_content(
            &mut reader,
            Owned(PhantomData),
            ArchiveValidation::default(),
        )
    }
}

impl<S> BlockTable<S, TypeDyn>
where
    S: BlockTableStorage,
{
    /// Deserialize a `BlockTable` from an already-opened [`ArchiveReader`].
    ///
    /// Reads the protobuf `BlockTableHeader`, resolves the codec and filters, validates the TOC,
    /// then dispatches to `storage.read_section()` for the raw block-data and block-offsets
    /// sections. The `storage` parameter determines how those sections are held in memory:
    /// [`Owned`] copies them into heap `Vec`s; [`Mmap`] returns zero-copy pointers into the
    /// mapping.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArchive` if the header is malformed (missing codec, unknown filter, wrong
    /// TOC section count, missing required sections, or bad dtype). Propagates I/O errors from
    /// `reader` and errors from [`BlockTable::new`].
    pub(crate) fn read_content<R>(
        reader: &mut ArchiveReader<R>,
        storage: S,
        validation: ArchiveValidation,
    ) -> Result<Self>
    where
        R: Read + Seek,
        S: BlockTableStorageRead,
    {
        let header = reader
            .read_message::<schema::BlockTableHeader>()
            .map_err(Error::io)?;
        let codec = header.codec.and_then(|c| c.kind).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidArchive,
                "unknown or missing codec in header",
            )
        })?;
        let codec = match codec {
            schema::codec::Kind::Zstd(()) => Codec::Zstd,
        };
        let filters = header
            .filters
            .iter()
            .map(|f| {
                Ok(match f.kind {
                    Some(schema::filter::Kind::ByteShuffle(())) => Filter::ByteShuffle,
                    Some(schema::filter::Kind::BitShuffle(())) => Filter::BitShuffle,
                    None => {
                        return Err(Error::new(
                            ErrorKind::InvalidArchive,
                            "unknown filter in header",
                        ));
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Read body data sections
        let (block_data, block_offsets) = match header.body_description {
            Some(schema::block_table_header::BodyDescription::ContinuousV1(())) => {
                let [block_offsets_section, block_data_section] = unsafe {
                    value_from_io::<[Section; 2]>(reader.reader_mut()).map_err(Error::io)?
                };

                let block_data = storage.read_section(reader, block_data_section)?;
                let block_offsets = storage.read_section(reader, block_offsets_section)?;
                (block_data, block_offsets)
            }
            None => bail!(InvalidArchive, "missing body description in header"),
        };

        // validation checks
        let check_offsets = match validation {
            ArchiveValidation::Minimal => false,
            ArchiveValidation::Strict => true,
        };
        if check_offsets && !block_offsets.as_ref().is_empty() {
            let block_offsets = block_offsets.as_ref();
            let monotonic = block_offsets.windows(2).all(|w| w[0] <= w[1]);
            // enough to check the last because of monotonicity
            let in_bounds = *block_offsets.last().unwrap() <= block_data.as_ref().len() as u64;
            ensure!(
                monotonic && in_bounds,
                InvalidArchive,
                "invalid block offsets: monotonic={monotonic}, in_bounds={in_bounds}"
            );
        }

        let decoder_config = DecoderCodecConfig {
            codec,
            filters: ArrayVec::from_slice(filters.as_slice()).ok_or_else(|| {
                Error::new(ErrorKind::InvalidArchive, "too many filters in header")
            })?,
            dtype: Dtype::from_proto(header.dtype.as_ref().unwrap()).unwrap(),
        };

        Self::new(
            block_data,
            block_offsets,
            header.block_size as BlockSize,
            decoder_config,
        )
    }
}

impl BlockTableStorageRead for Owned {
    /// Read `section` into a heap-allocated `Vec<T>`.
    ///
    /// Allocates uninitialised capacity, reads the raw bytes from `reader` directly into the
    /// buffer, then transmutes `Vec<MaybeUninit<T>>` to `Vec<T>` after the read succeeds.
    fn read_section<T, R>(
        &self,
        reader: &mut ArchiveReader<R>,
        section: Section,
    ) -> Result<Self::Data<T>>
    where
        T: Copy + 'static,
        R: Read + Seek,
    {
        reader.check_section_bounds(&section)?;

        ensure!(
            section.size.is_multiple_of(size_of::<T>() as u64),
            InvalidArchive,
            "section size is not a multiple of item size"
        );
        let len = section.size as usize / size_of::<T>();

        let mut data = Vec::<MaybeUninit<T>>::with_capacity(len);
        unsafe { data.set_len(len) };
        reader
            .read_section_into(&section, unsafe {
                cast_slice_mut::<MaybeUninit<T>, u8>(data.as_mut_slice())
            })
            .map_err(Error::io)?;
        Ok(unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(data) })
    }
}

impl BlockTableStorageRead for Mmap {
    /// Return a zero-copy [`MmapData<T>`] pointing into the memory-mapped region.
    ///
    /// No bytes are copied; `mmap` is cloned (an `Arc` bump) to keep the mapping alive for the
    /// lifetime of the returned `MmapData`. Fails if the section's byte offset within the mapping
    /// is not aligned to `T`.
    fn read_section<T, R>(
        &self,
        reader: &mut ArchiveReader<R>,
        section: Section,
    ) -> Result<Self::Data<T>>
    where
        T: Copy + 'static,
        R: Read + Seek,
    {
        reader.check_section_bounds(&section)?;

        ensure!(
            section.size.is_multiple_of(size_of::<T>() as u64),
            InvalidArchive,
            "section size is not a multiple of item size"
        );
        let len = section.size as usize / size_of::<T>();

        let offset = self.base_offset as i64 + section.offset;
        let offset = offset as usize;
        let data = self.mmap[offset..].as_ptr().cast::<T>();
        ensure!(
            data.is_aligned(),
            InvalidArchive,
            "data section offset is not properly aligned"
        );

        Ok(MmapData {
            mmap: self.mmap.clone(),
            data: (unsafe { SendSyncPtr::new(data) }, len),
        })
    }
}
