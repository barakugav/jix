use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::archive::common::{ArchiveReader, ArchiveWriter, Section};
use crate::archive::schema;
use crate::codec::{Codec, DecoderCodecConfig, Filter, MAX_FILTERS};
use crate::dtype::Dtype;
use crate::error::{bail, ensure, error, Error, Result};
use crate::storage::block::{
    BlockLocation2, BlockSize, BlockTable, BlockTableBuilder, BlockTableStorage, Mmap, MmapData,
    Owned,
};
use crate::util::arrayvec::ArrayVec;
use crate::util::{cast_slice, cast_slice_mut, value_as_bytes, value_from_io, Idx, SendSyncPtr};
use crate::{ArchiveValidation, ElementType, TypeDyn};

/// Extension of [`BlockTableStorage`] that can populate its `Data<T>` arrays from an archive.
///
/// [`BlockTable::read_content`] calls `read_section` twice - once for `block_data` (`T = u8`)
/// and once for `blocks_loc` (`T = BlockLocation2`) - delegating all I/O and lifetime management to
/// the storage-specific implementation:
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
    /// Drives a [`BlockArchiveWriter`] with an explicit loop, feeding it one block at a time by
    /// slicing directly into `self.block_data` at each block's recorded location (zero-copy - no
    /// re-compression). Use this when embedding into a larger archive that already has an open
    /// writer; use [`write_to`](Self::write_to) for standalone files.
    pub(crate) fn write_content<W>(&self, writer: &mut ArchiveWriter<W>) -> Result<()>
    where
        W: Write + Seek,
    {
        assert!(self.nitems.is_multiple_of(self.block_size as u64));
        let nblocks = self.nitems / self.block_size as u64;

        let block_data = self.block_data.as_ref();

        let mut block_writer =
            BlockArchiveWriter::start(writer, nblocks, self.block_size, self.decoder_config())?;
        for block_index in 0..nblocks {
            let (offset, len) = self.block_location(block_index);
            let data = &block_data[offset as usize..][..len as usize];
            block_writer.write_compressed_block(block_index, data)?;
        }
        block_writer.finalize()
    }
}

/// Stateful writer that serializes a block table's content into an already-opened
/// [`ArchiveWriter`], owning all the section-offset bookkeeping.
///
/// The caller drives an explicit loop: [`start`](Self::start) writes the header and TOC
/// placeholder and positions the stream at the data section; then for each block the caller calls
/// [`write_compressed_block`](Self::write_compressed_block); finally [`finalize`](Self::finalize)
/// closes it out.
///
/// # Wire layout (within the writer's section)
///
/// ```text
/// [ protobuf BlockTableHeader ]
/// [ TOC placeholder: 2 * Section (overwritten by finalize) ]
/// [ alignment padding to 8-byte boundary ]
/// [ blocks_loc:   ((nblocks + 1) >> 1) * BlockLocation2  (two blocks packed per entry) ]
/// [ block_data:    concatenated compressed block bytes ]
/// ```
pub(crate) struct BlockArchiveWriter<'w, W> {
    writer: &'w mut ArchiveWriter<W>,
    /// Absolute offset of the TOC placeholder, overwritten by `finalize`.
    toc_offset: u64,
    /// Absolute offset where the `blocks_loc` section begins (8-byte aligned).
    blocks_loc_offset: u64,
    /// Absolute offset where the `block_data` section begins.
    block_data_offset: u64,
    /// Packed block locations indexed by logical block index. Written to the locations section in
    /// `finalize`, since blocks may arrive out of order.
    blocks_loc: Vec<BlockLocation2>,
    /// Total bytes of compressed block data written so far (the running data-section offset).
    block_data_total_len: u64,
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
            body_description: Some(schema::block_table_header::BodyDescription::ContiguousV1(())),
        };
        writer.write_message(&header).map_err(Error::io)?;

        // Write table of contents (placeholder for now, overwritten by finalize)
        let table_of_contents = [Section::default(); 2];
        let toc_offset = writer.stream_position().map_err(Error::io)?;
        writer
            .write_all(unsafe { value_as_bytes(&table_of_contents) })
            .map_err(Error::io)?;

        let blocks_loc_offset = {
            let current_offset = writer.stream_position().map_err(Error::io)?;
            let blocks_loc_offset =
                current_offset.ceil_to_multiple(align_of::<BlockLocation2>() as u64);
            let padding = (blocks_loc_offset - current_offset) as usize;
            if padding > 0 {
                let padding_buf = [0u8; align_of::<BlockLocation2>()];
                writer
                    .write_all(&padding_buf[..padding])
                    .map_err(Error::io)?;
            }
            blocks_loc_offset
        };
        let block_data_offset =
            blocks_loc_offset + ((nblocks + 1) >> 1) * size_of::<BlockLocation2>() as u64;

        // Seek to data section; the loop writes there without seeking.
        writer
            .seek(SeekFrom::Start(block_data_offset))
            .map_err(Error::io)?;

        Ok(Self {
            writer,
            toc_offset,
            blocks_loc_offset,
            block_data_offset,
            blocks_loc: vec![BlockLocation2::default(); ((nblocks + 1) >> 1) as usize],
            block_data_total_len: 0,
        })
    }
}

impl<'w, W> BlockTableBuilder for BlockArchiveWriter<'w, W>
where
    W: Write + Seek,
{
    type Output = ();

    /// Write one compressed block's bytes at the current data-section position (no seek), in call
    /// order, and record its `(offset, len)` location at logical index `block_index`: the offset is
    /// the running data-section total before the write and the length is `compressed.len()`. The
    /// locations are buffered in memory and written out in [`finalize`](Self::finalize).
    fn write_compressed_block(&mut self, block_index: u64, compressed: &[u8]) -> Result<()> {
        let offset = self.block_data_total_len;
        self.writer.write_all(compressed).map_err(Error::io)?;
        self.block_data_total_len += compressed.len() as u64;

        let loc = &mut self.blocks_loc[(block_index >> 1) as usize];
        let lane = (block_index & 1) as usize;
        loc.offset[lane] = offset;
        loc.len[lane] = compressed.len() as u32;
        Ok(())
    }

    /// Write the accumulated locations section, overwrite the placeholder TOC with the real section
    /// positions/sizes, then restore the stream to the end of the data section. Consumes `self`,
    /// releasing the writer borrow.
    fn finalize(self) -> Result<()> {
        // The block data was written sequentially; now persist the full locations section.
        self.writer
            .seek(SeekFrom::Start(self.blocks_loc_offset))
            .map_err(Error::io)?;
        self.writer
            .write_all(unsafe { cast_slice::<BlockLocation2, u8>(self.blocks_loc.as_slice()) })
            .map_err(Error::io)?;

        let table_of_contents = [
            Section {
                offset: self.blocks_loc_offset as i64 - self.writer.base_offset as i64,
                size: self.blocks_loc.len() as u64 * size_of::<BlockLocation2>() as u64,
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
    /// then dispatches to `storage.read_section()` for the raw block-data and block-locations
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
        let codec = header
            .codec
            .and_then(|c| c.kind)
            .ok_or_else(|| error!(InvalidArchive, "unknown or missing codec in header"))?;
        let codec = match codec {
            schema::codec::Kind::Zstd(()) => Codec::Zstd,
        };
        ensure!(
            header.filters.len() <= MAX_FILTERS,
            InvalidArchive,
            "too many filters in header"
        );
        let filters = header
            .filters
            .iter()
            .map(|f| {
                Ok(match f.kind {
                    Some(schema::filter::Kind::ByteShuffle(())) => Filter::ByteShuffle,
                    Some(schema::filter::Kind::BitShuffle(())) => Filter::BitShuffle,
                    None => {
                        bail!(InvalidArchive, "unknown filter in header");
                    }
                })
            })
            .collect::<Result<ArrayVec<_, MAX_FILTERS>>>()?;

        // Read body data sections
        let (block_data, blocks_loc) = match header.body_description {
            Some(schema::block_table_header::BodyDescription::ContiguousV1(())) => {
                let [blocks_loc_section, block_data_section] = unsafe {
                    value_from_io::<[Section; 2]>(reader.reader_mut()).map_err(Error::io)?
                };

                let block_data = storage.read_section(reader, block_data_section)?;
                let blocks_loc =
                    storage.read_section::<BlockLocation2, _>(reader, blocks_loc_section)?;
                (block_data, blocks_loc)
            }
            None => bail!(
                InvalidArchive,
                "missing or unknown body description in header"
            ),
        };

        let block_size = header.block_size as BlockSize;
        ensure!(block_size > 0, InvalidArchive, "block_size must be > 0");
        ensure!(
            header.nitems.is_multiple_of(block_size as u64),
            InvalidArchive,
            "nitems {} is not a multiple of block_size {block_size}",
            header.nitems
        );
        let nblocks = header.nitems / block_size as u64;
        let expected_blocks_loc_len = (nblocks + 1) >> 1;
        ensure!(
            blocks_loc.as_ref().len() as u64 == expected_blocks_loc_len,
            InvalidArchive,
            "block locations length {} does not match expected {expected_blocks_loc_len} for {nblocks} blocks",
            blocks_loc.as_ref().len()
        );

        // validation checks
        let check_block_locations = match validation {
            ArchiveValidation::Minimal => false,
            ArchiveValidation::Strict => true,
        };
        if check_block_locations {
            let data_len = block_data.as_ref().len() as u64;
            let blocks_loc = blocks_loc.as_ref();
            // Each block's (offset, len) location must lie within the data section. Blocks may be
            // stored in any order, so there is no cross-block ordering to check.
            let valid = (0..nblocks).all(|i| {
                let loc = &blocks_loc[(i >> 1) as usize];
                let lane = (i & 1) as usize;
                let (offset, len) = (loc.offset[lane], loc.len[lane] as u64);
                offset <= data_len && len <= data_len - offset
            });
            ensure!(
                valid,
                InvalidArchive,
                "invalid block location: an (offset, len) pair is out of bounds"
            );
        }

        let decoder_config = DecoderCodecConfig {
            codec,
            filters,
            dtype: Dtype::from_proto(header.dtype.as_ref().unwrap()).unwrap(),
        };

        Self::new(block_data, blocks_loc, nblocks, block_size, decoder_config)
    }
}

impl BlockTableStorageRead for Owned {
    /// Read `section` into a heap-allocated `Vec<T>`.
    ///
    /// Allocates uninitialized capacity, reads the raw bytes from `reader` directly into the
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
