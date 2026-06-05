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
use crate::{ElementType, TypeDyn};

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
    /// Obtains a zero-copy [`BlockFn`] via [`BlockTable::to_block_fn`] and forwards to
    /// [`write_content_impl`]. Use this when embedding into a larger archive that already has an
    /// open writer; use [`write_to`](Self::write_to) for standalone files.
    pub(crate) fn write_content<W>(&self, writer: &mut ArchiveWriter<W>) -> Result<()>
    where
        W: Write + Seek,
    {
        assert!(self.nitems.is_multiple_of(self.block_size as u64));
        let nblocks = self.nitems / self.block_size as u64;

        let (mut block_fn, compressed_block_size_bound) = self.to_block_fn();
        write_content_impl(
            nblocks,
            self.block_size,
            self.decoder_config(),
            writer,
            compressed_block_size_bound,
            &mut block_fn,
        )
    }
}

/// Serialize a block table's content into an already-opened [`ArchiveWriter`].
///
/// This is the shared inner implementation used by both [`BlockTable::write_content`] (which
/// serializes an existing in-memory table) and the array compaction path (which compresses on the
/// fly via a closure-backed [`BlockFn`]).
///
/// # Wire layout (within the writer's section)
///
/// ```text
/// [ protobuf BlockTableHeader ]
/// [ TOC placeholder: 2 * Section (overwritten at end) ]
/// [ alignment padding to u64 boundary ]
/// [ block_offsets: (nblocks + 1) * u64, or 0 entries when nblocks == 0 ]
/// [ block_data:    concatenated compressed block bytes ]
/// ```
///
/// The function seeks between the offsets section and the data section during the loop
/// to flush accumulated offsets every ~8 192 entries, bounding memory use regardless of
/// `nblocks`. After all blocks are written, it seeks back to overwrite the placeholder TOC
/// with the real section positions and sizes, then restores the stream position to the end
/// of the data.
///
/// # Arguments
///
/// - `nblocks` - total block count; may be zero.
/// - `block_size` - items per block, written verbatim into the header.
/// - `decoder_config` - codec, filters, and dtype written into the header.
/// - `writer` - destination; must support `Seek` for the deferred TOC write-back.
/// - `compressed_block_size_bound` - passed straight through to the batch-sizing logic
///   (targets ~64 KB of compressed data per `block_fn` call).
/// - `block_fn` - data source; called once per batch of blocks.
///
/// # Errors
///
/// Returns an I/O or codec error on any failure.
pub(crate) fn write_content_impl<W>(
    nblocks: u64,
    block_size: BlockSize,
    decoder_config: &DecoderCodecConfig,
    writer: &mut ArchiveWriter<W>,
    compressed_block_size_bound: usize,
    block_fn: &mut impl BlockFn,
) -> Result<()>
where
    W: Write + Seek,
{
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

    // Write table of contents (placeholder for now, will be overwritten later)
    let mut table_of_contents = [Section::default(); 2];
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

    let mut offsets_write_buf = Vec::<u64>::new();
    let mut written_offsets_num = 0;

    let mut block_data_total_len = 0;
    let chunk = (64 * 1024 / compressed_block_size_bound.max(1)).max(1) as u64; // try to write 64KB at a time

    // seek to data section, as we dont seek inside the loop and assume we are already at the data section
    writer
        .seek(SeekFrom::Start(block_data_offset))
        .map_err(Error::io)?;

    for block_index in (0..nblocks).step_by(chunk as usize) {
        let blocks = block_index..(block_index + chunk).min(nblocks);
        let base_offset = block_data_total_len;

        // Get blocks data
        let (data, offsets) = block_fn.get_compressed_blocks(blocks, base_offset)?;

        // Write compressed data
        // Write without seek, assuming we already in the data section
        writer.write_all(data).map_err(Error::io)?;

        // Record offsets
        if block_index == 0 {
            offsets_write_buf.push(0);
        }
        debug_assert!(block_data_total_len <= *offsets.first().unwrap());
        debug_assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
        offsets_write_buf.extend_from_slice(offsets);

        // Actually persist the offsets from time to time
        if offsets_write_buf.len() > 8192 {
            let offsets_offset =
                block_offsets_offset + written_offsets_num * size_of::<u64>() as u64;
            let current_offset = writer.stream_position().map_err(Error::io)?;
            // seek to correct position in offsets section
            writer
                .seek(SeekFrom::Start(offsets_offset))
                .map_err(Error::io)?;
            // write offsets
            writer
                .write_all(unsafe { cast_slice::<u64, u8>(offsets_write_buf.as_slice()) })
                .map_err(Error::io)?;
            written_offsets_num += offsets_write_buf.len() as u64;
            offsets_write_buf.clear();
            // seek back to data section
            writer
                .seek(SeekFrom::Start(current_offset))
                .map_err(Error::io)?;
        }

        block_data_total_len = *offsets.last().unwrap();
    }
    let current_pos = writer.stream_position().map_err(Error::io)?;

    // Flush offsets write buf
    let offsets_offset = block_offsets_offset + written_offsets_num * size_of::<u64>() as u64;
    writer
        .seek(SeekFrom::Start(offsets_offset))
        .map_err(Error::io)?;
    writer
        .write_all(unsafe { cast_slice::<u64, u8>(offsets_write_buf.as_slice()) })
        .map_err(Error::io)?;

    // Go back and write table of contents
    table_of_contents = [
        Section {
            offset: block_offsets_offset as i64 - writer.base_offset as i64,
            size: block_offsets_num * size_of::<u64>() as u64,
        },
        Section {
            offset: block_data_offset as i64 - writer.base_offset as i64,
            size: block_data_total_len,
        },
    ];
    writer
        .seek(SeekFrom::Start(toc_offset))
        .map_err(Error::io)?;
    writer
        .write_all(unsafe { value_as_bytes(&table_of_contents) })
        .map_err(Error::io)?;
    writer
        .seek(SeekFrom::Start(current_pos))
        .map_err(Error::io)?;

    Ok(())
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
        Self::read_content(&mut reader, Owned(PhantomData))
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
    pub(crate) fn read_content<R>(reader: &mut ArchiveReader<R>, storage: S) -> Result<Self>
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
        let len = section.size as usize / std::mem::size_of::<T>();

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
        let len = section.size as usize / std::mem::size_of::<T>();

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
