use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use zerocopy::{FromBytes, IntoBytes};

use crate::archive::common::{ArchiveReader, ArchiveWriter, Section};
use crate::archive::schema;
use crate::codec::{Codec, DecoderCodecConfig, Filter};
use crate::dtype::Dtype;
use crate::error::{bail, ensure, Error, ErrorKind, Result};
use crate::storage::block::{
    BlockFn, BlockSize, BlockTable, BlockTableStorage, Mmap, MmapData, Owned,
};
use crate::util::{cast_slice, cast_slice_mut, Idx};

pub trait BlockTableStorageRead: BlockTableStorage {
    fn read_section<T, R>(
        &self,
        reader: &mut ArchiveReader<R>,
        section: Section,
    ) -> Result<Self::Data<T>>
    where
        T: Copy + 'static,
        R: Read + Seek;
}

impl<S> BlockTable<S>
where
    S: BlockTableStorage,
{
    #[allow(unused)]
    pub(crate) fn write_to<W>(&self, writer: W) -> Result<()>
    where
        W: Write + Seek,
    {
        let mut writer =
            ArchiveWriter::new(writer, schema::ArchiveType::BlockTable).map_err(Error::io)?;
        self.write_content(&mut writer)
    }

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
            &self.decoder_config,
            writer,
            compressed_block_size_bound,
            &mut block_fn,
        )
    }
}

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
        table_of_contents: vec![
            schema::block_table_header::TableOfContents::BlockOffsets as i32,
            schema::block_table_header::TableOfContents::BlockDataContinuous as i32,
        ],
    };
    writer.write_message(&header).map_err(Error::io)?;

    // Write table of contents (placeholder for now, will be overwritten later)
    let mut toc = [Section::default(); 2];
    let toc_offset = writer.stream_position().map_err(Error::io)?;
    writer.write_all(toc.as_bytes()).map_err(Error::io)?;

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
    let chunk = (64 * 1024 / compressed_block_size_bound).max(1) as u64; // try to write 64KB at a time

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
    toc = [
        Section {
            offset: block_offsets_offset as i64 - writer.base_offset as i64,
            size: (block_offsets_num * size_of::<u64>() as u64) as u64,
        },
        Section {
            offset: block_data_offset as i64 - writer.base_offset as i64,
            size: block_data_total_len,
        },
    ];
    writer
        .seek(SeekFrom::Start(toc_offset))
        .map_err(Error::io)?;
    writer.write_all(toc.as_bytes()).map_err(Error::io)?;
    writer
        .seek(SeekFrom::Start(current_pos))
        .map_err(Error::io)?;

    Ok(())
}

impl BlockTable<Owned> {
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
            "unexpected zix file type: expected {:?}, actual {:?}",
            schema::ArchiveType::BlockTable,
            schema::ArchiveType::try_from(f_meta.archive_type)
        );
        Self::read_content(&mut reader, Owned(PhantomData))
    }
}

impl<S> BlockTable<S>
where
    S: BlockTableStorage,
{
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

        ensure!(
            header.table_of_contents.len() == 2,
            InvalidArchive,
            "expected 2 sections in table of contents, got {}",
            header.table_of_contents.len()
        );

        let toc = <[Section; 2]>::read_from_io(reader.reader_mut()).map_err(Error::io)?;
        let mut block_data_section = None;
        let mut block_offsets_section = None;
        for (toc_idx, toc_entry) in header.table_of_contents().enumerate() {
            match toc_entry {
                schema::block_table_header::TableOfContents::Unspecified => {} // fail later
                schema::block_table_header::TableOfContents::BlockDataContinuous => {
                    block_data_section = Some(toc[toc_idx])
                }
                schema::block_table_header::TableOfContents::BlockOffsets => {
                    block_offsets_section = Some(toc[toc_idx])
                }
            }
        }
        let (Some(block_data_section), Some(block_offsets_section)) =
            (block_data_section, block_offsets_section)
        else {
            bail!(InvalidArchive, "missing sections in table of contents");
        };

        // Read body data sections
        let block_data = storage.read_section(reader, block_data_section)?;
        let block_offsets = storage.read_section(reader, block_offsets_section)?;

        let decoder_config = DecoderCodecConfig {
            codec,
            filters: filters
                .as_slice()
                .try_into()
                .map_err(|_| Error::new(ErrorKind::InvalidArchive, "too many filters in header"))?,
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
            data: (data, len),
        })
    }
}
