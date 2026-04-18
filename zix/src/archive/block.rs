use std::io::{Read, Seek, SeekFrom, Write};

use zerocopy::{FromBytes, IntoBytes};

use crate::archive::common::{ArchiveReader, ArchiveWriter, Section};
use crate::archive::schema;
use crate::codec::{Codec, DecoderCodecConfig, Filter};
use crate::dtype::Dtype;
use crate::error::{bail, ensure, Error, ErrorKind, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage, Mmap, Owned};
use crate::util::{cast_slice, cast_slice_mut};

impl<S> BlockTable<S> {
    #[allow(unused)]
    pub(crate) fn write_to<W>(&self, writer: W) -> Result<()>
    where
        W: Write + Seek,
        S: BlockTableStorage,
    {
        let mut writer =
            ArchiveWriter::new(writer, schema::ArchiveType::BlockTable).map_err(Error::io)?;
        self.write_content(&mut writer)
    }

    pub(crate) fn write_content<W>(&self, writer: &mut ArchiveWriter<W>) -> Result<()>
    where
        W: Write + Seek,
        S: BlockTableStorage,
    {
        // Write header
        let header = schema::BlockTableHeader {
            dtype: Some(self.dtype().to_proto()),
            nitems: self.nitems,
            block_size: self.block_size as u64,
            codec: Some(schema::Codec {
                kind: Some(match self.decoder_config.codec {
                    Codec::Zstd => schema::codec::Kind::Zstd(()),
                }),
            }),
            filters: self
                .decoder_config
                .filters
                .iter()
                .map(|f| schema::Filter {
                    kind: Some(match f {
                        Filter::ByteShuffle => schema::filter::Kind::ByteShuffle(()),
                    }),
                })
                .collect(),
            table_of_contents: vec![
                schema::block_table_header::TableOfContents::BlockOffsets as i32,
                schema::block_table_header::TableOfContents::Cdata as i32,
            ],
        };
        writer.write_message(&header).map_err(Error::io)?;

        // Write table of contents (placeholder for now, will be overwritten later)
        let mut toc = [Section::default(); 2];
        let toc_offset = writer.stream_position().map_err(Error::io)?;
        writer.write_all(toc.as_bytes()).map_err(Error::io)?;

        // Write body data sections
        let cdata = writer
            .write_section(self.storage.cdata(), align_of::<u8>())
            .map_err(Error::io)?;
        let block_offsets = writer
            .write_section(
                unsafe { cast_slice::<u64, u8>(self.storage.block_offsets()) },
                align_of::<u64>(),
            )
            .map_err(Error::io)?;

        // Go back and write table of contents
        toc = [block_offsets, cdata];
        let current_pos = writer.stream_position().map_err(Error::io)?;
        writer
            .seek(SeekFrom::Start(toc_offset))
            .map_err(Error::io)?;
        writer.write_all(toc.as_bytes()).map_err(Error::io)?;
        writer
            .seek(SeekFrom::Start(current_pos))
            .map_err(Error::io)?;

        Ok(())
    }
}
impl BlockTable<Owned> {
    #[allow(unused)]
    pub(crate) fn read_from<R>(reader: R, len: u64) -> Result<Self>
    where
        R: Read + Seek,
    {
        let mut reader = ArchiveReader::new(reader, len)?;
        let f_meta = reader.read_file_meta().map_err(Error::io)?;
        ensure!(
            f_meta.archive_type == schema::ArchiveType::BlockTable as i32,
            InvalidArchive,
            "unexpected zix file type: expected {:?}, actual {:?}",
            schema::ArchiveType::BlockTable,
            schema::ArchiveType::try_from(f_meta.archive_type)
        );
        Self::read_content(&mut reader, Owned::read_from)
    }
}

impl<S> BlockTable<S> {
    pub(crate) fn read_content<R>(
        reader: &mut ArchiveReader<R>,
        read_storage: impl FnOnce(&mut ArchiveReader<R>, Section, Section) -> Result<S>,
    ) -> Result<Self>
    where
        R: Read + Seek,
        S: BlockTableStorage,
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

        let toc = <[Section; 2]>::read_from_io(reader.inner_mut()).map_err(Error::io)?;
        let mut cdata_section = None;
        let mut block_offsets_section = None;
        for (toc_idx, toc_entry) in header.table_of_contents().enumerate() {
            match toc_entry {
                schema::block_table_header::TableOfContents::Unspecified => {} // fail later
                schema::block_table_header::TableOfContents::Cdata => {
                    cdata_section = Some(toc[toc_idx])
                }
                schema::block_table_header::TableOfContents::BlockOffsets => {
                    block_offsets_section = Some(toc[toc_idx])
                }
            }
        }
        let (Some(cdata_section), Some(block_offsets_section)) =
            (cdata_section, block_offsets_section)
        else {
            bail!(InvalidArchive, "missing sections in table of contents");
        };

        // Read body data sections
        let storage = read_storage(reader, cdata_section, block_offsets_section)?;

        let decoder_config = DecoderCodecConfig {
            codec,
            filters,
            dtype: Dtype::from_proto(header.dtype.as_ref().unwrap()).unwrap(),
        };

        Self::new(
            storage,
            header.nitems,
            header.block_size as BlockSize,
            decoder_config,
        )
    }
}

impl Owned {
    pub(crate) fn read_from<R>(
        reader: &mut ArchiveReader<R>,
        cdata_section: Section,
        block_offsets_section: Section,
    ) -> Result<Owned>
    where
        R: Read + Seek,
    {
        let cdata = reader.read_section(&cdata_section).map_err(Error::io)?;

        let block_offsets = {
            let block_offsets_section = &block_offsets_section;
            let block_offsets_len =
                block_offsets_section.size as usize / std::mem::size_of::<u64>();
            let mut block_offsets = Vec::<u64>::with_capacity(block_offsets_len);
            #[allow(clippy::uninit_vec)]
            unsafe {
                block_offsets.set_len(block_offsets_len)
            };
            reader
                .read_section_into(block_offsets_section, unsafe {
                    cast_slice_mut::<u64, u8>(block_offsets.as_mut_slice())
                })
                .map_err(Error::io)?;
            block_offsets
        };

        Ok(Owned {
            cdata,
            block_offsets,
        })
    }
}

impl Mmap {
    pub(crate) fn new(mmap: memmap2::Mmap, cdata: Section, block_offsets: Section) -> Self {
        let cdata = {
            let offset = cdata.offset as usize;
            let size = cdata.size as usize;
            let slice = &mmap[offset..offset + size];
            // SAFETY: We require that the mmap outlives the returned slice, and that the caller does not mutate the slice.
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(slice) }
        };
        let block_offsets = {
            let offset = block_offsets.offset as usize;
            let size = block_offsets.size as usize;
            let buf = &mmap[offset..offset + size];
            let slice = unsafe { cast_slice::<u8, u64>(buf) };
            unsafe { std::mem::transmute::<&[u64], &'static [u64]>(slice) }
        };
        Self {
            cdata,
            block_offsets,
            mmap,
        }
    }

    pub(crate) fn read_from<R>(
        reader: &mut ArchiveReader<R>,
        mut cdata_section: Section,
        mut block_offsets_section: Section,
        mmap: memmap2::Mmap,
    ) -> Result<Mmap>
    where
        R: Read + Seek,
    {
        let base_offset = reader.base_offset() as i64;
        cdata_section.offset += base_offset;
        block_offsets_section.offset += base_offset;
        Ok(Mmap::new(mmap, cdata_section, block_offsets_section))
    }
}
