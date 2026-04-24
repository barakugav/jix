use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;

use zerocopy::{FromBytes, IntoBytes};

use crate::archive::common::{ArchiveReader, ArchiveWriter, Section};
use crate::archive::schema;
use crate::codec::{Codec, DecoderCodecConfig, Filter};
use crate::dtype::Dtype;
use crate::error::{bail, ensure, Error, ErrorKind, Result};
use crate::storage::block::{BlockSize, BlockTable, BlockTableStorage, Mmap, MmapData, Owned};
use crate::util::{cast_slice, cast_slice_mut};

pub trait BlockTableStorageRead: BlockTableStorage {
    fn read_content<T, R>(
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

        // Write body data sections
        let block_data = writer
            .write_section(self.block_data.as_ref(), align_of::<u8>())
            .map_err(Error::io)?;
        let block_offsets = writer
            .write_section(
                unsafe { cast_slice::<u64, u8>(self.block_offsets.as_ref()) },
                align_of::<u64>(),
            )
            .map_err(Error::io)?;

        // Go back and write table of contents
        toc = [block_offsets, block_data];
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

        let toc = <[Section; 2]>::read_from_io(reader.inner_mut()).map_err(Error::io)?;
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
        let block_data = storage.read_content(reader, block_data_section)?;
        let block_offsets = storage.read_content(reader, block_offsets_section)?;

        let decoder_config = DecoderCodecConfig {
            codec,
            filters,
            dtype: Dtype::from_proto(header.dtype.as_ref().unwrap()).unwrap(),
        };

        Self::new(
            block_data,
            block_offsets,
            header.nitems,
            header.block_size as BlockSize,
            decoder_config,
        )
    }
}

impl BlockTableStorageRead for Owned {
    fn read_content<T, R>(
        &self,
        reader: &mut ArchiveReader<R>,
        section: Section,
    ) -> Result<Self::Data<T>>
    where
        T: Copy + 'static,
        R: Read + Seek,
    {
        ensure!(
            section.size.is_multiple_of(size_of::<T>() as u64),
            InvalidArchive,
            "section size is not a multiple of item size"
        );
        let len = section.size as usize / std::mem::size_of::<T>();
        let mut data = Vec::<T>::with_capacity(len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(len)
        };
        reader
            .read_section_into(&section, unsafe {
                cast_slice_mut::<T, u8>(data.as_mut_slice())
            })
            .map_err(Error::io)?;
        Ok(data)
    }
}

impl BlockTableStorageRead for Mmap {
    fn read_content<T, R>(
        &self,
        reader: &mut ArchiveReader<R>,
        section: Section,
    ) -> Result<Self::Data<T>>
    where
        T: Copy + 'static,
        R: Read + Seek,
    {
        ensure!(
            section.size.is_multiple_of(size_of::<T>() as u64),
            InvalidArchive,
            "section size is not a multiple of item size"
        );
        let len = section.size as usize / std::mem::size_of::<T>();
        let offset = reader.base_offset() as i64 + section.offset;
        let offset = offset as usize;
        let data = self.0[offset..].as_ptr().cast::<T>();
        ensure!(
            data.is_aligned(),
            InvalidArchive,
            "data offset is not properly aligned"
        );

        Ok(MmapData {
            mmap: self.0.clone(),
            data: (data, len),
        })
    }
}
