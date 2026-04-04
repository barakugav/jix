use std::io::{self, Read, Seek, Write};
use std::ops::{Deref, DerefMut};

use prost::{Message, bytes};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes};

use crate::schema::Section;
use crate::schema::{self, ArchiveType};

const MAGIC: &[u8; 4] = b"ZIX1";

#[derive(Default, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
struct Header {
    magic: [u8; 4],
}

#[derive(Default, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
struct Footer {
    main_section_size: u32,
    file_metadata_size: u32,
    magic: [u8; 4],
}

pub(crate) struct ArchiveWriter<W> {
    writer: W,
    base_offset: u64,
    tmp_buf: Vec<u8>,
}
impl<W> ArchiveWriter<W> {
    pub(crate) fn new(mut writer: W) -> io::Result<Self>
    where
        W: Write + Seek,
    {
        let base_offset = writer.stream_position()?;

        let header = Header { magic: *MAGIC };
        writer.write_all(header.as_bytes())?;

        Ok(Self {
            writer,
            base_offset,
            tmp_buf: Vec::new(),
        })
    }

    pub(crate) fn write_message(&mut self, message: &impl Message) -> io::Result<usize>
    where
        W: Write,
    {
        self.tmp_buf.clear();
        message.encode_length_delimited(&mut self.tmp_buf)?;
        let msg_len = self.tmp_buf.len();
        self.writer.write_all(&self.tmp_buf)?;
        self.tmp_buf.clear();
        Ok(msg_len)
    }

    pub(crate) fn into_inner(self) -> W {
        self.writer
    }

    pub(crate) fn inner_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    pub(crate) fn write_section(&mut self, data: &[u8], alignment: usize) -> io::Result<Section>
    where
        W: Write + Seek,
    {
        let offset = self.stream_position()?;
        let padded_offset = offset.div_ceil(alignment as u64) * alignment as u64;
        let padding = padded_offset - offset;
        if padding > 0 {
            self.tmp_buf.clear();
            self.tmp_buf.resize(padding as usize, 0);
            self.writer.write_all(self.tmp_buf.as_slice())?;
            self.tmp_buf.clear();
        }
        let offset = padded_offset;

        self.write_all(data)?;
        let size = data.len() as u64;
        Ok(Section {
            offset: offset as i64 - self.base_offset as i64,
            size,
        })
    }

    pub(crate) fn write_main_section_and_footer(
        &mut self,
        main_section: &impl Message,
        archive_type: ArchiveType,
    ) -> io::Result<()>
    where
        W: Write + Seek,
    {
        let main_section_size = self.write_message(main_section)?;
        let file_metadata_size = self.write_message(&schema::FileMetadata {
            archive_type: archive_type as i32,
            lib_version_semver: 0, // TODO
        })?;
        let footer = Footer {
            main_section_size: main_section_size.try_into().unwrap(),
            file_metadata_size: file_metadata_size.try_into().unwrap(),
            magic: *MAGIC,
        };
        self.write_all(footer.as_bytes())?;
        Ok(())
    }
}
impl<W> Deref for ArchiveWriter<W> {
    type Target = W;
    fn deref(&self) -> &W {
        &self.writer
    }
}
impl<W> DerefMut for ArchiveWriter<W> {
    fn deref_mut(&mut self) -> &mut W {
        &mut self.writer
    }
}

pub(crate) struct ArchiveReader<R> {
    reader: R,
    base_offset: u64,
    length: u64,
    tmp_buf: Vec<u8>,
}
impl<R> ArchiveReader<R> {
    pub(crate) fn new(mut reader: R, length: u64) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        if length < (size_of::<Header>() + size_of::<Footer>()) as u64 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "zix file too short",
            ));
        }

        let base_offset = reader.stream_position()?;

        let header = Header::read_from_io(&mut reader)?;
        if &header.magic != MAGIC {
            return Err(io::Error::other("invalid zix file: invalid header magic"));
        }

        Ok(Self {
            reader,
            base_offset,
            length,
            tmp_buf: Vec::new(),
        })
    }

    pub(crate) fn read_message<M>(&mut self) -> io::Result<M>
    where
        R: Read,
        M: Message + Default,
    {
        self.try_read_message::<M>()?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of file while reading message",
            )
        })
    }
    pub(crate) fn try_read_message<M>(&mut self) -> io::Result<Option<M>>
    where
        R: Read,
        M: Message + Default,
    {
        let length = match self.read_uleb128()? {
            Some(length) => length,
            None => return Ok(None),
        };
        let msg_bytes = self.read_slice(length as usize)?;
        Ok(Some(M::decode(msg_bytes)?))
    }

    fn read_uleb128(&mut self) -> std::io::Result<Option<u64>>
    where
        R: Read,
    {
        let mut x: u64 = 0;
        for i in 0.. {
            let b = match self.read_byte()? {
                Some(b) => b,
                None => {
                    if i == 0 {
                        return Ok(None);
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "unexpected end of file",
                        ));
                    }
                }
            };
            x |= u64::from(b & 0x7F) << (i * 7);
            if b < 0x80 {
                return Ok(Some(x));
            }
        }
        unreachable!()
    }

    fn read_byte(&mut self) -> std::io::Result<Option<u8>>
    where
        R: Read,
    {
        let mut byte = 0;
        Ok(match self.reader.read(std::slice::from_mut(&mut byte))? {
            0 => None,
            _ => Some(byte),
        })
    }

    fn read_slice(&mut self, len: usize) -> std::io::Result<&[u8]>
    where
        R: Read,
    {
        self.tmp_buf.clear();
        self.tmp_buf.reserve(len);
        unsafe { self.tmp_buf.set_len(len) };
        self.reader.read_exact(self.tmp_buf.as_mut_slice())?;
        Ok(self.tmp_buf.as_slice())
    }

    pub(crate) fn into_inner(self) -> R {
        self.reader
    }

    pub(crate) fn inner_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    pub(crate) fn read_section(&mut self, section: &Section) -> io::Result<Vec<u8>>
    where
        R: Read + Seek,
    {
        let mut buf = Vec::with_capacity(section.size as usize);
        unsafe { buf.set_len(section.size as usize) };
        self.read_section_into(section, buf.as_mut_slice())?;
        Ok(buf)
    }

    pub(crate) fn read_section_into(&mut self, section: &Section, buf: &mut [u8]) -> io::Result<()>
    where
        R: Read + Seek,
    {
        assert_eq!(buf.len() as u64, section.size);
        self.seek_relative_to_base(section.offset)?;
        self.read_exact(buf)?;
        Ok(())
    }

    pub(crate) fn read_file_meta_and_main_section(
        &mut self,
    ) -> io::Result<(schema::FileMetadata, &[u8])>
    where
        R: Read + Seek,
    {
        // Read footer
        debug_assert!(self.length >= (size_of::<Header>() + size_of::<Footer>()) as u64);
        let footer_offset = self.length - size_of::<Footer>() as u64;
        self.seek_relative_to_base(footer_offset as i64)?;
        let footer = Footer::read_from_io(&mut self.reader)?;
        if &footer.magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid footer magic",
            ));
        }

        // Read main section and file metadata bytes (in a single read)
        let main_section_offset = footer_offset
            .checked_sub(footer.file_metadata_size as u64 + footer.main_section_size as u64)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid footer: main section and file metadata sizes are too large",
                )
            })?;
        self.seek_relative_to_base(main_section_offset as i64)?;
        let (main_section_bytes, file_meta_bytes) = self
            .read_slice(footer.file_metadata_size as usize + footer.main_section_size as usize)?
            .split_at(footer.main_section_size as usize);

        // Decode file metadata
        let file_meta = schema::FileMetadata::decode_length_delimited(file_meta_bytes)?;

        Ok((file_meta, main_section_bytes))
    }

    pub(crate) fn seek_relative_to_base(&mut self, offset: i64) -> io::Result<u64>
    where
        R: Seek,
    {
        let pos: u64 = (self.base_offset as i64 + offset).try_into().unwrap();
        self.seek(std::io::SeekFrom::Start(pos))
    }
}
impl<R> Deref for ArchiveReader<R> {
    type Target = R;
    fn deref(&self) -> &R {
        &self.reader
    }
}
impl<R> DerefMut for ArchiveReader<R> {
    fn deref_mut(&mut self) -> &mut R {
        &mut self.reader
    }
}
