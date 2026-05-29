use std::io::{self, Read, Seek, Write};
use std::ops::{Deref, DerefMut};

use prost::Message;
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::archive::schema::{self, ArchiveType};
use crate::archive::version::VERSION_U64;
use crate::error::{ensure, Error, Result};

const MAGIC: &[u8; 4] = b"ZIX1";

#[derive(Default, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
struct Header {
    magic: [u8; 4],
}

#[derive(Default, Clone, Copy, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub(crate) struct Section {
    /// Offsets in bytes from the beginning of the archive.
    pub(crate) offset: i64,
    /// Size of the section in bytes.
    pub(crate) size: u64,
}

pub(crate) struct ArchiveWriter<W> {
    writer: W,
    pub(crate) base_offset: u64,
    tmp_buf: Vec<u8>,
}
impl<W> ArchiveWriter<W> {
    pub(crate) fn new(mut writer: W, archive_type: ArchiveType) -> io::Result<Self>
    where
        W: Write + Seek,
    {
        let base_offset = writer.stream_position()?;
        let mut writer = Self {
            writer,
            base_offset,
            tmp_buf: Vec::new(),
        };

        let header = Header { magic: *MAGIC };
        writer.write_all(header.as_bytes())?;

        let file_metadata = schema::FileMetadata {
            archive_type: archive_type as i32,
            lib_version_semver: VERSION_U64,
        };
        writer.write_message(&file_metadata)?;

        Ok(writer)
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

    // pub(crate) fn write_section(&mut self, data: &[u8], alignment: usize) -> io::Result<Section>
    // where
    //     W: Write + Seek,
    // {
    //     let offset = self.stream_position()?;
    //     let padded_offset = offset.ceil_to_multiple(alignment as u64);
    //     let padding = padded_offset - offset;
    //     if padding > 0 {
    //         self.tmp_buf.clear();
    //         self.tmp_buf.resize(padding as usize, 0);
    //         self.writer.write_all(self.tmp_buf.as_slice())?;
    //         self.tmp_buf.clear();
    //     }
    //     let offset = padded_offset;

    //     self.write_all(data)?;
    //     let size = data.len() as u64;
    //     Ok(Section {
    //         offset: offset as i64 - self.base_offset as i64,
    //         size,
    //     })
    // }
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
    reader: std::io::Take<R>,
    total_size: u64,
    tmp_buf: Vec<u8>,
}
impl<R> ArchiveReader<R> {
    pub(crate) fn new(reader: R, length: Option<u64>) -> Result<Self>
    where
        R: Read + Seek,
    {
        let total_size = length.unwrap_or(u64::MAX);
        let mut reader = reader.take(total_size);
        ensure!(
            reader.limit() >= size_of::<Header>() as u64,
            InvalidArchive,
            "zix file too short: length={}",
            reader.limit()
        );

        let header = Header::read_from_io(&mut reader).map_err(Error::io)?;
        ensure!(
            &header.magic == MAGIC,
            InvalidArchive,
            "invalid zix file: invalid header magic {:?}",
            header.magic
        );

        Ok(Self {
            reader,
            total_size,
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
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.tmp_buf.set_len(len)
        };
        self.reader.read_exact(self.tmp_buf.as_mut_slice())?;
        Ok(self.tmp_buf.as_slice())
    }

    pub(crate) fn reader_mut(&mut self) -> &mut impl Read
    where
        R: Read,
    {
        &mut self.reader
    }

    // pub(crate) fn read_section(&mut self, section: &Section) -> io::Result<Vec<u8>>
    // where
    //     R: Read + Seek,
    // {
    //     let mut buf = Vec::with_capacity(section.size as usize);
    //     #[allow(clippy::uninit_vec)]
    //     unsafe {
    //         buf.set_len(section.size as usize)
    //     };
    //     self.read_section_into(section, buf.as_mut_slice())?;
    //     Ok(buf)
    // }

    pub(crate) fn read_section_into(&mut self, section: &Section, buf: &mut [u8]) -> io::Result<()>
    where
        R: Read + Seek,
    {
        assert_eq!(buf.len() as u64, section.size);
        self.seek_relative_to_base(section.offset)?;
        self.reader.read_exact(buf)?;
        Ok(())
    }

    pub(crate) fn read_file_meta(&mut self) -> io::Result<schema::FileMetadata>
    where
        R: Read,
    {
        self.read_message()
    }

    pub(crate) fn seek_relative_to_base(&mut self, offset: i64) -> io::Result<()>
    where
        R: Seek,
    {
        // Prefer seek_relative over seek as BufReader always discard its buffer on seek
        let pos_relative = offset - self.reader.stream_position()? as i64;
        self.reader.seek_relative(pos_relative)
    }

    pub(crate) fn check_section_bounds(&self, section: &Section) -> Result<()> {
        ensure!(
            0 <= section.offset && section.offset as u64 + section.size <= self.total_size,
            InvalidArchive,
            "section is out of archive bounds"
        );
        Ok(())
    }
}
