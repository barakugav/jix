use std::io::{self, Read, Seek, Write};
use std::ops::{Deref, DerefMut};

use prost::Message;

use crate::schema;
use crate::schema::Section;

const HEADER_MAGIC: u32 = 0x20d95dac;

pub(crate) struct Writer<W> {
    writer: W,
    base_offset: u64,
    tmp_buf: Vec<u8>,
}
impl<W> Writer<W> {
    pub(crate) fn new(mut writer: W) -> io::Result<Self>
    where
        W: Write + Seek,
    {
        let base_offset = writer.stream_position()?;
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

    pub(crate) fn write_header(
        &mut self,
        footer_spec: Option<&schema::FooterSpec>,
    ) -> io::Result<()>
    where
        W: Write,
    {
        let header = schema::Header {
            magic: HEADER_MAGIC,
            version: env!("CARGO_PKG_VERSION").to_string(),
            footer_spec: footer_spec.cloned(),
        };
        self.write_message(&header)?;
        Ok(())
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
        footer_spec: &schema::FooterSpec,
    ) -> io::Result<()>
    where
        W: Write + Seek,
    {
        let main_section_offset = self.stream_position()?;
        let main_section_size = self.write_message(main_section)?;

        let footer = schema::Footer {
            magic: footer_spec.magic,
            main_section_offset: main_section_offset as i64 - self.base_offset as i64,
            main_section_length: main_section_size as u64,
        };
        let footer_size = self.write_message(&footer)?;
        assert_eq!(footer_size, footer_spec.size as usize);
        Ok(())
    }

    pub(crate) fn new_footer_spec(&self) -> schema::FooterSpec {
        let magic = {
            // Generate a random magic.
            // we dont want to depend on an additional crate just for this, and rand is not yet stable in std.
            // We can create a hash_map::RandomState and hash a fixed value with it to get a random value that is
            // different on every run.
            // hash_map::RandomState uses the unstable std random under the hood.
            use std::hash::BuildHasher;
            let magic = 0xd3be9ab084788933_u64;
            let magic = std::collections::hash_map::RandomState::new().hash_one(magic);
            let magic = (magic >> 32) as u32 ^ (magic as u32);
            magic
        };
        schema::FooterSpec { size: 24, magic }
    }
}
impl<W> Deref for Writer<W> {
    type Target = W;
    fn deref(&self) -> &W {
        &self.writer
    }
}
impl<W> DerefMut for Writer<W> {
    fn deref_mut(&mut self) -> &mut W {
        &mut self.writer
    }
}

pub(crate) struct Reader<R> {
    reader: R,
    base_offset: u64,
    length: u64,
    tmp_buf: Vec<u8>,
}
impl<R> Reader<R> {
    pub(crate) fn new(mut reader: R, length: u64) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        let base_offset = reader.stream_position()?;
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

    pub(crate) fn read_header(&mut self) -> io::Result<schema::Header>
    where
        R: Read,
    {
        let header = self.try_read_message::<schema::Header>()?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of file while reading header",
            )
        })?;
        if header.magic != HEADER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid header magic",
            ));
        }
        Ok(header)
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

    pub(crate) fn read_footer_and_main_section<M>(
        &mut self,
        footer_spec: &Option<schema::FooterSpec>,
    ) -> io::Result<(schema::Footer, M)>
    where
        R: Read + Seek,
        M: Message + Default,
    {
        let footer_spec = footer_spec.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing footer spec in header")
        })?;

        if footer_spec.size as u64 > self.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small to contain footer",
            ));
        }
        self.seek_relative_to_base(self.length as i64 - (footer_spec.size as i64))?;
        let footer_bytes = self.read_slice(footer_spec.size as usize)?;
        let footer = schema::Footer::decode_length_delimited(footer_bytes)?;
        if footer.magic != footer_spec.magic {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid footer magic",
            ));
        }

        self.seek_relative_to_base(footer.main_section_offset)?;
        let main_section_bytes = self.read_slice(footer.main_section_length as usize)?;
        let main_section = M::decode_length_delimited(main_section_bytes)?;

        Ok((footer, main_section))
    }

    pub(crate) fn seek_relative_to_base(&mut self, offset: i64) -> io::Result<u64>
    where
        R: Seek,
    {
        let pos = (self.base_offset as i64 + offset) as u64;
        self.seek(std::io::SeekFrom::Start(pos))
    }
}
impl<R> Deref for Reader<R> {
    type Target = R;
    fn deref(&self) -> &R {
        &self.reader
    }
}
impl<R> DerefMut for Reader<R> {
    fn deref_mut(&mut self) -> &mut R {
        &mut self.reader
    }
}
