use std::io;

pub(crate) struct Encoder {
    inner: zstd::bulk::Compressor<'static>,
}
impl Encoder {
    pub(crate) fn new(level: i32) -> io::Result<Self> {
        Ok(Self {
            inner: zstd::bulk::Compressor::new(level)?,
        })
    }

    pub(crate) fn encode(&mut self, data: &[u8], dst: &mut [u8]) -> io::Result<usize> {
        self.inner.compress_to_buffer(data, dst)
    }

    pub(crate) fn encode_bound(&self, src_size: usize) -> usize {
        zstd::zstd_safe::compress_bound(src_size)
    }
}
pub struct ReadContext {
    inner: zstd::bulk::Decompressor<'static>,
}
impl ReadContext {
    pub(crate) fn new() -> io::Result<Self> {
        Ok(Self {
            inner: zstd::bulk::Decompressor::new()?,
        })
    }

    pub(crate) fn decode(&mut self, cdata: &[u8], dst: &mut [u8]) -> io::Result<usize> {
        self.inner.decompress_to_buffer(cdata, dst)
    }
}
