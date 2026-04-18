use std::cell::UnsafeCell;

use crate::dtype::{Alignment, Dtype};
use crate::error::{ensure, Error, ErrorKind, Result};
use crate::util::{AlignedBytes, AlternatingBuffers};

#[derive(Clone, Debug)]
pub enum Codec {
    Zstd,
}

#[derive(Clone, Debug)]
pub struct EncoderParams {
    codec: Codec,
    level: u8,
    filters: arrayvec::ArrayVec<Filter, 4>,
}
impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            codec: Codec::Zstd,
            level: 3,
            filters: ([Filter::ByteShuffle].as_slice()).try_into().unwrap(),
        }
    }
}
impl EncoderParams {
    pub fn get_filters(&self) -> &[Filter] {
        &self.filters
    }
}

pub(crate) struct Encoder {
    pub(crate) dtype: Dtype,
    pub(crate) filters: Vec<Filter>,
    pub(crate) compressor: Compressor,
    tmp_buf1: AlignedBytes,
    tmp_buf2: AlignedBytes,
}
pub(crate) enum Compressor {
    #[cfg(not(miri))]
    Zstd(zstd::bulk::Compressor<'static>),
    #[cfg(miri)]
    Zstd(()),
}
impl Encoder {
    pub(crate) fn new(params: &EncoderParams, dtype: Dtype) -> Result<Self> {
        let tmp_buf1 = AlignedBytes::new(dtype.alignment() as usize);
        let tmp_buf2 = tmp_buf1.clone();
        Ok(Self {
            dtype,
            filters: params.filters.to_vec(),
            tmp_buf1,
            tmp_buf2,
            compressor: match params.codec {
                Codec::Zstd => Compressor::Zstd({
                    cfg_if::cfg_if! { if #[cfg(not(miri))] {
                        zstd::bulk::Compressor::new(params.level as _).map_err(|e| {
                            Error::new(
                                ErrorKind::CodecError,
                                format!("Failed to create Zstd compressor: {e}"),
                            )
                        })?
                    } else {
                        ()
                    }}
                }),
            },
        })
    }

    pub(crate) fn encode(&mut self, data: &[u8], dst: &mut [u8]) -> Result<usize> {
        let itemsize = self.dtype.itemsize() as usize;
        ensure!(
            data.len().is_multiple_of(itemsize),
            InvalidArgument,
            "Data length is not a multiple of item size"
        );

        let mut buffers =
            AlternatingBuffers::with_const_src(data, &mut self.tmp_buf1, &mut self.tmp_buf2);
        for filter in &self.filters {
            match filter {
                Filter::ByteShuffle => {
                    let (data, buf) = buffers.edit();
                    buf.clear();
                    buf.reserve(data.len());
                    unsafe { buf.set_len(data.len()) };

                    let n = data.len() / itemsize;
                    for i in 0..n {
                        for b in 0..itemsize {
                            buf[b * n + i] = data[i * itemsize + b];
                        }
                    }
                }
            }
        }
        let data = buffers.data();

        match &mut self.compressor {
            Compressor::Zstd(compressor) => {
                cfg_if::cfg_if! { if #[cfg(not(miri))] {
                    compressor.compress_to_buffer(data, dst).map_err(|e| {
                        Error::new(
                            ErrorKind::CodecError,
                            format!("Failed to compress data with Zstd: {e}"),
                        )
                    })
                } else {
                    dst.copy_from_slice(data);
                    Ok(data.len())
                } }
            }
        }
    }

    pub(crate) fn encode_bound(&self, src_size: usize) -> usize {
        match &self.compressor {
            Compressor::Zstd(_) => {
                cfg_if::cfg_if! { if #[cfg(not(miri))] {
                    zstd::zstd_safe::compress_bound(src_size)
                } else {
                    src_size
                } }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Filter {
    ByteShuffle,
}

#[derive(Clone, Debug, Default)]
pub struct DecoderParams {}
#[derive(Clone, Debug)]
pub struct DecoderCodecConfig {
    pub(crate) codec: Codec,
    pub(crate) filters: Vec<Filter>,
    pub(crate) dtype: Dtype,
}

pub(crate) struct Decoder<'a> {
    #[cfg(not(miri))]
    inner: UnsafeCell<zstd::bulk::Decompressor<'static>>,
    context: &'a ReadContext,

    dtype: &'a Dtype,
    filters: &'a [Filter],
}
impl<'a> Decoder<'a> {
    pub(crate) fn new(context: &'a ReadContext, config: &'a DecoderCodecConfig) -> Self {
        Self {
            #[cfg(not(miri))]
            inner: UnsafeCell::new(zstd::bulk::Decompressor::new().unwrap()),
            context,
            dtype: &config.dtype,
            filters: &config.filters,
        }
    }

    pub(crate) fn decode(&self, cdata: &[u8], dst: &mut [u8]) -> Result<usize> {
        let dst_len = dst.len();
        let dst_ptr = dst.as_mut_ptr();

        let tmp_buf1 = unsafe { &mut *self.context.tmp_buf1.get() };
        let tmp_buf2 = unsafe { &mut *self.context.tmp_buf2.get() };
        let mut buffers = AlternatingBuffers::new(tmp_buf1, tmp_buf2);
        let decompress_out = if self.filters.is_empty() {
            dst
        } else {
            let (_, tmp_buf) = buffers.edit();
            tmp_buf.clear();
            tmp_buf.reserve(dst.len());
            unsafe {
                #[allow(clippy::uninit_vec)]
                tmp_buf.set_len(dst.len())
            };
            tmp_buf.as_mut_slice()
        };

        cfg_if::cfg_if! { if #[cfg(not(miri))] {
            let inner = unsafe { &mut *self.inner.get() };
            let nbytes = inner.decompress_to_buffer(cdata, decompress_out).map_err(|e| {
                Error::new(
                    ErrorKind::CodecError,
                    format!("Failed to decompress data with Zstd: {e}"),
                )
            })?;
        } else {
            decompress_out.copy_from_slice(cdata);
            let nbytes = cdata.len();
        } }

        // Apply filters in reverse order
        for (f_idx, filter) in self.filters.iter().enumerate().rev() {
            match filter {
                Filter::ByteShuffle => {
                    let (data, buf) = buffers.edit();
                    let buf = if f_idx > 0 {
                        buf.clear();
                        buf.reserve(data.len());
                        unsafe {
                            #[allow(clippy::uninit_vec)]
                            buf.set_len(data.len())
                        };
                        buf.as_mut_slice()
                    } else {
                        unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) }
                    };

                    let itemsize = self.dtype.itemsize() as usize;
                    ensure!(
                        data.len().is_multiple_of(itemsize),
                        InvalidArgument,
                        "Data length is not a multiple of item size, cant apply byte shuffle filter"
                    );
                    let n = data.len() / itemsize;
                    for i in 0..n {
                        for b in 0..itemsize {
                            buf[i * itemsize + b] = data[b * n + i];
                        }
                    }
                }
            }
        }

        Ok(nbytes)
    }
}

pub struct ReadContext {
    tmp_buffers: TmpBufferPool,
    tmp_buf1: UnsafeCell<AlignedBytes>,
    tmp_buf2: UnsafeCell<AlignedBytes>,
}
impl ReadContext {
    pub fn new(#[allow(unused)] decoder_params: &DecoderParams) -> Result<Self> {
        let tmp_buf1 = AlignedBytes::new(8);
        let tmp_buf2 = tmp_buf1.clone();
        Ok(Self {
            tmp_buffers: TmpBufferPool::new(),
            tmp_buf1: UnsafeCell::new(tmp_buf1),
            tmp_buf2: UnsafeCell::new(tmp_buf2),
        })
    }

    pub(crate) fn decoder<'a>(&'a self, config: &'a DecoderCodecConfig) -> Decoder<'a> {
        Decoder::new(self, config)
    }

    pub(crate) fn tmp_buf(&self, size: usize, alignment: Alignment) -> TmpBuf<'_> {
        self.tmp_buffers.get(size, alignment)
    }
}

struct TmpBufferPool {
    align8: UnsafeCell<Vec<AlignedBytes>>,
    align_other: UnsafeCell<Vec<(Alignment, Vec<AlignedBytes>)>>,
}
impl TmpBufferPool {
    fn new() -> Self {
        Self {
            align8: UnsafeCell::new(Vec::new()),
            align_other: UnsafeCell::new(Vec::new()),
        }
    }

    fn get(&self, size: usize, alignment: Alignment) -> TmpBuf<'_> {
        let (pool, pool_align) = self.get_pool(alignment);
        let pool = unsafe { &mut *pool };
        let tmp_buf = pool
            .pop()
            .unwrap_or_else(|| AlignedBytes::with_capacity(pool_align as usize, size));
        let mut buf = TmpBuf {
            buf: tmp_buf,
            buffers: self,
        };
        buf.set_len(size);
        buf
    }

    fn return_buf(&self, mut buf: AlignedBytes) {
        buf.clear();
        let (pool, _) = self.get_pool(buf.alignment() as Alignment);
        let pool = unsafe { &mut *pool };
        pool.push(buf);
    }

    fn get_pool(&self, alignment: Alignment) -> (*mut Vec<AlignedBytes>, Alignment) {
        match alignment {
            1 | 2 | 4 | 8 => (self.align8.get(), 8),
            _ => {
                let align_other = unsafe { &mut *self.align_other.get() };
                let pool = align_other
                    .iter_mut()
                    .find(|(align, _)| *align == alignment)
                    .map(|(_, pool)| pool);
                let pool = match pool {
                    Some(pool) => pool,
                    None => {
                        align_other.push((alignment, Vec::new()));
                        &mut align_other.last_mut().unwrap().1
                    }
                };
                (pool, alignment)
            }
        }
    }
}
pub(crate) struct TmpBuf<'a> {
    buf: AlignedBytes,
    buffers: &'a TmpBufferPool,
}
impl TmpBuf<'_> {
    pub(crate) fn set_len(&mut self, new_len: usize) {
        self.buf.clear();
        self.buf.reserve(new_len);
        unsafe { self.buf.set_len(new_len) };
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
}
impl Drop for TmpBuf<'_> {
    fn drop(&mut self) {
        // take self.buf
        let mut buf = AlignedBytes::new(self.buf.alignment());
        std::mem::swap(&mut self.buf, &mut buf);

        self.buffers.return_buf(buf);
    }
}
