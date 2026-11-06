//! Internal block codec implementation: filter pipeline plus the codec (Zstd) used to compress and
//! decompress each array block. The user-facing documentation lives on
//! [`Compact`](crate::storage::Compact); configuration is exposed through
//! [`ArrayParams`](crate::ArrayParams).

mod filter;
pub use filter::*;
mod bit_shuffle;
mod byte_shuffle;

use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::buf_pool::BufferPool;
use crate::dtype::{Alignment, Dtype};
#[allow(unused_imports)]
use crate::error::{ensure, error, Result};
use crate::util::arrayvec::ArrayVec;
use crate::util::cpu_cache::CACHE_LINE_SIZE;
use crate::util::{AlignedBytes, AlternatingBuffers};
use crate::ReadContext;

/// The compression algorithm applied to each block.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Codec {
    /// [Zstandard](https://facebook.github.io/zstd/) - a fast general-purpose compressor.
    /// Compression level is controlled by [`ArrayParams::level`](crate::ArrayParams::level).
    Zstd,
}

/// Compression configuration used when encoding array blocks.
///
/// Controls the codec, compression level, and the pre-compression filter pipeline. Filters
/// are applied to the raw element bytes **before** compression and reversed **after**
/// decompression. For numeric data, filters significantly improve the compression ratio by
/// rearranging bytes or bits into a more compressible layout.
///
/// # Defaults
///
/// [`EncoderParams::default()`] uses Zstd level 3 with [`Filter::ByteShuffle`], which is a
/// good baseline for most numeric workloads: fast encoding, reasonable ratio, and effective
/// at exploiting the byte-level regularity of uniform-dtype arrays.
#[derive(Debug, Clone)]
pub(crate) struct EncoderParams {
    pub(crate) codec: Codec,
    pub(crate) level: i8,
    pub(crate) filters: ArrayVec<Filter, MAX_FILTERS>,
}
impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            codec: Codec::Zstd,
            level: 3,
            filters: ArrayVec::from_slice([Filter::ByteShuffle].as_slice()).unwrap(),
        }
    }
}
impl EncoderParams {
    /// Set the compression codec.
    pub fn codec(&mut self, codec: Codec) -> &mut Self {
        self.codec = codec;
        self
    }

    /// Set the compression level.
    ///
    /// Higher levels trade CPU time for better compression ratios. For zstd, level 3 is the default.
    pub fn level(&mut self, level: i32) -> Result<&mut Self> {
        self.level = level.try_into().map_err(|_| {
            error!(
                InvalidArgument,
                "compression level {level} is out of the supported range"
            )
        })?;
        Ok(self)
    }

    /// Set the pre-compression filter pipeline (up to 4 filters).
    ///
    /// Filters are applied in order before compression and reversed after decompression.
    /// For most numeric dtypes, [`Filter::ByteShuffle`] (the default) provides a good
    /// compression ratio improvement with low overhead. [`Filter::BitShuffle`] can yield
    /// better ratios for low-entropy data at the cost of higher CPU usage. Pass an empty
    /// slice to disable filtering entirely.
    ///
    /// Not all combinations of filters make sense; for example, a byte shuffle followed by a bit
    /// shuffle doesn't make sense because the bit shuffle will operate on the byte-shuffled data,
    /// and the bit shuffle filter will incorrectly assume the data is in the original byte order -
    /// it will not yield incorrect results, but it probably won't improve the compression ratio.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `filters` contains more than 4 elements.
    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self> {
        ensure!(
            filters.len() <= MAX_FILTERS,
            InvalidArgument,
            "At most {MAX_FILTERS} filters are supported"
        );
        self.filters = ArrayVec::from_slice(filters).unwrap();
        Ok(self)
    }
}

pub(crate) struct Encoder {
    pub(crate) dtype: Dtype,
    pub(crate) filters: ArrayVec<Filter, MAX_FILTERS>,
    pub(crate) compressor: Compressor,
    filters_tmp_buf1: AlignedBytes,
    filters_tmp_buf2: AlignedBytes,
    buf_pool: BufferPool,
}
pub(crate) enum Compressor {
    #[cfg(not(miri))]
    Zstd(zstd::bulk::Compressor<'static>),
    #[cfg(miri)]
    Zstd(()),
}
impl Encoder {
    pub(crate) fn new(params: &EncoderParams, dtype: Dtype) -> Result<Self> {
        let filters_tmp_buf1 = AlignedBytes::new_padded(dtype.alignment().as_usize());
        let filters_tmp_buf2 = filters_tmp_buf1.clone();
        Ok(Self {
            dtype,
            filters: params.filters.clone(),
            filters_tmp_buf1,
            filters_tmp_buf2,
            buf_pool: BufferPool::new(),
            compressor: match params.codec {
                Codec::Zstd => {
                    #[cfg(not(miri))]
                    let inner = zstd::bulk::Compressor::new(params.level as i32)
                        .map_err(|e| error!(CodecError, "Failed to create Zstd compressor: {e}"))?;
                    #[cfg(miri)]
                    let inner = ();
                    Compressor::Zstd(inner)
                }
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

        let data = if self.filters.is_empty() {
            data
        } else {
            let tmp_buf1 = &mut self.filters_tmp_buf1;
            tmp_buf1.clear();
            tmp_buf1.reserve(data.len());
            unsafe { tmp_buf1.set_len(data.len()) };
            let tmp_buf1 = tmp_buf1.as_mut_slice();
            if self.filters.len() == 1 {
                self.filters[0].encode(data, tmp_buf1, &self.dtype, &self.buf_pool);
                tmp_buf1
            } else {
                let tmp_buf2 = &mut self.filters_tmp_buf2;
                tmp_buf2.clear();
                tmp_buf2.reserve(data.len());
                unsafe { tmp_buf2.set_len(data.len()) };
                let tmp_buf2 = tmp_buf2.as_mut_slice();
                let mut buffers = AlternatingBuffers::with_const_src(data, tmp_buf1, tmp_buf2);
                for filter in &self.filters {
                    let (data, buf) = buffers.edit();
                    filter.encode(data, buf, &self.dtype, &self.buf_pool);
                }
                buffers.into_data()
            }
        };

        match &mut self.compressor {
            Compressor::Zstd(compressor) => {
                #[cfg(not(miri))]
                let result = compressor
                    .compress_to_buffer(data, dst)
                    .map_err(|e| error!(CodecError, "Failed to compress data with Zstd: {e}"));
                #[cfg(miri)]
                let result = {
                    let _ = compressor;
                    dst.copy_from_slice(data);
                    Ok(data.len())
                };
                result
            }
        }
    }

    pub(crate) fn encode_bound(&self, src_size: usize) -> usize {
        match &self.compressor {
            Compressor::Zstd(_) => {
                #[cfg(not(miri))]
                {
                    zstd::zstd_safe::compress_bound(src_size)
                }
                #[cfg(miri)]
                {
                    src_size
                }
            }
        }
    }
}

/// Decoder configuration supplied by the caller at read time.
///
/// Unlike the codec type, filters, and dtype of a compressed block, which are determined by how the
/// data was written and cannot be changed, `DecoderParams` holds settings that the caller can
/// freely choose for each read session. These parameters apply regardless of the codec stored in
/// the data.
///
/// Currently no configuration options are exposed, but future versions may add settings such
/// as:
/// - Number of threads to use for parallel block decompression.
/// - Memory budget for the block cache.
///
/// Use [`DecoderParams::default()`] in the meantime. Pass an explicit instance to
/// [`ReadContext::new`] if you want forward-compatible control over these settings.
#[derive(Clone, Debug, Default)]
pub(crate) struct DecoderParams {
    _phantom: PhantomData<()>,
}

pub(crate) const MAX_FILTERS: usize = 4;

/// The codec configuration encoded alongside the array data, required to decode it.
///
/// Every field in `DecoderCodecConfig` is fixed at write time and must match exactly what was
/// used when the data was encoded - none of it can be chosen or overridden at read time.
///
/// This struct is populated from the stored array metadata and passed to [`Decoder`] internally.
/// Users do not construct it directly; it is derived from the array's on-disk representation
/// when a compressed array is read back from an archive.
#[derive(Debug, Clone)]
pub(crate) struct DecoderCodecConfig {
    pub(crate) codec: Codec,
    pub(crate) filters: ArrayVec<Filter, MAX_FILTERS>,
    pub(crate) dtype: Dtype,
}

pub(crate) struct Decoder<'a> {
    #[cfg(not(miri))]
    inner: &'a UnsafeCell<zstd::bulk::Decompressor<'static>>,
    context: &'a ReadContext,

    dtype: &'a Dtype,
    filters: &'a [Filter],
}
impl<'a> Decoder<'a> {
    #[inline]
    pub(crate) fn new(context: &'a ReadContext, config: &'a DecoderCodecConfig) -> Self {
        Self {
            #[cfg(not(miri))]
            inner: &context.decompressor,
            context,
            dtype: &config.dtype,
            filters: &config.filters,
        }
    }

    pub(crate) fn decode(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        let dst_len = dst.len();
        let mut buffers = None;
        let mut tmp_buf;
        let decompress_out = if self.filters.is_empty() {
            dst
        } else {
            tmp_buf = self
                .context
                .allocate_buf(dst.len(), Alignment::new(CACHE_LINE_SIZE).unwrap());
            let tmp_buf = &mut tmp_buf.as_mut_slice()[..dst.len()];

            let dst_ptr = dst.as_mut_ptr();
            let dst_buf_odd = self.filters.len().is_multiple_of(2);
            let (buf1, buf2) = if dst_buf_odd {
                // we have 1 decompression + even number of filter, so we are going to write to a
                // total of odd number of buffers. For the last buffer to be dst, he also need to
                // be the first.
                // the 'secondary_buf' of AlternatingBuffers is the one returned as mut on the
                // first edit.
                (tmp_buf, dst)
            } else {
                (dst, tmp_buf)
            };

            buffers = Some(AlternatingBuffers::new(buf1, buf2));

            let (_, tmp_buf) = buffers.as_mut().unwrap().edit();
            debug_assert_eq!(dst_buf_odd, tmp_buf.as_mut_ptr() == dst_ptr);
            tmp_buf
        };

        #[cfg(not(miri))]
        let nbytes = {
            let inner = unsafe { &mut *self.inner.get() };
            inner
                .decompress_to_buffer(src, decompress_out)
                .map_err(|e| error!(CodecError, "Failed to decompress data with Zstd: {e}"))?
        };
        #[cfg(miri)]
        let nbytes = {
            decompress_out.copy_from_slice(src);
            src.len()
        };
        ensure!(
            nbytes == dst_len,
            InvalidArchive,
            "Unexpected decoded block size: got {nbytes}, expected {dst_len}"
        );

        // Apply filters in reverse order
        for filter in self.filters.iter().rev() {
            let (data, buf) = buffers.as_mut().unwrap().edit();
            filter.decode(data, buf, self.dtype, &self.context.buffer_pool);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn level_out_of_i8_range_errors() {
        for level in [128, 1000, i32::MAX, -129, -1000, i32::MIN] {
            let err = EncoderParams::default().level(level).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidArgument);
            assert!(err.message().contains("compression level"), "{err}");
        }
    }

    #[test]
    fn level_in_i8_range_is_accepted() {
        for level in [i8::MIN, -5, 0, 3, 19, 22, i8::MAX] {
            let mut params = EncoderParams::default();
            params.level(level as i32).unwrap();
            assert_eq!(params.level, level);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // the miri stand-in for zstd needs equal lengths
    fn a_short_decode_errors_before_the_filters_touch_dst() {
        let src = zstd::bulk::compress(&[1u8; 8], 3).unwrap();
        let ctx = ReadContext::default();
        for nfilters in [0, 1, 3] {
            let config = DecoderCodecConfig {
                codec: Codec::Zstd,
                filters: (0..nfilters).map(|_| Filter::ByteShuffle).collect(),
                dtype: Dtype::new_ref::<u8>().clone(),
            };
            let mut dst = [0xEEu8; 64];
            let err = ctx.decoder(&config).decode(&src, &mut dst).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidArchive);
            assert!(dst[8..].iter().all(|&b| b == 0xEE));
        }
    }

    #[test]
    fn a_two_filter_pipeline_round_trips() {
        use crate::codec::filter::Filter;
        use crate::{Array, ArrayParams};

        let nd = ndarray::Array2::from_shape_fn((16, 16), |(i, j)| (i * 16 + j) as i32);
        let params = ArrayParams::new()
            .block_shape(&[4, 4])
            .unwrap()
            .filters(&[Filter::ByteShuffle, Filter::BitShuffle])
            .unwrap();

        let array = Array::compact_ndarray_with(&nd, params).unwrap();
        assert_eq!(array.to_ndarray().unwrap(), nd);
    }
}
