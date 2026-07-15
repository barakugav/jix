use crate::codec::filter::bit_shuffle::BitShuffleFilter;
use crate::codec::filter::byte_shuffle::ByteShuffleFilter;
use crate::codec::TmpBufferPool;
use crate::dtype::Dtype;

mod bit_shuffle;
mod byte_shuffle;

/// A pre-compression byte transform applied to block data before encoding.
///
/// Filters rearrange the raw element bytes into a layout that compresses more efficiently,
/// then reverse the transform after decompression. They are applied in pipeline order during
/// encoding and reversed during decoding.
///
/// For most numeric workloads [`ByteShuffle`](Filter::ByteShuffle) is the right default.
/// It is fast and reliably improves compression ratios for uniform-dtype arrays by grouping
/// bytes of the same significance across consecutive elements (e.g. all the low bytes
/// together, then all the high bytes).
///
/// [`BitShuffle`](Filter::BitShuffle) applies the same idea at the bit level, which can
/// squeeze out more compression for data with low bit entropy, at the cost of higher CPU
/// usage.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Filter {
    /// Groups bytes by their position within each element across a block.
    ///
    /// For a block of `n` elements with itemsize `k`, byte `i` of element `j` is placed at
    /// position `i * n + j` in the output. This makes runs of similar values in the same
    /// byte position contiguous, which Zstd can exploit for much better compression ratios
    /// on numeric data.
    ByteShuffle,
    /// Groups bits by their position within each element across a block.
    ///
    /// Analogous to byte shuffle but operating at the bit level. Yields better compression
    /// ratios than byte shuffle for data with low bit entropy (e.g. arrays of small integers
    /// stored in a wider type, or arrays with many repeated values), at the cost of higher
    /// CPU usage for both encode and decode.
    BitShuffle,
}
impl Filter {
    pub(crate) fn encode(
        &self,
        src: &[u8],
        dst: &mut [u8],
        dtype: &Dtype,
        tmp_buffers: &TmpBufferPool,
    ) {
        match self {
            Filter::ByteShuffle => ByteShuffleFilter.encode(src, dst, dtype, tmp_buffers),
            Filter::BitShuffle => BitShuffleFilter::default().encode(src, dst, dtype, tmp_buffers),
        }
    }

    pub(crate) fn decode(
        &self,
        src: &[u8],
        dst: &mut [u8],
        dtype: &Dtype,
        tmp_buffers: &TmpBufferPool,
    ) {
        match self {
            Filter::ByteShuffle => ByteShuffleFilter.decode(src, dst, dtype, tmp_buffers),
            Filter::BitShuffle => BitShuffleFilter::default().decode(src, dst, dtype, tmp_buffers),
        }
    }
}

trait FilterImpl {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, tmp_buffers: &TmpBufferPool);
    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, tmp_buffers: &TmpBufferPool);
}

#[cfg(test)]
mod tests {
    use crate::codec::filter::FilterImpl;
    use crate::codec::TmpBufferPool;
    use crate::dtype::Dtyped;
    use crate::util::gen_data_bytes_from_slice;

    pub(crate) fn test_roundtrip<F, T>(items: &[T])
    where
        F: FilterImpl + Default,
        T: Dtyped,
    {
        let data = gen_data_bytes_from_slice::<T>(items);
        let src = data.as_slice();
        let dtype = T::DTYPE;
        let tmp_buffers = TmpBufferPool::new();
        let mut encoded = vec![0u8; src.len()];
        F::default().encode(src, &mut encoded, &dtype, &tmp_buffers);
        let mut decoded = vec![0u8; src.len()];
        F::default().decode(&encoded, &mut decoded, &dtype, &tmp_buffers);
        assert_eq!(decoded, src);
    }

    /// Shared proptest driver for filter tests.
    ///
    /// Generic over the dtype only, with the per-filter check passed as a fn pointer, to
    /// avoid per-filter monomorphization.
    #[inline(never)]
    pub(crate) fn run_bytes_proptest<T>(check: fn(&[T]))
    where
        T: crate::util::ScalarStrategy,
    {
        let strategy = proptest::collection::vec(
            <T as crate::util::ScalarStrategy>::any_strategy(),
            0..=1000usize,
        );
        let mut runner =
            proptest::test_runner::TestRunner::new(proptest::test_runner::Config::default());
        runner
            .run(&strategy, |data| {
                check(&data);
                Ok(())
            })
            .unwrap();
    }
}
