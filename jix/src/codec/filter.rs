use crate::buf_pool::BufferPool;
use crate::codec::bit_shuffle::BitShuffleFilter;
use crate::codec::byte_shuffle::ByteShuffleFilter;
use crate::dtype::Dtype;

/// A byte transform applied to block data before compression and reversed after decompression.
///
/// Filters rearrange the raw element bytes into a layout that compresses better. A pipeline of
/// filters is applied in order when encoding and in reverse order when decoding.
///
/// For most numeric workloads [`ByteShuffle`](Filter::ByteShuffle) is the right default: it is
/// fast and reliably improves the ratio by grouping bytes of the same significance across
/// elements (e.g. all the low bytes, then all the high bytes). [`BitShuffle`](Filter::BitShuffle)
/// does the same per bit, which can compress low-entropy data better at a higher CPU cost.
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
    /// Compresses better than byte shuffle on data with low bit entropy (e.g. small integers
    /// stored in a wider type, or many repeated values), at a higher CPU cost to both encode
    /// and decode.
    BitShuffle,
}
impl Filter {
    pub(crate) fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, buf_pool: &BufferPool) {
        match self {
            Filter::ByteShuffle => ByteShuffleFilter.encode(src, dst, dtype, buf_pool),
            Filter::BitShuffle => BitShuffleFilter.encode(src, dst, dtype, buf_pool),
        }
    }

    pub(crate) fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, buf_pool: &BufferPool) {
        match self {
            Filter::ByteShuffle => ByteShuffleFilter.decode(src, dst, dtype, buf_pool),
            Filter::BitShuffle => BitShuffleFilter.decode(src, dst, dtype, buf_pool),
        }
    }
}

pub(super) trait FilterImpl {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, buf_pool: &BufferPool);
    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, buf_pool: &BufferPool);
}

#[cfg(test)]
pub(super) mod tests {
    use crate::buf_pool::BufferPool;
    use crate::codec::filter::FilterImpl;
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
        let buf_pool = BufferPool::new();
        let mut encoded = vec![0u8; src.len()];
        F::default().encode(src, &mut encoded, &dtype, &buf_pool);
        let mut decoded = vec![0u8; src.len()];
        F::default().decode(&encoded, &mut decoded, &dtype, &buf_pool);
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
