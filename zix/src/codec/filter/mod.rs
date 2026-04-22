use crate::codec::filter::bit_shuffle::BitShuffleFilter;
use crate::codec::filter::byte_shuffle::ByteShuffleFilter;
use crate::codec::TmpBufferPool;
use crate::dtype::Dtype;

mod bit_shuffle;
mod byte_shuffle;

#[derive(Clone, Debug)]
pub enum Filter {
    ByteShuffle,
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
}
