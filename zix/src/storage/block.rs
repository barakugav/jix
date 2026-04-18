use crate::codec::{Codec, Compressor, DecoderCodecConfig, Encoder, ReadContext};
use crate::dtype::Dtype;
use crate::error::{Result, ensure};

const _: () = const {
    assert!(
        cfg!(target_endian = "little"),
        "Only little-endian is supported"
    );
};

pub(crate) type BlockSize = u32;

/// Storage of 1D array items, organized in blocks.
///
/// The number of items must be divisible by the block length, there is no support for partial blocks.
/// At all times the storage holds the invariants:
/// - `block_size > 0`
/// - `nitems % block_size == 0`
pub(crate) struct BlockTable<S> {
    pub(crate) storage: S,
    pub(crate) nitems: u64,

    /// The number of items in each block. All blocks are full (nitems is divisible by block_size).
    /// Note the units are items, not bytes.
    pub(crate) block_size: BlockSize,

    pub(crate) decoder_config: DecoderCodecConfig,
}
impl<S> BlockTable<S> {
    pub(crate) fn new(
        storage: S,
        nitems: u64,
        block_size: BlockSize,
        decoder_config: DecoderCodecConfig,
    ) -> Result<Self>
    where
        S: BlockTableStorage,
    {
        ensure!(block_size > 0, InvalidArgument, "block_size must be > 0");
        ensure!(
            nitems.is_multiple_of(block_size as u64),
            InvalidArgument,
            "nitems must be a multiple of block_size"
        );
        let nblocks = nitems / block_size as u64;
        let cdata = storage.cdata();
        let block_offsets = storage.block_offsets();
        ensure!(
            block_offsets.len() as u64 == if nblocks == 0 { 0 } else { nblocks + 1 },
            InvalidArgument,
            "block_offsets length mismatch"
        );
        if nblocks > 0 {
            debug_assert!(block_offsets.windows(2).all(|w| w[0] < w[1]));
            debug_assert!(*block_offsets.last().unwrap() <= cdata.len() as u64);
        }
        Ok(Self {
            storage,
            nitems,
            block_size,
            decoder_config,
        })
    }

    /// Get the dtype of items in this storage.
    pub(crate) fn dtype(&self) -> &Dtype {
        &self.decoder_config.dtype
    }

    /// Get the total number of items in this storage.
    pub(crate) fn nitems(&self) -> u64 {
        self.nitems
    }

    /// Get the length of a block in this storage.
    ///
    /// Note that the units are in items, not bytes.
    pub(crate) fn block_len(&self) -> BlockSize {
        self.block_size
    }

    /// Read a block of items into the provided buffer.
    ///
    /// # Arguments
    ///
    /// - `block_idx`: The index of the block to read, in the range `0..(nitems / block_len)`.
    /// - `buf`: The buffer to read the block into. Must be of size `block_len * dtype.itemsize()`.
    /// - `context`: a read context containing global configuration and reuseable buffers.
    pub(crate) fn read_block(
        &self,
        block_idx: u64,
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()>
    where
        S: BlockTableStorage,
    {
        let b_size_bytes = self.block_len() as usize * self.dtype().itemsize() as usize;
        ensure!(
            buf.len() == b_size_bytes,
            InvalidBufferSize,
            "Buffer size does not match block size: got {}, expected {b_size_bytes}",
            buf.len()
        );

        let block_offsets = self.storage.block_offsets();
        let begin = block_offsets[block_idx as usize] as usize;
        let end = block_offsets[block_idx as usize + 1] as usize;
        let b_cdata = &self.storage.cdata()[begin..end];

        let decoder = context.decoder(&self.decoder_config);
        let nbytes = decoder.decode(b_cdata, buf)?;
        debug_assert_eq!(nbytes, b_size_bytes);
        Ok(())
    }
}

impl BlockTable<Owned> {
    #[allow(unused)]
    pub(crate) fn build_from_data(
        data: &[u8],
        dtype: Dtype,
        block_size: BlockSize,
        encoder: Encoder,
    ) -> Result<Self> {
        let itemsize = dtype.itemsize();
        ensure!(itemsize > 0, InvalidArgument, "itemsize must be > 0");
        ensure!(block_size > 0, InvalidArgument, "block_size must be > 0");
        ensure!(
            data.len().is_multiple_of(itemsize as usize),
            InvalidArgument,
            "data length must be a multiple of itemsize"
        );
        let nitems = data.len() / itemsize as usize;
        ensure!(
            nitems.is_multiple_of(block_size as usize),
            InvalidArgument,
            "nitems must be a multiple of block_size"
        );

        let b_size_bytes = block_size as usize * itemsize as usize;
        let mut builder = BlockTableBuilder::new(dtype, block_size, encoder)?;
        for b_data in data.chunks(b_size_bytes) {
            builder.add_block(b_data)?;
        }
        builder.finish()
    }
}

#[doc(hidden)]
pub trait BlockTableStorage {
    fn cdata(&self) -> &[u8];
    fn block_offsets(&self) -> &[u64];
}
#[doc(hidden)]
pub struct Owned {
    pub(crate) cdata: Vec<u8>,
    pub(crate) block_offsets: Vec<u64>,
}
#[doc(hidden)]
pub struct Borrowed<'a> {
    pub(crate) cdata: &'a [u8],
    pub(crate) block_offsets: &'a [u64],
}
#[doc(hidden)]
pub struct Mmap {
    pub(crate) cdata: &'static [u8],
    pub(crate) block_offsets: &'static [u64],
    #[allow(unused)] // keep the mmap alive
    pub(crate) mmap: memmap2::Mmap,
}
impl BlockTableStorage for Owned {
    fn cdata(&self) -> &[u8] {
        &self.cdata
    }
    fn block_offsets(&self) -> &[u64] {
        &self.block_offsets
    }
}
impl<'a> BlockTableStorage for Borrowed<'a> {
    fn cdata(&self) -> &[u8] {
        self.cdata
    }
    fn block_offsets(&self) -> &[u64] {
        self.block_offsets
    }
}
impl BlockTableStorage for Mmap {
    fn cdata(&self) -> &[u8] {
        self.cdata
    }
    fn block_offsets(&self) -> &[u64] {
        self.block_offsets
    }
}

pub(crate) struct BlockTableBuilder {
    dtype: Dtype,
    block_size: BlockSize,
    encoder: Encoder,
    cdata: Vec<u8>,
    block_offsets: Vec<u64>,
    max_blk_cdata_len: usize,
}
impl BlockTableBuilder {
    pub(crate) fn new(dtype: Dtype, block_size: BlockSize, encoder: Encoder) -> Result<Self> {
        ensure!(
            dtype.itemsize() > 0,
            InvalidArgument,
            "itemsize must be > 0"
        );
        ensure!(block_size > 0, InvalidArgument, "block_size must be > 0");
        let b_size_bytes = block_size as usize * dtype.itemsize() as usize;
        let max_blk_cdata_len = encoder.encode_bound(b_size_bytes);
        Ok(Self {
            dtype,
            block_size,
            encoder,
            cdata: Vec::new(),
            block_offsets: Vec::new(),
            max_blk_cdata_len,
        })
    }

    pub(crate) fn add_block(&mut self, block_data: &[u8]) -> Result<()> {
        let b_size_bytes = self.block_size as usize * self.dtype.itemsize() as usize;
        ensure!(
            block_data.len() == b_size_bytes,
            InvalidArgument,
            "Block data size does not match block size: got {}, expected {b_size_bytes}",
            block_data.len()
        );

        let cdata_len = self.cdata.len();
        self.cdata.reserve(self.max_blk_cdata_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.cdata.set_len(cdata_len + self.max_blk_cdata_len)
        };
        let blk_buf = &mut self.cdata[cdata_len..];

        let blk_cdata_len = self.encoder.encode(block_data, blk_buf)?;
        debug_assert!(blk_cdata_len <= self.max_blk_cdata_len);
        unsafe { self.cdata.set_len(cdata_len + blk_cdata_len) };

        if self.block_offsets.is_empty() {
            self.block_offsets.push(0);
        }
        self.block_offsets.push(self.cdata.len() as u64);
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<BlockTable<Owned>> {
        let nblocks = self.block_offsets.len().saturating_sub(1);
        let nitems = nblocks * self.block_size as usize;
        let decoder_config = DecoderCodecConfig {
            codec: match &self.encoder.compressor {
                Compressor::Zstd(_) => Codec::Zstd,
            },
            filters: self.encoder.filters.clone(),
            dtype: self.dtype.clone(),
        };
        BlockTable::new(
            Owned {
                cdata: self.cdata,
                block_offsets: self.block_offsets,
            },
            nitems as u64,
            self.block_size,
            decoder_config,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{BlockSize, BlockTable};
    use crate::codec::{DecoderParams, Encoder, EncoderParams, ReadContext};
    use crate::dtype::{Dtype, Dtyped};
    use crate::error::Result;
    use crate::storage::block::{BlockTableStorage, Owned};
    use crate::util::{AlignedBytes, cast_slice};

    fn make_encoder(dtype: Dtype, params: &EncoderParams) -> Encoder {
        Encoder::new(params, dtype).unwrap()
    }

    fn decode_block<S>(table: &BlockTable<S>, idx: usize, context: &mut ReadContext) -> Vec<u8>
    where
        S: BlockTableStorage,
    {
        let block_bytes = table.block_len() as usize * table.dtype().itemsize() as usize;
        let mut buf = vec![0u8; block_bytes];
        table.read_block(idx as u64, &mut buf, context).unwrap();
        buf
    }

    fn build_from_items<T>(
        items: &[T],
        block_size: BlockSize,
        encoder: Encoder,
    ) -> Result<BlockTable<Owned>>
    where
        T: Dtyped,
    {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::DTYPE,
            block_size,
            encoder,
        )
    }

    #[test]
    fn build_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let encoder_params = EncoderParams::default();
        let encoder = make_encoder(u8::DTYPE, &encoder_params);
        let table = build_from_items(&items, 8, encoder).unwrap();
        assert_eq!(table.storage.block_offsets.len(), 2);
        assert_eq!(table.nitems, 8);
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        assert_eq!(decode_block(&table, 0, &mut context), items);
    }

    #[test]
    fn build_multiple_blocks_exact_divisor() {
        // 12 items, block_size=4 → 3 full blocks
        let items: Vec<u8> = (0u8..12).collect();
        let encoder_params = EncoderParams::default();
        let encoder = make_encoder(u8::DTYPE, &encoder_params);
        let table = build_from_items(&items, 4, encoder).unwrap();
        assert_eq!(table.storage.block_offsets.len(), 4);
        assert_eq!(table.nitems, 12);
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        assert_eq!(decode_block(&table, 0, &mut context), items[0..4]);
        assert_eq!(decode_block(&table, 1, &mut context), items[4..8]);
        assert_eq!(decode_block(&table, 2, &mut context), items[8..12]);
    }

    #[test]
    fn build_multiple_blocks_non_divisible_panics() {
        // 10 items, block_size=4 → not divisible, should panic
        let items: Vec<u8> = (0u8..10).collect();
        let encoder_params = EncoderParams::default();
        let encoder = make_encoder(u8::DTYPE, &encoder_params);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_from_items(&items, 4, encoder).unwrap();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn build_with_itemsize_greater_than_one() {
        // 4 u32 values, block_size=2
        let items: Vec<u32> = vec![10, 20, 30, 40];
        let encoder_params = EncoderParams::default();
        let encoder = make_encoder(u32::DTYPE, &encoder_params);
        let table = build_from_items(&items, 2, encoder).unwrap();
        assert_eq!(table.storage.block_offsets.len(), 3);
        assert_eq!(table.nitems, 4);
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        assert_eq!(decode_block(&table, 0, &mut context), unsafe {
            cast_slice::<u32, u8>(&items[0..2])
        });
        assert_eq!(decode_block(&table, 1, &mut context), unsafe {
            cast_slice::<u32, u8>(&items[2..4])
        });
    }

    // -----------------------------------------------------------------------
    // write_to / read_from round-trip
    // -----------------------------------------------------------------------

    fn round_trip<T: Dtyped>(items: &[T], block_size: BlockSize) -> BlockTable<Owned> {
        let encoder_params = EncoderParams::default();
        let encoder = make_encoder(T::DTYPE, &encoder_params);
        let table = build_from_items(items, block_size, encoder).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        table.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let len = bytes.len() as u64;
        BlockTable::read_from(Cursor::new(bytes), len).unwrap()
    }

    #[test]
    fn round_trip_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let table2 = round_trip(&items, 8);
        assert_eq!(table2.storage.block_offsets.len(), 2);
        assert_eq!(table2.nitems, 8);
        assert_eq!(table2.block_size, 8);
        assert_eq!(*table2.dtype(), u8::DTYPE);
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        assert_eq!(decode_block(&table2, 0, &mut context), items);
    }

    #[test]
    fn round_trip_multiple_blocks() {
        let items: Vec<u8> = (0u8..12).collect();
        let table = round_trip(&items, 4);
        assert_eq!(table.storage.block_offsets.len(), 4);
        assert_eq!(table.nitems, 12);
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        let recovered: Vec<u8> = (0..table.storage.block_offsets.len() - 1)
            .flat_map(|i| decode_block(&table, i, &mut context))
            .collect();
        assert_eq!(recovered, items);
    }

    #[test]
    fn round_trip_preserves_block_offsets_ordering() {
        let items: Vec<u8> = (0u8..12).collect();
        let table2 = round_trip(&items, 3);
        let offs = table2.storage.block_offsets();
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
    }

    // -----------------------------------------------------------------------
    // write_to_file / read_from_file round-trip
    // -----------------------------------------------------------------------

    #[cfg(not(miri))]
    #[test]
    fn round_trip_file() {
        let items: Vec<u32> = (0u32..18).collect();
        let encoder_params = EncoderParams::default();
        let encoder = make_encoder(u32::DTYPE, &encoder_params);
        let table = build_from_items(&items, 3, encoder).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path();
        table
            .write_to(&mut std::fs::File::create(path).unwrap())
            .unwrap();

        let file = std::fs::File::open(path).unwrap();
        let reader_len = file.metadata().unwrap().len();
        let table2 = BlockTable::read_from(file, reader_len).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(table2.storage.block_offsets.len(), 7);
        assert_eq!(table2.nitems, 18);
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        let recovered: Vec<u8> = (0..table2.storage.block_offsets.len() - 1)
            .flat_map(|i| decode_block(&table2, i, &mut context))
            .collect();
        assert_eq!(recovered, unsafe { cast_slice::<u32, u8>(&items) });
    }

    fn make_storage<T: Dtyped>(
        items: &[T],
        block_len: BlockSize,
        params: &EncoderParams,
    ) -> BlockTable<Owned> {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::DTYPE,
            block_len,
            make_encoder(T::DTYPE, params),
        )
        .unwrap()
    }

    fn read_block_items<T, S>(storage: &BlockTable<S>, idx: usize) -> Vec<T>
    where
        T: Dtyped,
        S: BlockTableStorage,
    {
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        let block_bytes = storage.block_len() as usize * storage.dtype().itemsize() as usize;
        let mut buf = AlignedBytes::with_capacity(T::DTYPE.alignment() as usize, block_bytes);
        unsafe { buf.set_len(block_bytes) };
        storage
            .read_block(idx as u64, &mut buf, &mut context)
            .unwrap();
        unsafe { cast_slice::<u8, T>(&buf) }.to_vec()
    }

    #[test]
    fn single_block_u8_round_trips() {
        let items: Vec<u8> = (0..8).collect();
        let encoder_params = EncoderParams::default();
        let s = make_storage(&items, 8, &encoder_params);
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 8);
        assert_eq!(s.dtype(), &u8::DTYPE);
        assert_eq!(read_block_items::<u8, _>(&s, 0), items);
    }

    #[test]
    fn two_blocks_i32_round_trips() {
        let items: Vec<i32> = (0..8).collect();
        let encoder_params = EncoderParams::default();
        let s = make_storage(&items, 4, &encoder_params);
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 4);
        assert_eq!(read_block_items::<i32, _>(&s, 0), items[..4]);
        assert_eq!(read_block_items::<i32, _>(&s, 1), items[4..]);
    }

    #[test]
    fn multiple_blocks_f32_round_trips() {
        let items: Vec<f32> = (0..12).map(|x| x as f32 * 0.5).collect();
        let encoder_params = EncoderParams::default();
        let s = make_storage(&items, 4, &encoder_params);
        assert_eq!(s.nitems(), 12);
        assert_eq!(s.block_len(), 4);
        for b in 0..3 {
            assert_eq!(read_block_items::<f32, _>(&s, b), items[b * 4..(b + 1) * 4]);
        }
    }

    #[test]
    fn buffer_too_small_returns_error() {
        let items: Vec<u8> = (0..4).collect();
        let encoder_params = EncoderParams::default();
        let s = make_storage(&items, 4, &encoder_params);
        let mut buf = vec![0u8; 3]; // one byte short
        let mut context = ReadContext::new(&DecoderParams::default()).unwrap();
        assert!(s.read_block(0, &mut buf, &mut context).is_err());
    }
}
