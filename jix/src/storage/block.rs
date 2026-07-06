use std::marker::PhantomData;
use std::sync::Arc;

use crate::codec::{Codec, Compressor, DecoderCodecConfig, Encoder, EncoderParams, ReadContext};
use crate::dtype::Dtype;
use crate::error::{ensure, Result};
use crate::util::{assert_unchecked_eq, SendSyncPtr};
use crate::ElementType;

const _: () = const {
    assert!(
        cfg!(target_endian = "little"),
        "Only little-endian is supported"
    );
};

/// Size of a block along one dimension.
pub type BlockSize = u32;

/// Packed locations of two blocks within the compressed data buffer.
///
/// A block's location is an `(offset, len)` pair. A naive `{ offset: u64, len: u32 }` struct would
/// be padded to 16 bytes (8-byte alignment), wasting the 4 bytes saved by storing `len` as a `u32`
/// rather than a `u64`. Block lengths always fit in `u32`, so two blocks are packed into one
/// 24-byte entry instead - 12 bytes per block, no padding. Block `2i` uses lane `[0]`, block
/// `2i + 1` uses lane `[1]`; an array of these holds `(nblocks + 1) >> 1` entries (when `nblocks`
/// is odd, the spare lane of the last entry is left zero and never read).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct BlockLocation2 {
    /// Byte offset into the data buffer where each block's compressed data begins.
    pub(crate) offset: [u64; 2],
    /// Byte length of each block's compressed data.
    pub(crate) len: [u32; 2],
}

/// Compressed 1D storage of typed items, divided into independently-encoded fixed-size blocks.
///
/// Items are stored as a flat sequence of `nitems` elements of type `dtype`. The sequence is
/// split into blocks of `block_size` items each; every block is compressed independently using
/// the codec pipeline described by `decoder_config`. All blocks are full - `nitems` must be an
/// exact multiple of `block_size`.
///
/// Internally the compressed bytes of all blocks live in a single continuous byte buffer
/// (`block_data`), but the blocks may be stored in any order within it. A parallel array of
/// [`BlockLocation2`] entries (`blocks_loc`) records where each block's data lives: block `i`
/// occupies `blocks_loc[i >> 1].len[i & 1]` bytes of `block_data` starting at
/// `blocks_loc[i >> 1].offset[i & 1]`. This enables O(1) random access to any block without
/// scanning the compressed data.
///
/// # Storage backends
///
/// The generic parameter `S: `[`BlockTableStorage`] determines how `block_data` and `blocks_loc`
/// are held in memory:
/// - [`Owned`] - heap-allocated; produced by [`OwnedBlockTableBuilder`] or [`Self::build_from_data`].
/// - [`Borrowed<'a>`] - borrowed slice; zero-copy view into an existing byte buffer.
/// - [`Mmap`] - memory-mapped file; data is read from disk on demand by the OS.
///
/// # Invariants
///
/// - `block_size > 0`
/// - `nitems % block_size == 0`
/// - `blocks_loc.len() == (nblocks + 1) >> 1`
/// - every block's location satisfies `offset + len <= block_data.len()`
pub(crate) struct BlockTable<S, ET>
where
    S: BlockTableStorage,
{
    pub(crate) block_data: S::Data<u8>,
    pub(crate) blocks_loc: S::Data<BlockLocation2>,

    pub(crate) nitems: u64,
    element_type: ET,

    /// The number of items in each block. All blocks are full (nitems is divisible by block_size).
    /// Note the units are items, not bytes.
    pub(crate) block_size: BlockSize,

    decoder_config: DecoderCodecConfig,
}
impl<S, ET> BlockTable<S, ET>
where
    S: BlockTableStorage,
{
    /// Construct a `BlockTable` from pre-encoded data, validating structural invariants.
    ///
    /// # Arguments
    ///
    /// - `block_data` - concatenated compressed bytes for all blocks.
    /// - `blocks_loc` - packed block locations, two blocks per [`BlockLocation2`] entry; must have
    ///   length `(nblocks + 1) >> 1`. Block `i`'s compressed bytes are the `len` bytes of
    ///   `block_data` starting at `offset`, where `(offset, len)` is lane `i & 1` of entry `i >> 1`.
    ///   Every block must satisfy `offset + len <= block_data.len()`; blocks need not be contiguous
    ///   or in order.
    /// - `nblocks` - number of blocks (the parity that `blocks_loc.len()` alone cannot recover).
    /// - `block_size` - number of items per block (must be `> 0`).
    /// - `decoder_config` - codec and dtype configuration used when decoding blocks.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if:
    /// - `block_size == 0`
    pub(crate) fn new(
        block_data: S::Data<u8>,
        blocks_loc: S::Data<BlockLocation2>,
        nblocks: u64,
        block_size: BlockSize,
        decoder_config: DecoderCodecConfig,
    ) -> Result<Self>
    where
        ET: ElementType,
    {
        ensure!(block_size > 0, InvalidArgument, "block_size must be > 0");
        let nitems = nblocks * block_size as u64;
        let data_len = block_data.as_ref().len() as u64;
        debug_assert_eq!(blocks_loc.as_ref().len() as u64, (nblocks + 1) >> 1);
        debug_assert!((0..nblocks).all(|i| {
            let loc = &blocks_loc.as_ref()[(i >> 1) as usize];
            let lane = (i & 1) as usize;
            let (offset, len) = (loc.offset[lane], loc.len[lane] as u64);
            offset <= data_len && len <= data_len - offset
        }));
        let element_type = ET::from_dtype(decoder_config.dtype.clone())?;
        Ok(Self {
            block_data,
            blocks_loc,
            nitems,
            element_type,
            block_size,
            decoder_config,
        })
    }

    /// Get the dtype of items in this storage.
    #[inline(always)]
    pub(crate) fn dtype(&self) -> &Dtype
    where
        ET: ElementType,
    {
        self.element_type.dtype()
    }

    /// Get the number of blocks in this storage.
    #[inline(always)]
    pub(crate) fn nblocks(&self) -> u64 {
        self.nitems / self.block_size as u64
    }

    /// Byte offset and length of block `block_idx`'s compressed data within `block_data`.
    #[inline(always)]
    pub(crate) fn block_location(&self, block_idx: u64) -> (u64, BlockSize) {
        let loc = &self.blocks_loc.as_ref()[(block_idx >> 1) as usize];
        let lane = (block_idx & 1) as usize;
        (loc.offset[lane], loc.len[lane])
    }

    /// Decompress one block into `buf`.
    ///
    /// # Arguments
    ///
    /// - `block_idx` - zero-based block index in `0..(nitems / block_size)`.
    ///   **Panics** if out of range.
    /// - `buf` - destination buffer. Must be exactly `block_size * dtype.itemsize()` bytes.
    /// - `context` - read context used for decoding.
    ///
    /// # Errors
    ///
    /// Returns `InvalidBufferSize` if `buf` has the wrong length.
    /// Propagates any codec error.
    #[inline]
    pub(crate) fn read_block(
        &self,
        block_idx: u64,
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()>
    where
        ET: ElementType,
    {
        let b_size_bytes = self.block_size as usize * self.dtype().itemsize() as usize;
        ensure!(
            buf.len() == b_size_bytes,
            InvalidBufferSize,
            "Buffer size does not match block size: got {}, expected {b_size_bytes}",
            buf.len()
        );

        let (offset, len) = self.block_location(block_idx);
        let block_data = &self.block_data.as_ref()[offset as usize..][..len as usize];

        let decoder = context.decoder(&self.decoder_config);
        let nbytes = decoder.decode(block_data, buf)?;
        debug_assert_eq!(nbytes, b_size_bytes);
        Ok(())
    }

    pub(crate) fn as_ref(&self) -> BlockTable<Borrowed<'_>, ET>
    where
        ET: ElementType,
    {
        BlockTable {
            block_data: self.block_data.as_ref(),
            blocks_loc: self.blocks_loc.as_ref(),
            nitems: self.nitems,
            element_type: self.element_type.clone(),
            block_size: self.block_size,
            decoder_config: self.decoder_config.clone(),
        }
    }

    #[inline(always)]
    pub(crate) fn decoder_config(&self) -> &DecoderCodecConfig
    where
        ET: ElementType,
    {
        unsafe { assert_unchecked_eq!(self.element_type.dtype(), &self.decoder_config.dtype) };
        &self.decoder_config
    }

    #[inline]
    pub(crate) fn element_type_change<NewET: ElementType>(self) -> Result<BlockTable<S, NewET>>
    where
        ET: ElementType,
    {
        let element_type = NewET::from_dtype(self.dtype().clone())?;
        Ok(BlockTable {
            block_data: self.block_data,
            blocks_loc: self.blocks_loc,
            nitems: self.nitems,
            element_type,
            block_size: self.block_size,
            decoder_config: self.decoder_config,
        })
    }
}

/// A sink that consumes a block table's compressed blocks one at a time and finalizes into some
/// output - either an in-memory [`BlockTable`] ([`OwnedBlockTableBuilder`]) or bytes streamed to an
/// archive ([`BlockArchiveWriter`](crate::archive::block::BlockArchiveWriter)).
pub(crate) trait BlockTableBuilder {
    /// What [`finalize`](Self::finalize) yields. For example the built [`BlockTable`] for an
    /// in-memory builder.
    type Output;

    /// Append one block's compressed bytes and record its `[offset, length]` pair at logical index
    /// `block_index`. Bytes are appended in call order, but `block_index` may be supplied in any
    /// order, so the data buffer's physical layout need not match the logical block order.
    fn write_compressed_block(&mut self, block_index: u64, compressed: &[u8]) -> Result<()>;

    /// Consume the builder, completing the block table and returning its [`Output`](Self::Output).
    fn finalize(self) -> Result<Self::Output>;
}

/// In-memory implementor of [`BlockTableBuilder`].
///
/// Accumulates compressed blocks into heap `Vec`s and produces a [`BlockTable<Owned>`].
pub(crate) struct OwnedBlockTableBuilder<ET> {
    block_data: Vec<u8>,
    blocks_loc: Vec<BlockLocation2>,
    block_size: BlockSize,
    decoder_config: DecoderCodecConfig,
    nblocks: u64,
    _marker: PhantomData<ET>,
}

impl<ET> OwnedBlockTableBuilder<ET>
where
    ET: ElementType,
{
    pub(crate) fn start(
        nblocks: u64,
        block_size: BlockSize,
        decoder_config: DecoderCodecConfig,
    ) -> Result<Self> {
        Ok(Self {
            block_data: Vec::new(),
            blocks_loc: vec![BlockLocation2::default(); ((nblocks + 1) >> 1) as usize],
            block_size,
            decoder_config,
            nblocks,
            _marker: PhantomData,
        })
    }
}

impl<ET> BlockTableBuilder for OwnedBlockTableBuilder<ET>
where
    ET: ElementType,
{
    type Output = BlockTable<Owned, ET>;

    /// Append one compressed block's bytes - in call order, contiguously - and record its
    /// `(offset, len)` location at `block_index`. The offset is the current end of the data buffer
    /// and the length is `compressed.len()`.
    fn write_compressed_block(&mut self, block_index: u64, compressed: &[u8]) -> Result<()> {
        let offset = self.block_data.len() as u64;
        self.block_data.extend_from_slice(compressed);
        let loc = &mut self.blocks_loc[(block_index >> 1) as usize];
        let lane = (block_index & 1) as usize;
        loc.offset[lane] = offset;
        loc.len[lane] = compressed.len() as u32;
        Ok(())
    }

    fn finalize(self) -> Result<BlockTable<Owned, ET>> {
        BlockTable::new(
            self.block_data,
            self.blocks_loc,
            self.nblocks,
            self.block_size,
            self.decoder_config,
        )
    }
}

impl<ET> BlockTable<Owned, ET> {
    /// Build a `BlockTable` by encoding raw item bytes in one shot.
    ///
    /// Splits `data` into chunks of `block_size * dtype.itemsize()` bytes, compresses each
    /// chunk with `encoder`, and returns the fully constructed table.
    /// This is the single-call alternative to the incremental [`OwnedBlockTableBuilder`].
    ///
    /// # Arguments
    ///
    /// - `data` - contiguous raw (uncompressed) item bytes; length must equal
    ///   `nitems * dtype.itemsize()`.
    /// - `dtype` - element type of the stored items.
    /// - `block_size` - number of items per block (must be `> 0`).
    /// - `encoder` - codec pipeline (filters + compressor) applied to each block. TODO
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if:
    /// - `dtype.itemsize() == 0`
    /// - `block_size == 0`
    /// - `data.len()` is not a multiple of `dtype.itemsize()`
    /// - The resulting `nitems` is not a multiple of `block_size`
    #[allow(unused)]
    pub(crate) fn build_from_data(
        data: &[u8],
        dtype: Dtype,
        block_size: BlockSize,
        encoder_params: &EncoderParams,
    ) -> Result<Self>
    where
        ET: ElementType,
    {
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

        ensure!(
            dtype.itemsize() > 0,
            InvalidArgument,
            "itemsize must be > 0"
        );
        ensure!(block_size > 0, InvalidArgument, "block_size must be > 0");
        let b_size_bytes = block_size as usize * dtype.itemsize() as usize;
        let mut encoder = Encoder::new(encoder_params, dtype.clone())?;
        let max_blk_cdata_len = encoder.encode_bound(b_size_bytes);

        let nblocks = nitems / block_size as usize;
        let mut block_data = Vec::<u8>::new();
        let mut blocks_loc = vec![BlockLocation2::default(); (nblocks + 1) >> 1];
        for (block_idx, plain_data) in data.chunks(b_size_bytes).enumerate() {
            let b_size_bytes = block_size as usize * dtype.itemsize() as usize;
            ensure!(
                plain_data.len() == b_size_bytes,
                InvalidArgument,
                "Block data size does not match block size: got {}, expected {b_size_bytes}",
                plain_data.len()
            );
            let block_data_len = block_data.len();
            block_data.reserve(max_blk_cdata_len);
            #[allow(clippy::uninit_vec)]
            unsafe {
                block_data.set_len(block_data_len + max_blk_cdata_len)
            };
            let blk_buf = &mut block_data[block_data_len..];
            let blk_cdata_len = encoder.encode(plain_data, blk_buf)?;
            debug_assert!(blk_cdata_len <= max_blk_cdata_len);
            unsafe { block_data.set_len(block_data_len + blk_cdata_len) };
            let loc = &mut blocks_loc[block_idx >> 1];
            loc.offset[block_idx & 1] = block_data_len as u64;
            loc.len[block_idx & 1] = blk_cdata_len as u32;
        }

        let decoder_config = DecoderCodecConfig {
            codec: match &encoder.compressor {
                Compressor::Zstd(_) => Codec::Zstd,
            },
            filters: encoder.filters.clone_slow(),
            dtype: dtype.clone(),
        };
        BlockTable::new(
            block_data,
            blocks_loc,
            nblocks as u64,
            block_size,
            decoder_config,
        )
    }
}

/// Abstraction over the backing storage of a [`BlockTable`]'s byte arrays.
///
/// The associated type `Data<T>` determines how a typed array is held in memory.
/// Three implementations are provided:
/// - [`Owned`] - heap-allocated `Vec<T>`; owns its data.
/// - [`Borrowed<'a>`] - a borrowed slice `&'a [T]`; zero-copy view into existing memory.
/// - [`Mmap`] - memory-mapped file via [`MmapData<T>`]; the `Arc<Mmap>` keeps the mapping
///   alive for as long as any `BlockTable<Mmap>` referencing it exists.
pub(crate) trait BlockTableStorage {
    type Data<T: 'static>: AsRef<[T]>;
}

pub(crate) struct Owned(pub(crate) PhantomData<()>);
pub(crate) struct Borrowed<'a>(pub(crate) PhantomData<&'a ()>);
pub(crate) struct Mmap {
    pub(crate) mmap: Arc<memmap2::Mmap>,
    pub(crate) base_offset: u64,
}
impl BlockTableStorage for Owned {
    type Data<T: 'static> = Vec<T>;
}
impl<'a> BlockTableStorage for Borrowed<'a> {
    type Data<T: 'static> = &'a [T];
}
/// The `BlockTableStorage::Data<T>` type for memory-mapped storage.
///
/// Pairs an `Arc<Mmap>` - which keeps the memory mapping alive - with a raw pointer and
/// length describing the typed slice within it. The pointer is derived directly from the
/// mapped region, so no allocation or copy takes place when reading.
pub(crate) struct MmapData<T: 'static> {
    #[allow(unused)]
    pub(crate) mmap: Arc<memmap2::Mmap>,
    pub(crate) data: (SendSyncPtr<T>, usize),
}
impl<T: 'static> AsRef<[T]> for MmapData<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.0.as_ptr(), self.data.1) }
    }
}
impl BlockTableStorage for Mmap {
    type Data<T: 'static> = MmapData<T>;
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{BlockSize, BlockTable};
    use crate::codec::{EncoderParams, ReadContext};
    use crate::dtype::Dtyped;
    use crate::error::Result;
    use crate::storage::block::{BlockTableStorage, Owned};
    use crate::storage::{ElementType, Ty};
    use crate::util::{cast_slice, AlignedBytes};

    fn decode_block<S, ET>(
        table: &BlockTable<S, ET>,
        idx: usize,
        context: &mut ReadContext,
    ) -> Vec<u8>
    where
        S: BlockTableStorage,
        ET: ElementType,
    {
        let block_bytes = table.block_size as usize * table.dtype().itemsize() as usize;
        let mut buf = vec![0u8; block_bytes];
        table.read_block(idx as u64, &mut buf, context).unwrap();
        buf
    }

    fn build_from_items<T>(
        items: &[T],
        block_size: BlockSize,
        encoder_params: &EncoderParams,
    ) -> Result<BlockTable<Owned, Ty<T>>>
    where
        T: Dtyped,
    {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::DTYPE,
            block_size,
            &encoder_params,
        )
    }

    #[test]
    fn build_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let table = build_from_items(&items, 8, &EncoderParams::default()).unwrap();
        assert_eq!(table.nblocks(), 1);
        assert_eq!(table.nitems, 8);
        let mut context = ReadContext::default();
        assert_eq!(decode_block(&table, 0, &mut context), items);
    }

    #[test]
    fn build_multiple_blocks_exact_divisor() {
        // 12 items, block_size=4 -> 3 full blocks
        let items: Vec<u8> = (0u8..12).collect();
        let table = build_from_items(&items, 4, &EncoderParams::default()).unwrap();
        assert_eq!(table.nblocks(), 3);
        assert_eq!(table.nitems, 12);
        let mut context = ReadContext::default();
        assert_eq!(decode_block(&table, 0, &mut context), items[0..4]);
        assert_eq!(decode_block(&table, 1, &mut context), items[4..8]);
        assert_eq!(decode_block(&table, 2, &mut context), items[8..12]);
    }

    #[test]
    fn build_multiple_blocks_non_divisible_panics() {
        // 10 items, block_size=4 -> not divisible, should panic
        let items: Vec<u8> = (0u8..10).collect();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_from_items(&items, 4, &EncoderParams::default()).unwrap();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn build_with_itemsize_greater_than_one() {
        // 4 u32 values, block_size=2
        let items: Vec<u32> = vec![10, 20, 30, 40];
        let table = build_from_items(&items, 2, &EncoderParams::default()).unwrap();
        assert_eq!(table.nblocks(), 2);
        assert_eq!(table.nitems, 4);
        let mut context = ReadContext::default();
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

    fn round_trip<T: Dtyped>(items: &[T], block_size: BlockSize) -> BlockTable<Owned, Ty<T>> {
        let table = build_from_items(items, block_size, &EncoderParams::default()).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        table.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let len = bytes.len() as u64;
        BlockTable::read_from(Cursor::new(bytes), len)
            .unwrap()
            .element_type_change::<Ty<T>>()
            .unwrap()
    }

    #[test]
    fn round_trip_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let table2 = round_trip(&items, 8);
        assert_eq!(table2.nblocks(), 1);
        assert_eq!(table2.nitems, 8);
        assert_eq!(table2.block_size, 8);
        assert_eq!(*table2.dtype(), u8::DTYPE);
        let mut context = ReadContext::default();
        assert_eq!(decode_block(&table2, 0, &mut context), items);
    }

    #[test]
    fn round_trip_multiple_blocks() {
        let items: Vec<u8> = (0u8..12).collect();
        let table = round_trip(&items, 4);
        assert_eq!(table.nblocks(), 3);
        assert_eq!(table.nitems, 12);
        let mut context = ReadContext::default();
        let recovered: Vec<u8> = (0..table.nblocks() as usize)
            .flat_map(|i| decode_block(&table, i, &mut context))
            .collect();
        assert_eq!(recovered, items);
    }

    #[test]
    fn round_trip_preserves_blocks_loc_ordering() {
        let items: Vec<u8> = (0u8..12).collect();
        let table2 = round_trip(&items, 3);
        // Each block has a non-empty range, and blocks are stored back-to-back (no gaps).
        let mut expected_offset = 0u64;
        for i in 0..table2.nblocks() {
            let (offset, len) = table2.block_location(i);
            assert_eq!(offset, expected_offset);
            assert!(len > 0);
            expected_offset += len as u64;
        }
    }

    // -----------------------------------------------------------------------
    // write_to_file / read_from_file round-trip
    // -----------------------------------------------------------------------

    #[cfg(not(miri))]
    #[test]
    fn round_trip_file() {
        let items: Vec<u32> = (0u32..18).collect();
        let table = build_from_items(&items, 3, &EncoderParams::default()).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path();
        table
            .write_to(&mut std::fs::File::create(path).unwrap())
            .unwrap();

        let file = std::fs::File::open(path).unwrap();
        let reader_len = file.metadata().unwrap().len();
        let table2 = BlockTable::read_from(file, reader_len).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(table2.nblocks(), 6);
        assert_eq!(table2.nitems, 18);
        let mut context = ReadContext::default();
        let recovered: Vec<u8> = (0..table2.nblocks() as usize)
            .flat_map(|i| decode_block(&table2, i, &mut context))
            .collect();
        assert_eq!(recovered, unsafe { cast_slice::<u32, u8>(&items) });
    }

    fn make_storage<T: Dtyped>(
        items: &[T],
        block_size: BlockSize,
        encoder_params: &EncoderParams,
    ) -> BlockTable<Owned, Ty<T>> {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::DTYPE,
            block_size,
            &encoder_params,
        )
        .unwrap()
    }

    fn read_block_items<T, S>(storage: &BlockTable<S, Ty<T>>, idx: usize) -> Vec<T>
    where
        T: Dtyped,
        S: BlockTableStorage,
    {
        let mut context = ReadContext::default();
        let block_bytes = storage.block_size as usize * storage.dtype().itemsize() as usize;
        let mut buf =
            AlignedBytes::with_capacity_exact(T::DTYPE.alignment().as_usize(), block_bytes);
        unsafe { buf.set_len(block_bytes) };
        storage
            .read_block(idx as u64, &mut buf, &mut context)
            .unwrap();
        unsafe { cast_slice::<u8, T>(&buf) }.to_vec()
    }

    #[test]
    fn single_block_u8_round_trips() {
        let items: Vec<u8> = (0..8).collect();
        let s = make_storage(&items, 8, &EncoderParams::default());
        assert_eq!(s.nitems, 8);
        assert_eq!(s.block_size, 8);
        assert_eq!(s.dtype(), &u8::DTYPE);
        assert_eq!(read_block_items::<u8, _>(&s, 0), items);
    }

    #[test]
    fn two_blocks_i32_round_trips() {
        let items: Vec<i32> = (0..8).collect();
        let s = make_storage(&items, 4, &EncoderParams::default());
        assert_eq!(s.nitems, 8);
        assert_eq!(s.block_size, 4);
        assert_eq!(read_block_items::<i32, _>(&s, 0), items[..4]);
        assert_eq!(read_block_items::<i32, _>(&s, 1), items[4..]);
    }

    #[test]
    fn multiple_blocks_f32_round_trips() {
        let items: Vec<f32> = (0..12).map(|x| x as f32 * 0.5).collect();
        let s = make_storage(&items, 4, &EncoderParams::default());
        assert_eq!(s.nitems, 12);
        assert_eq!(s.block_size, 4);
        for b in 0..3 {
            assert_eq!(read_block_items::<f32, _>(&s, b), items[b * 4..(b + 1) * 4]);
        }
    }

    #[test]
    fn buffer_too_small_returns_error() {
        let items: Vec<u8> = (0..4).collect();
        let s = make_storage(&items, 4, &EncoderParams::default());
        let mut buf = vec![0u8; 3]; // one byte short
        let mut context = ReadContext::default();
        assert!(s.read_block(0, &mut buf, &mut context).is_err());
    }
}
