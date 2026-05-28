use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use crate::codec::{Codec, Compressor, DecoderCodecConfig, Encoder, EncoderParams, ReadContext};
use crate::dtype::Dtype;
use crate::error::{ensure, Result};
use crate::storage::ElementType;
use crate::util::{assert_unchecked_eq, SendSyncPtr};

const _: () = const {
    assert!(
        cfg!(target_endian = "little"),
        "Only little-endian is supported"
    );
};

pub type BlockSize = u32;

/// Compressed 1D storage of typed items, divided into independently-encoded fixed-size blocks.
///
/// Items are stored as a flat sequence of `nitems` elements of type `dtype`. The sequence is
/// split into blocks of `block_size` items each; every block is compressed independently using
/// the codec pipeline described by `decoder_config`. All blocks are full - `nitems` must be an
/// exact multiple of `block_size`.
///
/// Internally the compressed bytes of all blocks are concatenated into a single byte buffer
/// (`block_data`). A parallel array of `nblocks + 1` byte offsets (`block_offsets`) records where
/// each block's data begins and ends, enabling O(1) random access to any block without
/// scanning the compressed data.
///
/// # Storage backends
///
/// The generic parameter `S: `[`BlockTableStorage`] determines how `block_data` and `block_offsets`
/// are held in memory:
/// - [`Owned`] - heap-allocated; produced by [`BlockTableBuilder`] or [`Self::build_from_data`].
/// - [`Borrowed<'a>`] - borrowed slice; zero-copy view into an existing byte buffer.
/// - [`Mmap`] - memory-mapped file; data is read from disk on demand by the OS.
///
/// # Invariants
///
/// - `block_size > 0`
/// - `nitems % block_size == 0`
/// - `block_offsets.len() == nblocks + 1` when `nblocks > 0`, or `0` when `nblocks == 0`
/// - `block_offsets` is strictly increasing; the last entry equals `block_data.len()`
pub(crate) struct BlockTable<S, ET>
where
    S: BlockTableStorage,
{
    pub(crate) block_data: S::Data<u8>,
    pub(crate) block_offsets: S::Data<u64>,

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
    /// - `block_offsets` - byte positions into `block_data` that delimit each block.
    ///   Block `i`'s compressed bytes are `block_data[block_offsets[i]..block_offsets[i+1]]`.
    ///   Must have length `nblocks + 1` when `nblocks > 0`, or length `0` when `nblocks == 0`.
    ///   Entries must be strictly increasing and the last entry must not exceed `block_data.len()`.
    /// - `block_size` - number of items per block (must be `> 0`).
    /// - `decoder_config` - codec and dtype configuration used when decoding blocks.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if:
    /// - `block_size == 0`
    pub(crate) fn new(
        block_data: S::Data<u8>,
        block_offsets: S::Data<u64>,
        block_size: BlockSize,
        decoder_config: DecoderCodecConfig,
    ) -> Result<Self>
    where
        ET: ElementType,
    {
        ensure!(block_size > 0, InvalidArgument, "block_size must be > 0");
        let nblocks = block_offsets.as_ref().len().saturating_sub(1);
        let nitems = nblocks as u64 * block_size as u64;
        if nblocks > 0 {
            debug_assert!(block_offsets.as_ref().windows(2).all(|w| w[0] < w[1]));
            debug_assert!(
                *block_offsets.as_ref().last().unwrap() <= block_data.as_ref().len() as u64
            );
        }
        let element_type = ET::from_dtype(decoder_config.dtype.clone())?;
        Ok(Self {
            block_data,
            block_offsets,
            nitems,
            element_type,
            block_size,
            decoder_config,
        })
    }

    /// Get the dtype of items in this storage.
    pub(crate) fn dtype(&self) -> &Dtype
    where
        ET: ElementType,
    {
        self.element_type.dtype()
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

    /// Decompress one block into `buf`.
    ///
    /// # Arguments
    ///
    /// - `block_idx` - zero-based block index in `0..(nitems / block_len)`.
    ///   **Panics** if out of range.
    /// - `buf` - destination buffer. Must be exactly `block_len * dtype.itemsize()` bytes.
    /// - `context` - read context used for decoding.
    ///
    /// # Errors
    ///
    /// Returns `InvalidBufferSize` if `buf` has the wrong length.
    /// Propagates any codec error.
    pub(crate) fn read_block(
        &self,
        block_idx: u64,
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()>
    where
        ET: ElementType,
    {
        let b_size_bytes = self.block_len() as usize * self.dtype().itemsize() as usize;
        ensure!(
            buf.len() == b_size_bytes,
            InvalidBufferSize,
            "Buffer size does not match block size: got {}, expected {b_size_bytes}",
            buf.len()
        );

        let block_offsets = self.block_offsets.as_ref();
        let begin = block_offsets[block_idx as usize] as usize;
        let end = block_offsets[block_idx as usize + 1] as usize;
        let block_data = &self.block_data.as_ref()[begin..end];

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
            block_offsets: self.block_offsets.as_ref(),
            nitems: self.nitems,
            element_type: self.element_type.clone(),
            block_size: self.block_size,
            decoder_config: self.decoder_config.clone(),
        }
    }

    pub(crate) fn decoder_config(&self) -> &DecoderCodecConfig
    where
        ET: ElementType,
    {
        unsafe { assert_unchecked_eq!(self.element_type.dtype(), &self.decoder_config.dtype) };
        &self.decoder_config
    }

    pub(crate) fn into_type<NewET: ElementType>(self) -> Result<BlockTable<S, NewET>>
    where
        ET: ElementType,
    {
        let element_type = NewET::from_dtype(self.dtype().clone())?;
        Ok(BlockTable {
            block_data: self.block_data,
            block_offsets: self.block_offsets,
            nitems: self.nitems,
            element_type,
            block_size: self.block_size,
            decoder_config: self.decoder_config,
        })
    }
}

/// Build a [`BlockTable<Owned>`] by pulling compressed blocks from `block_fn`.
///
/// Iterates over all `nblocks` blocks in batches sized to produce roughly 64 KB of compressed
/// output per call, delegating each batch to [`BlockFn::get_compressed_blocks`]. The returned
/// bytes and end-offsets are accumulated into the final `BlockTable`.
///
/// # Arguments
///
/// - `nblocks` - total number of blocks to build; may be zero.
/// - `block_size` - items per block (passed through to [`BlockTable::new`]).
/// - `decoder_config` - codec/dtype configuration stored in the table.
/// - `compressed_block_size_bound` - upper bound on a single block's compressed byte size;
///   used only to size the iteration chunk (no correctness requirement, just a performance hint).
/// - `block_fn` - the data source; called once per batch.
///
/// # Errors
///
/// Propagates any error returned by `block_fn` or by [`BlockTable::new`].
pub(crate) fn build_block_table<ET>(
    nblocks: u64,
    block_size: BlockSize,
    decoder_config: DecoderCodecConfig,
    compressed_block_size_bound: usize,
    block_fn: &mut impl BlockFn,
) -> Result<BlockTable<Owned, ET>>
where
    ET: ElementType,
{
    let mut block_data = Vec::<u8>::new();
    let mut block_offsets = Vec::<u64>::new();

    let mut block_data_total_len = 0;
    let chunk = (64 * 1024 / compressed_block_size_bound).max(1) as u64; // try to write 64KB at a time

    for block_index in (0..nblocks).step_by(chunk as usize) {
        let blocks = block_index..(block_index + chunk).min(nblocks);
        let base_offset = block_data_total_len;

        // Get blocks data
        let (data, offsets) = block_fn.get_compressed_blocks(blocks.clone(), base_offset)?;
        debug_assert_eq!(offsets.len(), (blocks.end - blocks.start) as usize);

        // Write compressed data
        block_data.extend_from_slice(data);

        // Record offsets
        if block_index == 0 {
            block_offsets.push(0);
        }
        let x = *offsets.last().unwrap();
        debug_assert!(block_data_total_len <= x);
        if !(offsets.windows(2).all(|w| w[0] <= w[1])) {
            println!("Offsets are not non-decreasing: {offsets:?}");
        }
        debug_assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
        block_offsets.extend_from_slice(offsets);

        block_data_total_len = *offsets.last().unwrap();
    }

    debug_assert_eq!(
        block_offsets.len(),
        if nblocks == 0 {
            0
        } else {
            nblocks as usize + 1
        }
    );
    BlockTable::new(block_data, block_offsets, block_size, decoder_config)
}

impl<ET> BlockTable<Owned, ET> {
    /// Build a `BlockTable` by encoding raw item bytes in one shot.
    ///
    /// Splits `data` into chunks of `block_size * dtype.itemsize()` bytes, compresses each
    /// chunk with `encoder`, and returns the fully constructed table.
    /// This is the single-call alternative to the incremental [`BlockTableBuilder`].
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

        let mut block_data = Vec::<u8>::new();
        let mut block_offsets = Vec::<u64>::new();
        for plain_data in data.chunks(b_size_bytes) {
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
            if block_offsets.is_empty() {
                block_offsets.push(0);
            }
            block_offsets.push(block_data.len() as u64);
        }

        let decoder_config = DecoderCodecConfig {
            codec: match &encoder.compressor {
                Compressor::Zstd(_) => Codec::Zstd,
            },
            filters: encoder.filters.clone(),
            dtype: dtype.clone(),
        };
        BlockTable::new(block_data, block_offsets, block_size, decoder_config)
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
pub trait BlockTableStorage {
    type Data<T: 'static>: AsRef<[T]>;
}

#[doc(hidden)]
pub struct Owned(pub(crate) PhantomData<()>);
#[doc(hidden)]
pub struct Borrowed<'a>(pub(crate) PhantomData<&'a ()>);
#[doc(hidden)]
pub struct Mmap {
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
pub struct MmapData<T: 'static> {
    #[allow(unused)]
    pub(crate) mmap: Arc<memmap2::Mmap>,
    pub(crate) data: (SendSyncPtr<T>, usize),
}
impl<T: 'static> AsRef<[T]> for MmapData<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.0.as_ptr(), self.data.1) }
    }
}
impl BlockTableStorage for Mmap {
    type Data<T: 'static> = MmapData<T>;
}

/// A source of pre-compressed block data consumed by [`build_block_table`] and
/// `write_content_impl`.
///
/// Both consumers iterate over blocks in batches and call `get_compressed_blocks` once per batch.
/// The implementor is responsible for compressing the requested blocks and returning them together
/// with their cumulative end-offsets.
///
/// Two implementations are provided in this module:
/// - [`BlockFnWithState`] - closure-based, used when encoding an [`Array`](crate::Array) into a
///   new `BlockTable`.
/// - The inner `BlockFnImpl` returned by [`BlockTable::to_block_fn`] - zero-copy slice into an
///   existing `BlockTable`, used when re-serializing already-compressed data.
pub(crate) trait BlockFn {
    /// Produce compressed data for a contiguous range of block indices.
    ///
    /// # Arguments
    ///
    /// - `blocks` - half-open range of block indices to compress, e.g. `4..8`.
    /// - `base_offset` - the caller's accumulated byte count *before* this batch; equal to the
    ///   absolute byte offset where `blocks.start`'s compressed data begins. Used by
    ///   implementations that need to produce absolute offsets.
    ///
    /// # Returns
    ///
    /// A pair `(data, offsets)` where:
    /// - `data` - concatenated compressed bytes for all blocks in `blocks`.
    /// - `offsets` - cumulative end-offsets of each block within the *entire* data stream
    ///   (not relative to this batch). `offsets[i]` is the absolute byte position immediately
    ///   after block `blocks.start + i`. Length must equal `blocks.end - blocks.start`.
    fn get_compressed_blocks(
        &mut self,
        blocks: Range<u64>,
        base_offset: u64,
    ) -> Result<(&[u8], &[u64])>;
}

/// Closure-based [`BlockFn`] implementation that carries its own mutable state.
///
/// Wraps a closure `F` of the form
/// `FnMut(Range<u64>, u64, &mut E) -> Result<(&[u8], &[u64])>` together with a mutable extension
/// value `E` that can hold reusable scratch buffers (e.g., pre-allocated compressed-data and
/// offset vectors). This avoids per-call allocations while keeping the closure signature clean.
///
/// Construct with [`BlockFnWithState::from_fn`].
pub(crate) struct BlockFnWithState<F, E> {
    impl_fn: F,
    extension: E,
}
impl<F, E> BlockFnWithState<F, E> {
    pub(crate) fn from_fn(extension: E, impl_fn: F) -> Self
    where
        F: for<'a> FnMut(Range<u64>, u64, &'a mut E) -> Result<(&'a [u8], &'a [u64])>,
    {
        Self { impl_fn, extension }
    }
}
impl<F, E> BlockFn for BlockFnWithState<F, E>
where
    F: for<'a> FnMut(Range<u64>, u64, &'a mut E) -> Result<(&'a [u8], &'a [u64])>,
{
    fn get_compressed_blocks(
        &mut self,
        blocks: Range<u64>,
        base_offset: u64,
    ) -> Result<(&[u8], &[u64])> {
        (self.impl_fn)(blocks, base_offset, &mut self.extension)
    }
}

impl<S, ET> BlockTable<S, ET>
where
    S: BlockTableStorage,
{
    /// Adapt this `BlockTable` into a [`BlockFn`] for use with [`build_block_table`] or
    /// `write_content_impl`.
    ///
    /// The returned `BlockFn` slices directly into `self.block_data` and `self.block_offsets`
    /// with no copying or re-compression. The second return value is the maximum compressed size
    /// of any single block, used by callers to compute a batch size that targets ~64 KB per call.
    pub(crate) fn to_block_fn<'a>(&'a self) -> (impl BlockFn + 'a, usize) {
        assert!(self.nitems.is_multiple_of(self.block_size as u64));
        let compressed_block_size_bound = self
            .block_offsets
            .as_ref()
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0);

        struct BlockFnImpl<'a, S, ET>
        where
            S: BlockTableStorage,
        {
            table: &'a BlockTable<S, ET>,
        }
        impl<'a, S, ET> BlockFn for BlockFnImpl<'a, S, ET>
        where
            S: BlockTableStorage,
        {
            fn get_compressed_blocks(
                &mut self,
                blocks: Range<u64>,
                base_offset: u64,
            ) -> Result<(&[u8], &[u64])> {
                let start = blocks.start as usize;
                let end = blocks.end as usize;
                let all_offsets = self.table.block_offsets.as_ref();

                assert_eq!(base_offset, all_offsets[start]);

                let data_start = all_offsets[start] as usize;
                let data_end = all_offsets[end] as usize;
                let data = &self.table.block_data.as_ref()[data_start..data_end];
                let offsets = &all_offsets[start + 1..=end];

                Ok((data, offsets))
            }
        }
        (
            BlockFnImpl { table: self },
            compressed_block_size_bound as usize,
        )
    }
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
        let block_bytes = table.block_len() as usize * table.dtype().itemsize() as usize;
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
        assert_eq!(table.block_offsets.len(), 2);
        assert_eq!(table.nitems, 8);
        let mut context = ReadContext::default();
        assert_eq!(decode_block(&table, 0, &mut context), items);
    }

    #[test]
    fn build_multiple_blocks_exact_divisor() {
        // 12 items, block_size=4 -> 3 full blocks
        let items: Vec<u8> = (0u8..12).collect();
        let table = build_from_items(&items, 4, &EncoderParams::default()).unwrap();
        assert_eq!(table.block_offsets.len(), 4);
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
        assert_eq!(table.block_offsets.len(), 3);
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
            .into_type::<Ty<T>>()
            .unwrap()
    }

    #[test]
    fn round_trip_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let table2 = round_trip(&items, 8);
        assert_eq!(table2.block_offsets.len(), 2);
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
        assert_eq!(table.block_offsets.len(), 4);
        assert_eq!(table.nitems, 12);
        let mut context = ReadContext::default();
        let recovered: Vec<u8> = (0..table.block_offsets.len() - 1)
            .flat_map(|i| decode_block(&table, i, &mut context))
            .collect();
        assert_eq!(recovered, items);
    }

    #[test]
    fn round_trip_preserves_block_offsets_ordering() {
        let items: Vec<u8> = (0u8..12).collect();
        let table2 = round_trip(&items, 3);
        let offs = table2.block_offsets;
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
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

        assert_eq!(table2.block_offsets.len(), 7);
        assert_eq!(table2.nitems, 18);
        let mut context = ReadContext::default();
        let recovered: Vec<u8> = (0..table2.block_offsets.len() - 1)
            .flat_map(|i| decode_block(&table2, i, &mut context))
            .collect();
        assert_eq!(recovered, unsafe { cast_slice::<u32, u8>(&items) });
    }

    fn make_storage<T: Dtyped>(
        items: &[T],
        block_len: BlockSize,
        encoder_params: &EncoderParams,
    ) -> BlockTable<Owned, Ty<T>> {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::DTYPE,
            block_len,
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
        let block_bytes = storage.block_len() as usize * storage.dtype().itemsize() as usize;
        let mut buf = AlignedBytes::with_capacity(T::DTYPE.alignment().as_usize(), block_bytes);
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
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 8);
        assert_eq!(s.dtype(), &u8::DTYPE);
        assert_eq!(read_block_items::<u8, _>(&s, 0), items);
    }

    #[test]
    fn two_blocks_i32_round_trips() {
        let items: Vec<i32> = (0..8).collect();
        let s = make_storage(&items, 4, &EncoderParams::default());
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 4);
        assert_eq!(read_block_items::<i32, _>(&s, 0), items[..4]);
        assert_eq!(read_block_items::<i32, _>(&s, 1), items[4..]);
    }

    #[test]
    fn multiple_blocks_f32_round_trips() {
        let items: Vec<f32> = (0..12).map(|x| x as f32 * 0.5).collect();
        let s = make_storage(&items, 4, &EncoderParams::default());
        assert_eq!(s.nitems(), 12);
        assert_eq!(s.block_len(), 4);
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
