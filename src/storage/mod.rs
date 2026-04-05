pub(crate) mod archive;
pub(crate) mod block;
pub(crate) mod codec;
mod plain;

use std::io;

use crate::{dtype::Dtype, storage::codec::ReadContext};

pub(crate) type BlockSize = u32;

/// Storage of 1D array items, organized in blocks.
///
/// The number of items must be divisible by the block length, there is not support for partial blocks.
/// At all times the storage hold the invariants:
/// - `block_len > 0`
/// - `nitems % block_len == 0`
pub trait Storage {
    /// Get the dtype of items in this storage.
    fn dtype(&self) -> &Dtype;

    /// Get the total number of items in this storage.
    fn nitems(&self) -> usize;

    /// Get the length of a block in this storage.
    ///
    /// Note that the units are in items, not bytes.
    fn block_len(&self) -> BlockSize;

    /// Read a block of items into the provided buffer.
    ///
    /// # Arguments
    ///
    /// - `block_idx`: The index of the block to read, in the range `0..(nitems / block_len)`.
    /// - `buf`: The buffer to read the block into. Must be of size `block_len * dtype.itemsize()`.
    /// - `context`: a read context containing global configuration and reuseable buffers.
    fn read_block(
        &self,
        block_idx: usize,
        buf: &mut [u8],
        context: &mut ReadContext,
    ) -> io::Result<()>;
}
