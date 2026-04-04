use std::io;

use crate::dtype::Dtype;

mod block;
pub(crate) use block::BlockSize;
mod codec;
mod common;
mod compressed;
mod plain;

pub(crate) trait Storage {
    fn dtype(&self) -> &Dtype;
    fn nitems(&self) -> usize;
    fn block_len(&self) -> BlockSize;
    fn read_block(&self, block_idx: usize, buf: &mut [u8]) -> io::Result<()>;
}
