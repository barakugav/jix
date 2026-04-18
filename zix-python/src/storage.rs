use std::ops::Range;
use std::sync::Arc;

use zix_core::codec::ReadContext;
use zix_core::dtype::Dtype;
use zix_core::storage::{ArrayStorage, ArrayStorageSpec};

#[derive(Clone)]
pub(crate) struct DynStorage(pub(crate) Arc<dyn ArrayStorage + Send + Sync>);
impl ArrayStorage for DynStorage {
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> zix_core::error::Result<()> {
        self.0.read_data(index, buf, context)
    }

    fn shape(&self) -> &[u64] {
        self.0.shape()
    }

    fn dtype(&self) -> &Dtype {
        self.0.dtype()
    }

    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.0.spec()
    }
}
