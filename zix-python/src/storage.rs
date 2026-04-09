use std::sync::Arc;

use zix_core::array::BlocksLayout;
use zix_core::codec::ReadContext;
use zix_core::dtype::Dtype;
use zix_core::storage::ArrayStorage;

#[derive(Clone)]
pub(crate) struct DynStorage(pub(crate) Arc<dyn ArrayStorage + Send + Sync>);
impl ArrayStorage for DynStorage {
    fn dtype(&self) -> &Dtype {
        self.0.dtype()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        self.0.blocks_layout()
    }

    fn read_data(
        &self,
        index: &[std::ops::Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> std::io::Result<()> {
        self.0.read_data(index, buf, context)
    }
}
