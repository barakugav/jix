use std::ops::Range;
use std::sync::Arc;

use zix_core::array::BlocksLayout;
use zix_core::codec::{DecoderParams, EncoderParams, ReadContext};
use zix_core::dtype::Dtype;
use zix_core::storage::ArrayStorage;

#[derive(Clone)]
pub(crate) struct DynStorage(pub(crate) Arc<dyn ArrayStorage + Send + Sync>);
impl ArrayStorage for DynStorage {
    fn shape(&self) -> &[u64] {
        self.0.shape()
    }

    fn dtype(&self) -> &Dtype {
        self.0.dtype()
    }

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> std::io::Result<()> {
        self.0.read_data(index, buf, context)
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        self.0.blocks_layout()
    }

    fn codec_params(&self) -> (&EncoderParams, &DecoderParams) {
        self.0.codec_params()
    }
}
