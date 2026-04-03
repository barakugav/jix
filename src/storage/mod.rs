use crate::dtype::Dtype;
use crate::error::Error;
use crate::util::DimArray;

mod block;
mod codec;
mod common;
mod plain;

pub(crate) trait ArrayStorage {
    fn dtype(&self) -> &Dtype;
    fn shape(&self) -> &[usize];
    fn chunks_layout(&self) -> &ChunksLayout;
    fn get_chunk_data(
        &self,
        chunk_global_id: usize,
        chunk_idx: &[usize],
        buf: &mut [u8],
    ) -> Result<(), Error>;
}
pub(crate) struct ChunksLayout {
    pub(crate) chunk_shape: DimArray<usize>,
    pub(crate) chunk_space_shape: DimArray<usize>,
    pub(crate) chunk_size: usize, // chunk_shape.iter().product()
}
impl ChunksLayout {
    pub(crate) fn new(chunk_shape: &[usize], shape: &[usize]) -> Self {
        let chunk_shape = chunk_shape.iter().cloned().collect::<DimArray<_>>();
        let chunk_space_shape = shape
            .iter()
            .zip(&chunk_shape)
            .map(|(&s, &c)| s.div_ceil(c))
            .collect();
        let chunk_size = chunk_shape.iter().product();
        Self {
            chunk_shape,
            chunk_space_shape,
            chunk_size,
        }
    }
}

pub(crate) struct ChunkDesc {
    // global_index: usize,
    chunk_idx: DimArray<usize>,
}

// pub(crate) struct DynStorage {
//     storage: Box<dyn ArrayStorage>,
//     dtype: Dtype,
//     shape: DimVec<usize>,
//     chunks_layout: ChunksLayoutInfo,
// }
// impl ArrayStorage for DynStorage {
//     fn dtype(&self) -> &Dtype {
//         &self.dtype
//     }
//     fn shape(&self) -> &[usize] {
//         &self.shape
//     }
//     fn chunks_layout(&self) -> &ChunksLayoutInfo {
//         &self.chunks_layout
//     }
//     fn get_chunk(&self, chunk_idx: &[usize], chunk_buf: &mut ChunkBuf) -> Result<(), Error> {
//         self.storage.get_chunk(chunk_idx, chunk_buf)
//     }
// }
