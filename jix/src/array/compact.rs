use crate::codec::{DecoderCodecConfig, Encoder};
use crate::error::Result;
use crate::storage::block::{BlockTableBuilder, OwnedBlockTableBuilder};
use crate::storage::{ArrayBlockTableStorageBase, BlockSize, Compact, StridedBuf};
use crate::util::iter::NdIter;
use crate::util::{scale_read_shape, AlignedBytes, IterExt, NdCopier, PtrMutNoalias, PtrNoalias};
use crate::{
    default_logical_strides, default_strides, ArrayParams, ArrayStorage, Dim, DimDyn, DimVec,
    Dimension, ReadContext,
};

use super::Array;

impl<S: ArrayStorage> Array<S> {
    /// Compress the data of this array into a new `Array<Compact>` with new blocks.
    ///
    /// The primary use of `compact` is to materialize a lazy operation chain:
    /// An `Array<S>` can have an arbitrary storage implementation, often a lazy view of one or
    /// more computations, for example `Array<Floor<Map<Compact>>>` (see the examples).
    /// Reads to such lazy view arrays always perform the whole computation pipeline on the fly,
    /// which is very flexible but can be inefficient for repeated access. Copying the data and
    /// re-compressing it into a new array with `compact` breaks the lazy storage chain and materializes
    /// the result as a standalone `Array<Compact>`.
    ///
    /// In contrast to "simple" views such as unary element-wise operations, lazy ops that change the
    /// shape of the array (e.g. `reshape`, `broadcast`, `permute_dims`) can cause block boundaries
    /// to no longer align with the logical layout of the array, causing reads to decompress excess
    /// data. Calling `compact` on the result of such an operation re-encodes the data with a freshly
    /// derived block shape that matches the new layout. The block shape of copied arrays is
    /// automatically derived and tuned from the underlying storage(s), using a heuristic that aims
    /// to preserve user choices (that may depend on the user knowledge of the access pattern), but
    /// it's not perfect - you may want to explicitly pass some parameters via
    /// [`compact_with`](Array::compact_with).
    ///
    /// It's also possible to materialize a lazy operation chain directly into a file without holding
    /// the whole result (compressed or decompressed) in memory.
    /// See [`write_to_file`](Array::write_to_file) and its variants for details and examples.
    ///
    /// Codec settings (compression level, filters, etc.) are also inherited from the source storage.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - compression or decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// let result =                  // Array<Compact>
    ///     a.map(|x| x * 7.399_f32)  // Array<Map<Compact>>
    ///    .floor()                   // Array<Floor<Map<Compact>>>
    ///    .compact()?;               // Array<Compact> - materialize the pipeline
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact(&self) -> Result<Array<Compact<S::ElementType>>> {
        let context = self.read_ctx();
        self.compact_with(ArrayParams::default(), &context)
    }

    /// Compress the data of this array into a new `Array<Compact>` with explicit control over parameters.
    ///
    /// Like [`compact`](Array::compact) (see its documentation), but with explicit [`ArrayParams`].
    /// Any optional field in `params` that is not set will be inherited from the source storage
    /// if possible.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - compression or decompression fails.
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a_params = ArrayParams::default().block_shape(&[1, 2])?;
    /// let a = Array::compact_ndarray_with(&array![[1.5f32, 2.0], [3.14, 6.17]], a_params)?;
    ///
    /// // Let's say a is given to us, and we prepare to access it many times with a specific access
    /// // pattern. We re-compress it with a matching block shape.
    /// let b = a.compact_with(ArrayParams::default().block_shape(&[2, 1])?, &a.read_ctx())?;
    ///
    /// assert_eq!(
    ///     b.to_ndarray_sub(&[0..2, 0..1], &b.read_ctx())?,
    ///     array![[1.5], [3.14]]
    /// );
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_with(
        &self,
        mut params: ArrayParams,
        context: &ReadContext,
    ) -> Result<Array<Compact<S::ElementType>>> {
        fn builder_init(
            nblocks: u64,
            block_shape: &[BlockSize],
            decoder_config: DecoderCodecConfig,
        ) -> Result<OwnedBlockTableBuilder> {
            let block_size = block_shape.iter().cloned().try_shape_product().unwrap();
            OwnedBlockTableBuilder::start(nblocks, block_size, decoder_config)
        }

        let builder = self.compact_into_builder(&mut params, context, builder_init)?;
        let blocks = unsafe { builder.into_table::<S::ElementType>()? };
        let shape = DimDyn::from_slice(self.shape());
        Ok(Array {
            storage: Compact(ArrayBlockTableStorageBase::new(blocks, shape, params)?),
        })
    }

    /// Read this array's data, compress it block by block, and feed each compressed block to a
    /// builder created by `builder_init`.
    ///
    /// # Block layout
    ///
    /// `params.block_shape` divides the array into an N-dimensional grid of target blocks. To read
    /// from `self` efficiently, data is fetched in larger *chunks* - each spanning a whole number of
    /// target blocks per dimension - with a single `read_data` call per chunk, then carved into
    /// individual target blocks in memory. The chunk shape is derived from `self`'s own block shape
    /// and scaled toward its preferred read size (see `scale_read_shape`), keeping reads aligned to
    /// how `self` is stored. Blocks are therefore produced in chunk-traversal order rather than
    /// global C order, so each block is handed to the builder together with its logical (C-order)
    /// `block_index`. Boundary blocks - those that extend beyond the array's shape - are zero-padded
    /// to fill the full `block_shape` before compression.
    pub(crate) fn compact_into_builder<B>(
        &self,
        params: &mut ArrayParams,
        context: &ReadContext,
        builder_init: impl FnOnce(
            /* nblocks */ u64,
            /* block_shape */ &[BlockSize],
            /* decoder_config */ DecoderCodecConfig,
        ) -> Result<B>,
    ) -> Result<B::Output>
    where
        B: BlockTableBuilder,
    {
        // Erase the storage to reduce monomorphization bloat
        let storage: &dyn ArrayStorage = &self.storage;
        context.top_level_read(storage.spec().dynamic().arrays_to_cache.clone(), || {
            compact_into_builder_dyn_dispatch(storage, params, context, builder_init)
        })
    }
}

#[inline(never)]
fn compact_into_builder_dyn_dispatch<B: BlockTableBuilder>(
    storage: &dyn ArrayStorage,
    params: &mut ArrayParams,
    context: &ReadContext,
    builder_init: impl FnOnce(
        /* nblocks */ u64,
        /* block_shape */ &[BlockSize],
        /* decoder_config */ DecoderCodecConfig,
    ) -> Result<B>,
) -> Result<B::Output> {
    let compact_fn = match storage.shape().len() {
        1 => compact_into_builder_dyn::<_, Dim<1>>,
        2 => compact_into_builder_dyn::<_, Dim<2>>,
        3 => compact_into_builder_dyn::<_, Dim<3>>,
        4 => compact_into_builder_dyn::<_, Dim<4>>,
        _ => compact_into_builder_dyn::<_, DimDyn>,
    };
    compact_fn(storage, params, context, builder_init)
}
#[inline(never)]
fn compact_into_builder_dyn<B: BlockTableBuilder, D: Dimension>(
    storage: &dyn ArrayStorage,
    params: &mut ArrayParams,
    context: &ReadContext,
    builder_init: impl FnOnce(
        /* nblocks */ u64,
        /* block_shape */ &[BlockSize],
        /* decoder_config */ DecoderCodecConfig,
    ) -> Result<B>,
) -> Result<B::Output> {
    let shape = storage.shape();
    let ndim = shape.len();
    let dtype = storage.dtype();

    params.inherit_unset_from_storage(&storage);
    params.tune(shape, dtype)?;

    let block_shape = params.block_shape.as_ref().unwrap();
    let block_shape = D::vec(ndim, |dim| block_shape[dim]);
    let block_size = block_shape
        .as_ref()
        .iter()
        .cloned()
        .try_shape_product()
        .unwrap();
    let block_grid_shape = D::vec(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));
    let nblocks = block_grid_shape
        .as_ref()
        .iter()
        .cloned()
        .try_shape_product()
        .unwrap();
    // C-order strides over the block grid, used to map a block's grid position to its logical
    // index (blocks are produced out of C order, so each one carries its own index).
    let block_grid_lstrides = default_logical_strides(&block_grid_shape);

    let encoder_params = params.encoder_params.clone().unwrap_or_default();
    let mut encoder = Encoder::new(&encoder_params, dtype.clone())?;
    let decoder_cfg = DecoderCodecConfig {
        codec: encoder_params.codec,
        filters: encoder_params.filters,
        dtype: dtype.clone(),
    };

    let itemsize = encoder.dtype.itemsize() as usize;
    let alignment = encoder.dtype.alignment().as_usize();
    let block_size_bytes = block_size as usize * itemsize;
    let block_strides = default_strides(&block_shape, itemsize);
    let block_compressed_bound = encoder.encode_bound(block_size_bytes);
    let copier = NdCopier::new(dtype);

    let spec = storage.spec();
    let current_block_shape = spec.block_shape();
    let mut chunk_shape_in_blocks = D::from_fn(ndim, |dim| {
        (current_block_shape[dim] / block_shape[dim]).max(1) as u64
    });
    let read_size = spec.read_size();
    let (min_chunk, max_chunk) = (
        (read_size.min / block_size_bytes as u64).max(1),
        (read_size.max / block_size_bytes as u64).max(1),
    );
    scale_read_shape(
        chunk_shape_in_blocks.as_mut_slice(),
        block_grid_shape.as_ref(),
        block_grid_shape.as_ref(),
        (min_chunk, max_chunk),
        spec.read_shape_scale_weight(),
        spec.read_shape_scale_order().iter().map(|&i| i as usize),
    );

    // A chunk spans `chunk_shape_in_blocks` target blocks per dimension (element units). We read
    // a whole chunk from `storage` in one pass, then carve the target blocks out of it.
    let chunk_shape = D::vec(ndim, |dim| {
        block_shape[dim] as u64 * chunk_shape_in_blocks[dim]
    });
    let chunk_grid_shape = D::vec(ndim, |dim| shape[dim].div_ceil(chunk_shape[dim]));

    let mut chunk_buf = AlignedBytes::new_padded(alignment);
    let mut tmp_block_plain = AlignedBytes::new_padded(alignment);
    let mut tmp_block_compressed = AlignedBytes::new_padded(alignment);
    let mut builder = builder_init(nblocks, block_shape.as_ref(), decoder_cfg)?;

    // Outer loop over chunks. The extension yields each chunk's active element extent, clamped
    // to the array at the high boundary.
    let chunk_iter = NdIter::builder(chunk_grid_shape)
        .with_block_offset_size_ext(
            &D::vec(ndim, |_| 0),
            &D::vec(ndim, |dim| shape[dim]),
            chunk_shape.clone(),
        )
        .build();
    for (chunk_idx, (chunk_inner_offset, chunk_size)) in chunk_iter {
        debug_assert!(chunk_inner_offset.as_ref().iter().all(|&off| off == 0));
        let read_range = D::vec(ndim, |dim| {
            let start = chunk_idx[dim] * chunk_shape[dim];
            start..start + chunk_size[dim]
        });
        let chunk_bytes = chunk_size.as_ref().iter().product::<u64>() as usize * itemsize;
        chunk_buf.clear();
        chunk_buf.reserve(chunk_bytes);
        unsafe { chunk_buf.set_len(chunk_bytes) };
        let chunk_strides = default_strides(&chunk_size, itemsize);
        storage.read_data(
            read_range.as_ref(),
            context,
            Some(&mut unsafe {
                StridedBuf::from_slice_mut(chunk_buf.as_mut_slice(), chunk_strides.as_ref())
            }),
        )?;
        let chunk_offset_base = (0..ndim)
            .map(|dim| chunk_idx[dim] * chunk_shape_in_blocks[dim] * block_grid_lstrides[dim])
            .sum::<u64>();

        // Inner loop over the target blocks within the chunk.
        let block_iter = NdIter::builder(D::vec(ndim, |dim| {
            chunk_size[dim].div_ceil(block_shape[dim] as u64)
        }))
        .with_block_offset_size_ext(
            &D::vec(ndim, |_| 0),
            &chunk_size,
            D::vec(ndim, |dim| block_shape[dim] as u64),
        )
        .build();
        for (block_in_chunk_idx, (block_inner_offset, block_active_size)) in block_iter {
            debug_assert!(block_inner_offset.as_ref().iter().all(|&off| off == 0));
            // Logical (C-order) index of this block in the full grid.
            let block_index = chunk_offset_base
                + (0..ndim)
                    .map(|dim| block_in_chunk_idx[dim] * block_grid_lstrides[dim])
                    .sum::<u64>();
            let full_block = (0..ndim).all(|dim| block_active_size[dim] == block_shape[dim] as u64);

            // Gather the block out of the chunk buffer with the target block strides,
            // zero-padding boundary blocks first.
            tmp_block_plain.clear();
            tmp_block_plain.reserve(block_size_bytes);
            unsafe { tmp_block_plain.set_len(block_size_bytes) };
            if !full_block {
                tmp_block_plain.fill(0); // zero-pad
            }
            let src_byte_offset = (0..ndim)
                .map(|dim| {
                    block_in_chunk_idx[dim] as usize
                        * block_shape[dim] as usize
                        * chunk_strides[dim]
                })
                .sum::<usize>();
            // TODO: this nd_copy may be redundant if the chunk and block strides are identical.
            // In that case, we can compress directly from the chunk buffer
            unsafe {
                copier.copy(
                    PtrNoalias::from_slice(chunk_buf.as_slice().get_unchecked(src_byte_offset..)),
                    PtrMutNoalias::from_slice(tmp_block_plain.as_mut_slice()),
                    D::vec(ndim, |dim| block_active_size[dim] as usize).as_ref(),
                    chunk_strides.as_ref(),
                    block_strides.as_ref(),
                    dtype,
                )
            };
            let plain_data = tmp_block_plain.as_slice();

            // Compress block data
            tmp_block_compressed.clear();
            tmp_block_compressed.reserve(block_compressed_bound);
            unsafe { tmp_block_compressed.set_len(block_compressed_bound) };
            let cdata_len = encoder.encode(plain_data, tmp_block_compressed.as_mut_slice())?;
            unsafe { tmp_block_compressed.set_len(cdata_len) };

            builder.write_compressed_block(block_index, tmp_block_compressed.as_slice())?;
        }
    }

    builder.finalize()
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::array;

    use super::Array;

    use crate::dtype::Dtyped;

    use crate::util::arr_params;
    use crate::ArrayStorage;

    use crate::array::tests::*;

    // -----------------------------------------------------------------------
    // compact
    // -----------------------------------------------------------------------

    #[test]
    fn compact_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], [4], &[4]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.ndim(), 1);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.storage.spec().block_shape()[..], [4]);
        assert_eq!(b.to_ndarray().unwrap(), array![0, 1, 2, 3]);
    }

    #[test]
    fn compact_1d_multi_block() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], [6], &[3]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[6]);
        assert_eq!(b.storage.spec().block_shape()[..], [3]);
        assert_eq!(b.to_ndarray().unwrap(), array![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn compact_1d_with_padding() {
        // shape [5], block [3] -> stored as 6 elements (padded)
        let src = array![0u8, 1, 2, 3, 4];
        let a = Array::compact_ndarray_with(&src, arr_params(&[3])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[5]);
        assert_eq!(b.storage.spec().block_shape()[..], [3]);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], [8], &[4]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[8]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(
            b.to_ndarray().unwrap(),
            array![10, 20, 30, 40, 50, 60, 70, 80]
        );
    }

    #[test]
    fn compact_2d_single_block() {
        // shape=[2,3], block=[2,3] - one block, no partial-block path
        let a = array(&[&[0u8, 1, 2, 3, 4, 5]], [2, 3], &[2, 3]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.storage.spec().block_shape()[..], [2, 3]);
        assert_eq!(b.to_ndarray().unwrap(), array![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn compact_2d_multi_block() {
        // shape=[4,6], block=[2,3] - 4 blocks, exercises the full-block compact path
        // Block layout (row-major grid):
        //   block0=[0,0]: rows 0-1, cols 0-2 -> 0,1,2,6,7,8
        //   block1=[0,1]: rows 0-1, cols 3-5 -> 3,4,5,9,10,11
        //   block2=[1,0]: rows 2-3, cols 0-2 -> 12,13,14,18,19,20
        //   block3=[1,1]: rows 2-3, cols 3-5 -> 15,16,17,21,22,23
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            [4, 6],
            &[2, 3],
        );
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[4, 6]);
        assert_eq!(b.storage.spec().block_shape()[..], [2, 3]);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ndarray::Array::from_shape_vec([4, 6], (0u8..24).collect()).unwrap()
        );
    }

    #[test]
    fn compact_2d_with_padding() {
        // shape=[3,5], block=[2,3] -> padded to [4,6]; shape preserved as [3,5].
        // Block grid 2*2:
        //   [0,0]: size [2,3] - full block
        //   [0,1]: size [2,2] - partial in dim1
        //   [1,0]: size [1,3] - partial in dim0
        //   [1,1]: size [1,2] - partial in BOTH dims (corner block)
        #[rustfmt::skip]
        let src = array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 3])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[3, 5]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_3d_with_padding_in_all_dims() {
        // shape=[3,3,5], block=[2,2,3] -> padded to [4,4,6].
        // Block grid 2*2*2 = 8 blocks; every boundary block is partial in at least
        // one dimension, and the single corner block [1,1,1] is partial in all three:
        //   size [1,1,2] vs block_shape [2,2,3].
        let src = ndarray::Array3::<u8>::from_shape_vec([3, 3, 5], (0u8..45).collect()).unwrap();
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 2, 3])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[3, 3, 5]);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.storage.spec().block_shape()[..], [2, 2, 3]);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_preserves_block_shape() {
        // Verify the copied array has the same block layout as the source.
        let src = array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::compact_ndarray_with(&src, arr_params(&[4])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(
            a.storage.spec().block_shape()[..],
            b.storage.spec().block_shape()[..]
        );
    }

    #[test]
    fn compact_result_is_independent() {
        // Mutating the source array should not affect the compact (they are independent).
        // Since Array<Compact> doesn't expose mutation, we verify by round-tripping
        // both through write/read and checking values remain consistent.
        let src = array![10u8, 20, 30, 40];
        let a = Array::compact_ndarray_with(&src, arr_params(&[4])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(a.to_ndarray().unwrap(), src);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    // -----------------------------------------------------------------------
    // to_plain
    // -----------------------------------------------------------------------

    #[test]
    fn to_plain_i32() {
        let src = array![[10i32, 20, 30], [40, 50, 60]];
        let a = Array::compact_ndarray(&src).unwrap();

        // Materializing returns an equal array.
        let plain = a.view().to_plain().unwrap();
        assert_eq!(plain.shape(), &[2, 3]);
        assert_eq!(plain.to_ndarray().unwrap(), src);

        // It also materializes a lazy view.
        let plain = a.map(|x| x + 1).to_plain().unwrap();
        assert_eq!(
            plain.to_ndarray().unwrap(),
            array![[11i32, 21, 31], [41, 51, 61]]
        );
    }

    #[test]
    fn to_plain_carries_the_hinted_layout_into_the_materialized_leaf() {
        let a = Array::compact_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let plain = a.view().permute_dims(&[1, 0]).to_plain().unwrap();
        assert_eq!(plain.shape(), &[3, 2]);
        assert_eq!(
            plain.to_ndarray().unwrap(),
            ndarray::array![[1i32, 4], [2, 5], [3, 6]]
        );
        // `to_plain` goes through `to_ndarray`, so the buffer is F-laid-out; `Plain::new` then
        // re-derives the hint from those strides, and the layout survives into the new leaf.
        assert_eq!(plain.storage().spec().read_layout_order(), &[1, 0]);
    }
}
