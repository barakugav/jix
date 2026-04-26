use crate::codec::{DecoderParams, EncoderParams};
use crate::dtype::Dtype;
use crate::error::{check_ndim, Result};
use crate::storage::block::BlockSize;
use crate::storage::{ArrayStorage, BlockShapeTag, BlocksLayout};
use crate::util::DimArray;
use crate::Array;

/// Parameters controlling the encoding/decoding configs of an [`Array`], and its block layout.
///
/// `ArrayParams` groups two independent sets of configuration:
///
/// - **Codec** — the compression ([`EncoderParams`]) and decompression ([`DecoderParams`])
///   configuration used when decoding blocks from an existing array, or when encoding blocks for a
///   new array. These affect the compression ratio and CPU usage of the codec, but not the block
///   layout.
///
/// - **Block layout** — the nd-block shape used to divide the array into blocks, each compressed
///   independently, and other related hints that are propagated through lazy view storage
///   operations. A good block layout is critical for performance, and should match the access
///   pattern of your workload.
///
/// # When are params applied?
///
/// - When a new array is constructed, such as via [`Array::compact_array`]: the data is split into
///   blocks according to the block layout params, and each block is compressed using the encoder
///   params before being written to storage.
/// - When an array is accessed for read, such as via [`Array::to_ndarray`]: each compressed block
///   is decompressed using the decoder params. Sometimes readers of an array might want to read
///   smaller chunks of data, that is aligned to the block shape (or preferred read shape) to avoid
///   decompressing more data than necessary.
/// - When an array is copied, such as via [`Array::copy`] or [`Array::copy_with`]: a new compressed
///   array is constructed, inheriting any unset params from the source array's storage spec. When
///   the copied array is a compressed array (i.e. not a lazy view), the block shape and codec
///   params are preserved identically by default. Arrays with lazy view storage
///   (e.g. from `Add`, `Reshape`, etc.) may modify the params as best as it can, trying to preserve
///   user-specified params where possible, but it is an approximate heuristic.
///   Shape modifying operations (e.g. `Reshape`, `PermuteAxes`, etc.) are especially likely to
///   change the block layout params - consider passing explicit params to `copy_with` after these
///   ops, or verifying the resulting block layout is reasonable for your access pattern.
///
/// # Recommended usage
///
/// Use `ArrayParams::new()` (equivalent to `ArrayParams::default()`) for most cases — the
/// defaults select a block shape that fits in the L1 data cache using Zstd level 3 with byte
/// shuffling. For latency-sensitive workloads where you know the access pattern, set `block_shape`
/// explicitly and call `copy_with` instead of `copy` after shape-changing ops.
///
/// ```
/// use zix::{Array, ArrayParams};
///
/// // Construct an array with a specific block shape.
/// let data = ndarray::Array2::<f32>::zeros((1024, 1024));
/// let mut params = ArrayParams::new();
/// params.block_shape(&[64, 64]);
/// let za = Array::compact_array_with(&data, params)?;
///
/// // After a shape-changing op, pin the block shape explicitly.
/// let mut out_params = ArrayParams::new();
/// out_params.block_shape(&[128, 128]);
/// let ctx = za.read_ctx();
/// let transposed = za.permute_axes(&[1, 0]).copy_with(out_params, &ctx)?;
/// # Ok::<(), zix::error::Error>(())
/// ```
#[derive(Clone, Default, Debug)]
pub struct ArrayParams {
    pub(crate) block_shape: Option<DimArray<BlockSize>>,
    pub(crate) block_shape_tag: Option<DimArray<BlockShapeTag>>,
    pub(crate) block_size_hint: Option<u64>,
    pub(crate) preferred_read_shape: Option<DimArray<BlockSize>>,
    pub(crate) preferred_read_size_hint: Option<u64>,
    pub(crate) encoder_params: Option<EncoderParams>,
    pub(crate) decoder_params: Option<DecoderParams>,
}
impl ArrayParams {
    /// Creates a new `ArrayParams` with all fields unset (equivalent to [`Default::default()`]).
    ///
    /// Block layout and other params are automatically selected according to cache size heuristics
    /// when not set explicitly.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the explicit storage block shape, in items per dimension.
    ///
    /// When set, the array is stored in nd-blocks of exactly this shape (subject to boundary
    /// clamping at the array edges). This overrides any auto-computed shape.
    ///
    /// Setting the block shape also tags every dimension as [`BlockShapeTag::Fixed`] unless
    /// [`block_shape_tag`](Self::block_shape_tag) is also set, meaning the shape will be
    /// preserved as-is if this `ArrayParams` is later used as a propagation source.
    pub fn block_shape(&mut self, block_shape: &[BlockSize]) -> &mut Self {
        check_ndim(block_shape.len()).unwrap();
        self.block_shape = Some(block_shape.iter().cloned().collect());
        self
    }

    /// Returns the explicit block shape, or `None` if it has not been set.
    pub fn get_block_shape(&self) -> Option<&[BlockSize]> {
        self.block_shape.as_deref()
    }

    /// Clears the explicit block shape.
    pub fn clear_block_shape(&mut self) -> &mut Self {
        self.block_shape = None;
        self
    }

    /// Sets per-dimension tags that control how [`block_shape`](Self::block_shape) is scaled
    /// when the block shape is auto-computed during propagation.
    ///
    /// `tags` must have the same length as `block_shape`. Requires `block_shape` to also be set.
    /// See [`BlockShapeTag`] for the available options.
    pub fn block_shape_tag(&mut self, tags: &[BlockShapeTag]) -> &mut Self {
        check_ndim(tags.len()).unwrap();
        self.block_shape_tag = Some(tags.iter().cloned().collect());
        self
    }

    /// Returns the per-dimension block shape tags, or `None` if not set.
    pub fn get_block_shape_tag(&self) -> Option<&[BlockShapeTag]> {
        self.block_shape_tag.as_deref()
    }

    /// Clears the block shape tags.
    pub fn clear_block_shape_tag(&mut self) -> &mut Self {
        self.block_shape_tag = None;
        self
    }

    /// Sets the target block size in bytes, used when auto-computing the block shape.
    ///
    /// When `block_shape` is not set, or when some dimensions are not [`BlockShapeTag::Fixed`],
    /// the auto-computation scales the block shape so that each block is approximately this many
    /// bytes. Defaults to the L1 data cache size when not provided.
    pub fn block_size_hint(&mut self, size_hint: u64) -> &mut Self {
        self.block_size_hint = Some(size_hint);
        self
    }

    /// Returns the block size hint in bytes, or `None` if not set.
    pub fn get_block_size_hint(&self) -> Option<u64> {
        self.block_size_hint
    }

    /// Clears the block size hint, reverting to the default.
    pub fn clear_block_size_hint(&mut self) -> &mut Self {
        self.block_size_hint = None;
        self
    }

    /// Sets the preferred read shape, in items per dimension.
    ///
    /// This is a hint to the read path: reads are most efficient when they cover a region of
    /// approximately this shape. It is typically larger than the storage block shape, targeting
    /// the L2 cache. When not set, it is auto-computed from [`preferred_read_size_hint`](Self::preferred_read_size_hint).
    pub fn preferred_read_shape(&mut self, read_shape: &[BlockSize]) -> &mut Self {
        check_ndim(read_shape.len()).unwrap();
        self.preferred_read_shape = Some(read_shape.iter().cloned().collect());
        self
    }

    /// Returns the preferred read shape, or `None` if not set.
    pub fn get_preferred_read_shape(&self) -> Option<&[BlockSize]> {
        self.preferred_read_shape.as_deref()
    }

    /// Clears the preferred read shape, reverting to auto-computation.
    pub fn clear_preferred_read_shape(&mut self) -> &mut Self {
        self.preferred_read_shape = None;
        self
    }

    /// Sets the target size in bytes for a single preferred read region.
    ///
    /// Analogous to [`block_size_hint`](Self::block_size_hint) but for the preferred read shape.
    /// Defaults to the L2 cache size when not set.
    pub fn preferred_read_size_hint(&mut self, size_hint: u64) -> &mut Self {
        self.preferred_read_size_hint = Some(size_hint);
        self
    }

    /// Returns the preferred read size hint in bytes, or `None` if not set.
    pub fn get_preferred_read_size_hint(&self) -> Option<u64> {
        self.preferred_read_size_hint
    }

    /// Clears the preferred read size hint, reverting to the default (L2 cache size).
    pub fn clear_preferred_read_size_hint(&mut self) -> &mut Self {
        self.preferred_read_size_hint = None;
        self
    }

    /// Sets the encoder (compression) parameters used when writing blocks.
    ///
    /// Controls the codec (e.g. Zstd), compression level, and pre-compression filters (byte shuffle).
    /// Defaults `EncoderParams::default()` when not set.
    pub fn encoder_params(&mut self, encoder_params: EncoderParams) -> &mut Self {
        self.encoder_params = Some(encoder_params);
        self
    }

    /// Returns the encoder params, or `None` if not set.
    pub fn get_encoder_params(&self) -> Option<&EncoderParams> {
        self.encoder_params.as_ref()
    }

    /// Clears the encoder params, reverting to the default configuration.
    pub fn clear_encoder_params(&mut self) -> &mut Self {
        self.encoder_params = None;
        self
    }

    /// Sets the decoder (decompression) parameters used when reading blocks.
    pub fn decoder_params(&mut self, decoder_params: DecoderParams) -> &mut Self {
        self.decoder_params = Some(decoder_params);
        self
    }

    /// Returns the decoder params, or `None` if not set.
    pub fn get_decoder_params(&self) -> Option<&DecoderParams> {
        self.decoder_params.as_ref()
    }

    /// Clears the decoder params, reverting to the default.
    pub fn clear_decoder_params(&mut self) -> &mut Self {
        self.decoder_params = None;
        self
    }

    /// Fills in any unset fields in `self` from `array`'s storage params.
    ///
    /// Fields that are already set in `self` are not overwritten. This mirrors what
    /// [`Array::copy_with`] does internally, and is useful when building params that should
    /// inherit most settings from an existing array while overriding specific ones.
    ///
    /// # Example
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let source = Array::compact_array(&array![1i32, 2, 3, 4, 5, 6, 7, 8])?;
    ///
    /// // Override just the block shape; inherit codec params from `source`.
    /// let mut params = ArrayParams::new();
    /// params.block_shape(&[4]);
    /// params.override_from_array(&source);
    ///
    /// let copy = source.copy_with(params, &source.read_ctx())?;
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    pub fn override_from_array<S>(&mut self, array: &Array<S>)
    where
        S: ArrayStorage,
    {
        self.override_from_storage(array.storage());
    }

    pub(crate) fn override_from_storage(&mut self, storage: &impl ArrayStorage) {
        let spec = storage._spec();
        self.encoder_params
            .get_or_insert_with(|| spec.encoder_params.cloned().unwrap_or_default());
        self.decoder_params
            .get_or_insert_with(|| spec.decoder_params.cloned().unwrap_or_default());

        let blocks_layout = spec.blocks_layout;
        self.block_shape
            .get_or_insert_with(|| blocks_layout.block_shape_hint.clone());
        self.block_shape_tag
            .get_or_insert_with(|| blocks_layout.block_shape_tag.clone());
        self.block_size_hint
            .get_or_insert(blocks_layout.block_size_hint);
        self.preferred_read_shape
            .get_or_insert_with(|| blocks_layout.preferred_read_shape.clone());
        self.preferred_read_size_hint
            .get_or_insert(blocks_layout.preferred_read_size_hint);
    }

    pub(crate) fn tune(&mut self, shape: &[u64], dtype: &Dtype) -> Result<()> {
        let b_layout = BlocksLayout::new(
            self.block_shape.clone(),
            self.block_shape_tag.clone(),
            self.block_size_hint,
            self.preferred_read_shape.clone(),
            self.preferred_read_size_hint,
            shape,
            dtype.itemsize() as _,
        )?;
        self.block_shape = Some(b_layout.block_shape_hint);
        self.block_shape_tag = Some(b_layout.block_shape_tag);
        self.block_size_hint = Some(b_layout.block_size_hint);
        self.preferred_read_shape = Some(b_layout.preferred_read_shape);
        self.preferred_read_size_hint = Some(b_layout.preferred_read_size_hint);
        Ok(())
    }
}
