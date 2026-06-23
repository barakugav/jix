use std::marker::PhantomPinned;
use std::pin::Pin;

use crate::codec::{Codec, DecoderParams, EncoderParams, Filter};
use crate::dtype::{Dtype, Itemsize};
use crate::error::{check_ndim, ensure, Result};
use crate::storage::block::BlockSize;
use crate::util::{scale_read_shape, DimArray, Idx, IterExt, SendSyncPtr};
use crate::{dim_arr, Array, ArrayStorage, DimDyn, Dimension};

/// Parameters controlling the encoding/decoding configs of an [`Array`], and its block layout.
///
/// `ArrayParams` groups two independent sets of configuration:
///
/// - **Codec** - the compression configuration (codec, level, filter pipeline) applied when
///   encoding blocks for a new array, set via [`codec`](Self::codec), [`level`](Self::level), and
///   [`filters`](Self::filters). These affect the compression ratio and CPU usage of the codec, but
///   not the block layout.
///
/// - **Block layout** - the nd-block shape used to divide the array into blocks, each compressed
///   independently, and other related hints that are propagated through lazy view storage
///   operations. A good block layout is critical for performance, and should match the access
///   pattern of your workload.
///
/// # When are params applied?
///
/// - When a new array is constructed, such as via [`Array::compact_ndarray`]: the data is split into
///   blocks according to the block layout params, and each block is compressed using the encoder
///   params before being written to storage.
/// - When an array is accessed for read, such as via [`Array::to_ndarray`]: each compressed block
///   is decompressed using the decoder params. Sometimes readers of an array might want to read
///   smaller chunks of data, that is aligned to the block shape (or preferred read shape) to avoid
///   decompressing more data than necessary.
/// - When an array is copied, such as via [`Array::compact`] or [`Array::compact_with`]: a new compressed
///   array is constructed, inheriting any unset params from the source array's storage spec. When
///   the copied array is a compressed array (i.e. not a lazy view), the block shape and codec
///   params are preserved identically by default. Arrays with lazy view storage
///   (e.g. from `Add`, `Reshape`, etc.) may modify the params as best as it can, trying to preserve
///   user-specified params where possible, but it is an approximate heuristic.
///   Shape modifying operations (e.g. `Reshape`, `PermuteAxes`, etc.) are especially likely to
///   change the block layout params - consider passing explicit params to `compact_with` after these
///   ops, or verifying the resulting block layout is reasonable for your access pattern.
///
/// # Recommended usage
///
/// Use `ArrayParams::new()` (equivalent to `ArrayParams::default()`) for most cases - the
/// defaults select a block shape that fits in the L1 data cache using Zstd level 3 with byte
/// shuffling. For latency-sensitive workloads where you know the access pattern, set `block_shape`
/// explicitly and call `compact_with` instead of `compact` after shape-changing ops.
///
/// ```
/// use jix::{Array, ArrayParams};
///
/// // Construct an array with a specific block shape.
/// let data = ndarray::Array2::<f32>::zeros((1024, 1024));
/// let mut params = ArrayParams::new();
/// params.block_shape(&[64, 64]);
/// let za = Array::compact_ndarray_with(&data, params)?;
///
/// // After a shape-changing op, pin the block shape explicitly.
/// let mut out_params = ArrayParams::new();
/// out_params.block_shape(&[128, 128]);
/// let ctx = za.read_ctx();
/// let transposed = za.permute_axes(&[1, 0]).compact_with(out_params, &ctx)?;
/// # Ok::<(), jix::Error>(())
/// ```
#[derive(Clone, Default, Debug)]
pub struct ArrayParams {
    pub(crate) block_shape: Option<DimArray<BlockSize>>,
    pub(crate) block_shape_tag: Option<DimArray<BlockShapeTag>>,
    pub(crate) block_size: Option<u64>,
    pub(crate) read_size: Option<u64>,
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
        check_ndim::<DimDyn>(block_shape.len()).unwrap();
        self.block_shape = Some(DimArray::from_slice(block_shape).unwrap());
        self
    }

    /// Sets per-dimension tags that control how [`block_shape`](Self::block_shape) is scaled
    /// when the block shape is auto-computed during propagation.
    ///
    /// `tags` must have the same length as `block_shape`. Requires `block_shape` to also be set.
    /// See [`BlockShapeTag`] for the available options.
    pub fn block_shape_tag(&mut self, tags: &[BlockShapeTag]) -> &mut Self {
        check_ndim::<DimDyn>(tags.len()).unwrap();
        self.block_shape_tag = Some(DimArray::from_slice(tags).unwrap());
        self
    }

    /// Sets the target block size in bytes, used when auto-computing the block shape.
    ///
    /// When `block_shape` is not set, or when some dimensions are not [`BlockShapeTag::Fixed`],
    /// the auto-computation scales the block shape so that each block is approximately this many
    /// bytes.
    ///
    /// When not provided, defaults to `block_shape.product() * itemsize` (the block size in
    /// bytes), or the L1 data cache size if no block shape is given.
    pub fn block_size(&mut self, size_hint: u64) -> &mut Self {
        self.block_size = Some(size_hint);
        self
    }

    /// Sets the target size in bytes for a single preferred read region.
    ///
    /// Defaults to the L1 cache size when not set explicitly.
    pub fn read_size(&mut self, size_hint: u64) -> &mut Self {
        self.read_size = Some(size_hint);
        self
    }

    /// Sets the compression codec used when writing blocks.
    ///
    /// Defaults to [`Codec::Zstd`] when not set. Setting any codec field materializes the codec
    /// configuration to its defaults, then applies this override.
    pub fn codec(&mut self, codec: Codec) -> &mut Self {
        self.encoder_params
            .get_or_insert_with(EncoderParams::default)
            .codec(codec);
        self
    }

    /// Sets the compression level (0-19).
    ///
    /// Higher levels trade CPU time for a better compression ratio. For Zstd, level 3 is the
    /// default.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `level` is out of the valid range (0-19 for Zstd).
    pub fn level(&mut self, level: u32) -> Result<&mut Self> {
        self.encoder_params
            .get_or_insert_with(EncoderParams::default)
            .level(level)?;
        Ok(self)
    }

    /// Sets the pre-compression filter pipeline (up to 4 filters).
    ///
    /// Filters are applied in order before compression and reversed after decompression. For most
    /// numeric dtypes [`Filter::ByteShuffle`] (the default) gives a good ratio improvement at low
    /// cost; [`Filter::BitShuffle`] can do better on low-entropy data at higher CPU cost. Pass an
    /// empty slice to disable filtering.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `filters` contains more than 4 elements.
    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self> {
        self.encoder_params
            .get_or_insert_with(EncoderParams::default)
            .filters(filters)?;
        Ok(self)
    }

    /// Fills in any unset fields in `self` from `array`'s storage params.
    ///
    /// Fields that are already set in `self` are not overwritten. This mirrors what
    /// [`Array::compact_with`] does internally, and is useful when building params that should
    /// inherit most settings from an existing array while overriding specific ones.
    ///
    /// # Example
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let source = Array::compact_ndarray(&array![1i32, 2, 3, 4, 5, 6, 7, 8])?;
    ///
    /// // Override just the block shape; inherit codec params from `source`.
    /// let mut params = ArrayParams::new();
    /// params.block_shape(&[4]);
    /// params.override_from_array(&source);
    ///
    /// let copy = source.compact_with(params, &source.read_ctx())?;
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn override_from_array<S>(&mut self, array: &Array<S>)
    where
        S: ArrayStorage,
    {
        self.override_from_storage(array.storage());
    }

    pub(crate) fn override_from_storage(&mut self, storage: &impl ArrayStorage) {
        let spec = storage.spec();
        self.encoder_params
            .get_or_insert_with(|| spec.encoder_params().clone());
        self.decoder_params
            .get_or_insert_with(|| spec.decoder_params().clone());

        self.block_shape
            .get_or_insert_with(|| spec.block_shape().clone());
        self.block_shape_tag
            .get_or_insert_with(|| spec.block_shape_tag().clone());
        self.block_size.get_or_insert(spec.block_size());
        self.read_size.get_or_insert(spec.read_size());
    }

    /// Compute and validate the block geometry for an array.
    ///
    /// Both the storage block shape and the preferred read block shape are resolved here;
    /// either can be supplied explicitly or left as `None` to be auto-computed from a
    /// target byte size.
    ///
    /// # Arguments
    ///
    /// - `block_shape` - shape of one storage block in items per dimension.
    ///   When `None`, a shape is chosen automatically so that each block is approximately
    ///   `block_size` bytes.
    /// - `block_shape_tag` - per-dimension constraint on how the block shape may be scaled;
    ///   requires `block_shape` to also be provided. Defaults to all-[`BlockShapeTag::Fixed`].
    ///   See [`BlockShapeTag`] for the available options.
    /// - `block_size` - target block size in bytes used when auto-computing or scaling
    ///   the block shape. Defaults to the L1 data cache size when the shape is not fully
    ///   [`BlockShapeTag::Fixed`].
    /// - `read_size` - target size for the preferred read region in bytes.
    ///   Defaults to the L1 cache size.
    /// - `shape` - the array shape, used to clamp block dimensions that would exceed the array.
    /// - `itemsize` - bytes per array element.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if:
    /// - `block_shape_tag` is provided without `block_shape`
    /// - the length of `block_shape_tag` does not match `ndim`
    /// - `ndim` exceeds [`crate::NDIM_MAX`]
    pub(crate) fn tune(&mut self, shape: &[u64], dtype: &Dtype) -> Result<()> {
        let block_shape = self.block_shape.clone();
        let block_shape_tag = self.block_shape_tag.clone();
        let mut block_size = self.block_size;

        let ndim = shape.len();
        check_ndim::<DimDyn>(ndim)?;
        let itemsize = dtype.itemsize() as u64;

        let cache_sizes = crate::util::cpu_cache::cache_sizes();

        ensure!(
            block_shape_tag.is_none() || block_shape.is_some(),
            InvalidArgument,
            "block_shape_tag is specified but block_shape is not specified"
        );
        let block_shape_tag =
            block_shape_tag.unwrap_or_else(|| dim_arr(ndim, |_| BlockShapeTag::Fixed));
        ensure!(
            ndim == block_shape_tag.len(),
            InvalidArgument,
            "ndim does not match block_shape_tag length: expected {}, got {}",
            ndim,
            block_shape_tag.len()
        );
        let fixed_block_shape = block_shape_tag
            .iter()
            .all(|&tag| tag == BlockShapeTag::Fixed);
        // Compute block_size if not specified, and if it cant be computed from block_shape
        if block_size.is_none() && (block_shape.is_none() || !fixed_block_shape) {
            block_size = Some(cache_sizes.l1_data as u64);
        }
        // Compute block shape
        let mut block_shape = block_shape.unwrap_or_else(|| {
            Self::scale_block_shape(
                &dim_arr(ndim, |_| 1),
                &dim_arr(ndim, |_| true),
                block_size.unwrap() / itemsize,
                shape,
            )
        });
        ensure!(
            ndim == block_shape.len(),
            InvalidArgument,
            "ndim does not match block_shape length: expected {}, got {}",
            ndim,
            block_shape.len()
        );
        // Scale block_shape up to block_size
        if !fixed_block_shape {
            block_shape = Self::scale_block_shape(
                &dim_arr(ndim, |dim| match block_shape_tag[dim] {
                    BlockShapeTag::Fixed | BlockShapeTag::MultipleOf => block_shape[dim],
                    BlockShapeTag::Any => 1,
                }),
                &dim_arr(ndim, |dim| block_shape_tag[dim] != BlockShapeTag::Fixed),
                block_size.unwrap() / itemsize,
                shape,
            );
        }
        ensure!(
            block_shape
                .iter()
                .zip(shape)
                .all(|(&b, &s)| b > 0 && b as u64 <= s.max(1)),
            InvalidArgument,
            "block_shape {:?} is invalid for array shape {:?}",
            block_shape,
            shape
        );
        // Update block_size to block_shape.product() if it is not specified
        let block_size = block_size
            .unwrap_or_else(|| {
                block_shape.iter().map(|&b| b as u64).try_product().unwrap() * itemsize
            })
            .max(1);
        // read_size default to L1 cache size
        let read_size = self.read_size.unwrap_or(cache_sizes.l1_data as u64).max(1);

        self.block_shape = Some(block_shape);
        self.block_shape_tag = Some(block_shape_tag);
        self.block_size = Some(block_size);
        self.read_size = Some(read_size);
        Ok(())
    }

    fn scale_block_shape(
        block_shape: &[BlockSize],
        scale_dim: &[bool],
        block_size_max: u64,
        shape: &[u64],
    ) -> DimArray<BlockSize> {
        let ndim = shape.len();
        let mut scaled_block_shape = (0..ndim)
            .rev()
            .scan(1, |inner_block_volume, dim| {
                let mut block_len = block_shape[dim];
                if scale_dim[dim] {
                    block_len = Self::block_len_heuristic(
                        block_len,
                        shape[dim],
                        block_size_max,
                        *inner_block_volume,
                    )
                };
                *inner_block_volume *= block_len as u64;
                Some(block_len)
            })
            .collect::<DimArray<_>>();
        scaled_block_shape.reverse();
        scaled_block_shape
    }

    fn block_len_heuristic(
        base_block_len: BlockSize,
        dim_len: u64,
        max_volume: u64,
        inner_block_volume: u64,
    ) -> BlockSize {
        if dim_len <= 1 {
            return 1;
        }
        let base_block_len = base_block_len as u64;
        let max_block_len = (max_volume / inner_block_volume)
            .min(dim_len)
            .min(1 << 30)
            .floor_to_multiple(base_block_len)
            .max(1);
        let base_block_len = base_block_len.max(1).min(max_block_len);
        let block_len = if max_block_len == dim_len {
            dim_len
        } else {
            // multiple_of should a power of 2, on the order of dim_len//8
            let multiple_of = base_block_len
                * ((dim_len / (16 * base_block_len)) + 1)
                    .next_power_of_two()
                    .min(1 << 20);

            // Use the largest block length that is a multiple of multiple_of and require
            // less than 12.5% padding
            (1..=(max_block_len / multiple_of))
                .rev()
                .map(|m| m * multiple_of)
                .find(|&block_len| {
                    let padding = dim_len.ceil_to_multiple(block_len) - dim_len;
                    padding <= dim_len / 8
                })
                .unwrap_or(multiple_of)
        };
        debug_assert!(1 <= block_len && block_len <= dim_len);
        block_len.try_into().unwrap()
    }

    pub(crate) fn into_spec(self, shape: &[u64], dtype: &Dtype) -> Result<ArraySpecOwned> {
        let mut params = self;
        params.tune(shape, dtype)?;
        let spec = ArraySpecOwned::new(
            params.block_shape.unwrap(),
            params.block_shape_tag.unwrap(),
            params.block_size.unwrap(),
            params.read_size.unwrap(),
            params.encoder_params.unwrap_or_default(),
            params.decoder_params.unwrap_or_default(),
        );

        {
            let spec = spec.as_ref();
            let ndim = shape.len();
            assert_eq!(spec.block_shape().len(), ndim);
            assert!(spec
                .block_shape()
                .iter()
                .zip(shape)
                .all(|(&b, &s)| (0..=s.max(1)).contains(&(b as u64))));
            assert_eq!(spec.block_shape_tag().len(), ndim);
            assert!(spec.block_size() > 0);
            assert!(spec.read_size() > 0);
        }

        Ok(spec)
    }
}

/// Per-dimension tag describing how a block shape dimension may be automatically scaled
/// when a new array is constructed without an explicit block shape.
///
/// Users typically choose a block shape based on their access patterns, so `Fixed`
/// is the default - it preserves that choice in downstream arrays. Operations that
/// change the logical shape (reduction, broadcast, reshape, etc.) may tag affected
/// dimensions as `Any` or `MultipleOf` to let the heuristic freely pick a suitable
/// size for those dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum BlockShapeTag {
    /// The block size for this dimension is exactly the value in `block_shape_hint` and
    /// must not be changed. Used for most user-specified block shapes to preserve the
    /// user's intent.
    Fixed,
    /// The block size must be a multiple of the value in `block_shape_hint`, but may be
    /// scaled up to fit the target byte size. Used when an operation constrains the
    /// granularity without fixing the exact size.
    MultipleOf,
    /// The block size for this dimension can be freely chosen up to the target byte size.
    /// The value in `block_shape_hint` is ignored. Used when an operation makes the
    /// original block size irrelevant (e.g. a dimension added by broadcast).
    Any,
}

/// Internal specs of an array.
pub struct ArraySpec<'a> {
    shared: Pin<&'a (ArraySpecShared, PhantomPinned)>,
    block: &'a ArrayBlockSpec,
}
/// Owned version of [`ArraySpec`].
///
/// The structs holds two sets of parameters:
/// - "shared" parameters: these are parameters that an array allocated on the heap, and any views
///   derived from it hold a raw pointer to it, using [`ArraySpecPtr`].
/// - "block" parameters: these are parameters that are stored directly in the array struct. With
///   the intention that these parameters are more likely to be modified by view operations.
///
/// The idea behind this structure is to let views modify some of the parameters without having to
/// allocate a full `ArraySpec`. We could have used Rc/Arc, but Rc locks you from multithreading, and
/// Arc creates contention on cache lines between CPUs. Raw pointers are not safe, but views always
/// hold a reference to the source array, so they are guaranteed to be valid as long as the source
/// array is alive.
/// This resemble self referential structs.
#[derive(Clone)]
pub(crate) struct ArraySpecOwned {
    shared: Pin<Box<(ArraySpecShared, PhantomPinned)>>,
    block: ArrayBlockSpec,
}
/// See [`ArraySpecOwned`] docs.
#[derive(Clone)]
pub(crate) struct ArraySpecShared {
    block_size: u64,
    read_size: u64,
    encoder_params: EncoderParams,
    decoder_params: DecoderParams,
}
/// See [`ArraySpecOwned`] docs.
#[derive(Clone)]
pub(crate) struct ArrayBlockSpec {
    pub(crate) block_shape: DimArray<BlockSize>,
    pub(crate) block_shape_tag: DimArray<BlockShapeTag>,
}
impl ArraySpecOwned {
    pub(crate) fn new(
        block_shape: DimArray<BlockSize>,
        block_shape_tag: DimArray<BlockShapeTag>,
        block_size: u64,
        read_size: u64,
        encoder_params: EncoderParams,
        decoder_params: DecoderParams,
    ) -> Self {
        let shared = ArraySpecShared {
            block_size,
            read_size,
            encoder_params,
            decoder_params,
        };
        let block = ArrayBlockSpec {
            block_shape,
            block_shape_tag,
        };
        Self {
            shared: Box::pin((shared, PhantomPinned)),
            block,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> ArraySpec<'_> {
        ArraySpec {
            shared: self.shared.as_ref(),
            block: &self.block,
        }
    }
}
impl<'a> ArraySpec<'a> {
    #[inline]
    pub(crate) fn with_block_spec(&self, block: &'a ArrayBlockSpec) -> Self {
        Self {
            shared: self.shared,
            block,
        }
    }

    #[inline(always)]
    fn shared(&self) -> &'a ArraySpecShared {
        let inner = &self.shared.0;
        unsafe { std::mem::transmute::<&ArraySpecShared, &'a ArraySpecShared>(inner) }
    }
    #[inline(always)]
    fn block(&self) -> &'a ArrayBlockSpec {
        self.block
    }

    #[inline(always)]
    pub(crate) fn block_size(&self) -> u64 {
        self.shared().block_size
    }
    #[inline(always)]
    pub(crate) fn read_size(&self) -> u64 {
        self.shared().read_size
    }
    #[inline(always)]
    pub(crate) fn encoder_params(&self) -> &'a EncoderParams {
        &self.shared().encoder_params
    }
    #[inline(always)]
    pub(crate) fn decoder_params(&self) -> &'a DecoderParams {
        &self.shared().decoder_params
    }
    #[inline(always)]
    pub(crate) fn block_shape(&self) -> &'a DimArray<BlockSize> {
        &self.block().block_shape
    }
    #[inline(always)]
    pub(crate) fn block_shape_tag(&self) -> &'a DimArray<BlockShapeTag> {
        &self.block().block_shape_tag
    }

    pub(crate) fn read_shape_heuristic<D>(
        &self,
        total_read_shape: &[u64],
        shape: &[u64],
        itemsize: Itemsize,
    ) -> D
    where
        D: Dimension,
    {
        self.read_shape_heuristic_with_scale_order(
            total_read_shape,
            shape,
            itemsize,
            (0..total_read_shape.len()).rev(),
        )
    }

    pub(crate) fn read_shape_heuristic_with_scale_order<D>(
        &self,
        total_read_shape: &[u64],
        shape: &[u64],
        itemsize: Itemsize,
        scale_order: impl Iterator<Item = usize>,
    ) -> D
    where
        D: Dimension,
    {
        let block_shape = self.block_shape();
        let mut read_shape = D::from_fn(shape.len(), |dim| block_shape[dim] as u64);
        scale_read_shape(
            &mut read_shape,
            total_read_shape,
            shape,
            (self.read_size() / itemsize as u64).max(1),
            scale_order,
        );
        read_shape
    }
}

/// See [`ArraySpecOwned`] docs.
#[derive(Clone)]
pub(crate) struct ArraySpecPtr {
    shared: SendSyncPtr<(ArraySpecShared, PhantomPinned)>,
    block: ArrayBlockSpec,
}
impl ArraySpecPtr {
    pub(crate) fn new(spec: ArraySpec<'_>) -> Self {
        let shared = spec.shared.get_ref();
        let shared = unsafe { SendSyncPtr::new(shared) };
        Self {
            shared,
            block: spec.block.clone(),
        }
    }

    /// Returns a reference to the underlying `ArraySpec`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the source of this ArraySpecPtr is still alive and has not been
    /// modified in a way that would invalidate the reference.
    pub(crate) unsafe fn as_ref<'a>(
        &self,
        #[allow(unused_variables)] source_spec: impl FnOnce() -> ArraySpec<'a>,
    ) -> ArraySpec<'_> {
        #[cfg(debug_assertions)]
        {
            let source_spec = source_spec();
            let source_shared = source_spec.shared.get_ref();
            debug_assert!(
                std::ptr::eq(
                    self.shared.as_ptr(),
                    source_shared as *const (ArraySpecShared, PhantomPinned),
                ),
                "ArraySpecPtr::as_ref() called with a different source spec than the one used to create the pointer"
            );
        }

        ArraySpec {
            shared: unsafe { Pin::new_unchecked(&*self.shared.as_ptr()) },
            block: &self.block,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Array, ArrayParams};

    #[test]
    fn example() {
        // Construct an array with a specific block shape.
        let data = ndarray::Array2::<f32>::zeros((1024, 1024));
        let mut params = ArrayParams::new();
        params.block_shape(&[64, 64]);
        let za = Array::compact_ndarray_with(&data, params).unwrap();

        // After a shape-changing op, pin the block shape explicitly.
        let mut out_params = ArrayParams::new();
        out_params.block_shape(&[128, 128]);
        let ctx = za.read_ctx();
        let transposed = za
            .permute_axes(&[1, 0])
            .compact_with(out_params, &ctx)
            .unwrap();
        assert_eq!(transposed.shape(), &[1024, 1024]);
    }

    #[test]
    fn flattened_codec_methods_roundtrip() {
        use crate::codec::{Codec, Filter};

        let mut params = ArrayParams::new();
        // Unset by default -> getters return None.
        assert!(params.encoder_params.is_none());

        params.codec(Codec::Zstd);
        params.level(7).unwrap();
        params
            .filters(&[Filter::ByteShuffle, Filter::BitShuffle])
            .unwrap();

        let encoder_params = params.encoder_params.as_ref().unwrap();
        assert!(matches!(encoder_params.codec, Codec::Zstd));
        assert_eq!(encoder_params.level, 7);
        assert_eq!(encoder_params.filters.len(), 2);
        assert_eq!(encoder_params.filters.len(), 2);

        // Validation still propagates from the internal encoder configuration.
        assert!(params.level(99).is_err());
    }
}
