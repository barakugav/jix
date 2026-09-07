use std::marker::PhantomPinned;
use std::pin::Pin;

use crate::codec::{Codec, DecoderParams, EncoderParams, Filter};
use crate::dtype::{Dtype, Itemsize};
use crate::error::{check_dtype_size_nonzero, check_ndim, ensure, Result};
use crate::storage::block::BlockSize;
use crate::util::{scale_read_shape, DimArray, DimIdx, Idx, IterExt, ScaleWeight, SendSyncPtr};
use crate::{dim_arr, Array, ArrayStorage, DimBitmap, DimDyn, Dimension, SliceExt};

/// Target byte range for a single read region.
///
/// Stores a `(min, max)` range in bytes.
/// Used as a per-spec tuning hint that controls how much data a single read pass
/// pulls through memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadSize {
    pub(crate) min: u64,
    pub(crate) max: u64,
}

impl ReadSize {
    pub(crate) fn new(min: u64, max: u64) -> Self {
        let min = min.max(1);
        let max = max.max(min);
        Self { min, max }
    }

    /// The range expressed in element counts for `itemsize`, each floored at 1.
    #[inline]
    pub(crate) fn nitems(self, itemsize: Itemsize) -> (u64, u64) {
        debug_assert!(itemsize > 0, "itemsize must be non-zero");
        let itemsize = itemsize as u64;
        ((self.min / itemsize).max(1), (self.max / itemsize).max(1))
    }
}

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
/// defaults select a block shape automatically according to the CPU cache sizes using Zstd level 3
/// with byte shuffling. For latency-sensitive workloads where you know the access pattern, set `block_shape`
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
    pub(crate) block_shape_fixed_dims: Option<DimBitmap>,
    pub(crate) block_size: Option<u64>,
    pub(crate) read_size: Option<ReadSize>,
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
    /// Setting the block shape also marks every dimension as fixed unless
    /// [`block_shape_fixed_dims`](Self::block_shape_fixed_dims) is also used, meaning the shape will
    /// be preserved as-is if this `ArrayParams` is later used as a propagation source.
    pub fn block_shape(&mut self, block_shape: &[BlockSize]) -> &mut Self {
        check_ndim::<DimDyn>(block_shape.len()).unwrap();
        self.block_shape = Some(DimArray::from_slice(block_shape).unwrap());
        self
    }

    /// Sets, per dimension, whether that dimension of [`block_shape`](Self::block_shape) is
    /// fixed.
    ///
    /// A fixed dimension (`true`) keeps its exact block-shape length during any later
    /// auto-scaling (e.g. when a downstream operation recomputes the block shape). A dimension that
    /// is not fixed (`false`) may be freely resized to fit the target block size - it is used when
    /// an operation makes the original block size irrelevant (e.g. a broadcast or reduced
    /// dimension).
    ///
    /// `fixed` must have one entry per dimension (the same length as `block_shape`). Requires
    /// [`block_shape`](Self::block_shape) to also be set. When this is not set, the default depends
    /// on the block shape: an explicitly-set [`block_shape`](Self::block_shape) is all-fixed
    /// (preserved exactly), while an auto-computed block shape is all-non-fixed.
    pub fn block_shape_fixed_dims(&mut self, fixed: &[bool]) -> &mut Self {
        check_ndim::<DimDyn>(fixed.len()).unwrap();
        self.block_shape_fixed_dims = Some(fixed.iter().copied().collect());
        self
    }

    /// Sets the target block size in bytes, used when auto-computing the block shape.
    ///
    /// When `block_shape` is not set, or when some dimensions are not fixed, the
    /// auto-computation scales the block shape so that each block is approximately this many
    /// bytes.
    ///
    /// When not provided, defaults to `block_shape.product() * itemsize` (the block size in
    /// bytes), or a size chosen automatically according to the CPU cache sizes if no block shape
    /// is given.
    pub fn block_size(&mut self, size_hint: u64) -> &mut Self {
        self.block_size = Some(size_hint);
        self
    }

    /// Sets the target byte range for a single preferred read region as `(min, max)`.
    ///
    /// A read region is the rectangular slab the engine pulls and decompresses in one pass
    /// when materializing a lazy pipeline or a sub-region read. Its shape is derived by
    /// scaling the storage block shape toward this byte budget. The two bounds steer that
    /// scaling differently:
    ///
    /// - `max` is the *scale-down ceiling*: an oversized read shape is shrunk only until it
    ///   fits within `max`. Keeping `max` large lets reads stay big.
    /// - `min` is the *scale-up floor*: an undersized read shape is grown only up to `min`.
    ///   Reads already at or above `min` are left as the scale-down step produced them.
    ///
    /// The motivation is block-grid misalignment. When the source array's block shape differs
    /// from the output's, a read that straddles source-block boundaries forces whole-block
    /// decompression for a partial slice, and neighboring reads re-decompress the same block.
    /// Larger read regions span more blocks per call and dilute that wasted work (no alignment
    /// guarantee, but the waste shrinks as the region grows); the counter-pressure is cache
    /// residency, which the `max` ceiling bounds.
    ///
    /// When unset, the range is chosen automatically according to the CPU cache sizes.
    pub fn read_size(&mut self, size_hint: (u64, u64)) -> &mut Self {
        self.read_size = Some(ReadSize::new(size_hint.0, size_hint.1));
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

    /// Set the compression level.
    ///
    /// Higher levels trade CPU time for better compression ratios. For zstd, level 3 is the default.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `level` is not in the valid range.
    pub fn level(&mut self, level: i32) -> Result<&mut Self> {
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

        if self.block_shape.is_none() {
            self.block_shape = Some(spec.block_shape().clone());
            self.block_shape_fixed_dims = Some(spec.block_shape_fixed_dims());
        }
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
    /// - `block_shape_fixed_dims` - per-dimension bitmap of which block-shape dimensions are fixed
    ///   (must not be scaled); requires `block_shape` to also be provided. When `None`, an explicit
    ///   `block_shape` defaults to all-fixed (preserved exactly) and an auto-computed one to
    ///   all-non-fixed. See [`DimBitmap`].
    /// - `block_size` - target block size in bytes used when auto-computing or scaling
    ///   the block shape. Defaults to a size chosen automatically according to the CPU cache
    ///   sizes when the shape is not fully fixed.
    /// - `read_size` - target size for the preferred read region in bytes.
    ///   Defaults to a range chosen automatically according to the CPU cache sizes.
    /// - `shape` - the array shape, used to clamp block dimensions that would exceed the array.
    /// - `itemsize` - bytes per array element.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if:
    /// - `block_shape_fixed_dims` is provided without `block_shape`
    /// - the length of `block_shape_fixed_dims` does not match `ndim`
    /// - `ndim` exceeds [`NDIM_MAX`]
    pub(crate) fn tune(&mut self, shape: &[u64], dtype: &Dtype) -> Result<()> {
        let block_shape = self.block_shape.clone();
        let block_shape_fixed_dims = self.block_shape_fixed_dims;
        let mut block_size = self.block_size;

        let ndim = shape.len();
        check_ndim::<DimDyn>(ndim)?;
        let itemsize = dtype.itemsize() as u64;
        check_dtype_size_nonzero(dtype)?;

        let cache_sizes = crate::util::cpu_cache::cache_sizes();

        ensure!(
            block_shape_fixed_dims.is_none() || block_shape.is_some(),
            InvalidArgument,
            "block_shape_fixed_dims is specified but block_shape is not specified"
        );
        let block_shape_fixed_dims = block_shape_fixed_dims
            .unwrap_or_else(|| DimBitmap::filled(ndim, block_shape.is_some()));
        ensure!(
            ndim == block_shape_fixed_dims.len(),
            InvalidArgument,
            "ndim does not match block_shape_fixed_dims length: expected {}, got {}",
            ndim,
            block_shape_fixed_dims.len()
        );
        let fixed_block_shape = block_shape_fixed_dims.all();
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
                &dim_arr(ndim, |dim| {
                    if block_shape_fixed_dims.get(dim) {
                        block_shape[dim]
                    } else {
                        1
                    }
                }),
                &dim_arr(ndim, |dim| !block_shape_fixed_dims.get(dim)),
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
        // read_size defaults to a window derived from the cache sizes.
        let read_size = self.read_size.unwrap_or_else(|| {
            let l2 = cache_sizes.l2 as u64;
            let read_size_min =
                std::cmp::max(cache_sizes.l1_data as u64 / 2, l2 / 16).max(block_size);
            let read_size_max = l2 / 2;
            ReadSize::new(read_size_min, read_size_max)
        });

        self.block_shape = Some(block_shape);
        self.block_shape_fixed_dims = Some(block_shape_fixed_dims);
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
        let mut volume = block_shape.iter().map(|&b| b as u64).product::<u64>();
        let mut scaled_block_shape = (0..ndim)
            .rev()
            .map(|dim| {
                let mut block_len = block_shape[dim];
                if scale_dim[dim] {
                    block_len =
                        Self::block_len_heuristic(block_len, shape[dim], block_size_max, volume);
                    volume = volume / (block_shape[dim].max(1) as u64) * block_len as u64;
                };
                block_len
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
                * ((dim_len.min(max_block_len) / (16 * base_block_len)) + 1)
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

    pub(crate) fn into_spec(
        self,
        shape: &[u64],
        dtype: &Dtype,
        flags: ArraySpecFlags,
    ) -> Result<ArraySpecOwned> {
        let mut params = self;
        params.tune(shape, dtype)?;
        let spec = ArraySpecOwned::new(
            params.block_shape.unwrap(),
            params.block_shape_fixed_dims.unwrap(),
            params.block_size.unwrap(),
            params.read_size.unwrap(),
            params.encoder_params.unwrap_or_default(),
            params.decoder_params.unwrap_or_default(),
            flags,
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
            assert_eq!(spec.block_shape_fixed_dims().len(), ndim);
            assert!(spec.block_size() > 0);
            assert!(spec.read_size().min > 0);
        }

        Ok(spec)
    }
}

/// Internal specs of an array.
pub struct ArraySpec<'a> {
    shared: Pin<&'a (ArraySpecShared, PhantomPinned)>,
    dynamic: &'a ArraySpecDynamic,
    flags: ArraySpecFlags,
}
/// Owned version of [`ArraySpec`].
///
/// The structs holds two sets of parameters:
/// - "shared" parameters: these are parameters that an array allocated on the heap, and any views
///   derived from it hold a raw pointer to it, using [`ArraySpecPtr`].
/// - "dynamic" parameters: these are parameters that are stored directly in the array struct. With
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
    dynamic: ArraySpecDynamic,
    flags: ArraySpecFlags,
}
/// See [`ArraySpecOwned`] docs.
#[derive(Clone)]
pub(crate) struct ArraySpecShared {
    block_size: u64,
    read_size: ReadSize,
    encoder_params: EncoderParams,
    decoder_params: DecoderParams,
}
/// See [`ArraySpecOwned`] docs.
#[derive(Clone)]
pub(crate) struct ArraySpecDynamic {
    /// Per-dimension block length, in items.
    ///
    /// For **Compact**/**CompactMmap** storage this is the literal storage block shape (blocks are
    /// compressed independently at this granularity). For **every other** storage or lazy view it is
    /// the *minimum read shape that doesn't waste work*: the smallest per-dim read tile below which a
    /// read would redundantly re-read or recompute underlying data (e.g. a broadcast dim's whole
    /// length, so the single source element is read once rather than once per tile). It seeds
    /// [`scale_read_shape`](crate::util::scale_read_shape), which scales the read region down/up from
    /// it.
    pub(crate) block_shape: DimArray<BlockSize>,

    /// Per-dimension "fixed" flags for [`block_shape`](Self::block_shape).
    ///
    /// A fixed dim keeps its exact block length when a **new Compact array** is materialized
    /// (`ArrayParams::tune`/`scale_block_shape`); a non-fixed dim may be freely resized to fit the
    /// target block size. This flag only affects Compact materialization - it does *not* enter the
    /// read path (`scale_read_shape` ignores it).
    pub(crate) block_shape_fixed_dims: DimBitmap,

    /// Estimated cost of reading a single element from this array, ignoring any broadcasting or
    /// duplication (see [`combine_elementwise_hints`] and the read-hint design).
    pub(crate) element_cost: f32,

    /// Per-dimension coverage weight: the fraction of this array's per-element read cost that is
    /// *redone once per tile band* along that dim. See [`ScaleWeight`].
    ///
    /// [`NONE`](ScaleWeight::NONE) for a leaf (nothing is duplicated, so a wider tile saves no
    /// work), [`FULL`](ScaleWeight::FULL) for a dim the whole view is duplicated along
    /// ([`Broadcast`](crate::ops::Broadcast), [`Tile`](crate::ops::Tile)): there, a tile that does
    /// not cover the dim re-reads the underlying data once per band. Element-wise ops mix their
    /// inputs' weights by [`element_cost`](Self::element_cost), so an expensive duplicated operand
    /// dominates a cheap one.
    ///
    /// Purely a *coverage* quantity: it says nothing about storage granularity, which lives in
    /// [`block_shape`](Self::block_shape).
    pub(crate) read_shape_scale_weight: DimArray<ScaleWeight>,

    /// The order in which [`scale_read_shape`](crate::util::scale_read_shape) scales the dims: a
    /// permutation of `0..ndim` in **ascending coverage priority** - `[0]` is the dim least worth
    /// covering, the last entry the one most worth covering.
    ///
    /// The two scans run from opposite ends: the down-scan shrinks from the front, the up-scan
    /// grows from the back. So the *last* entry is the first dim to be grown and the last to be
    /// shrunk. For a C-contiguous leaf the order is `[0, 1, .., ndim-1]`, which grows the innermost
    /// dim first.
    ///
    /// It is [`read_shape_scale_weight`](Self::read_shape_scale_weight) sorted ascending, with
    /// [`read_layout_order`](Self::read_layout_order) as the (stable) tie-break - so it runs in the
    /// same direction as the layout order, and for an array with no duplication at all (every
    /// weight [`NONE`](ScaleWeight::NONE)) the two are equal. Cached here so a read does not have
    /// to sort.
    pub(crate) read_shape_scale_order: DimArray<DimIdx>,

    /// The memory layout that is cheapest to read this array into: a permutation of `0..ndim`,
    /// **outermost (largest stride) first**. C-order is `[0, 1, .., ndim-1]`, F-order is
    /// `[ndim-1, .., 0]`.
    ///
    /// A reader that allocates its own destination (e.g.
    /// [`to_ndarray_sub`](crate::Array::to_ndarray_sub), or pull-mode
    /// [`materialize_out_buf`](crate::storage::materialize_out_buf)) lays the buffer out in this
    /// order so the copy runs straight through instead of transposing. It is a pure hint: it never
    /// changes which values a read produces, only the strides of a freshly allocated destination.
    pub(crate) read_layout_order: DimArray<DimIdx>,
}
impl ArraySpecOwned {
    pub(crate) fn new(
        block_shape: DimArray<BlockSize>,
        block_shape_fixed_dims: DimBitmap,
        block_size: u64,
        read_size: ReadSize,
        encoder_params: EncoderParams,
        decoder_params: DecoderParams,
        flags: ArraySpecFlags,
    ) -> Self {
        let shared = ArraySpecShared {
            block_size,
            read_size,
            encoder_params,
            decoder_params,
        };
        let ndim = block_shape.len();
        let dynamic = ArraySpecDynamic::new(
            block_shape,
            block_shape_fixed_dims,
            1.0,
            dim_arr(ndim, |_| ScaleWeight::NONE),
            dim_arr(ndim, |i| i as DimIdx), // default to C-order
        );
        Self {
            shared: Box::pin((shared, PhantomPinned)),
            dynamic,
            flags,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> ArraySpec<'_> {
        ArraySpec {
            shared: self.shared.as_ref(),
            dynamic: &self.dynamic,
            flags: self.flags,
        }
    }

    pub(crate) fn dynamic_mut(&mut self) -> &mut ArraySpecDynamic {
        &mut self.dynamic
    }
}
impl<'a> ArraySpec<'a> {
    #[inline]
    pub(crate) fn with_dynamic_spec(self, dynamic: &'a ArraySpecDynamic) -> Self {
        Self { dynamic, ..self }
    }

    // #[inline(always)]
    // pub(crate) fn with_flags(self, flags: ArraySpecFlags) -> Self {
    //     Self { flags, ..self }
    // }

    #[inline(always)]
    pub(crate) fn map_flags(self, f: impl FnOnce(ArraySpecFlags) -> ArraySpecFlags) -> Self {
        Self {
            flags: f(self.flags),
            ..self
        }
    }

    pub(crate) fn with_cleared_flags(mut self) -> Self {
        self.flags = ArraySpecFlags::default();
        self
    }

    #[inline(always)]
    fn shared(&self) -> &'a ArraySpecShared {
        let inner = &self.shared.0;
        unsafe { std::mem::transmute::<&ArraySpecShared, &'a ArraySpecShared>(inner) }
    }
    #[inline(always)]
    pub(crate) fn dynamic(&self) -> &'a ArraySpecDynamic {
        self.dynamic
    }

    #[inline(always)]
    pub(crate) fn block_size(&self) -> u64 {
        self.shared().block_size
    }
    #[inline(always)]
    pub(crate) fn read_size(&self) -> ReadSize {
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
        &self.dynamic().block_shape
    }
    #[inline(always)]
    pub(crate) fn block_shape_fixed_dims(&self) -> DimBitmap {
        self.dynamic().block_shape_fixed_dims
    }
    #[inline(always)]
    pub(crate) fn element_cost(&self) -> f32 {
        self.dynamic().element_cost
    }
    #[inline(always)]
    pub(crate) fn read_shape_scale_weight(&self) -> &'a [ScaleWeight] {
        &self.dynamic().read_shape_scale_weight
    }
    #[inline(always)]
    pub(crate) fn read_shape_scale_order(&self) -> &'a DimArray<DimIdx> {
        &self.dynamic().read_shape_scale_order
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn read_layout_order(&self) -> &'a [DimIdx] {
        &self.dynamic().read_layout_order
    }

    // internal use only
    #[doc(hidden)]
    #[inline(always)]
    pub fn flags(&self) -> ArraySpecFlags {
        self.flags
    }

    pub(crate) fn read_shape_heuristic<D>(
        &self,
        max_shape: &[u64],
        shape: &[u64],
        itemsize: Itemsize,
    ) -> D
    where
        D: Dimension,
    {
        self.read_shape_scale_dims(max_shape, shape, self.read_size().nitems(itemsize), |_| {
            true
        })
    }

    /// Scale a read tile covering only the dims selected by `include`, to `target_nitems` (in
    /// items); dims not included are left at length 1. Seeds the included dims from `block_shape`
    /// and scales them in order by [`read_shape_scale_order`](ArraySpecDynamic::read_shape_scale_order).
    pub(crate) fn read_shape_scale_dims<D>(
        &self,
        max_shape: &[u64],
        array_shape: &[u64],
        target_nitems: (u64, u64),
        include: impl Fn(usize) -> bool,
    ) -> D
    where
        D: Dimension,
    {
        let block_shape = self.block_shape();
        let mut read_shape = D::from_fn(max_shape.len(), |dim| {
            if include(dim) {
                block_shape[dim] as u64
            } else {
                1
            }
        });
        let order = self.read_shape_scale_order();
        scale_read_shape(
            read_shape.as_mut_slice(),
            max_shape,
            array_shape,
            target_nitems,
            self.read_shape_scale_weight(),
            order.iter().map(|&d| d as usize).filter(|&d| include(d)),
        );
        read_shape
    }
}

impl ArraySpecDynamic {
    pub(crate) fn new(
        block_shape: DimArray<BlockSize>,
        block_shape_fixed_dims: DimBitmap,
        element_cost: f32,
        read_shape_scale_weight: DimArray<ScaleWeight>,
        read_layout_order: DimArray<DimIdx>,
    ) -> Self {
        debug_assert_eq!(read_shape_scale_weight.len(), block_shape.len());
        debug_assert_eq!(read_layout_order.len(), block_shape.len());
        let read_shape_scale_order =
            scale_order_from_weights(&read_shape_scale_weight, &read_layout_order);
        Self {
            block_shape,
            block_shape_fixed_dims,
            element_cost,
            read_shape_scale_weight,
            read_shape_scale_order,
            read_layout_order,
        }
    }

    pub(crate) fn add_elementwise_cost(&mut self, extra: f32) {
        let old_cost = self.element_cost;
        let new_cost = old_cost + extra;
        self.element_cost = new_cost;

        let dilution = if new_cost > 0.0 {
            (old_cost / new_cost) as f64
        } else {
            0.0
        };
        for weight in self.read_shape_scale_weight.as_mut_slice() {
            if !weight.is_none() {
                *weight = ScaleWeight::new(weight.f64() * dilution);
            }
        }
        self.read_shape_scale_order =
            scale_order_from_weights(&self.read_shape_scale_weight, &self.read_layout_order);
    }
}

pub(crate) fn scale_order_from_weights(
    weight: &[ScaleWeight],
    layout_order: &[DimIdx],
) -> DimArray<DimIdx> {
    let mut order = DimArray::from_slice(layout_order).unwrap();
    order.sort_by_key(|&d| weight[d as usize]);
    order
}

/// Insert `dim` into a `read_layout_order` when its position in the order carries no meaning.
pub(crate) fn read_layout_order_insert_dont_care_dim(order: &mut DimArray<DimIdx>, dim: usize) {
    debug_assert!(!order.contains(&(dim as DimIdx)));
    let pos = order
        .iter()
        .position(|&d| d as usize > dim)
        .unwrap_or(order.len());
    order.insert(pos, dim as DimIdx);
}

/// One input of a multi-input op's hint combine:
/// (element cost, per-dim coverage weight, layout order)
pub(crate) type HintInput<'a> = (f32, &'a [ScaleWeight], &'a [DimIdx]);

/// Combine the read hints of the inputs of a *selection* op - one where each output element reads
/// exactly one input (e.g. [`Concatenate`](crate::ops::Concatenate)/[`Stack`](crate::ops::Stack)).
///
/// `weights[i]` is the number of output elements taken from input `i` (any consistent unit - only
/// the ratios matter). A selection reads one input per output element, so both the cost and the
/// weights are the *weighted average* over the inputs, not a sum.
pub(crate) fn combine_select_hints(
    inputs: &[HintInput<'_>],
    weights: &[f64],
) -> (f32, DimArray<ScaleWeight>, DimArray<DimIdx>) {
    debug_assert_eq!(inputs.len(), weights.len());
    let total = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    combine_hints(inputs, |i| weights[i] / total)
}

/// Combine the read hints of the inputs of an element-wise op (same shape).
///
/// Every output element reads *every* input, so the costs add (weight 1 each).
pub(crate) fn combine_elementwise_hints(
    inputs: &[HintInput<'_>],
) -> (f32, DimArray<ScaleWeight>, DimArray<DimIdx>) {
    combine_hints(inputs, |_| 1.0)
}

fn combine_hints(
    inputs: &[HintInput<'_>],
    input_weight: impl Fn(usize) -> f64,
) -> (f32, DimArray<ScaleWeight>, DimArray<DimIdx>) {
    let ndim = inputs[0].1.len();
    let weighted_cost = |i: usize| input_weight(i) * inputs[i].0 as f64;
    let element_cost = (0..inputs.len()).map(weighted_cost).sum::<f64>() + 1.0;
    let weight = dim_arr(ndim, |d| {
        let duplicated = (0..inputs.len())
            .map(|i| weighted_cost(i) * inputs[i].1[d].f64())
            .sum::<f64>();
        ScaleWeight::new(duplicated / element_cost)
    });
    (element_cost as f32, weight, max_cost_layout_order(inputs))
}

fn max_cost_layout_order(inputs: &[HintInput<'_>]) -> DimArray<DimIdx> {
    let max_cost_index = inputs
        .iter()
        .map(|(cost, _, _)| cost)
        .enumerate()
        .max_by(|(i_a, a), (i_b, b)| a.partial_cmp(b).unwrap().then(i_b.cmp(i_a)))
        .unwrap()
        .0;
    inputs[max_cost_index].2.to_dim_vec::<DimDyn>()
}

/// Combine the block layout (`block_shape` + `block_shape_fixed_dims`) of several equal-ndim inputs
/// of an element-wise or selection op (binary, `where`, map-multiple, concatenate, stack).
pub(crate) fn combine_block_layout(
    inputs: &[(&[BlockSize], DimBitmap)],
) -> (DimArray<BlockSize>, DimBitmap) {
    let ndim = inputs[0].0.len();
    let block_shape = dim_arr(ndim, |d| {
        inputs.iter().map(|&(bs, _)| bs[d]).max().unwrap_or(1)
    });
    let block_shape_fixed_dims = (0..ndim)
        .map(|d| {
            let all_equal = inputs.iter().all(|&(bs, _)| bs[d] == block_shape[d]);
            let any_fixed = inputs.iter().any(|&(_, fixed)| fixed.get(d));
            all_equal && any_fixed
        })
        .collect::<DimBitmap>();
    (block_shape, block_shape_fixed_dims)
}

/// See [`ArraySpecOwned`] docs.
#[derive(Clone)]
pub(crate) struct ArraySpecPtr {
    shared: SendSyncPtr<(ArraySpecShared, PhantomPinned)>,
    dynamic: ArraySpecDynamic,
    flags: ArraySpecFlags,
}
impl ArraySpecPtr {
    pub(crate) fn new(spec: ArraySpec<'_>) -> Self {
        let shared = spec.shared.get_ref();
        let shared = unsafe { SendSyncPtr::new(shared) };
        Self {
            shared,
            dynamic: spec.dynamic.clone(),
            flags: spec.flags,
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
            dynamic: &self.dynamic,
            flags: self.flags,
        }
    }
}

pub(crate) use flags::ArraySpecFlags;
pub(crate) mod flags {
    #[doc(hidden)]
    #[derive(Clone, Copy, Default)]
    pub struct ArraySpecFlags(u8);

    /// The array is stored in a compact layout, divided into nd-blocks, each compressed independently.
    const IS_COMPACT: u8 = 0b0000_0001;
    /// The array read operation is a simple strided copy.
    ///
    /// This is a hint to the caller, useful for Plain, Scalar, insert_axis, remove_axis, etc.
    const PLAIN_READ: u8 = 0b0000_0010;

    #[allow(dead_code)]
    impl ArraySpecFlags {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        #[doc(hidden)]
        pub fn is_compact(&self) -> bool {
            self.0 & IS_COMPACT != 0
        }
        pub(crate) fn set_compact(mut self) -> Self {
            self.0 |= IS_COMPACT;
            self
        }
        pub(crate) fn clear_compact(mut self) -> Self {
            self.0 &= !IS_COMPACT;
            self
        }

        #[doc(hidden)]
        pub fn plain_read(&self) -> bool {
            self.0 & PLAIN_READ != 0
        }
        pub(crate) fn set_plain_read(mut self) -> Self {
            self.0 |= PLAIN_READ;
            self
        }
        pub(crate) fn clear_plain_read(mut self) -> Self {
            self.0 &= !PLAIN_READ;
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::params::{
        combine_block_layout, combine_elementwise_hints, combine_select_hints, ArraySpecFlags,
        DimBitmap, ReadSize,
    };
    use ndarray::ShapeBuilder;

    use super::scale_order_from_weights;
    use crate::util::ScaleWeight;
    use crate::{Array, ArrayParams, ArrayStorage};

    fn bits(bm: DimBitmap) -> Vec<bool> {
        bm.into_iter().collect()
    }

    /// Assert a weight equals `fraction`, up to the quantization step of [`ScaleWeight`].
    #[track_caller]
    fn assert_weight(weight: ScaleWeight, fraction: f64) {
        let step = 1.0 / u16::MAX as f64;
        let diff = (weight.f64() - fraction).abs();
        assert!(diff <= step, "{weight:?} is not {fraction} (diff {diff})");
    }

    #[test]
    fn combine_elementwise_hints_weights_by_cost() {
        // Both operands are fully duplicated along one dim, but along different dims. The weights
        // are carried across as absolute duplicated cost and renormalized, so the expensive
        // operand's dim ends up worth far more - and it supplies the layout order too.
        let cheap_layout = [0u8, 1];
        let costly_layout = [1u8, 0];
        let full = [ScaleWeight::FULL, ScaleWeight::NONE];
        let (ec, weight, layout) = combine_elementwise_hints(&[
            (1.0, &full, &cheap_layout),
            (1000.0, &[full[1], full[0]], &costly_layout),
        ]);
        assert_eq!(ec, 1.0 + 1000.0 + 1.0);
        assert_weight(weight[0], 1.0 / 1002.0);
        assert_weight(weight[1], 1000.0 / 1002.0);
        assert_eq!(layout.as_slice(), &costly_layout);
        // The derived order runs lowest-weight first, so the costly operand's dim is scaled first.
        assert_eq!(
            scale_order_from_weights(&weight, &costly_layout).as_slice(),
            &[0, 1]
        );
    }

    #[test]
    fn combine_select_hints_weights_by_output_share() {
        // Each output element reads exactly one input, so cost and weight are the *weighted
        // average* over the inputs - here the costly input supplies three quarters of the output.
        let layout = [0u8, 1];
        let full = [ScaleWeight::FULL, ScaleWeight::NONE];
        let (ec, weight, _) = combine_select_hints(
            &[
                (1.0, &full, &layout),
                (1000.0, &[full[1], full[0]], &layout),
            ],
            &[1.0, 3.0],
        );
        let expected_ec = 0.25 * 1.0 + 0.75 * 1000.0 + 1.0;
        assert_eq!(ec, expected_ec as f32);
        assert_weight(weight[0], 0.25 / expected_ec);
        assert_weight(weight[1], (0.75 * 1000.0) / expected_ec);
    }

    #[test]
    fn max_cost_layout_order_breaks_ties_toward_the_first_input() {
        let first = [1u8, 0];
        let second = [0u8, 1];
        let none = [ScaleWeight::NONE; 2];
        let (_, _, layout) =
            combine_elementwise_hints(&[(8.0, &none, &first), (8.0, &none, &second)]);
        assert_eq!(layout.as_slice(), &first);
    }

    #[test]
    fn scale_weight_round_trips_and_keeps_a_positive_fraction_positive() {
        assert_eq!(ScaleWeight::new(0.0), ScaleWeight::NONE);
        assert_eq!(ScaleWeight::new(-1.0), ScaleWeight::NONE);
        assert_eq!(ScaleWeight::new(1.0), ScaleWeight::FULL);
        assert_eq!(ScaleWeight::new(2.0), ScaleWeight::FULL);
        assert_eq!(ScaleWeight::NONE.f64(), 0.0);
        assert_eq!(ScaleWeight::FULL.f64(), 1.0);
        for fraction in [1e-9, 1e-5, 0.25, 0.47, 0.999] {
            let w = ScaleWeight::new(fraction);
            assert!(!w.is_none(), "a positive fraction must not quantize away");
            assert_weight(w, fraction.max(1.0 / u16::MAX as f64));
        }
        // Ordering the u16 directly orders the fractions, which is what the scale order relies on.
        assert!(ScaleWeight::new(0.1) < ScaleWeight::new(0.2));
    }

    #[test]
    fn a_unary_op_dilutes_the_scale_weight() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap(),
        )
        .unwrap();
        // A compact leaf reads at cost 8; broadcasting dim 1 duplicates the whole of it.
        let bc = a.sum(1).insert_axis(1).broadcast(&[3, 4]);
        let cost = bc.storage().spec().element_cost();
        assert_eq!(
            bc.storage().spec().read_shape_scale_weight(),
            &[ScaleWeight::NONE, ScaleWeight::FULL]
        );

        // Each unary op adds one unit of work per *output* element, which a tile boundary never
        // makes anyone redo - so the duplicated share of the total shrinks.
        let once = std::ops::Neg::neg(bc.view());
        assert_eq!(once.storage().spec().element_cost(), cost + 1.0);
        assert_weight(
            once.storage().spec().read_shape_scale_weight()[1],
            (cost / (cost + 1.0)) as f64,
        );

        let twice = std::ops::Neg::neg(once.view());
        assert_eq!(twice.storage().spec().element_cost(), cost + 2.0);
        assert_weight(
            twice.storage().spec().read_shape_scale_weight()[1],
            (cost / (cost + 2.0)) as f64,
        );

        // Diluting never claims a dim stopped being duplicated, and never invents duplication.
        assert!(!twice.storage().spec().read_shape_scale_weight()[1].is_none());
        assert_eq!(
            twice.storage().spec().read_shape_scale_weight()[0],
            ScaleWeight::NONE
        );
        // The order is a pure function of the weights, so it survives the rescale.
        assert_eq!(
            twice.storage().spec().read_shape_scale_order().as_slice(),
            bc.storage().spec().read_shape_scale_order().as_slice()
        );
    }

    #[test]
    fn scale_order_is_the_layout_order_when_nothing_is_duplicated() {
        // With no duplication every weight ties, and the stable sort leaves the layout order
        // intact - which is what keeps a strided leaf's contiguity preference.
        for layout in [[0u8, 1, 2], [2, 0, 1], [1, 2, 0]] {
            assert_eq!(
                scale_order_from_weights(&[ScaleWeight::NONE; 3], &layout).as_slice(),
                &layout
            );
        }
    }

    #[test]
    fn read_hints_propagate_through_reduction_and_broadcast() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap(),
        )
        .unwrap();
        // A compact leaf reads at element_cost 8, with C-order scaling (inner dim first).
        {
            let sp = a.storage().spec();
            assert_eq!(sp.element_cost(), 8.0);
            assert_eq!(sp.read_shape_scale_weight(), &[ScaleWeight::NONE; 2]);
            assert_eq!(sp.read_shape_scale_order().as_slice(), &[0, 1]);
        }
        // Reduce over axis 1 (extent 4), re-insert the axis, and broadcast back to [3, 4].
        let bc = a.sum(1).insert_axis(1).broadcast(&[3, 4]);
        let sp = bc.storage().spec();
        // Reduction folds the whole reduced extent per output: 8 * (4 + 4) = 64.
        assert_eq!(sp.element_cost(), 64.0);
        // The broadcast dim (1) is the one worth covering, so it scales first - last in the order.
        assert_eq!(
            sp.read_shape_scale_weight(),
            &[ScaleWeight::NONE, ScaleWeight::FULL]
        );
        assert_eq!(
            *sp.read_shape_scale_order().last().unwrap(),
            1,
            "broadcast dim should scale first, got {:?}",
            sp.read_shape_scale_order()
        );
    }

    #[test]
    fn read_hints_combine_in_binary_op() {
        let plain = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i64).collect()).unwrap(),
        )
        .unwrap();
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap(),
        )
        .unwrap();
        // `sum` promotes i32 -> i64, matching `plain`'s dtype for the binary op.
        let bc = a.sum(1).insert_axis(1).broadcast(&[3, 4]);
        let out = plain.maximum(bc);
        let sp = out.storage().spec();
        // element_cost = plain (8) + broadcasted reduction (64) + 1.
        assert_eq!(sp.element_cost(), 8.0 + 64.0 + 1.0);
        // The costlier operand (the broadcasted reduction) dominates the gain: its broadcast dim
        // is scaled first, i.e. last in the order.
        assert!(sp.read_shape_scale_weight()[1] > sp.read_shape_scale_weight()[0]);
        assert_eq!(*sp.read_shape_scale_order().last().unwrap(), 1);
    }

    #[test]
    fn read_shape_scale_order_is_layout_order_for_compact_and_follows_gain_for_views() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4, 5], (0..60i32).collect()).unwrap(),
        )
        .unwrap();
        // A compact leaf duplicates nothing, so the order is just its layout order.
        assert_eq!(
            a.storage().spec().read_shape_scale_order().as_slice(),
            a.storage().spec().read_layout_order()
        );
        assert_eq!(
            a.storage().spec().read_shape_scale_order().as_slice(),
            &[0, 1, 2]
        );

        // A broadcast makes its dim the one worth covering: highest gain, so it sorts last and is
        // grown first.
        let bc = a.sum(1).insert_axis(1).broadcast(&[3, 4, 5]);
        let order = bc.storage().spec().read_shape_scale_order();
        assert_eq!(
            order.as_slice(),
            &[0, 2, 1],
            "broadcast dim should sort last, got {order:?}"
        );
    }

    #[test]
    fn read_layout_order_is_c_order_for_a_compact_leaf() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4, 5], (0..60i32).collect()).unwrap(),
        )
        .unwrap();
        // Compact blocks are stored C-order, so the outermost-first layout order is [0, 1, 2].
        assert_eq!(a.storage().spec().read_layout_order(), &[0, 1, 2]);
    }

    #[test]
    fn read_layout_order_follows_strides_for_a_plain_leaf() {
        // An F-order ndarray: dim 0 has the smallest stride, so dim 1 is the outermost axis.
        let arr =
            ndarray::Array::from_shape_vec([3, 4].f(), (0..12i32).collect::<Vec<_>>()).unwrap();
        let a = Array::plain_ndarray(arr).unwrap();
        assert_eq!(a.storage().spec().read_layout_order(), &[1, 0]);
    }

    #[test]
    fn read_layout_order_keeps_a_size_one_dim_at_its_shape_position() {
        // C-order with a size-1 axis stays exactly C-order, so readers' "already C-order" fast
        // path keeps firing.
        let c = ndarray::Array::from_shape_vec([4, 1, 5], (0..20i32).collect::<Vec<_>>()).unwrap();
        let a = Array::plain_ndarray(c).unwrap();
        assert_eq!(a.storage().spec().read_layout_order(), &[0, 1, 2]);

        // F-order [4, 5, 1]: the size-1 axis carries the largest stride (20 items) but is never
        // stepped, so it must not be mistaken for the outermost axis.
        let f =
            ndarray::Array::from_shape_vec([4, 5, 1].f(), (0..20i32).collect::<Vec<_>>()).unwrap();
        let a = Array::plain_ndarray(f).unwrap();
        assert_eq!(a.storage().spec().read_layout_order(), &[1, 0, 2]);
    }

    #[test]
    fn read_layout_order_puts_a_zero_stride_dim_outermost() {
        // A broadcast ndarray view has stride 0 on the expanded axis. Outermost is the only
        // placement that leaves the copy's inner run contiguous for both operands.
        let base =
            ndarray::Array::from_shape_vec([4, 1, 5], (0..20i32).collect::<Vec<_>>()).unwrap();
        let view = base.broadcast([4, 3, 5]).unwrap();
        let a = Array::plain_ndarray_view(view).unwrap();
        assert_eq!(a.storage().spec().read_layout_order(), &[1, 0, 2]);
    }

    fn compact_i32(
        shape: [usize; 3],
    ) -> Array<crate::storage::Compact<crate::Ty<i32>, crate::Dim<3>>> {
        let n = shape.iter().product::<usize>() as i32;
        Array::compact_ndarray(
            &ndarray::Array::from_shape_vec(shape, (0..n).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn read_layout_order_reindexes_through_permute_axes() {
        let a = compact_i32([3, 4, 5]);
        // Reversing the axes reverses the layout: the inner dim (2) becomes the outermost output
        // dim (0), so the outermost-first order is [2, 1, 0].
        let t = a.view().permute_axes(&[2, 1, 0]);
        assert_eq!(t.storage().spec().read_layout_order(), &[2, 1, 0]);
        // A rotation: output dim i reads input axis axes[i], so input dim 0 (outermost) lands on
        // output dim 1, input dim 1 on output dim 2, and input dim 2 on output dim 0.
        let t = a.view().permute_axes(&[2, 0, 1]);
        assert_eq!(t.storage().spec().read_layout_order(), &[1, 2, 0]);
    }

    #[test]
    fn read_layout_order_drops_reduced_dims() {
        let a = compact_i32([3, 4, 5]);
        // Layout [2, 1, 0]; reducing dim 1 leaves dims 2 and 0 in that relative order, renumbered
        // to the output's [1, 0].
        let r = a.view().permute_axes(&[2, 1, 0]).sum(1);
        assert_eq!(r.storage().spec().read_layout_order(), &[1, 0]);
    }

    #[test]
    fn read_layout_order_drops_removed_dims() {
        let a = compact_i32([3, 1, 5]);
        // Layout [2, 1, 0]; removing the size-1 dim 1 leaves dims 2 and 0, renumbered to [1, 0].
        let r = a.view().permute_axes(&[2, 1, 0]).remove_axis(1);
        assert_eq!(r.storage().spec().read_layout_order(), &[1, 0]);
    }

    #[test]
    fn read_layout_order_splices_an_inserted_axis_at_its_shape_position() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap();
        // A C-order input must stay exactly C-order so the readers' fast path keeps firing.
        let i = a.view().insert_axis(1);
        assert_eq!(i.storage().spec().read_layout_order(), &[0, 1, 2]);
        // Transposed: old dims 0, 1 map to new 0, 2 and keep their [1, 0] relative order as
        // [2, 0]; the new axis splices in ahead of the first higher-numbered dim.
        let i = a.view().permute_axes(&[1, 0]).insert_axis(1);
        assert_eq!(i.storage().spec().read_layout_order(), &[1, 2, 0]);
    }

    #[test]
    fn read_layout_order_survives_a_one_to_one_reshape_only() {
        let a = compact_i32([3, 4, 5]);
        // Every output dim maps 1:1 to a source dim, so the layout carries through.
        let r = a.view().permute_axes(&[2, 1, 0]).reshape([5, 4, 3]);
        assert_eq!(r.storage().spec().read_layout_order(), &[2, 1, 0]);
        // Merging dims 0 and 1 has no honest layout - a reshape is defined on C-order flattening -
        // so the output falls back to C-order.
        let r = a.view().permute_axes(&[2, 1, 0]).reshape([20, 3]);
        assert_eq!(r.storage().spec().read_layout_order(), &[0, 1]);
    }

    #[test]
    fn read_layout_order_forwards_through_elementwise_and_broadcast() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 1], (0..3i32).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap();
        // A unary elementwise op only bumps element_cost; the layout passes straight through.
        let n = a.view().permute_axes(&[1, 0]).map(|x: i32| x * 2);
        assert_eq!(n.storage().spec().read_layout_order(), &[1, 0]);
        // Broadcast forwards too: a duplicated dim keeps its position.
        let b = a.view().permute_axes(&[1, 0]).broadcast(&[4, 3]);
        assert_eq!(b.storage().spec().read_layout_order(), &[1, 0]);
    }

    fn compact_2d_i32(
        shape: [usize; 2],
    ) -> Array<crate::storage::Compact<crate::Ty<i32>, crate::Dim<2>>> {
        let n = shape.iter().product::<usize>() as i32;
        Array::compact_ndarray(
            &ndarray::Array::from_shape_vec(shape, (0..n).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn read_layout_order_takes_the_costliest_operand_in_a_binary_op() {
        // A transposed compact leaf: F-order, and cheap at element_cost 8.
        let t = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i64).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap();
        // A broadcast reduction over [4, 3]: C-order, and expensive at 8 * (3 + 4) = 56.
        let src = compact_2d_i32([4, 3]);
        let costly = || src.view().sum(1).insert_axis(1).broadcast(&[4, 3]);
        assert_eq!(costly().storage().spec().element_cost(), 8.0 * (3.0 + 4.0));

        // The costlier operand supplies the layout, whichever side it sits on.
        let out = t.view().permute_axes(&[1, 0]).maximum(costly());
        assert_eq!(out.storage().spec().read_layout_order(), &[0, 1]);
        let out = costly().maximum(t.view().permute_axes(&[1, 0]));
        assert_eq!(out.storage().spec().read_layout_order(), &[0, 1]);
    }

    #[test]
    fn read_layout_order_takes_the_costliest_input_in_concatenate() {
        let c_order = compact_2d_i32([4, 3]);
        let src = compact_2d_i32([3, 4]);
        // Three maps over a transposed leaf: F-order at element_cost 8 + 3 = 11.
        let costly = src
            .view()
            .permute_axes(&[1, 0])
            .map(|x: i32| x)
            .map(|x: i32| x)
            .map(|x: i32| x);
        assert_eq!(costly.storage().spec().element_cost(), 11.0);

        // Input 1 is the costliest, so its F-order layout wins over input 0's C-order.
        let out = crate::ops::concatenate((c_order.view(), costly), 0);
        assert_eq!(out.storage().spec().read_layout_order(), &[1, 0]);
    }

    #[test]
    fn read_layout_order_takes_the_costliest_input_in_where() {
        let cond = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([4, 3], vec![true; 12]).unwrap(),
        )
        .unwrap();
        let c_order = compact_2d_i32([4, 3]);
        let src = compact_2d_i32([3, 4]);
        let costly = src
            .view()
            .permute_axes(&[1, 0])
            .map(|x: i32| x)
            .map(|x: i32| x)
            .map(|x: i32| x);

        // `y` is the costliest of the three, so its F-order layout wins - not `x`'s.
        let out = crate::ops::where_condition(cond.view(), c_order.view(), costly);
        assert_eq!(out.storage().spec().read_layout_order(), &[1, 0]);
    }

    #[test]
    fn read_layout_order_splices_stacks_new_axis_at_its_shape_position() {
        let a = compact_2d_i32([3, 4]);
        // Both inputs are transposed [4, 3] views with layout [1, 0]. Stacking at axis 1 relabels
        // them to [2, 0], then the new axis splices in ahead of the first higher-numbered dim.
        let out = crate::ops::stack(
            (
                a.view().permute_axes(&[1, 0]),
                a.view().permute_axes(&[1, 0]),
            ),
            1,
        );
        assert_eq!(out.storage().spec().read_layout_order(), &[1, 2, 0]);
    }

    #[test]
    fn combine_block_layout_takes_max_and_fixes_on_agreement() {
        let a_fixed: DimBitmap = [true, false, false].iter().copied().collect();
        let b_fixed: DimBitmap = [false, true, false].iter().copied().collect();
        // dim 0: equal len (4) and `a` fixed -> fixed. dim 1: differing len (2 vs 8) -> not fixed
        // even though `b` is fixed. dim 2: equal len (5) but neither fixed -> not fixed.
        let (bs, fx) =
            combine_block_layout(&[(&[4u32, 2, 5][..], a_fixed), (&[4u32, 8, 5][..], b_fixed)]);
        assert_eq!(bs.as_slice(), &[4, 8, 5]);
        assert_eq!(bits(fx), [true, false, false]);
    }

    #[test]
    fn binary_op_keeps_block_shape_a_granularity_and_moves_coverage_to_gain() {
        // A small explicit block on dim 1. The broadcast must NOT inflate it: `block_shape` is the
        // storage granularity, and a tile cut below it re-decodes every block once per band. The
        // "cover this dim" wish lives in `read_shape_scale_weight` instead.
        // The ops consume their receiver, so build two identical arrays.
        let mk = || {
            let mut params = ArrayParams::new();
            params.block_shape(&[3, 2]);
            Array::compact_ndarray_with(
                &ndarray::Array::from_shape_vec([3, 4], (0..12i64).collect()).unwrap(),
                params,
            )
            .unwrap()
        };
        let a = mk();
        assert_eq!(a.storage().spec().block_shape()[1], 2);

        // std-like: reduce axis 1, re-insert it, broadcast back.
        let bc = mk().sum(1).insert_axis(1).broadcast(&[3, 4]);
        {
            let sp = bc.storage().spec();
            // The reduction dropped dim 1 and `insert_axis` put back a length-1 dim, so the
            // granularity there really is 1 - and the broadcast leaves it alone instead of
            // inflating it to the full extent 4, which is what used to force the tile off the
            // block grid.
            assert_eq!(sp.block_shape()[1], 1, "broadcast forwards the granularity");
            assert_eq!(
                sp.read_shape_scale_weight(),
                &[ScaleWeight::NONE, ScaleWeight::FULL]
            );
        }

        let out = a.maximum(bc);
        let sp = out.storage().spec();
        assert_eq!(sp.block_shape().as_slice(), &[3, 2]);
        // dim 0: equal block (3) and `a` fixed -> fixed. dim 1: differing block (2 vs 1) -> not fixed.
        assert_eq!(bits(sp.block_shape_fixed_dims()), [true, false]);
        // The broadcasted operand is much the costlier (64 vs 8), so it dominates the combined
        // gain: 64 * 1.0 / (8 + 64 + 1).
        assert_eq!(sp.element_cost(), 8.0 + 64.0 + 1.0);
        assert_eq!(sp.read_shape_scale_weight()[0], ScaleWeight::NONE);
        assert_weight(sp.read_shape_scale_weight()[1], 64.0 / 73.0);
    }

    #[test]
    fn plain_leaf_read_shape_scale_order_follows_strides() {
        // C-contiguous: most contiguous (last) dim scales first, like a compact leaf - i.e. it
        // sorts last in the order.
        let c = Array::plain_ndarray(
            ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            c.storage().spec().read_shape_scale_order().as_slice(),
            &[0, 1]
        );

        // A size-1 dim reads the same regardless of coverage, so it scales last.
        let s = Array::plain_ndarray(
            ndarray::Array::from_shape_vec([3, 1], (0..3i32).collect()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            s.storage().spec().read_shape_scale_order().as_slice(),
            &[0, 1]
        );
    }

    #[test]
    fn dim_bitmap_filled_and_len() {
        let all = DimBitmap::filled(3, true);
        assert_eq!(all.len(), 3);
        assert_eq!(bits(all), [true, true, true]);

        let none = DimBitmap::filled(3, false);
        assert_eq!(none.len(), 3);
        assert_eq!(bits(none), [false, false, false]);

        assert_eq!(DimBitmap::filled(0, true).len(), 0);
        assert_eq!(DimBitmap::filled(8, true).len(), 8);
    }

    #[test]
    fn dim_bitmap_get_set() {
        let mut bm = DimBitmap::filled(3, false);
        assert!(!bm.get(0));
        bm.set(2, true);
        assert!(bm.get(2));
        assert!(!bm.get(1));
        bm.set(2, false);
        assert!(!bm.get(2));

        assert!(DimBitmap::filled(8, true).get(0));
        assert!(DimBitmap::filled(8, true).get(7));
    }

    #[test]
    fn dim_bitmap_all() {
        assert!(DimBitmap::filled(8, true).all());
        assert!(DimBitmap::filled(0, true).all());
        assert!(!DimBitmap::filled(1, false).all());

        let mut bm = DimBitmap::filled(2, false);
        bm.set(0, true);
        assert!(!bm.all());
        bm.set(1, true);
        assert!(bm.all());
    }

    #[test]
    fn dim_bitmap_insert() {
        // [fixed, free, fixed], insert a non-fixed dim at pos 1 -> [fixed, free(new), free, fixed]
        let mut bm = DimBitmap::filled(3, false);
        bm.set(0, true);
        bm.set(2, true);
        bm.insert(1, false);
        assert_eq!(bm.len(), 4);
        assert_eq!(bits(bm), [true, false, false, true]);

        // insert at the front shifts everything up.
        let mut bm = DimBitmap::filled(1, false);
        bm.set(0, true);
        bm.insert(0, false);
        assert_eq!(bits(bm), [false, true]);
    }

    #[test]
    fn dim_bitmap_from_into_iter_roundtrips() {
        let src = [true, false, true, false];
        let bm: DimBitmap = src.iter().copied().collect();
        assert_eq!(bm.len(), 4);
        assert_eq!(bits(bm), src);

        // Reorder/select via iterators (what the ops do): keep dims 0 and 2.
        let selected: DimBitmap = bm
            .into_iter()
            .enumerate()
            .filter_map(|(dim, c)| (dim == 0 || dim == 2).then_some(c))
            .collect();
        assert_eq!(bits(selected), [true, true]);
    }

    #[test]
    fn read_size_normalizes_and_converts() {
        // min floored at 1, max raised to at least min.
        assert_eq!(ReadSize::new(0, 0), ReadSize::new(1, 1));
        let rs = ReadSize::new(10, 4); // max < min -> max raised to min
        assert_eq!((rs.min, rs.max), (10, 10));

        let rs = ReadSize::new(32 * 1024, 256 * 1024);
        assert_eq!((rs.min, rs.max), (32 * 1024, 256 * 1024));

        // nitems divides by itemsize, flooring each at 1.
        let (min_n, max_n) = ReadSize::new(32, 256).nitems(4u16);
        assert_eq!((min_n, max_n), (8, 64));
        let (min_n, max_n) = ReadSize::new(2, 256).nitems(4u16); // 2/4 -> floored to 1
        assert_eq!((min_n, max_n), (1, 64));
    }

    #[test]
    fn block_shape_fixed_dims_controls_scaling() {
        use crate::dtype::Dtyped;

        // All dims fixed by default: the explicit block shape is preserved exactly.
        let mut params = ArrayParams::new();
        params.block_shape(&[4, 4]);
        let spec = params
            .into_spec(&[1024, 1024], &i32::DTYPE, ArraySpecFlags::default())
            .unwrap();
        assert_eq!(spec.as_ref().block_shape().as_slice(), &[4, 4]);
        assert!(spec.as_ref().block_shape_fixed_dims().all());

        // Release dim 1: it may grow to fill the block-size budget, while dim 0 stays pinned.
        let mut params = ArrayParams::new();
        params.block_shape(&[4, 4]);
        params.block_shape_fixed_dims(&[true, false]);
        params.block_size(4 * 1024 * i32::DTYPE.itemsize() as u64);
        let spec = params
            .into_spec(&[1024, 1024], &i32::DTYPE, ArraySpecFlags::default())
            .unwrap();
        let bs = spec.as_ref().block_shape();
        assert_eq!(bs[0], 4, "fixed dim 0 must be preserved");
        assert!(bs[1] > 4, "non-fixed dim 1 should scale up: {bs:?}");
        assert!(spec.as_ref().block_shape_fixed_dims().get(0));
        assert!(!spec.as_ref().block_shape_fixed_dims().get(1));
    }

    #[test]
    fn block_shape_fixed_dims_without_block_shape_errors() {
        use crate::dtype::Dtyped;
        let mut params = ArrayParams::new();
        params.block_shape_fixed_dims(&[false]);
        let result = params.into_spec(&[8], &i32::DTYPE, ArraySpecFlags::default());
        match result {
            Err(e) => assert!(matches!(e.kind(), crate::error::ErrorKind::InvalidArgument)),
            Ok(_) => panic!("expected InvalidArgument error"),
        }
    }

    #[test]
    fn block_shape_fixed_dims_length_mismatch_errors() {
        use crate::dtype::Dtyped;
        // A 3-element fixed-dims mask on a 2-D array is a length mismatch.
        let mut params = ArrayParams::new();
        params.block_shape(&[4, 4]);
        params.block_shape_fixed_dims(&[true, false, true]);
        let result = params.into_spec(&[1024, 1024], &i32::DTYPE, ArraySpecFlags::default());
        match result {
            Err(e) => assert!(matches!(e.kind(), crate::error::ErrorKind::InvalidArgument)),
            Ok(_) => panic!("expected InvalidArgument error"),
        }
    }

    #[test]
    fn example() {
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
    }

    #[test]
    fn read_size_defaults_to_cache_window() {
        use crate::dtype::Dtyped;
        use crate::util::cpu_cache::cache_sizes;
        let mut params = ArrayParams::new();
        params.block_shape(&[8]);
        let spec = params
            .into_spec(&[8], &i32::DTYPE, ArraySpecFlags::default())
            .unwrap();
        let rs = spec.as_ref().read_size();
        let cs = cache_sizes();
        let block_size = 8 * i32::DTYPE.itemsize() as u64;
        // When unset, the read-size window is derived automatically from the CPU cache sizes:
        // a valid, non-degenerate range no smaller than the block and bounded by the caches.
        assert!(
            rs.min >= block_size,
            "min {} < block_size {block_size}",
            rs.min
        );
        assert!(rs.min <= rs.max, "min {} > max {}", rs.min, rs.max);
        assert!(rs.max <= cs.l2 as u64, "max {} > l2 {}", rs.max, cs.l2);
    }

    #[test]
    fn read_size_setter_roundtrips_tuple() {
        use crate::dtype::Dtyped;
        let mut params = ArrayParams::new();
        params.block_shape(&[8]);
        params.read_size((4096, 65536));
        let spec = params
            .into_spec(&[8], &i32::DTYPE, ArraySpecFlags::default())
            .unwrap();
        let rs = spec.as_ref().read_size();
        assert_eq!((rs.min, rs.max), (4096, 65536));
    }

    // ---- block-shape scaling heuristics ----
    //
    // `scale_block_shape` chooses a per-dimension block length so that the total block volume
    // (product of the lengths) stays approximately within `block_size_max` items, filling that
    // budget greedily from the innermost (last) axis outward. `scale_dim[d] == false` marks a
    // fixed dim that must keep its input length; scaled dims are sized by `block_len_heuristic`
    // and clamped to `[1, shape[d]]`. Because the heuristics only approximate the target, the
    // budget assertions below check "not wildly over", not a strict ceiling.

    use crate::storage::block::BlockSize;

    fn sbs(
        block_shape: &[BlockSize],
        scale_dim: &[bool],
        block_size_max: u64,
        shape: &[u64],
    ) -> Vec<BlockSize> {
        ArrayParams::scale_block_shape(block_shape, scale_dim, block_size_max, shape).to_vec()
    }

    fn vol(block: &[BlockSize]) -> u64 {
        block.iter().map(|&b| b as u64).product()
    }

    #[test]
    fn scale_block_shape_output_len_matches_ndim() {
        assert_eq!(
            sbs(&[1, 1, 1], &[true, true, true], 1000, &[100, 100, 100]).len(),
            3
        );
        assert_eq!(sbs(&[4], &[false], 1000, &[100]).len(), 1);
    }

    #[test]
    fn scale_block_shape_preserves_fixed_dims() {
        // dims 0 and 2 are fixed and must survive unchanged; dim 1 is scaled.
        let out = sbs(&[7, 1, 3], &[false, true, false], 10_000, &[100, 100, 100]);
        assert_eq!(out[0], 7);
        assert_eq!(out[2], 3);
    }

    #[test]
    fn scale_block_shape_clamps_within_shape() {
        // Every chosen block length must land in [1, shape[dim]], scaled or fixed.
        for (bs, sd, budget, shape) in [
            (vec![1, 1], vec![true, true], 64u64, vec![50u64, 50]),
            (vec![1, 1, 1], vec![true, true, true], 1000, vec![7, 200, 3]),
            (vec![1], vec![true], 10, vec![1000]),
            (vec![1], vec![true], 1_000_000, vec![5]),
        ] {
            let out = sbs(&bs, &sd, budget, &shape);
            for (d, &b) in out.iter().enumerate() {
                assert!(b >= 1, "dim {d}: block len {b} < 1 for shape {shape:?}");
                assert!(
                    b as u64 <= shape[d].max(1),
                    "dim {d}: block len {b} exceeds shape {}",
                    shape[d]
                );
            }
        }
    }

    #[test]
    fn scale_block_shape_singleton_scaled_dim_is_one() {
        // A scaled dimension of extent 1 must collapse to a block length of 1.
        let out = sbs(&[1, 1], &[true, true], 1000, &[1, 500]);
        assert_eq!(out[0], 1);
        // A fully-degenerate shape stays all ones.
        assert_eq!(sbs(&[1, 1], &[true, true], 1000, &[1, 1]), vec![1, 1]);
    }

    #[test]
    fn scale_block_shape_fully_auto_stays_near_budget() {
        // The all-scaled path must not blow far past the budget. Also guards the `multiple_of`
        // cap fix (dim_len.min(max_block_len)) via a huge dim with a tiny budget.
        for (budget, shape) in [
            (64u64, vec![1_000_000u64]),
            (100, vec![1000]),
            (1000, vec![1000, 1000]),
            (256, vec![10_000, 10_000]),
            (4096, vec![64, 64, 64]),
        ] {
            let ndim = shape.len();
            let out = sbs(&vec![1; ndim], &vec![true; ndim], budget, &shape);
            assert!(
                vol(&out) <= budget * 8,
                "fully-auto volume {} >> budget {budget} for shape {shape:?}: {out:?}",
                vol(&out)
            );
        }
    }

    #[test]
    fn scale_block_shape_reserves_budget_for_outer_fixed_dim() {
        // Regression: an OUTER fixed dim (index 0, processed last) must still be reserved out of
        // the budget when an inner scaled dim is sized, or the block overshoots wildly.
        let out = sbs(&[50, 1], &[false, true], 100, &[1000, 1000]);
        assert_eq!(out[0], 50, "fixed dim must be preserved");
        assert!(
            vol(&out) <= 100 * 4,
            "outer fixed dim not reserved out of budget: volume {} for {out:?}",
            vol(&out)
        );
    }

    #[test]
    fn scale_block_shape_inner_fixed_dim_stays_near_budget() {
        // Symmetric case: an INNER fixed dim (last index, processed first) is already accounted
        // for even before the fix. Ensures the fix does not regress this direction.
        let out = sbs(&[1, 50], &[true, false], 100, &[1000, 1000]);
        assert_eq!(out[1], 50);
        assert!(vol(&out) <= 100 * 4, "volume {} for {out:?}", vol(&out));
    }

    #[test]
    fn scale_block_shape_scaled_dim_yields_to_fixed_dims() {
        // Two fixed dims already exceed the budget; the remaining scaled dim must shrink to 1
        // rather than greedily taking the whole budget on top of them.
        let out = sbs(&[40, 40, 1], &[false, false, true], 100, &[100, 100, 100]);
        assert_eq!(out[0], 40);
        assert_eq!(out[1], 40);
        assert_eq!(
            out[2], 1,
            "scaled dim should collapse to 1 under budget pressure: {out:?}"
        );
    }

    #[test]
    fn block_len_heuristic_returns_one_for_degenerate_dim() {
        assert_eq!(ArrayParams::block_len_heuristic(5, 1, 1000, 1), 1);
        assert_eq!(ArrayParams::block_len_heuristic(1, 0, 1000, 1), 1);
    }

    #[test]
    fn block_len_heuristic_never_exceeds_dim_len_or_budget() {
        // Result stays within the dimension extent and roughly within the per-dim budget
        // (max_volume / inner_block_volume). Guards the `multiple_of` cap fix.
        for (base, dim_len, max_volume, inner) in [
            (1u32, 1_000_000u64, 64u64, 1u64),
            (1, 1000, 100, 1),
            (1, 1000, 100, 50),
            (8, 500, 4096, 2),
            (1, 10, 1_000_000, 1),
        ] {
            let b = ArrayParams::block_len_heuristic(base, dim_len, max_volume, inner) as u64;
            assert!(b >= 1 && b <= dim_len, "block len {b} out of [1,{dim_len}]");
            let budget = (max_volume / inner).max(1);
            assert!(
                b <= budget * 4,
                "block len {b} >> per-dim budget {budget} (base={base}, dim_len={dim_len})"
            );
        }
    }

    #[test]
    fn block_len_heuristic_budget_shrinks_with_inner_volume() {
        // A larger inner-block volume leaves less budget, so the chosen length cannot grow.
        let big = ArrayParams::block_len_heuristic(1, 10_000, 4096, 1);
        let small = ArrayParams::block_len_heuristic(1, 10_000, 4096, 64);
        assert!(
            small <= big,
            "expected {small} <= {big} as inner volume grows"
        );
    }

    // ---- Zero-itemsize dtypes ----
    //
    // `Dtype` supports itemsize 0 (an empty struct, `[T; 0]`), but an array of such a dtype does
    // not: the block layout is derived by dividing a byte budget by the itemsize. `tune` is the
    // single gate every leaf storage passes through, so it is where the rejection belongs.

    fn zero_itemsize_dtype() -> crate::dtype::Dtype {
        let d = crate::dtype::Dtype::from_fields(vec![]).unwrap();
        assert_eq!(d.itemsize(), 0);
        d
    }

    #[test]
    fn tune_rejects_zero_itemsize() {
        let mut params = ArrayParams::new();
        let err = params.tune(&[8], &zero_itemsize_dtype()).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::UnsupportedDtype);
    }

    #[test]
    fn into_spec_rejects_zero_itemsize() {
        let err = ArrayParams::new()
            .into_spec(&[8], &zero_itemsize_dtype(), ArraySpecFlags::default())
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::UnsupportedDtype);
    }

    // The leaf storages, through their public entry points.

    #[test]
    fn compact_nd_ptr_rejects_zero_itemsize() {
        let buf = [0u8; 8];
        let result = unsafe {
            Array::<crate::storage::Compact<crate::TypeDyn, crate::Dim<1>>>::compact_nd_ptr(
                buf.as_ptr(),
                [4u64],
                &[0usize],
                zero_itemsize_dtype(),
                ArrayParams::new(),
            )
        };
        assert_eq!(
            result.unwrap_err().kind(),
            crate::ErrorKind::UnsupportedDtype
        );
    }

    #[test]
    fn plain_ndarray_ptr_rejects_zero_itemsize() {
        let buf = [0u8; 8];
        let result = unsafe {
            Array::<crate::storage::Plain<&(), crate::TypeDyn, crate::Dim<1>>>::plain_ndarray_ptr(
                buf.as_ptr(),
                [4u64],
                &[0usize],
                zero_itemsize_dtype(),
                ArrayParams::new(),
            )
        };
        assert_eq!(
            result.unwrap_err().kind(),
            crate::ErrorKind::UnsupportedDtype
        );
    }

    // `[i32; 0]` is a `Dtyped` type with itemsize 0, so it reaches `tune` through `T::DTYPE`
    // rather than through a caller-supplied runtime dtype.
    #[test]
    fn compact_fn_rejects_zero_itemsize_element() {
        let result =
            Array::<crate::storage::Compact<crate::Ty<[i32; 0]>, crate::Dim<1>>>::compact_fn(
                [4u64],
                |_| [],
            );
        assert_eq!(
            result.unwrap_err().kind(),
            crate::ErrorKind::UnsupportedDtype
        );
    }
}
