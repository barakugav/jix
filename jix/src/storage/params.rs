use std::hint::assert_unchecked;
use std::marker::PhantomPinned;
use std::pin::Pin;

use crate::codec::{Codec, DecoderParams, EncoderParams, Filter};
use crate::dtype::{Dtype, Itemsize};
use crate::error::{check_ndim, ensure, Result};
use crate::storage::block::BlockSize;
use crate::util::{scale_read_shape, DimArray, Idx, IterExt, SendSyncPtr};
use crate::{dim_arr, Array, ArrayStorage, DimDyn, Dimension, NDIM_MAX};

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

    /// Sets the compression level.
    ///
    /// Higher levels trade CPU time for a better compression ratio. For Zstd, level 3 is the
    /// default.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `level` is out of the valid range (0-19 for Zstd).
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

/// A compact, fixed-length set of per-dimension boolean flags.
///
/// Bit `d` (the low bit is dimension 0) stores one boolean for dimension `d`; callers decide what a
/// set bit means. The bitmap tracks its own dimension count ([`len`](Self::len)), so it builds (via
/// [`FromIterator`]), iterates (via [`IntoIterator`]), and compares as a fixed-length sequence of
/// booleans. Since the maximum number of dimensions is [`NDIM_MAX`] (which is 8), all flags fit in a
/// single `u8`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DimBitmap {
    bits: u8,
    len: u8,
}

impl DimBitmap {
    /// The low `n` bits set (and nothing above), used to mask off unused high bits.
    #[inline]
    fn low_mask(n: usize) -> u8 {
        debug_assert!(n <= NDIM_MAX);
        if n >= NDIM_MAX {
            u8::MAX
        } else {
            (1u8 << n) - 1
        }
    }

    /// A bitmap of `len` dimensions, every flag set to `value`.
    #[inline]
    pub(crate) fn filled(len: usize, value: bool) -> Self {
        assert!(len <= NDIM_MAX);
        Self {
            bits: if value { Self::low_mask(len) } else { 0 },
            len: len as u8,
        }
    }

    /// The number of dimensions the bitmap covers.
    #[inline]
    pub(crate) fn len(self) -> usize {
        let len = self.len as usize;
        unsafe { assert_unchecked(len <= NDIM_MAX) };
        len
    }

    /// Returns the flag for dimension `dim`.
    #[inline]
    pub(crate) fn get(self, dim: usize) -> bool {
        assert!(dim < self.len());
        self.bits & (1u8 << dim) != 0
    }

    /// Sets the flag for dimension `dim` to `value`.
    #[inline]
    pub(crate) fn set(&mut self, dim: usize, value: bool) {
        assert!(dim < self.len());
        let bit = 1u8 << dim;
        if value {
            self.bits |= bit;
        } else {
            self.bits &= !bit;
        }
    }

    /// Returns whether every dimension's flag is set.
    #[inline]
    pub(crate) fn all(self) -> bool {
        let mask = Self::low_mask(self.len());
        self.bits & mask == mask
    }

    /// Inserts a new dimension at position `pos`, shifting higher dimensions up by one, and
    /// grows the length by one. The inserted dimension takes `value`.
    pub(crate) fn insert(&mut self, pos: usize, value: bool) {
        assert!(pos <= self.len() && self.len() < NDIM_MAX);
        let low_mask = Self::low_mask(pos);
        self.bits = (self.bits & low_mask) | ((self.bits & !low_mask) << 1);
        self.len += 1;
        self.set(pos, value);
    }
}

impl FromIterator<bool> for DimBitmap {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut bitmap = Self { bits: 0, len: 0 };
        for value in iter {
            assert!(bitmap.len() < NDIM_MAX);
            let dim = bitmap.len();
            bitmap.len += 1;
            bitmap.set(dim, value);
        }
        bitmap
    }
}

/// Iterator over a [`DimBitmap`], yielding the flag of each dimension (dimension 0 first).
pub(crate) struct DimBitmapIter {
    bitmap: DimBitmap,
    pos: usize,
}
impl Iterator for DimBitmapIter {
    type Item = bool;
    #[inline]
    fn next(&mut self) -> Option<bool> {
        (self.pos < self.bitmap.len()).then(|| {
            let value = self.bitmap.get(self.pos);
            self.pos += 1;
            value
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bitmap.len() - self.pos;
        (remaining, Some(remaining))
    }
}
impl ExactSizeIterator for DimBitmapIter {}
impl IntoIterator for DimBitmap {
    type Item = bool;
    type IntoIter = DimBitmapIter;
    #[inline]
    fn into_iter(self) -> DimBitmapIter {
        DimBitmapIter {
            bitmap: self,
            pos: 0,
        }
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
    /// the *minimum read shape that doesn't waste work*: the smallest read tile, per dim, below which
    /// a read would redundantly re-read or recompute underlying data (e.g. a broadcast dim's whole
    /// length, so the single source element is read once instead of once per tile). It seeds
    /// [`scale_read_shape`](crate::util::scale_read_shape) - the read region is scaled down/up from
    /// it - so propagating a sensible value here is what lets a reader avoid recomputing an expensive
    /// broadcasted view per output tile.
    pub(crate) block_shape: DimArray<BlockSize>,
    /// Per-dimension "fixed" flags for [`block_shape`](Self::block_shape).
    ///
    /// A fixed dim keeps its exact block length when a **new Compact array** is materialized
    /// (`ArrayParams::tune`/`scale_block_shape`); a non-fixed dim may be freely resized to fit the
    /// target block size. This flag only affects Compact materialization - it does *not* enter the
    /// read path (`scale_read_shape` ignores it).
    pub(crate) block_shape_fixed_dims: DimBitmap,
    /// Estimated cost of reading a single element from this array, ignoring any broadcasting or
    /// duplication (see [`normalize_dim_scale_weights`]/[`combine_elementwise_hints`] and the
    /// read-hint design).
    pub(crate) element_cost: f32,
    /// Per-dimension relative scaling priority (higher = prefer to scale/cover that dim first, to
    /// avoid recomputing an expensive broadcasted view). See [`DimScaleWeight`].
    ///
    /// These weights are normalized *within a single array* and are meaningless across arrays on
    /// their own: two arrays' weights are only comparable once each is scaled by its own array's
    /// [`element_cost`](Self::element_cost). Any op that combines the weights of several inputs
    /// (element-wise, concatenate, stack, ...) must therefore multiply each input's weight by that
    /// input's `element_cost` before comparing or summing them.
    pub(crate) dim_scale_weights: DimArray<DimScaleWeight>,
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
        let dynamic = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims,
            element_cost: 1.0,
            // Default to C-order priority: the last, most-contiguous dim ranks highest. Leaves with
            // a different layout (e.g. Plain) override this.
            dim_scale_weights: normalize_dim_scale_weights(
                dim_arr(ndim, |i| (i + 1) as f64).as_slice(),
            ),
        };
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
    pub(crate) fn dim_scale_weights(&self) -> &'a DimArray<DimScaleWeight> {
        &self.dynamic().dim_scale_weights
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
        self.read_shape_heuristic_with_scale_order(
            max_shape,
            shape,
            itemsize,
            (0..max_shape.len()).rev(),
        )
    }

    pub(crate) fn read_shape_heuristic_with_scale_order<D>(
        &self,
        max_shape: &[u64],
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
            read_shape.as_mut_slice(),
            max_shape,
            shape,
            self.read_size().nitems(itemsize),
            scale_order,
        );
        read_shape
    }
}

/// A per-dimension scaling-priority weight, normalized into `[0, 255]`. Interpret it as a fraction
/// of 255 (i.e. a value in `[0, 1]`) via [`f64`](Self::f64).
///
/// Construct one only through [`normalize_dim_scale_weights`] or [`DimScaleWeight::zero`], so the
/// "normalized" invariant is preserved - most ops build raw `f64` weights and re-normalize.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub(crate) struct DimScaleWeight(u8);
impl DimScaleWeight {
    /// The lowest weight - a dimension not worth covering.
    #[inline]
    pub(crate) fn zero() -> Self {
        Self(0)
    }
    /// This weight as a fraction in `[0, 1]`.
    #[inline]
    pub(crate) fn f64(self) -> f64 {
        self.0 as f64 / 255.0
    }
}

/// Normalize raw per-dimension scale weights into stored [`DimScaleWeight`]s.
///
/// A `0.0` (or negative) weight maps to [`DimScaleWeight::zero`]. Positive weights map to distinct
/// values in `[1, 255]`, ordered so a larger raw weight gets a larger stored weight, and rounded to
/// preserve relative magnitude where possible. Ties (equal raw weight) are always broken so the
/// inner (greater-index) dimension gets the lower weight. `weights.len()` must be `<= NDIM_MAX`.
pub(crate) fn normalize_dim_scale_weights(weights: &[f64]) -> DimArray<DimScaleWeight> {
    let n = weights.len();
    debug_assert!(n <= NDIM_MAX);
    let max = weights.iter().copied().fold(0.0f64, f64::max);
    if max <= 0.0 {
        return dim_arr(n, |_| DimScaleWeight::zero());
    }
    // Order the positive dims from lowest priority to highest: by raw weight ascending, breaking
    // ties so the inner (greater-index) dim ranks lower (i.e. sorts earlier).
    let mut order: DimArray<usize> = (0..n).filter(|&d| weights[d] > 0.0).collect();
    order.sort_by(|&a, &b| weights[a].total_cmp(&weights[b]).then(b.cmp(&a)));
    // Assign strictly increasing values in [1, 255], seeded from a magnitude-preserving round and
    // separated as needed (forward pass), then capped at 255 (backward pass). For k <= NDIM_MAX
    // this always yields k distinct values in [1, 255].
    let k = order.len();
    let mut vals = dim_arr(k, |_| 0u16);
    let mut prev = 0u16;
    for j in 0..k {
        let rounded = (weights[order[j]] / max * 254.0).round() as u16 + 1;
        vals[j] = rounded.max(prev + 1);
        prev = vals[j];
    }
    let mut next = 256u16;
    for j in (0..k).rev() {
        vals[j] = vals[j].min((next - 1).min(255));
        next = vals[j];
    }
    let mut out = dim_arr(n, |_| DimScaleWeight::zero());
    for j in 0..k {
        out[order[j]] = DimScaleWeight(vals[j] as u8);
    }
    out
}

/// Combine the read hints of the inputs of a *selection* op - one where each output element reads
/// exactly one input (e.g. [`Concatenate`](crate::ops::Concatenate)/[`Stack`](crate::ops::Stack)).
///
/// There is no re-reading of any input, so `element_cost` is the worst case across inputs, plus one.
/// Per dim, the weight is `normalize( max_i weight_i[d] * cost_i )` - each input's weight is rescaled
/// by its own `element_cost` first, since a [`DimScaleWeight`] is not comparable across arrays.
/// `inputs` must be non-empty with equal-length weight slices.
pub(crate) fn combine_select_hints(
    inputs: &[(f32, &[DimScaleWeight])],
) -> (f32, DimArray<DimScaleWeight>) {
    let ndim = inputs[0].1.len();
    let element_cost = inputs.iter().map(|&(cost, _)| cost).fold(0.0f32, f32::max) + 1.0;
    let weights = dim_arr(ndim, |d| {
        inputs
            .iter()
            .map(|&(cost, sw)| sw[d].f64() * cost as f64)
            .fold(0.0f64, f64::max)
    });
    (
        element_cost,
        normalize_dim_scale_weights(weights.as_slice()),
    )
}

/// Combine the read hints of the inputs of an element-wise op (same shape).
///
/// Returns the combined `(element_cost, dim_scale_weights)`: `element_cost = sum(costs) + 1` and,
/// per dim, `weight[d] = normalize( sum_i weight_i[d] * cost_i )`. Multiplying each input's
/// normalized weight by its own cost re-injects the cross-input magnitude that per-input
/// normalization strips. `inputs` must be non-empty and all weight slices the same length.
pub(crate) fn combine_elementwise_hints(
    inputs: &[(f32, &[DimScaleWeight])],
) -> (f32, DimArray<DimScaleWeight>) {
    let ndim = inputs[0].1.len();
    let element_cost = (inputs.iter().map(|&(cost, _)| cost as f64).sum::<f64>() + 1.0) as f32;
    let weights = dim_arr(ndim, |d| {
        inputs
            .iter()
            .map(|&(cost, sw)| sw[d].f64() * cost as f64)
            .sum::<f64>()
    });
    (
        element_cost,
        normalize_dim_scale_weights(weights.as_slice()),
    )
}

/// Combine the block layout (`block_shape` + `block_shape_fixed_dims`) of several equal-ndim inputs
/// of an element-wise or selection op (binary, `where`, map-multiple, concatenate, stack).
///
/// Per dimension the output block length is the `max` over inputs - the coarsest "minimum read shape
/// that doesn't waste work" wins, so a broadcasted operand's full-dim coverage is preserved rather
/// than dropped in favor of a smaller operand's block. A dimension is marked fixed only if every
/// input agrees on that block length *and* at least one input marks it fixed: a disagreement means
/// there is no single meaningful storage boundary to preserve. `inputs` must be non-empty with
/// equal-length block-shape slices, each matching its `block_shape_fixed_dims` length.
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
        combine_block_layout, combine_elementwise_hints, combine_select_hints,
        normalize_dim_scale_weights, ArraySpecFlags, DimBitmap, DimScaleWeight, ReadSize,
    };
    use crate::{Array, ArrayParams, ArrayStorage};

    fn bits(bm: DimBitmap) -> Vec<bool> {
        bm.into_iter().collect()
    }

    /// Assert `out` is a valid normalization of `raw`: `0` -> `zero`; positive weights get distinct
    /// values in `[1, 255]`; a larger raw weight gets a larger stored weight; and ties (equal raw)
    /// are broken so the inner (greater-index) dim ranks lower.
    fn assert_normalized(raw: &[f64], out: &[DimScaleWeight]) {
        use std::cmp::Ordering;
        assert_eq!(raw.len(), out.len());
        for i in 0..raw.len() {
            if raw[i] <= 0.0 {
                assert_eq!(out[i], DimScaleWeight::zero());
            } else {
                assert!((1..=255).contains(&out[i].0));
            }
            for j in 0..raw.len() {
                let expected = if raw[i] <= 0.0 && raw[j] <= 0.0 {
                    Ordering::Equal
                } else if raw[i] <= 0.0 {
                    Ordering::Less
                } else if raw[j] <= 0.0 {
                    Ordering::Greater
                } else if raw[i] != raw[j] {
                    raw[i].partial_cmp(&raw[j]).unwrap()
                } else {
                    // Equal positive raw: inner (greater index) ranks lower.
                    j.cmp(&i)
                };
                assert_eq!(
                    out[i].cmp(&out[j]),
                    expected,
                    "({i},{j}) raw {raw:?} out {out:?}"
                );
            }
        }
    }

    #[test]
    fn normalize_dim_scale_weights_orders_and_breaks_ties() {
        for input in [
            vec![],
            vec![0.0f64],
            vec![5.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            vec![3.0, 1.0, 2.0],
            vec![0.0, 7.0, 7.0, 100.0],
            vec![1e12, 1.0, 1e12, 0.0],
            vec![10.0, 20.0, 20.0, 5.0, 0.0, 1e6],
        ] {
            let out = normalize_dim_scale_weights(&input);
            assert_normalized(&input, out.as_slice());
        }
    }

    #[test]
    fn normalize_dim_scale_weights_ties_favor_outer_dim() {
        // All-equal positive weights strictly decrease by dim index (inner ranks lower).
        let out = normalize_dim_scale_weights(&[4.0, 4.0, 4.0]);
        assert!(out[0] > out[1] && out[1] > out[2]);
    }

    #[test]
    fn normalize_dim_scale_weights_close_distinct_values_stay_distinct() {
        // Two distinct weights that both round to the top must not collapse together.
        let out = normalize_dim_scale_weights(&[1e18, 1e18 - 1.0]);
        assert!(out[0] > out[1]);
    }

    #[test]
    fn combine_elementwise_hints_costlier_operand_dominates() {
        // Operand b weights dim 1 and is far more expensive per element, so it dominates dim 1.
        let a_sw = normalize_dim_scale_weights(&[10.0, 1.0]);
        let b_sw = normalize_dim_scale_weights(&[1.0, 10.0]);
        let (ec, sw) =
            combine_elementwise_hints(&[(1.0, a_sw.as_slice()), (1000.0, b_sw.as_slice())]);
        assert_eq!(ec, 1.0 + 1000.0 + 1.0);
        assert!(sw[1] > sw[0], "expected dim1 to dominate, got {sw:?}");
    }

    #[test]
    fn read_hints_propagate_through_reduction_and_broadcast() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap(),
        )
        .unwrap();
        // A compact leaf reads at element_cost 8, with C-order dim priority (last dim highest).
        {
            let sp = a.storage().spec();
            assert_eq!(sp.element_cost(), 8.0);
            let sw = sp.dim_scale_weights();
            assert!(sw[0] < sw[1] && sw[0] > DimScaleWeight::zero());
        }
        // Reduce over axis 1 (extent 4), re-insert the axis, and broadcast back to [3, 4].
        let bc = a.sum(1).insert_axis(1).broadcast(&[3, 4]);
        let sp = bc.storage().spec();
        // Reduction folds the whole reduced extent per output: 8 * (4 + 4) = 64.
        assert_eq!(sp.element_cost(), 64.0);
        // The broadcast dim (1) must outrank the untouched dim (0).
        let sw = sp.dim_scale_weights();
        assert!(sw[1] > sw[0], "broadcast dim should dominate, got {sw:?}");
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
        // The expensive broadcast dim carries through the binary combine.
        let sw = sp.dim_scale_weights();
        assert!(sw[1] > sw[0], "expected dim1 to dominate, got {sw:?}");
    }

    #[test]
    fn combine_select_hints_is_cost_weighted() {
        // Each output reads one input, so element_cost is the max plus one, and per-dim weights are
        // compared only after scaling by each input's own cost.
        let a_sw = normalize_dim_scale_weights(&[10.0, 1.0]); // cheap input weights dim 0
        let b_sw = normalize_dim_scale_weights(&[1.0, 10.0]); // expensive input weights dim 1
        let (ec, sw) = combine_select_hints(&[(1.0, a_sw.as_slice()), (1000.0, b_sw.as_slice())]);
        assert_eq!(ec, 1000.0 + 1.0);
        assert!(sw[1] > sw[0], "expected dim1 to dominate, got {sw:?}");
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
    fn binary_op_preserves_broadcast_full_dim_block() {
        // A small explicit block on dim 1, so the broadcast's full-dim coverage is strictly larger.
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

        // std-like: reduce axis 1, re-insert it, broadcast back. The broadcast dim's block covers
        // the whole extent so the reduction is not recomputed per column-tile.
        let bc = mk().sum(1).insert_axis(1).broadcast(&[3, 4]);
        assert_eq!(bc.storage().spec().block_shape()[1], 4);

        // The binary op must keep the broadcasted full-dim coverage: max(2, 4) = 4, not `a`'s 2.
        let out = a.maximum(bc);
        let sp = out.storage().spec();
        assert_eq!(sp.block_shape().as_slice(), &[3, 4]);
        // dim 0: equal block (3) and `a` fixed -> fixed. dim 1: differing block (2 vs 4) -> not fixed.
        assert_eq!(bits(sp.block_shape_fixed_dims()), [true, false]);
    }

    #[test]
    fn plain_leaf_dim_scale_weights_follow_strides() {
        // C-contiguous: most contiguous (last) dim ranks highest, like a compact leaf.
        let c = Array::plain_ndarray(
            ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap(),
        )
        .unwrap();
        let sw = c.storage().spec().dim_scale_weights().clone();
        assert!(sw[0] < sw[1] && sw[0] > DimScaleWeight::zero());

        // A size-1 dim reads the same regardless of coverage, so it gets the lowest weight (zero).
        let s = Array::plain_ndarray(
            ndarray::Array::from_shape_vec([3, 1], (0..3i32).collect()).unwrap(),
        )
        .unwrap();
        let sw = s.storage().spec().dim_scale_weights().clone();
        assert_eq!(sw[1], DimScaleWeight::zero());
        assert!(sw[0] > DimScaleWeight::zero());
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
}
