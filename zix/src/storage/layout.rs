use crate::dtype::Itemsize;
use crate::error::{check_ndim, ensure, Result};
use crate::storage::block::BlockSize;
use crate::util::{dim_arr, DimArray, Idx};

/// Block geometry hints for an nd-array storage.
///
/// Carries two independent hints that describe the recommended block shape for an array:
///
/// - **Storage block shape** (`block_shape_hint`, `block_shape_tag`, `block_size_hint`) —
///   the recommended nd-block shape to use when encoding array data into block storage.
///   For the baseline compressed storage this matches the actual block shape, but in
///   general it is only a hint that may differ from the true underlying layout.
///
/// - **Preferred read shape** (`preferred_read_shape`, `preferred_read_size_hint`) —
///   the recommended region size to request in a single read, typically larger than the
///   storage block shape and targeting the L2 cache.
///
/// Element-wise operations (e.g. negation, `exp`) propagate this layout unchanged.
/// Shape-changing operations (reshape, permute, broadcast, reduction, etc.) update the
/// hints to reflect a layout that would work well for arrays subsequently constructed
/// from the view.
#[derive(Clone)]
pub struct BlocksLayout {
    /// See [`BlocksLayout::block_shape_hint`].
    pub(crate) block_shape_hint: DimArray<BlockSize>,

    /// See [`BlocksLayout::block_shape_tag`].
    pub(crate) block_shape_tag: DimArray<BlockShapeTag>,

    /// See [`BlocksLayout::block_size_hint`].
    pub(crate) block_size_hint: u64,

    /// See [`BlocksLayout::preferred_read_shape`].
    pub(crate) preferred_read_shape: DimArray<BlockSize>,

    /// See [`BlocksLayout::preferred_read_size_hint`].
    pub(crate) preferred_read_size_hint: u64,
}

/// Per-dimension constraint on how a block shape dimension may be automatically scaled
/// when a new array is constructed without an explicit block shape.
///
/// See [`BlocksLayout::block_shape_tag`](BlocksLayout).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

impl BlocksLayout {
    /// Compute and validate the block geometry for an array.
    ///
    /// Both the storage block shape and the preferred read block shape are resolved here;
    /// either can be supplied explicitly or left as `None` to be auto-computed from a
    /// target byte size.
    ///
    /// # Arguments
    ///
    /// - `block_shape` — shape of one storage block in items per dimension.
    ///   When `None`, a shape is chosen automatically so that each block is approximately
    ///   `block_size_hint` bytes.
    /// - `block_shape_tag` — per-dimension constraint on how the block shape may be scaled;
    ///   requires `block_shape` to also be provided. Defaults to all-[`BlockShapeTag::Fixed`].
    ///   See [`BlockShapeTag`] for the available options.
    /// - `block_size_hint` — target block size in bytes used when auto-computing or scaling
    ///   the block shape. Defaults to the L1 data cache size when the shape is not fully
    ///   [`BlockShapeTag::Fixed`].
    /// - `preferred_read_shape` — region size the read path prefers to request at once,
    ///   in items per dimension. When `None`, auto-computed from `preferred_read_size_hint`.
    /// - `preferred_read_size_hint` — target size for the preferred read region in bytes.
    ///   Defaults to the L2 cache size.
    /// - `shape` — the array shape, used to clamp block dimensions that would exceed the array.
    /// - `itemsize` — bytes per array element.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if:
    /// - `block_shape_tag` is provided without `block_shape`
    /// - the length of `block_shape_tag` or `preferred_read_shape` does not match `ndim`
    /// - `ndim` exceeds [`crate::NDIM_MAX`]
    pub(crate) fn new(
        block_shape: Option<DimArray<BlockSize>>,
        block_shape_tag: Option<DimArray<BlockShapeTag>>,
        mut block_size_hint: Option<u64>,
        preferred_read_shape: Option<DimArray<BlockSize>>,
        mut preferred_read_size_hint: Option<u64>,

        shape: &[u64],
        itemsize: Itemsize,
    ) -> Result<Self> {
        let ndim = shape.len();
        check_ndim(ndim)?;
        let itemsize = itemsize as u64;

        let cache_sizes = crate::util::cache_size::cache_sizes();

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
            "ndim does not match block_shape_tag length"
        );
        let fixed_block_shape = block_shape_tag
            .iter()
            .all(|&tag| tag == BlockShapeTag::Fixed);
        // Compute block_size_hint if not specified, and if it cant be computed from block_shape
        if block_size_hint.is_none() && (block_shape.is_none() || !fixed_block_shape) {
            block_size_hint = Some(cache_sizes.l1_data as u64);
        }
        // Compute block shape
        let mut block_shape = block_shape.unwrap_or_else(|| {
            Self::scale_block_shape(
                &dim_arr(ndim, |_| 1),
                &dim_arr(ndim, |_| true),
                block_size_hint.unwrap() / itemsize,
                shape,
            )
        });
        // Scale block_shape up to block_size_hint
        if !fixed_block_shape {
            block_shape = Self::scale_block_shape(
                &dim_arr(ndim, |dim| match block_shape_tag[dim] {
                    BlockShapeTag::Fixed | BlockShapeTag::MultipleOf => block_shape[dim],
                    BlockShapeTag::Any => 1,
                }),
                &dim_arr(ndim, |dim| block_shape_tag[dim] != BlockShapeTag::Fixed),
                block_size_hint.unwrap() / itemsize,
                shape,
            );
        }
        // Update block_size_hint to block_shape.product() if it is not specified
        let block_size_hint = block_size_hint
            .unwrap_or_else(|| block_shape.iter().map(|&b| b as u64).product::<u64>() * itemsize);
        // Compute preferred_read_size_hint if not specified, and if it cant be computed from preferred_read_shape
        if preferred_read_size_hint.is_none() && preferred_read_shape.is_none() {
            preferred_read_size_hint = Some(cache_sizes.l2 as u64);
        }
        // Compute preferred_read_shape
        let preferred_read_shape = match preferred_read_shape {
            Some(preferred_read_shape) => {
                ensure!(
                    ndim == preferred_read_shape.len(),
                    InvalidArgument,
                    "ndim does not match preferred_read_shape length"
                );
                dim_arr(ndim, |dim| {
                    (preferred_read_shape[dim] as u64)
                        .max(block_shape[dim] as u64)
                        .min(shape[dim]) as BlockSize
                })
            }
            None => Self::scale_block_shape(
                &block_shape,
                &dim_arr(ndim, |_| true),
                preferred_read_size_hint.unwrap() / itemsize,
                shape,
            ),
        };
        // Update preferred_read_size_hint to preferred_read_shape.product() if it is not specified
        let preferred_read_size_hint = preferred_read_size_hint.unwrap_or_else(|| {
            preferred_read_shape
                .iter()
                .map(|&b| b as u64)
                .product::<u64>()
                * itemsize
        });

        Ok(BlocksLayout {
            block_shape_hint: block_shape,
            block_shape_tag,
            block_size_hint,
            preferred_read_shape,
            preferred_read_size_hint,
        })
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
        block_len as BlockSize
    }

    /// Recommended storage block shape, in items per dimension.
    ///
    /// This is a *hint*, not an exact description of the underlying storage. For the
    /// baseline compressed storage it matches the actual block shape, but lazy operations
    /// (reduction, broadcast, reshape, etc.) may update it to reflect a recommended shape
    /// for arrays constructed from their views. Callers should treat it as a recommendation,
    /// not a guarantee.
    pub fn block_shape_hint(&self) -> &[BlockSize] {
        &self.block_shape_hint
    }

    /// Per-dimension tag describing how `block_shape_hint` should be treated when a new
    /// array needs to choose a block shape automatically (i.e. without an explicit user
    /// override).
    ///
    /// Users typically choose a block shape based on their access patterns, so `Fixed`
    /// is the default — it preserves that choice in downstream arrays. Operations that
    /// change the logical shape (reduction, broadcast, reshape, etc.) may tag affected
    /// dimensions as `Any` or `MultipleOf` to let the heuristic freely pick a suitable
    /// size for those dimensions.
    pub fn block_shape_tag(&self) -> &[BlockShapeTag] {
        &self.block_shape_tag
    }

    /// Target byte size hint for a storage block.
    ///
    /// Used as the budget when auto-choosing a block shape for a new array. It is a hint
    /// only — it may differ from `block_shape_hint.iter().product() * itemsize` when both
    /// a shape and a hint were provided independently, or when a lazy operation updated
    /// one without changing the other. Defaults to the L1 data cache size when no block
    /// shape or hint has been set explicitly.
    pub fn block_size_hint(&self) -> u64 {
        self.block_size_hint
    }

    /// Recommended nd-region size to request in a single read, in items per dimension.
    ///
    /// A hint to the read path: reads are most efficient when they cover a region of
    /// approximately this shape. Typically larger than `block_shape_hint` (targeting the
    /// L2 cache), it guides operations like `copy` to issue larger read requests. Like
    /// `block_shape_hint`, lazy operations may update this independently.
    pub fn preferred_read_shape(&self) -> &[BlockSize] {
        &self.preferred_read_shape
    }

    /// Target byte size hint for a single read region.
    ///
    /// Analogous to `block_size_hint` but for the preferred read shape. May differ from
    /// `preferred_read_shape.iter().product() * itemsize` for the same reasons.
    /// Defaults to the L2 cache size when not set explicitly.
    pub fn preferred_read_size_hint(&self) -> u64 {
        self.preferred_read_size_hint
    }
}
