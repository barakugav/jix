use std::ops::Range;

use crate::codec::{DecoderParams, EncoderParams, ReadContext};
use crate::dtype::{Dtype, Itemsize};
use crate::error::{check_ndim, ensure, Result};
use crate::storage::block::BlockSize;
use crate::util::{dim_arr, DimArray, Idx};

mod compressed;
pub use compressed::*;

mod plain;
pub use plain::*;

mod scalar;
pub use scalar::*;

pub(crate) mod block;

pub trait ArrayStorage {
    /// Read the specified slice of the array into the provided buffer.
    ///
    /// # Arguments
    ///
    /// - `index`: A slice of ranges, one per dimension, specifying the slice of the array to read.
    ///   Each range is half-open: `start..end`, where `start` is inclusive and `end` is exclusive.
    /// - `buf`: A mutable byte slice to store the read data.
    ///   The size of the buffer must be exactly equal to the number of elements in the specified
    ///   slice multiplied by the item size of the array's dtype.
    ///   The buffer base pointer must be suitably aligned for the array's dtype.
    ///   Elements should be laid out in row-major order (C-style contiguous) in the buffer.
    /// - `context`: A context object that may be used for caching or other purposes during the
    ///   read operation. See `ReadContext` for more details.
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()>;

    fn shape(&self) -> &[u64];
    fn dtype(&self) -> &Dtype;

    fn spec(&self) -> ArrayStorageSpec<'_>;
}
pub struct ArrayStorageSpec<'a> {
    pub(crate) blocks_layout: &'a BlocksLayout,
    pub(crate) encoder_params: Option<&'a EncoderParams>,
    pub(crate) decoder_params: Option<&'a DecoderParams>,
    // pub(crate) decoder_config: Option<&'a DecoderCodecConfig>,
}

#[derive(Clone)]
pub struct BlocksLayout {
    /// === how preferred read block shape is transformed by view ops ===
    /// permute_axis - permute the block shape
    /// insert_axis - insert a block dim of size 1
    /// broadcast - broadcasted dims will be set to full dim size
    /// reduce_axis - just remove the block dim, or set it to 1 if keepdims
    /// reshape:
    ///   - dims that kept the logical stride and length will keep the same block shape.
    ///   - dims that kept the logical stride and reduced length will keep the same block shape.
    ///   - dims that kept the logical stride and increased length will use the dim's block
    ///     shape multiplied by some factor (see later).
    ///   - other dims will use 1, and will be scaled up by some factor (see later).
    ///   - After the initial block shape is determined, without the factors, the block shape is
    ///     scaled to block_size_hint by scaling each dim by a factor, starting
    ///     with the last dim, until the block size is at most block_size_hint.
    pub(crate) block_shape_hint: DimArray<BlockSize>,
    pub(crate) block_shape_tag: DimArray<BlockShapeTag>,
    pub(crate) block_size_hint: u64, // in bytes units

    /// === how preferred read block shape is transformed by view ops ===
    /// permute_axis - permute the block shape
    /// insert_axis - insert a block dim of size 1
    /// broadcast - broadcasted dims will be set to full dim size
    /// reduce_axis - just remove the block dim, or set it to 1 if keepdims
    /// reshape:
    ///   - dims that kept the logical stride and length will keep the same block shape.
    ///   - dims that kept the logical stride and reduced length will keep the same block shape.
    ///   - dims that kept the logical stride and increased length will use the dim's block
    ///     shape multiplied by some factor (see later).
    ///   - other dims will use 1, and will be scaled up by some factor (see later).
    ///   - After the initial block shape is determined, without the factors, the block shape is
    ///     scaled to preferred_read_block_size_hint by scaling each dim by a factor, starting
    ///     with the last dim, until the block size is at most preferred_read_block_size_hint.
    pub(crate) preferred_read_block_shape: DimArray<BlockSize>,
    pub(crate) preferred_read_block_size_hint: u64, // in bytes units
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BlockShapeTag {
    Fixed,
    MultipleOf,
    Any,
}

impl BlocksLayout {
    pub(crate) fn new(
        block_shape: Option<DimArray<BlockSize>>,
        block_shape_tag: Option<DimArray<BlockShapeTag>>,
        mut block_size_hint: Option<u64>,
        preferred_read_block_shape: Option<DimArray<BlockSize>>,
        mut preferred_read_block_size_hint: Option<u64>,

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
        // Compute preferred_read_block_size_hint if not specified, and if it cant be computed from preferred_read_block_shape
        if preferred_read_block_size_hint.is_none() && preferred_read_block_shape.is_none() {
            preferred_read_block_size_hint = Some(cache_sizes.l2 as u64);
        }
        // Compute preferred_read_block_shape
        let preferred_read_block_shape = match preferred_read_block_shape {
            Some(preferred_read_block_shape) => {
                ensure!(
                    ndim == preferred_read_block_shape.len(),
                    InvalidArgument,
                    "ndim does not match preferred_read_block_shape length"
                );
                dim_arr(ndim, |dim| {
                    (preferred_read_block_shape[dim] as u64)
                        .max(block_shape[dim] as u64)
                        .min(shape[dim]) as BlockSize
                })
            }
            None => Self::scale_block_shape(
                &block_shape,
                &dim_arr(ndim, |_| true),
                preferred_read_block_size_hint.unwrap() / itemsize,
                shape,
            ),
        };
        // Update preferred_read_block_size_hint to preferred_read_block_shape.product() if it is not specified
        let preferred_read_block_size_hint = preferred_read_block_size_hint.unwrap_or_else(|| {
            preferred_read_block_shape
                .iter()
                .map(|&b| b as u64)
                .product::<u64>()
                * itemsize
        });

        Ok(BlocksLayout {
            block_shape_hint: block_shape,
            block_shape_tag,
            block_size_hint,
            preferred_read_block_shape,
            preferred_read_block_size_hint,
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
}

pub struct Ref<'a, S>(pub(crate) &'a S);
impl_array_storage_forward!(Ref<'a, S> where S: ArrayStorage);

macro_rules! impl_array_storage_forward {
    ($wrapper:ident $(<$($gen:tt),*>)? $(where $($wh:tt)*)?) => {
        impl $(<$($gen),*>)? crate::storage::ArrayStorage for $wrapper $(<$($gen),*>)?
        where
            $($($wh)*)?
        {
            fn read_data(
                &self,
                index: &[core::ops::Range<u64>],
                buf: &mut [u8],
                context: &crate::codec::ReadContext,
            ) -> crate::error::Result<()> {
                self.0.read_data(index, buf, context)
            }
            fn shape(&self) -> &[u64] {
                self.0.shape()
            }
            fn dtype(&self) -> &crate::dtype::Dtype {
                self.0.dtype()
            }
            fn spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
                self.0.spec()
            }
        }
    };
}
pub(crate) use impl_array_storage_forward;
