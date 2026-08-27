use std::mem::MaybeUninit;
use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::{Alignment, Dtype, Dtyped, Itemsize};
use crate::error::{bail, check_get_range, check_ndim, ensure, Result};
use crate::ops::common::AxesArg;
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{
    check_out_buf, materialize_out_buf, ArraySpec, ArrayStorageInfo, ArrayStorageTyped, StridedBuf,
};
use crate::util::iter::NdIter;
use crate::util::SliceExt;
use crate::util::{calc_block_end, scale_read_shape, DimArray, USE_NEW_READ_SCALING};
use crate::{
    array_from_fn_inline, default_strides, Array, ArrayExt, ArrayStorage, DimVec, Dimension,
    IterExt, Ty,
};

pub(crate) struct ReductionOp<S: ArrayStorage, K, D> {
    kernel: K,

    array: S,
    is_reduced: <S::Dimension as Dimension>::Vec<bool>,

    shape: D,
    spec: ArraySpecDynamic,
}
pub(crate) trait ReductionOpKernel<T> {
    type Output;

    type State;

    /// Build the initial accumulator. `init_item` is the first stream element together with
    /// its TRUE global stream position (0-based), or `None` when the kernel was invoked on an
    /// empty reduction. Kernels with [`supports_empty`](Self::supports_empty) returning
    /// `false` may unwrap `init_item` - the caller guarantees it is `Some` for those kernels.
    ///
    /// The bundled index is not always `0`: a lane accumulator is seeded from an interior
    /// item, and a cell can be re-seeded partway through the stream when the reduced axis
    /// spans several bulks. Kernels whose result depends on element position (argmax/argmin)
    /// record it; the others ignore it.
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State;

    /// Fold `item` (at true global stream position `idx`) into `state`.
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State;

    /// Combine two partial accumulators folded over DISJOINT subsets of the stream.
    ///
    /// MUST be associative and commutative with respect to the folded result (up to float
    /// accuracy): it collapses the interleaved lane accumulators of a single cell, and continues
    /// a cell's accumulator across the bulks of a large reduced axis. The two subsets never
    /// overlap and together cover the folded elements exactly once.
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State;

    /// Produce the final result. `nitems` is the total number of stream elements that
    /// were folded into `state`, so `nitems == 0` exactly when the reduction was empty.
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output;

    fn supports_empty(&self) -> bool;
}

impl<S: ArrayStorage, K, D> ReductionOp<S, K, D> {
    pub(crate) fn new<Ax>(array: S, kernel: K, axes: Ax) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: ReductionOpKernel<S::Item, Output: Dtyped>,
        D: Dimension,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        let input_ndim = array.shape().len();
        let mut is_reduced = S::Dimension::vec(input_ndim, |_| false);
        for i in 0..axes.len() {
            let ax = axes.get(i);
            ensure!(
                ax < input_ndim,
                InvalidArgument,
                "axis {ax} out of bounds for array of ndim {input_ndim}"
            );

            ensure!(!is_reduced[ax], InvalidArgument, "duplicate axis {ax}");
            is_reduced[ax] = true;
        }

        if !kernel.supports_empty()
            && array
                .shape()
                .iter()
                .zip(is_reduced.as_ref())
                .any(|(&s, &reduced)| reduced && s == 0)
        {
            bail!(
                InvalidArgument,
                "reduction on empty dimension not supported"
            );
        }

        let dim_new2orig = is_reduced
            .as_ref()
            .iter()
            .enumerate()
            .filter_map(|(dim, &reduced)| reduced.not().then_some(dim))
            .collect::<DimArray<_>>();
        let shape = dim_new2orig
            .iter()
            .map(|&dim| array.shape()[dim])
            .collect::<DimArray<_>>();
        let shape = D::from_slice(&shape);

        let spec = array.spec();
        let reduced_prod = (0..input_ndim)
            .filter(|&dim| is_reduced[dim])
            .map(|dim| array.shape()[dim])
            .product::<u64>();
        let element_cost = (spec.element_cost() as f64 * (reduced_prod + 4) as f64) as f32;
        let dim_orig2new = |d: usize| (d - (0..d).filter(|&j| is_reduced[j]).count()) as u8;
        let read_shape_scale_order = spec
            .read_shape_scale_order()
            .iter()
            .filter(|&&d| !is_reduced[d as usize])
            .map(|&d| dim_orig2new(d as usize))
            .collect();
        let spec = ArraySpecDynamic {
            block_shape: dim_new2orig
                .iter()
                .map(|&dim| spec.block_shape()[dim])
                .collect(),
            block_shape_fixed_dims: spec
                .block_shape_fixed_dims()
                .into_iter()
                .enumerate()
                .filter_map(|(dim, c)| is_reduced[dim].not().then_some(c))
                .collect(),
            element_cost,
            read_shape_scale_order,
        };

        Ok(Self {
            kernel,
            shape,
            spec,
            array,
            is_reduced,
        })
    }
}

impl<S, K, D> ArrayStorage for ReductionOp<S, K, D>
where
    S: ArrayStorageTyped,
    K: ReductionOpKernel<S::Item, Output: Dtyped>,
    D: Dimension,
{
    type ElementType = Ty<K::Output>;
    type Dimension = D;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        read_data_impl::<S::Dimension, D>(
            &self.array,
            self.shape(),
            self.dtype(),
            size_of::<K::State>(),
            Alignment::of::<K::State>(),
            &self.is_reduced,
            &|args| reduce_tile::<S::Item, K, D>(&self.kernel, args),
            &|args| {
                finalize_states::<S::Item, K, D>(&self.kernel, args);
            },
            index,
            context,
            out,
        )
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        const { &K::Output::DTYPE }
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("ReductionOp", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = ReductionOp<S, K, NewD>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let shape = self.shape.as_slice();
        check_ndim::<NewD>(shape.len())?;
        let shape = NewD::from_slice(shape);

        Ok(ReductionOp {
            kernel: self.kernel,
            array: self.array,
            is_reduced: self.is_reduced,
            shape,
            spec: self.spec,
        })
    }

    crate::ops::impl_element_type_change_default!();
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn read_data_impl<'a, InnerD, OuterD>(
    inner_array: &dyn ArrayStorage,
    outer_shape: &[u64],
    output_dtype: &Dtype,
    kernel_state_sizeof: usize,
    kernel_state_alignof: Alignment,
    is_reduced: &InnerD::Vec<bool>,
    reduce_tile_fn: &dyn Fn(ReduceTileArgs<OuterD>),
    finalize_state_fn: &dyn Fn(FinalizeStateArgs<OuterD>),
    index: &[Range<u64>],
    context: &'a ReadContext,
    out: Option<&'a mut StridedBuf<'_>>,
) -> Result<StridedBuf<'a>>
where
    InnerD: Dimension,
    OuterD: Dimension,
{
    // This method accept some &dyn fns to avoid monomorphizing the whole method for every combination
    // of kernel, dimension, dtype, and backing storage.

    check_get_range(outer_shape, index)?;
    check_out_buf(out.as_deref(), outer_shape)?;
    let out_shape_usize = OuterD::vec(index.len(), |d| (index[d].end - index[d].start) as usize);
    let mut out = materialize_out_buf(out, context, out_shape_usize.as_ref(), output_dtype);
    let (out_buf, out_strides) = out.data_mut();
    let out_strides = out_strides.to_dim_vec::<OuterD>();

    // Streams the reduction over a two-level chunking of the inner array so peak scratch
    // memory stays bounded *and* each downstream `self.array.read_data` call is sized to
    // roughly fit the caller's request in cache.
    //
    // # What it computes
    //
    // For every output position `O` covered by `index`, evaluates
    //
    //   stream(O) = iterate inner elements with non-reduced coords matching O,
    //               in row-major order over the reduced axes
    //   buf[O]    = K::finalize_state(fold(stream(O), K::init_state, K::update_state),
    //                                 nitems = stream length)
    //
    // # Two-level chunking: bulks and tiles
    //
    // The inner index range `inner_range_full` (reduced dims spanning the full source
    // extent, non-reduced dims forwarded from `index`) is partitioned at two granularities:
    //
    //   * **bulk** - the outer chunk. Splits *non-reduced* (output) dims only.
    //     `bulk_shape[d] = inner_shape[d]` for reduced dims (the full reduced extent sits
    //     in one bulk) and `tile_shape[d]` for non-reduced. So each bulk owns a *disjoint
    //     block of outputs* and contains that block's *entire* reduction stream - the
    //     block is fully reduced before moving on, and consecutive bulks never re-walk the
    //     same outputs. The live state working set is therefore one bulk's outputs, not
    //     the whole output buffer.
    //   * **tile** - the inner chunk. Splits *both* dim groups, but is shaped so the
    //     non-reduced part exactly matches one bulk's output block (one tile-along-non-
    //     reduced per bulk). Within a bulk the tile iterator sweeps the reduced axes in
    //     row-major order, so a tile is `bulk-non-reduced * tile-reduced`. Each
    //     `(bulk, tile)` pair produces *one* `self.array.read_data` call, sized to land
    //     within the source's `read_size` `(min, max)` window.
    //
    // `tile_shape` sizes each tile; it is chosen once per call by `reduction_tile_shape` (which keeps
    // enough reduced-dim volume that the kernel's per-tile overhead is amortized). Setting
    // `bulk_shape[non-reduced] = tile_shape[non-reduced]` then makes the inner read shape come out to
    // `tile_shape` for every tile.
    //
    // Both iterators are driven by `NdIterExtBlockOffsetSize`. Each yields
    // `(blk_idx, (inner_offset, blk_size))`: the absolute element start in dim `d` is
    // `blk_idx[d] * block_shape[d] + inner_offset[d]`, length `blk_size[d]`. Interior
    // blocks carry `inner_offset = 0, blk_size = block_shape`; border blocks carry the
    // partial values produced by the iterator.
    //
    // # Per-tile processing
    //
    // ```text
    // full_reduction_size = product(inner_range_full[d] len for d in reduced dims)
    // for each bulk:                      // a disjoint block of outputs
    //   bulk_base_item_idx = 0            // reduced-stream position within this bulk
    //   bulk_initialized   = false
    //   for each tile in this bulk:       // walks the reduced axes; one non-reduced tile
    //     reduction_size = product(tile_size[d] for d in reduced dims)
    //     items_buf <- self.array.read_data(tile = bulk-non-reduced * tile-reduced)
    //     for each output position O in this bulk's output block:
    //       fold the tile's `reduction_size` items at O's offset into state_buf[O],
    //       at stream indices [bulk_base_item_idx, bulk_base_item_idx + reduction_size).
    //       The first tile of the bulk seeds the slot (K::init_state); later tiles merge.
    //     bulk_base_item_idx += reduction_size
    //     bulk_initialized = true
    //   // assert bulk_base_item_idx == full_reduction_size (block fully reduced)
    // ```
    //
    // The first-seed vs. merge split is per-bulk: the *first* tile of a bulk takes the
    // seed branch (`K::init_state` writes each slot it covers - and one non-reduced tile
    // per bulk means it covers the bulk's whole output block); every later tile of the
    // same bulk merges its partial fold into those slots via `K::merge`. `bulk_initialized`
    // resets at the start of each bulk, so every output block is seeded exactly once, by
    // its own bulk.
    //
    // # Finalization
    //
    // After all bulks (every output has been fully reduced within its bulk):
    //   - If any output was written (`state_initialized`): finalize every state into `buf`
    //     via `K::finalize_state(state, full_reduction_size)`. When `state_in_out_buf`,
    //     the state and output pointers for each slot alias the same bytes, so each
    //     iteration `assume_init_read`s the state *before* writing the result.
    //   - Otherwise (empty reduction - only reachable when a reduced dim is empty and the
    //     kernel supports empty - which produces zero bulks): write
    //     `K::finalize_state(K::init_state(None), 0)` to every output.
    //
    // # Scratch buffers
    //
    // - `items_buf`: raw input elements for the current *tile*. Resized each tile.
    // - `state_buf`: `out_nitems` slots of `MaybeUninit<K::State>`. Seeded one output
    //   block at a time (the first tile of each bulk seeds that bulk's block); finalized
    //   in the post-loop pass. When `K::State` matches `K::Output` in size and is no more
    //   strictly aligned (`state_in_out_buf`), we skip the scratch allocation entirely and
    //   reuse the caller's `buf` as the state buffer - `finalize_state` then reads each
    //   slot and writes the output into the same byte range it just consumed. The
    //   finalization loop reads the state out of each slot *before* writing the result
    //   back, since state and output pointers alias in that mode (see the `CAREFUL`
    //   comments).
    //
    // # Invariants (also enforced by `debug_assert!`s)
    //
    // - Reduced dims produce at most one bulk-block per call
    //   (`bulk_shape[reduced] == inner_shape[reduced]`).
    // - Each bulk has exactly one tile-along-non-reduced
    //   (`tile_shape[non-reduced] == bulk_shape[non-reduced]`).
    // - Every tile's absolute element range is contained in `inner_range_full`.
    // - At the end of each bulk, `bulk_base_item_idx == full_reduction_size` - i.e. the
    //   bulk folded its outputs over exactly the full reduced stream.

    let out_nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;

    let inner_shape = inner_array.shape();
    let inner_ndim = inner_shape.len();
    let item_dtype = inner_array.dtype();
    let out_ndim = index.len();

    let inner_range_full = {
        let mut out_dim = 0;
        InnerD::vec(inner_ndim, |dim| {
            if is_reduced[dim] {
                0..inner_shape[dim]
            } else {
                let r = index[out_dim].clone();
                out_dim += 1;
                r
            }
        })
    };

    let out_shape = OuterD::vec(index.len(), |dim| index[dim].end - index[dim].start);

    // The read tile is chosen so the reduced dims keep enough volume to amortize per-tile overhead
    // (see `reduction_tile_shape`). Setting `bulk_shape[non-reduced] = tile_shape[non-reduced]` then
    // makes the inner read shape come out to `tile_shape` for every tile.
    let tile_max_shape = InnerD::vec(inner_ndim, |dim| {
        inner_range_full[dim].end - inner_range_full[dim].start
    });
    let tile_shape = reduction_tile_shape::<InnerD>(
        &inner_array.spec(),
        is_reduced.as_ref(),
        tile_max_shape.as_ref(),
        inner_shape,
        item_dtype.itemsize(),
    );

    // Bulk shape: full source extent on reduced dims (so an output's entire reduction
    // stream sits inside one bulk and is finished there), tile shape on non-reduced dims
    // (so each bulk owns a disjoint block of outputs, one tile wide - consecutive bulks
    // never re-walk the same outputs and double-count).
    let bulk_shape = InnerD::vec(inner_ndim, |dim| {
        if is_reduced[dim] {
            inner_shape[dim].max(1)
        } else {
            tile_shape[dim]
        }
    });
    let bulk_grid_begin = InnerD::vec(inner_ndim, |dim| {
        inner_range_full[dim].start / bulk_shape[dim]
    });
    let bulk_grid_end = InnerD::vec(inner_ndim, |dim| {
        calc_block_end(
            inner_range_full[dim].start,
            inner_range_full[dim].end,
            bulk_shape[dim],
        )
    });
    debug_assert!(
        (0..inner_ndim).all(|d| !is_reduced[d] || bulk_grid_end[d] - bulk_grid_begin[d] <= 1),
        "reduced dim must produce at most one bulk-block",
    );
    let bulk_iter = NdIter::builder_with_begin(bulk_grid_begin, bulk_grid_end)
        .with_block_offset_size_ext(
            &InnerD::vec(inner_ndim, |dim| inner_range_full[dim].start),
            &InnerD::vec(inner_ndim, |dim| inner_range_full[dim].end),
            bulk_shape.clone(),
        )
        .build();

    // Every output is fully reduced within its own bulk, so this is the stream length
    // folded into each output cell - and the `nitems` passed to `finalize_state`.
    let full_reduction_size = (0..inner_ndim)
        .filter(|&d| is_reduced[d])
        .map(|d| inner_range_full[d].end - inner_range_full[d].start)
        .product::<u64>();

    let output_align = output_dtype.alignment().as_usize();
    let state_in_out_buf = kernel_state_sizeof == output_dtype.itemsize() as usize
        && kernel_state_alignof.as_usize() <= output_align
        && (out_buf.as_ptr() as usize).is_multiple_of(output_align)
        && out_strides
            .as_ref()
            .iter()
            .all(|&s| s.is_multiple_of(output_align));
    let out_ptr = out_buf.as_mut_ptr();
    let out_buf_len = out_buf.len();
    let mut tmp_state_buf;
    // CAREFUL: state_buf and out_ptr may alias
    let (state_buf, state_strides) = if state_in_out_buf {
        // Reuse the output bytes as the state buffer
        let state_buf = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_buf_len) };
        (state_buf, out_strides.clone())
    } else {
        tmp_state_buf =
            context.allocate_buf(out_nitems * kernel_state_sizeof, kernel_state_alignof);
        (
            tmp_state_buf.as_mut_slice(),
            default_strides(&out_shape, kernel_state_sizeof),
        )
    };
    debug_assert!(state_strides
        .as_ref()
        .iter()
        .all(|&s| s.is_multiple_of(kernel_state_alignof.as_usize())));
    let mut state_initialized = false;

    let mut items_buf = context.allocate_buf(0, item_dtype.alignment());
    for (bulk_idx, (bulk_inner_offset, bulk_size)) in bulk_iter {
        // The bulk's absolute element range, used as the tile iterator's universe.
        let bulk_begin = InnerD::vec(inner_ndim, |dim| {
            bulk_idx[dim] * bulk_shape[dim] + bulk_inner_offset[dim]
        });
        let bulk_end = InnerD::vec(inner_ndim, |dim| bulk_begin[dim] + bulk_size[dim]);

        // Tile iterator: walks the bulk's range partitioned by `tile_shape`. Non-reduced
        // dims have exactly one tile per bulk (tile_shape[non-reduced] == bulk non-reduced
        // width), reduced dims are subdivided - and their tiles are swept in row-major
        // order so each output's reduced stream is folded front-to-back across the bulk.
        let tile_grid_begin = InnerD::vec(inner_ndim, |dim| bulk_begin[dim] / tile_shape[dim]);
        let tile_grid_end = InnerD::vec(inner_ndim, |dim| {
            calc_block_end(bulk_begin[dim], bulk_end[dim], tile_shape[dim])
        });
        debug_assert!(
            (0..inner_ndim)
                .all(|d| { is_reduced[d] || tile_grid_end[d] - tile_grid_begin[d] <= 1 }),
            "non-reduced dim must produce at most one tile per bulk",
        );
        let tile_iter = NdIter::builder_with_begin(tile_grid_begin, tile_grid_end)
            .with_block_offset_size_ext(
                &bulk_begin,
                &bulk_end,
                InnerD::vec(inner_ndim, |d| tile_shape[d]), // TODO: clone
            )
            .build();

        let mut bulk_base_item_idx = 0u64;
        let mut bulk_initialized = false;
        for (tile_idx, (tile_inner_offset, tile_size)) in tile_iter {
            let tile = InnerD::vec(inner_ndim, |dim| {
                let start = tile_idx[dim] * tile_shape[dim] + tile_inner_offset[dim];
                start..start + tile_size[dim]
            });
            debug_assert!(
                (0..inner_ndim).all(|d| {
                    inner_range_full[d].start <= tile[d].start
                        && tile[d].end <= inner_range_full[d].end
                }),
                "tile not contained in inner_range_full",
            );

            let tile_reduction_size = (0..inner_ndim)
                .filter(|&d| is_reduced[d])
                .map(|d| tile_size[d] as usize)
                .product::<usize>();

            // Read this tile's items into a `(output, reduced)` layout - reduced dims
            // innermost/contiguous - so each output's reduced stream is a unit-stride run.
            // `perm_lstrides[d]` is the element stride of source dim `d` in that layout.
            let mut perm_strides = InnerD::vec(inner_ndim, |_| 0usize);
            {
                let mut red_acc = item_dtype.itemsize() as usize;
                let mut out_acc = tile_reduction_size * item_dtype.itemsize() as usize;
                for d in (0..inner_ndim).rev() {
                    if is_reduced[d] {
                        perm_strides[d] = red_acc;
                        red_acc *= tile_size[d] as usize;
                    } else {
                        perm_strides[d] = out_acc;
                        out_acc *= tile_size[d] as usize;
                    }
                }
            }
            items_buf.set_len(
                (tile_size.as_ref().iter().product::<u64>() * item_dtype.itemsize() as u64)
                    as usize,
            );
            let items_buf = items_buf.as_mut_slice();
            // Read the tile into its `(output, reduced)` permuted layout.
            // SAFETY: `perm_strides` describes the tile's `(output, reduced)` sub-region within
            // `items_buf`, which is sized to the full tile.
            let mut inner_out_buf =
                unsafe { StridedBuf::from_slice_mut(items_buf, perm_strides.as_ref()) };
            inner_array.read_data(tile.as_ref(), context, Some(&mut inner_out_buf))?;
            drop(inner_out_buf);

            // Output-iterator setup. `tile_out_shape` is the tile's output sub-region;
            // `tile_state_base` shifts `state_buf` to its first slot. Items strides come
            // from the permuted layout (innermost output stride == reduction_size).
            let items_buf_strides_for_out_iter = perm_strides
                .as_ref()
                .iter()
                .zip(is_reduced.as_ref())
                .filter_map(|(&s, &reduced)| reduced.not().then_some(s))
                .collect_dim_vec::<OuterD>(out_ndim);

            let tile_out_shape = (0..inner_ndim)
                .filter(|&d| !is_reduced[d])
                .map(|d| tile_size[d])
                .collect_dim_vec::<OuterD>(out_ndim);
            let state_offset = (0..inner_ndim)
                .filter(|&d| !is_reduced[d])
                .enumerate()
                .map(|(out_d, d)| {
                    (tile[d].start - inner_range_full[d].start) as usize * state_strides[out_d]
                })
                .sum::<usize>();
            let tile_state_base = unsafe { state_buf.as_mut_ptr().add(state_offset) };

            reduce_tile_fn(ReduceTileArgs {
                tile_out_shape,
                items_buf: items_buf.as_ptr(),
                items_buf_strides: items_buf_strides_for_out_iter,
                state_buf: tile_state_base,
                state_buf_strides: state_strides.clone(),
                merge_into_existing: bulk_initialized,
                tile_reduction_size,
                bulk_base_item_idx,
            });

            if tile_reduction_size > 0 {
                bulk_base_item_idx += tile_reduction_size as u64;
                bulk_initialized = true;
                state_initialized = true;
            }
        }

        debug_assert_eq!(
            bulk_base_item_idx, full_reduction_size,
            "bulk did not fold each of its outputs over the full reduced stream",
        );
    }

    // finalize_state. Every output was fully reduced inside its own bulk, so each cell
    // folded exactly `full_reduction_size` items.
    let reduction_size_overall = full_reduction_size;
    // CAREFUL: state_buf and out_ptr may alias
    let state_ptr = state_buf.as_mut_ptr();
    // From here on the state/output buffers are touched only through `state_ptr` and
    // `out_ptr`. Dont use `state_buf`.
    finalize_state_fn(FinalizeStateArgs {
        out_shape,
        state_buf: state_ptr,
        state_buf_strides: state_strides,
        out_buf: out_ptr,
        out_buf_strides: out_strides,
        reduction_size_overall,
        state_initialized,
    });
    Ok(out)
}

struct ReduceTileArgs<D: Dimension> {
    tile_out_shape: D::Vec<u64>,
    items_buf: *const u8,
    items_buf_strides: D::Vec<usize>,
    state_buf: *mut u8,
    state_buf_strides: D::Vec<usize>,
    merge_into_existing: bool,
    tile_reduction_size: usize,
    bulk_base_item_idx: u64,
}
fn reduce_tile<T, K, D>(kernel: &K, args: ReduceTileArgs<D>)
where
    T: Dtyped,
    K: ReductionOpKernel<T, Output: Dtyped>,
    D: Dimension,
{
    let ReduceTileArgs {
        tile_out_shape,
        items_buf,
        items_buf_strides,
        state_buf,
        state_buf_strides,
        merge_into_existing,
        tile_reduction_size,
        bulk_base_item_idx,
    } = args;

    let out_iter = NdIter::builder(tile_out_shape)
        .with_strides_ptr_ext(items_buf_strides, items_buf)
        .with_strides_ptr_mut_ext(state_buf_strides, state_buf)
        .build();

    // The first tile of a bulk seeds each output slot; later tiles of the same
    // bulk merge their partial fold into it. (Snapshot the flag so mutating
    // `bulk_initialized` after the fold doesn't clash with the closure's borrow.)
    // let merge_into_existing = bulk_initialized;
    // Store one cell's freshly-folded tile state into its slot. `state_ptr` is
    // this cell's slot in `state_buf`.
    let store = |cell_state: K::State, state_ptr: *mut MaybeUninit<K::State>| {
        // SAFETY: `state_ptr` is this cell's slot in `state_buf`.
        let slot = unsafe { &mut *state_ptr };
        if merge_into_existing {
            // CAREFUL: with `state_in_out_buf`, `slot` aliases the output buffer;
            // read the state out before writing the merged result back into the
            // same bytes.
            let prev = unsafe { slot.assume_init_read() };
            slot.write(kernel.merge_states(prev, cell_state));
        } else {
            slot.write(cell_state);
        }
    };
    let out_iter = out_iter.map(|(_out_idx, (src_base, state_ptr))| {
        // SAFETY: `src_base` points at this cell's contiguous reduced stream
        // of `reduction_size` items in the permuted (output, reduced) buffer.
        let src = unsafe { std::slice::from_raw_parts(src_base.cast::<T>(), tile_reduction_size) };
        let state_ptr = state_ptr.cast::<MaybeUninit<K::State>>();
        (src, state_ptr)
    });

    const LANES: usize = 16;
    if tile_reduction_size >= LANES {
        for (src, state_ptr) in out_iter {
            let state = fold_cell::<T, K, LANES>(kernel, src, bulk_base_item_idx);
            store(state, state_ptr);
        }
    } else if tile_reduction_size > 0 {
        for (src, state_ptr) in out_iter {
            let state = fold_cell::<T, K, 1>(kernel, src, bulk_base_item_idx);
            store(state, state_ptr);
        }
    }
}

/// Fold one output cell's contiguous reduced stream (`items`, which must hold at least
/// `LANES` items) into a single accumulator.
#[inline(always)]
fn fold_cell<T, K, const LANES: usize>(kernel: &K, items: &[T], base: u64) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    let n = items.len();
    let mut i = 0;

    // Seed one accumulator per lane from the first LANES items.
    debug_assert!(n >= LANES);
    let mut states: [K::State; LANES] = array_from_fn_inline(|b| {
        let item = unsafe { *items.get_unchecked(b) };
        kernel.init_state(Some((item, base + b as u64)))
    });
    i += LANES;

    // Process the main bulk of the stream in LANES-sized chunks.
    while i + LANES <= n {
        let items: [T; LANES] = array_from_fn_inline(|b| unsafe { *items.get_unchecked(i + b) });
        let mut it = states.into_iter();
        states = array_from_fn_inline(|b| {
            let state = it.next().unwrap();
            kernel.update_state(state, items[b], base + (i + b) as u64)
        });
        i += LANES;
    }

    // merge the LANES states to a single one
    let mut state = merge_states::<T, K, LANES>(kernel, states);

    // Fold any remaining tail sequentially.
    while i < n {
        let item = unsafe { *items.get_unchecked(i) };
        state = kernel.update_state(state, item, base + i as u64);
        i += 1;
    }
    state
}

/// Collapse `LANES` lane accumulators into one via a bottom-up pairwise tree (dependency
/// depth `log2(LANES)`). `LANES` must be a power of two (the dispatch only picks 4/8/16).
#[inline(always)]
fn merge_states<T, K, const LANES: usize>(kernel: &K, states: [K::State; LANES]) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    // TODO: this code is not panic-safe.
    assert!(LANES > 0 && LANES.is_power_of_two());
    let mut states = states.map_inline(MaybeUninit::new);
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for j in 0..width {
            let a = unsafe { states[j].assume_init_read() };
            let b = unsafe { states[j + width].assume_init_read() };
            states[j].write(kernel.merge_states(a, b));
        }
    }
    unsafe { states[0].assume_init_read() }
}

struct FinalizeStateArgs<D: Dimension> {
    out_shape: D::Vec<u64>,
    state_buf: *mut u8,
    state_buf_strides: D::Vec<usize>,
    out_buf: *mut u8,
    out_buf_strides: D::Vec<usize>,
    reduction_size_overall: u64,
    state_initialized: bool,
}
fn finalize_states<T, K, D>(kernel: &K, args: FinalizeStateArgs<D>)
where
    K: ReductionOpKernel<T>,
    D: Dimension,
{
    let FinalizeStateArgs {
        out_shape,
        state_buf,
        state_buf_strides,
        out_buf,
        out_buf_strides,
        reduction_size_overall,
        state_initialized,
    } = args;

    if state_initialized {
        // CAREFUL: state_ptr and out_ptr may alias
        let out_iter = NdIter::builder(out_shape)
            .with_strides_ptr_mut_ext(state_buf_strides, state_buf)
            .with_strides_ptr_mut_ext(out_buf_strides, out_buf)
            .build();
        for (_idx, (state, out_ptr)) in out_iter {
            // CAREFUL: state and out_ptr may alias
            let res = {
                let state = state.cast::<MaybeUninit<K::State>>();
                let state = unsafe { (&*state).assume_init_read() };
                kernel.finalize_state(state, reduction_size_overall)
            };
            unsafe { out_ptr.cast::<K::Output>().write_unaligned(res) };
        }
    } else {
        // Empty reduction: write the empty-stream result to every output.
        let out_iter = NdIter::builder(out_shape)
            .with_strides_ptr_mut_ext(out_buf_strides, out_buf)
            .build();
        // debug_assert!( // TODO
        //     out_nitems == 0 || full_reduction_size == 0,
        //     "output left unseeded despite a non-empty output and non-empty reduction",
        // );
        for (_idx, out_ptr) in out_iter {
            let state = kernel.init_state(None);
            let res = kernel.finalize_state(state, 0);
            unsafe { out_ptr.cast::<K::Output>().write_unaligned(res) };
        }
    }
}

/// Choose the per-call read tile for a reduction, given the source array's `spec`.
///
/// The reduction kernel walks the reduced axes inside each tile, so the tile must keep enough volume
/// on the reduced dims to amortize per-tile overhead. Which strategy runs is gated by
/// [`USE_NEW_READ_SCALING`]:
///
/// - Balanced (default): seed from `block_shape` and scale with a reduced-dims-first order (each
///   dim rightmost first), so the reduced dims get first claim on the budget.
/// - Priority (parked): scale the tile normally
///   ([`read_shape_heuristic`](ArraySpec::read_shape_heuristic)); if that leaves the reduced dims'
///   volume below `min(REDUCED_TILE_MIN_NITEMS, full_reduced_extent)`, rebuild it from two
///   independent scales over the disjoint reduced / non-reduced dim groups (reduced dims to that
///   floor, non-reduced dims to the remaining budget `max_nitems / floor`), which merge cleanly
///   since [`read_shape_scale_dims`](ArraySpec::read_shape_scale_dims) scales only its selected dims.
///
/// `tile_max_shape` is the per-dim length of the region being read, `array_shape` the full source
/// shape, and `is_reduced[d]` marks the reduced dims - all with one entry per source dim.
fn reduction_tile_shape<D: Dimension>(
    spec: &ArraySpec,
    is_reduced: &[bool],
    tile_max_shape: &[u64],
    array_shape: &[u64],
    itemsize: Itemsize,
) -> D {
    let ndim = tile_max_shape.len();
    if USE_NEW_READ_SCALING {
        const REDUCED_TILE_MIN_NITEMS: u64 = 512;
        let full_reduced = (0..ndim)
            .filter(|&d| is_reduced[d])
            .fold(1u64, |v, d| v.saturating_mul(tile_max_shape[d]));
        let reduced_tile_floor = REDUCED_TILE_MIN_NITEMS.min(full_reduced);

        let tile_shape = spec.read_shape_heuristic::<D>(tile_max_shape, array_shape, itemsize);
        let reduced_tile_volume = (0..ndim)
            .filter(|&d| is_reduced[d])
            .fold(1u64, |v, d| v.saturating_mul(tile_shape[d]));
        if reduced_tile_volume >= reduced_tile_floor {
            return tile_shape;
        }

        // The regular tile starved the reduced dims: rebuild from two disjoint-group scales.
        let max_nitems = spec.read_size().nitems(itemsize).1;
        let non_reduced_target = (max_nitems / reduced_tile_floor).max(1);
        let reduced_tile = spec.read_shape_scale_dims::<D>(
            tile_max_shape,
            array_shape,
            (reduced_tile_floor, reduced_tile_floor),
            |d| is_reduced[d],
        );
        let non_reduced_tile = spec.read_shape_scale_dims::<D>(
            tile_max_shape,
            array_shape,
            (non_reduced_target, non_reduced_target),
            |d| !is_reduced[d],
        );
        D::from_fn(ndim, |d| {
            if is_reduced[d] {
                reduced_tile[d]
            } else {
                non_reduced_tile[d]
            }
        })
    } else {
        // Balanced (default): seed from the source block shape and scale with reduced dims first
        // (rightmost first), then non-reduced (rightmost first), so the reduced dims claim the
        // budget before the free dims. Uses the balanced `scale_read_shape` (see there).
        let block_shape = spec.block_shape();
        let mut read_shape = D::from_fn(ndim, |dim| block_shape[dim] as u64);
        let order = (0..ndim)
            .rev()
            .filter(|&d| is_reduced[d])
            .chain((0..ndim).rev().filter(|&d| !is_reduced[d]));
        scale_read_shape(
            read_shape.as_mut_slice(),
            tile_max_shape,
            array_shape,
            spec.read_size().nitems(itemsize),
            order,
        );
        read_shape
    }
}

/// Emits the wrapper storage struct (`$Op<S, D>` or `$Op<S>`), its `ArrayStorage` impl,
/// and the kernel struct declaration (`pub(crate) struct $Kernel;` - with fields if
/// `extra_args` were declared). The `ReductionOpKernel` impl for the kernel is still
/// written by hand next to the macro invocation.
///
/// Invocation shape:
/// ```ignore
/// define_reduction_op!(
///     /// docs...
///     Op, Kernel { extra_arg: Type, ... },           // `{ ... }` is optional
///     where { S: ArrayStorageTyped, S::Item: ... },  // full where-clause, must end with `,`
///     output = <S::Item as Trait>::Output,
///     single_axis,                                 // optional; omit for multi-axis
/// );
/// ```
macro_rules! define_reduction_op {
    // single-axis variant
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident $( { $($extra_arg:ident: $extra_ty:ty),+ $(,)? } )?,
        where { $($where_:tt)+ }
        output = $output_ty:ty,
        single_axis $(,)?
    ) => {
        struct $Kernel { $($($extra_arg: $extra_ty),+)? }

        $(#[$meta])*
        pub struct $Op<S>(
            crate::ops::reduction::ReductionOp<S, $Kernel, <S::Dimension as crate::Dimension>::Smaller>,
        )
        where
            S: crate::ArrayStorage;

        impl<S> $Op<S>
        where
            $($where_)+
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S, axis: usize $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<Self> {
                let kernel = $Kernel { $($($extra_arg,)+)? };
                Ok(Self(crate::ops::reduction::ReductionOp::new(array, kernel, &[axis])?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>, axis: usize $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage(), axis $($(, $extra_arg)+)?).map(crate::Array::from_storage)
            }
        }

        impl<S> crate::ArrayStorage for $Op<S>
        where
            $($where_)+
        {
            type ElementType = crate::Ty<$output_ty>;
            type Dimension = <S::Dimension as crate::Dimension>::Smaller;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> ArrayStorageInfo<'_> {
                ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }
            crate::ops::impl_dimension_change_default!();
            crate::ops::impl_element_type_change_default!();
        }
    };

    // multi-axis variant
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident $( { $($extra_arg:ident: $extra_ty:ty),+ $(,)? } )?,
        where { $($where_:tt)+ }
        output = $output_ty:ty $(,)?
    ) => {
        struct $Kernel { $($($extra_arg: $extra_ty),+)? }

        $(#[$meta])*
        pub struct $Op<S: crate::ArrayStorage, D>(crate::ops::reduction::ReductionOp<S, $Kernel, D>);

        impl<S, D> $Op<S, D>
        where
            $($where_)+
            D: crate::Dimension,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new<Ax>(array: S, axes: Ax $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<Self>
            where
                Ax: crate::ops::AxesArg<ReducedDimension<S::Dimension> = D>,
            {
                let kernel = $Kernel { $($($extra_arg,)+)? };
                Ok(Self(crate::ops::reduction::ReductionOp::new(array, kernel, axes)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array<Ax>(array: crate::Array<S>, axes: Ax $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<crate::Array<Self>>
            where
                Ax: crate::ops::AxesArg<ReducedDimension<S::Dimension> = D>,
            {
                Self::new(array.into_storage(), axes $($(, $extra_arg)+)?).map(crate::Array::from_storage)
            }
        }

        impl<S, D> crate::ArrayStorage for $Op<S, D>
        where
            $($where_)+
            D: crate::Dimension,
        {
            type ElementType = crate::Ty<$output_ty>;
            type Dimension = D;
            crate::storage::impl_array_storage_forward!(<S, D>);

            fn info(&self) -> ArrayStorageInfo<'_> {
                ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }

            type DimensionChange<NewD: crate::Dimension> = $Op<S, NewD>;
            #[inline]
            fn dimension_change<NewD: crate::Dimension>(
                self,
            ) -> crate::error::Result<Self::DimensionChange<NewD>> {
                Ok($Op(self.0.dimension_change()?))
            }

            crate::ops::impl_element_type_change_default!();
        }
    };
}

/// Public scalar-level traits implemented by primitive element types and consumed by the
/// reduction op storage wrappers ([`Sum`], [`Product`], [`Mean`], [`Variance`], ...).
///
/// Each trait describes how to fold a stream of `Self` values into a result. Some are simple
/// (e.g. `Sum` only needs an `Output` accumulator); others (`Mean`, `Variance`)
/// expose a richer state machine (`type State`, `init`, `update`, `finalize`) because the
/// final result is not just the accumulator.
///
/// Max/Min/argmax/argmin do **not** appear here - those ops are bounded directly by
/// [`crate::scalar::Maximum`] / [`crate::scalar::Minimum`] / [`PartialOrd`].
pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    /// Scalar kernel trait for the element-wise `sum` reduction.
    ///
    /// Integer types accumulate into a wider output to reduce overflow risk (`i8..=i64 ->
    /// i64`; `u8..=u64` and `bool -> u64`). Floating-point and complex types keep their input
    /// width (`f16 -> f16`, `f32 -> f32`, `Complex<f32> -> Complex<f32>`, ...), matching NumPy.
    pub trait Sum {
        /// The sum element type: `i64`/`u64` for integers and `bool`, otherwise the input type.
        type Output;
        /// Return the initial accumulator (zero).
        fn init() -> Self::Output;
        /// Fold `item` into the running sum.
        fn update(state: Self::Output, item: Self) -> Self::Output;
        /// Combine two partial sums (used to merge interleaved lane accumulators).
        fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output;
    }

    macro_rules! impl_sum {
        ($item_ty:ty, $output_ty:ty) => {
            impl Sum for $item_ty {
                type Output = $output_ty;

                #[inline(always)]
                fn init() -> Self::Output {
                    <f32 as crate::scalar::Cast<Self::Output>>::cast(-0.0)
                }
                #[inline(always)]
                fn update(state: Self::Output, item: Self) -> Self::Output {
                    state + <_ as crate::scalar::Cast<Self::Output>>::cast(item)
                }
                #[inline(always)]
                fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output {
                    a + b
                }
            }
        };
    }
    impl_sum!(i8, i64);
    impl_sum!(i16, i64);
    impl_sum!(i32, i64);
    impl_sum!(i64, i64);
    impl_sum!(u8, u64);
    impl_sum!(u16, u64);
    impl_sum!(u32, u64);
    impl_sum!(u64, u64);
    #[cfg(feature = "half")]
    impl_sum!(f16, f16);
    impl_sum!(f32, f32);
    impl_sum!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f32>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f64>, Complex<f64>);
    impl_sum!(bool, u64);

    /// Scalar kernel trait for the element-wise `product` reduction.
    ///
    /// Integer types accumulate into a wider output to reduce overflow risk (`i8..=i64 ->
    /// i64`; `u8..=u64 -> u64`). Floating-point and complex types keep their input width
    /// (`f16 -> f16`, `f32 -> f32`, `Complex<f32> -> Complex<f32>`, ...), matching NumPy.
    pub trait Product {
        /// The product element type: `i64`/`u64` for integers, otherwise the input type.
        type Output;
        /// Return the initial accumulator (one).
        fn init() -> Self::Output;
        /// Fold `item` into the running product.
        fn update(state: Self::Output, item: Self) -> Self::Output;
        /// Combine two partial products (used to merge interleaved lane accumulators).
        fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output;
    }
    macro_rules! impl_product {
        ($item_ty:ty, $output_ty:ty) => {
            impl Product for $item_ty {
                type Output = $output_ty;

                #[inline(always)]
                fn init() -> Self::Output {
                    <i32 as crate::scalar::Cast<Self::Output>>::cast(1)
                }
                #[inline(always)]
                fn update(state: Self::Output, item: Self) -> Self::Output {
                    state * <_ as crate::scalar::Cast<Self::Output>>::cast(item)
                }
                #[inline(always)]
                fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output {
                    a * b
                }
            }
        };
    }
    impl_product!(i8, i64);
    impl_product!(i16, i64);
    impl_product!(i32, i64);
    impl_product!(i64, i64);
    impl_product!(u8, u64);
    impl_product!(u16, u64);
    impl_product!(u32, u64);
    impl_product!(u64, u64);
    #[cfg(feature = "half")]
    impl_product!(f16, f16);
    impl_product!(f32, f32);
    impl_product!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f32>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f64>, Complex<f64>);

    /// Scalar kernel trait for the element-wise `mean` reduction.
    ///
    /// The mean is computed as the sum divided by the count. Integer and `bool` inputs promote
    /// to `f64`; floating-point and complex inputs keep their input width (`f16 -> f16`,
    /// `f32 -> f32`, `Complex<f32> -> Complex<f32>`, ...), matching NumPy.
    ///
    /// The count is tracked **outside** the accumulator: callers thread the number of
    /// folded items into [`finalize`](Self::finalize) themselves.
    pub trait Mean {
        /// The output element type: `f64` for integer and `bool` inputs, otherwise the input type.
        type Output;
        /// Accumulator state - the running sum.
        type State;
        /// Return the initial (empty) accumulator.
        fn init() -> Self::State;
        /// Fold `item` into the running sum.
        fn update(state: Self::State, item: Self) -> Self::State;
        /// Combine two partial sums (used to merge interleaved lane accumulators).
        fn merge_states(a: Self::State, b: Self::State) -> Self::State;
        /// Finalize `state` into the mean. Returns `None` if `nitems == 0`; otherwise
        /// returns `state / nitems` (cast to the output domain).
        fn finalize(state: Self::State, nitems: u64) -> Option<Self::Output>;
    }
    macro_rules! impl_mean {
        ($item_ty:ty, $output_ty:ty) => {
            impl Mean for $item_ty {
                type Output = $output_ty;
                type State = <Self as Sum>::Output;

                #[inline(always)]
                fn init() -> Self::State {
                    <Self as Sum>::init()
                }
                #[inline(always)]
                fn update(state: Self::State, item: Self) -> Self::State {
                    <Self as Sum>::update(state, item)
                }
                #[inline(always)]
                fn merge_states(a: Self::State, b: Self::State) -> Self::State {
                    <Self as Sum>::merge_states(a, b)
                }
                #[inline(always)]
                fn finalize(state: Self::State, nitems: u64) -> Option<Self::Output> {
                    if nitems == 0 {
                        return None;
                    }
                    Some(
                        <_ as crate::scalar::Cast<Self::Output>>::cast(state)
                            / <_ as crate::scalar::Cast<Self::Output>>::cast(nitems),
                    )
                }
            }
        };
    }
    impl_mean!(i8, f64);
    impl_mean!(i16, f64);
    impl_mean!(i32, f64);
    impl_mean!(i64, f64);
    impl_mean!(u8, f64);
    impl_mean!(u16, f64);
    impl_mean!(u32, f64);
    impl_mean!(u64, f64);
    #[cfg(feature = "half")]
    impl_mean!(f16, f16);
    impl_mean!(f32, f32);
    impl_mean!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f32>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f64>, Complex<f64>);
    impl_mean!(bool, f64);

    /// Welford accumulator used by [`Variance`]. The count of folded items is tracked
    /// **inside** this struct (`count`) so each interleaved lane accumulator counts only
    /// the items it saw and [`Variance::merge_states`] can recombine lanes with Chan's
    /// algorithm.
    #[derive(Clone, Copy)]
    pub struct VarianceState<M> {
        /// Running mean in the type-specific accumulator domain.
        mean: M,
        /// Running sum of squared deviations from the mean.
        m2: f64,
        /// Number of items folded into this accumulator.
        count: u64,
    }

    /// Scalar kernel trait for the `var` (variance) and `std` (standard deviation) reductions.
    ///
    /// The degree-of-freedom correction is controlled by `ddof` passed to [`finalize`]:
    /// use `0.0` for population variance (`N` denominator) and `1.0` for sample variance
    /// (`N-1` denominator).
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// [`finalize`]: Variance::finalize
    pub trait Variance {
        /// The output element type - always a real `Float`: `f64` for integer and `bool`
        /// inputs, the input width for real floats (`f16`/`f32`/`f64`), and the real component
        /// type for complex inputs (`Complex<f32> -> f32`, `Complex<f64> -> f64`).
        type Output;
        /// Welford accumulator state.
        type State;
        /// Return the initial (empty) accumulator.
        fn init() -> Self::State;
        /// Fold `item` into the running Welford accumulator. `idx` is ignored: the count is
        /// tracked inside the state so interleaved lanes stay correct.
        fn update(state: Self::State, item: Self, idx: u64) -> Self::State;
        /// Combine two Welford accumulators computed over disjoint subsets (Chan's parallel
        /// algorithm). Associative + commutative up to float error.
        fn merge_states(a: Self::State, b: Self::State) -> Self::State;
        /// Finalize `state` into the variance using `ddof` degrees-of-freedom correction.
        /// `nitems` is the total number of elements folded in.
        ///
        /// Returns `NaN` if the effective denominator (`nitems - ddof`) is non-positive.
        fn finalize(state: Self::State, ddof: f64, nitems: u64) -> Self::Output;
    }
    macro_rules! impl_variance {
        ($item_ty:ty => $output_ty:ty, MeanT = $mean_ty:ty, |$delta:ident, $delta2:ident| $m2_expr:expr) => {
            impl Variance for $item_ty {
                type Output = $output_ty;
                type State = VarianceState<$mean_ty>;

                #[inline(always)]
                fn init() -> Self::State {
                    VarianceState {
                        mean: <i32 as crate::scalar::Cast<$mean_ty>>::cast(0),
                        m2: 0.0,
                        count: 0,
                    }
                }
                #[inline(always)]
                fn update(mut state: Self::State, item: Self, _idx: u64) -> Self::State {
                    state.count += 1;
                    let x = <_ as crate::scalar::Cast<$mean_ty>>::cast(item);
                    let $delta = x - state.mean;
                    state.mean += $delta / state.count as f64;
                    let $delta2 = x - state.mean;
                    state.m2 += $m2_expr;
                    state
                }
                #[inline(always)]
                fn merge_states(a: Self::State, b: Self::State) -> Self::State {
                    if a.count == 0 {
                        return b;
                    }
                    if b.count == 0 {
                        return a;
                    }
                    let na = a.count;
                    let nb = b.count;
                    let n = na + nb;
                    // Chan's parallel Welford combine. The m2 cross-term reuses the per-type
                    // squared-deviation expression with `delta2 = delta` (real: delta^2;
                    // complex: |delta|^2).
                    let $delta = b.mean - a.mean;
                    let mean = a.mean
                        + $delta
                            * <f64 as crate::scalar::Cast<$mean_ty>>::cast(nb as f64 / n as f64);
                    let $delta2 = $delta;
                    let m2 = a.m2 + b.m2 + ($m2_expr) * (na as f64 * nb as f64 / n as f64);
                    VarianceState {
                        mean,
                        m2,
                        count: a.count + b.count,
                    }
                }
                #[inline(always)]
                fn finalize(state: Self::State, ddof: f64, nitems: u64) -> Self::Output {
                    let denom = nitems as f64 - ddof;
                    let res = if denom <= 0.0 {
                        f64::NAN
                    } else {
                        state.m2 / denom
                    };
                    <_ as crate::scalar::Cast<Self::Output>>::cast(res)
                }
            }
        };
    }
    impl_variance!(i8 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(i16 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(i32 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(i64 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(u8 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(u16 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(u32 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(u64 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    #[cfg(feature = "half")]
    impl_variance!(f16 => f16, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(f32 => f32, MeanT = f64, |delta, delta2| delta * delta2);
    impl_variance!(f64 => f64, MeanT = f64, |delta, delta2| delta * delta2);
    #[cfg(feature = "num-complex")]
    impl_variance!(Complex<f32> => f32, MeanT = Complex<f64>, |delta, delta2| delta.re
        * delta2.re
        + delta.im * delta2.im);
    #[cfg(feature = "num-complex")]
    impl_variance!(Complex<f64> => f64, MeanT = Complex<f64>, |delta, delta2| delta.re
        * delta2.re
        + delta.im * delta2.im);
    impl_variance!(bool => f64, MeanT = f64, |delta, delta2| delta * delta2);
}

define_reduction_op!(
    /// Reduces one or more axes by taking the maximum element.
    ///
    /// For **float** types, `NaN` is propagated: if any element is `NaN`, the result
    /// is `NaN`. This matches the element-wise [`Maximum`](crate::ops::Maximum) op and
    /// `numpy.max` (not `numpy.nanmax`, which would ignore `NaN`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::max()`](crate::Array::max).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all axes -> scalar
    /// let scalar = Array::compact_ndarray(&nd)?
    ///     .max((0, 1)).to_ndarray()?;
    /// assert_eq!(scalar[[]], 6);
    ///
    /// // Reduce axis 0 -> shape [3]
    /// let col_max = Array::compact_ndarray(&nd)?
    ///     .max(0).to_ndarray()?;
    /// assert_eq!(col_max.as_slice().unwrap(), &[4, 5, 6]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Max,
    MaxKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Maximum<Output = S::Item> + Dtyped,
    }
    output = S::Item,
);
impl<T> ReductionOpKernel<T> for MaxKernel
where
    T: crate::scalar::Maximum<Output = T>,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        init_item.unwrap().0
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        state.maximum(item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a.maximum(b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces one or more axes by taking the minimum element.
    ///
    /// For **float** types, `NaN` is propagated: if any element is `NaN`, the result
    /// is `NaN`. This matches the element-wise [`Minimum`](crate::ops::Minimum) op and
    /// `numpy.min` (not `numpy.nanmin`, which would ignore `NaN`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::min()`](crate::Array::min).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all axes -> scalar
    /// let scalar = Array::compact_ndarray(&nd)?
    ///     .min((0, 1)).to_ndarray()?;
    /// assert_eq!(scalar[[]], 1);
    ///
    /// // Reduce axis 0 -> shape [3]
    /// let col_min = Array::compact_ndarray(&nd)?
    ///     .min(0).to_ndarray()?;
    /// assert_eq!(col_min.as_slice().unwrap(), &[1, 2, 3]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Min,
    MinKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Minimum<Output = S::Item> + Dtyped,
    }
    output = S::Item,
);
impl<T> ReductionOpKernel<T> for MinKernel
where
    T: crate::scalar::Minimum<Output = T>,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        init_item.unwrap().0
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        state.minimum(item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a.minimum(b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces a single axis by returning the index of the maximum element.
    ///
    /// Output dtype is `u64`.
    ///
    /// Unlike [`Max`], this op accepts only a single axis. If multiple elements share
    /// the maximum value, the index of the *first* such element is returned, matching
    /// `numpy.argmax`. For **float** types, `NaN` propagates: if any element along the
    /// reduced axis is `NaN`, the returned index is that of some `NaN` (not necessarily the
    /// first). This differs from `numpy.argmax`, which returns the index of the *first* `NaN`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::argmax()`](crate::Array::argmax).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of max along axis 1 (per row) -> shape [2]
    /// let idx = Array::compact_ndarray(&nd)?
    ///     .argmax(1).to_ndarray()?;
    /// assert_eq!(idx.as_slice().unwrap(), &[1, 2]); // max of row 0 at col 1, row 1 at col 2
    ///
    /// // Index of max along axis 0 (per column) -> shape [3]
    /// let col_idx = Array::compact_ndarray(&nd)?
    ///     .argmax(0).to_ndarray()?;
    /// assert_eq!(col_idx.as_slice().unwrap(), &[1, 0, 1]); // max of col 0 at row 1, col 1 at row 0, col 2 at row 1
    /// # Ok::<(), jix::Error>(())
    /// ```
    ArgMax,
    ArgMaxKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: PartialOrd,
    }
    output = u64,
    single_axis,
);
// `item != item` / `bv != bv` are deliberate `NaN` tests (a value is `NaN` iff it is not
// equal to itself), so the `eq_op` lint does not apply.
#[allow(clippy::eq_op)]
impl<T> ReductionOpKernel<T> for ArgMaxKernel
where
    T: PartialOrd,
{
    type Output = u64;
    /// `(best_idx, best_val)`.
    type State = (u64, T);

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let (item, idx) = init_item.unwrap();
        (idx, item)
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        // `NaN` propagates: `item != item` holds only for `NaN`, so a `NaN` becomes - and,
        // being neither `>` nor `!=`-equal to a later value, sticks as - the running best.
        // For integer types the `NaN` term folds away, leaving the plain `item > best_val`.
        let (best_idx, best_val) = state;
        if item > best_val || item != item {
            (idx, item)
        } else {
            (best_idx, best_val)
        }
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        // The larger value wins, and any `NaN` wins (propagating as in `update_state`). On an
        // exact value tie the *smaller* index wins - together with `update_state` keeping the
        // earlier index on ties, this makes argmax report the first occurrence of the maximum,
        // matching `numpy.argmax`. The two subsets folded into `a`/`b` need not be contiguous
        // index ranges (lane interleaving, tree merge), so the tie-break must compare indices
        // rather than assume one side is "earlier". A `NaN` tie's index is still unspecified.
        let (ai, av) = a;
        let (bi, bv) = b;
        if bv > av || bv != bv {
            (bi, bv)
        } else if av > bv || av != av {
            (ai, av)
        } else {
            // av == bv and neither is `NaN`: keep the earlier index.
            if bi < ai {
                (bi, bv)
            } else {
                (ai, av)
            }
        }
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        let (best_idx, _best_val) = state;
        best_idx
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces a single axis by returning the index of the minimum element.
    ///
    /// Output dtype is `u64`.
    ///
    /// Unlike [`Min`], this op accepts only a single axis. If multiple elements share
    /// the minimum value, the index of the *first* such element is returned, matching
    /// `numpy.argmin`. For **float** types, `NaN` propagates: if any element along the
    /// reduced axis is `NaN`, the returned index is that of some `NaN` (not necessarily the
    /// first). This differs from `numpy.argmin`, which returns the index of the *first* `NaN`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::argmin()`](crate::Array::argmin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of min along axis 1 (per row) -> shape [2]
    /// let idx = Array::compact_ndarray(&nd)?
    ///     .argmin(1).to_ndarray()?;
    /// assert_eq!(idx.as_slice().unwrap(), &[0, 1]); // min of row 0 at col 0, row 1 at col 1
    ///
    /// // Index of min along axis 0 (per column) -> shape [3]
    /// let col_idx = Array::compact_ndarray(&nd)?
    ///     .argmin(0).to_ndarray()?;
    /// assert_eq!(col_idx.as_slice().unwrap(), &[0, 1, 0]); // min of col 0 at row 0, col 1 at row 1, col 2 at row 0
    /// # Ok::<(), jix::Error>(())
    /// ```
    ArgMin,
    ArgMinKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: PartialOrd,
    }
    output = u64,
    single_axis,
);
// `item != item` / `bv != bv` are deliberate `NaN` tests, so `eq_op` does not apply.
#[allow(clippy::eq_op)]
impl<T> ReductionOpKernel<T> for ArgMinKernel
where
    T: PartialOrd,
{
    type Output = u64;
    /// `(best_idx, best_val)`.
    type State = (u64, T);

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let (item, idx) = init_item.unwrap();
        (idx, item)
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        // `NaN` propagates (see [`ArgMaxKernel::update_state`]): `item != item` holds only
        // for `NaN`, so a `NaN` becomes and sticks as the running best. For integer types the
        // `NaN` term folds away, leaving the plain `item < best_val`.
        let (best_idx, best_val) = state;
        if item < best_val || item != item {
            (idx, item)
        } else {
            (best_idx, best_val)
        }
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        // The smaller value wins, and any `NaN` wins (propagating as in `update_state`). On an
        // exact value tie the *smaller* index wins - together with `update_state` keeping the
        // earlier index on ties, this makes argmin report the first occurrence of the minimum,
        // matching `numpy.argmin`. The two subsets folded into `a`/`b` need not be contiguous
        // index ranges (lane interleaving, tree merge), so the tie-break must compare indices
        // rather than assume one side is "earlier". A `NaN` tie's index is still unspecified.
        let (ai, av) = a;
        let (bi, bv) = b;
        if bv < av || bv != bv {
            (bi, bv)
        } else if av < bv || av != av {
            (ai, av)
        } else {
            // av == bv and neither is `NaN`: keep the earlier index.
            if bi < ai {
                (bi, bv)
            } else {
                (ai, av)
            }
        }
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        let (best_idx, _best_val) = state;
        best_idx
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces one or more axes by summing all elements along those axes.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64`, `bool` | `u64` |
    /// | `f16` | `f16` |
    /// | `f32` | `f32` |
    /// | `f64` | `f64` |
    /// | `Complex<f32>` | `Complex<f32>` |
    /// | `Complex<f64>` | `Complex<f64>` |
    ///
    /// Integer inputs are widened to a 64-bit accumulator to reduce overflow on large
    /// reductions; floating-point and complex inputs keep their width, matching NumPy (except
    /// `bool`, which jix sums into `u64` whereas NumPy uses `int64`).
    /// An empty reduction (zero elements along the reduced axes) returns `0`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sum()`](crate::Array::sum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Sum all elements -> i64
    /// let total = Array::compact_ndarray(&nd)?
    ///     .sum((0, 1)).to_ndarray()?;
    /// assert_eq!(total[[]], 21);
    ///
    /// // Sum along axis 0 -> shape [3]
    /// let col_sums = Array::compact_ndarray(&nd)?
    ///     .sum(0).to_ndarray()?;
    /// assert_eq!(col_sums.as_slice().unwrap(), &[5, 7, 9]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sum,
    SumKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Sum<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Sum>::Output,
);
impl<T> ReductionOpKernel<T> for SumKernel
where
    T: crate::scalar::Sum,
{
    type Output = <T as crate::scalar::Sum>::Output;
    type State = <T as crate::scalar::Sum>::Output;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let mut state = <T as crate::scalar::Sum>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        <T as crate::scalar::Sum>::update(state, item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Sum>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        true
    }
}

define_reduction_op!(
    /// Reduces one or more axes by multiplying all elements along those axes.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64` | `u64` |
    /// | `f16` | `f16` |
    /// | `f32` | `f32` |
    /// | `f64` | `f64` |
    /// | `Complex<f32>` | `Complex<f32>` |
    /// | `Complex<f64>` | `Complex<f64>` |
    ///
    /// Integer inputs are widened to a 64-bit accumulator to reduce overflow on large
    /// reductions; floating-point and complex inputs keep their width, matching NumPy. `bool`
    /// is not supported (NumPy would promote it to `int64`).
    /// An empty reduction (zero elements along the reduced axes) returns `1`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::product()`](crate::Array::product).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Product of all elements -> i64
    /// let total = Array::compact_ndarray(&nd)?
    ///     .product((0, 1)).to_ndarray()?;
    /// assert_eq!(total[[]], 720);
    ///
    /// // Product along axis 0 -> shape [3]
    /// let col_products = Array::compact_ndarray(&nd)?
    ///     .product(0).to_ndarray()?;
    /// assert_eq!(col_products.as_slice().unwrap(), &[4, 10, 18]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Product,
    ProductKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Product<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Product>::Output,
);
impl<T> ReductionOpKernel<T> for ProductKernel
where
    T: crate::scalar::Product,
{
    type Output = <T as crate::scalar::Product>::Output;
    type State = <T as crate::scalar::Product>::Output;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let mut state = <T as crate::scalar::Product>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        <T as crate::scalar::Product>::update(state, item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Product>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        true
    }
}

define_reduction_op!(
    /// Reduces one or more axes by computing the arithmetic mean.
    ///
    /// Integer and `bool` inputs promote to `f64`; floating-point and complex inputs keep
    /// their input width (`f16 -> f16`, `f32 -> f32`, `f64 -> f64`, `Complex<f32> ->
    /// Complex<f32>`, `Complex<f64> -> Complex<f64>`), matching NumPy.
    ///
    /// Reducing an empty slice (zero elements) panics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::mean()`](crate::Array::mean).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Mean of all elements -> f64
    /// let total = Array::compact_ndarray(&nd)?
    ///     .mean((0, 1)).to_ndarray()?;
    /// assert_eq!(total[[]], 3.5);
    ///
    /// // Mean along axis 0 -> shape [3]
    /// let col_means = Array::compact_ndarray(&nd)?
    ///     .mean(0).to_ndarray()?;
    /// assert_eq!(col_means.as_slice().unwrap(), &[2.5, 3.5, 4.5]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Mean,
    MeanKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Mean<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Mean>::Output,
);
impl<T> ReductionOpKernel<T> for MeanKernel
where
    T: crate::scalar::Mean,
{
    type Output = <T as crate::scalar::Mean>::Output;
    type State = <T as crate::scalar::Mean>::State;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let mut state = <T as crate::scalar::Mean>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        <T as crate::scalar::Mean>::update(state, item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Mean>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output {
        <T as crate::scalar::Mean>::finalize(state, nitems).unwrap()
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces one or more axes by computing the variance.
    ///
    /// The variance is real-valued and its dtype matches NumPy: integer and `bool` inputs
    /// promote to `f64`, real floats keep their width (`f16 -> f16`, `f32 -> f32`, `f64 ->
    /// f64`), and complex inputs reduce to their real component type (`Complex<f32> -> f32`,
    /// `Complex<f64> -> f64`), computing `E[|x - mean|^2]`.
    ///
    /// The `ddof` parameter (delta degrees of freedom) adjusts the divisor: the variance
    /// is computed as `sum((x - mean)^2) / (n - ddof)`. Use `ddof=0` for the population
    /// variance and `ddof=1` for the sample variance (Bessel's correction). If
    /// `n - ddof <= 0`, the result is `NaN`.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::var()`](crate::Array::var).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population variance (ddof=0) of all elements -> f64
    /// let var_all = Array::compact_ndarray(&nd)?
    ///     .var((0, 1), 0.0).to_ndarray()?;
    /// assert!((var_all[[]] - 2.9167).abs() < 0.001);
    ///
    /// // Sample variance (ddof=1) along axis 0 -> shape [3]
    /// let col_vars = Array::compact_ndarray(&nd)?
    ///     .var(0, 1.0).to_ndarray()?;
    /// assert_eq!(col_vars.as_slice().unwrap(), &[4.5, 4.5, 4.5]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Variance,
    VarianceKernel { ddof: f64 },
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Variance<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Variance>::Output,
);
impl<T> ReductionOpKernel<T> for VarianceKernel
where
    T: crate::scalar::Variance,
{
    type Output = <T as crate::scalar::Variance>::Output;
    type State = <T as crate::scalar::Variance>::State;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let mut state = <T as crate::scalar::Variance>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        <T as crate::scalar::Variance>::update(state, item, idx)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Variance>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output {
        <T as crate::scalar::Variance>::finalize(state, self.ddof, nitems)
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces one or more axes by computing the standard deviation.
    ///
    /// The standard deviation is real-valued and follows the same dtype rules as [`Variance`]:
    /// integer and `bool` inputs promote to `f64`, real floats keep their width, and complex
    /// inputs reduce to their real component type (`Complex<f32> -> f32`, `Complex<f64> ->
    /// f64`), computing `sqrt(E[|x - mean|^2])`.
    ///
    /// Equivalent to `sqrt(variance)`. The `ddof` parameter has the same meaning as in
    /// [`Variance`]: use `ddof=0` for population std and `ddof=1` for sample std.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::std()`](crate::Array::std).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population std (ddof=0) of all elements -> f64
    /// let std_all = Array::compact_ndarray(&nd)?
    ///     .std((0, 1), 0.0).to_ndarray()?;
    /// assert!((std_all[[]] - 1.7078).abs() < 0.001);
    ///
    /// // Sample std (ddof=1) along axis 0 -> shape [3]
    /// let col_stds = Array::compact_ndarray(&nd)?
    ///     .std(0, 1.0).to_ndarray()?;
    /// assert!((col_stds[[0]] - 2.1213).abs() < 0.001);
    /// # Ok::<(), jix::Error>(())
    /// ```
    StandardDeviation,
    StandardDeviationKernel { ddof: f64 },
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Variance<Output: num_traits::Float + Dtyped>,
    }
    output = <S::Item as crate::scalar::Variance>::Output,
);
impl<T> ReductionOpKernel<T> for StandardDeviationKernel
where
    T: crate::scalar::Variance<Output: num_traits::Float>,
{
    type Output = <T as crate::scalar::Variance>::Output;
    type State = <T as crate::scalar::Variance>::State;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        let mut state = <T as crate::scalar::Variance>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        <T as crate::scalar::Variance>::update(state, item, idx)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Variance>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output {
        let var = <T as crate::scalar::Variance>::finalize(state, self.ddof, nitems);
        <_ as num_traits::Float>::sqrt(var)
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}

define_reduction_op!(
    /// Reduces one or more axes by testing whether all elements are `true`.
    ///
    /// The input array must contain `bool` elements. Output dtype is `bool`.
    /// Returns `true` only when every element is `true`. An empty reduction returns `true`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::all()`](crate::Array::all).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[true, false, true], [true, true, true]];
    ///
    /// // All elements true? -> false (contains false)
    /// let all_true = Array::compact_ndarray(&nd)?
    ///     .all((0, 1)).to_ndarray()?;
    /// assert_eq!(all_true[[]], false);
    ///
    /// // All true along axis 0 (per column) -> shape [3]
    /// let col_all = Array::compact_ndarray(&nd)?
    ///     .all(0).to_ndarray()?;
    /// assert_eq!(col_all.as_slice().unwrap(), &[true, false, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    All,
    AllKernel,
    where {
        S: ArrayStorageTyped<Item = bool>,
    }
    output = bool,
);
impl ReductionOpKernel<bool> for AllKernel {
    type Output = bool;
    type State = bool;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(bool, u64)>) -> Self::State {
        let mut state = true;
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: bool, _idx: u64) -> Self::State {
        state && item
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a && b
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        true
    }
}

define_reduction_op!(
    /// Reduces one or more axes by testing whether any element is `true`.
    ///
    /// The input array must contain `bool` elements. Output dtype is `bool`.
    /// Returns `true` when at least one element is `true`. An empty reduction returns `false`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::any()`](crate::Array::any).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[false, false, false], [true, true, true]];
    ///
    /// // Any element true? -> true
    /// let any_true = Array::compact_ndarray(&nd)?
    ///     .any((0, 1)).to_ndarray()?;
    /// assert_eq!(any_true[[]], true);
    ///
    /// // Any true along axis 0 (per column) -> shape [3]
    /// let col_any = Array::compact_ndarray(&nd)?
    ///     .any(0).to_ndarray()?;
    /// assert_eq!(col_any.as_slice().unwrap(), &[true, true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Any,
    AnyKernel,
    where {
        S: ArrayStorageTyped<Item = bool>,
    }
    output = bool,
);
impl ReductionOpKernel<bool> for AnyKernel {
    type Output = bool;
    type State = bool;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(bool, u64)>) -> Self::State {
        let mut state = false;
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: bool, _idx: u64) -> Self::State {
        state || item
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a || b
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        true
    }
}

/// Reduces one or more axes by combining the elements along those axes with a user-supplied
/// binary closure, in an **unspecified order**.
///
/// The output dtype is the same as the input dtype (`S::Item`), and the closure has signature
/// `Fn(S::Item, S::Item) -> S::Item`. There is no separate seed value: the first element each
/// accumulator sees becomes its initial value. `f` is used for two distinct jobs, and cannot
/// tell them apart:
///
/// - folding one element into a running accumulator - `f(acc, x)`;
/// - combining two accumulators built from *disjoint* subsets of the same output cell's
///   elements - `f(acc_a, acc_b)`.
///
/// # The closure MUST be associative and commutative
///
/// Despite the name, this is **not** [`Iterator::reduce`]: it is not a left fold, and that
/// holds even when a single axis is reduced. One output cell's elements are spread over multiple
/// interleaved lane accumulators that are then collapsed by a pairwise tree, and when the
/// reduced extent is larger than one read tile a cell's accumulator is merged across tiles.
/// Which elements land in which accumulator, and in which order the accumulators are combined,
/// is an implementation detail that shifts with the array shape, the block shape, the read
/// window size, and the library version.
///
/// So `f` must be associative *and* commutative for the result to be well-defined - `min`,
/// `max`, `+`, `*`, `&`, `|`, `^`, `gcd` all qualify. A non-commutative closure such as
/// `|a, b| a - b` compiles and runs but yields an arbitrary value.
///
/// Float `+` / `*` are only associative up to rounding. That is fine and expected: the
/// built-in [`Sum`] / [`Product`] reductions reassociate exactly the same way.
///
/// # Empty reductions
///
/// Empty reductions (any reduced dimension has length `0`) are **not supported**: with no seed
/// value and no first element there is nothing to return. [`Reduce::new`] returns an
/// `Err`, and [`Array::reduce_unordered`] panics at construction time.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation; the operation is also available as
/// [`Array::reduce_unordered()`](crate::Array::reduce_unordered).
///
/// # Examples
///
/// Custom maximum over a single axis:
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1i32, 5, 3], [4, 2, 6]];
/// let row_max = Array::compact_ndarray(&nd)?
///     .reduce_unordered(1, |a, b| if a > b { a } else { b })
///     .to_ndarray()?;
/// assert_eq!(row_max.as_slice().unwrap(), &[5, 6]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Multi-axis reduction with a closure that has no built-in equivalent:
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[0b0011u8, 0b0101], [0b1001, 0b0001]];
/// let xor = Array::compact_ndarray(&nd)?
///     .reduce_unordered((0, 1), |a, b| a ^ b)
///     .to_ndarray()?;
/// assert_eq!(xor[[]], 0b1110);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Reduce<S: ArrayStorage, D, F>(ReductionOp<S, ReduceKernel<F>, D>);
impl<S: ArrayStorage, D, F> Reduce<S, D, F> {
    /// Constructs a [`Reduce`] storage. See the struct docs for semantics and examples.
    pub fn new<Ax>(array: S, axes: Ax, f: F) -> Result<Self>
    where
        S: ArrayStorageTyped,
        D: Dimension,
        F: Fn(S::Item, S::Item) -> S::Item,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        Ok(Self(ReductionOp::new(array, ReduceKernel(f), axes)?))
    }

    /// Constructs an array with [`Reduce`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<Ax>(array: Array<S>, axes: Ax, f: F) -> Result<Array<Reduce<S, D, F>>>
    where
        S: ArrayStorageTyped,
        D: Dimension,
        F: Fn(S::Item, S::Item) -> S::Item,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        Self::new(array.into_storage(), axes, f).map(Array::from_storage)
    }
}
struct ReduceKernel<F>(F);
impl<T, F> ReductionOpKernel<T> for ReduceKernel<F>
where
    F: Fn(T, T) -> T,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, u64)>) -> Self::State {
        init_item.unwrap().0
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        (self.0)(state, item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        (self.0)(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    #[inline(always)]
    fn supports_empty(&self) -> bool {
        false
    }
}
impl<S, D, F> ArrayStorage for Reduce<S, D, F>
where
    S: ArrayStorageTyped,
    D: Dimension,
    F: Fn(S::Item, S::Item) -> S::Item,
{
    type ElementType = Ty<S::Item>;
    type Dimension = D;
    crate::storage::impl_array_storage_forward!(<S, D, F>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Reduce", [&self.0.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Reduce<S, NewD, F>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Reduce(self.0.dimension_change::<NewD>()?))
    }

    crate::ops::impl_element_type_change_default!();
}

/*
/// Reduces one or more axes by folding the elements along those axes through a
/// user-supplied closure, starting from an explicit initial accumulator.
///
/// The output dtype is the accumulator type `B` (which can differ from the input element
/// type `S::Item`). The closure has signature `Fn(B, S::Item) -> B` and is applied with
/// the running accumulator on the left, mirroring [`Iterator::fold`]: for a stream
/// `x0, x1, ..., xn`, the result is `f(f(f(init, x0), x1), ..., xn)`.
///
/// `B` must implement [`Dtyped`](crate::dtype::Dtyped) (i.e. `Copy + Send + Sync + 'static`
/// plus the jix dtype contract). The initial value is stored inside the storage and cloned
/// (via `Copy`) once per output cell.
///
/// # Traversal order
///
/// - **Single reduced axis**: elements along that axis are visited in logical order (index
///   `0` upward). The result is well-defined for non-commutative / non-associative
///   closures.
/// - **Multiple reduced axes**: elements are visited in an *implementation-defined* order
///   driven by the storage's internal tiling. The order is not stable across array shapes,
///   block sizes, or library versions. Closures used here MUST be both associative and
///   commutative for the result to be well-defined.
///
/// # Empty reductions
///
/// Unlike [`Reduce`], `Fold` supports empty reductions: when a reduced axis has length
/// `0`, every output cell receives `init` unchanged (the closure is never invoked).
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation; the operation is also available as
/// [`Array::fold()`](crate::Array::fold).
///
/// # Examples
///
/// Sum with a wider accumulator (the input is `u8`, the result is `i64`):
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1u8, 2, 3], [4, 5, 6]];
/// let total = Array::compact_ndarray(&nd)?
///     .fold((0, 1), 0i64, |a, x| a + x as i64)
///     .to_ndarray()?;
/// assert_eq!(total[[]], 21);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Single-axis fold: order is guaranteed to be logical, so the non-commutative closure
/// produces a deterministic result:
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // ((((100 - 10) - 1) - 2) - 3) = 84
/// let nd = array![10i32, 1, 2, 3];
/// let result = Array::compact_ndarray(&nd)?
///     .fold(0, 100i32, |a, x| a - x)
///     .to_ndarray()?;
/// assert_eq!(result[[]], 84);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Counting elements satisfying a predicate (per row):
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1i32, 5, 3, 7], [4, 2, 8, 1]];
/// // Count elements > 3 along each row.
/// let counts = Array::compact_ndarray(&nd)?
///     .fold(1, 0u64, |c, x| c + (x > 3) as u64)
///     .to_ndarray()?;
/// assert_eq!(counts.as_slice().unwrap(), &[2, 2]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Fold<S: ArrayStorage, D, B, F>(ReductionOp<S, VanillaFoldKernel<B, F>, D>);
impl<S: ArrayStorage, D, B, F> Fold<S, D, B, F> {
    /// Constructs a [`Fold`] storage. See the struct docs for semantics and examples.
    pub fn new<Ax>(array: S, axes: Ax, init: B, f: F) -> Result<Self>
    where
        S: ArrayStorageTyped,
        D: Dimension,
        B: Dtyped,
        F: Fn(B, S::Item) -> B,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        Ok(Self(ReductionOp::new(
            array,
            VanillaFoldKernel { init, f },
            axes,
        )?))
    }

    /// Constructs an array with [`Fold`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<Ax>(
        array: Array<S>,
        axes: Ax,
        init: B,
        f: F,
    ) -> Result<Array<Fold<S, D, B, F>>>
    where
        S: ArrayStorageTyped,
        D: Dimension,
        B: Dtyped,
        F: Fn(B, S::Item) -> B,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        Self::new(array.into_storage(), axes, init, f).map(Array::from_storage)
    }
}
struct VanillaFoldKernel<B, F> {
    init: B,
    f: F,
}
impl<T, B, F> ReductionOpKernel<T> for VanillaFoldKernel<B, F>
where
    B: Dtyped,
    F: Fn(B, T) -> B,
{
    type Output = B;
    type State = B;

    #[inline(always)]
    fn init_state(&self, first: Option<T>) -> Self::State {
        let mut state = self.init;
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, 0);
        }
        state
    }

    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        (self.f)(state, item)
    }

    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }

    #[inline(always)]
    fn supports_empty(&self) -> bool {
        true
    }
}
impl<S, D, B, F> ArrayStorage for Fold<S, D, B, F>
where
    S: ArrayStorageTyped,
    D: Dimension,
    B: Dtyped,
    F: Fn(B, S::Item) -> B,
{
    type ElementType = Ty<B>;
    type Dimension = D;
    crate::storage::impl_array_storage_forward!(<S, D, B, F>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Fold", [&self.0.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Fold<S, NewD, B, F>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Fold(self.0.dimension_change::<NewD>()?))
    }

    crate::ops::impl_element_type_change_default!();
}
*/

/// Emits an `Array::$method(...)` helper that forwards to `$Op::new_array(...)`. The full
/// where-clause on `S` (and its `Item`) is supplied verbatim by the caller so each op can
/// pick its own bound (`PartialOrd`, `Maximum`, `Sum`, `Item = bool`, ...).
macro_rules! define_array_reduction_method {
    // single-axis variant
    (
        $method:ident: $Op:ident,
        where { $($where_:tt)+ }
        $(, extra_args = ($($extra_arg:ident: $extra_ty:ty),*))?,
        single_axis $(,)?
    ) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self, axis: usize $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S>>
        where $($where_)+
        {
            $Op::new_array(self, axis $($(, $extra_arg)*)?).unwrap()
        }
    };

    // multi-axis variant
    (
        $method:ident: $Op:ident,
        where { $($where_:tt)+ }
        $(, extra_args = ($($extra_arg:ident: $extra_ty:ty),*))?
        $(,)?
    ) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<Ax>(self, axis: Ax $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S, Ax::ReducedDimension<S::Dimension>>>
        where
            $($where_)+
            Ax: AxesArg,
        {
            $Op::new_array(self, axis $($(, $extra_arg)*)?).unwrap()
        }
    };
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(
        max: Max,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Maximum<Output = S::Item> + Dtyped,
        }
    );
    define_array_reduction_method!(
        min: Min,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Minimum<Output = S::Item> + Dtyped,
        }
    );
    define_array_reduction_method!(
        argmax: ArgMax,
        where {
            S: ArrayStorageTyped,
            S::Item: PartialOrd,
        },
        single_axis
    );
    define_array_reduction_method!(
        argmin: ArgMin,
        where {
            S: ArrayStorageTyped,
            S::Item: PartialOrd,
        },
        single_axis
    );
    define_array_reduction_method!(
        sum: Sum,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Sum<Output: Dtyped>,
        }
    );
    define_array_reduction_method!(
        product: Product,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Product<Output: Dtyped>,
        }
    );
    define_array_reduction_method!(
        mean: Mean,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Mean<Output: Dtyped>,
        }
    );
    define_array_reduction_method!(
        var: Variance,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Variance<Output: Dtyped>,
        },
        extra_args = (ddof: f64)
    );
    define_array_reduction_method!(
        std: StandardDeviation,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Variance<Output: num_traits::Float + Dtyped>,
        },
        extra_args = (ddof: f64)
    );
    define_array_reduction_method!(
        all: All,
        where {
            S: ArrayStorageTyped<Item = bool>,
        }
    );
    define_array_reduction_method!(
        any: Any,
        where {
            S: ArrayStorageTyped<Item = bool>,
        }
    );

    /// Applies the [`Reduce`] operation, see the op struct docs for details.
    ///
    /// `f` MUST be associative and commutative: the elements are visited in an unspecified
    /// order, and `f` combines partial accumulators as well as single elements.
    #[track_caller]
    pub fn reduce_unordered<F, Ax>(
        self,
        axes: Ax,
        f: F,
    ) -> Array<Reduce<S, Ax::ReducedDimension<S::Dimension>, F>>
    where
        S: ArrayStorageTyped,
        F: Fn(S::Item, S::Item) -> S::Item,
        Ax: AxesArg,
    {
        Reduce::new_array(self, axes, f).unwrap()
    }

    /* TODO
    /// Applies the [`Fold`] operation, see the op struct docs for details.
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn fold<F, B, Ax>(
        self,
        axes: Ax,
        init: B,
        f: F,
    ) -> Array<Fold<S, Ax::ReducedDimension<S::Dimension>, B, F>>
    where
        S: ArrayStorageTyped,
        B: Dtyped,
        F: Fn(B, S::Item) -> B,
        Ax: AxesArg,
    {
        Fold::new_array(self, axes, init, f).unwrap()
    }
    */
}

#[cfg(test)]
pub(crate) mod tests {
    use std::rc::Rc;

    use ndarray::{array, ArrayD};

    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use proptest::prelude::*;

    use super::Reduce;
    use crate::array::Array;
    use crate::storage::Compact;
    use crate::{DimDyn, Ty};

    #[test]
    fn reduction_tile_shape_two_scale_and_regular() {
        use super::reduction_tile_shape;
        use crate::dtype::Dtyped;
        use crate::storage::params::ArraySpecFlags;
        use crate::{ArrayParams, Dimension};

        // Build a source spec directly (no data allocated): block [8, 8] and a read window in bytes
        // that, for i32 (itemsize 4), gives known item counts. REDUCED_TILE_MIN_NITEMS is 512.
        let spec_of = |shape: &[u64], read_min: u64, read_max: u64| {
            let mut params = ArrayParams::new();
            params.block_shape(&[8, 8]);
            params.read_size((read_min, read_max));
            params
                .into_spec(shape, &i32::DTYPE, ArraySpecFlags::default())
                .unwrap()
        };

        // Reduce axis 0 (the low-priority outer dim). Window (2048, 16384) bytes -> (512, 4096)
        // items. The regular scale spends the min budget on the inner dim, starving the reduced dim
        // (8 < 512 floor), so the tile is rebuilt: the reduced dim grows to the 512 floor and the
        // non-reduced dim fills the rest (4096 / 512 = 8).
        let a = spec_of(&[1024, 64], 2048, 16384);
        let tile = reduction_tile_shape::<DimDyn>(
            &a.as_ref(),
            &[true, false],
            &[1024, 64],
            &[1024, 64],
            4,
        );
        if crate::util::USE_NEW_READ_SCALING {
            assert_eq!(
                tile.as_slice(),
                &[512, 8],
                "two-scale: reduced dim reaches the floor"
            );
        } else {
            // Balanced/reduced-first: the reduced dim (0) claims the budget before the free dim.
            assert_eq!(
                tile.as_slice(),
                &[64, 8],
                "reduced-first: reduced dim scaled before the free dim"
            );
        }

        // Reduce axis 1 (the high-priority inner dim) with a large min budget (16384, 16384) bytes ->
        // (4096, 4096) items, so the regular scale already covers the reduced dim past the floor and
        // `reduction_tile_shape` returns it unchanged.
        let b = spec_of(&[64, 1024], 16384, 16384);
        let sp = b.as_ref();
        let tile = reduction_tile_shape::<DimDyn>(&sp, &[false, true], &[64, 1024], &[64, 1024], 4);
        let regular = sp.read_shape_heuristic::<DimDyn>(&[64, 1024], &[64, 1024], 4);
        assert_eq!(
            tile.as_slice(),
            regular.as_slice(),
            "regular tile already covers the reduced dim"
        );
        assert!(tile[1] >= 512, "reduced dim meets the floor, got {tile:?}");
    }

    /// Per-dtype comparison policy for the reduction property tests.
    ///
    /// A reduction reads its input block-by-block, so a floating accumulator reassociates and can
    /// land a few ULP away from a sequential reference fold. Float and complex reductions are
    /// therefore compared with [`ApproxEq`], using a per-dtype tolerance (looser for the narrower
    /// `f32` / `Complex<f32>`, tighter for `f64` / `Complex<f64>`). Integer and `bool` reductions
    /// are exact and compared bit-for-bit.
    ///
    /// `f16` only ever surfaces here as a `max` / `min` result (its `sum`/`product`/`mean` value
    /// parity is skipped - native `f16` accumulation drifts/overflows too far to pin with a
    /// tolerance, mirroring the Python suite), and `max`/`min` select an input element, so `f16`
    /// is compared exactly like the integers.
    ///
    /// [`ApproxEq`]: crate::scalar::ApproxEq
    trait ReductionCompare: crate::dtype::Dtyped + std::fmt::Debug + Clone {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>);
    }

    /// Implements [`ReductionCompare`] with exact, bit-for-bit comparison.
    macro_rules! reduction_compare_exact {
        ($($t:ty),* $(,)?) => {$(
            impl ReductionCompare for $t {
                fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
                    crate::util::assert_array_matches(actual, expected);
                }
            }
        )*};
    }
    reduction_compare_exact!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
    #[cfg(feature = "half")]
    reduction_compare_exact!(f16);

    impl ReductionCompare for f32 {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
            crate::util::assert_array_matches_approx(actual, expected, 1e-3, 1e-1);
        }
    }
    impl ReductionCompare for f64 {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
            crate::util::assert_array_matches_approx(actual, expected, 1e-9, 1e-6);
        }
    }
    #[cfg(feature = "num-complex")]
    impl ReductionCompare for complex_f32 {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
            crate::util::assert_array_matches_approx(
                actual,
                expected,
                1e-3,
                Complex::new(1e-1, 1e-1),
            );
        }
    }
    #[cfg(feature = "num-complex")]
    impl ReductionCompare for complex_f64 {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
            crate::util::assert_array_matches_approx(
                actual,
                expected,
                1e-9,
                Complex::new(1e-6, 1e-6),
            );
        }
    }

    /// Asserts a reduction result matches its reference, dispatching exact vs. approximate
    /// comparison (and the per-dtype tolerance) on the output dtype via [`ReductionCompare`].
    fn assert_reduction_matches<S: crate::ArrayStorage, T: ReductionCompare>(
        actual: &Array<S>,
        expected: &ArrayD<T>,
    ) {
        T::assert_matches(actual, expected);
    }

    pub(crate) fn axis_strategy(ndim: usize) -> impl proptest::strategy::Strategy<Value = usize> {
        0..ndim
    }

    pub(crate) fn axes_strategy(ndim: usize) -> proptest::strategy::BoxedStrategy<Vec<usize>> {
        if ndim == 0 {
            return proptest::strategy::Just(vec![]).boxed();
        }
        let axis_strategy = axis_strategy(ndim).prop_map(|axis| vec![axis]);
        let multi_axes_strategy = prop::collection::vec(0..ndim, 1..=ndim).prop_map(|mut axes| {
            axes.sort_unstable();
            axes.dedup();
            axes
        });
        prop::strategy::Union::new_weighted(vec![
            (3, axis_strategy.boxed()),
            (1, multi_axes_strategy.boxed()),
        ])
        .boxed()
    }

    fn reduction_shape_strategy() -> impl proptest::strategy::Strategy<Value = Vec<usize>> {
        prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=100, 1)),
            (2, proptest::collection::vec(100..=1000, 1)),
            // 2D
            (8, proptest::collection::vec(1..=16, 2)),
            (2, proptest::collection::vec(16..=37, 2)),
            // 3D
            (5, proptest::collection::vec(1..=12, 3)),
            // 4D
            (5, proptest::collection::vec(1..=8, 4)),
            // Many dims
            (3, proptest::collection::vec(1..=4, 1..=8)),
        ])
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn carray_strategy_for_reduction<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<
        Value = (ArrayD<T>, Rc<Array<Compact<Ty<T>, DimDyn>>>, Vec<usize>),
    > {
        let shape = reduction_shape_strategy();
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axes = axes_strategy(nd.ndim());
                (Just(nd), Just(za), axes)
            })
    }

    // pub(crate) fn carray_strategy_for_reduction_single_axis<T: crate::util::ScalarStrategy>(
    //     elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    // ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<Array<Compact<Ty<T>, DimDyn>>>, usize)>
    // {
    //     let shape = reduction_shape_strategy();
    //     let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
    //     array
    //         .prop_map(|(nd, za)| (nd, Rc::new(za)))
    //         .prop_flat_map(|(nd, za)| {
    //             let axis = axis_strategy(nd.ndim());
    //             (Just(nd), Just(za), axis)
    //         })
    // }

    use proptest::test_runner::{Config, TestRunner};

    /// Shared proptest driver for reduction-op tests.
    ///
    /// Generic over the dtype only, with the op passed as a fn pointer, to avoid per-op
    /// monomorphization.
    #[inline(never)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn check_reduction<T>(
        cases: proptest::strategy::BoxedStrategy<(
            ArrayD<T>,
            Rc<Array<Compact<Ty<T>, DimDyn>>>,
            Vec<usize>,
        )>,
        check: fn(&ArrayD<T>, &Array<Compact<Ty<T>, DimDyn>>, &[usize]),
    ) where
        T: crate::util::ScalarStrategy,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(&cases, |(nd, za, axes)| {
                check(&nd, &za, &axes);
                Ok(())
            })
            .unwrap();
    }

    macro_rules! test_reduction_dtype {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                // `$body` is shared across all dtypes this macro is invoked with, including
                // `bool` (where ordering comparisons like `x > m` are boolean, not numeric)
                // and same-width integer arms (where a widening cast like `x as u64` is a
                // no-op for the widest dtype in the list).
                #[test]
                #[allow(clippy::bool_comparison, clippy::unnecessary_cast)]
                fn [<$op_method _ $dtype>]() {
                    crate::ops::reduction::tests::check_reduction::<$dtype>(
                        proptest::strategy::Strategy::boxed(
                            crate::ops::reduction::tests::carray_strategy_for_reduction::<$dtype>(
                                <$dtype as crate::util::ScalarStrategy>::$strategy()
                            )
                        ),
                        |nd, za, axes| {
                            let result = za.as_ref().$op_method(axes);
                            let expected = crate::ops::reduction::tests::ndarray_reduce(
                                nd, axes,
                                |arr| {
                                    let $items = arr.iter().cloned();
                                    $body
                                }
                            );
                            crate::ops::reduction::tests::assert_reduction_matches(&result, &expected);
                        },
                    );
                }
            }
        };

        (
            $op_method:ident,
            single_axis = true,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axis) in crate::ops::reduction::tests::carray_strategy_for_reduction_single_axis::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(axis);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &[axis],
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::ops::reduction::tests::assert_reduction_matches(&result, &expected);
                    }
                }
            }
        };

    }

    macro_rules! test_reduction {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };

        (
            $op_method:ident,
            single_axis = true,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                single_axis = true,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    single_axis = true,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };

        // Approximate variants for order-dependent float/complex `sum`/`product`/`mean`. Those
        // ops preserve the input width and accumulate in it, so the block fold reassociates. The
        // reference `$body` folds in a wide accumulator; here it is cast to the op's output dtype
        // (`$dtype`, which for these ops equals the input dtype) so the reference and the
        // natively-accumulated jix result share a dtype. The cast is the only difference from the
        // plain arms - comparison then goes through the usual (approximate, for floats)
        // `ReductionCompare` dispatch.
        (
            $op_method:ident,
            approx,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { <_ as crate::scalar::Cast<$dtype>>::cast($body) },
                $dtype,
                $strategy
            );)+
        };
    }

    #[allow(unused_imports)]
    pub(crate) use {test_reduction, test_reduction_dtype};

    test_reduction!(
        max,
        |items| { items.reduce(|m, x| if x > m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[test]
    fn min_concrete() {
        // i32: negatives and the signed dtype MIN/MAX edges; axis 0, axis 1, and all axes.
        // A non-default block shape ([2, 2]) also crosses a block boundary on both the
        // reduced and the kept axis.
        let nd = array![[i32::MIN, 3, -5], [7, i32::MAX, -9]];
        let za = Array::compact_ndarray_with(&nd, crate::util::arr_params(&[2, 2])).unwrap();
        // axis 0 (per column): min(MIN,7)=MIN; min(3,MAX)=3; min(-5,-9)=-9.
        crate::util::assert_array_matches(&za.as_ref().min(0usize), &array![i32::MIN, 3, -9]);
        // axis 1 (per row): min(MIN,3,-5)=MIN; min(7,MAX,-9)=-9.
        crate::util::assert_array_matches(&za.as_ref().min(1usize), &array![i32::MIN, -9]);
        // all axes: overall minimum is i32::MIN.
        crate::util::assert_array_matches(&za.as_ref().min((0, 1)), &ndarray::arr0(i32::MIN));

        // u8: the unsigned dtype MIN (0) / MAX (255) edges - distinct from the signed case
        // above since unsigned types have no negatives.
        let ndu = array![[0u8, 200, 255], [10, 255, 5]];
        let zau = Array::compact_ndarray(&ndu).unwrap();
        crate::util::assert_array_matches(&zau.as_ref().min(0usize), &array![0u8, 200, 5]);
        crate::util::assert_array_matches(&zau.as_ref().min(1usize), &array![0u8, 5]);
        crate::util::assert_array_matches(&zau.as_ref().min((0, 1)), &ndarray::arr0(0u8));

        // f32: negatives, zero, +/-infinity. `NaN` is deliberately not used here: for float
        // outputs `ReductionCompare` compares via `assert_array_matches_approx`, whose
        // `ApproxEq` - like IEEE 754 - never treats `NaN` as approximately equal to `NaN`, so
        // a `NaN` result can't be asserted through that helper.
        let ndf = array![
            [1.5f32, -2.0, 0.0],
            [f32::INFINITY, f32::NEG_INFINITY, -3.5]
        ];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        crate::util::assert_array_matches_approx(
            &zaf.as_ref().min(0usize),
            &array![1.5f32, f32::NEG_INFINITY, -3.5],
            1e-3,
            1e-1,
        );
        crate::util::assert_array_matches_approx(
            &zaf.as_ref().min(1usize),
            &array![-2.0f32, f32::NEG_INFINITY],
            1e-3,
            1e-1,
        );
        crate::util::assert_array_matches_approx(
            &zaf.as_ref().min((0, 1)),
            &ndarray::arr0(f32::NEG_INFINITY),
            1e-3,
            1e-1,
        );
    }

    #[test]
    fn argmax_concrete() {
        // i32 with negatives and a deliberate 2-way tie on both axes: the kernel must return
        // the FIRST index among tied maxima, not an arbitrary one.
        let nd = array![[5i32, 5, 2], [5, -1, -8]];
        let za = Array::compact_ndarray(&nd).unwrap();
        // axis 0 (per column): col 0 ties row 0/row 1 at 5 -> first row (0) wins.
        crate::util::assert_array_matches(&za.as_ref().argmax(0usize), &array![0u64, 0, 0]);
        // axis 1 (per row): row 0 ties col 0/col 1 at 5 -> first col (0) wins.
        crate::util::assert_array_matches(&za.as_ref().argmax(1usize), &array![0u64, 0]);

        // f32 path, same tie-break rule, negatives included. (`NaN` excluded - see
        // `min_concrete` for why.)
        let ndf = array![[2.5f32, 2.5, -1.0], [3.0, 3.0, 0.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        crate::util::assert_array_matches(&zaf.as_ref().argmax(0usize), &array![1u64, 1, 1]);
        // row 0 ties col 0/col 1 at 2.5, row 1 ties col 0/col 1 at 3.0 -> first col (0) wins.
        crate::util::assert_array_matches(&zaf.as_ref().argmax(1usize), &array![0u64, 0]);
    }

    #[test]
    fn argmin_concrete() {
        // Same arrays as `argmax_concrete`, exercising the minimum side of the tie-break rule.
        let nd = array![[5i32, 5, 2], [5, -1, -8]];
        let za = Array::compact_ndarray(&nd).unwrap();
        // axis 0 (per column): col 0 ties row 0/row 1 at 5 -> first row (0) wins.
        crate::util::assert_array_matches(&za.as_ref().argmin(0usize), &array![0u64, 1, 1]);
        // axis 1 (per row): both rows have a unique minimum (2, then -8).
        crate::util::assert_array_matches(&za.as_ref().argmin(1usize), &array![2u64, 2]);

        let ndf = array![[2.5f32, 2.5, -1.0], [3.0, 3.0, 0.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        // col 0/col 1 tie at 2.5 vs. 3.0 -> row 0 (index 0) wins both.
        crate::util::assert_array_matches(&zaf.as_ref().argmin(0usize), &array![0u64, 0, 0]);
        crate::util::assert_array_matches(&zaf.as_ref().argmin(1usize), &array![2u64, 2]);
    }

    test_reduction!(
        sum,
        |items| { items.fold(0u64, |m, x| m + x as u64) },
        [u8, u16, u32, u64, bool],
        op_safe_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0i64, |m, x| m + x as i64) },
        [i8, i16, i32, i64],
        op_safe_strategy
    );
    // Float/complex sum preserves the input width and accumulates in it, so the block fold
    // reassociates - compare approximately, and skip f16 (its native accumulation drifts too
    // far to pin with a tolerance; covered by max/min instead).
    test_reduction!(
        sum,
        approx,
        |items| { items.fold(0.0f64, |m, x| m + <_ as crate::scalar::Cast<f64>>::cast(x)) },
        [f32, f64],
        op_safe_strategy
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        sum,
        approx,
        |items| {
            items.fold(Complex::<f64>::default(), |m, x| {
                m + <_ as crate::scalar::Cast<Complex<f64>>>::cast(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    #[test]
    fn product_concrete() {
        // i32 -> i64 (integer product widens; see `_traits::Product`). Includes negatives and
        // a `0` (zero-absorption edge); values stay small so the widened i64 accumulator can't
        // overflow.
        let nd = array![[2i32, -3, 0], [-1, 5, -2]];
        let za = Array::compact_ndarray(&nd).unwrap();
        // axis 0 (per column): 2*-1=-2; -3*5=-15; 0*-2=0.
        crate::util::assert_array_matches(&za.as_ref().product(0usize), &array![-2i64, -15, 0]);
        // axis 1 (per row): 2*-3*0=0; -1*5*-2=10.
        crate::util::assert_array_matches(&za.as_ref().product(1usize), &array![0i64, 10]);
        // all axes: the `0` term collapses the whole product to 0.
        crate::util::assert_array_matches(&za.as_ref().product((0, 1)), &ndarray::arr0(0i64));

        // f32 -> f32 (float product keeps its input width). Negatives, no zero this time so
        // the full multiplication chain is exercised (the i32 case above covers the zero edge).
        let ndf = array![[1.5f32, -2.0, 0.5], [2.0, -1.0, 3.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        // axis 0 (per column): 1.5*2.0=3.0; -2.0*-1.0=2.0; 0.5*3.0=1.5.
        crate::util::assert_array_matches_approx(
            &zaf.as_ref().product(0usize),
            &array![3.0f32, 2.0, 1.5],
            1e-3,
            1e-1,
        );
        // axis 1 (per row): 1.5*-2.0*0.5=-1.5; 2.0*-1.0*3.0=-6.0.
        crate::util::assert_array_matches_approx(
            &zaf.as_ref().product(1usize),
            &array![-1.5f32, -6.0],
            1e-3,
            1e-1,
        );
        crate::util::assert_array_matches_approx(
            &zaf.as_ref().product((0, 1)),
            &ndarray::arr0(9.0f32),
            1e-3,
            1e-1,
        );
    }
    // mean. Integer/bool inputs widen to f64 (exact-in-f64 sum, then divide) - compared via the
    // f64 `ReductionCompare` (approx). Float/complex inputs preserve their width and reassociate,
    // so they take the approx path; f16 is skipped (see `sum`).
    test_reduction!(
        mean,
        |items| {
            {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<f64>>::cast(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [i8, i16, i32, i64, u8, u16, u32, u64],
        op_safe_strategy
    );
    test_reduction!(
        mean,
        approx,
        |items| {
            {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<f64>>::cast(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [f32, f64],
        op_safe_strategy
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        mean,
        approx,
        |items| {
            {
                let mut sum: Complex<f64> = Complex::default();
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<Complex<f64>>>::cast(x);
                    count += 1;
                }
                sum / Complex::new(count as f64, 0.0)
            }
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    #[test]
    fn variance() {
        let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let var_all = a.as_ref().var((0, 1), 0.0).to_ndarray().unwrap();
        assert!((var_all[[]] - 2.9166).abs() < 0.001);
        let var_col = a.as_ref().var(0, 0.0).to_ndarray().unwrap();
        assert!((var_col[[0]] - 2.25).abs() < 0.001);
        let var_row = a.as_ref().var(1, 0.0).to_ndarray().unwrap();
        assert!((var_row[[0]] - 0.6666).abs() < 0.001);
    }
    #[test]
    fn std() {
        let a = Array::compact_ndarray(&array![[7i32, 8, 9], [4, 5, 6]]).unwrap();
        let std_all = a.as_ref().std((0, 1), 0.0).to_ndarray().unwrap();
        assert!((std_all[[]] - 1.7078).abs() < 0.001);
        let std_col = a.as_ref().std(0, 0.0).to_ndarray().unwrap();
        assert!((std_col[[0]] - 1.5).abs() < 0.001);
        let std_row = a.as_ref().std(1, 0.0).to_ndarray().unwrap();
        assert!((std_row[[0]] - 0.8164).abs() < 0.001);
    }

    /// Drives a `sum` reduction through the two-level bulk/tile chunking, with `block_shape`
    /// and a small `read_size` chosen so the tile lands at a genuine sub-block (smaller than
    /// the array along both a reduced and a non-reduced axis). That exercises both non-trivial
    /// paths at once: more than one *bulk*, and more than one *tile* within a bulk.
    ///
    /// The reduction's `read_data` is driven **directly** over the full output range in one
    /// call. The top-level readers (`to_ndarray`, `to_ndarray_sub`) chunk the output by the
    /// reduction op's block shape before calling `read_data`, so each call's non-reduced extent
    /// equals one block and the read-shape heuristic snaps the tile to it - never subdividing
    /// the non-reduced axis. Handing `read_data` the whole non-reduced extent at once makes it
    /// split that axis into several tiles per bulk. Correctness is exact: the i64 sum of small
    /// signed values can't reassociate or overflow. During development, path coverage is
    /// confirmed with the temporary debug print in `ReductionOp::read_data`.
    fn check_sum_divided(
        shape: &[usize],
        block_shape: &[u32],
        read_size: (u64, u64),
        axes: &[usize],
    ) {
        use crate::ArrayStorage;

        // Deterministic values in a small signed range so a wrong tile offset or a
        // double-counted / dropped item shows up as a mismatch.
        let n: usize = shape.iter().product();
        let nd = ndarray::ArrayD::from_shape_vec(
            shape.to_vec(),
            (0..n as i32).map(|x| (x % 97) - 48).collect(),
        )
        .unwrap();

        let mut params = crate::ArrayParams::new();
        params.block_shape(block_shape);
        params.read_size(read_size);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();

        let reduced = za.as_ref().sum(axes);
        let out_shape: Vec<u64> = reduced.shape().to_vec();
        let full_index: Vec<std::ops::Range<u64>> = out_shape.iter().map(|&s| 0..s).collect();
        let n_out: usize = out_shape.iter().product::<u64>() as usize;
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        let mut buf = vec![0i64; n_out.max(1)];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    buf.as_mut_ptr().cast::<u8>(),
                    n_out * size_of::<i64>(),
                )
            };
            let sh = full_index
                .iter()
                .map(|r| (r.end - r.start) as usize)
                .collect::<Vec<_>>();
            let c = crate::util::default_strides_slice(&sh, size_of::<i64>());
            let mut out = unsafe { crate::storage::StridedBuf::from_slice_mut(bytes, c.as_ref()) };
            storage
                .read_data(&full_index, &ctx, Some(&mut out))
                .unwrap();
        }

        let expected = ndarray_reduce(&nd, axes, |v| v.iter().map(|&x| x as i64).sum::<i64>());
        assert_eq!(&buf[..n_out], expected.as_slice().unwrap());
    }

    /// The `out=` destination carries no alignment guarantee, and its strides need not be multiples
    /// of the output dtype's alignment either. Both cases must fall back to a pooled state buffer
    /// and unaligned output stores instead of reusing the output bytes as the state buffer. Run
    /// under Miri to check the accesses.
    ///
    /// `extra_stride` is the byte gap added between consecutive outputs (0 = packed).
    #[allow(clippy::single_range_in_vec_init)]
    fn check_sum_into_misaligned_out(misalign: usize, extra_stride: usize) {
        use crate::ArrayStorage;

        let nd = ndarray::ArrayD::from_shape_vec(vec![3usize, 4], (0..12i32).collect()).unwrap();
        let za = Array::compact_ndarray(&nd).unwrap();
        let reduced = za.as_ref().sum([1usize]);
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        // 3 i64 outputs, at `stride` bytes apart, starting `misalign` bytes past an 8-aligned
        // address inside an over-aligned byte buffer.
        let stride = size_of::<i64>() + extra_stride;
        let mut backing = vec![0u8; 3 * stride + 8 + misalign];
        let off = backing.as_ptr().align_offset(8) + misalign;
        {
            let bytes = &mut backing[off..off + 2 * stride + size_of::<i64>()];
            let mut out = unsafe { crate::storage::StridedBuf::from_slice_mut(bytes, &[stride]) };
            storage.read_data(&[0..3], &ctx, Some(&mut out)).unwrap();
        }

        let got = (0..3)
            .map(|i| unsafe {
                backing
                    .as_ptr()
                    .add(off + i * stride)
                    .cast::<i64>()
                    .read_unaligned()
            })
            .collect::<Vec<_>>();
        assert_eq!(got, [6, 22, 38]);
    }

    #[test]
    fn sum_into_misaligned_out_buf() {
        check_sum_into_misaligned_out(1, 0);
    }

    #[test]
    fn sum_into_out_buf_with_unaligned_strides() {
        // Base is aligned but the stride is not a multiple of 8, so most output slots are
        // misaligned even though the first one is not.
        check_sum_into_misaligned_out(0, 3);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn reduction_empty_output_subrange() {
        // Reading an EMPTY output sub-range (a non-reduced dim with zero extent) must succeed
        // and yield an empty result - it must not touch the empty-reduction path's assumptions.
        // Regression: the swapped bulk/tile layout finalizes with `nitems = full_reduction_size`
        // (the product of reduced extents), so an empty output range - which seeds no state and
        // falls into the empty path - must not assume that count is zero.
        let nd =
            ndarray::ArrayD::from_shape_vec(vec![5usize, 5], (0..25i32).map(|x| x as i8).collect())
                .unwrap();
        let za = Array::compact_ndarray(&nd).unwrap();
        let reduced = za.as_ref().max(1usize); // output shape [5]
        let ctx = reduced.read_ctx();
        // An empty sub-range along the (sole) output axis.
        let got = reduced.to_ndarray_sub(&[1..1], &ctx).unwrap();
        assert_eq!(got.shape(), &[0]);
    }

    #[test]
    fn sum_multi_bulk_multi_tile_2d_reduce_outer() {
        // Reduce axis 0 (the strided/outer axis), tile == block == [2, 3]. Current scheme:
        // bulks split the reduced axis 0 (8/2 = 4 bulks); within each bulk, tiles split the
        // non-reduced axis 1 (6/3 = 2 tiles/bulk).
        check_sum_divided(&[8, 6], &[2, 3], (32, 64), &[0]);
    }

    #[test]
    fn sum_multi_bulk_multi_tile_2d_reduce_inner() {
        // Reduce axis 1 (the contiguous/inner axis), tile == block == [3, 2]. bulks split
        // reduced axis 1 (8/2 = 4); tiles split non-reduced axis 0 (6/3 = 2).
        check_sum_divided(&[6, 8], &[3, 2], (32, 64), &[1]);
    }

    #[test]
    fn sum_multi_bulk_multi_tile_3d_reduce_middle() {
        // 3D, reduce the middle axis, tile == block == [2, 2, 2]. bulks split reduced axis 1
        // (4/2 = 2); tiles split the two non-reduced axes 0 and 2 ((4/2)*(4/2) = 4 tiles/bulk).
        check_sum_divided(&[4, 4, 4], &[2, 2, 2], (32, 64), &[1]);
    }

    #[test]
    fn sum_into_strided_2d_output_multi_bulk() {
        // A reduction must write straight into a *strided* (non-contiguous) destination using the
        // caller's own byte-strides - not stage through a contiguous scratch and scatter. This
        // exercises the `state_in_out_buf` path: the i64 sum state has the same size/alignment as
        // the i64 output, so the state buffer aliases the strided output and adopts its layout.
        // Small blocks + read_size force multiple bulks/tiles, so the strided `state_offset`
        // (base slot of each bulk's output block) is exercised with a non-trivial 2D stride.
        use crate::storage::StridedBuf;
        use crate::ArrayStorage;

        // [4, 4, 4] i8, reduce the middle axis -> [4, 4] i64.
        let nd = ndarray::ArrayD::from_shape_vec(
            vec![4usize, 4, 4],
            (0..64i32).map(|x| (x % 97) as i8).collect(),
        )
        .unwrap();
        let mut params = crate::ArrayParams::new();
        params.block_shape(&[2, 2, 2]);
        params.read_size((32, 64));
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let reduced = za.as_ref().sum(1usize); // [4, 4] i64
        let index = [0..4u64, 0..4];
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        // Reference: a plain contiguous read.
        let mut expected = vec![0i64; 16];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(expected.as_mut_ptr().cast::<u8>(), 16 * 8)
            };
            let sh = index
                .iter()
                .map(|r| (r.end - r.start) as usize)
                .collect::<Vec<_>>();
            let c = crate::util::default_strides_slice(&sh, size_of::<i64>());
            let mut out = unsafe { crate::storage::StridedBuf::from_slice_mut(bytes, c.as_ref()) };
            storage.read_data(&index, &ctx, Some(&mut out)).unwrap();
        }

        // Strided destination: element strides [8, 2] (bytes [64, 16]), so slot (r, c) lands at
        // element offset r*8 + c*2 and every other element is an untouched gap. Backed by a
        // `Vec<i64>` so the base and every itemsize-multiple slot are 8-aligned.
        const SENTINEL: i64 = i64::MIN;
        let mut backing = vec![SENTINEL; 32];
        let byte_strides = [64usize, 16];
        let slot = |r: usize, c: usize| r * 8 + c * 2;
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), 32 * 8)
            };
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, &byte_strides) };
            storage.read_data(&index, &ctx, Some(&mut out)).unwrap();
        }

        let mut is_slot = [false; 32];
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(backing[slot(r, c)], expected[r * 4 + c], "slot ({r},{c})");
                is_slot[slot(r, c)] = true;
            }
        }
        for (i, &v) in backing.iter().enumerate() {
            if !is_slot[i] {
                assert_eq!(v, SENTINEL, "gap at element {i} was overwritten");
            }
        }
    }

    #[test]
    fn var_into_strided_output_non_aliasing_state() {
        // The variance (Welford) state is larger than its f64 output, so `state_in_out_buf` is
        // false: the state lives in a separate contiguous scratch and `finalize_states` scatters
        // each result into the strided destination. Check that scatter honors the caller's strides
        // and matches a contiguous read exactly (identical compute order -> bit-identical).
        use crate::storage::StridedBuf;
        use crate::ArrayStorage;

        let nd = ndarray::ArrayD::from_shape_vec(vec![3usize, 4], (0..12i32).collect()).unwrap();
        let za = Array::compact_ndarray(&nd).unwrap();
        let reduced = za.as_ref().var(1usize, 0.0); // [3] f64
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        let mut expected = vec![0f64; 3];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(expected.as_mut_ptr().cast::<u8>(), 3 * 8)
            };
            let c = crate::util::default_strides_slice(&[3usize], size_of::<f64>());
            let mut out = unsafe { crate::storage::StridedBuf::from_slice_mut(bytes, c.as_ref()) };
            storage.read_data(&[0..3], &ctx, Some(&mut out)).unwrap();
        }

        // Strided destination: byte stride 16 (element stride 2), so slot i is at element 2*i and
        // 2*i+1 is an untouched gap. Backed by `Vec<f64>` for 8-alignment.
        const SENTINEL: f64 = -98765.0;
        let mut backing = vec![SENTINEL; 6];
        let byte_strides = [16usize];
        {
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), 6 * 8) };
            let mut out = unsafe { StridedBuf::from_slice_mut(bytes, &byte_strides) };
            storage.read_data(&[0..3], &ctx, Some(&mut out)).unwrap();
        }

        for i in 0..3 {
            assert_eq!(backing[i * 2], expected[i], "slot {i}");
            assert_eq!(backing[i * 2 + 1], SENTINEL, "gap {i} was overwritten");
        }
    }

    // --- Reduce ------------------------------------------------------------------

    // Combiners for the `reduce_unordered` property tests. Each one is associative AND
    // commutative, as the op requires - which is what lets a plain sequential fold serve as the
    // reference for the op's unspecified order.
    fn xor<T: std::ops::BitXor<Output = T>>(a: T, b: T) -> T {
        a ^ b
    }
    fn maximum<T: PartialOrd>(a: T, b: T) -> T {
        if a > b {
            a
        } else {
            b
        }
    }
    #[cfg(feature = "num-complex")]
    fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
        a + b
    }

    /// Proptest driver for `reduce_unordered` over one dtype: the op must agree with a plain
    /// sequential fold of the same combiner `$f`, for every shape/axes combination the shared
    /// reduction strategy generates.
    macro_rules! test_reduce_unordered_dtype {
        ($f:path, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                #[test]
                fn [<reduce_unordered_ $dtype>]() {
                    check_reduction::<$dtype>(
                        proptest::strategy::Strategy::boxed(
                            carray_strategy_for_reduction::<$dtype>(
                                <$dtype as crate::util::ScalarStrategy>::$strategy()
                            )
                        ),
                        |nd, za, axes| {
                            let result = za.as_ref().reduce_unordered(axes, $f);
                            let expected = ndarray_reduce(nd, axes, |arr| {
                                arr.iter().cloned().reduce($f).unwrap()
                            });
                            assert_reduction_matches(&result, &expected);
                        },
                    );
                }
            }
        };
        ($f:path, [$($dtype:ident),+ $(,)?], $strategy:ident) => {
            $(test_reduce_unordered_dtype!($f, $dtype, $strategy);)+
        };
    }

    // `xor` is exact and has no built-in op counterpart, so it pins the whole custom-closure
    // path (lane seeding, lane tree merge, cross-tile merge) without needing a tolerance.
    test_reduce_unordered_dtype!(xor, [i8, i16, i32, i64], any_strategy);
    test_reduce_unordered_dtype!(xor, [u8, u16, u32, u64, bool], any_strategy);
    // Floats have no `xor`; `maximum` is the exact order-independent stand-in. `any_strategy`
    // never yields `NaN` for floats, so `>` is a total order here and the reference agrees
    // exactly (this is also the only group `f16` can join - see `ReductionCompare`).
    test_reduce_unordered_dtype!(maximum, [f32, f64], any_strategy);
    #[cfg(feature = "half")]
    test_reduce_unordered_dtype!(maximum, [f16], any_strategy);
    // Complex has no ordering, so use `add` and let the per-dtype approximate comparison absorb
    // the reassociation (same deal as the built-in `sum` on floats).
    #[cfg(feature = "num-complex")]
    test_reduce_unordered_dtype!(add, [complex_f32, complex_f64], op_safe_strategy);

    #[test]
    fn reduce_unordered_concrete() {
        // Axis 0, axis 1 and all-axes, with a non-default block shape ([2, 2]) so both the
        // reduced and the kept axis cross a block boundary.
        let nd = array![[1i32, 5, 3], [4, 2, 6]];
        let za = Array::compact_ndarray_with(&nd, crate::util::arr_params(&[2, 2])).unwrap();
        // axis 0 (per column): max(1,4)=4; max(5,2)=5; max(3,6)=6.
        crate::util::assert_array_matches(
            &za.as_ref().reduce_unordered(0usize, maximum::<i32>),
            &array![4, 5, 6],
        );
        // axis 1 (per row): max(1,5,3)=5; max(4,2,6)=6.
        crate::util::assert_array_matches(
            &za.as_ref().reduce_unordered(1usize, maximum::<i32>),
            &array![5, 6],
        );
        // all axes -> scalar.
        crate::util::assert_array_matches(
            &za.as_ref().reduce_unordered((0, 1), maximum::<i32>),
            &ndarray::arr0(6),
        );
        // A reduction over zero axes is a no-op copy: every cell is its own accumulator.
        crate::util::assert_array_matches(
            &za.as_ref()
                .reduce_unordered([0usize; 0].as_slice(), maximum::<i32>),
            &nd,
        );
    }

    #[test]
    fn reduce_unordered_preserves_dtype() {
        // The output dtype must equal the input dtype (no widening, unlike `sum`/`product`).
        use crate::dtype::Dtyped;
        let a = Array::compact_ndarray(&array![1i8, 2, 3]).unwrap();
        let r = a
            .as_ref()
            .reduce_unordered(0usize, |a, b| a.wrapping_add(b));
        assert_eq!(r.dtype(), &<i8 as Dtyped>::DTYPE);
        assert_eq!(r.to_ndarray().unwrap()[[]], 6i8);
    }

    #[test]
    fn reduce_unordered_errs_on_empty_axis() {
        // With no seed value and no first element there is nothing to return, so construction
        // fails (`supports_empty() == false`).
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let err = Reduce::new_array(a.as_ref(), 1usize, |a: i32, b: i32| a + b)
            .expect_err("empty reduced axis must be rejected");
        assert!(
            err.to_string().contains("empty dimension"),
            "unexpected error: {err}"
        );
        // An empty *kept* axis is fine - the reduced axis is non-empty and the output is empty.
        let empty: Array2<i32> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let r = a
            .as_ref()
            .reduce_unordered(1usize, |a, b| a + b)
            .to_ndarray()
            .unwrap();
        assert_eq!(r.shape(), &[0]);
    }

    #[test]
    #[should_panic(expected = "empty dimension")]
    fn reduce_unordered_panics_on_empty_axis() {
        // `Array::reduce_unordered` unwraps the construction error.
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let _ = a.as_ref().reduce_unordered(1usize, |a, b| a + b);
    }

    #[test]
    fn reduce_unordered_multi_bulk_multi_tile_2d_reduce_outer() {
        // Same chunking scenarios as the `sum` bulk/tile tests, driven through the custom
        // closure instead. Reduce axis 0: bulks split the reduced axis, tiles split the kept one.
        check_reduce_unordered_divided(&[8, 6], &[2, 3], (32, 64), &[0]);
    }

    #[test]
    fn reduce_unordered_multi_bulk_multi_tile_2d_reduce_inner() {
        check_reduce_unordered_divided(&[6, 8], &[3, 2], (32, 64), &[1]);
    }

    #[test]
    fn reduce_unordered_multi_bulk_multi_tile_3d_reduce_middle() {
        check_reduce_unordered_divided(&[4, 4, 4], &[2, 2, 2], (32, 64), &[1]);
    }

    #[test]
    fn reduce_unordered_multi_bulk_long_reduced_axis() {
        // A long, prime-length reduced axis with a tiny read window: the axis is chopped into
        // many bulks whose partial accumulators all have to be merged back together, and 137 is
        // not a multiple of the block extent so the last bulk is a short one.
        //
        // (The *lane* interleaving inside a single tile - which needs a per-tile reduced extent
        // of at least LANES - is covered by the 1D property-test shapes of 100..=1000 elements,
        // which are read in one tile with the default read window.)
        check_reduce_unordered_divided(&[137, 3], &[8, 1], (32, 64), &[0]);
    }

    /// Drives `reduce_unordered` through the two-level bulk/tile chunking (see
    /// [`check_sum_divided`] for why `read_data` is called directly with the whole output range)
    /// and checks two independent things:
    ///
    /// 1. **Value**: `xor` is associative, commutative and exact, so the result must equal a
    ///    sequential reference fold no matter how the stream got partitioned.
    /// 2. **Call count**: combining `n` elements into one accumulator is an `n`-leaf tree, which
    ///    has exactly `n - 1` internal nodes - so the closure must be invoked exactly
    ///    `n_out * (reduction_size - 1)` times. Counting them catches a dropped, duplicated or
    ///    re-seeded element even when `xor` would happen to mask it (e.g. an element folded in
    ///    twice cancels out).
    fn check_reduce_unordered_divided(
        shape: &[usize],
        block_shape: &[u32],
        read_size: (u64, u64),
        axes: &[usize],
    ) {
        use std::cell::Cell;

        use crate::ArrayStorage;

        let n: usize = shape.iter().product();
        let nd = ndarray::ArrayD::from_shape_vec(
            shape.to_vec(),
            (0..n as i32).map(|x| (x % 251) as u8).collect(),
        )
        .unwrap();

        let mut params = crate::ArrayParams::new();
        params.block_shape(block_shape);
        params.read_size(read_size);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();

        let calls = Cell::new(0u64);
        let reduced = za.as_ref().reduce_unordered(axes, |a: u8, b: u8| {
            calls.set(calls.get() + 1);
            xor(a, b)
        });
        let out_shape: Vec<u64> = reduced.shape().to_vec();
        let full_index: Vec<std::ops::Range<u64>> = out_shape.iter().map(|&s| 0..s).collect();
        let n_out: usize = out_shape.iter().product::<u64>() as usize;
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        let mut buf = vec![0u8; n_out.max(1)];
        {
            let sh = full_index
                .iter()
                .map(|r| (r.end - r.start) as usize)
                .collect::<Vec<_>>();
            let c = crate::util::default_strides_slice(&sh, size_of::<u8>());
            let mut out = unsafe {
                crate::storage::StridedBuf::from_slice_mut(&mut buf[..n_out], c.as_ref())
            };
            storage
                .read_data(&full_index, &ctx, Some(&mut out))
                .unwrap();
        }

        let expected = ndarray_reduce(&nd, axes, |v| v.iter().cloned().reduce(xor).unwrap());
        assert_eq!(&buf[..n_out], expected.as_slice().unwrap());

        let reduction_size = n / n_out;
        assert_eq!(
            calls.get(),
            (n_out * (reduction_size - 1)) as u64,
            "each output cell must combine its {reduction_size} elements with exactly \
             {} closure calls",
            reduction_size - 1
        );
    }

    /* TODO(reduction-merge): fold tests removed along with the op; restore later.
    #[test]
    fn fold_single_axis_in_logical_order() {
        // Single-axis fold with a non-commutative closure (subtraction). Logical-order
        // traversal makes the result deterministic:
        // ((((100 - 10) - 1) - 2) - 3) = 84.
        let a = Array::compact_ndarray(&array![10i32, 1, 2, 3]).unwrap();
        let r = a
            .as_ref()
            .fold(0, 100i32, |a, x| a - x)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 84);
    }

    #[test]
    fn fold_widens_output_dtype() {
        // The accumulator type `B` is independent of the input element type. Here we sum
        // a u8 array into an i64.
        let a = Array::compact_ndarray(&array![[1u8, 2, 3], [4, 5, 6]]).unwrap();
        let r = a
            .as_ref()
            .fold((0, 1), 0i64, |a, x| a + x as i64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 21);
        // dtype is the accumulator type, not the input type.
        use crate::dtype::Dtyped;
        let r = a.as_ref().fold((0, 1), 0i64, |a, x| a + x as i64);
        assert_eq!(r.dtype(), &<i64 as Dtyped>::DTYPE);
    }

    #[test]
    fn fold_multi_axis_count_predicate() {
        // Per-row count via fold along axis 1. The closure is commutative + associative
        // (addition), so single-axis order doesn't matter here - but axis 1 is a single
        // axis, so order is still guaranteed.
        let a = Array::compact_ndarray(&array![[1i32, 5, 3, 7], [4, 2, 8, 1]]).unwrap();
        let r = a
            .as_ref()
            .fold(1, 0u64, |c, x| c + (x > 3) as u64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2, 2]);
    }

    #[test]
    fn fold_empty_axis_returns_init() {
        // Folding over an empty axis must produce the init accumulator at every output
        // cell - the closure is never invoked. Reducing axis 1 of a `(2, 0)` array
        // yields shape `[2]` with both cells equal to `init`.
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let r = a
            .as_ref()
            .fold(1, 42i64, |a, x| a + x as i64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r.as_slice().unwrap(), &[42, 42]);
    }

    #[test]
    fn fold_init_passed_through_when_closure_never_runs() {
        // A scalar reduction (reduce all axes) over a 0-element input collapses to a
        // single output cell whose value is exactly `init`.
        use ndarray::Array1;
        let empty: Array1<i32> = Array1::from_shape_vec(0, vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let r = a
            .as_ref()
            .fold(0, 999i64, |a, x| a + x as i64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 999);
    }
    */

    // test_reduction!(
    //     all,
    //     |items| { items.fold(true, |m, x| m && <_ as crate::scalar::Cast<bool>>::cast(x)) },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     logical_op_strategy,
    //     #[cfg(feature = "half")]
    //     [f16],
    //     #[cfg(feature = "num-complex")]
    //     [complex_f32, complex_f64]
    // );
    // test_reduction!(
    //     any,
    //     |items| { items.fold(false, |m, x| m || <_ as crate::scalar::Cast<bool>>::cast(x)) },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     logical_op_strategy,
    //     #[cfg(feature = "half")]
    //     [f16],
    //     #[cfg(feature = "num-complex")]
    //     [complex_f32, complex_f64]
    // );

    fn ndarray_reduce<'a, S, D, O>(
        array: &'a ndarray::ArrayBase<S, D>,
        axes: &[usize],
        f: impl Fn(&ndarray::ArrayViewD<'a, S::Elem>) -> O,
    ) -> ndarray::ArrayD<O>
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        // Output shape = original with reduction axes removed
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();

        let out_shape = array
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &s)| s)
            .collect::<Vec<_>>();

        let values = ndarray_reduction_iter(array, &axes)
            .map(|(_, view)| f(&view))
            .collect::<Vec<_>>();

        ndarray::ArrayD::from_shape_vec(out_shape, values).unwrap()
    }

    /// Iterates over all index combinations of the **kept** axes (i.e. axes NOT in `axes`),
    /// yielding for each combination the multi-index into the kept axes and a view spanning
    /// the reduction axes.
    fn ndarray_reduction_iter<'a, S, D>(
        array: &'a ndarray::ArrayBase<S, D>,
        axes: &[usize],
    ) -> impl Iterator<Item = (Vec<usize>, ndarray::ArrayViewD<'a, S::Elem>)> + 'a
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();

        // Kept axes = all axes not being reduced
        let ndim = array.ndim();
        let kept_axes = (0..ndim).filter(|i| !axes.contains(i)).collect::<Vec<_>>();

        // Shape of the kept axes - this is what we iterate over
        let kept_shape = kept_axes
            .iter()
            .map(|&ax| array.shape()[ax])
            .collect::<Vec<_>>();
        let total: usize = kept_shape.iter().product();

        (0..total).map(move |flat_idx| {
            // Convert flat index to multi-index over the kept axes
            let mut remaining = flat_idx;
            let mut kept_indices = Vec::with_capacity(kept_axes.len());
            for &dim_size in kept_shape.iter().rev() {
                kept_indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            kept_indices.reverse();

            // Fix each kept axis to its index, remove in descending order.
            // We remove kept axes (which are the non-reduction axes), leaving
            // a view over the reduction axes.
            let mut view = array.view().into_dyn();

            // We must track axis offset: as we remove axes, remaining axis
            // indices shift down. Process kept axes in descending order.
            let mut pairs: Vec<(usize, usize)> = kept_axes
                .iter()
                .copied()
                .zip(kept_indices.iter().copied())
                .collect();
            pairs.sort_unstable_by_key(|p| core::cmp::Reverse(p.0));

            for (ax, idx) in &pairs {
                view = view.index_axis_move(ndarray::Axis(*ax), *idx);
            }

            (kept_indices, view)
        })
    }
}
