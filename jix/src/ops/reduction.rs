use std::mem::MaybeUninit;
use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::{Alignment, Dtype, Dtyped};
use crate::error::{bail, check_get_buffer_size, check_get_range, check_ndim, ensure, Result};
use crate::iter::strides::NdIterExtStridesOffset;
use crate::ops::common::AxesArg;
use crate::ops::LanesInfo;
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{ArraySpec, ArrayStorageInfo, ArrayStorageTyped, OutBuf};
use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::util::iter::NdIter;
use crate::util::{calc_block_end, cast_slice_mut, default_logical_strides, DimArray};
use crate::{
    array_from_fn_inline, array_map2_inline, Array, ArrayExt, ArrayStorage, DimVec, Dimension,
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
    /// Build the initial accumulator. `first` is the first element of the reduction stream
    /// when the stream is non-empty, or `None` when the kernel was invoked on an empty
    /// reduction. Kernels with [`supports_empty`](Self::supports_empty) returning `false`
    /// may unwrap `first` - the caller guarantees it is `Some` for those kernels.
    ///
    /// The first element is at position `0`; subsequent calls to [`update_state`] receive
    /// the 0-based stream `idx` of each item.
    ///
    /// [`update_state`]: Self::update_state
    fn init_state(&self, first: Option<T>) -> Self::State;
    /// Fold `item` (at 0-based stream position `idx`) into `state`. `idx` is always `>= 1`
    /// since position `0` is consumed by [`init_state`](Self::init_state).
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State;
    /// Produce the final result. `nitems` is the total number of stream elements that
    /// were folded into `state` (one for `init_state` + one per `update_state` call),
    /// so `nitems == 0` exactly when `first` was `None`.
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

        let shape = array
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(dim, &s)| is_reduced[dim].not().then_some(s))
            .collect::<DimArray<_>>();
        let shape = D::from_slice(&shape);

        let spec = array.spec();
        let spec = ArraySpecDynamic {
            block_shape: (0..input_ndim)
                .filter_map(|dim| is_reduced[dim].not().then_some(spec.block_shape()[dim]))
                .collect(),
            block_shape_tag: (0..input_ndim)
                .filter_map(|dim| is_reduced[dim].not().then_some(spec.block_shape_tag()[dim]))
                .collect(),
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
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        // this is a compile time check, the compiler knows the value of `LANES`
        let read_fn = match <S::Item as LanesInfo>::LANES {
            1 => Self::read_data_impl::<1>,
            2 => Self::read_data_impl::<2>,
            4 => Self::read_data_impl::<4>,
            8 => Self::read_data_impl::<8>,
            16 => Self::read_data_impl::<16>,
            32 => Self::read_data_impl::<32>,
            64 => Self::read_data_impl::<64>,
            128 => Self::read_data_impl::<128>,
            256 => Self::read_data_impl::<256>,
            512 => Self::read_data_impl::<512>,
            _ => Self::read_data_impl::<1024>,
        };

        check_get_range(self.shape(), index)?;
        let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
        let mut buf = buf.get_contiguous_mut(nitems, self.dtype(), context)?;
        read_fn(self, index, buf.as_mut_slice(), context)?;
        let out_shape = D::vec(index.len(), |d| (index[d].end - index[d].start) as usize);
        buf.finalize(out_shape.as_ref(), self.dtype());
        Ok(())
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
impl<S, K, D> ReductionOp<S, K, D>
where
    S: ArrayStorageTyped,
    K: ReductionOpKernel<S::Item, Output: Dtyped>,
    D: Dimension,
{
    #[inline(always)] // weird to inline(always), but its only called from read_data
    fn read_data_impl<const LANES: usize>(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()> {
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
        //   * **bulk** - the outer chunk. Splits *reduced* dims only. Walking bulks in
        //     row-major order is what advances `base_item_idx`: each bulk delivers
        //     `reduction_size = product(bulk_size[d] for d reduced)` items per output position.
        //     `bulk_shape[d] = tile_shape[d]` for reduced dims and `inner_shape[d]`
        //     for non-reduced - i.e. the full non-reduced extent always sits in one bulk
        //     along that dim, so consecutive bulks never re-walk the same outputs.
        //   * **tile** - the inner chunk. Splits *both* dim groups, but is shaped so the
        //     reduced part exactly matches one bulk's reduced range (one tile-along-reduced
        //     per bulk). Within a bulk the tile iterator sweeps the non-reduced output
        //     region. Each `(bulk, tile)` pair produces *one* `self.array.read_data` call,
        //     sized to land within the source's `read_size` `(min, max)` window.
        //
        // `tile_shape` is chosen once per call by the source spec's
        // [`read_shape_heuristic_with_scale_order`], passing a custom scale order that
        // visits reduced dims first (rightmost first), then non-reduced dims (rightmost
        // first). The heuristic seeds every dim from the source storage block hint and
        // greedily scales each dim up in that order until the byte budget is spent.
        // Setting `bulk_shape[reduced] = tile_shape[reduced]` then makes the inner read
        // shape come out to `tile_shape` for every tile.
        //
        // [`read_shape_heuristic_with_scale_order`]: crate::params::ArraySpec::read_shape_heuristic_with_scale_order
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
        // for each bulk:
        //   reduction_size = product(bulk_size[d] for d in reduced dims)
        //   item_idx_end   = base_item_idx + reduction_size
        //   for each tile in this bulk:
        //     items_buf <- self.array.read_data(tile = bulk-reduced * tile-non-reduced)
        //     for each output position O in this tile's non-reduced sub-region:
        //       walk the tile's `reduction_size` items at the right offset into items_buf,
        //       folding them into state_buf[O] via K::init_state / K::update_state. item_idx
        //       runs `[base_item_idx, item_idx_end)` regardless of init-vs-continue branch.
        //   base_item_idx = item_idx_end
        //   if reduction_size > 0 { state_initialized = true }
        // ```
        //
        // The first-init vs. continue split is per-slot inside the inner output loop: when
        // `!state_initialized`, the first reduction item per output is consumed by
        // `K::init_state(first)` to write the slot; control then falls into the same
        // `K::update_state` loop as later bulks. Within the first bulk, *all* tiles take
        // the init branch - each tile only initializes the slots it covers, and together
        // the first bulk's tiles cover every output exactly once. The flag flips only at
        // the *end* of a bulk that contributed items, never inside a bulk.
        //
        // # Finalization
        //
        // After all bulks:
        //   - If any bulk was visited (`state_initialized`): finalize every state into `buf`
        //     via `K::finalize_state(state, reduction_size_overall)`. When
        //     `state_in_out_buf`, the state and output pointers for each slot alias the same
        //     bytes, so each iteration `assume_init_read`s the state *before* writing the
        //     result.
        //   - Otherwise (empty reduction - only reachable when a reduced dim is empty and the
        //     kernel supports empty): write `K::finalize_state(K::init_state(None), 0)` to
        //     every output.
        //
        // # Scratch buffers
        //
        // - `items_buf`: raw input elements for the current *tile*. Resized each tile.
        // - `state_buf`: `out_nitems` slots of `MaybeUninit<K::State>`. Initialized lazily
        //   one tile at a time during the first bulk; finalized in the post-loop pass.
        //   When `K::State` matches `K::Output` in size and is no more strictly aligned
        //   (`state_in_out_buf`), we skip the scratch allocation entirely and reuse the
        //   caller's `buf` as the state buffer - `finalize_state` then reads each slot and
        //   writes the output into the same byte range it just consumed. The finalization
        //   loop reads the state out of each slot *before* writing the result back, since
        //   state and output pointers alias in that mode (see the `CAREFUL` comments).
        //
        // # Invariants (also enforced by `debug_assert!`s)
        //
        // - Non-reduced dims produce at most one bulk-block per call.
        // - Each bulk has exactly one tile-along-reduced
        //   (`tile_shape[reduced] == bulk_shape[reduced]`).
        // - Every tile's absolute element range is contained in `inner_range_full`.
        // - On the first state init pass, `base_item_idx == 0`.
        // - After all bulks, each output's reduction stream consumes exactly
        //   `product(inner_range_full[d].end - .start for d in reduced dims)` items (when
        //   `out_nitems > 0`).

        let out_nitems = check_get_buffer_size(index, &K::Output::DTYPE, buf)?;

        let inner_shape = self.array.shape();
        let inner_ndim = inner_shape.len();
        let out_ndim = index.len();

        let inner_range_full = {
            let mut out_dim = 0;
            S::Dimension::vec(inner_ndim, |dim| {
                if self.is_reduced[dim] {
                    0..inner_shape[dim]
                } else {
                    let r = index[out_dim].clone();
                    out_dim += 1;
                    r
                }
            })
        };

        let out_shape = D::vec(index.len(), |dim| index[dim].end - index[dim].start);

        // Greedy scale-up: reduced dims first (rightmost first), then non-reduced
        // (rightmost first). The reduction kernel walks the reduced axes inside one tile,
        // so giving them first claim on the budget produces fewer outer iterations. Each
        // dim grows by an integer multiplier of its seed (the storage block hint), so the
        // tile stays a multiple of the source's natural block size along that dim.
        let tile_scale_order = (0..inner_ndim)
            .rev()
            .filter(|&dim| self.is_reduced[dim])
            .chain((0..inner_ndim).rev().filter(|&dim| !self.is_reduced[dim]));
        let tile_shape: S::Dimension = self.array.spec().read_shape_heuristic_with_scale_order(
            S::Dimension::vec(inner_ndim, |dim| {
                inner_range_full[dim].end - inner_range_full[dim].start
            })
            .as_ref(),
            self.array.shape(),
            size_of::<S::Item>() as _,
            tile_scale_order,
        );

        // Bulk shape: tile shape on reduced dims (so each bulk has exactly one
        // tile-along-reduced), full source extent on non-reduced dims (so the requested
        // output range sits in one bulk along that dim - otherwise consecutive bulks would
        // re-walk the same outputs and double-count).
        let bulk_shape = S::Dimension::vec(inner_ndim, |dim| {
            if self.is_reduced[dim] {
                tile_shape[dim]
            } else {
                inner_shape[dim].max(1)
            }
        });
        let bulk_grid_begin = S::Dimension::vec(inner_ndim, |dim| {
            inner_range_full[dim].start / bulk_shape[dim]
        });
        let bulk_grid_end = S::Dimension::vec(inner_ndim, |dim| {
            calc_block_end(
                inner_range_full[dim].start,
                inner_range_full[dim].end,
                bulk_shape[dim],
            )
        });
        debug_assert!(
            (0..inner_ndim)
                .all(|d| self.is_reduced[d] || bulk_grid_end[d] - bulk_grid_begin[d] <= 1),
            "non-reduced dim must produce at most one bulk-block",
        );
        let bulk_iter = NdIter::new_with_begin(
            bulk_grid_begin,
            bulk_grid_end,
            NdIterExtBlockOffsetSize::new(
                &S::Dimension::vec(inner_ndim, |dim| inner_range_full[dim].start),
                &S::Dimension::vec(inner_ndim, |dim| inner_range_full[dim].end),
                bulk_shape.clone(),
            ),
        );

        let state_in_out_buf = size_of::<K::State>() == size_of::<K::Output>()
            && align_of::<K::State>() <= align_of::<K::Output>();
        let out_ptr = buf.as_mut_ptr().cast::<K::Output>();
        let mut tmp_state_buf;
        // CAREFUL: state_buf and out_ptr may alias
        let state_buf: &mut [MaybeUninit<K::State>] = if state_in_out_buf {
            unsafe {
                std::slice::from_raw_parts_mut(out_ptr.cast::<MaybeUninit<K::State>>(), out_nitems)
            }
        } else {
            tmp_state_buf = context.tmp_buf_typed::<MaybeUninit<K::State>>(out_nitems);
            unsafe { cast_slice_mut::<_, MaybeUninit<K::State>>(tmp_state_buf.as_mut_slice()) }
        };
        let state_lstrides = default_logical_strides(&out_shape);
        let mut state_initialized = false;

        let mut items_buf = context.tmp_buf(0, Alignment::of::<S::Item>());
        let mut base_item_idx = 0;
        for (bulk_idx, (bulk_inner_offset, bulk_size)) in bulk_iter {
            // The bulk's absolute element range, used as the tile iterator's universe.
            let bulk_begin = S::Dimension::vec(inner_ndim, |dim| {
                bulk_idx[dim] * bulk_shape[dim] + bulk_inner_offset[dim]
            });
            let bulk_end = S::Dimension::vec(inner_ndim, |dim| bulk_begin[dim] + bulk_size[dim]);

            // Each bulk contributes `reduction_size` items to every output it covers
            // (which is every output, since non-reduced dims sit in one bulk). That count
            // is determined by the bulk along reduced dims and is independent of tiling.
            let reduction_size = (0..inner_ndim)
                .filter(|&d| self.is_reduced[d])
                .map(|d| bulk_size[d])
                .product::<u64>();
            let item_idx_end = base_item_idx + reduction_size;

            // Tile iterator: walks the bulk's range partitioned by `tile_shape`. Reduced
            // dims have exactly one tile per bulk (tile_shape[reduced] == bulk reduced
            // width), non-reduced dims are subdivided.
            let tile_grid_begin =
                S::Dimension::vec(inner_ndim, |dim| bulk_begin[dim] / tile_shape[dim]);
            let tile_grid_end = S::Dimension::vec(inner_ndim, |dim| {
                calc_block_end(bulk_begin[dim], bulk_end[dim], tile_shape[dim])
            });
            debug_assert!(
                (0..inner_ndim)
                    .all(|d| { !self.is_reduced[d] || tile_grid_end[d] - tile_grid_begin[d] <= 1 }),
                "reduced dim must produce at most one tile per bulk",
            );
            let tile_iter = NdIter::new_with_begin(
                tile_grid_begin,
                tile_grid_end,
                NdIterExtBlockOffsetSize::new(
                    &bulk_begin,
                    &bulk_end,
                    S::Dimension::vec(inner_ndim, |d| tile_shape[d]), // TODO: clone
                ),
            );

            for (tile_idx, (tile_inner_offset, tile_size)) in tile_iter {
                let tile = S::Dimension::vec(inner_ndim, |dim| {
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

                // Read this tile's items
                items_buf.set_len(
                    (tile_size.as_ref().iter().product::<u64>() * size_of::<S::Item>() as u64)
                        as usize,
                );
                let items_buf = items_buf.as_mut_slice();
                self.array
                    .read_data(tile.as_ref(), &mut OutBuf::new(items_buf), context)?;

                // Output-iterator setup. `tile_out_shape` is the tile's output sub-region;
                // `tile_state_base` shifts `state_buf` to its first slot.
                let items_buf_lstrides = default_logical_strides(&tile_size);
                let items_buf_lstrides_for_out_iter = items_buf_lstrides
                    .as_ref()
                    .iter()
                    .zip(self.is_reduced.as_ref())
                    .filter_map(|(&s, &reduced)| reduced.not().then_some(s))
                    .collect_dim_vec::<D>(out_ndim);

                let tile_out_shape = (0..inner_ndim)
                    .filter(|&d| !self.is_reduced[d])
                    .map(|d| tile_size[d])
                    .collect_dim_vec::<D>(out_ndim);
                let state_offset = (0..inner_ndim)
                    .filter(|&d| !self.is_reduced[d])
                    .enumerate()
                    .map(|(out_d, d)| {
                        (tile[d].start - inner_range_full[d].start) * state_lstrides[out_d]
                    })
                    .sum::<u64>();
                let tile_state_base = unsafe { state_buf.as_mut_ptr().add(state_offset as usize) };

                let mut out_iter = NdIter::new(
                    tile_out_shape,
                    (
                        NdIterExtStridesPtr::new(
                            items_buf_lstrides_for_out_iter,
                            items_buf.as_ptr().cast::<S::Item>(),
                        ),
                        NdIterExtStridesPtrMut::new(
                            default_logical_strides(&out_shape),
                            tile_state_base,
                        ),
                    ),
                );
                // Reduction-axis walk inside `items_buf`. `tile_size[reduced] == bulk_size[reduced]`
                // so this equals `reduction_size`.
                let reduction_shape = S::Dimension::vec(inner_ndim, |dim| {
                    if self.is_reduced[dim] {
                        tile_size[dim]
                    } else {
                        1
                    }
                });
                debug_assert_eq!(
                    reduction_shape.as_ref().iter().product::<u64>(),
                    reduction_size
                );
                self.inner_loop::<LANES>(
                    &mut out_iter,
                    &reduction_shape,
                    &tile_size,
                    reduction_size,
                    base_item_idx,
                    item_idx_end,
                    state_initialized,
                );
                self.inner_loop::<1>(
                    &mut out_iter,
                    &reduction_shape,
                    &tile_size,
                    reduction_size,
                    base_item_idx,
                    item_idx_end,
                    state_initialized,
                );
            }

            base_item_idx = item_idx_end;
            if reduction_size > 0 {
                state_initialized = true;
            }
        }

        // finalize_state
        let reduction_size_overall = base_item_idx;
        debug_assert!(
            out_nitems == 0
                || reduction_size_overall
                    == (0..inner_ndim)
                        .filter(|&d| self.is_reduced[d])
                        .map(|d| inner_range_full[d].end - inner_range_full[d].start)
                        .product::<u64>(),
            "total items folded per output does not match product of reduced-dim sizes",
        );
        // CAREFUL: state_buf and out_ptr may alias
        let state_ptr = state_buf.as_mut_ptr();
        // From here on the state/output buffers are touched only through `state_ptr` and
        // `out_ptr`. Dont use `state_buf`.
        let out_lstrides = default_logical_strides(&out_shape);
        if state_initialized {
            let out_iter = NdIter::new(
                out_shape,
                (
                    // CAREFUL: state_ptr and out_ptr may alias
                    NdIterExtStridesPtrMut::new(state_lstrides, state_ptr),
                    NdIterExtStridesPtrMut::new(out_lstrides, out_ptr),
                ),
            );
            for (_idx, (state, out_ptr)) in out_iter {
                // CAREFUL: state and out_ptr may alias
                let res = {
                    let state = unsafe { (&*state).assume_init_read() };
                    self.kernel.finalize_state(state, reduction_size_overall)
                };
                unsafe { out_ptr.write(res) };
            }
        } else {
            // Empty reduction: write the empty-stream result to every output.
            let out_iter = NdIter::new(
                out_shape,
                NdIterExtStridesPtrMut::new(out_lstrides, out_ptr),
            );
            debug_assert_eq!(reduction_size_overall, 0);
            for (_idx, out_ptr) in out_iter {
                let state = self.kernel.init_state(None);
                let res = self.kernel.finalize_state(state, 0);
                unsafe { out_ptr.write(res) };
            }
        }

        Ok(())
    }

    #[inline]
    fn inner_loop<const LANES: usize>(
        &self,
        out_iter: &mut NdIter<
            D,
            (
                NdIterExtStridesPtr<D, S::Item, u64>,
                NdIterExtStridesPtrMut<D, MaybeUninit<K::State>, u64>,
            ),
        >,
        reduction_shape: &<S::Dimension as Dimension>::Vec<u64>,
        tile_shape: &<S::Dimension as Dimension>::Vec<u64>,
        reduction_size: u64,
        base_item_idx: u64,
        item_idx_end: u64,
        state_initialized: bool,
    ) {
        while out_iter.len() >= LANES as u64 {
            let src_base_and_state = array_from_fn_inline::<_, LANES>(|_| unsafe {
                out_iter.next().unwrap_unchecked().1
            });
            let src_base = src_base_and_state.map_inline_ref(|(src_base, _state_ptr)| *src_base);
            let state_ptr = src_base_and_state.map_inline_ref(|(_src_base, state_ptr)| *state_ptr);
            let reduction_iter = NdIter::new(
                reduction_shape.clone(),
                NdIterExtStridesOffset::new(default_logical_strides(tile_shape), 0),
            );
            debug_assert_eq!(reduction_size, reduction_iter.len());
            let mut reduction_iter = reduction_iter.map(|(_idx, offset)| {
                src_base.map_inline_ref(|src_base| unsafe { src_base.add(offset as usize).read() })
            });
            let mut item_idx = base_item_idx;

            let state_ref = state_ptr.map_inline_ref(|&state_ptr| unsafe { &mut *state_ptr });
            if !state_initialized {
                // init state with the first item
                debug_assert_eq!(item_idx, 0);
                let first = reduction_iter.next();
                match first {
                    Some(first) => {
                        for i in 0..LANES {
                            state_ref[i].write(self.kernel.init_state(Some(first[i])));
                        }
                    }
                    None => {
                        for i in 0..LANES {
                            state_ref[i].write(self.kernel.init_state(None));
                        }
                    }
                }
                if first.is_some() {
                    item_idx += 1;
                }
            }
            // SAFETY: every state was written during the first bulk.
            let mut state =
                state_ref.map_inline_ref(|state_ref| unsafe { state_ref.assume_init_read() });

            while item_idx < item_idx_end {
                let item = reduction_iter.next();
                let item = unsafe { item.unwrap_unchecked() };
                state = array_map2_inline(state, item, |state_i, item_i| {
                    self.kernel.update_state(state_i, item_i, item_idx)
                });

                item_idx += 1;
            }
            debug_assert!(reduction_iter.next().is_none());
            for (state_ref, state) in state_ref.into_iter().zip(state) {
                state_ref.write(state);
            }
        }
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
    /// Accumulates into a wider output type to reduce overflow risk: integer types accumulate
    /// into `i64`/`u64`, floating-point types accumulate into `f64`.
    pub trait Sum {
        /// The sum element type (wider than the input for most types).
        type Output;
        /// Return the initial accumulator (zero).
        fn init() -> Self::Output;
        /// Fold `item` into the running sum.
        fn update(state: Self::Output, item: Self) -> Self::Output;
    }

    macro_rules! impl_sum {
        ($item_ty:ty, $output_ty:ty) => {
            impl Sum for $item_ty {
                type Output = $output_ty;

                #[inline(always)]
                fn init() -> Self::Output {
                    <i32 as crate::scalar::Cast<Self::Output>>::cast(0)
                }
                #[inline(always)]
                fn update(state: Self::Output, item: Self) -> Self::Output {
                    state + <_ as crate::scalar::Cast<Self::Output>>::cast(item)
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
    impl_sum!(f16, f64);
    impl_sum!(f32, f64);
    impl_sum!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f64>, Complex<f64>);
    impl_sum!(bool, u64);

    /// Scalar kernel trait for the element-wise `product` reduction.
    ///
    /// Accumulates into a wider output type to reduce overflow risk: integer types accumulate
    /// into `i64`/`u64`, floating-point types accumulate into `f64`.
    pub trait Product {
        /// The product element type (wider than the input for most types).
        type Output;
        /// Return the initial accumulator (one).
        fn init() -> Self::Output;
        /// Fold `item` into the running product.
        fn update(state: Self::Output, item: Self) -> Self::Output;
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
    impl_product!(f16, f64);
    impl_product!(f32, f64);
    impl_product!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f64>, Complex<f64>);

    /// Scalar kernel trait for the element-wise `mean` reduction.
    ///
    /// The mean is computed as the sum divided by the count; the output is always `f64`
    /// (or `Complex<f64>` for complex inputs) to preserve precision.
    ///
    /// The count is tracked **outside** the accumulator: callers thread the number of
    /// folded items into [`finalize`](Self::finalize) themselves.
    pub trait Mean {
        /// The output element type - always `f64` or `Complex<f64>`.
        type Output;
        /// Accumulator state - the running sum.
        type State;
        /// Return the initial (empty) accumulator.
        fn init() -> Self::State;
        /// Fold `item` into the running sum.
        fn update(state: Self::State, item: Self) -> Self::State;
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
                fn finalize(state: Self::State, nitems: u64) -> Option<Self::Output> {
                    if nitems == 0 {
                        return None;
                    }
                    Some(<_ as crate::scalar::Cast<Self::Output>>::cast(state) / nitems as f64)
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
    impl_mean!(f16, f64);
    impl_mean!(f32, f64);
    impl_mean!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f64>, Complex<f64>);
    impl_mean!(bool, f64);

    /// Welford accumulator used by [`Variance`]. The count of folded items is tracked
    /// **outside** this struct - callers thread it into [`Variance::update`] (as `n`)
    /// and [`Variance::finalize`] (as `nitems`).
    pub struct VarianceState<M> {
        /// Running mean in the type-specific accumulator domain.
        mean: M,
        /// Running sum of squared deviations from the mean (`f64`).
        m2: f64,
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
        /// The output element type - always a `Float` (i.e. `f64` for most inputs).
        type Output;
        /// Welford accumulator state.
        type State;
        /// Return the initial (empty) accumulator.
        fn init() -> Self::State;
        /// Fold `item` into the running Welford accumulator. `idx` is the 0-based stream position
        /// of `item.
        fn update(state: Self::State, item: Self, idx: u64) -> Self::State;
        /// Finalize `state` into the variance using `ddof` degrees-of-freedom correction.
        /// `nitems` is the total number of elements folded in.
        ///
        /// Returns `NaN` if the effective denominator (`nitems - ddof`) is non-positive.
        fn finalize(state: Self::State, ddof: f64, nitems: u64) -> Self::Output;
    }
    macro_rules! impl_variance {
        ($item_ty:ty, $mean_ty:ty, |$delta:ident, $delta2:ident| $m2_expr:expr) => {
            impl Variance for $item_ty {
                type Output = f64;
                type State = VarianceState<$mean_ty>;

                #[inline(always)]
                fn init() -> Self::State {
                    VarianceState {
                        mean: <i32 as crate::scalar::Cast<$mean_ty>>::cast(0),
                        m2: 0.0,
                    }
                }
                #[inline(always)]
                fn update(mut state: Self::State, item: Self, idx: u64) -> Self::State {
                    let x = <_ as crate::scalar::Cast<$mean_ty>>::cast(item);
                    let $delta = x - state.mean;
                    state.mean += $delta / (idx + 1) as f64;
                    let $delta2 = x - state.mean;
                    state.m2 += $m2_expr;
                    state
                }
                #[inline(always)]
                fn finalize(state: Self::State, ddof: f64, nitems: u64) -> Self::Output {
                    let denom = nitems as f64 - ddof;
                    if denom <= 0.0 {
                        f64::NAN
                    } else {
                        state.m2 / denom
                    }
                }
            }
        };
    }
    impl_variance!(i8, f64, |delta, delta2| delta * delta2);
    impl_variance!(i16, f64, |delta, delta2| delta * delta2);
    impl_variance!(i32, f64, |delta, delta2| delta * delta2);
    impl_variance!(i64, f64, |delta, delta2| delta * delta2);
    impl_variance!(u8, f64, |delta, delta2| delta * delta2);
    impl_variance!(u16, f64, |delta, delta2| delta * delta2);
    impl_variance!(u32, f64, |delta, delta2| delta * delta2);
    impl_variance!(u64, f64, |delta, delta2| delta * delta2);
    #[cfg(feature = "half")]
    impl_variance!(f16, f64, |delta, delta2| delta * delta2);
    impl_variance!(f32, f64, |delta, delta2| delta * delta2);
    impl_variance!(f64, f64, |delta, delta2| delta * delta2);
    #[cfg(feature = "num-complex")]
    impl_variance!(Complex<f32>, Complex<f64>, |delta, delta2| delta.re
        * delta2.re
        + delta.im * delta2.im);
    #[cfg(feature = "num-complex")]
    impl_variance!(Complex<f64>, Complex<f64>, |delta, delta2| delta.re
        * delta2.re
        + delta.im * delta2.im);
    impl_variance!(bool, f64, |delta, delta2| delta * delta2);
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        first.unwrap()
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        state.maximum(item)
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        first.unwrap()
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        state.minimum(item)
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
    /// the maximum value, the index of the first occurrence is returned.
    /// For **float** types, a `NaN` never displaces the running best, because every
    /// comparison against `NaN` evaluates to `false`. A `NaN` index is therefore returned
    /// only when the first element along the reduced axis is `NaN`; otherwise `NaN`
    /// values are skipped. This differs from `numpy.argmax`, which returns the index of
    /// the first `NaN`.
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
impl<T> ReductionOpKernel<T> for ArgMaxKernel
where
    T: PartialOrd,
{
    type Output = u64;
    /// `(best_idx, best_val)`.
    type State = (u64, T);

    #[inline(always)]
    fn init_state(&self, first: Option<T>) -> Self::State {
        (0, first.unwrap())
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        let (best_idx, best_val) = state;
        if item > best_val {
            (idx, item)
        } else {
            (best_idx, best_val)
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
    /// the minimum value, the index of the first occurrence is returned.
    /// For **float** types, a `NaN` never displaces the running best, because every
    /// comparison against `NaN` evaluates to `false`. A `NaN` index is therefore returned
    /// only when the first element along the reduced axis is `NaN`; otherwise `NaN`
    /// values are skipped. This differs from `numpy.argmin`, which returns the index of
    /// the first `NaN`.
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
impl<T> ReductionOpKernel<T> for ArgMinKernel
where
    T: PartialOrd,
{
    type Output = u64;
    /// `(best_idx, best_val)`.
    type State = (u64, T);

    #[inline(always)]
    fn init_state(&self, first: Option<T>) -> Self::State {
        (0, first.unwrap())
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        let (best_idx, best_val) = state;
        if item < best_val {
            (idx, item)
        } else {
            (best_idx, best_val)
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
    /// | `f16`, `f32`, `f64` | `f64` |
    /// | `Complex<f32>`, `Complex<f64>` | `Complex<f64>` |
    ///
    /// The output dtype is always widened to avoid overflow on large reductions.
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        let mut state = <T as crate::scalar::Sum>::init();
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        <T as crate::scalar::Sum>::update(state, item)
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
    /// | `f16`, `f32`, `f64` | `f64` |
    /// | `Complex<f32>`, `Complex<f64>` | `Complex<f64>` |
    ///
    /// Note: `bool` is not supported. The output dtype is always widened.
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        let mut state = <T as crate::scalar::Product>::init();
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        <T as crate::scalar::Product>::update(state, item)
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
    /// Output dtype is `f64` for all scalar inputs; `Complex<f64>` for `Complex<f32>` and
    /// `Complex<f64>` inputs.
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        let mut state = <T as crate::scalar::Mean>::init();
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        <T as crate::scalar::Mean>::update(state, item)
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
    /// Output dtype is `f64`. For complex inputs the result is the real-valued variance
    /// `E[|x - mean|^2]`.
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        let mut state = <T as crate::scalar::Variance>::init();
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        <T as crate::scalar::Variance>::update(state, item, idx)
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
    /// Output dtype is `f64`. For complex inputs the result is the real-valued standard
    /// deviation `sqrt(E[|x - mean|^2])`.
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
    fn init_state(&self, first: Option<T>) -> Self::State {
        let mut state = <T as crate::scalar::Variance>::init();
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: u64) -> Self::State {
        <T as crate::scalar::Variance>::update(state, item, idx)
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
    fn init_state(&self, first: Option<bool>) -> Self::State {
        let mut state = true;
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: bool, _idx: u64) -> Self::State {
        state && item
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
    fn init_state(&self, first: Option<bool>) -> Self::State {
        let mut state = false;
        if let Some(item) = first {
            state = self.update_state(state, item, 0);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: bool, _idx: u64) -> Self::State {
        state || item
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

/// Reduces one or more axes by repeatedly applying a user-supplied binary closure to the
/// elements along those axes.
///
/// The output dtype is the same as the input dtype (`S::Item`). The closure has signature
/// `Fn(S::Item, S::Item) -> S::Item` and is applied with the running accumulator on the
/// left, mirroring [`Iterator::reduce`]: for a non-empty stream `x0, x1, ..., xn`, the
/// result is `f(f(f(x0, x1), x2), ..., xn)`.
///
/// # Traversal order
///
/// - **Single reduced axis**: elements along that axis are visited in logical order (index
///   `0` upward). The result is well-defined for non-commutative / non-associative closures.
/// - **Multiple reduced axes**: elements are visited in an *implementation-defined* order
///   driven by the storage's internal tiling. The order is not stable across array shapes,
///   block sizes, or library versions. Closures used here MUST be both associative and
///   commutative for the result to be well-defined.
///
/// # Empty reductions
///
/// Empty reductions (any reduced dimension has length `0`) are **not supported**: there is
/// no initial accumulator and no first element to seed it. Calling [`Array::reduce`] in
/// that case panics at construction time, and [`Reduce::new`] returns an `Err`.
/// Use [`Fold`] when an explicit empty-case value is needed.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation; the operation is also available as
/// [`Array::reduce()`](crate::Array::reduce).
///
/// # Examples
///
/// Custom maximum over a single axis (closure is commutative and associative):
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1i32, 5, 3], [4, 2, 6]];
/// let row_max = Array::compact_ndarray(&nd)?
///     .reduce(1, |a, b| if a > b { a } else { b })
///     .to_ndarray()?;
/// assert_eq!(row_max.as_slice().unwrap(), &[5, 6]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Single-axis subtraction: order matters, and a single reduced axis is guaranteed to be
/// visited in logical order:
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // (((1 - 2) - 3) - 4) = -8
/// let nd = array![1i64, 2, 3, 4];
/// let diff = Array::compact_ndarray(&nd)?
///     .reduce(0, |a, b| a - b)
///     .to_ndarray()?;
/// assert_eq!(diff[[]], -8);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Multi-axis reduction (closure MUST be associative and commutative):
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1i32, 2, 3], [4, 5, 6]];
/// let total = Array::compact_ndarray(&nd)?
///     .reduce((0, 1), |a, b| a + b)
///     .to_ndarray()?;
/// assert_eq!(total[[]], 21);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Reduce<S: ArrayStorage, D, F>(ReductionOp<S, VanillaReduceKernel<F>, D>);
impl<S: ArrayStorage, D, F> Reduce<S, D, F> {
    /// Constructs a [`Reduce`] storage. See the struct docs for semantics and examples.
    pub fn new<Ax>(array: S, axes: Ax, f: F) -> Result<Self>
    where
        S: ArrayStorageTyped,
        D: Dimension,
        F: Fn(S::Item, S::Item) -> S::Item,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        Ok(Self(ReductionOp::new(array, VanillaReduceKernel(f), axes)?))
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
struct VanillaReduceKernel<F>(F);
impl<T, F> ReductionOpKernel<T> for VanillaReduceKernel<F>
where
    F: Fn(T, T) -> T,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, first: Option<T>) -> Self::State {
        first.unwrap()
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: u64) -> Self::State {
        (self.0)(state, item)
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
        if let Some(item) = first {
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
    #[track_caller]
    pub fn reduce<F, Ax>(
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

    use crate::array::Array;
    use crate::storage::Compact;
    use crate::{DimDyn, Ty};

    /// Per-dtype comparison policy for the reduction property tests.
    ///
    /// A reduction reads its input block-by-block, so a floating accumulator
    /// reassociates and can land a few ULP away from the sequential reference
    /// fold. Every float and complex reduction folds into the widening
    /// `f64` / `Complex<f64>` accumulator (which is therefore the element type of
    /// `expected`), so those two types are compared with [`ApproxEq`]; integer
    /// and `bool` reductions are exact and compared bit-for-bit. `f32` / `f16`
    /// (and `Complex<f32>`) only ever surface here as `max` / `min` results,
    /// which just select an input element and so also compare exactly.
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
    reduction_compare_exact!(i8, i16, i32, i64, u8, u16, u32, u64, bool, f32);
    #[cfg(feature = "half")]
    reduction_compare_exact!(f16);
    #[cfg(feature = "num-complex")]
    reduction_compare_exact!(complex_f32);

    impl ReductionCompare for f64 {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
            crate::util::assert_array_matches_approx(actual, expected, 1e-9, 1e-6);
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

    /// Asserts a reduction result matches its reference, dispatching exact vs.
    /// approximate comparison on the accumulator type via [`ReductionCompare`].
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

    pub(crate) fn carray_strategy_for_reduction_small<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<
        Value = (ArrayD<T>, Rc<Array<Compact<Ty<T>, DimDyn>>>, Vec<usize>),
    > {
        let shape = prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=4, 1)),
            // 2D
            (8, proptest::collection::vec(1..=2, 2)),
        ]);
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axes = axes_strategy(nd.ndim());
                (Just(nd), Just(za), axes)
            })
    }

    macro_rules! test_reduction_dtype {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axes) in crate::ops::reduction::tests::carray_strategy_for_reduction::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(&axes);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &axes,
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

        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident,
            small_data = true
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axes) in crate::ops::reduction::tests::carray_strategy_for_reduction_small::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(&axes);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &axes,
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
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*,
            small_data = true
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy,
                small_data = true
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy,
                    small_data = true
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
    test_reduction!(
        min,
        |items| { items.reduce(|m, x| if x < m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    // test_reduction!( // TODO
    //     argmax,
    //     single_axis = true,
    //     |items| {
    //         items
    //             .enumerate()
    //             .reduce(|(m_i, m), (i, x)| if x > m { (i, x) } else { (m_i, m) })
    //             .unwrap()
    //             .0 as u64
    //     },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     any_strategy,
    //     #[cfg(feature = "half")]
    //     [f16]
    // );
    // test_reduction!(
    //     argmin,
    //     single_axis = true,
    //     |items| {
    //         items
    //             .enumerate()
    //             .reduce(|(m_i, m), (i, x)| if x < m { (i, x) } else { (m_i, m) })
    //             .unwrap()
    //             .0 as u64
    //     },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     any_strategy,
    //     #[cfg(feature = "half")]
    //     [f16]
    // );
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
    test_reduction!(
        sum,
        |items| { items.fold(0.0f64, |m, x| m + <_ as crate::scalar::Cast<f64>>::cast(x)) },
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        sum,
        |items| {
            items.fold(Complex::<f64>::default(), |m, x| {
                m + <_ as crate::scalar::Cast<Complex<f64>>>::cast(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    test_reduction!(
        product,
        |items| { items.fold(1u64, |m, x| m * x as u64) },
        [u8, u16, u32, u64],
        op_safe_strategy,
        small_data = true
    );
    test_reduction!(
        product,
        |items| { items.fold(1i64, |m, x| m * x as i64) },
        [i8, i16, i32, i64],
        op_safe_strategy,
        small_data = true
    );
    test_reduction!(
        product,
        |items| { items.fold(1.0f64, |m, x| m * <_ as crate::scalar::Cast<f64>>::cast(x)) },
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        small_data = true
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        product,
        |items| {
            items.fold(Complex::<f64>::new(1.0, 0.0), |m, x| {
                m * <_ as crate::scalar::Cast<Complex<f64>>>::cast(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy,
        small_data = true
    );
    // mean
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
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        mean,
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

    #[test]
    fn reduce_single_axis_in_logical_order() {
        // A single reduced axis must be visited in logical (index-ascending) order, so
        // non-commutative closures produce a well-defined result. Subtraction is the
        // canonical witness: (((1 - 2) - 3) - 4) = -8.
        let a = Array::compact_ndarray(&array![1i64, 2, 3, 4]).unwrap();
        let r = a.as_ref().reduce(0, |a, b| a - b).to_ndarray().unwrap();
        assert_eq!(r[[]], -8);
    }

    #[test]
    fn reduce_single_axis_per_row() {
        // Single axis on a 2D array: reduce axis 1 (columns). Each row is folded
        // left-to-right with subtraction.
        // row 0: ((10 - 1) - 2) = 7
        // row 1: ((20 - 5) - 3) = 12
        let a = Array::compact_ndarray(&array![[10i32, 1, 2], [20, 5, 3]]).unwrap();
        let r = a.as_ref().reduce(1, |a, b| a - b).to_ndarray().unwrap();
        assert_eq!(r.as_slice().unwrap(), &[7, 12]);
    }

    #[test]
    fn reduce_multi_axis_sum() {
        // Multi-axis reduction. The closure here is associative + commutative, which is
        // required for the result to be well-defined (the traversal order across multiple
        // reduced axes is implementation-defined).
        let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let r = a
            .as_ref()
            .reduce((0, 1), |a, b| a + b)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 21);
    }

    #[test]
    fn reduce_preserves_dtype() {
        // The output dtype must equal the input dtype.
        use crate::dtype::Dtyped;
        let a = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let r = a.as_ref().reduce(0, |a, b| a + b);
        assert_eq!(r.dtype(), &<i32 as Dtyped>::DTYPE);
    }

    #[test]
    #[should_panic]
    fn reduce_panics_on_empty_axis() {
        // Reducing along an empty axis is unsupported: there's no first element to seed
        // the accumulator. `Array::reduce` unwraps the construction error and panics.
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let _ = a.as_ref().reduce(1, |a, b| a + b);
    }

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

        let out_shape: Vec<usize> = array
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &s)| s)
            .collect();

        let values: Vec<O> = ndarray_reduction_iter(array, &axes)
            .map(|(_, view)| f(&view))
            .collect();

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
        let kept_axes: Vec<usize> = (0..ndim).filter(|i| !axes.contains(i)).collect();

        // Shape of the kept axes - this is what we iterate over
        let kept_shape: Vec<usize> = kept_axes.iter().map(|&ax| array.shape()[ax]).collect();
        let total: usize = kept_shape.iter().product();

        (0..total).map(move |flat_idx| {
            // Convert flat index to multi-index over the kept axes
            let mut remaining = flat_idx;
            let mut kept_indices: Vec<usize> = Vec::with_capacity(kept_axes.len());
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
            pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));

            for (ax, idx) in &pairs {
                view = view.index_axis_move(ndarray::Axis(*ax), *idx);
            }

            (kept_indices, view)
        })
    }

    mod ndarray_reduce_tests {
        use super::{ndarray_reduce, ndarray_reduction_iter};

        #[cfg(test)]
        mod tests {
            use ndarray::{array, Array};

            use super::*;

            #[test]
            fn single_axis_0() {
                // Shape [2, 3], reduce axis 0 -> 3 views of shape [2]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0]).collect();

                assert_eq!(views.len(), 3);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[2]);
                }

                // kept axis is 1, so indices are [0], [1], [2]
                // view[0] = a[:, 0] = [1, 4]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(views[0].1, array![1, 4].into_dyn());
                // view[1] = a[:, 1] = [2, 5]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(views[1].1, array![2, 5].into_dyn());
                // view[2] = a[:, 2] = [3, 6]
                assert_eq!(views[2].0, vec![2]);
                assert_eq!(views[2].1, array![3, 6].into_dyn());
            }

            #[test]
            fn single_axis_1() {
                // Shape [2, 3], reduce axis 1 -> 2 views of shape [3]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[1]).collect();

                assert_eq!(views.len(), 2);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[3]);
                }

                // kept axis is 0, so indices are [0], [1]
                // view[0] = a[0, :] = [1, 2, 3]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(views[0].1, array![1, 2, 3].into_dyn());
                // view[1] = a[1, :] = [4, 5, 6]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(views[1].1, array![4, 5, 6].into_dyn());
            }

            #[test]
            fn multi_axis_3d() {
                // Shape [2, 3, 4], reduce axes [0, 2] -> 3 views of shape [2, 4]
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0, 2]).collect();

                assert_eq!(views.len(), 3);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[2, 4]);
                }

                // kept axis is 1, indices are [0], [1], [2]
                // view[0] = a[:, 0, :] = [[0,1,2,3],[12,13,14,15]]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(
                    views[0].1,
                    array![[0, 1, 2, 3], [12, 13, 14, 15]].into_dyn()
                );
                // view[1] = a[:, 1, :] = [[4,5,6,7],[16,17,18,19]]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(
                    views[1].1,
                    array![[4, 5, 6, 7], [16, 17, 18, 19]].into_dyn()
                );
                // view[2] = a[:, 2, :] = [[8,9,10,11],[20,21,22,23]]
                assert_eq!(views[2].0, vec![2]);
                assert_eq!(
                    views[2].1,
                    array![[8, 9, 10, 11], [20, 21, 22, 23]].into_dyn()
                );
            }

            #[test]
            fn reduce_all_axes() {
                // Shape [2, 3], reduce both -> 1 view of shape [2, 3] (no kept axes)
                let a = Array::from_shape_vec(vec![2, 3], vec![10, 20, 30, 40, 50, 60]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0, 1]).collect();

                assert_eq!(views.len(), 1);
                assert_eq!(views[0].0, Vec::<usize>::new());
                assert_eq!(views[0].1, array![[10, 20, 30], [40, 50, 60]].into_dyn());
            }

            #[test]
            fn no_axes_returns_scalar_views() {
                // Reduce no axes -> 6 scalar views (iterate over everything)
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[]).collect();

                assert_eq!(views.len(), 6);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[] as &[usize]);
                }

                assert_eq!(views[0].0, vec![0, 0]);
                assert_eq!(*views[0].1.first().unwrap(), 1);
                assert_eq!(views[1].0, vec![0, 1]);
                assert_eq!(*views[1].1.first().unwrap(), 2);
                assert_eq!(views[5].0, vec![1, 2]);
                assert_eq!(*views[5].1.first().unwrap(), 6);
            }

            #[test]
            fn axes_order_independent() {
                // [0, 2] and [2, 0] should yield identical results
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();

                let v1: Vec<_> = ndarray_reduction_iter(&a, &[0, 2]).collect();
                let v2: Vec<_> = ndarray_reduction_iter(&a, &[2, 0]).collect();

                assert_eq!(v1.len(), v2.len());
                for ((idx1, view1), (idx2, view2)) in v1.iter().zip(v2.iter()) {
                    assert_eq!(idx1, idx2);
                    assert_eq!(view1, view2);
                }
            }

            #[test]
            fn dim_1_reduce_axis_0() {
                // Shape [5], reduce axis 0 -> 1 view of shape [5] (no kept axes)
                let a = Array::from_shape_vec(vec![5], vec![10, 20, 30, 40, 50]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0]).collect();

                assert_eq!(views.len(), 1);
                assert_eq!(views[0].0, Vec::<usize>::new());
                assert_eq!(views[0].1, array![10, 20, 30, 40, 50].into_dyn());
            }

            #[test]
            fn reduce_middle_axis() {
                // Shape [2, 3, 4], reduce axis 1 -> 2*4=8 views of shape [3]
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[1]).collect();

                assert_eq!(views.len(), 8);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[3]);
                }

                // kept axes are [0, 2]
                // view[0]: kept=[0,0] -> a[0, :, 0] = [0, 4, 8]
                assert_eq!(views[0].0, vec![0, 0]);
                assert_eq!(views[0].1, array![0, 4, 8].into_dyn());
                // view[3]: kept=[0,3] -> a[0, :, 3] = [3, 7, 11]
                assert_eq!(views[3].0, vec![0, 3]);
                assert_eq!(views[3].1, array![3, 7, 11].into_dyn());
                // view[4]: kept=[1,0] -> a[1, :, 0] = [12, 16, 20]
                assert_eq!(views[4].0, vec![1, 0]);
                assert_eq!(views[4].1, array![12, 16, 20].into_dyn());
                // view[7]: kept=[1,3] -> a[1, :, 3] = [15, 19, 23]
                assert_eq!(views[7].0, vec![1, 3]);
                assert_eq!(views[7].1, array![15, 19, 23].into_dyn());
            }

            // --- ndarray_reduce tests ---

            #[test]
            fn reduce_sum_axis_0() {
                // np.sum(a, axis=0) for shape [2, 3]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[0], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[3]);
                assert_eq!(result, array![5, 7, 9].into_dyn());
            }

            #[test]
            fn reduce_sum_axis_1() {
                // np.sum(a, axis=1) for shape [2, 3]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[1], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[2]);
                assert_eq!(result, array![6, 15].into_dyn());
            }

            #[test]
            fn reduce_sum_multi_axis() {
                // np.sum(a, axis=(0, 2)) for shape [2, 3, 4]
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
                let result = ndarray_reduce(&a, &[0, 2], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[3]);
                // axis 1 index 0: sum of a[:, 0, :] = sum(0..4) + sum(12..16) = 6 + 54 = 60
                // axis 1 index 1: sum of a[:, 1, :] = sum(4..8) + sum(16..20) = 22 + 70 = 92
                // axis 1 index 2: sum of a[:, 2, :] = sum(8..12) + sum(20..24) = 38 + 86 = 124
                assert_eq!(result, array![60, 92, 124].into_dyn());
            }

            #[test]
            fn reduce_all_axes_to_scalar() {
                // np.sum(a) - reduce everything
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[0, 1], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[] as &[usize]);
                assert_eq!(*result.first().unwrap(), 21);
            }

            #[test]
            fn reduce_no_axes_identity() {
                // Reducing no axes -> same shape, each element passed through f
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[], |v| *v.first().unwrap());

                assert_eq!(result.shape(), &[2, 3]);
                assert_eq!(result, array![[1, 2, 3], [4, 5, 6]].into_dyn());
            }

            #[test]
            fn reduce_max_axis() {
                // np.max(a, axis=0)
                let a = Array::from_shape_vec(vec![3, 2], vec![5, 1, 3, 8, 7, 2]).unwrap();
                let result = ndarray_reduce(&a, &[0], |v| *v.iter().max().unwrap());

                assert_eq!(result.shape(), &[2]);
                assert_eq!(result, array![7, 8].into_dyn());
            }
        }
    }
}
