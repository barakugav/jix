use std::mem::MaybeUninit;
use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::{Alignment, Dtype, Dtyped};
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::ops::common::AxesArg;
use crate::ops::BulkInfo;
#[allow(unused_imports)]
use crate::scalar::{f16, Complex};
use crate::storage::{ArrayStorageSpec, ArrayStorageTyped, BlocksLayout};
use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::util::iter::NdIter;
use crate::util::{assert_unchecked_eq, cast_slice_mut, default_strides, dim_arr, DimArray};
use crate::{Array, ArrayStorage, Dimension, Ty};

pub(crate) struct ReductionOp<S, K, D> {
    kernel: K,

    array: S,
    is_reduced: DimArray<bool>,

    out_dtype_: Dtype,
    shape: D,
    blocks_layout: BlocksLayout,
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

impl<S, K, D> ReductionOp<S, K, D> {
    pub(crate) fn new<Ax>(array: S, kernel: K, axes: Ax) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: ReductionOpKernel<S::Item, Output: Dtyped>,
        D: Dimension,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        let input_ndim = array.shape().len();
        let mut is_reduced = dim_arr(input_ndim, |_| false);
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
                .zip(&is_reduced)
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
        let shape = D::from_slice(&shape).unwrap();

        let mut b_layout = array.spec().blocks_layout.clone();
        b_layout.block_shape_hint = (0..input_ndim)
            .filter_map(|dim| {
                is_reduced[dim]
                    .not()
                    .then_some(b_layout.block_shape_hint[dim])
            })
            .collect();
        b_layout.block_shape_tag = (0..input_ndim)
            .filter_map(|dim| {
                is_reduced[dim]
                    .not()
                    .then_some(b_layout.block_shape_tag[dim])
            })
            .collect();
        b_layout.preferred_read_shape = (0..input_ndim)
            .filter_map(|dim| {
                is_reduced[dim]
                    .not()
                    .then_some(b_layout.preferred_read_shape[dim])
            })
            .collect();

        Ok(Self {
            kernel,
            out_dtype_: K::Output::DTYPE,
            shape,
            blocks_layout: b_layout,
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
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        // this is a compile time check, the compiler knows the value of `BULK`
        let read_fn = match <S::Item as BulkInfo>::BULK {
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
        read_fn(self, index, buf, context)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        let dtype = &self.out_dtype_;
        unsafe { assert_unchecked_eq!(dtype, &K::Output::DTYPE) };
        dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.spec()
        }
    }
}
impl<S, K, D> ReductionOp<S, K, D>
where
    S: ArrayStorageTyped,
    K: ReductionOpKernel<S::Item, Output: Dtyped>,
    D: Dimension,
{
    #[inline]
    fn read_data_impl<const SIMD: usize>(
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
        //     sized to roughly fit the source's `preferred_read_size_hint`.
        //
        // `tile_shape` is chosen once per call by [`choose_tile_shape`]: seed every dim
        // from the source storage block hint (clamped to `inner_range_full[d].len()`), then
        // greedily scale each dim up by an integer multiplier of its seed - reduced dims
        // first (rightmost first), then non-reduced (rightmost first) - until the running
        // product would exceed `target_nitems = preferred_read_size_hint /
        // size_of::<S::Item>()`. The per-dim multiplier is also capped so the tile can't
        // overshoot the requested range. A final fixup snaps any dim whose tile already
        // covers the full requested range to `inner_shape[d]`: tiles are aligned to
        // absolute multiples of `tile_shape[d]`, so a tile sized exactly to the range can
        // still get split when `inner_range_full[d].start` is not a multiple of the tile
        // shape - using the full source extent guarantees one tile along that dim. Setting
        // `bulk_shape[reduced] = tile_shape[reduced]` then makes the inner read shape come
        // out to `tile_shape` for every tile.
        //
        // [`choose_tile_shape`]: Self::choose_tile_shape
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

        check_get_range(self.shape(), index)?;
        let out_nitems = check_get_buffer_size(index, &K::Output::DTYPE, buf)?;

        let inner_shape = self.array.shape();
        let inner_ndim = inner_shape.len();

        let inner_range_full = {
            let mut out_dim = 0;
            dim_arr(inner_ndim, |dim| {
                if self.is_reduced[dim] {
                    0..inner_shape[dim]
                } else {
                    let r = index[out_dim].clone();
                    out_dim += 1;
                    r
                }
            })
        };

        let out_shape = dim_arr(index.len(), |dim| index[dim].end - index[dim].start);

        let tile_shape = self.choose_tile_shape(&inner_range_full);

        // Bulk shape: tile shape on reduced dims (so each bulk has exactly one
        // tile-along-reduced), full source extent on non-reduced dims (so the requested
        // output range sits in one bulk along that dim - otherwise consecutive bulks would
        // re-walk the same outputs and double-count).
        let bulk_shape = dim_arr(inner_ndim, |dim| {
            if self.is_reduced[dim] {
                tile_shape[dim]
            } else {
                inner_shape[dim].max(1)
            }
        });
        let bulk_grid_begin = dim_arr(inner_ndim, |dim| {
            inner_range_full[dim].start / bulk_shape[dim]
        });
        let bulk_grid_end = dim_arr(inner_ndim, |dim| {
            inner_range_full[dim].end.div_ceil(bulk_shape[dim])
        });
        debug_assert!(
            (0..inner_ndim)
                .all(|d| self.is_reduced[d] || bulk_grid_end[d] - bulk_grid_begin[d] <= 1),
            "non-reduced dim must produce at most one bulk-block",
        );
        let mut bulk_iter = NdIter::new_with_begin(
            S::Dimension::from_slice(&bulk_grid_begin).unwrap(),
            S::Dimension::from_slice(&bulk_grid_end).unwrap(),
            NdIterExtBlockOffsetSize::new(
                &dim_arr(inner_ndim, |dim| inner_range_full[dim].start),
                &dim_arr(inner_ndim, |dim| inner_range_full[dim].end),
                &bulk_shape,
            ),
        );

        let state_in_out_buf = size_of::<K::State>() == size_of::<K::Output>()
            && align_of::<K::State>() <= align_of::<K::Output>();
        let mut state_buf;
        // CAREFUL: state_buf and out_ptr may alias
        let (state_buf, out_ptr) = if state_in_out_buf {
            // use the output buffer as the state buffer
            let out_ptr = buf.as_mut_ptr();
            (buf, out_ptr)
        } else {
            state_buf = context.tmp_buf_typed::<MaybeUninit<K::State>>(out_nitems);
            (state_buf.as_mut_slice(), buf.as_mut_ptr())
        };
        let state_buf = unsafe { cast_slice_mut::<_, MaybeUninit<K::State>>(state_buf) };
        let state_strides = default_strides(&out_shape, size_of::<MaybeUninit<K::State>>() as u64);
        let mut state_initialized = false;

        let mut items_buf = context.tmp_buf(0, Alignment::of::<S::Item>());
        let mut base_item_idx = 0;
        while let Some((bulk_idx, (bulk_inner_offset, bulk_size))) = bulk_iter.next() {
            // The bulk's absolute element range, used as the tile iterator's universe.
            let bulk_begin = dim_arr(inner_ndim, |dim| {
                bulk_idx[dim] * bulk_shape[dim] + bulk_inner_offset[dim]
            });
            let bulk_end = dim_arr(inner_ndim, |dim| bulk_begin[dim] + bulk_size[dim]);

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
            let tile_grid_begin = dim_arr(inner_ndim, |dim| bulk_begin[dim] / tile_shape[dim]);
            let tile_grid_end = dim_arr(inner_ndim, |dim| bulk_end[dim].div_ceil(tile_shape[dim]));
            let mut tile_iter = NdIter::new_with_begin(
                S::Dimension::from_slice(&tile_grid_begin).unwrap(),
                S::Dimension::from_slice(&tile_grid_end).unwrap(),
                NdIterExtBlockOffsetSize::new(&bulk_begin, &bulk_end, &tile_shape),
            );
            debug_assert!(
                (0..inner_ndim)
                    .all(|d| { !self.is_reduced[d] || tile_grid_end[d] - tile_grid_begin[d] <= 1 }),
                "reduced dim must produce at most one tile per bulk",
            );

            while let Some((tile_idx, (tile_inner_offset, tile_size))) = tile_iter.next() {
                let tile = dim_arr(inner_ndim, |dim| {
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
                    (tile_size.iter().product::<u64>() * size_of::<S::Item>() as u64) as usize,
                );
                let items_buf = items_buf.as_mut_slice();
                self.array.read_data(&tile, items_buf, context)?;

                // Output-iterator setup. `tile_out_shape` is the tile's output sub-region;
                // `tile_state_base` shifts `state_buf` to its first slot.
                let items_buf_strides = default_strides(tile_size, size_of::<S::Item>() as u64);
                let items_buf_strides_for_out_iter = items_buf_strides
                    .iter()
                    .zip(&self.is_reduced)
                    .filter_map(|(&s, &reduced)| reduced.not().then_some(s))
                    .collect::<DimArray<_>>();
                let tile_out_shape = (0..inner_ndim)
                    .filter(|&d| !self.is_reduced[d])
                    .map(|d| tile_size[d])
                    .collect::<DimArray<_>>();
                let state_offset_bytes = (0..inner_ndim)
                    .filter(|&d| !self.is_reduced[d])
                    .enumerate()
                    .map(|(out_d, d)| {
                        (tile[d].start - inner_range_full[d].start) * state_strides[out_d]
                    })
                    .sum::<u64>();
                let tile_state_base = unsafe {
                    state_buf
                        .as_mut_ptr()
                        .cast::<u8>()
                        .offset(state_offset_bytes as isize)
                };

                let mut out_iter = NdIter::new(
                    D::from_slice(&tile_out_shape).unwrap(),
                    (
                        NdIterExtStridesPtr::new(
                            &items_buf_strides_for_out_iter,
                            items_buf.as_ptr(),
                        ),
                        NdIterExtStridesPtrMut::new(&state_strides, tile_state_base),
                    ),
                );

                // Reduction-axis walk inside `items_buf`. `tile_size[reduced] == bulk_size[reduced]`
                // so this equals `reduction_size`.
                let reduction_shape = dim_arr(inner_ndim, |dim| {
                    if self.is_reduced[dim] {
                        tile_size[dim]
                    } else {
                        1
                    }
                });
                debug_assert_eq!(reduction_shape.iter().product::<u64>(), reduction_size);

                while let Some((_idx, (src_base, state))) = out_iter.next() {
                    let reduction_iter = NdIter::new(
                        S::Dimension::from_slice(&reduction_shape).unwrap(),
                        NdIterExtStridesPtr::new(&items_buf_strides, src_base),
                    );
                    debug_assert_eq!(reduction_size, reduction_iter.len());
                    let mut reduction_iter = reduction_iter
                        .map(|(_idx, in_ptr)| unsafe { in_ptr.cast::<S::Item>().read() });
                    let mut item_idx = base_item_idx;

                    let state_ref = unsafe { &mut *state.cast::<MaybeUninit<K::State>>() };
                    if !state_initialized {
                        // init state with the first item
                        debug_assert_eq!(item_idx, 0);
                        let first = reduction_iter.next();
                        state_ref.write(self.kernel.init_state(first));
                        if first.is_some() {
                            item_idx += 1;
                        }
                    }
                    // SAFETY: every state was written during the first bulk.
                    let mut state = unsafe { state_ref.assume_init_read() };

                    // Fold the first SIMD block
                    while item_idx < item_idx_end.min(SIMD as u64) {
                        let item = reduction_iter.next();
                        let item = unsafe { item.unwrap_unchecked() };
                        state = self.kernel.update_state(state, item, item_idx);
                        item_idx += 1;
                    }
                    // Fold all SIMD blocks
                    let simd_idx_end = item_idx_end.saturating_sub(SIMD as u64);
                    while item_idx < simd_idx_end {
                        let items: [_; SIMD] = std::array::from_fn(|_| unsafe {
                            reduction_iter.next().unwrap_unchecked()
                        });
                        for (i, item) in items.into_iter().enumerate() {
                            state = self.kernel.update_state(state, item, item_idx + i as u64);
                        }
                        item_idx += SIMD as u64;
                    }
                    // Fold the tail
                    while item_idx < item_idx_end {
                        let item = reduction_iter.next();
                        let item = unsafe { item.unwrap_unchecked() };
                        state = self.kernel.update_state(state, item, item_idx);
                        item_idx += 1;
                    }
                    debug_assert!(reduction_iter.next().is_none());
                    state_ref.write(state);
                }
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
        // drop doesn't do anything, but indicate we don't hold a mut ref to the state buf, as
        // out_ptr may be an alias to it
        #[allow(dropping_references)]
        drop(state_buf);
        let out_strides = default_strides(&out_shape, size_of::<K::Output>() as u64);
        if state_initialized {
            let mut out_iter = NdIter::new(
                D::from_slice(&out_shape).unwrap(),
                (
                    // CAREFUL: state_ptr and out_ptr may alias
                    NdIterExtStridesPtrMut::new(&state_strides, state_ptr.cast()),
                    NdIterExtStridesPtrMut::new(&out_strides, out_ptr),
                ),
            );
            while let Some((_idx, (state, out_ptr))) = out_iter.next() {
                // CAREFUL: state and out_ptr may alias
                let res = {
                    let state = unsafe { &*state.cast::<MaybeUninit<K::State>>() };
                    let state = unsafe { state.assume_init_read() };
                    self.kernel.finalize_state(state, reduction_size_overall)
                };
                unsafe { out_ptr.cast::<K::Output>().write(res) };
            }
        } else {
            // Empty reduction: write the empty-stream result to every output.
            let mut out_iter = NdIter::new(
                D::from_slice(&out_shape).unwrap(),
                NdIterExtStridesPtrMut::new(&out_strides, out_ptr),
            );
            debug_assert_eq!(reduction_size_overall, 0);
            while let Some((_idx, out_ptr)) = out_iter.next() {
                let state = self.kernel.init_state(None);
                let res = self.kernel.finalize_state(state, 0);
                unsafe { out_ptr.cast::<K::Output>().write(res) };
            }
        }

        Ok(())
    }

    /// Choose the per-tile inner read shape.
    ///
    /// Tiles seed from the source array's storage-block hint (the natural unit of work
    /// for the inner reader) and scale up greedily - reduced dims first, rightmost first
    /// within each group - until the byte volume reaches the source's preferred read size
    /// hint (with a cache floor for pathologically small hints). Every per-dim choice is
    /// clamped to `full_read_shape[d].len()` so the budget isn't spent on dims that the
    /// requested range can't actually consume.
    ///
    /// Final fixup: any dim whose tile already covers the full requested range is snapped
    /// to `inner_shape[d]`. Tiles are aligned to absolute multiples of `tile_shape[d]`, so
    /// a tile sized exactly to the range can still get split when `full_read_shape[d].start`
    /// is not a multiple of the tile shape; using the full source extent guarantees the
    /// range fits in one tile along that dim.
    fn choose_tile_shape(&self, full_read_shape: &[Range<u64>]) -> DimArray<u64> {
        let ndim = full_read_shape.len();
        let blocks_layout = &self.array.spec().blocks_layout;

        // Target byte volume per inner read; floor to keep tiny hints from collapsing tiles.
        let target_nitems =
            (blocks_layout.preferred_read_size_hint / size_of::<S::Item>() as u64).max(1);

        // Seed from the source storage block hint, clamped to the requested range.
        let mut tile_shape = dim_arr(ndim, |dim| {
            (blocks_layout.block_shape_hint[dim] as u64)
                .min(full_read_shape[dim].end - full_read_shape[dim].start)
                .max(1)
        });

        // Greedy scale-up: reduced dims first (rightmost first), then non-reduced
        // (rightmost first). The reduction kernel walks the reduced axes inside one tile,
        // so giving them first claim on the budget produces fewer outer iterations. Each
        // dim grows by an integer multiplier of its seed (the storage block hint), so the
        // tile stays a multiple of the source's natural block size along that dim.
        let scale_order = (0..ndim)
            .rev()
            .filter(|&dim| self.is_reduced[dim])
            .chain((0..ndim).rev().filter(|&dim| !self.is_reduced[dim]));
        let mut current_volume = tile_shape.iter().product::<u64>();
        for dim in scale_order {
            let dim_len = full_read_shape[dim].end - full_read_shape[dim].start;
            let mult_by_budget = target_nitems / current_volume.max(1);
            let mult_by_range = dim_len.div_ceil(tile_shape[dim]);
            let multiplier = mult_by_budget.min(mult_by_range).max(1);
            let new_tile_size = (tile_shape[dim] * multiplier).min(dim_len);
            current_volume = current_volume / tile_shape[dim] * new_tile_size;
            tile_shape[dim] = new_tile_size;
        }

        // Snap any dim already covering its full requested range to `inner_shape[d]` so
        // the tile boundary doesn't accidentally split the range along an unaligned start.
        let inner_shape = self.array.shape();
        for dim in 0..ndim {
            if tile_shape[dim] == (full_read_shape[dim].end - full_read_shape[dim].start) {
                tile_shape[dim] = inner_shape[dim].max(1);
            }
        }

        tile_shape
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
        pub struct $Op<S, D>(crate::ops::reduction::ReductionOp<S, $Kernel, D>);

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
    #[allow(unused_imports)]
    use crate::scalar::{f16, Complex};

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
    /// For **float** types, `NaN` values are ignored: if at least one non-`NaN` value
    /// is present, the result is the maximum of the non-`NaN` values. If all elements
    /// are `NaN`, the result is `NaN`. This deviates from the element-wise [`Maximum`](crate::ops::Maximum)
    /// op (which propagates `NaN`) but matches `numpy.max`.
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Maximum<Output = S::Item> + crate::dtype::Dtyped,
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
    /// For **float** types, `NaN` values are ignored: if at least one non-`NaN` value
    /// is present, the result is the minimum of the non-`NaN` values. If all elements
    /// are `NaN`, the result is `NaN`. This deviates from the element-wise [`Minimum`](crate::ops::Minimum)
    /// op (which propagates `NaN`) but matches `numpy.min`.
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Minimum<Output = S::Item> + crate::dtype::Dtyped,
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
    /// For **float** types, `NaN` values are treated as less than any non-`NaN` value,
    /// so they are never selected unless all elements are `NaN`.
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
        S: crate::storage::ArrayStorageTyped,
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
    /// For **float** types, `NaN` values are treated as greater than any non-`NaN` value,
    /// so they are never selected unless all elements are `NaN`.
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
        S: crate::storage::ArrayStorageTyped,
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Sum<Output: crate::dtype::Dtyped>,
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Product<Output: crate::dtype::Dtyped>,
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Mean<Output: crate::dtype::Dtyped>,
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Variance<Output: crate::dtype::Dtyped>,
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
        S: crate::storage::ArrayStorageTyped,
        S::Item: crate::scalar::Variance<Output: num_traits::Float + crate::dtype::Dtyped>,
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
        S: crate::storage::ArrayStorageTyped<Item = bool>,
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
        S: crate::storage::ArrayStorageTyped<Item = bool>,
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
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Maximum<Output = S::Item> + crate::dtype::Dtyped,
        }
    );
    define_array_reduction_method!(
        min: Min,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Minimum<Output = S::Item> + crate::dtype::Dtyped,
        }
    );
    define_array_reduction_method!(
        argmax: ArgMax,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: PartialOrd,
        },
        single_axis
    );
    define_array_reduction_method!(
        argmin: ArgMin,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: PartialOrd,
        },
        single_axis
    );
    define_array_reduction_method!(
        sum: Sum,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Sum<Output: crate::dtype::Dtyped>,
        }
    );
    define_array_reduction_method!(
        product: Product,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Product<Output: crate::dtype::Dtyped>,
        }
    );
    define_array_reduction_method!(
        mean: Mean,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Mean<Output: crate::dtype::Dtyped>,
        }
    );
    define_array_reduction_method!(
        var: Variance,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Variance<Output: crate::dtype::Dtyped>,
        },
        extra_args = (ddof: f64)
    );
    define_array_reduction_method!(
        std: StandardDeviation,
        where {
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::Variance<Output: num_traits::Float + crate::dtype::Dtyped>,
        },
        extra_args = (ddof: f64)
    );
    define_array_reduction_method!(
        all: All,
        where {
            S: crate::storage::ArrayStorageTyped<Item = bool>,
        }
    );
    define_array_reduction_method!(
        any: Any,
        where {
            S: crate::storage::ArrayStorageTyped<Item = bool>,
        }
    );
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
                        crate::util::assert_array_matches(&result, &expected);
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
                        crate::util::assert_array_matches(&result, &expected);
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
                        crate::util::assert_array_matches(&result, &expected);
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
