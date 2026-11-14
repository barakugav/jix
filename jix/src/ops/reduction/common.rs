use std::mem::MaybeUninit;
use std::ops::Not;

use fearless_simd::{Level as SimdLevel, Simd, SimdBase};

use crate::ops::prelude::*;
use crate::{array2_map_inline, const_max, const_min, prev_power_of_two, Dim};

pub(crate) struct ReductionOp<S: ArrayStorage, K> {
    kernel: K,

    pub(crate) array: S,
    is_reduced: DimArray<bool>,

    shape: DimDyn,
    cache_id: ArrayCacheId,
    cache_scope: CacheScope,
    spec: ArraySpecDynamic,
}
pub(crate) trait ReductionOpKernel<T> {
    type Output;

    type State;

    /// Build the initial accumulator. `init_item` is the first stream element together with
    /// its true global stream position (0-based), or `None` when the kernel was invoked on an
    /// empty reduction. Kernels with [`SUPPORTS_EMPTY`](Self::SUPPORTS_EMPTY) set to
    /// `false` may unwrap `init_item` - the caller guarantees it is `Some` for those kernels.
    ///
    /// The bundled index is not always `0`: a lane accumulator is seeded from an interior
    /// item, and a cell can be re-seeded partway through the stream when the reduced dim
    /// spans several bulks. It is `Some` exactly for the kernels that
    /// [`NEEDS_INDICES`](Self::NEEDS_INDICES) (argmax/argmin), which record it.
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State;

    #[inline(always)]
    fn init_state_bulk<const N: usize, S: Simd>(
        &self,
        items: [T; N],
        indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        let _ = simd;
        items.map_enumerate(
            #[inline(always)]
            |i, item| self.init_state(Some((item, indices.map(|idx| idx[i])))),
        )
    }

    /// Fold `item` (at true global stream position `idx`, `Some` exactly for the kernels that
    /// [`NEEDS_INDICES`](Self::NEEDS_INDICES)) into `state`.
    fn update_state(&self, state: Self::State, item: T, idx: Option<u64>) -> Self::State;

    /// [`update_state`](Self::update_state) on each lane: `items[i]` (at `indices[i]`) folded
    /// into `states[i]`, vectorized with `simd` where the kernel has SIMD support.
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        let _ = simd;
        array2_map_inline(
            states,
            items,
            #[inline(always)]
            |i, state, item| self.update_state(state, item, indices.map(|idx| idx[i])),
        )
    }

    /// Combine two partial accumulators folded over DISJOINT subsets of the stream.
    ///
    /// MUST be associative and commutative with respect to the folded result (up to float
    /// accuracy): it collapses the interleaved lane accumulators of a single cell, and continues
    /// a cell's accumulator across the bulks of a large reduced dim. The two subsets never
    /// overlap and together cover the folded elements exactly once.
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State;

    /// Produce the final result. `nitems` is the total number of stream elements that
    /// were folded into `state`, so `nitems == 0` exactly when the reduction was empty.
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output;

    /// Whether the kernel can produce a result for an empty stream - see
    /// [`init_state`](Self::init_state), which receives `None` exactly then.
    const SUPPORTS_EMPTY: bool;

    /// Whether [`finalize_state`](Self::finalize_state) does any actual work.
    ///
    /// `false` lets the caller skip the whole finalization pass and treat the state bytes as the
    /// output bytes.
    ///
    /// # Safety
    ///
    /// Setting this to `false` is an unsafe promise. `false` asserts that
    /// `finalize_state(state, nitems)` is the identity for every `state` and `nitems`:
    /// [`Output`](Self::Output) must have the same layout as [`State`](Self::State), and the
    /// returned value must be bit-identical to `state`. The caller relies on this to reinterpret
    /// one buffer as the other without touching it.
    const NEEDS_FINALIZE: bool = true;

    /// Whether a balanced merge tree is worth building for this kernel.
    ///
    /// `true` asks the fold to split a long run in half recursively, so that
    /// [`merge_states`](Self::merge_states) combines partials of similar size and the result is
    /// reached through a `log(n)`-deep tree rather than one linear chain. That is what floating
    /// point accumulation wants. `false` says the kernel's result does not depend on how the
    /// stream is grouped - exact integer or comparison folds - so the whole run is folded in one
    /// pass and the tree is skipped entirely.
    const PREFER_TREE_MERGE: bool;

    /// The lane accumulators of the one-cell fold over contiguous items, if fixed: by default
    /// they are sized by the SIMD level's vectors.
    const ONE_CELL_LANES: Option<usize> = None;

    /// Whether the kernel reads the items' stream positions: the indices passed to it are `Some`
    /// exactly then, and the caller does not compute them otherwise.
    const NEEDS_INDICES: bool = true;
}

impl<S: ArrayStorage, K> ReductionOp<S, K> {
    pub(crate) fn new(array: S, kernel: K, dims: impl DimsArg) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: ReductionOpKernel<S::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        let input_ndim = array.shape().len();
        let mut is_reduced = dim_arr(input_ndim, |_| false);
        for i in 0..dims.len() {
            let d = dims.get(i);
            ensure!(
                d < input_ndim,
                InvalidArgument,
                "dim {d} out of bounds for array of ndim {input_ndim}"
            );

            ensure!(!is_reduced[d], InvalidArgument, "duplicate dim {d}");
            is_reduced[d] = true;
        }
        debug_assert!(
            !K::NEEDS_INDICES || dims.len() <= 1,
            "index-tracking reductions support a single reduced dim"
        );

        if !K::SUPPORTS_EMPTY
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
        let shape = DimDyn::from_slice(&shape);
        let item_bytes = (K::Output::DTYPE.itemsize() as usize).max(size_of::<K::State>());
        check_shape_overflow(shape.as_slice(), item_bytes as _)?;

        let inner_spec = array.spec();
        let reduced_prod = (0..input_ndim)
            .filter(|&dim| is_reduced[dim])
            .map(|dim| array.shape()[dim])
            .product::<u64>();
        let element_cost = (inner_spec.element_cost() as f64 * (reduced_prod as f64 + 4.0)) as f32;
        let dim_orig2new = |d: usize| (d - (0..d).filter(|&j| is_reduced[j]).count()) as DimIdx;
        let read_shape_scale_weight = inner_spec
            .read_shape_scale_weight()
            .iter()
            .enumerate()
            .filter_map(|(d, &w)| (!is_reduced[d]).then_some(w))
            .collect();
        let read_layout_order = inner_spec
            .read_layout_order()
            .iter()
            .filter(|&&d| !is_reduced[d as usize])
            .map(|&d| dim_orig2new(d as usize))
            .collect();
        let mut spec = ArraySpecDynamic::new(
            dim_new2orig
                .iter()
                .map(|&dim| inner_spec.block_shape()[dim])
                .collect(),
            inner_spec
                .block_shape_fixed_dims()
                .into_iter()
                .enumerate()
                .filter_map(|(dim, c)| is_reduced[dim].not().then_some(c))
                .collect(),
            element_cost,
            read_shape_scale_weight,
            read_layout_order,
        );

        let cache_id = ArrayCacheId::new();
        spec.add_cache_ids_from([&inner_spec]);
        let cache_scope = spec.with_cache_scope();
        spec.add_cache_id(cache_id);

        Ok(Self {
            kernel,
            shape,
            cache_id,
            cache_scope,
            spec,
            array,
            is_reduced,
        })
    }
}

impl<S, K> crate::storage::ArrayStorageImpl for ReductionOp<S, K>
where
    S: ArrayStorageTyped,
    K: ReductionOpKernel<S::Item, Output: Dtyped>,
{
}
impl<S, K> ArrayStorage for ReductionOp<S, K>
where
    S: ArrayStorageTyped,
    K: ReductionOpKernel<S::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        context.read_cached(self.cache_id, index, self.dtype(), out, |out| {
            let _cache_scope = self.cache_scope.enter(context);
            read_data_dispatch(
                &self.array,
                self.shape(),
                self.dtype(),
                size_of::<K::State>(),
                Alignment::of::<K::State>(),
                &self.is_reduced,
                K::NEEDS_FINALIZE,
                &|args| reduce_tile::<S::Item, K>(&self.kernel, args),
                &|args| finalize_states_inner_loop::<S::Item, K>(&self.kernel, args),
                &|out, len, out_stride| {
                    finalize_states_empty_inner_loop(&self.kernel, out, len, out_stride)
                },
                index,
                context,
                out,
                self.spec().read_layout_order(),
            )
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<K::Output>()
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

    crate::ops::impl_element_type_change_default!();
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn read_data_dispatch<'a>(
    inner_array: &dyn ArrayStorage,
    outer_shape: &[u64],
    output_dtype: &Dtype,
    kernel_state_sizeof: usize,
    kernel_state_alignof: Alignment,
    is_reduced: &[bool],
    needs_finalize: bool,
    reduce_tile_fn: &dyn Fn(ReduceTileArgs<'_>),
    finalize_state_fn: &dyn Fn(Finalize1DArgs),
    finalize_state_empty_fn: &dyn Fn(*mut u8, usize, usize),
    index: &Index<'_>,
    context: &'a ReadContext,
    out: Option<&'a mut StridedBuf<'_>>,
    read_layout_order: &[DimIdx],
) -> Result<StridedBuf<'a>> {
    let read_data_fn = match is_reduced.len() {
        1 => read_data_impl::<Dim<1>>,
        2 => read_data_impl::<Dim<2>>,
        3 => read_data_impl::<Dim<3>>,
        4 => read_data_impl::<Dim<4>>,
        _ => read_data_impl::<DimDyn>,
    };
    read_data_fn(
        inner_array,
        outer_shape,
        output_dtype,
        kernel_state_sizeof,
        kernel_state_alignof,
        is_reduced,
        needs_finalize,
        reduce_tile_fn,
        finalize_state_fn,
        finalize_state_empty_fn,
        index,
        context,
        out,
        read_layout_order,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn read_data_impl<'a, InnerD>(
    inner_array: &dyn ArrayStorage,
    outer_shape: &[u64],
    output_dtype: &Dtype,
    kernel_state_sizeof: usize,
    kernel_state_alignof: Alignment,
    is_reduced: &[bool],
    needs_finalize: bool,
    reduce_tile_fn: &dyn Fn(ReduceTileArgs<'_>),
    finalize_state_fn: &dyn Fn(Finalize1DArgs),
    finalize_state_empty_fn: &dyn Fn(*mut u8, usize, usize),
    index: &Index<'_>,
    context: &'a ReadContext,
    out: Option<&'a mut StridedBuf<'_>>,
    read_layout_order: &[DimIdx],
) -> Result<StridedBuf<'a>>
where
    InnerD: Dimension,
{
    // This method accept some &dyn fns to avoid monomorphizing the whole method for every combination
    // of kernel, dimension, dtype, and backing storage.
    check_out_buf(
        out.as_deref(),
        outer_shape,
        index,
        output_dtype.itemsize() as usize,
    )?;
    let out_shape_usize = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
    let mut out = materialize_out_buf(
        out,
        context,
        out_shape_usize.as_ref(),
        output_dtype,
        read_layout_order,
    );
    let (out_buf, out_strides) = out.data_mut();

    // Streams the reduction over a two-level chunking of the inner array so peak scratch
    // memory stays bounded *and* each downstream `self.array.read_data` call is sized to
    // roughly fit the caller's request in cache.
    //
    // # What it computes
    //
    // For every output position `O` covered by `index`, evaluates
    //
    //   stream(O) = iterate inner elements with non-reduced coords matching O,
    //               in row-major order over the reduced dims
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
    //     reduced per bulk). Within a bulk the tile iterator sweeps the reduced dims in
    //     row-major order, so a tile is `bulk-non-reduced * tile-reduced`. Each
    //     `(bulk, tile)` pair produces *one* `self.array.read_data` call, sized to land
    //     within the source's `read_size` `(min, max)` window.
    //
    // # Per-tile processing
    //
    // ```text
    // full_reduction_size = product(inner_range_full[d] len for d in reduced dims)
    // for each bulk:                      // a disjoint block of outputs
    //   bulk_base_item_idx = 0            // reduced-stream position within this bulk
    //   bulk_initialized   = false
    //   for each tile in this bulk:       // walks the reduced dims; one non-reduced tile
    //     reduction_size = product(tile_size[d] for d in reduced dims)
    //     items_buf <- self.array.read_data(tile = bulk-non-reduced * tile-reduced)
    //     for each output position O in this bulk's output block:
    //       fold the tile's `reduction_size` items at O's offset into state_buf[O],
    //       at stream indices [bulk_base_item_idx, bulk_base_item_idx + reduction_size).
    //       The first tile of the bulk seeds the slot (K::init_state); later tiles merge.
    //     bulk_base_item_idx += reduction_size
    //     bulk_initialized = true
    // ```

    let out_nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
    if out_nitems == 0 {
        return Ok(out);
    }

    let inner_shape = inner_array.shape();
    let inner_ndim = inner_shape.len();
    let item_dtype = inner_array.dtype();

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

    let out_shape = dim_arr(index.len(), |dim| index[dim].end - index[dim].start);

    let tile_max_shape = InnerD::vec(inner_ndim, |dim| {
        inner_range_full[dim].end - inner_range_full[dim].start
    });
    let tile_shape = inner_array.spec().read_shape_heuristic::<InnerD>(
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

    // Row-major strides over the reduced sub-shape, so an element's position in the reduced
    // stream is the dot product of its coordinates with these. Zero on the non-reduced dims, which
    // drops them from that dot product
    let reduced_shape_logical_strides = {
        let mut reduced_shape_logical_strides = InnerD::vec(inner_ndim, |_| 0u64);
        let mut logical_stride = 1;
        for d in (0..inner_ndim).rev() {
            if is_reduced[d] {
                reduced_shape_logical_strides[d] = logical_stride;
                logical_stride *= inner_range_full[d].end - inner_range_full[d].start;
            }
        }
        debug_assert_eq!(logical_stride, full_reduction_size);
        reduced_shape_logical_strides
    };

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
        (state_buf, DimArray::from_slice(out_strides).unwrap())
    } else {
        check_region_size(index, kernel_state_sizeof)?; // states may be bigger than the output
        let state_buf_len = out_nitems * kernel_state_sizeof;
        tmp_state_buf = context.allocate_buf(state_buf_len, kernel_state_alignof);
        (
            &mut tmp_state_buf.as_mut_slice()[..state_buf_len],
            default_strides(&out_shape, kernel_state_sizeof),
        )
    };
    debug_assert!(state_strides
        .as_ref()
        .iter()
        .all(|&s| s.is_multiple_of(kernel_state_alignof.as_usize())));
    // The state strides lifted to the inner array's dims: zero on a reduced dim, since a reduced
    // dim does not index the state.
    let state_strides_inner = {
        let mut state_strides = state_strides.as_ref().iter().copied();
        InnerD::vec(inner_ndim, |d| {
            if is_reduced[d] {
                0
            } else {
                state_strides.next().unwrap()
            }
        })
    };

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
                InnerD::vec(inner_ndim, |d| tile_shape[d]),
            )
            .build();

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

            let items = inner_array.read_data(
                &unsafe { Index::new_unchecked(tile.as_ref()) },
                context,
                None,
            )?;

            let state_offset = (0..inner_ndim)
                .map(|d| {
                    ((tile[d].start - inner_range_full[d].start) as usize) * state_strides_inner[d]
                })
                .sum::<usize>();
            let tile_base_reduced_idx = (0..inner_ndim)
                .map(|d| {
                    (tile[d].start - inner_range_full[d].start) * reduced_shape_logical_strides[d]
                })
                .sum::<u64>();

            let tile_shape = InnerD::vec(inner_ndim, |d| tile_size[d] as usize);
            let tile_state_base =
                unsafe { PtrMutNoalias::from_slice(state_buf.get_unchecked_mut(state_offset..)) };

            reduce_tile_fn(ReduceTileArgs {
                tile_shape: tile_shape.as_ref(),
                items: &items,
                states_buf: tile_state_base,
                state_strides: state_strides_inner.as_ref(),
                tile_base_reduced_idx,
                reduced_shape_logical_strides: reduced_shape_logical_strides.as_ref(),
            });
        }
    }

    // CAREFUL: state_buf and out_ptr may alias
    let state_ptr = state_buf.as_mut_ptr();
    // From here on the state/output buffers are touched only through `state_ptr` and
    // `out_ptr`. Dont use `state_buf`.
    finalize_states(
        FinalizeStateArgs {
            out_shape: out_shape_usize.as_ref(),
            state_buf: state_ptr,
            state_buf_strides: state_strides.as_ref(),
            out_buf: out_ptr,
            out_buf_strides: out_strides,
            full_reduction_size,
        },
        (kernel_state_sizeof as Itemsize, kernel_state_alignof),
        (output_dtype.itemsize(), output_dtype.alignment()),
        needs_finalize,
        finalize_state_fn,
        finalize_state_empty_fn,
    );
    Ok(out)
}

struct ReduceTileArgs<'a> {
    tile_shape: &'a [usize],
    items: &'a StridedBuf<'a>,
    states_buf: PtrMutNoalias<'a, u8>,
    state_strides: &'a [usize],

    tile_base_reduced_idx: u64,
    reduced_shape_logical_strides: &'a [u64],
}

fn reduce_tile<T, K>(kernel: &K, args: ReduceTileArgs<'_>)
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    reduce_tile_impl(
        args,
        (size_of::<T>() as Itemsize, Alignment::of::<T>()),
        (
            size_of::<K::State>() as Itemsize,
            Alignment::of::<K::State>(),
        ),
        &|args: FoldInnerLoopArgs<'_>, flags: FoldInnerLoopFlags| {
            let init = args.base_item_idx == 0;
            let fold_fn = match (flags.one_cell, flags.contiguous, init) {
                (true, true, _) => fold_one_cell::<T, K, true>,
                (true, false, _) => fold_one_cell::<T, K, false>,
                (false, true, true) => fold_multi_cell::<T, K, true, true>,
                (false, true, false) => fold_multi_cell::<T, K, true, false>,
                (false, false, true) => fold_multi_cell::<T, K, false, true>,
                (false, false, false) => fold_multi_cell::<T, K, false, false>,
            };
            fold_fn(kernel, args)
        },
    );
}

type FoldInnerLoops<'a> = dyn Fn(FoldInnerLoopArgs<'_>, FoldInnerLoopFlags) + 'a;

#[derive(Clone, Copy)]
struct FoldInnerLoopFlags {
    one_cell: bool,
    contiguous: bool,
}

#[inline(never)]
fn reduce_tile_impl(
    args: ReduceTileArgs<'_>,
    item_layout: (Itemsize, Alignment),
    state_layout: (Itemsize, Alignment),
    inner_loop: &FoldInnerLoops<'_>,
) {
    let ReduceTileArgs {
        tile_shape,
        items,
        mut states_buf,
        state_strides,
        tile_base_reduced_idx,
        reduced_shape_logical_strides,
    } = args;

    let (items_buf, items_strides) = items.data();

    let [tile_shape_u64, items_strides_u64, state_strides_u64] =
        [tile_shape, items_strides, state_strides]
            .map(|a| a.iter().map(|s| *s as u64).collect::<DimArray<_>>());
    let iter = NdIterUnordered::new_with(
        &tile_shape_u64,
        [
            &items_strides_u64,
            &state_strides_u64,
            reduced_shape_logical_strides,
        ],
        [
            item_layout,
            state_layout,
            (1, Alignment::of::<u8>()), // arbitrary
        ],
        // reduced_shape_logical_strides are logical strides, not memory strides, ignore them
        // when choosing an iteration dim order
        [true, true, false],
    );

    let inner_len = iter.inner_len();
    if inner_len == 0 {
        return; // empty tile
    }
    let [_items_inner_stride, state_inner_stride, reduced_inner_stride] = iter.inner_strides();
    let [items_contiguous, state_contiguous, _] = *iter.is_contiguous();
    let [items_strides_aligned, state_strides_aligned, _] = *iter.is_strides_aligned();
    let items_aligned = !REQUIRE_ALIGNED
        || (items_strides_aligned
            && (items_buf.as_ptr() as usize).is_multiple_of(item_layout.1.as_usize()));
    // The state buffer is either a fresh pooled allocation made at the state's alignment, or the
    // caller's output buffer - which `state_in_out_buf` only accepts when it is aligned
    debug_assert!(
        state_strides_aligned
            && (states_buf.as_ptr() as usize).is_multiple_of(state_layout.1.as_usize()),
        "the state buffer must be aligned for the kernel state",
    );

    // The inner loop is either over (1) items for a single output cell, or (2) output cells, one
    // item each.
    let flags = if state_inner_stride == 0 {
        FoldInnerLoopFlags {
            one_cell: true,
            contiguous: items_contiguous && items_aligned,
        }
    } else {
        debug_assert!(
            reduced_inner_stride == 0 || inner_len == 1,
            "a non-reduced innermost dim has no reduced-shape stride",
        );
        FoldInnerLoopFlags {
            one_cell: false,
            contiguous: items_contiguous && items_aligned && state_contiguous,
        }
    };
    let items_buf = PtrNoalias::<u8>::from_slice(items_buf);
    iter.foreach_inner_1d(move |offsets, inner_len, strides| {
        let [item_offset, state_offset, base_item_idx] = offsets;
        let [items_stride, state_stride, idx_stride] = strides;

        let items_buf = unsafe { items_buf.bytes_offset(item_offset as usize) };
        let states_buf = unsafe { states_buf.bytes_offset(state_offset as usize) };

        let base_item_idx = tile_base_reduced_idx + base_item_idx;

        inner_loop(
            FoldInnerLoopArgs {
                inner_len: inner_len as usize,
                items: items_buf,
                states_buf,
                items_stride: items_stride as usize,
                state_stride: state_stride as usize,
                idx_stride,
                base_item_idx,
            },
            flags,
        );
    });
}

// Number of lanes, unless SIMD is used (contiguous folds)
const ONE_CELL_LANES: usize = 16;

// Runs longer than this are split recursively in the pairwise tee fold.
const PAIRWISE_SPLIT_THRESHOLD: usize = 2048;

struct FoldInnerLoopArgs<'a> {
    inner_len: usize,
    items: PtrNoalias<'a, u8>,
    states_buf: PtrMutNoalias<'a, u8>,
    items_stride: usize,
    state_stride: usize,
    idx_stride: u64,
    base_item_idx: u64,
}
fn fold_one_cell<T, K, const CONTIGUOUS: bool>(kernel: &K, args: FoldInnerLoopArgs<'_>)
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    let FoldInnerLoopArgs {
        inner_len,
        items,
        states_buf,
        items_stride,
        state_stride,
        idx_stride,
        base_item_idx,
    } = args;

    debug_assert_eq!(state_stride, 0);
    debug_assert!(inner_len >= 1);

    let ctx = FoldOneCellCtx {
        kernel,
        items: items.cast::<T>(),
        items_stride,
        idx_stride,
        base_item_idx,
    };

    let mut state = if CONTIGUOUS {
        fearless_simd::dispatch!(SimdLevel::new(), simd => {
            fold_one_cell_per_simd::<T, K, CONTIGUOUS, _>(&ctx, inner_len, simd)
        })
    } else {
        crate::scalar::simd::dispatch_baseline!(simd => {
            fold_one_cell_per_simd::<T, K, CONTIGUOUS, _>(&ctx, inner_len, simd)
        })
    };
    let state_ref = unsafe { &mut *states_buf.as_mut_ptr().cast::<MaybeUninit<K::State>>() };
    if base_item_idx > 0 {
        let prev = unsafe { state_ref.assume_init_read() };
        state = kernel.merge_states(prev, state);
    }
    state_ref.write(state);
}

#[inline(always)]
fn fold_one_cell_per_simd<T, K, const CONTIGUOUS: bool, S: Simd>(
    ctx: &FoldOneCellCtx<'_, T, K>,
    len: usize,
    simd: S,
) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    let lanes = const {
        let lanes = if !CONTIGUOUS {
            ONE_CELL_LANES
        } else if let Some(lanes) = K::ONE_CELL_LANES {
            lanes
        } else {
            let per_vector = prev_power_of_two(S::u8s::LEN / size_of::<K::State>());
            let lanes = 8 * per_vector;
            const_max(const_min(lanes, PAIRWISE_SPLIT_THRESHOLD / 2), 8)
        };
        let lane_bytes = const_max(const_max(size_of::<T>(), size_of::<K::State>()), 1);
        let max_lanes = prev_power_of_two(4096 / lane_bytes);
        let lanes = const_min(lanes, max_lanes);
        assert!(lanes.is_power_of_two() && lanes <= 512);
        lanes
    };

    macro_rules! dispatch_lanes {
        ($($lanes:literal)*) => {
            match lanes {
                $($lanes => fold_one_cell_impl::<T, K, $lanes, CONTIGUOUS, S>(ctx, len, simd),)*
                _ => unreachable!(),
            }
        };
    }
    dispatch_lanes!(1 2 4 8 16 32 64 128 256 512)
}

#[inline(always)]
fn fold_one_cell_impl<T, K, const LANES: usize, const CONTIGUOUS: bool, S>(
    ctx: &FoldOneCellCtx<'_, T, K>,
    len: usize,
    simd: S,
) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
    S: Simd,
{
    debug_assert!(len >= 1);

    // Fold the body of the run as a pairwise tree
    let body_len = len - len % LANES;
    let mut state =
        (body_len > 0).then(|| fold_pairwise::<T, K, LANES, CONTIGUOUS, S>(ctx, 0, body_len, simd));

    // Fold the tail sequentially
    let kernel = ctx.kernel;
    for i in body_len..len {
        let item = ctx.read_item::<CONTIGUOUS>(i);
        let idx = K::NEEDS_INDICES.then(|| ctx.item_idx(i));
        state = Some(match state {
            None => kernel.init_state(Some((item, idx))),
            Some(state) => kernel.update_state(state, item, idx),
        });
    }
    state.unwrap()
}

#[inline(never)]
fn fold_pairwise<T, K, const LANES: usize, const CONTIGUOUS: bool, S>(
    ctx: &FoldOneCellCtx<'_, T, K>,
    begin: usize,
    len: usize,
    simd: S,
) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
    S: Simd,
{
    const { assert!(PAIRWISE_SPLIT_THRESHOLD >= 2 * LANES) };
    debug_assert!(len >= LANES && len.is_multiple_of(LANES));

    if len <= PAIRWISE_SPLIT_THRESHOLD || !K::PREFER_TREE_MERGE {
        let vectorize = {
            #[inline(never)]
            move || {
                simd.vectorize(
                    #[inline(always)]
                    || fold_pairwise_leaf::<T, K, LANES, CONTIGUOUS, S>(ctx, begin, len, simd),
                )
            }
        };
        return vectorize();
    }

    let half = len / 2;
    let left_len = half - half % LANES;
    debug_assert!(left_len >= LANES && len - left_len >= LANES);
    let left = fold_pairwise::<T, K, LANES, CONTIGUOUS, S>(ctx, begin, left_len, simd);
    let right =
        fold_pairwise::<T, K, LANES, CONTIGUOUS, S>(ctx, begin + left_len, len - left_len, simd);
    ctx.kernel.merge_states(left, right)
}

#[inline(always)]
fn fold_pairwise_leaf<T, K, const LANES: usize, const CONTIGUOUS: bool, S>(
    ctx: &FoldOneCellCtx<'_, T, K>,
    begin: usize,
    len: usize,
    simd: S,
) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
    S: Simd,
{
    debug_assert!(len >= LANES && len.is_multiple_of(LANES));
    if CONTIGUOUS {
        unsafe { std::hint::assert_unchecked(ctx.items_stride == size_of::<T>()) };
    }
    let kernel = ctx.kernel;

    let mut indices = K::NEEDS_INDICES.then(
        #[inline(always)]
        || {
            array_from_fn_inline(
                #[inline(always)]
                |b| ctx.item_idx(begin + b),
            )
        },
    );
    let idx_step = LANES as u64 * ctx.idx_stride;

    let mut states = kernel.init_state_bulk(
        ctx.read_items_bulk::<LANES, CONTIGUOUS>(begin),
        indices,
        simd,
    );

    let mut i = begin;
    for _ in 1..len / LANES {
        i += LANES;
        let items = ctx.read_items_bulk::<LANES, CONTIGUOUS>(i);
        if let Some(indices) = &mut indices {
            *indices = indices.map_inline(
                #[inline(always)]
                |idx| idx + idx_step,
            );
        }
        states = kernel.update_state_bulk(states, items, indices, simd);
    }
    merge_states::<T, K, LANES>(kernel, states)
}

struct FoldOneCellCtx<'a, T, K> {
    kernel: &'a K,
    items: PtrNoalias<'a, T>,
    items_stride: usize,
    idx_stride: u64,
    base_item_idx: u64,
}
impl<T: Dtyped, K> FoldOneCellCtx<'_, T, K> {
    #[inline(always)]
    fn item_idx(&self, i: usize) -> u64 {
        self.base_item_idx + (i as u64 * self.idx_stride)
    }

    #[inline(always)]
    fn read_item<const CONTIGUOUS: bool>(&self, i: usize) -> T {
        let items = self.items.as_ptr();
        if CONTIGUOUS {
            unsafe { items.add(i).read_maybe_aligned::<REQUIRE_ALIGNED>() }
        } else {
            unsafe { items.byte_add(i * self.items_stride).read_unaligned() }
        }
    }

    #[inline(always)]
    fn read_items_bulk<const LANES: usize, const CONTIGUOUS: bool>(&self, i: usize) -> [T; LANES] {
        let items = self.items.as_ptr();
        if CONTIGUOUS {
            unsafe {
                items
                    .add(i)
                    .cast::<[T; LANES]>()
                    .read_maybe_aligned::<REQUIRE_ALIGNED>()
            }
        } else {
            unsafe {
                array_from_fn_inline(|b| {
                    items.byte_add((i + b) * self.items_stride).read_unaligned()
                })
            }
        }
    }
}

/// Collapse `LANES` lane accumulators into one via a bottom-up pairwise tree (dependency
/// depth `log2(LANES)`). `LANES` must be a power of two.
#[inline(always)]
fn merge_states<T, K, const LANES: usize>(kernel: &K, states: [K::State; LANES]) -> K::State
where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    // TODO: this function is not panic-safe.

    const { assert!(LANES > 0 && LANES.is_power_of_two()) };
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

/// Inner-loop shape for a non-reduced innermost dim: one item folded into each of the run's cells.
fn fold_multi_cell<T, K, const CONTIGUOUS: bool, const INIT: bool>(
    kernel: &K,
    args: FoldInnerLoopArgs<'_>,
) where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    let FoldInnerLoopArgs {
        inner_len: len,
        items,
        states_buf: states,
        items_stride,
        state_stride,
        idx_stride: _,
        base_item_idx,
    } = args;

    let items = items.cast::<T>();
    let states = states.cast::<MaybeUninit<K::State>>();
    let strides = (items_stride, state_stride);

    if const { CONTIGUOUS } {
        fold_multi_cell_contiguous::<T, K, INIT>(kernel, items, states, base_item_idx, len);
    } else {
        crate::scalar::simd::dispatch_baseline!(simd => fold_multi_cell_impl::<T, K, CONTIGUOUS, INIT, _>(
            kernel, items, states, strides, base_item_idx, len, simd
        ));
    }
}

#[inline(never)]
fn fold_multi_cell_contiguous<T, K, const INIT: bool>(
    kernel: &K,
    items: PtrNoalias<'_, T>,
    states: PtrMutNoalias<'_, MaybeUninit<K::State>>,
    base_item_idx: u64,
    len: usize,
) where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    let strides = (size_of::<T>(), size_of::<K::State>());
    fearless_simd::dispatch!(SimdLevel::new(), simd => fold_multi_cell_impl::<T, K, true, INIT, _>(
        kernel, items, states, strides, base_item_idx, len, simd
    ))
}

#[inline(always)]
fn fold_multi_cell_impl<T, K, const CONTIGUOUS: bool, const INIT: bool, S>(
    kernel: &K,
    items: PtrNoalias<'_, T>,
    states: PtrMutNoalias<'_, MaybeUninit<K::State>>,
    (items_stride, state_stride): (usize, usize),
    base_item_idx: u64,
    len: usize,
    _simd: S,
) where
    T: Dtyped,
    K: ReductionOpKernel<T>,
{
    if CONTIGUOUS {
        unsafe { std::hint::assert_unchecked(items_stride == size_of::<T>()) };
        unsafe { std::hint::assert_unchecked(state_stride == size_of::<K::State>()) };
    }

    let items = items.as_ptr();
    let states = states.as_mut_ptr();
    let idx = K::NEEDS_INDICES.then_some(base_item_idx);
    for i in 0..len {
        let item = if CONTIGUOUS {
            unsafe { items.add(i).read_maybe_aligned::<REQUIRE_ALIGNED>() }
        } else {
            unsafe { items.byte_add(i * items_stride).read_unaligned() }
        };
        let state = if CONTIGUOUS {
            unsafe { &mut *states.add(i) }
        } else {
            unsafe { &mut *states.byte_add(i * state_stride) }
        };
        if INIT {
            state.write(kernel.init_state(Some((item, idx))));
        } else {
            let prev = unsafe { state.assume_init_read() };
            state.write(kernel.update_state(prev, item, idx));
        }
    }
}

struct FinalizeStateArgs<'a> {
    out_shape: &'a [usize],
    state_buf: *mut u8,
    state_buf_strides: &'a [usize],
    out_buf: *mut u8,
    out_buf_strides: &'a [usize],
    full_reduction_size: u64,
}
#[inline(never)]
fn finalize_states(
    args: FinalizeStateArgs<'_>,
    state_layout: (Itemsize, Alignment),
    output_layout: (Itemsize, Alignment),
    needs_finalize: bool,
    finalize_run: &dyn Fn(Finalize1DArgs),
    empty_run: &dyn Fn(*mut u8, usize, usize),
) {
    let FinalizeStateArgs {
        out_shape,
        state_buf,
        state_buf_strides,
        out_buf,
        out_buf_strides,
        full_reduction_size,
    } = args;

    if full_reduction_size > 0 {
        let state_in_out_buf = core::ptr::eq(state_buf, out_buf);
        if !needs_finalize && state_in_out_buf {
            // SAFETY: `NEEDS_FINALIZE == false` promises `finalize_state` is the identity and that
            // `K::Output` has `K::State`'s layout. With `state_in_out_buf` the states were folded
            // straight into the output buffer at the output strides, so those bytes already are the
            // final output and there is nothing left to do.
            return;
        }

        // CAREFUL: state_buf and out_buf may alias
        let iter = NdIterUnordered::new(
            out_shape,
            [out_buf_strides, state_buf_strides],
            [output_layout, state_layout],
        );
        iter.foreach_inner_1d(|offsets, len, inner_strides| {
            let [out_offset, state_offset] = offsets;
            let [out_stride, state_stride] = inner_strides;
            finalize_run(Finalize1DArgs {
                state: unsafe { state_buf.add(state_offset) },
                state_stride,
                out: unsafe { out_buf.add(out_offset) },
                out_stride,
                len,
                full_reduction_size,
            });
        });
    } else {
        // Empty reduction: write the empty-stream result to every output.
        let iter = NdIterUnordered::new(out_shape, [out_buf_strides], [output_layout]);
        iter.foreach_inner_1d(|[offset], len, [inner_stride]| {
            empty_run(unsafe { out_buf.add(offset) }, len, inner_stride);
        });
    }
}

struct Finalize1DArgs {
    state: *mut u8,
    state_stride: usize,
    out: *mut u8,
    out_stride: usize,
    len: usize,
    full_reduction_size: u64,
}
fn finalize_states_inner_loop<T, K>(kernel: &K, args: Finalize1DArgs)
where
    K: ReductionOpKernel<T>,
{
    let Finalize1DArgs {
        state,
        state_stride,
        out,
        out_stride,
        len,
        full_reduction_size,
    } = args;

    let contiguous = state_stride == size_of::<K::State>() && out_stride == size_of::<K::Output>();
    if contiguous {
        fearless_simd::dispatch!(SimdLevel::new(), simd => finalize_states_inner_loop_impl::<T, K, true, _>(
            kernel,
            (state, state_stride),
            (out, out_stride),
            full_reduction_size,
            len,
            simd,
        ))
    } else {
        crate::scalar::simd::dispatch_baseline!(simd => finalize_states_inner_loop_impl::<T, K, false, _>(
            kernel,
            (state, state_stride),
            (out, out_stride),
            full_reduction_size,
            len,
            simd,
        )
        );
    }
}

#[inline(always)]
fn finalize_states_inner_loop_impl<T, K, const CONTIGUOUS: bool, S: Simd>(
    kernel: &K,
    (state, state_stride): (*mut u8, usize),
    (out, out_stride): (*mut u8, usize),
    full_reduction_size: u64,
    len: usize,
    _simd: S,
) where
    K: ReductionOpKernel<T>,
{
    if CONTIGUOUS {
        unsafe { std::hint::assert_unchecked(state_stride == size_of::<K::State>()) };
        unsafe { std::hint::assert_unchecked(out_stride == size_of::<K::Output>()) };
    }

    for i in 0..len {
        // CAREFUL: state and out may alias
        let state = unsafe { state.add(i * state_stride).cast::<K::State>() };
        let out = unsafe { out.add(i * out_stride).cast::<K::Output>() };
        let state = unsafe { state.read_unaligned() };
        let res = kernel.finalize_state(state, full_reduction_size);
        unsafe { out.write_unaligned(res) };
    }
}
fn finalize_states_empty_inner_loop<T, K>(kernel: &K, out: *mut u8, len: usize, out_stride: usize)
where
    K: ReductionOpKernel<T>,
{
    for i in 0..len {
        let out = unsafe { out.add(i * out_stride) };
        let state = kernel.init_state(None);
        let res = kernel.finalize_state(state, 0);
        unsafe { out.cast::<K::Output>().write_unaligned(res) };
    }
}

/// Emits the wrapper storage struct (`$Op<S>`), its `ArrayStorage` impl,
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
///     single_dim,                                 // optional; omit for multi-dim
/// );
/// ```
macro_rules! define_reduction_op {
    // single-dim variant
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident $( { $($extra_arg:ident: $extra_ty:ty),+ $(,)? } )?,
        where { $($where_:tt)+ }
        output = $output_ty:ty,
        single_dim $(,)?
    ) => {
        struct $Kernel { $($($extra_arg: $extra_ty),+)? }

        $(#[$meta])*
        pub struct $Op<S>(crate::ops::reduction::common::ReductionOp<S, $Kernel>)
        where
            S: crate::ArrayStorage;

        impl<S> $Op<S>
        where
            $($where_)+
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S, dim: usize $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<Self> {
                let kernel = $Kernel { $($($extra_arg,)+)? };
                Ok(Self(crate::ops::reduction::common::ReductionOp::new(array, kernel, &[dim])?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>, dim: usize $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage(), dim $($(, $extra_arg)+)?).map(crate::Array::from_storage)
            }
        }

        impl<S> crate::storage::ArrayStorageImpl for $Op<S>
        where
            $($where_)+
        {
        }
        impl<S> crate::ArrayStorage for $Op<S>
        where
            $($where_)+
        {
            type ElementType = crate::Ty<$output_ty>;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> ArrayStorageInfo<'_> {
                ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }
            crate::ops::impl_element_type_change_default!();
        }
    };

    // multi-dim variant
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident $( { $($extra_arg:ident: $extra_ty:ty),+ $(,)? } )?,
        where { $($where_:tt)+ }
        output = $output_ty:ty $(,)?
    ) => {
        struct $Kernel { $($($extra_arg: $extra_ty),+)? }

        $(#[$meta])*
        pub struct $Op<S: crate::ArrayStorage>(crate::ops::reduction::common::ReductionOp<S, $Kernel>);

        impl<S> $Op<S>
        where
            $($where_)+
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S, dims: impl crate::ops::DimsArg $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<Self> {
                let kernel = $Kernel { $($($extra_arg,)+)? };
                Ok(Self(crate::ops::reduction::common::ReductionOp::new(array, kernel, dims)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>, dims: impl crate::ops::DimsArg $($(, $extra_arg: $extra_ty)+)?) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage(), dims $($(, $extra_arg)+)?).map(crate::Array::from_storage)
            }
        }

        impl<S> crate::storage::ArrayStorageImpl for $Op<S>
        where
            $($where_)+
        {
        }
        impl<S> crate::ArrayStorage for $Op<S>
        where
            $($where_)+
        {
            type ElementType = crate::Ty<$output_ty>;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> ArrayStorageInfo<'_> {
                ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }

            crate::ops::impl_element_type_change_default!();
        }
    };
}

/// Emits an `Array::$method(...)` helper that forwards to `$Op::new_array(...)`. The full
/// where-clause on `S` (and its `Item`) is supplied verbatim by the caller so each op can
/// pick its own bound (`ArgMax`, `Maximum`, `Sum`, `Item = bool`, ...).
macro_rules! define_array_reduction_method {
    // single-dim variant
    (
        $method:ident: $Op:ident,
        where { $($where_:tt)+ }
        $(, extra_args = ($($extra_arg:ident: $extra_ty:ty),*))?,
        single_dim $(,)?
    ) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self, dim: usize $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S>>
        where $($where_)+
        {
            $Op::new_array(self, dim $($(, $extra_arg)*)?).unwrap()
        }
    };

    // multi-dim variant
    (
        $method:ident: $Op:ident,
        where { $($where_:tt)+ }
        $(, extra_args = ($($extra_arg:ident: $extra_ty:ty),*))?
        $(,)?
    ) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self, dim: impl DimsArg $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S>>
        where
            $($where_)+
        {
            $Op::new_array(self, dim $($(, $extra_arg)*)?).unwrap()
        }
    };
}

pub(crate) use {define_array_reduction_method, define_reduction_op};

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(crate) mod tests {
    use std::rc::Rc;

    use crate::scalar::bool8;
    use crate::storage::Index;
    use crate::ArrayParams;

    use ndarray::{array, ArrayD, Ix0, Ix1};

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

    /// Per-dtype comparison policy for the reduction property tests.
    ///
    /// A reduction reads its input block-by-block, so a floating accumulator reassociates and can
    /// land a few ULP away from a sequential reference fold. Float and complex reductions are
    /// therefore compared with [`ApproxEq`], using a per-dtype tolerance (looser for the narrower
    /// `f32` / `Complex<f32>`, tighter for `f64` / `Complex<f64>`). Integer and `bool8` reductions
    /// are exact and compared bit-for-bit.
    ///
    /// `f16` only ever surfaces here as a `max` / `min` result (its `sum`/`product`/`mean` value
    /// parity is skipped - native `f16` accumulation drifts/overflows too far to pin with a
    /// tolerance, mirroring the Python suite), and `max`/`min` select an input element, so `f16`
    /// is compared exactly like the integers.
    ///
    /// [`ApproxEq`]: crate::scalar::ApproxEq
    pub(crate) trait ReductionCompare:
        crate::dtype::Dtyped + std::fmt::Debug + Clone
    {
        fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>);
    }

    /// Implements [`ReductionCompare`] with exact comparison, see [`crate::util::assert_array_matches`].
    macro_rules! reduction_compare_exact {
        ($($t:ty),* $(,)?) => {$(
            impl ReductionCompare for $t {
                fn assert_matches<S: crate::ArrayStorage>(actual: &Array<S>, expected: &ArrayD<Self>) {
                    crate::util::assert_array_matches(actual, expected);
                }
            }
        )*};
    }
    reduction_compare_exact!(i8, i16, i32, i64, u8, u16, u32, u64, bool8);
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
    pub(crate) fn assert_reduction_matches<S: crate::ArrayStorage, T: ReductionCompare>(
        actual: &Array<S>,
        expected: &ArrayD<T>,
    ) {
        T::assert_matches(actual, expected);
    }

    pub(crate) fn dim_strategy(ndim: usize) -> impl proptest::strategy::Strategy<Value = usize> {
        0..ndim
    }

    pub(crate) fn dims_strategy(ndim: usize) -> proptest::strategy::BoxedStrategy<Vec<usize>> {
        if ndim == 0 {
            return proptest::strategy::Just(vec![]).boxed();
        }
        let dim_strategy = dim_strategy(ndim).prop_map(|dim| vec![dim]);
        let multi_dims_strategy = prop::collection::vec(0..ndim, 1..=ndim).prop_map(|mut dims| {
            dims.sort_unstable();
            dims.dedup();
            dims
        });
        prop::strategy::Union::new_weighted(vec![
            (3, dim_strategy.boxed()),
            (1, multi_dims_strategy.boxed()),
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
    pub(crate) fn array_strategy_for_reduction<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<crate::util::TestArray<T>>, Vec<usize>)>
    {
        let shape = reduction_shape_strategy();
        let array = crate::util::array_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let dims = dims_strategy(nd.ndim());
                (Just(nd), Just(za), dims)
            })
    }

    // pub(crate) fn carray_strategy_for_reduction_single_dim<T: crate::util::ScalarStrategy>(
    //     elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    // ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<CompactArray<T>>, usize)>
    // {
    //     let shape = reduction_shape_strategy();
    //     let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
    //     array
    //         .prop_map(|(nd, za)| (nd, Rc::new(za)))
    //         .prop_flat_map(|(nd, za)| {
    //             let dim = dim_strategy(nd.ndim());
    //             (Just(nd), Just(za), dim)
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
            Rc<crate::util::TestArray<T>>,
            Vec<usize>,
        )>,
        check: fn(&ArrayD<T>, &crate::util::TestArray<T>, &[usize]),
    ) where
        T: crate::util::ScalarStrategy,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(&cases, |(nd, za, dims)| {
                check(&nd, &za, &dims);
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
                // `bool8` (where ordering comparisons like `x > m` are boolean, not numeric)
                // and same-width integer arms (where a widening cast like `x as u64` is a
                // no-op for the widest dtype in the list).
                #[test]
                #[allow(clippy::bool_comparison, clippy::unnecessary_cast)]
                fn [<$op_method _ $dtype:lower>]() {
                    crate::ops::reduction::common::tests::check_reduction::<$dtype>(
                        proptest::strategy::Strategy::boxed(
                            crate::ops::reduction::common::tests::array_strategy_for_reduction::<$dtype>(
                                <$dtype as crate::util::ScalarStrategy>::$strategy()
                            )
                        ),
                        |nd, za, dims| {
                            let result = za.view().$op_method(dims);
                            let expected = crate::ops::reduction::common::tests::ndarray_reduce(
                                nd, dims,
                                |arr| {
                                    let $items = arr.iter().cloned();
                                    $body
                                }
                            );
                            crate::ops::reduction::common::tests::assert_reduction_matches(&result, &expected);
                        },
                    );
                }
            }
        };

        (
            $op_method:ident,
            single_dim = true,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, dim) in crate::ops::reduction::common::tests::carray_strategy_for_reduction_single_dim::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(dim);
                        let expected = crate::ops::reduction::common::tests::ndarray_reduce(
                            &nd, &[dim],
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::ops::reduction::common::tests::assert_reduction_matches(&result, &expected);
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
            $(crate::ops::reduction::common::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::common::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };

        (
            $op_method:ident,
            single_dim = true,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::common::tests::test_reduction_dtype!(
                $op_method,
                single_dim = true,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::common::tests::test_reduction_dtype!(
                    $op_method,
                    single_dim = true,
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
            $(crate::ops::reduction::common::tests::test_reduction_dtype!(
                $op_method,
                |$items| { <_ as crate::scalar::Cast<$dtype>>::cast($body) },
                $dtype,
                $strategy
            );)+
        };
    }

    #[allow(unused_imports)]
    pub(crate) use {test_reduction, test_reduction_dtype};

    #[test]
    fn reduction_over_plain_with_hand_picked_strides() {
        use crate::{Array, ArrayParams, Ty};

        // The reduction reads its source in pull mode, and `Plain::read_data` in pull mode hands
        // back a zero-copy view with exactly the strides it was built with. So a `Plain` array
        // feeds the in-tile loop a chosen stride pattern directly - one row per shape that loop
        // can see, including the degenerate and unaligned ones the fast path has to reject.
        const SHAPE: [usize; 3] = [3, 4, 5];
        let shape_u64 = SHAPE.iter().map(|&s| s as u64).collect::<Vec<u64>>();

        // Deterministic bytes. Every bit pattern is a valid `i32`, so a read at any offset - even
        // an unaligned one - is well defined, and both sides read through the same strides, so the
        // values themselves never matter.
        let backing = (0..4096u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect::<Vec<u8>>();

        // (name, byte strides, base byte offset)
        let rows: &[(&str, [usize; 3], usize)] = &[
            ("row major", [80, 20, 4], 0),
            ("reversed, dim 0 innermost", [4, 12, 48], 0),
            ("permuted, dim 1 innermost", [80, 4, 16], 0),
            ("zero stride on dim 1", [80, 0, 4], 0),
            ("zero stride on dim 0", [0, 20, 4], 0),
            ("gaps on every dim", [320, 80, 8], 0),
            ("unaligned base", [80, 20, 4], 1),
            ("unaligned base and strides", [82, 22, 6], 1),
        ];

        // The default window puts the whole array in one tile; the tiny one forces the reduced
        // extent across many tiles, so a cell's state is merged rather than only seeded.
        let windows: &[Option<(u64, u64)>] = &[None, Some((16, 64))];

        for &(name, strides, base) in rows {
            let val = |idx: [usize; 3]| -> i32 {
                let off = base + (0..3).map(|d| idx[d] * strides[d]).sum::<usize>();
                // SAFETY: every row's furthest reachable byte is well inside `backing`.
                unsafe { backing.as_ptr().add(off).cast::<i32>().read_unaligned() }
            };
            // Output index of a full index, dropping the reduced dims.
            let out_idx = |full: [usize; 3], kept: &[usize]| {
                ndarray::IxDyn(&kept.iter().map(|&d| full[d]).collect::<Vec<_>>())
            };

            for &window in windows {
                let mut params = ArrayParams::new();
                if let Some(w) = window {
                    params = params.read_size(w);
                }
                let arr: Array<crate::storage::Plain<_, Ty<i32>>> = unsafe {
                    // SAFETY: `backing` outlives `arr`, and every index combination stays in bounds.
                    Array::plain_ndarray_ptr(
                        backing.as_ptr().add(base),
                        &shape_u64,
                        &strides,
                        <i32 as crate::dtype::Dtyped>::DTYPE,
                        params,
                    )
                }
                .unwrap();

                for mask in 1u8..8 {
                    let dims = (0..3)
                        .filter(|d| mask >> d & 1 == 1)
                        .collect::<Vec<usize>>();
                    let kept = (0..3)
                        .filter(|d| mask >> d & 1 == 0)
                        .collect::<Vec<usize>>();
                    let out_shape = kept.iter().map(|&d| SHAPE[d]).collect::<Vec<usize>>();
                    eprintln!("row={name} window={window:?} dims={dims:?}");

                    let mut exp_sum = ndarray::ArrayD::<i64>::zeros(ndarray::IxDyn(&out_shape));
                    let mut exp_max =
                        ndarray::ArrayD::<i32>::from_elem(ndarray::IxDyn(&out_shape), i32::MIN);
                    for i in 0..SHAPE[0] {
                        for j in 0..SHAPE[1] {
                            for k in 0..SHAPE[2] {
                                let oi = out_idx([i, j, k], &kept);
                                let v = val([i, j, k]);
                                exp_sum[oi.clone()] += v as i64;
                                if v > exp_max[oi.clone()] {
                                    exp_max[oi] = v;
                                }
                            }
                        }
                    }
                    crate::util::assert_array_matches(&arr.view().sum(dims.as_slice()), &exp_sum);
                    crate::util::assert_array_matches(&arr.view().max(dims.as_slice()), &exp_max);
                }

                // `argmax` is single-dim, and the only op whose result depends on an element's
                // position in the reduced stream.
                for dim in 0..3 {
                    let kept = (0..3).filter(|&d| d != dim).collect::<Vec<usize>>();
                    let out_shape = kept.iter().map(|&d| SHAPE[d]).collect::<Vec<usize>>();
                    eprintln!("row={name} window={window:?} argmax dim={dim}");

                    let dyn_shape = ndarray::IxDyn(&out_shape);
                    let mut exp = ndarray::ArrayD::<u64>::zeros(dyn_shape.clone());
                    let mut best = ndarray::ArrayD::<Option<i32>>::from_elem(dyn_shape, None);
                    for i in 0..SHAPE[0] {
                        for j in 0..SHAPE[1] {
                            for k in 0..SHAPE[2] {
                                // Row-major nesting visits a given output cell with a strictly
                                // increasing `dim` coordinate, so a strict `>` keeps the first
                                // maximum - matching the kernel's tie-break.
                                let full = [i, j, k];
                                let oi = out_idx(full, &kept);
                                let v = val(full);
                                if best[oi.clone()].is_none_or(|b| v > b) {
                                    best[oi.clone()] = Some(v);
                                    exp[oi] = full[dim] as u64;
                                }
                            }
                        }
                    }
                    crate::util::assert_array_matches(&arr.view().argmax(dim), &exp);
                }
            }
        }
    }

    /// Drives a `sum` reduction through the two-level bulk/tile chunking, with `block_shape`
    /// and a small `read_size` chosen so the tile lands at a genuine sub-block (smaller than
    /// the array along both a reduced and a non-reduced dim). That exercises both non-trivial
    /// paths at once: more than one *bulk*, and more than one *tile* within a bulk.
    ///
    /// The reduction's `read_data` is driven **directly** over the full output range in one
    /// call. The top-level readers (`to_ndarray`, `to_ndarray_sub`) chunk the output by the
    /// reduction op's block shape before calling `read_data`, so each call's non-reduced extent
    /// equals one block and the read-shape heuristic snaps the tile to it - never subdividing
    /// the non-reduced dim. Handing `read_data` the whole non-reduced extent at once makes it
    /// split that dim into several tiles per bulk. Correctness is exact: the i64 sum of small
    /// signed values can't reassociate or overflow. During development, path coverage is
    /// confirmed with the temporary debug print in `ReductionOp::read_data`.
    fn check_sum_divided(
        shape: &[usize],
        block_shape: &[u32],
        read_size: (u64, u64),
        dims: &[usize],
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

        let params = ArrayParams::new()
            .block_shape(block_shape)
            .unwrap()
            .read_size(read_size);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();

        let reduced = za.view().sum(dims);
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
                .read_data(&Index::new(&full_index).unwrap(), &ctx, Some(&mut out))
                .unwrap();
        }

        let expected = ndarray_reduce(&nd, dims, |v| v.iter().map(|&x| x as i64).sum::<i64>());
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
        let reduced = za.view().sum([1usize]);
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
            storage
                .read_data(&Index::new(&[0..3]).unwrap(), &ctx, Some(&mut out))
                .unwrap();
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
        let reduced = za.view().max(1usize); // output shape [5]
        let ctx = reduced.read_ctx();
        // An empty sub-range along the (sole) output dim.
        let got = reduced.to_ndarray_sub::<Ix1>(&[1..1], &ctx).unwrap();
        assert_eq!(got.shape(), &[0]);
    }

    #[test]
    fn sum_multi_bulk_multi_tile_2d_reduce_outer() {
        // Reduce dim 0 (the strided/outer dim), tile == block == [2, 3]. Current scheme:
        // bulks split the reduced dim 0 (8/2 = 4 bulks); within each bulk, tiles split the
        // non-reduced dim 1 (6/3 = 2 tiles/bulk).
        check_sum_divided(&[8, 6], &[2, 3], (32, 64), &[0]);
    }

    #[test]
    fn sum_multi_bulk_multi_tile_2d_reduce_inner() {
        // Reduce dim 1 (the contiguous/inner dim), tile == block == [3, 2]. bulks split
        // reduced dim 1 (8/2 = 4); tiles split non-reduced dim 0 (6/3 = 2).
        check_sum_divided(&[6, 8], &[3, 2], (32, 64), &[1]);
    }

    #[test]
    fn sum_multi_bulk_multi_tile_3d_reduce_middle() {
        // 3D, reduce the middle dim, tile == block == [2, 2, 2]. bulks split reduced dim 1
        // (4/2 = 2); tiles split the two non-reduced dims 0 and 2 ((4/2)*(4/2) = 4 tiles/bulk).
        check_sum_divided(&[4, 4, 4], &[2, 2, 2], (32, 64), &[1]);
    }

    #[test]
    fn sum_f32_pairwise_error_does_not_grow_with_run_length() {
        // `fold_pairwise` splits a run longer than `PAIRWISE_SPLIT_THRESHOLD` in half, so a long f32 sum
        // accumulates in a log-deep tree rather than one linear chain per lane. One huge leading
        // value followed by ones makes the difference visible: at 2^24 an f32 cannot represent an
        // odd neighbour, so every `1.0` folded straight onto the big running total is lost. A
        // linear chain loses ~len/LANES of them and gets worse the longer the run, while pairwise
        // loses only the few inside the one leaf holding the big value, however long the run is.
        const N: usize = 65536;
        let mut data = vec![1.0f32; N];
        data[0] = (1u32 << 24) as f32;
        // Exact in f64 - both 2^24 and the count of ones are represented exactly.
        let exact = f64::from(1u32 << 24) + (N - 1) as f64;

        let nd = ndarray::ArrayD::from_shape_vec(vec![N], data).unwrap();
        // One block and one tile over the whole array, so the run reaches a single
        // `fold_pairwise` call rather than being split across tiles and merged sequentially.
        let params = ArrayParams::new()
            .block_shape(&[N as u32])
            .unwrap()
            .read_size((1 << 19, 1 << 20));
        let za = Array::compact_ndarray_with(&nd, params).unwrap();

        let got = f64::from(za.view().sum(0usize).to_ndarray::<Ix0>().unwrap()[[]]);
        let err = (exact - got).abs();
        // Pairwise loses 7 of the ones; a 16-lane linear chain over the whole run loses 4095.
        // The bound is loose enough to survive retuning LANES / PAIRWISE_SPLIT_THRESHOLD, tight enough that
        // going back to a single linear pass fails it.
        assert!(
            err <= 64.0,
            "f32 sum lost {err} (got {got}, exact {exact}) - is the run still folded pairwise?",
        );
    }

    #[test]
    fn reduce_huge_lazy_shape() {
        let a = Array::compact_ndarray(&array![[1u8]]).unwrap();
        let a = crate::ops::Broadcast::new_array(a, &[i64::MAX as u64, 2]).unwrap();
        let s = crate::ops::Max::new_array(a, (0, 1)).unwrap();
        assert_eq!(s.shape(), &[] as &[u64]);
    }

    #[test]
    fn sum_f32_long_runs_with_tails() {
        // Runs around and past `PAIRWISE_SPLIT_THRESHOLD`, most not a multiple of the lanes: the tree folds
        // the body, then the tail is folded after it. Small integers sum exactly in any order, so
        // the sum checks that every item is folded exactly once.
        for n in [1, 15, 16, 17, 2047, 2048, 2049, 4095, 4111, 8193, 20011] {
            let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
            let exact = (0..n).map(|i| i % 7).sum::<usize>() as f32;
            let nd = ndarray::ArrayD::from_shape_vec(vec![n], data).unwrap();
            // One block and one tile: the whole run is one `fold_pairwise` call.
            let params = ArrayParams::new()
                .block_shape(&[n as u32])
                .unwrap()
                .read_size((1 << 19, 1 << 20));
            let za = Array::compact_ndarray_with(&nd, params).unwrap();
            assert_eq!(
                za.view().sum(0usize).to_ndarray::<Ix0>().unwrap()[[]],
                exact,
                "n = {n}"
            );
        }
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

        // [4, 4, 4] i8, reduce the middle dim -> [4, 4] i64.
        let nd = ndarray::ArrayD::from_shape_vec(
            vec![4usize, 4, 4],
            (0..64i32).map(|x| (x % 97) as i8).collect(),
        )
        .unwrap();
        let params = ArrayParams::new()
            .block_shape(&[2, 2, 2])
            .unwrap()
            .read_size((32, 64));
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let reduced = za.view().sum(1usize); // [4, 4] i64
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
            storage
                .read_data(&Index::new(&index).unwrap(), &ctx, Some(&mut out))
                .unwrap();
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
            storage
                .read_data(&Index::new(&index).unwrap(), &ctx, Some(&mut out))
                .unwrap();
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
        let reduced = za.view().var(1usize, 0.0); // [3] f64
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        let mut expected = vec![0f64; 3];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(expected.as_mut_ptr().cast::<u8>(), 3 * 8)
            };
            let c = crate::util::default_strides_slice(&[3usize], size_of::<f64>());
            let mut out = unsafe { crate::storage::StridedBuf::from_slice_mut(bytes, c.as_ref()) };
            storage
                .read_data(&Index::new(&[0..3]).unwrap(), &ctx, Some(&mut out))
                .unwrap();
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
            storage
                .read_data(&Index::new(&[0..3]).unwrap(), &ctx, Some(&mut out))
                .unwrap();
        }

        for i in 0..3 {
            assert_eq!(backing[i * 2], expected[i], "slot {i}");
            assert_eq!(backing[i * 2 + 1], SENTINEL, "gap {i} was overwritten");
        }
    }

    pub(crate) fn ndarray_reduce<'a, S, D, O>(
        array: &'a ndarray::ArrayBase<S, D>,
        dims: &[usize],
        f: impl Fn(&ndarray::ArrayViewD<'a, S::Elem>) -> O,
    ) -> ndarray::ArrayD<O>
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        // Output shape = original with reduction dims removed
        let mut dims = dims.to_vec();
        dims.sort_unstable();
        dims.dedup();

        let out_shape = array
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &s)| s)
            .collect::<Vec<_>>();

        let values = ndarray_reduction_iter(array, &dims)
            .map(|(_, view)| f(&view))
            .collect::<Vec<_>>();

        ndarray::ArrayD::from_shape_vec(out_shape, values).unwrap()
    }

    /// Iterates over all index combinations of the **kept** dims (i.e. dims NOT in `dims`),
    /// yielding for each combination the multi-index into the kept dims and a view spanning
    /// the reduction dims.
    fn ndarray_reduction_iter<'a, S, D>(
        array: &'a ndarray::ArrayBase<S, D>,
        dims: &[usize],
    ) -> impl Iterator<Item = (Vec<usize>, ndarray::ArrayViewD<'a, S::Elem>)> + 'a
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        let mut dims = dims.to_vec();
        dims.sort_unstable();
        dims.dedup();

        // Kept dims = all dims not being reduced
        let ndim = array.ndim();
        let kept_dims = (0..ndim).filter(|i| !dims.contains(i)).collect::<Vec<_>>();

        // Shape of the kept dims - this is what we iterate over
        let kept_shape = kept_dims
            .iter()
            .map(|&d| array.shape()[d])
            .collect::<Vec<_>>();
        let total: usize = kept_shape.iter().product();

        (0..total).map(move |flat_idx| {
            // Convert flat index to multi-index over the kept dims
            let mut remaining = flat_idx;
            let mut kept_indices = Vec::with_capacity(kept_dims.len());
            for &dim_size in kept_shape.iter().rev() {
                kept_indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            kept_indices.reverse();

            // Fix each kept dim to its index, remove in descending order.
            // We remove kept dims (which are the non-reduction dims), leaving
            // a view over the reduction dims.
            let mut view = array.view().into_dyn();

            // We must track dim offset: as we remove dims, remaining dim
            // indices shift down. Process kept dims in descending order.
            let mut pairs: Vec<(usize, usize)> = kept_dims
                .iter()
                .copied()
                .zip(kept_indices.iter().copied())
                .collect();
            pairs.sort_unstable_by_key(|p| core::cmp::Reverse(p.0));

            for (d, idx) in &pairs {
                view = view.index_axis_move(ndarray::Axis(*d), *idx);
            }

            (kept_indices, view)
        })
    }

    #[test]
    fn reduction_rejects_zero_itemsize_kernel_output() {
        use crate::Array;

        // A reduction kernel picks its own output type, so it is another way a fresh dtype reaches an
        // array without passing through `ArrayParams::tune` (a lazy view never re-tunes). `Empty`
        // is a `Dtyped` type with itemsize 0.
        //
        // `Fold` is the user-facing op that exposes a caller-chosen accumulator, but it is commented
        // out pending the reduction merge, so the kernel is spelled out here instead.
        #[derive(Copy, Clone, crate::dtype::Dtyped)]
        #[dtyped(crate = "crate")]
        #[repr(C)]
        struct Empty {}

        struct ZeroSizedOutputKernel;
        impl super::ReductionOpKernel<i32> for ZeroSizedOutputKernel {
            type Output = Empty;
            type State = ();
            fn init_state(&self, _init_item: Option<(i32, Option<u64>)>) -> Self::State {}
            fn update_state(
                &self,
                _state: Self::State,
                _item: i32,
                _idx: Option<u64>,
            ) -> Self::State {
            }
            fn merge_states(&self, _a: Self::State, _b: Self::State) -> Self::State {}
            fn finalize_state(&self, _state: Self::State, _nitems: u64) -> Self::Output {
                Empty {}
            }
            const SUPPORTS_EMPTY: bool = true;
            const PREFER_TREE_MERGE: bool = false;
        }

        let za = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let err = super::ReductionOp::<_, _>::new(
            za.into_storage(),
            ZeroSizedOutputKernel,
            [0usize].as_slice(),
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::UnsupportedDtype);
    }

    #[test]
    #[cfg(all(target_pointer_width = "32", not(debug_assertions)))] // slow in debug builds
    fn reduce_more_than_2pow32_items_on_32_bit() {
        let n = (1u64 << 32) + 16;
        let ones = Array::compact_ndarray(&array![[1u8, 1]]).unwrap();
        let sum = ones.view().broadcast(&[n, 2]).sum(0usize);
        assert_eq!(sum.to_ndarray::<Ix1>().unwrap(), array![n, n]);
        let sum = ones.view().broadcast(&[n, 2]).sum((0, 1));
        assert_eq!(sum.to_ndarray::<Ix0>().unwrap()[[]], 2 * n);
        let mean = ones
            .view()
            .map(|x: u8| x as f32)
            .broadcast(&[n, 2])
            .mean(0usize);
        assert_eq!(mean.to_ndarray::<Ix1>().unwrap(), array![1.0, 1.0]);
        let zeros = Array::compact_ndarray(&array![0u8])
            .unwrap()
            .broadcast(&[n]);
        let one = Array::compact_ndarray(&array![1u8])
            .unwrap()
            .broadcast(&[1]);
        let argmax = crate::ops::concatenate((zeros, one), 0).argmax(0);
        assert_eq!(argmax.to_ndarray::<Ix0>().unwrap()[[]], n);
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    fn var_state_buffer_must_fit_in_memory_on_32_bit() {
        use crate::ArrayStorage;
        // The 4-byte output fits, the 24-byte states do not.
        let n = (1u64 << 32) / 24 + 1;
        let a = Array::compact_ndarray(&array![[1.0f32]]).unwrap();
        let v = a.broadcast(&[n, 1]).var(1, 0.0);
        let ctx = crate::ReadContext::default();
        assert!(v.storage().read_data(&[0..n], &ctx, None).is_err());
    }
}
