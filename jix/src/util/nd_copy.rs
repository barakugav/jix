use std::ptr;

use crate::dtype::{Alignment, Dtype, Itemsize};
use crate::util::{
    copy_nonoverlapping_nontemporal, strided_span_bytes, DimArray, NONTEMPORAL_MIN_RUN_BYTES,
};
use crate::{NdIterUnordered, PtrExt, PtrMutExt, PtrMutNoalias, PtrNoalias};

/// A copy at least this large, between two layouts that order the dims differently, is cut into
/// tiles (see [`NdCopier::copy`]). Below it the whole region sits in cache anyway.
const BLOCKED_COPY_MIN_BYTES: usize = 1 << 20;
/// Size of one tile of a blocked copy. A tile is staged through two compact scratch buffers of
/// this size, which together should sit in L1.
const BLOCKED_COPY_TILE_BYTES: usize = 16 << 10;
/// How long a run a tile of a blocked copy writes to the destination, when the region allows.
/// Long runs touch every destination cache line once; short ones leave lines half written, to be
/// fetched again when the neighbouring tile completes them.
const BLOCKED_COPY_DST_RUN_BYTES: usize = 1024;
/// A run that is contiguous in both layouts and at least this long needs no blocking: every run
/// is a streaming copy of its own, whatever order the runs are visited in.
const BLOCKED_COPY_MAX_RUN_BYTES: usize = 4096;

/// The dims a blocked copy should tile over, outermost first in the destination's order, or none
/// if the copy needs no blocking.
///
/// Order the dims of each side outermost first (by stride). Where the two orders agree the copy
/// already streams through both sides. Where they differ, the dims *outside* the common innermost
/// suffix are the ones to tile over - unless that common suffix is already a long contiguous run,
/// in which case each run streams on its own and the order the runs are visited in is free.
/// Dims of extent 1 take no part in either order; a broadcast (stride 0) source is never blocked.
fn blocked_copy_split_dims(
    shape: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
    itemsize: usize,
) -> DimArray<usize> {
    if (0..shape.len()).any(|d| shape[d] > 1 && src_strides[d] == 0) {
        return DimArray::new();
    }
    let (src_order, dst_order) = (
        dims_outermost_first(shape, src_strides),
        dims_outermost_first(shape, dst_strides),
    );
    let suffix_len = src_order
        .iter()
        .rev()
        .zip(dst_order.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let n = dst_order.len();
    if suffix_len == n {
        return DimArray::new();
    }
    let run_bytes = dst_order[n - suffix_len..]
        .iter()
        .map(|&d| shape[d])
        .product::<usize>()
        * itemsize;
    if run_bytes >= BLOCKED_COPY_MAX_RUN_BYTES {
        return DimArray::new();
    }
    DimArray::from_slice(&dst_order[..n - suffix_len]).unwrap()
}

/// The dims of extent above 1, sorted by descending stride: the layout's outermost dim first.
fn dims_outermost_first(shape: &[usize], strides: &[usize]) -> DimArray<usize> {
    let mut order = (0..shape.len())
        .filter(|&d| shape[d] > 1)
        .collect::<DimArray<usize>>();
    order
        .as_mut_slice()
        .sort_by_key(|&d| std::cmp::Reverse(strides[d]));
    order
}

/// Packed strides for `shape` that nest the dims in the same order as `like_strides` does.
fn packed_strides_like(
    shape: &[usize],
    like_strides: &[usize],
    itemsize: usize,
) -> DimArray<usize> {
    let mut order = (0..shape.len()).collect::<DimArray<usize>>();
    order
        .as_mut_slice()
        .sort_by_key(|&d| std::cmp::Reverse(like_strides[d]));
    let mut strides = DimArray::from_slice(like_strides).unwrap();
    let mut acc = itemsize;
    for &d in order.iter().rev() {
        strides[d] = acc;
        acc *= shape[d];
    }
    strides
}

/// A reusable, dtype-specialized copier that moves a rectangular n-dimensional region between two
/// byte slices under independent source and destination strides.
///
/// [`new`](Self::new) inspects the dtype once and picks the cheapest copy routine up front: a
/// monomorphized scalar copy for the common power-of-two `(itemsize, alignment)` pairs, a
/// field-by-field copy for structs that decompose into at most four scalar fields, or a generic
/// byte-wise fallback for everything else. Each [`copy`](Self::copy) then moves one region by
/// driving an [`NdIterUnordered`] over `shape` (which sorts, coalesces and walks the dims) and
/// copying one contiguous-or-strided inner loop at each visited position.
pub(crate) struct NdCopier {
    copy_fn: NdCopyFn,
    /// Write contiguous runs of the destination with non-temporal stores, see
    /// [`nontemporal`](Self::nontemporal).
    nontemporal: bool,
}

type NdCopyFn = unsafe fn(NdCopyArgs);
type InnerLoopFn<T> = unsafe fn(PtrNoalias<'_, T>, PtrMutNoalias<'_, T>, usize, usize, usize);
struct NdCopyArgs<'a> {
    src: PtrNoalias<'a, u8>,
    dst: PtrMutNoalias<'a, u8>,
    shape: &'a [usize],
    src_strides: &'a [usize],
    dst_strides: &'a [usize],
    dtype: &'a Dtype,
    nontemporal: bool,
}

impl NdCopier {
    #[inline]
    pub(crate) const fn new(dtype: &Dtype) -> Self {
        let copy_fn: NdCopyFn = match (dtype.itemsize(), dtype.alignment().as_usize()) {
            (1, 1) => Self::scalar_fn::<u8>,
            (2, 2) => Self::scalar_fn::<u16>,
            (4, 4) => Self::scalar_fn::<u32>,
            (8, 4) => Self::scalar_fn::<[u32; 2]>,
            (8, 8) => Self::scalar_fn::<u64>,
            (16, 8) => Self::scalar_fn::<[u64; 2]>,
            _ => Self::copy_untyped,
        };
        Self {
            copy_fn,
            nontemporal: false,
        }
    }

    /// Write the destination's contiguous runs with non-temporal stores (see
    /// [`copy_nonoverlapping_nontemporal`]). For a destination far larger than the cache: it is
    /// then written at about twice the throughput, but is not left in the cache. Whoever sets this
    /// runs [`nontemporal_fence`](crate::util::nontemporal_fence) once the copies are done.
    #[inline]
    pub(crate) const fn nontemporal(mut self, nontemporal: bool) -> Self {
        self.nontemporal = nontemporal;
        self
    }

    /// Copies a rectangular `shape` region from `src` to `dst` under independent byte strides.
    ///
    /// `src` and `dst` are [`PtrNoalias`] / [`PtrMutNoalias`] - references in the ABI, so LLVM
    /// keeps on the inner loops' operands the `noalias` tag it would drop on a bare raw pointer.
    ///
    /// # Safety
    ///
    /// - Each wrapped byte range must be a superset of every byte the copy touches. For a region
    ///   with the given `shape`/strides that is `sum_d (shape[d] - 1) * stride[d] + itemsize` bytes
    ///   from the start (all byte strides are non-negative). A shorter range is undefined
    ///   behavior even though no explicit bounds checks run.
    /// - The `src` and `dst` regions must not overlap.
    /// - `dtype` must have the same itemsize and alignment as the dtype passed to
    ///   [`new`](Self::new). The copy is sized by that dtype, and a mismatch is only caught by a
    ///   `debug_assert`.
    #[inline(always)]
    pub(crate) unsafe fn copy(
        &self,
        src: PtrNoalias<'_, u8>,
        dst: PtrMutNoalias<'_, u8>,
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        dtype: &Dtype,
    ) {
        if shape.contains(&0) {
            return;
        }
        // A big copy between layouts that disagree on the dim order (a transpose) walks one side
        // with a huge stride: a cache and TLB miss per element. Cut it into tiles instead.
        let itemsize = dtype.itemsize() as usize;
        let nbytes = shape.iter().product::<usize>() * itemsize;
        if nbytes >= BLOCKED_COPY_MIN_BYTES {
            let split_dims = blocked_copy_split_dims(shape, src_strides, dst_strides, itemsize);
            if !split_dims.is_empty() {
                return unsafe {
                    self.copy_blocked(
                        src.as_ptr(),
                        dst.as_mut_ptr(),
                        shape,
                        src_strides,
                        dst_strides,
                        dtype,
                        &split_dims,
                    )
                };
            }
        }
        // Dim ordering, size-1-dim dropping and contiguous-run coalescing all happen inside
        // `nd_iter_unordered` (see `scalar_fn` / `copy_untyped`); here we just dispatch by dtype.
        let args = NdCopyArgs {
            src,
            dst,
            shape,
            src_strides,
            dst_strides,
            dtype,
            nontemporal: self.nontemporal,
        };
        unsafe { (self.copy_fn)(args) }
    }

    /// [`copy`](Self::copy) for a large region whose two layouts order the dims differently.
    ///
    /// The region is cut into small tiles over `split_dims`, and each tile takes three plain
    /// copies through two compact scratch buffers:
    ///
    /// 1. gather the tile from the source into a buffer packed in the *source's* dim order,
    /// 2. reorder it into a buffer packed in the *destination's* dim order,
    /// 3. write that out to the destination.
    ///
    /// Steps 1 and 3 run along their side's own innermost dim, so every cache line of the source
    /// and of the destination is touched once, in whole runs. Only step 2 jumps around, and it
    /// stays in L1. A copy straight from source to destination would instead revisit each line
    /// once per element of the other side's run - and with a power-of-two stride (an array whose
    /// dims are powers of two) those lines all fall in the same few cache sets and evict each
    /// other between visits.
    ///
    /// A tile is long along the destination's innermost dim ([`BLOCKED_COPY_DST_RUN_BYTES`]),
    /// which matters most, and spends the rest of its budget on the source's innermost dim.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    unsafe fn copy_blocked(
        &self,
        src: *const u8,
        dst: *mut u8,
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        dtype: &Dtype,
        split_dims: &[usize],
    ) {
        let ndim = shape.len();
        let itemsize = dtype.itemsize() as usize;

        // Tile shape: dims outside `split_dims` keep their whole extent.
        let mut tile = DimArray::from_slice(shape).unwrap();
        for &d in split_dims {
            tile[d] = 1;
        }
        let cell_bytes = tile.iter().product::<usize>() * itemsize;
        let mut budget = (BLOCKED_COPY_TILE_BYTES / cell_bytes).max(1);
        let dst_inner = *split_dims.last().unwrap();
        let src_inner = split_dims
            .iter()
            .copied()
            .min_by_key(|&d| src_strides[d])
            .unwrap();
        let run_len = (BLOCKED_COPY_DST_RUN_BYTES / cell_bytes).max(1);
        // The destination's innermost dim first, then the source's, then the rest inside out.
        let grow_order = [dst_inner, src_inner]
            .into_iter()
            .chain(split_dims.iter().rev().copied())
            .enumerate();
        for (i, d) in grow_order {
            if tile[d] != 1 {
                continue; // already grown
            }
            let cap = if i == 0 { run_len.min(budget) } else { budget };
            tile[d] = shape[d].min(cap).max(1);
            budget /= tile[d];
        }

        let tile_bytes = tile.iter().product::<usize>() * itemsize;
        let alignment = dtype.alignment().as_usize();
        let mut gathered = crate::util::AlignedBytes::with_capacity_exact(alignment, tile_bytes);
        let mut reordered = crate::util::AlignedBytes::with_capacity_exact(alignment, tile_bytes);
        unsafe {
            gathered.set_len(tile_bytes);
            reordered.set_len(tile_bytes);
        }

        // Odometer over the tiles of `split_dims`, the destination's outermost dim slowest.
        let mut start = DimArray::from_slice(shape).unwrap();
        start.as_mut_slice().fill(0);
        let mut size = tile.clone();
        loop {
            for &d in split_dims {
                size[d] = tile[d].min(shape[d] - start[d]);
            }
            let (src_off, dst_off) = (0..ndim).fold((0, 0), |(s, t), d| {
                (s + start[d] * src_strides[d], t + start[d] * dst_strides[d])
            });
            let nbytes = size.iter().product::<usize>() * itemsize;
            let gathered_strides = packed_strides_like(&size, src_strides, itemsize);
            let reordered_strides = packed_strides_like(&size, dst_strides, itemsize);
            let step = |src: PtrNoalias<'_, u8>,
                        dst: PtrMutNoalias<'_, u8>,
                        src_strides: &[usize],
                        dst_strides: &[usize],
                        nontemporal: bool| {
                let args = NdCopyArgs {
                    src,
                    dst,
                    shape: &size,
                    src_strides,
                    dst_strides,
                    dtype,
                    nontemporal,
                };
                unsafe { (self.copy_fn)(args) }
            };
            unsafe {
                step(
                    PtrNoalias::new(
                        src.add(src_off),
                        strided_span_bytes(&size, src_strides, itemsize),
                    ),
                    PtrMutNoalias::from_slice(&mut gathered.as_mut_slice()[..nbytes]),
                    src_strides,
                    &gathered_strides,
                    false,
                );
                step(
                    PtrNoalias::from_slice(&gathered.as_slice()[..nbytes]),
                    PtrMutNoalias::from_slice(&mut reordered.as_mut_slice()[..nbytes]),
                    &gathered_strides,
                    &reordered_strides,
                    false,
                );
                step(
                    PtrNoalias::from_slice(&reordered.as_slice()[..nbytes]),
                    PtrMutNoalias::new(
                        dst.add(dst_off),
                        strided_span_bytes(&size, dst_strides, itemsize),
                    ),
                    &reordered_strides,
                    dst_strides,
                    self.nontemporal,
                );
            }

            // Advance to the next tile.
            let mut advanced = false;
            for &d in split_dims.iter().rev() {
                start[d] += tile[d];
                if start[d] < shape[d] {
                    advanced = true;
                    break;
                }
                start[d] = 0;
            }
            if !advanced {
                break;
            }
        }
    }

    unsafe fn scalar_fn<T: Copy + 'static>(args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            dst,
            shape,
            src_strides,
            dst_strides,
            dtype,
            nontemporal,
        } = args;
        debug_assert_eq!(size_of::<T>(), dtype.itemsize() as usize);
        debug_assert_eq!(align_of::<T>(), dtype.alignment().as_usize());
        let (src, mut dst) = (src.cast::<T>(), dst.cast::<T>());
        let base_ptrs_aligned = dst.as_ptr().is_aligned() && src.as_ptr().is_aligned();

        if shape.len() == 1 {
            // Skip the iterator for single dim arrays

            let len = shape[0];
            if len == 1 {
                unsafe {
                    let value = src.as_ptr().read_unaligned();
                    dst.as_mut_ptr().write_unaligned(value);
                };
                return;
            }

            let src_stride = src_strides[0];
            let dst_stride = dst_strides[0];
            let aligned = base_ptrs_aligned
                && src_stride.is_multiple_of(align_of::<T>())
                && dst_stride.is_multiple_of(align_of::<T>());

            let inner_loop_fn = Self::pick_inner_loop::<T>(
                len,
                aligned,
                src_stride == size_of::<T>(),
                dst_stride == size_of::<T>(),
                src_stride == 0,
                nontemporal,
            );
            unsafe { inner_loop_fn(src, dst, len, src_stride, dst_stride) };
            return;
        }

        // Operand 0 is the destination (write; sort-primary), operand 1 the source (read). Both are
        // byte-addressed with element layout `(size_of::<T>(), align_of::<T>())`.
        let iter = NdIterUnordered::new(
            shape,
            [dst_strides, src_strides],
            [(size_of::<T>() as Itemsize, Alignment::of::<T>()); 2],
        );
        let [dst_strides_aligned, src_strides_aligned] = *iter.is_strides_aligned();
        let is_aligned = dst_strides_aligned && src_strides_aligned && base_ptrs_aligned;
        let [dst_contiguous, src_contiguous] = *iter.is_contiguous();
        let inner_loop_fn = Self::pick_inner_loop::<T>(
            iter.inner_len(),
            is_aligned,
            src_contiguous,
            dst_contiguous,
            iter.inner_strides()[1] == 0,
            nontemporal,
        );

        iter.foreach_inner_1d_impl::<true>(
            move |[dst_offset, src_offset], len, [dst_stride, src_stride]| unsafe {
                inner_loop_fn(
                    src.bytes_offset(src_offset),
                    dst.bytes_offset(dst_offset),
                    len,
                    src_stride,
                    dst_stride,
                )
            },
        );
    }

    #[inline(always)]
    fn pick_inner_loop<T: Copy>(
        inner_len: usize,
        aligned: bool,
        src_contiguous: bool,
        dst_contiguous: bool,
        src_broadcast: bool,
        nontemporal: bool,
    ) -> InnerLoopFn<T> {
        if nontemporal
            && src_contiguous
            && dst_contiguous
            && inner_len * size_of::<T>() >= NONTEMPORAL_MIN_RUN_BYTES
        {
            Self::inner_loop_contiguous_nontemporal::<T>
        } else if src_contiguous && dst_contiguous {
            // For the both-contiguous run, peel a small fixed length into a single `[T; N]` move
            // (branchless); longer runs fall through to `copy_nonoverlapping`.
            match (inner_len, aligned) {
                (1, true) => Self::inner_loop_contiguous_const_len::<T, 1, true>,
                (1, false) => Self::inner_loop_contiguous_const_len::<T, 1, false>,
                (2, true) => Self::inner_loop_contiguous_const_len::<T, 2, true>,
                (2, false) => Self::inner_loop_contiguous_const_len::<T, 2, false>,
                (4, true) => Self::inner_loop_contiguous_const_len::<T, 4, true>,
                (4, false) => Self::inner_loop_contiguous_const_len::<T, 4, false>,
                (8, true) => Self::inner_loop_contiguous_const_len::<T, 8, true>,
                (8, false) => Self::inner_loop_contiguous_const_len::<T, 8, false>,
                (16, true) => Self::inner_loop_contiguous_const_len::<T, 16, true>,
                (16, false) => Self::inner_loop_contiguous_const_len::<T, 16, false>,
                (32, true) if const { size_of::<T>() <= 16 } => {
                    Self::inner_loop_contiguous_const_len::<T, 32, true>
                }
                (32, false) if const { size_of::<T>() <= 16 } => {
                    Self::inner_loop_contiguous_const_len::<T, 32, false>
                }
                (64, true) if const { size_of::<T>() <= 8 } => {
                    Self::inner_loop_contiguous_const_len::<T, 64, true>
                }
                (64, false) if const { size_of::<T>() <= 8 } => {
                    Self::inner_loop_contiguous_const_len::<T, 64, false>
                }
                (_, true) => Self::inner_loop::<T, true, true, true>,
                (_, false) => Self::inner_loop::<T, false, true, true>,
            }
        } else if src_broadcast {
            match (aligned, dst_contiguous) {
                (true, true) => Self::inner_loop_broadcast::<T, true, true>,
                (true, false) => Self::inner_loop_broadcast::<T, true, false>,
                (false, true) => Self::inner_loop_broadcast::<T, false, true>,
                (false, false) => Self::inner_loop_broadcast::<T, false, false>,
            }
        } else {
            match (aligned, [src_contiguous, dst_contiguous]) {
                (_, [true, true]) => unreachable!(),
                (true, [true, false]) => Self::inner_loop::<T, true, true, false>,
                (false, [true, false]) => Self::inner_loop::<T, false, true, false>,
                (true, [false, true]) => Self::inner_loop::<T, true, false, true>,
                (false, [false, true]) => Self::inner_loop::<T, false, false, true>,
                (true, [false, false]) => Self::inner_loop::<T, true, false, false>,
                (false, [false, false]) => Self::inner_loop::<T, false, false, false>,
            }
        }
    }

    unsafe fn inner_loop<
        T: Copy,
        const ALIGNED: bool,
        const SRC_CONTIGUOUS: bool,
        const DST_CONTIGUOUS: bool,
    >(
        src: PtrNoalias<T>,
        dst: PtrMutNoalias<T>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        unsafe { std::hint::assert_unchecked(SRC_CONTIGUOUS == (src_stride == size_of::<T>())) };
        unsafe { std::hint::assert_unchecked(DST_CONTIGUOUS == (dst_stride == size_of::<T>())) };
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        unsafe {
            if SRC_CONTIGUOUS && DST_CONTIGUOUS {
                if ALIGNED {
                    ptr::copy_nonoverlapping::<T>(src, dst, len);
                } else {
                    ptr::copy_nonoverlapping::<u8>(
                        src.cast::<u8>(),
                        dst.cast::<u8>(),
                        len * size_of::<T>(),
                    );
                }
            } else {
                for i in 0..len {
                    let s = if SRC_CONTIGUOUS {
                        src.add(i)
                    } else {
                        src.byte_add(i * src_stride)
                    };
                    let d = if DST_CONTIGUOUS {
                        dst.add(i)
                    } else {
                        dst.byte_add(i * dst_stride)
                    };
                    let val = s.read_maybe_aligned::<ALIGNED>();
                    d.write_maybe_aligned::<ALIGNED>(val);
                }
            }
        }
    }

    unsafe fn inner_loop_contiguous_nontemporal<T: Copy>(
        src: PtrNoalias<T>,
        dst: PtrMutNoalias<T>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        debug_assert_eq!(src_stride, size_of::<T>());
        debug_assert_eq!(dst_stride, size_of::<T>());
        unsafe {
            copy_nonoverlapping_nontemporal(
                src.as_ptr().cast::<u8>(),
                dst.as_mut_ptr().cast::<u8>(),
                len * size_of::<T>(),
            )
        };
    }

    unsafe fn inner_loop_contiguous_const_len<T: Copy, const LEN: usize, const ALIGNED: bool>(
        src: PtrNoalias<T>,
        dst: PtrMutNoalias<T>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        debug_assert_eq!(len, LEN);
        debug_assert_eq!(src_stride, size_of::<T>());
        debug_assert_eq!(dst_stride, size_of::<T>());
        let src = src.as_ptr().cast::<[T; LEN]>();
        let dst = dst.as_mut_ptr().cast::<[T; LEN]>();
        let val = unsafe { src.read_maybe_aligned::<ALIGNED>() };
        unsafe { dst.write_maybe_aligned::<ALIGNED>(val) };
    }

    unsafe fn inner_loop_broadcast<T: Copy, const ALIGNED: bool, const DST_CONTIGUOUS: bool>(
        src: PtrNoalias<T>,
        dst: PtrMutNoalias<T>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        debug_assert_eq!(src_stride, 0);
        unsafe { std::hint::assert_unchecked(DST_CONTIGUOUS == (dst_stride == size_of::<T>())) };

        let dst = dst.as_mut_ptr();
        unsafe {
            let val = src.as_ptr().read_maybe_aligned::<ALIGNED>();
            for i in 0..len {
                let d = if DST_CONTIGUOUS {
                    dst.add(i)
                } else {
                    dst.byte_add(i * dst_stride)
                };
                d.write_maybe_aligned::<ALIGNED>(val);
            }
        }
    }

    unsafe fn inner_loop_untyped<const SRC_DST_CONTIGUOUS: bool>(
        src: PtrNoalias<u8>,
        dst: PtrMutNoalias<u8>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        itemsize: usize,
    ) {
        if SRC_DST_CONTIGUOUS {
            debug_assert_eq!(src_stride, itemsize);
            debug_assert_eq!(dst_stride, itemsize);
        }
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        unsafe {
            if SRC_DST_CONTIGUOUS {
                ptr::copy_nonoverlapping(src, dst, len * itemsize);
            } else {
                for i in 0..len {
                    let s = src.byte_add(i * src_stride);
                    let d = dst.byte_add(i * dst_stride);
                    ptr::copy_nonoverlapping(s, d, itemsize);
                }
            }
        }
    }

    unsafe fn inner_loop_untyped_nontemporal(
        src: PtrNoalias<u8>,
        dst: PtrMutNoalias<u8>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        itemsize: usize,
    ) {
        debug_assert_eq!(src_stride, itemsize);
        debug_assert_eq!(dst_stride, itemsize);
        unsafe { copy_nonoverlapping_nontemporal(src.as_ptr(), dst.as_mut_ptr(), len * itemsize) };
    }

    unsafe fn inner_loop_untyped_broadcast(
        src: PtrNoalias<u8>,
        dst: PtrMutNoalias<u8>,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        itemsize: usize,
    ) {
        debug_assert_eq!(src_stride, 0);
        debug_assert_eq!(dst_stride, itemsize);
        if len == 0 {
            return;
        }
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        // Write the element once, then keep doubling the filled prefix, so the run takes
        // `log2(len)` copies rather than `len` of them.
        let total = len * itemsize;
        unsafe { ptr::copy_nonoverlapping(src, dst, itemsize) };
        let mut filled = itemsize;
        while filled < total {
            let n = filled.min(total - filled);
            unsafe { ptr::copy_nonoverlapping(dst, dst.add(filled), n) };
            filled += n;
        }
    }

    unsafe fn copy_untyped(args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            mut dst,
            shape,
            src_strides,
            dst_strides,
            dtype,
            nontemporal,
        } = args;
        let itemsize = dtype.itemsize() as usize;
        let pick_inner_loop = |src_stride: usize, dst_stride: usize, len: usize| {
            let dst_contiguous = dst_stride == itemsize;
            if dst_contiguous && src_stride == itemsize {
                if nontemporal && len * itemsize >= NONTEMPORAL_MIN_RUN_BYTES {
                    return Self::inner_loop_untyped_nontemporal
                        as unsafe fn(
                            PtrNoalias<u8>,
                            PtrMutNoalias<u8>,
                            usize,
                            usize,
                            usize,
                            usize,
                        );
                }
                Self::inner_loop_untyped::<true>
            } else if dst_contiguous && src_stride == 0 {
                Self::inner_loop_untyped_broadcast
            } else {
                Self::inner_loop_untyped::<false>
            }
        };

        if shape.len() == 1 {
            // Skip the iterator for single dim arrays

            let len = shape[0];
            if len == 1 {
                unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), itemsize) };
                return;
            }
            let src_stride = src_strides[0];
            let dst_stride = dst_strides[0];

            let inner_loop_fn = pick_inner_loop(src_stride, dst_stride, len);
            unsafe { inner_loop_fn(src, dst, len, src_stride, dst_stride, itemsize) };
            return;
        }

        // Byte-wise fallback: each element is `itemsize` opaque bytes copied via `copy_nonoverlapping`
        // (always the unaligned `u8` path). Operand 0 is the destination, operand 1 the source.
        let iter = NdIterUnordered::new(
            shape,
            [dst_strides, src_strides],
            [(dtype.itemsize(), Alignment::of::<u8>()); 2],
        );
        let [dst_inner_stride, src_inner_stride] = iter.inner_strides();
        let inner_loop_fn = pick_inner_loop(src_inner_stride, dst_inner_stride, iter.inner_len());
        // `move` for the same reason as in `scalar_fn`.
        iter.foreach_inner_1d_impl::<true>(
            move |[dst_offset, src_offset], len, [dst_stride, src_stride]| unsafe {
                inner_loop_fn(
                    src.bytes_offset(src_offset),
                    dst.bytes_offset(dst_offset),
                    len,
                    src_stride,
                    dst_stride,
                    itemsize,
                )
            },
        );
    }
}

/// Write `n` copies of the `itemsize`-byte element at `src` contiguously into `dst`.
pub(crate) unsafe fn fill_contiguous(src: *const u8, dst: *mut u8, n: usize, itemsize: usize) {
    let (src, dst) = unsafe {
        (
            PtrNoalias::new(src, itemsize),
            PtrMutNoalias::new(dst, n * itemsize),
        )
    };
    unsafe {
        match itemsize {
            1 => NdCopier::inner_loop_broadcast::<u8, false, true>(src, dst, n, 0, 1),
            2 => {
                NdCopier::inner_loop_broadcast::<u16, false, true>(src.cast(), dst.cast(), n, 0, 2)
            }
            4 => {
                NdCopier::inner_loop_broadcast::<u32, false, true>(src.cast(), dst.cast(), n, 0, 4)
            }
            8 => {
                NdCopier::inner_loop_broadcast::<u64, false, true>(src.cast(), dst.cast(), n, 0, 8)
            }
            16 => NdCopier::inner_loop_broadcast::<[u64; 2], false, true>(
                src.cast(),
                dst.cast(),
                n,
                0,
                16,
            ),
            _ => NdCopier::inner_loop_untyped_broadcast(src, dst, n, 0, itemsize, itemsize),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtyped;

    // A struct dtype with no padding (itemsize 12, alignment 4). It misses every
    // specialized (itemsize, alignment) case in `create_scalar_fn`, so `NdCopier`
    // takes the `copy_struct` route. Because there is no padding between or after
    // fields, copying each field is equivalent to copying the whole element - which
    // is exactly what the byte-level `reference_copy` does, so the two agree.
    #[derive(Copy, Clone, crate::dtype::Dtyped)]
    #[dtyped(crate = "crate")]
    #[repr(C)]
    struct StructNoPad {
        a: i32,
        b: i16,
        c: i16,
        d: i32,
    }

    // Byte strides for a row-major array whose backing shape is `shape[d] * mult[d]`,
    // with logical elements sampled every `mult[d]` slots along dim `d`. `mult` all
    // ones gives a fully contiguous layout; any `mult[d] > 1` leaves gaps, exercising
    // the strided (non-coalesced) path - including a strided innermost dim.
    fn strided_strides(shape: &[usize], mult: &[usize], itemsize: usize) -> Vec<usize> {
        let ndim = shape.len();
        let backing = (0..ndim).map(|d| shape[d] * mult[d]).collect::<Vec<_>>();
        let mut cstr = vec![0usize; ndim];
        let mut acc = itemsize;
        for d in (0..ndim).rev() {
            cstr[d] = acc;
            acc *= backing[d];
        }
        (0..ndim).map(|d| cstr[d] * mult[d]).collect()
    }

    // Number of bytes needed to hold every element of `shape` at the given byte strides.
    fn buf_len(shape: &[usize], strides: &[usize], itemsize: usize) -> usize {
        if shape.contains(&0) {
            return 0;
        }
        let max_off: usize = shape
            .iter()
            .zip(strides)
            .map(|(&s, &st)| (s - 1) * st)
            .sum();
        max_off + itemsize
    }

    // Independent, naive nd copy over raw bytes: for every element (row-major order)
    // copy `itemsize` bytes from its source offset to its destination offset. Bytes in
    // `dst` that no element maps to are left zero, so the assert also catches any write
    // outside the intended region (e.g. into a strided gap).
    fn reference_copy(
        src: &[u8],
        dst_len: usize,
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        itemsize: usize,
    ) -> Vec<u8> {
        let ndim = shape.len();
        let mut dst = vec![0u8; dst_len];
        let total: usize = shape.iter().product();
        let mut idx = vec![0usize; ndim];
        for _ in 0..total {
            let src_off: usize = (0..ndim).map(|d| idx[d] * src_strides[d]).sum();
            let dst_off: usize = (0..ndim).map(|d| idx[d] * dst_strides[d]).sum();
            dst[dst_off..dst_off + itemsize].copy_from_slice(&src[src_off..src_off + itemsize]);
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        dst
    }

    // Owns a source/destination byte buffer and hands out a byte pointer into it. The aligned
    // variant (`misalign == 0`) is a `Vec<T>`, naturally aligned to `align_of::<T>()` - exactly what
    // the aligned copy path needs. The unaligned variant is a `Vec<u8>` shifted one byte past an
    // aligned offset, forcing the unaligned path for any dtype with alignment > 1. Both are only ever
    // touched as raw bytes, so no `T` is constructed and the `Vec<T>` (len 0) drops nothing.
    enum TestBuf<T> {
        Aligned(Vec<T>),
        Unaligned(Vec<u8>),
    }
    impl<T> TestBuf<T> {
        fn new(len: usize, misalign: usize) -> Self {
            if misalign == 0 {
                Self::Aligned(Vec::with_capacity(len.div_ceil(size_of::<T>()) + 1))
            } else {
                Self::Unaligned(vec![0u8; len + align_of::<T>() + 1])
            }
        }
        fn ptr(&mut self) -> *mut u8 {
            match self {
                Self::Aligned(v) => v.as_mut_ptr().cast::<u8>(),
                Self::Unaligned(v) => {
                    let base = v.as_mut_ptr();
                    let pad = base.align_offset(align_of::<T>());
                    assert_ne!(pad, usize::MAX, "cannot align within the buffer");
                    // SAFETY: `pad < align_of::<T>()` and the buffer was sized with that much
                    // headroom plus one byte for the misalignment shift, so this stays in bounds.
                    unsafe { base.add(pad + 1) }
                }
            }
        }
    }

    // A `strided_strides` multiplier vector that leaves a one-element gap only along `dim`
    // (all ones - fully contiguous - when `dim` is out of range, e.g. for a 0-D shape).
    fn strided_dim(ndim: usize, dim: usize) -> Vec<usize> {
        (0..ndim).map(|d| if d == dim { 2 } else { 1 }).collect()
    }

    // Run a single copy through `NdCopier` for element type `T` at a chosen base alignment
    // (`misalign == 0` aligned, `misalign == 1` deliberately misaligned) and assert it byte-matches
    // the independent reference.
    fn check<T: Dtyped>(
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        misalign: usize,
    ) {
        let dtype = T::DTYPE;
        let itemsize = dtype.itemsize() as usize;
        assert_eq!(src_strides.len(), shape.len());
        assert_eq!(dst_strides.len(), shape.len());

        let src_len = buf_len(shape, src_strides, itemsize);
        let dst_len = buf_len(shape, dst_strides, itemsize);

        let mut src_buf = TestBuf::<T>::new(src_len, misalign);
        let mut dst_buf = TestBuf::<T>::new(dst_len, misalign);
        let src_ptr = src_buf.ptr();
        let dst_ptr = dst_buf.ptr();

        // Fill the source with a deterministic byte pattern; zero the destination so that gaps the
        // copy never writes match the reference's zero-filled gaps. Both go through raw bytes, so
        // the aligned `Vec<T>`'s uninitialized capacity is only ever accessed as bytes, never `T`.
        for i in 0..src_len {
            unsafe {
                src_ptr
                    .add(i)
                    .write((i as u8).wrapping_mul(37).wrapping_add(11))
            };
        }
        unsafe { dst_ptr.write_bytes(0, dst_len) };

        let src = unsafe { std::slice::from_raw_parts(src_ptr, src_len) };
        let expected = reference_copy(src, dst_len, shape, src_strides, dst_strides, itemsize);

        // Regular and non-temporal stores must write the same bytes.
        for nontemporal in [false, true] {
            unsafe { dst_ptr.write_bytes(0, dst_len) };
            let copier = NdCopier::new(&dtype).nontemporal(nontemporal);
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) };
            unsafe {
                copier.copy(
                    PtrNoalias::from_slice(src),
                    PtrMutNoalias::from_slice(dst),
                    shape,
                    src_strides,
                    dst_strides,
                    &dtype,
                );
            }
            crate::util::nontemporal_fence();

            let actual = unsafe { std::slice::from_raw_parts(dst_ptr, dst_len) };
            assert_eq!(
                actual,
                expected.as_slice(),
                "itemsize={itemsize} misalign={misalign} nontemporal={nontemporal} shape={shape:?} \
                 src_strides={src_strides:?} dst_strides={dst_strides:?}"
            );
        }
    }

    // For a fixed (element type, shape), exercise a representative set of source/dst layouts, each
    // aligned and deliberately misaligned:
    //   - contiguous/contiguous: fully coalesces into a single run;
    //   - strided-outer/contiguous: inner dims coalesce, driving the 1D fast path;
    //   - strided-inner/strided-inner: nothing coalesces, driving the general strided nd iterator.
    // Gaps stay on a single dim, so buffers never exceed twice the contiguous size - small enough
    // to run the whole matrix under Miri.
    fn check_rank<T: Dtyped>(shape: &[usize]) {
        let itemsize = T::DTYPE.itemsize() as usize;
        let ndim = shape.len();
        let cont = strided_strides(shape, &vec![1; ndim], itemsize);
        let outer = strided_strides(shape, &strided_dim(ndim, 0), itemsize);
        let inner = strided_strides(shape, &strided_dim(ndim, ndim.wrapping_sub(1)), itemsize);
        for (src, dst) in [(&cont, &cont), (&outer, &cont), (&inner, &inner)] {
            for misalign in [0, 1] {
                check::<T>(shape, src, dst, misalign);
            }
        }
    }

    // Ranks 0 through 8 (the maximum) for one element type. The trailing `[3, k]` shapes put a
    // contiguous inner run of length k behind a strided outer dim, so the run coalesces to
    // `n_contiguous == k` and drives each specialized `copy_1d` arm (k in 1/2/4/8/16/32/64, subject
    // to the size guards) plus the `_` fallback (k = 3).
    fn check_all_dims<T: Dtyped>() {
        check_rank::<T>(&[]);
        check_rank::<T>(&[7]);
        check_rank::<T>(&[3, 4]);
        check_rank::<T>(&[2, 3, 4]);
        check_rank::<T>(&[2, 2, 3, 3]);
        check_rank::<T>(&[2, 2, 2, 3, 3]);
        check_rank::<T>(&[2, 2, 2, 2, 3, 3]);
        check_rank::<T>(&[2, 2, 2, 2, 2, 2, 3]);
        check_rank::<T>(&[2, 2, 2, 2, 2, 2, 2, 2]);

        for k in [1, 2, 3, 4, 8, 16, 32, 64] {
            check_rank::<T>(&[3, k]);
        }
    }

    // Zero-length dimensions: the copy is a no-op (the empty-region guard in `copy` fires before
    // any element or struct-field pointer is computed), so the zeroed destination must stay zeroed.
    fn check_zero_extent<T: Dtyped>() {
        let itemsize = T::DTYPE.itemsize() as usize;
        for shape in [vec![0usize], vec![3, 0], vec![2, 0, 4]] {
            let strides = strided_strides(&shape, &vec![1; shape.len()], itemsize);
            for misalign in [0, 1] {
                check::<T>(&shape, &strides, &strides, misalign);
            }
        }
    }

    // Randomized coverage via `proptest`: deterministic per element type (seeded from the itemsize),
    // exploring random ranks, extents, per-dim strides and base alignment against the reference.
    // Buffers are bounded so every draw stays cheap enough to also run under Miri.
    fn fuzz<T: Dtyped>() {
        use proptest::prelude::*;
        use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

        let itemsize = T::DTYPE.itemsize() as usize;
        let strategy = (0usize..=6).prop_flat_map(|ndim| {
            (
                prop::collection::vec(1usize..=3, ndim),
                prop::collection::vec(1usize..=3, ndim),
                prop::collection::vec(1usize..=3, ndim),
                0usize..=1,
            )
        });
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&(0xC0FF_EE12_3456_789A_u64 ^ itemsize as u64).to_le_bytes());
        let mut runner = TestRunner::new_with_rng(
            Config {
                cases: 64,
                ..Config::default()
            },
            TestRng::from_seed(RngAlgorithm::ChaCha, &seed),
        );
        runner
            .run(&strategy, |(shape, src_mult, dst_mult, misalign)| {
                let src_strides = strided_strides(&shape, &src_mult, itemsize);
                let dst_strides = strided_strides(&shape, &dst_mult, itemsize);
                if buf_len(&shape, &src_strides, itemsize).max(buf_len(
                    &shape,
                    &dst_strides,
                    itemsize,
                )) > 8 * 1024
                {
                    return Ok(());
                }
                check::<T>(&shape, &src_strides, &dst_strides, misalign);
                Ok(())
            })
            .unwrap();
    }

    // Generates one `copy_<name>` test per dtype, each running the full rank/layout/alignment
    // matrix. One dtype is tested per distinct `scalar_fn` specialization - dtypes that share a
    // byte layout (e.g. `i32`/`f32` both hit the `u32` path) would exercise identical copy code.
    macro_rules! copy_tests {
        ( $( $(#[$m:meta])* $name:ident : $ty:ty ),* $(,)? ) => {
            $( paste::paste! {
                $(#[$m])*
                #[test]
                fn [<copy_ $name>]() { check_all_dims::<$ty>(); }
            } )*
        };
    }

    copy_tests! {
        u8: u8,                                                        // (1, 1) -> u8
        u16: u16,                                                      // (2, 2) -> u16
        u32: u32,                                                      // (4, 4) -> u32
        u64: u64,                                                      // (8, 8) -> u64
        #[cfg(feature = "half")] f16: crate::scalar::f16,              // (2, 2) -> u16
        #[cfg(feature = "num-complex")] complex_f32: crate::scalar::Complex<f32>, // (8, 4) -> [u32; 2]
        #[cfg(feature = "num-complex")] complex_f64: crate::scalar::Complex<f64>, // (16, 8) -> [u64; 2]
    }

    // A unit (size-1) dim is iterated once at index 0, so its stride never contributes an offset;
    // the coalescing scan treats it as contiguous whatever stride it carries and never lets it block
    // neighboring dims from merging. Each layout gives a unit dim a stride that would otherwise halt
    // the scan (a broadcast `0` and an arbitrary multiple), with contiguous dims around it, covering
    // both the scalar (`scalar_fn`) and byte-wise (`copy_untyped`) coalescing routes.
    fn check_unit_dim<T: Dtyped>() {
        let is = T::DTYPE.itemsize() as usize;
        for unit in [0, 5 * is] {
            // interior unit dim: contiguous both sides (full coalesce), then a strided outer dim
            check::<T>(&[4, 1, 8], &[8 * is, unit, is], &[8 * is, unit, is], 0);
            check::<T>(&[4, 1, 8], &[16 * is, unit, is], &[16 * is, unit, is], 0);
        }
        // leading and trailing unit dims, contiguous elsewhere
        check::<T>(&[1, 4, 8], &[7 * is, 8 * is, is], &[7 * is, 8 * is, is], 0);
        check::<T>(&[4, 8, 1], &[8 * is, is, 3 * is], &[8 * is, is, 3 * is], 0);
        check::<T>(&[4, 1, 8], &[8 * is, 0, is], &[8 * is, 0, is], 1); // misaligned base
    }
    #[test]
    fn copy_unit_dim() {
        check_unit_dim::<u32>(); // scalar route
        check_unit_dim::<u64>();
        check_unit_dim::<[i32; 3]>(); // byte-wise copy_untyped route
        check_unit_dim::<[u8; 3]>();
    }

    // The 1D copy splits on per-side contiguity: an outer dim that survives coalescing with its byte
    // stride still equal to the inner run's length is contiguous and is copied as whole blocks (a
    // single `copy_nonoverlapping` when both sides are, a per-side loop when one is, the general loop
    // otherwise). Folding needs *both* strides to match, so at most one side reaches this path
    // contiguous with `len > 1`. An inner run of `inner` behind a strided outer dim pins each arm; a
    // power-of-two `inner` routes through the const-generic `copy_1d`, others through
    // `inner_loop_untyped`. Each combo runs aligned and misaligned so the misaligned arm reaches every
    // per-side contiguity case - in particular the neither-side-contiguous + unaligned branch of
    // `inner_loop_untyped`, which no other deterministic test pins (only the randomized `fuzz` does).
    fn check_1d_sides<T: Dtyped>(inner: usize) {
        let is = T::DTYPE.itemsize() as usize;
        let cont = [inner * is, is]; // outer stride == inner run -> that side contiguous
        let strided = [2 * inner * is, is]; // outer 2x gap -> that side strided
        for (src, dst) in [
            (&cont, &cont),
            (&cont, &strided),
            (&strided, &cont),
            (&strided, &strided),
        ] {
            for misalign in [0, 1] {
                check::<T>(&[4, inner], src, dst, misalign);
            }
        }
    }
    #[test]
    fn copy_1d_sides() {
        check_1d_sides::<u32>(8); // power-of-two run -> const-generic copy_1d arms
        check_1d_sides::<u32>(3); // other run length -> inner_loop_untyped
        check_1d_sides::<u64>(8); // wider element type
        check_1d_sides::<u64>(3);
    }

    // Struct dtype (the `copy_struct` field-by-field route) and non-scalar array dtypes (the
    // `copy_untyped` byte-wise fallback): array-of-scalar dtypes carry an inner shape, so they miss
    // every specialized `(itemsize, alignment)` case and are not decomposed as structs - `[i32; 3]`
    // gives itemsize 12, `[u8; 3]` an odd itemsize (3) with no power-of-two run.
    #[test]
    fn copy_struct() {
        check_all_dims::<StructNoPad>();
    }
    #[test]
    fn copy_dynamic_itemsize_12() {
        check_all_dims::<[i32; 3]>();
    }
    #[test]
    fn copy_dynamic_odd_itemsize() {
        check_all_dims::<[u8; 3]>();
    }

    // A zero source stride repeats one element. Along the innermost axis that is the broadcast loop;
    // along an outer axis it only re-reads whole runs. Each layout runs aligned and misaligned, into
    // a contiguous and a strided destination, in 1-D (the iterator-free path) and in 2-D.
    fn check_broadcast_src<T: Dtyped>() {
        let is = T::DTYPE.itemsize() as usize;
        for misalign in [0, 1] {
            for n in [1, 3, 64, 100] {
                check::<T>(&[n], &[0], &[is], misalign);
                check::<T>(&[n], &[0], &[2 * is], misalign);
            }
            // broadcast along the inner axis
            check::<T>(&[3, 17], &[is, 0], &[17 * is, is], misalign);
            check::<T>(&[3, 17], &[is, 0], &[34 * is, 2 * is], misalign);
            // broadcast along every axis: one element spread over the whole region
            check::<T>(&[3, 17], &[0, 0], &[17 * is, is], misalign);
            check::<T>(&[3, 17], &[0, 0], &[34 * is, 2 * is], misalign);
            // broadcast along the outer axis only: the same contiguous run, repeated
            check::<T>(&[3, 17], &[0, is], &[17 * is, is], misalign);
        }
    }
    #[test]
    fn copy_broadcast_src() {
        check_broadcast_src::<u8>();
        check_broadcast_src::<u16>();
        check_broadcast_src::<u32>();
        check_broadcast_src::<u64>();
        #[cfg(feature = "num-complex")]
        check_broadcast_src::<crate::scalar::Complex<f32>>(); // (8, 4) -> [u32; 2]
        #[cfg(feature = "num-complex")]
        check_broadcast_src::<crate::scalar::Complex<f64>>(); // (16, 8) -> [u64; 2], via `Reg16`
        check_broadcast_src::<StructNoPad>();
        check_broadcast_src::<[i32; 3]>();
        check_broadcast_src::<[u8; 3]>();
    }

    // `fill_contiguous` for every item size it dispatches on, plus odd ones for the byte-wise loop,
    // from a misaligned source into a misaligned destination.
    #[test]
    fn fill_contiguous_repeats_the_element() {
        for itemsize in [1, 2, 3, 4, 8, 12, 16, 24] {
            for n in [1, 3, 64, 100] {
                let elem = (0..itemsize as u8).map(|i| i.wrapping_mul(37).wrapping_add(11));
                let src = std::iter::once(0).chain(elem.clone()).collect::<Vec<u8>>();
                let mut dst = vec![0u8; 1 + n * itemsize + 1];
                unsafe { fill_contiguous(src[1..].as_ptr(), dst[1..].as_mut_ptr(), n, itemsize) };
                let expected = std::iter::once(0)
                    .chain(elem.cycle().take(n * itemsize))
                    .chain(std::iter::once(0))
                    .collect::<Vec<u8>>();
                assert_eq!(dst, expected, "itemsize={itemsize} n={n}");
            }
        }
    }

    // Zero-length dimensions across the scalar, struct and dynamic dispatch routes.
    #[test]
    fn copy_zero_extent() {
        check_zero_extent::<u8>();
        check_zero_extent::<u64>();
        #[cfg(feature = "num-complex")]
        check_zero_extent::<crate::scalar::Complex<f64>>();
        check_zero_extent::<StructNoPad>();
        check_zero_extent::<[u8; 3]>();
    }

    // Randomized fuzzing over one dtype per dispatch route (scalar, struct, byte-wise fallback).
    #[test]
    fn fuzz_random() {
        fuzz::<u32>();
        fuzz::<u64>();
        #[cfg(feature = "num-complex")]
        fuzz::<crate::scalar::Complex<f64>>();
        fuzz::<StructNoPad>();
        fuzz::<[u8; 3]>();
    }

    // Strides that nest the dims in `order` (outermost first), leaving `mult - 1` empty slots after
    // every element along every dim (`mult == 1` is packed).
    fn order_strides(shape: &[usize], order: &[usize], itemsize: usize, mult: usize) -> Vec<usize> {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = itemsize;
        for &d in order.iter().rev() {
            strides[d] = acc * mult;
            acc *= shape[d] * mult;
        }
        strides
    }

    // A copy between two layouts given by their dim orders, over a region big enough for the
    // blocked path, checked against the naive reference. The source bytes come from a hash of
    // their offset - `check`'s pattern repeats every 256 bytes, which a tile landing a whole
    // number of periods off would get past. `blocked` asserts which path the copy takes.
    fn check_orders<T: Dtyped>(
        shape: &[usize],
        src_order: &[usize],
        dst_order: &[usize],
        (src_mult, dst_mult): (usize, usize),
        misalign: usize,
        blocked: bool,
    ) {
        let dtype = T::DTYPE;
        let itemsize = dtype.itemsize() as usize;
        let src_strides = order_strides(shape, src_order, itemsize, src_mult);
        let dst_strides = order_strides(shape, dst_order, itemsize, dst_mult);
        let nbytes = shape.iter().product::<usize>() * itemsize;
        let takes_blocked_path = nbytes >= BLOCKED_COPY_MIN_BYTES
            && !blocked_copy_split_dims(shape, &src_strides, &dst_strides, itemsize).is_empty();
        assert_eq!(takes_blocked_path, blocked, "shape={shape:?}");

        let src_len = buf_len(shape, &src_strides, itemsize);
        let dst_len = buf_len(shape, &dst_strides, itemsize);
        let mut src_buf = TestBuf::<T>::new(src_len, misalign);
        let mut dst_buf = TestBuf::<T>::new(dst_len, misalign);
        let (src_ptr, dst_ptr) = (src_buf.ptr(), dst_buf.ptr());
        for i in 0..src_len {
            let h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            unsafe { src_ptr.add(i).write((h >> 56) as u8) };
        }
        unsafe { dst_ptr.write_bytes(0, dst_len) };

        let src = unsafe { std::slice::from_raw_parts(src_ptr, src_len) };
        let expected = reference_copy(src, dst_len, shape, &src_strides, &dst_strides, itemsize);
        for nontemporal in [false, true] {
            unsafe { dst_ptr.write_bytes(0, dst_len) };
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) };
            unsafe {
                NdCopier::new(&dtype).nontemporal(nontemporal).copy(
                    PtrNoalias::from_slice(src),
                    PtrMutNoalias::from_slice(dst),
                    shape,
                    &src_strides,
                    &dst_strides,
                    &dtype,
                );
            }
            crate::util::nontemporal_fence();
            let actual = unsafe { std::slice::from_raw_parts(dst_ptr, dst_len) };
            assert!(
                actual == expected.as_slice(),
                "itemsize={itemsize} misalign={misalign} nontemporal={nontemporal} shape={shape:?} \
                 src_order={src_order:?} dst_order={dst_order:?} mult={:?}",
                (src_mult, dst_mult)
            );
        }
    }

    // Big transposed copies take the blocked path: 2D transposes whose extents are not multiples
    // of the tile, each dispatch route, aligned and not, packed and gapped.
    #[test]
    #[cfg_attr(miri, ignore)] // megabyte-sized regions
    fn blocked_copy_2d_transpose() {
        for misalign in [0, 1] {
            check_orders::<u32>(&[300, 1100], &[0, 1], &[1, 0], (1, 1), misalign, true);
            check_orders::<u32>(&[1100, 300], &[1, 0], &[0, 1], (1, 1), misalign, true);
            check_orders::<u64>(&[1030, 257], &[0, 1], &[1, 0], (1, 1), misalign, true);
            check_orders::<u8>(&[2000, 600], &[0, 1], &[1, 0], (1, 1), misalign, true);
            check_orders::<u16>(&[4099, 131], &[0, 1], &[1, 0], (1, 1), misalign, true);
            check_orders::<StructNoPad>(&[300, 400], &[0, 1], &[1, 0], (1, 1), misalign, true);
            check_orders::<[u8; 3]>(&[700, 600], &[0, 1], &[1, 0], (1, 1), misalign, true);
        }
        // Power-of-two extents: the strides that alias in the cache.
        check_orders::<u32>(&[1024, 512], &[0, 1], &[1, 0], (1, 1), 0, true);
        // Gaps in the source, in the destination, in both.
        check_orders::<u32>(&[300, 1100], &[0, 1], &[1, 0], (2, 1), 0, true);
        check_orders::<u32>(&[300, 1100], &[0, 1], &[1, 0], (1, 2), 0, true);
        check_orders::<u64>(&[500, 300], &[0, 1], &[1, 0], (3, 2), 1, true);
        // A very thin region: one dim shorter than a tile's run.
        check_orders::<u32>(&[3, 100_000], &[0, 1], &[1, 0], (1, 1), 0, true);
        check_orders::<u32>(&[100_000, 3], &[0, 1], &[1, 0], (1, 1), 0, true);
    }

    // Every way two layouts of a 3D / 4D region can disagree, including size-1 dims.
    #[test]
    #[cfg_attr(miri, ignore)] // megabyte-sized regions
    fn blocked_copy_nd_permutations() {
        let shape = [70usize, 90, 60];
        let orders: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for src_order in orders {
            for dst_order in orders {
                // Blocked exactly when the orders differ: the longest shared inner run here is
                // 60 * 4 bytes, short of the size that skips blocking.
                let blocked = src_order != dst_order;
                check_orders::<u32>(&shape, &src_order, &dst_order, (1, 1), 0, blocked);
            }
        }
        check_orders::<u16>(
            &[20, 30, 25, 41],
            &[0, 1, 2, 3],
            &[3, 2, 1, 0],
            (1, 1),
            0,
            true,
        );
        check_orders::<u16>(
            &[20, 30, 25, 41],
            &[0, 1, 2, 3],
            &[2, 3, 0, 1],
            (1, 1),
            1,
            true,
        );
        check_orders::<u64>(
            &[1, 300, 1, 500],
            &[0, 1, 2, 3],
            &[3, 2, 1, 0],
            (1, 1),
            0,
            true,
        );
        check_orders::<u64>(
            &[300, 1, 500, 1],
            &[0, 1, 2, 3],
            &[2, 0, 3, 1],
            (1, 1),
            0,
            true,
        );
    }

    // Copies that must NOT be blocked: agreeing layouts, a long run shared by both layouts, a
    // region under the size threshold.
    #[test]
    #[cfg_attr(miri, ignore)] // megabyte-sized regions
    fn blocked_copy_is_skipped_when_it_cannot_help() {
        check_orders::<u32>(&[300, 1100], &[0, 1], &[0, 1], (1, 1), 0, false);
        check_orders::<u32>(&[300, 1100], &[0, 1], &[0, 1], (2, 1), 0, false);
        // Dims 0 and 1 swap, but dim 2 is an 8000-byte run in both layouts.
        check_orders::<u32>(&[16, 16, 2000], &[0, 1, 2], &[1, 0, 2], (1, 1), 0, false);
        // Transposed, but small.
        check_orders::<u32>(&[100, 100], &[0, 1], &[1, 0], (1, 1), 0, false);
    }
}
