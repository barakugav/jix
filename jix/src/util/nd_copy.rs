use crate::arrayvec::ArrayVec;
use crate::dtype::{Dtype, Itemsize};
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::iter::NdIter;
use crate::{dim_arr, Dim, DimArray, Dimension, SliceExt};

/// A reusable, dtype-specialized copier that moves a rectangular n-dimensional region between two
/// raw byte buffers under independent source and destination strides.
///
/// [`new`](Self::new) inspects the dtype once and picks the cheapest copy routine up front: a
/// monomorphized scalar copy for the common power-of-two `(itemsize, alignment)` pairs, a
/// field-by-field copy for structs that decompose into at most four scalar fields, or a generic
/// byte-wise fallback for everything else. Each [`copy`](Self::copy) then moves one region by
/// walking `shape` with an [`NdIter`] and copying the appropriate number of bytes at each element.
pub(crate) struct NdCopier<'a>(NdCopierInner<'a>);
enum NdCopierInner<'a> {
    Simple(NdCopyFn),
    Struct(NdCopierStruct<'a>),
}
type NdCopyFn = fn(NdCopyArgs);
struct NdCopierStruct<'a> {
    scalar_fns: ArrayVec<NdCopyFn, 4>,
    offsets: ArrayVec<Itemsize, 4>,
    dtypes: ArrayVec<&'a Dtype, 4>,
}
struct NdCopyArgs<'a> {
    src: &'a [u8],
    dst: &'a mut [u8],
    shape: &'a [usize],
    src_strides: &'a [usize],
    dst_strides: &'a [usize],
    dtype: &'a Dtype,
}

impl<'a> NdCopier<'a> {
    #[inline(always)]
    pub(crate) fn new(dtype: &'a Dtype) -> Self {
        match Self::create_scalar_fn(dtype) {
            Some(f) => Self(NdCopierInner::Simple(f)),
            None => Self::new_slow(dtype),
        }
    }

    fn new_slow(dtype: &'a Dtype) -> Self {
        #[inline]
        fn collect_struct_fields<'a>(
            mut copier: NdCopierStruct<'a>,
            offset: Itemsize,
            dtype: &'a Dtype,
        ) -> Option<NdCopierStruct<'a>> {
            if dtype.try_to_scalar().is_some() {
                if copier.scalar_fns.is_full() {
                    return None;
                }
                let scalar_fn = NdCopier::create_scalar_fn(dtype).unwrap();
                copier.scalar_fns.push(scalar_fn);
                copier.offsets.push(offset);
                copier.dtypes.push(dtype);
                Some(copier)
            } else if !dtype.shape().is_empty() {
                None
            } else if let Some(fields) = dtype.fields() {
                #[inline(never)] // make collect_struct_fields inlinable
                fn collect_struct_fields_inner<'a>(
                    copier: NdCopierStruct<'a>,
                    offset: Itemsize,
                    dtype: &'a Dtype,
                ) -> Option<NdCopierStruct<'a>> {
                    collect_struct_fields(copier, offset, dtype)
                }

                for (_f_name, field_offset, field_dtype) in fields {
                    let field_offset = offset + field_offset;
                    copier = collect_struct_fields_inner(copier, field_offset, field_dtype)?;
                }
                Some(copier)
            } else {
                None
            }
        }

        let struct_copier = NdCopierStruct {
            scalar_fns: ArrayVec::new(),
            offsets: ArrayVec::new(),
            dtypes: ArrayVec::new(),
        };
        let struct_copier = collect_struct_fields(struct_copier, 0, dtype);

        Self(if let Some(struct_copier) = struct_copier {
            NdCopierInner::Struct(struct_copier)
        } else {
            NdCopierInner::Simple(Self::copy_dynamic)
        })
    }

    #[inline(always)]
    pub(crate) unsafe fn copy(
        &self,
        src: *const u8,
        dst: *mut u8,
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        dtype: &Dtype,
    ) {
        if shape.contains(&0) {
            return;
        }
        // Reorder axes so the destination's strides are non-increasing (largest axis outermost,
        // smallest - most contiguous - innermost). This turns a scattered-write transpose into a
        // sequential-write copy and lets the coalescing scans below merge more runs; it is a
        // no-op for the already-C-order layouts that dominate.
        let dim_perm = compute_dim_permutation(dst_strides);
        let shape_permuted;
        let src_strides_permuted;
        let dst_strides_permuted;
        let (shape, src_strides, dst_strides) = if let Some(dim_perm) = dim_perm {
            shape_permuted = apply_dim_permutation(shape, &dim_perm);
            src_strides_permuted = apply_dim_permutation(src_strides, &dim_perm);
            dst_strides_permuted = apply_dim_permutation(dst_strides, &dim_perm);
            (
                shape_permuted.as_slice(),
                src_strides_permuted.as_slice(),
                dst_strides_permuted.as_slice(),
            )
        } else {
            (shape, src_strides, dst_strides)
        };

        // The internal pipeline works over slices (so the reference-derived pointers carry
        // `noalias`), but the public API is raw pointers. Carry each pointer as a zero-length
        // slice beginning at it: the length is never read (callers pass an explicit `shape`),
        // the pointer is recovered via `as_ptr`/`as_mut_ptr`, and a zero-length `u8` slice is
        // always well-aligned. SAFETY: `copy`'s contract already requires `src`/`dst` to be
        // valid, non-overlapping regions for the described `shape`/strides.
        let args = NdCopyArgs {
            src: unsafe { std::slice::from_raw_parts(src, 0) },
            dst: unsafe { std::slice::from_raw_parts_mut(dst, 0) },
            shape,
            src_strides,
            dst_strides,
            dtype,
        };
        match &self.0 {
            NdCopierInner::Simple(f) => f(args),
            NdCopierInner::Struct(struct_copier) => Self::copy_struct(struct_copier, args),
        }
    }

    #[inline]
    const fn create_scalar_fn(dtype: &Dtype) -> Option<fn(NdCopyArgs)> {
        Some(match (dtype.itemsize(), dtype.alignment().as_usize()) {
            (1, 1) => Self::scalar_fn::<u8>,
            (2, 2) => Self::scalar_fn::<u16>,
            (4, 4) => Self::scalar_fn::<u32>,
            (8, 4) => Self::scalar_fn::<[u32; 2]>,
            (8, 8) => Self::scalar_fn::<u64>,
            (16, 8) => Self::scalar_fn::<[u64; 2]>,
            _ => return None,
        })
    }

    fn scalar_fn<T: Copy + 'static>(args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            dst,
            shape,
            src_strides, // TODO accept Option<>,
            dst_strides,
            dtype,
        } = args;
        debug_assert_eq!(size_of::<T>(), dtype.itemsize() as usize);
        debug_assert_eq!(align_of::<T>(), dtype.alignment().as_usize());

        // Collapse the axes: drop size-1 axes, merge stride-compatible neighbors, and peel the
        // trailing both-contiguous run as `n_contiguous_items`. What remains are strided axes.
        let merged = merge_dims(shape, src_strides, dst_strides, size_of::<T>());
        let mut shape = merged.shape.as_slice();
        let mut src_strides = merged.src_strides.as_slice();
        let mut dst_strides = merged.dst_strides.as_slice();
        let n_contiguous_items = merged.n_contiguous_items;
        if shape.is_empty() {
            shape = &[1];
            src_strides = &[0];
            dst_strides = &[0];
        }

        if shape.len() > 1 {
            return unsafe {
                Self::copy_nd::<T>(
                    shape,
                    src_strides,
                    dst_strides,
                    src,
                    dst,
                    n_contiguous_items,
                )
            };
        }
        // 1D copy

        let len = shape[0];
        let src_stride = src_strides[0];
        let dst_stride = dst_strides[0];

        let aligned = (src.as_ptr().cast::<T>().is_aligned()
            && src_stride.is_multiple_of(align_of::<T>()))
            && (dst.as_ptr().cast::<T>().is_aligned()
                && dst_stride.is_multiple_of(align_of::<T>()));

        unsafe {
            Self::copy_1d_dispatch::<T>(
                src,
                dst,
                len,
                src_stride,
                dst_stride,
                n_contiguous_items,
                aligned,
            )
        }
    }

    /// Copy a single strided axis of `len` runs, each `n_contiguous_items` contiguous elements,
    /// selecting the const-generic [`copy_1d`](Self::copy_1d) arm for the common run sizes and
    /// falling back to [`copy_1d_unsized`](Self::copy_1d_unsized) otherwise. `src`/`dst` carry the
    /// run's base pointer (as zero-length slices; see [`copy`](Self::copy)). Shared by the 1D fast
    /// path and by `copy_nd`'s peeled innermost axis.
    #[inline(always)]
    unsafe fn copy_1d_dispatch<T: Copy>(
        src: &[u8],
        dst: &mut [u8],
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        n_contiguous_items: usize,
        aligned: bool,
    ) {
        unsafe {
            match n_contiguous_items {
                1 => Self::copy_1d::<T, 1>(src, dst, len, src_stride, dst_stride, aligned),
                2 => Self::copy_1d::<T, 2>(src, dst, len, src_stride, dst_stride, aligned),
                4 => Self::copy_1d::<T, 4>(src, dst, len, src_stride, dst_stride, aligned),
                8 => Self::copy_1d::<T, 8>(src, dst, len, src_stride, dst_stride, aligned),
                16 => Self::copy_1d::<T, 16>(src, dst, len, src_stride, dst_stride, aligned),
                32 if size_of::<T>() <= 8 => {
                    Self::copy_1d::<T, 32>(src, dst, len, src_stride, dst_stride, aligned)
                }
                64 if size_of::<T>() <= 4 => {
                    Self::copy_1d::<T, 64>(src, dst, len, src_stride, dst_stride, aligned)
                }
                _ => Self::copy_1d_unsized::<T>(
                    src,
                    dst,
                    len,
                    src_stride,
                    dst_stride,
                    n_contiguous_items,
                    aligned,
                ),
            }
        }
    }

    #[inline(always)]
    unsafe fn copy_1d<T: Copy, const N_CONTIGUOUS: usize>(
        src: &[u8],
        dst: &mut [u8],
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        aligned: bool,
    ) {
        if aligned {
            unsafe {
                Self::copy_1d_aligned::<T, N_CONTIGUOUS>(src, dst, len, src_stride, dst_stride)
            }
        } else {
            unsafe {
                Self::copy_1d_unaligned::<T, N_CONTIGUOUS>(src, dst, len, src_stride, dst_stride)
            }
        }
    }
    #[inline]
    unsafe fn copy_1d_aligned<T: Copy, const N_CONTIGUOUS: usize>(
        src: &[u8],
        dst: &mut [u8],
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let src_contiguous = len <= 1 || src_stride == size_of::<T>() * N_CONTIGUOUS;
        let dst_contiguous = len <= 1 || dst_stride == size_of::<T>() * N_CONTIGUOUS;
        if src_contiguous && dst_contiguous {
            unsafe {
                std::ptr::copy_nonoverlapping::<[T; N_CONTIGUOUS]>(
                    src.cast::<[T; N_CONTIGUOUS]>(),
                    dst.cast::<[T; N_CONTIGUOUS]>(),
                    len,
                )
            };
        } else if src_contiguous {
            let src = src.cast::<[T; N_CONTIGUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i);
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTIGUOUS]>();
                    dst.write(src.read());
                }
            }
        } else if dst_contiguous {
            let dst = dst.cast::<[T; N_CONTIGUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTIGUOUS]>();
                    let dst = dst.add(i);
                    dst.write(src.read());
                }
            }
        } else {
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTIGUOUS]>();
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTIGUOUS]>();
                    dst.write(src.read());
                }
            }
        }
    }
    #[inline(never)]
    unsafe fn copy_1d_unaligned<T: Copy, const N_CONTIGUOUS: usize>(
        src: &[u8],
        dst: &mut [u8],
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let src_contiguous = len <= 1 || src_stride == size_of::<T>() * N_CONTIGUOUS;
        let dst_contiguous = len <= 1 || dst_stride == size_of::<T>() * N_CONTIGUOUS;
        if src_contiguous && dst_contiguous {
            let src = src.cast::<[T; N_CONTIGUOUS]>();
            let dst = dst.cast::<[T; N_CONTIGUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i);
                    let dst = dst.add(i);
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        } else if src_contiguous {
            let src = src.cast::<[T; N_CONTIGUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i);
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTIGUOUS]>();
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        } else if dst_contiguous {
            let dst = dst.cast::<[T; N_CONTIGUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTIGUOUS]>();
                    let dst = dst.add(i);
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        } else {
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTIGUOUS]>();
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTIGUOUS]>();
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        }
    }

    #[inline(never)]
    unsafe fn copy_1d_unsized<T: Copy>(
        src: &[u8],
        dst: &mut [u8],
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        n_contiguous_items: usize,
        aligned: bool,
    ) {
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        unsafe {
            if aligned {
                let src_contiguous = len <= 1 || src_stride == size_of::<T>() * n_contiguous_items;
                let dst_contiguous = len <= 1 || dst_stride == size_of::<T>() * n_contiguous_items;
                if src_contiguous && dst_contiguous {
                    std::ptr::copy_nonoverlapping::<T>(
                        src.cast::<T>(),
                        dst.cast::<T>(),
                        len * n_contiguous_items,
                    );
                } else if src_contiguous {
                    let src = src.cast::<T>();
                    for i in 0..len {
                        let src = src.add(i * n_contiguous_items);
                        let dst = dst.add(i * dst_stride).cast::<T>();
                        std::ptr::copy_nonoverlapping::<T>(src, dst, n_contiguous_items);
                    }
                } else if dst_contiguous {
                    let dst = dst.cast::<T>();
                    for i in 0..len {
                        let src = src.add(i * src_stride).cast::<T>();
                        let dst = dst.add(i * n_contiguous_items);
                        std::ptr::copy_nonoverlapping::<T>(src, dst, n_contiguous_items);
                    }
                } else {
                    for i in 0..len {
                        let src = src.add(i * src_stride).cast::<T>();
                        let dst = dst.add(i * dst_stride).cast::<T>();
                        std::ptr::copy_nonoverlapping::<T>(src, dst, n_contiguous_items);
                    }
                }
            } else {
                let n_contiguous_bytes = size_of::<T>() * n_contiguous_items;
                for i in 0..len {
                    let src = src.add(i * src_stride);
                    let dst = dst.add(i * dst_stride);
                    std::ptr::copy_nonoverlapping::<u8>(src, dst, n_contiguous_bytes);
                }
            }
        }
    }

    /// Copy a region whose innermost axes did not fully coalesce (at least two strided axes
    /// remain). We peel the innermost remaining axis and copy each of its runs with the
    /// specialized [`copy_1d_dispatch`](Self::copy_1d_dispatch) path, walking only the OUTER axes
    /// with the [`NdIter`]. Unlike a flat nd-walk that copies `n_contiguous_items` at every
    /// element, this exploits one-sided contiguity of the innermost axis - the common shape after
    /// the destination-stride permutation, where the destination is unit-stride along the
    /// innermost axis but the source is not - turning each run into a tight sequential-store /
    /// strided-load loop and cutting the iterator trip count by a factor of the innermost extent.
    #[inline(never)]
    unsafe fn copy_nd<T: Copy>(
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        src: &[u8],
        dst: &mut [u8],
        n_contiguous_items: usize,
    ) {
        let ndim = shape.len();
        debug_assert!(ndim >= 2);
        // Peel the innermost axis; the NdIter walks only the outer axes `[0, ndim - 1)`.
        let inner = ndim - 1;
        let inner_len = shape[inner];
        let inner_src_stride = src_strides[inner];
        let inner_dst_stride = dst_strides[inner];

        // `aligned` covers every stride and both bases, so each row base stays aligned and the
        // innermost strides are aligned too - the precondition `copy_1d`'s aligned path needs.
        let aligned = (src.as_ptr().cast::<T>().is_aligned()
            && src_strides
                .iter()
                .all(|s| s.is_multiple_of(align_of::<T>())))
            && (dst.as_ptr().cast::<T>().is_aligned()
                && dst_strides
                    .iter()
                    .all(|s| s.is_multiple_of(align_of::<T>())));

        unsafe fn copy_nd_inner<T: Copy, D: Dimension>(
            outer_shape: &[usize],
            outer_src_strides: &[usize],
            outer_dst_strides: &[usize],
            src: &[u8],
            dst: &mut [u8],
            inner_len: usize,
            inner_src_stride: usize,
            inner_dst_stride: usize,
            n_contiguous_items: usize,
            aligned: bool,
        ) {
            let src = src.as_ptr();
            let dst = dst.as_mut_ptr();
            let iter = NdIter::new(
                D::vec(outer_shape.len(), |i| outer_shape[i] as u64),
                (
                    NdIterExtStridesPtr::new(outer_src_strides.to_dim_vec::<D>(), src),
                    NdIterExtStridesPtrMut::new(outer_dst_strides.to_dim_vec::<D>(), dst),
                ),
            );
            for (_, (src_ptr, dst_ptr)) in iter {
                // Zero-length slices carrying each row's base pointer (see `copy`).
                let src_row = unsafe { std::slice::from_raw_parts(src_ptr, 0) };
                let dst_row = unsafe { std::slice::from_raw_parts_mut(dst_ptr, 0) };
                unsafe {
                    NdCopier::copy_1d_dispatch::<T>(
                        src_row,
                        dst_row,
                        inner_len,
                        inner_src_stride,
                        inner_dst_stride,
                        n_contiguous_items,
                        aligned,
                    );
                }
            }
        }

        let outer_shape = &shape[..inner];
        let outer_src_strides = &src_strides[..inner];
        let outer_dst_strides = &dst_strides[..inner];
        let copy_nd_fn = match outer_shape.len() {
            0 => unreachable!(),
            1 => copy_nd_inner::<T, Dim<1>>,
            2 => copy_nd_inner::<T, Dim<2>>,
            3 => copy_nd_inner::<T, Dim<3>>,
            4 => copy_nd_inner::<T, Dim<4>>,
            5 => copy_nd_inner::<T, Dim<5>>,
            6 => copy_nd_inner::<T, Dim<6>>,
            7 => copy_nd_inner::<T, Dim<7>>,
            _ => unimplemented!(),
        };
        unsafe {
            copy_nd_fn(
                outer_shape,
                outer_src_strides,
                outer_dst_strides,
                src,
                dst,
                inner_len,
                inner_src_stride,
                inner_dst_stride,
                n_contiguous_items,
                aligned,
            )
        }
    }

    #[inline(never)]
    fn copy_struct(struct_copier: &NdCopierStruct, args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            dst,
            shape,
            src_strides,
            dst_strides,
            dtype: _,
        } = args;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        for ((f, &offset), field_dtype) in struct_copier
            .scalar_fns
            .iter()
            .zip(struct_copier.offsets.iter())
            .zip(struct_copier.dtypes.iter())
        {
            // Rebuild the per-field zero-length slices from the base pointers. SAFETY: the field
            // at `offset` lies within the element; distinct fields do not overlap; the lengths are
            // unused and a zero-length u8 slice is always aligned.
            let field_args = NdCopyArgs {
                src: unsafe { std::slice::from_raw_parts(src_ptr.add(offset as usize), 0) },
                dst: unsafe { std::slice::from_raw_parts_mut(dst_ptr.add(offset as usize), 0) },
                shape,
                src_strides,
                dst_strides,
                dtype: field_dtype,
            };
            f(field_args)
        }
    }

    fn copy_dynamic(args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            dst,
            shape,
            src_strides,
            dst_strides,
            dtype,
        } = args;

        let itemsize = dtype.itemsize() as usize;
        // Same axis collapse as the scalar path (see `merge_dims`), keyed on the dtype's itemsize.
        let merged = merge_dims(shape, src_strides, dst_strides, itemsize);
        let mut shape = merged.shape.as_slice();
        let mut src_strides = merged.src_strides.as_slice();
        let mut dst_strides = merged.dst_strides.as_slice();
        let n_contiguous_bytes = merged.n_contiguous_items * itemsize;
        if shape.is_empty() {
            shape = &[1];
            src_strides = &[0];
            dst_strides = &[0];
        }

        unsafe fn copy_dynamic_inner<D: Dimension>(
            shape: &[usize],
            src_strides: &[usize],
            dst_strides: &[usize],
            src: &[u8],
            dst: &mut [u8],
            n_contiguous_bytes: usize,
        ) {
            let src = src.as_ptr();
            let dst = dst.as_mut_ptr();
            let iter = NdIter::new(
                D::vec(shape.len(), |i| shape[i] as u64),
                (
                    NdIterExtStridesPtr::new(src_strides.to_dim_vec::<D>(), src),
                    NdIterExtStridesPtrMut::new(dst_strides.to_dim_vec::<D>(), dst),
                ),
            );
            for (_, (src_ptr, dst_ptr)) in iter {
                unsafe {
                    std::ptr::copy_nonoverlapping::<u8>(src_ptr, dst_ptr, n_contiguous_bytes);
                }
            }
        }

        let copy_fn = match shape.len() {
            0 => unreachable!(),
            1 => copy_dynamic_inner::<Dim<1>>,
            2 => copy_dynamic_inner::<Dim<2>>,
            3 => copy_dynamic_inner::<Dim<3>>,
            4 => copy_dynamic_inner::<Dim<4>>,
            5 => copy_dynamic_inner::<Dim<5>>,
            6 => copy_dynamic_inner::<Dim<6>>,
            7 => copy_dynamic_inner::<Dim<7>>,
            8 => copy_dynamic_inner::<Dim<8>>,
            _ => unimplemented!(),
        };
        unsafe {
            copy_fn(
                shape,
                src_strides,
                dst_strides,
                src,
                dst,
                n_contiguous_bytes,
            )
        }
    }
}

/// Result of [`merge_dims`]: the remaining strided axes (outermost-first) plus the size of the
/// trailing run of elements that are contiguous on both sides.
struct MergedDims {
    shape: DimArray<usize>,
    src_strides: DimArray<usize>,
    dst_strides: DimArray<usize>,
    n_contiguous_items: usize,
}

/// Collapse a copy region to the fewest axes: drop size-1 axes (they contribute offset 0), merge
/// any adjacent pair that is contiguous with respect to itself on BOTH src and dst
/// (`stride_outer == stride_inner * extent_inner`), then peel the innermost surviving axis as the
/// contiguous run when its stride equals `itemsize` on both sides. Byte strides / byte `itemsize`
/// throughout, so it serves both the scalar (`size_of::<T>()`) and byte-wise (`dtype.itemsize()`)
/// callers.
///
/// This generalizes the old "merge only the trailing both-contiguous axes" scan: those trailing
/// axes still collapse into one unit-stride innermost axis - yielding the same `n_contiguous_items`
/// - but interior/outer axes now coalesce too, cutting the iterated rank (e.g. a wholly
/// stride-compatible region collapses to a single 1D copy, and a transpose whose outer axes are
/// stride-compatible loses those extra iterator dimensions).
#[inline]
fn merge_dims(
    shape: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
    itemsize: usize,
) -> MergedDims {
    debug_assert!(shape.len() == src_strides.len() && shape.len() == dst_strides.len());
    // Coalesce right-to-left, so `tmp_*[0]` is the innermost surviving group.
    let mut tmp_shape = DimArray::new();
    let mut tmp_src = DimArray::new();
    let mut tmp_dst = DimArray::new();
    for d in (0..shape.len()).rev() {
        let sz = shape[d];
        if sz == 1 {
            continue; // size-1 axis: offset always 0 - drop it (this also lets its neighbors meet)
        }
        let (ss, ds) = (src_strides[d], dst_strides[d]);
        if let Some(cur) = tmp_shape.len().checked_sub(1) {
            // Merge into the current innermost group when contiguous-adjacent on both sides; the
            // merged group keeps the inner strides and grows its extent.
            if ss == tmp_src[cur] * tmp_shape[cur] && ds == tmp_dst[cur] * tmp_shape[cur] {
                tmp_shape[cur] *= sz;
                continue;
            }
        }
        tmp_shape.push(sz);
        tmp_src.push(ss);
        tmp_dst.push(ds);
    }

    let m = tmp_shape.len();
    // Peel the innermost group as the contiguous run iff it is unit-stride on both sides.
    let (n_contiguous_items, strided_start) =
        if m > 0 && tmp_src[0] == itemsize && tmp_dst[0] == itemsize {
            (tmp_shape[0], 1)
        } else {
            (1, 0)
        };
    let n_strided = m - strided_start;
    // Reverse the surviving strided groups into outermost-first order.
    MergedDims {
        shape: dim_arr(n_strided, |i| tmp_shape[m - 1 - i]),
        src_strides: dim_arr(n_strided, |i| tmp_src[m - 1 - i]),
        dst_strides: dim_arr(n_strided, |i| tmp_dst[m - 1 - i]),
        n_contiguous_items,
    }
}

#[inline]
fn compute_dim_permutation(dst_strides: &[usize]) -> Option<DimArray<usize>> {
    if dst_strides.windows(2).all(|w| w[0] >= w[1]) {
        None
    } else {
        let ndim = dst_strides.len();
        let mut perm = dim_arr(ndim, |d| d);
        perm[..ndim].sort_by(|&a, &b| dst_strides[b].cmp(&dst_strides[a]));
        Some(perm)
    }
}
#[inline]
fn apply_dim_permutation<T: Copy>(arr: &[T], perm: &[usize]) -> DimArray<T> {
    dim_arr(perm.len(), |d| arr[perm[d]])
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
    #[repr(C)]
    struct StructNoPad {
        a: i32,
        b: i16,
        c: i16,
        d: i32,
    }

    // Byte strides for a row-major array whose backing shape is `shape[d] * mult[d]`,
    // with logical elements sampled every `mult[d]` slots along axis `d`. `mult` all
    // ones gives a fully contiguous layout; any `mult[d] > 1` leaves gaps, exercising
    // the strided (non-coalesced) path - including a strided innermost axis.
    fn strided_strides(shape: &[usize], mult: &[usize], itemsize: usize) -> Vec<usize> {
        let ndim = shape.len();
        let backing: Vec<usize> = (0..ndim).map(|d| shape[d] * mult[d]).collect();
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

    // A `strided_strides` multiplier vector that leaves a one-element gap only along axis `ax`
    // (all ones - fully contiguous - when `ax` is out of range, e.g. for a 0-D shape).
    fn strided_axis(ndim: usize, ax: usize) -> Vec<usize> {
        (0..ndim).map(|d| if d == ax { 2 } else { 1 }).collect()
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

        let copier = NdCopier::new(&dtype);
        unsafe {
            copier.copy(src_ptr, dst_ptr, shape, src_strides, dst_strides, &dtype);
        }

        let actual = unsafe { std::slice::from_raw_parts(dst_ptr, dst_len) };
        assert_eq!(
            actual,
            expected.as_slice(),
            "itemsize={itemsize} misalign={misalign} shape={shape:?} \
             src_strides={src_strides:?} dst_strides={dst_strides:?}"
        );
    }

    // For a fixed (element type, shape), exercise a representative set of source/dst layouts, each
    // aligned and deliberately misaligned:
    //   - contiguous/contiguous: fully coalesces into a single run;
    //   - strided-outer/contiguous: inner axes coalesce, driving the 1D fast path;
    //   - strided-inner/strided-inner: nothing coalesces, driving the general strided nd iterator.
    // Gaps stay on a single axis, so buffers never exceed twice the contiguous size - small enough
    // to run the whole matrix under Miri.
    fn check_rank<T: Dtyped>(shape: &[usize]) {
        let itemsize = T::DTYPE.itemsize() as usize;
        let ndim = shape.len();
        let cont = strided_strides(shape, &vec![1; ndim], itemsize);
        let outer = strided_strides(shape, &strided_axis(ndim, 0), itemsize);
        let inner = strided_strides(shape, &strided_axis(ndim, ndim.wrapping_sub(1)), itemsize);
        for (src, dst) in [(&cont, &cont), (&outer, &cont), (&inner, &inner)] {
            for misalign in [0, 1] {
                check::<T>(shape, src, dst, misalign);
            }
        }
    }

    // Ranks 0 through 8 (the maximum) for one element type. The trailing `[3, k]` shapes put a
    // contiguous inner run of length k behind a strided outer axis, so the run coalesces to
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
    // exploring random ranks, extents, per-axis strides and base alignment against the reference.
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

    // A unit (size-1) axis is iterated once at index 0, so its stride never contributes an offset;
    // the coalescing scan treats it as contiguous whatever stride it carries and never lets it block
    // neighboring axes from merging. Each layout gives a unit axis a stride that would otherwise halt
    // the scan (a broadcast `0` and an arbitrary multiple), with contiguous axes around it, covering
    // both the scalar (`scalar_fn`) and byte-wise (`copy_dynamic`) coalescing routes.
    fn check_unit_axis<T: Dtyped>() {
        let is = T::DTYPE.itemsize() as usize;
        for unit in [0, 5 * is] {
            // interior unit axis: contiguous both sides (full coalesce), then a strided outer axis
            check::<T>(&[4, 1, 8], &[8 * is, unit, is], &[8 * is, unit, is], 0);
            check::<T>(&[4, 1, 8], &[16 * is, unit, is], &[16 * is, unit, is], 0);
        }
        // leading and trailing unit axes, contiguous elsewhere
        check::<T>(&[1, 4, 8], &[7 * is, 8 * is, is], &[7 * is, 8 * is, is], 0);
        check::<T>(&[4, 8, 1], &[8 * is, is, 3 * is], &[8 * is, is, 3 * is], 0);
        check::<T>(&[4, 1, 8], &[8 * is, 0, is], &[8 * is, 0, is], 1); // misaligned base
    }
    #[test]
    fn copy_unit_axis() {
        check_unit_axis::<u32>(); // scalar route
        check_unit_axis::<u64>();
        check_unit_axis::<[i32; 3]>(); // byte-wise copy_dynamic route
        check_unit_axis::<[u8; 3]>();
    }

    // The 1D copy splits on per-side contiguity: an outer axis that survives coalescing with its byte
    // stride still equal to the inner run's length is contiguous and is copied as whole blocks (a
    // single `copy_nonoverlapping` when both sides are, a per-side loop when one is, the general loop
    // otherwise). Folding needs *both* strides to match, so at most one side reaches this path
    // contiguous with `len > 1`. An inner run of `inner` behind a strided outer axis pins each arm; a
    // power-of-two `inner` routes through the const-generic `copy_1d`, others through
    // `copy_1d_unsized`. `misalign = 1` adds the unaligned arm.
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
            check::<T>(&[4, inner], src, dst, 0);
        }
        check::<T>(&[4, inner], &cont, &strided, 1);
    }
    #[test]
    fn copy_1d_sides() {
        check_1d_sides::<u32>(8); // power-of-two run -> const-generic copy_1d arms
        check_1d_sides::<u32>(3); // other run length -> copy_1d_unsized
        check_1d_sides::<u64>(8); // wider element type
        check_1d_sides::<u64>(3);
    }

    // Struct dtype (the `copy_struct` field-by-field route) and non-scalar array dtypes (the
    // `copy_dynamic` byte-wise fallback): array-of-scalar dtypes carry an inner shape, so they miss
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
}
