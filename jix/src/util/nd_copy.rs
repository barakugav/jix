use std::ptr;

use crate::arrayvec::ArrayVec;
use crate::dtype::{Alignment, Dtype, Itemsize};
use crate::{NdIterUnordered, PtrExt, PtrMutExt};

/// A reusable, dtype-specialized copier that moves a rectangular n-dimensional region between two
/// byte slices under independent source and destination strides.
///
/// [`new`](Self::new) inspects the dtype once and picks the cheapest copy routine up front: a
/// monomorphized scalar copy for the common power-of-two `(itemsize, alignment)` pairs, a
/// field-by-field copy for structs that decompose into at most four scalar fields, or a generic
/// byte-wise fallback for everything else. Each [`copy`](Self::copy) then moves one region by
/// driving an [`NdIterUnordered`] over `shape` (which sorts, coalesces and walks the axes) and
/// copying one contiguous-or-strided inner loop at each visited position.
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
            NdCopierInner::Simple(Self::copy_untyped)
        })
    }

    /// Copies a rectangular `shape` region from `src` to `dst` under independent byte strides.
    ///
    /// `src` and `dst` are the backing byte slices.
    /// Passing slices - rather than raw pointers - lets the inner loops derive `noalias`-tagged pointers.
    ///
    /// # Safety
    ///
    /// - Each slice's byte range must be a superset of every byte the copy touches. For a region
    ///   with the given `shape`/strides that is `sum_d (shape[d] - 1) * stride[d] + itemsize` bytes
    ///   from the slice start (all byte strides are non-negative). A shorter slice is undefined
    ///   behavior even though no explicit bounds checks run.
    /// - The `src` and `dst` regions must not overlap.
    #[inline(always)]
    pub(crate) unsafe fn copy(
        &self,
        src: &[u8],
        dst: &mut [u8],
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        dtype: &Dtype,
    ) {
        if shape.contains(&0) {
            return;
        }
        // Axis ordering, size-1-axis dropping and contiguous-run coalescing all happen inside
        // `nd_iter_unordered` (see `scalar_fn` / `copy_untyped`); here we just dispatch by dtype.
        let args = NdCopyArgs {
            src,
            dst,
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
            src_strides,
            dst_strides,
            dtype,
        } = args;
        debug_assert_eq!(size_of::<T>(), dtype.itemsize() as usize);
        debug_assert_eq!(align_of::<T>(), dtype.alignment().as_usize());

        // Operand 0 is the destination (write; sort-primary), operand 1 the source (read). Both are
        // byte-addressed with element layout `(size_of::<T>(), align_of::<T>())`.
        let iter = NdIterUnordered::new(
            shape,
            [dst_strides, src_strides],
            [(size_of::<T>() as Itemsize, Alignment::of::<T>()); 2],
        );
        // The iterator's aligned flags only cover the strides; AND-in the base pointers so the
        // aligned `read`/`write` path runs only when the data really is.
        let [dst_aligned, src_aligned] = iter.is_aligned();
        let aligned = (dst_aligned && dst.as_ptr().cast::<T>().is_aligned())
            && (src_aligned && src.as_ptr().cast::<T>().is_aligned());
        let [dst_contiguous, src_contiguous] = iter.is_contiguous();

        // For the both-contiguous run, peel a small fixed length into a single `[T; N]` move
        // (branchless); longer runs fall through to `copy_nonoverlapping`.
        let inner_loop_fn = if src_contiguous && dst_contiguous {
            match (iter.inner_len(), aligned) {
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
                (32, true) if size_of::<T>() <= 8 => {
                    Self::inner_loop_contiguous_const_len::<T, 32, true>
                }
                (32, false) if size_of::<T>() <= 8 => {
                    Self::inner_loop_contiguous_const_len::<T, 32, false>
                }
                (64, true) if size_of::<T>() <= 4 => {
                    Self::inner_loop_contiguous_const_len::<T, 64, true>
                }
                (64, false) if size_of::<T>() <= 4 => {
                    Self::inner_loop_contiguous_const_len::<T, 64, false>
                }
                (_, true) => Self::inner_loop::<T, true, true, true>,
                (_, false) => Self::inner_loop::<T, false, true, true>,
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
        };

        iter.foreach_inner_1d(
            |[dst_offset, src_offset], len, [dst_stride, src_stride]| unsafe {
                inner_loop_fn(
                    src.get_unchecked(src_offset..),
                    dst.get_unchecked_mut(dst_offset..),
                    len,
                    src_stride,
                    dst_stride,
                )
            },
        );
    }

    unsafe fn inner_loop<
        T: Copy,
        const ALIGNED: bool,
        const SRC_CONTIGUOUS: bool,
        const DST_CONTIGUOUS: bool,
    >(
        src: &[u8],
        dst: &mut [u8],
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        if SRC_CONTIGUOUS {
            debug_assert_eq!(src_stride, size_of::<T>());
        }
        if DST_CONTIGUOUS {
            debug_assert_eq!(dst_stride, size_of::<T>());
        }
        let src = src.as_ptr().cast::<T>();
        let dst = dst.as_mut_ptr().cast::<T>();
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
                        src.cast::<u8>().add(i * src_stride).cast::<T>()
                    };
                    let d = if DST_CONTIGUOUS {
                        dst.add(i)
                    } else {
                        dst.cast::<u8>().add(i * dst_stride).cast::<T>()
                    };
                    let val = s.read_maybe_aligned::<ALIGNED>();
                    d.write_maybe_aligned::<ALIGNED>(val);
                }
            }
        }
    }

    /// One inner run where both operands are contiguous and its length is the compile-time `N`:
    /// a single `[T; N]` load/store (branchless), the common small-run fast path.
    unsafe fn inner_loop_contiguous_const_len<T: Copy, const LEN: usize, const ALIGNED: bool>(
        src: &[u8],
        dst: &mut [u8],
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

    unsafe fn inner_loop_untyped<const SRC_DST_CONTIGUOUS: bool>(
        src: &[u8],
        dst: &mut [u8],
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
                    let s = src.add(i * src_stride);
                    let d = dst.add(i * dst_stride);
                    ptr::copy_nonoverlapping(s, d, itemsize);
                }
            }
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
        for ((f, &offset), field_dtype) in struct_copier
            .scalar_fns
            .iter()
            .zip(struct_copier.offsets.iter())
            .zip(struct_copier.dtypes.iter())
        {
            let field_args = NdCopyArgs {
                src: unsafe { src.get_unchecked(offset as usize..) },
                dst: unsafe { dst.get_unchecked_mut(offset as usize..) },
                shape,
                src_strides,
                dst_strides,
                dtype: field_dtype,
            };
            f(field_args)
        }
    }

    fn copy_untyped(args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            dst,
            shape,
            src_strides,
            dst_strides,
            dtype,
        } = args;
        let itemsize = dtype.itemsize() as usize;

        // Byte-wise fallback: each element is `itemsize` opaque bytes copied via `copy_nonoverlapping`
        // (always the unaligned `u8` path). Operand 0 is the destination, operand 1 the source.
        let iter = NdIterUnordered::new(
            shape,
            [dst_strides, src_strides],
            [(dtype.itemsize(), Alignment::of::<u8>()); 2],
        );
        let [dst_contiguous, src_contiguous] = iter.is_contiguous();
        let inner_loop_fn = match dst_contiguous && src_contiguous {
            true => Self::inner_loop_untyped::<true>,
            false => Self::inner_loop_untyped::<false>,
        };
        iter.foreach_inner_1d(
            |[dst_offset, src_offset], len, [dst_stride, src_stride]| unsafe {
                inner_loop_fn(
                    src.get_unchecked(src_offset..),
                    dst.get_unchecked_mut(dst_offset..),
                    len,
                    src_stride,
                    dst_stride,
                    itemsize,
                )
            },
        );
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
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) };
        unsafe {
            copier.copy(src, dst, shape, src_strides, dst_strides, &dtype);
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
    // both the scalar (`scalar_fn`) and byte-wise (`copy_untyped`) coalescing routes.
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
        check_unit_axis::<[i32; 3]>(); // byte-wise copy_untyped route
        check_unit_axis::<[u8; 3]>();
    }

    // The 1D copy splits on per-side contiguity: an outer axis that survives coalescing with its byte
    // stride still equal to the inner run's length is contiguous and is copied as whole blocks (a
    // single `copy_nonoverlapping` when both sides are, a per-side loop when one is, the general loop
    // otherwise). Folding needs *both* strides to match, so at most one side reaches this path
    // contiguous with `len > 1`. An inner run of `inner` behind a strided outer axis pins each arm; a
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
