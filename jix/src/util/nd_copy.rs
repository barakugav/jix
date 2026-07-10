use crate::arrayvec::ArrayVec;
use crate::dtype::{Dtype, Itemsize};
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::iter::NdIter;
use crate::{Dim, DimArray, DimDyn, Dimension, SliceExt};

/// A reusable, dtype-specialized copier that moves a rectangular n-dimensional region between two
/// raw byte buffers under independent source and destination strides.
///
/// [`new`](Self::new) inspects the dtype once and picks the cheapest copy routine up front: a
/// monomorphized scalar copy for the common power-of-two `(itemsize, alignment)` pairs, a
/// field-by-field copy for structs that decompose into at most four scalar fields, or a generic
/// byte-wise fallback for everything else. Each [`copy`](Self::copy) then moves one region by
/// walking `shape` with an [`NdIter`] and copying the appropriate number of bytes at each element.
pub(crate) struct NdCopier<'a, D: Dimension>(NdCopierInner<'a, D>);
// TODO: remove the D generic
enum NdCopierInner<'a, D: Dimension> {
    Simple(NdCopyFn<D>),
    Struct(NdCopierStruct<'a, D>),
}
type NdCopyFn<D> = fn(NdCopyArgs<D>);
struct NdCopierStruct<'a, D: Dimension> {
    scalar_fns: ArrayVec<NdCopyFn<D>, 4>,
    offsets: ArrayVec<Itemsize, 4>,
    dtypes: ArrayVec<&'a Dtype, 4>,
}
struct NdCopyArgs<'a, D: Dimension> {
    src: *const u8,
    dst: *mut u8,
    shape: &'a D::Vec<usize>,
    src_strides: &'a D::Vec<usize>,
    dst_strides: &'a D::Vec<usize>,
    dtype: &'a Dtype,
}
impl<'a, D: Dimension> Clone for NdCopyArgs<'a, D> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            dst: self.dst,
            shape: self.shape,
            src_strides: self.src_strides,
            dst_strides: self.dst_strides,
            dtype: self.dtype,
        }
    }
}

impl<'a, D: Dimension> NdCopier<'a, D> {
    #[inline(always)]
    pub(crate) fn new(dtype: &'a Dtype) -> Self {
        match Self::create_scalar_fn(dtype) {
            Some(f) => Self(NdCopierInner::Simple(f)),
            None => Self::new_slow(dtype),
        }
    }

    fn new_slow(dtype: &'a Dtype) -> Self {
        #[inline]
        fn collect_struct_fields<'a, D: Dimension>(
            mut copier: NdCopierStruct<'a, D>,
            offset: Itemsize,
            dtype: &'a Dtype,
        ) -> Option<NdCopierStruct<'a, D>> {
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
                fn collect_struct_fields_inner<'a, D: Dimension>(
                    copier: NdCopierStruct<'a, D>,
                    offset: Itemsize,
                    dtype: &'a Dtype,
                ) -> Option<NdCopierStruct<'a, D>> {
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
        shape: &D::Vec<usize>,
        src_strides: &D::Vec<usize>,
        dst_strides: &D::Vec<usize>,
        dtype: &Dtype,
    ) {
        if shape.as_ref().contains(&0) {
            return;
        }
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
    const fn create_scalar_fn(dtype: &Dtype) -> Option<fn(NdCopyArgs<D>)> {
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

    fn scalar_fn<T: Copy + 'static>(args: NdCopyArgs<D>) {
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

        let mut shape = shape.as_ref();
        let ndim = shape.len();
        let mut src_strides = src_strides.as_ref();
        let mut dst_strides = dst_strides.as_ref();
        assert!(ndim == src_strides.len() && ndim == dst_strides.len());
        let mut n_continuous_items = 1;

        // copy more then itemsize if the last dim(s) is contiguous
        let n_continuous_dims = (0..ndim)
            .rev()
            .scan(size_of::<T>(), |expected_stride, dim| {
                let is_contiguous = shape[dim] <= 1
                    || (src_strides[dim] == *expected_stride
                        && dst_strides[dim] == *expected_stride);
                *expected_stride *= shape[dim];
                Some(is_contiguous)
            })
            .take_while(|&is_contiguous| is_contiguous)
            .count();
        if n_continuous_dims > 0 {
            let n_strided_dims = ndim - n_continuous_dims;
            n_continuous_items = shape[n_strided_dims..].iter().product::<usize>();
            shape = &shape[..n_strided_dims];
            src_strides = &src_strides[..n_strided_dims];
            dst_strides = &dst_strides[..n_strided_dims];
        }
        if shape.len() == 0 {
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
                    n_continuous_items,
                )
            };
        }
        // 1D copy

        let len = shape[0];
        let src_stride = src_strides[0];
        let dst_stride = dst_strides[0];

        let aligned = (src.cast::<T>().is_aligned() && src_stride.is_multiple_of(align_of::<T>()))
            && (dst.cast::<T>().is_aligned() && dst_stride.is_multiple_of(align_of::<T>()));

        unsafe {
            match n_continuous_items {
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
                _ => {
                    if aligned {
                        for i in 0..len {
                            let src = src.add(i * src_stride).cast::<T>();
                            let dst = dst.add(i * dst_stride).cast::<T>();
                            std::ptr::copy_nonoverlapping::<T>(src, dst, n_continuous_items);
                        }
                    } else {
                        let n_continuous_bytes = size_of::<T>() * n_continuous_items;
                        for i in 0..len {
                            let src = src.add(i * src_stride);
                            let dst = dst.add(i * dst_stride);
                            std::ptr::copy_nonoverlapping::<u8>(src, dst, n_continuous_bytes);
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn copy_1d<T: Copy, const N_CONTINUOUS: usize>(
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        aligned: bool,
    ) {
        if aligned {
            unsafe {
                Self::copy_1d_aligned::<T, N_CONTINUOUS>(src, dst, len, src_stride, dst_stride)
            }
        } else {
            unsafe {
                Self::copy_1d_unaligned::<T, N_CONTINUOUS>(src, dst, len, src_stride, dst_stride)
            }
        }
    }
    #[inline]
    unsafe fn copy_1d_aligned<T: Copy, const N_CONTINUOUS: usize>(
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        for i in 0..len {
            unsafe {
                let src = src.add(i * src_stride).cast::<[T; N_CONTINUOUS]>();
                let dst = dst.add(i * dst_stride).cast::<[T; N_CONTINUOUS]>();
                dst.write(src.read());
            }
        }
    }
    #[inline(never)]
    unsafe fn copy_1d_unaligned<T: Copy, const N_CONTINUOUS: usize>(
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
    ) {
        for i in 0..len {
            unsafe {
                let src = src.add(i * src_stride).cast::<[T; N_CONTINUOUS]>();
                let dst = dst.add(i * dst_stride).cast::<[T; N_CONTINUOUS]>();
                dst.write_unaligned(src.read_unaligned());
            }
        }
    }

    #[inline(never)]
    unsafe fn copy_nd<T: Copy>(
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
        src: *const u8,
        dst: *mut u8,
        n_continuous_items: usize,
    ) {
        unsafe fn copy_nd_inner<T: Copy, D: Dimension>(
            shape: D::Vec<u64>,
            src_strides: D::Vec<usize>,
            dst_strides: D::Vec<usize>,
            src: *const u8,
            dst: *mut u8,
            n_continuous_items: usize,
            aligned: bool,
        ) {
            let iter = NdIter::new(
                shape,
                (
                    NdIterExtStridesPtr::new(src_strides, src),
                    NdIterExtStridesPtrMut::new(dst_strides, dst),
                ),
            );
            if aligned {
                for (_, (src_ptr, dst_ptr)) in iter {
                    unsafe {
                        std::ptr::copy_nonoverlapping::<T>(
                            src_ptr.cast::<T>(),
                            dst_ptr.cast::<T>(),
                            n_continuous_items,
                        );
                    }
                }
            } else {
                let n_continuous_bytes = size_of::<T>() * n_continuous_items;
                for (_, (src_ptr, dst_ptr)) in iter {
                    unsafe {
                        std::ptr::copy_nonoverlapping::<u8>(src_ptr, dst_ptr, n_continuous_bytes);
                    }
                }
            }
        }

        let aligned = (src.cast::<T>().is_aligned()
            && src_strides
                .iter()
                .all(|s| s.is_multiple_of(align_of::<T>())))
            && (dst.cast::<T>().is_aligned()
                && dst_strides
                    .iter()
                    .all(|s| s.is_multiple_of(align_of::<T>())));

        unsafe {
            match shape.len() {
                0 | 1 => unreachable!(),
                2 => copy_nd_inner::<T, Dim<2>>(
                    Dim::<2>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<2>>(),
                    dst_strides.to_dim_vec::<Dim<2>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                3 => copy_nd_inner::<T, Dim<3>>(
                    Dim::<3>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<3>>(),
                    dst_strides.to_dim_vec::<Dim<3>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                4 => copy_nd_inner::<T, Dim<4>>(
                    Dim::<4>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<4>>(),
                    dst_strides.to_dim_vec::<Dim<4>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                5 => copy_nd_inner::<T, Dim<5>>(
                    Dim::<5>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<5>>(),
                    dst_strides.to_dim_vec::<Dim<5>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                6 => copy_nd_inner::<T, Dim<6>>(
                    Dim::<6>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<6>>(),
                    dst_strides.to_dim_vec::<Dim<6>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                7 => copy_nd_inner::<T, Dim<7>>(
                    Dim::<7>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<7>>(),
                    dst_strides.to_dim_vec::<Dim<7>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                8 => copy_nd_inner::<T, Dim<8>>(
                    Dim::<8>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<8>>(),
                    dst_strides.to_dim_vec::<Dim<8>>(),
                    src,
                    dst,
                    n_continuous_items,
                    aligned,
                ),
                _ => unimplemented!(),
            }
        }
    }

    #[inline(never)]
    fn copy_struct(struct_copier: &NdCopierStruct<D>, args: NdCopyArgs<D>) {
        for ((f, &offset), field_dtype) in struct_copier
            .scalar_fns
            .iter()
            .zip(struct_copier.offsets.iter())
            .zip(struct_copier.dtypes.iter())
        {
            let field_args = NdCopyArgs {
                src: unsafe { args.src.add(offset as usize) },
                dst: unsafe { args.dst.add(offset as usize) },
                dtype: field_dtype,
                ..args.clone()
            };
            f(field_args)
        }
    }

    fn copy_dynamic(args: NdCopyArgs<D>) {
        let NdCopyArgs {
            src,
            dst,
            shape,
            src_strides,
            dst_strides,
            dtype,
        } = args;

        let shape = shape.as_ref();
        let ndim = shape.len();
        assert!(ndim == src_strides.as_ref().len() && ndim == dst_strides.as_ref().len());
        let itemsize = dtype.itemsize() as usize;

        // copy more then itemsize if the last dim(s) is contiguous
        let n_continuous_dims = (0..ndim)
            .rev()
            .scan(itemsize, |expected_stride, dim| {
                let is_contiguous =
                    src_strides[dim] == *expected_stride && dst_strides[dim] == *expected_stride;
                *expected_stride *= shape[dim];
                Some(is_contiguous)
            })
            .take_while(|&is_contiguous| is_contiguous)
            .count();
        let itemsize = itemsize * shape[ndim - n_continuous_dims..].iter().product::<usize>();
        let shape = &shape[..ndim - n_continuous_dims];
        let src_strides = &src_strides[..ndim - n_continuous_dims];
        let dst_strides = &dst_strides[..ndim - n_continuous_dims];

        let iter = NdIter::new(
            shape.iter().map(|&s| s as u64).collect::<DimArray<_>>(),
            (
                NdIterExtStridesPtr::new(src_strides.to_dim_vec::<DimDyn>(), src),
                NdIterExtStridesPtrMut::new(dst_strides.to_dim_vec::<DimDyn>(), dst),
            ),
        );
        for (_, (src_ptr, dst_ptr)) in iter {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtyped;
    use crate::Dim;

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

    // Owns a source/destination byte buffer for one copy and hands out a byte pointer into it.
    //
    // The aligned variant (`misalign == 0`) is a `Vec<T>`, whose allocation is aligned to
    // `align_of::<T>()` - exactly the alignment the aligned copy path requires, since for every
    // specialized `(itemsize, alignment)` case the read-type's alignment equals the dtype's and
    // struct fields are read at no stricter alignment. The unaligned variant (`misalign == 1`) is a
    // `Vec<u8>` positioned one byte past an aligned offset (a plain `Vec<u8>` has no useful
    // alignment of its own, so we align within it first), forcing the unaligned path for every dtype
    // whose alignment exceeds 1. Both are only ever touched as raw bytes, so no `T` value is needed
    // and, since the `Vec<T>` stays `len == 0`, nothing is dropped.
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

    // Run a single copy through `NdCopier<D>` for element type `T` at a chosen base alignment
    // (`misalign == 0` aligned, `misalign == 1` deliberately misaligned) and assert it byte-matches
    // the independent reference.
    fn check<T: Dtyped, D: Dimension>(
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

        let copier = NdCopier::<D>::new(&dtype);
        let shape_v = D::vec(shape.len(), |i| shape[i]);
        let src_v = D::vec(shape.len(), |i| src_strides[i]);
        let dst_v = D::vec(shape.len(), |i| dst_strides[i]);
        unsafe {
            copier.copy(src_ptr, dst_ptr, &shape_v, &src_v, &dst_v, &dtype);
        }

        let actual = unsafe { std::slice::from_raw_parts(dst_ptr, dst_len) };
        assert_eq!(
            actual,
            expected.as_slice(),
            "itemsize={itemsize} misalign={misalign} shape={shape:?} \
             src_strides={src_strides:?} dst_strides={dst_strides:?}"
        );
    }

    // For a fixed (element type, dimension type, shape), exercise a small but representative set of
    // source/destination layouts, each at an aligned and a deliberately misaligned base:
    //   - contiguous/contiguous: fully coalesces into a single run;
    //   - strided-outer/contiguous: inner axes coalesce, driving the 1D fast path;
    //   - strided-inner/strided-inner: nothing coalesces, driving the byte-wise `copy_nd` iterator.
    // Gaps stay on a single axis, so buffers never exceed twice the contiguous size - small enough
    // to run the whole matrix under Miri with no special-casing.
    fn check_rank<T: Dtyped, D: Dimension>(shape: &[usize]) {
        let itemsize = T::DTYPE.itemsize() as usize;
        let ndim = shape.len();
        let cont = strided_strides(shape, &vec![1; ndim], itemsize);
        let outer = strided_strides(shape, &strided_axis(ndim, 0), itemsize);
        let inner = strided_strides(shape, &strided_axis(ndim, ndim.wrapping_sub(1)), itemsize);
        for (src, dst) in [(&cont, &cont), (&outer, &cont), (&inner, &inner)] {
            for misalign in [0, 1] {
                check::<T, D>(shape, src, dst, misalign);
            }
        }
    }

    // Ranks 0 through 8 (the maximum supported) for one element type. Every rank runs with runtime
    // `DimDyn` dims, which drives each `copy_nd` rank arm; a few ranks also run with static `Dim<N>`
    // dims to exercise those containers. Then a contiguous inner run of length K behind a strided
    // outer axis (which coalesces to `n_continuous == K`) drives each specialized `copy_1d` arm
    // (K in 1/2/4/8/16/32/64, subject to the size guards) plus the `_` fallback (K = 3) - the exact
    // shape that regressed when `N_CONTINUOUS` was misused as an outer-loop unroll factor.
    fn check_all_dims<T: Dtyped>() {
        check_rank::<T, DimDyn>(&[]);
        check_rank::<T, DimDyn>(&[7]);
        check_rank::<T, DimDyn>(&[3, 4]);
        check_rank::<T, DimDyn>(&[2, 3, 4]);
        check_rank::<T, DimDyn>(&[2, 2, 3, 3]);
        check_rank::<T, DimDyn>(&[2, 2, 2, 3, 3]);
        check_rank::<T, DimDyn>(&[2, 2, 2, 2, 3, 3]);
        check_rank::<T, DimDyn>(&[2, 2, 2, 2, 2, 2, 3]);
        check_rank::<T, DimDyn>(&[2, 2, 2, 2, 2, 2, 2, 2]);

        check_rank::<T, Dim<0>>(&[]);
        check_rank::<T, Dim<4>>(&[2, 2, 3, 3]);
        check_rank::<T, Dim<8>>(&[2, 2, 2, 2, 2, 2, 2, 2]);

        for k in [1, 2, 3, 4, 8, 16, 32, 64] {
            check_rank::<T, Dim<2>>(&[3, k]);
        }
    }

    // Zero-length dimensions: the copy is a no-op (the empty-region guard in `copy` fires before
    // any element or struct-field pointer is computed), so the zeroed destination must stay zeroed.
    fn check_zero_extent<T: Dtyped>() {
        let itemsize = T::DTYPE.itemsize() as usize;
        for shape in [vec![0usize], vec![3, 0], vec![2, 0, 4]] {
            let strides = strided_strides(&shape, &vec![1; shape.len()], itemsize);
            for misalign in [0, 1] {
                check::<T, DimDyn>(&shape, &strides, &strides, misalign);
            }
        }
    }

    // Randomized coverage via `fastrand`: deterministic per element type (seeded from the itemsize),
    // exploring random ranks, extents, per-axis strides and base alignment against the reference.
    // Buffers are bounded so every draw stays cheap enough to also run under Miri.
    fn fuzz<T: Dtyped>() {
        let itemsize = T::DTYPE.itemsize() as usize;
        let mut rng = fastrand::Rng::with_seed(0xC0FF_EE12_3456_789A ^ itemsize as u64);
        for _ in 0..64 {
            let ndim = rng.usize(0..=6);
            let shape: Vec<usize> = (0..ndim).map(|_| rng.usize(1..=3)).collect();
            let src_mult: Vec<usize> = (0..ndim).map(|_| rng.usize(1..=3)).collect();
            let dst_mult: Vec<usize> = (0..ndim).map(|_| rng.usize(1..=3)).collect();
            let src_strides = strided_strides(&shape, &src_mult, itemsize);
            let dst_strides = strided_strides(&shape, &dst_mult, itemsize);
            if buf_len(&shape, &src_strides, itemsize).max(buf_len(&shape, &dst_strides, itemsize))
                > 8 * 1024
            {
                continue;
            }
            let misalign = rng.usize(0..=1);
            check::<T, DimDyn>(&shape, &src_strides, &dst_strides, misalign);
        }
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

    // Unit (size-1) axes: iterated once at index 0, so their stride never contributes an offset.
    // The coalescing scan treats them as contiguous regardless of the stride stored for them, so an
    // axis carrying a broadcast-style `0` or an otherwise non-matching stride no longer stops
    // neighboring axes from merging. Each layout below gives a unit axis a stride that would halt
    // the pre-change scan (`0` and an arbitrary `5*is`/`7*is`/`3*is`, none equal to the contiguous
    // value), and asserts the copy still matches the layout-agnostic reference - both when the
    // surrounding axes fully coalesce to the 1D fast path and when the outer axis stays strided.
    #[test]
    fn copy_unit_axis() {
        let is = size_of::<u32>();
        for unit_stride in [0, 5 * is] {
            // Contiguous either side of the interior unit axis -> one 32-element run (fast path).
            check::<u32, DimDyn>(
                &[4, 1, 8],
                &[8 * is, unit_stride, is],
                &[8 * is, unit_stride, is],
                0,
            );
            // Strided outer axis (2x gap) -> unit axis folds out, leaving the 1D strided path.
            check::<u32, DimDyn>(
                &[4, 1, 8],
                &[16 * is, unit_stride, is],
                &[16 * is, unit_stride, is],
                0,
            );
        }
        // Leading and trailing unit axes with arbitrary strides, contiguous elsewhere.
        check::<u32, DimDyn>(&[1, 4, 8], &[7 * is, 8 * is, is], &[7 * is, 8 * is, is], 0);
        check::<u32, DimDyn>(&[4, 8, 1], &[8 * is, is, 3 * is], &[8 * is, is, 3 * is], 0);
        // A misaligned base drives the unaligned path; `u64` covers a wider specialization.
        check::<u32, DimDyn>(&[4, 1, 8], &[8 * is, 0, is], &[8 * is, 0, is], 1);
        let js = size_of::<u64>();
        check::<u64, DimDyn>(&[4, 1, 8], &[8 * js, 0, js], &[8 * js, 0, js], 0);
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
