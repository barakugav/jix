use crate::arrayvec::ArrayVec;
use crate::dtype::{Dtype, Itemsize};
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::iter::NdIter;
use crate::{Dim, Dimension, SliceExt};

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
    src: *const u8,
    dst: *mut u8,
    shape: &'a [usize],
    src_strides: &'a [usize],
    dst_strides: &'a [usize],
    dtype: &'a Dtype,
}
impl Clone for NdCopyArgs<'_> {
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
            mut shape,
            mut src_strides, // TODO accept Option<>,
            mut dst_strides,
            dtype,
        } = args;
        debug_assert_eq!(size_of::<T>(), dtype.itemsize() as usize);
        debug_assert_eq!(align_of::<T>(), dtype.alignment().as_usize());

        let ndim = shape.len();
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
                _ => Self::copy_1d_unsized::<T>(
                    src,
                    dst,
                    len,
                    src_stride,
                    dst_stride,
                    n_continuous_items,
                    aligned,
                ),
            }
        }
    }

    #[inline(never)]
    unsafe fn copy_1d_unsized<T: Copy>(
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_stride: usize,
        dst_stride: usize,
        n_continuous_items: usize,
        aligned: bool,
    ) {
        unsafe {
            if aligned {
                let src_continuous = len <= 1 || src_stride == size_of::<T>() * n_continuous_items;
                let dst_continuous = len <= 1 || dst_stride == size_of::<T>() * n_continuous_items;
                if src_continuous && dst_continuous {
                    std::ptr::copy_nonoverlapping::<T>(
                        src.cast::<T>(),
                        dst.cast::<T>(),
                        len * n_continuous_items,
                    );
                } else if src_continuous {
                    let src = src.cast::<T>();
                    for i in 0..len {
                        let src = src.add(i * n_continuous_items);
                        let dst = dst.add(i * dst_stride).cast::<T>();
                        std::ptr::copy_nonoverlapping::<T>(src, dst, n_continuous_items);
                    }
                } else if dst_continuous {
                    let dst = dst.cast::<T>();
                    for i in 0..len {
                        let src = src.add(i * src_stride).cast::<T>();
                        let dst = dst.add(i * n_continuous_items);
                        std::ptr::copy_nonoverlapping::<T>(src, dst, n_continuous_items);
                    }
                } else {
                    for i in 0..len {
                        let src = src.add(i * src_stride).cast::<T>();
                        let dst = dst.add(i * dst_stride).cast::<T>();
                        std::ptr::copy_nonoverlapping::<T>(src, dst, n_continuous_items);
                    }
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
        let src_continuous = len <= 1 || src_stride == size_of::<T>() * N_CONTINUOUS;
        let dst_continuous = len <= 1 || dst_stride == size_of::<T>() * N_CONTINUOUS;
        if src_continuous && dst_continuous {
            unsafe {
                std::ptr::copy_nonoverlapping::<[T; N_CONTINUOUS]>(
                    src.cast::<[T; N_CONTINUOUS]>(),
                    dst.cast::<[T; N_CONTINUOUS]>(),
                    len,
                )
            };
        } else if src_continuous {
            let src = src.cast::<[T; N_CONTINUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i);
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTINUOUS]>();
                    dst.write(src.read());
                }
            }
        } else if dst_continuous {
            let dst = dst.cast::<[T; N_CONTINUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTINUOUS]>();
                    let dst = dst.add(i);
                    dst.write(src.read());
                }
            }
        } else {
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTINUOUS]>();
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTINUOUS]>();
                    dst.write(src.read());
                }
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
        let src_continuous = len <= 1 || src_stride == size_of::<T>() * N_CONTINUOUS;
        let dst_continuous = len <= 1 || dst_stride == size_of::<T>() * N_CONTINUOUS;
        if src_continuous && dst_continuous {
            let src = src.cast::<[T; N_CONTINUOUS]>();
            let dst = dst.cast::<[T; N_CONTINUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i);
                    let dst = dst.add(i);
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        } else if src_continuous {
            let src = src.cast::<[T; N_CONTINUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i);
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTINUOUS]>();
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        } else if dst_continuous {
            let dst = dst.cast::<[T; N_CONTINUOUS]>();
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTINUOUS]>();
                    let dst = dst.add(i);
                    dst.write_unaligned(src.read_unaligned());
                }
            }
        } else {
            for i in 0..len {
                unsafe {
                    let src = src.add(i * src_stride).cast::<[T; N_CONTINUOUS]>();
                    let dst = dst.add(i * dst_stride).cast::<[T; N_CONTINUOUS]>();
                    dst.write_unaligned(src.read_unaligned());
                }
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
    fn copy_struct(struct_copier: &NdCopierStruct, args: NdCopyArgs) {
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

    fn copy_dynamic(args: NdCopyArgs) {
        let NdCopyArgs {
            src,
            dst,
            mut shape,
            mut src_strides,
            mut dst_strides,
            dtype,
        } = args;

        let ndim = shape.len();
        assert!(ndim == src_strides.len() && ndim == dst_strides.len());
        let mut n_continuous_bytes = dtype.itemsize() as usize;

        // copy more then itemsize if the last dim(s) is contiguous
        let n_continuous_dims = (0..ndim)
            .rev()
            .scan(dtype.itemsize() as usize, |expected_stride, dim| {
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
            n_continuous_bytes *= shape[n_strided_dims..].iter().product::<usize>();
            shape = &shape[..n_strided_dims];
            src_strides = &src_strides[..n_strided_dims];
            dst_strides = &dst_strides[..n_strided_dims];
        }
        if shape.len() == 0 {
            shape = &[1];
            src_strides = &[0];
            dst_strides = &[0];
        }

        unsafe fn copy_dynamic_inner<D: Dimension>(
            shape: D::Vec<u64>,
            src_strides: D::Vec<usize>,
            dst_strides: D::Vec<usize>,
            src: *const u8,
            dst: *mut u8,
            n_continuous_bytes: usize,
        ) {
            let iter = NdIter::new(
                shape,
                (
                    NdIterExtStridesPtr::new(src_strides, src),
                    NdIterExtStridesPtrMut::new(dst_strides, dst),
                ),
            );
            for (_, (src_ptr, dst_ptr)) in iter {
                unsafe {
                    std::ptr::copy_nonoverlapping::<u8>(src_ptr, dst_ptr, n_continuous_bytes);
                }
            }
        }

        unsafe {
            match shape.len() {
                0 => unreachable!(),
                1 => copy_dynamic_inner::<Dim<1>>(
                    Dim::<1>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<1>>(),
                    dst_strides.to_dim_vec::<Dim<1>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                2 => copy_dynamic_inner::<Dim<2>>(
                    Dim::<2>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<2>>(),
                    dst_strides.to_dim_vec::<Dim<2>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                3 => copy_dynamic_inner::<Dim<3>>(
                    Dim::<3>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<3>>(),
                    dst_strides.to_dim_vec::<Dim<3>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                4 => copy_dynamic_inner::<Dim<4>>(
                    Dim::<4>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<4>>(),
                    dst_strides.to_dim_vec::<Dim<4>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                5 => copy_dynamic_inner::<Dim<5>>(
                    Dim::<5>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<5>>(),
                    dst_strides.to_dim_vec::<Dim<5>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                6 => copy_dynamic_inner::<Dim<6>>(
                    Dim::<6>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<6>>(),
                    dst_strides.to_dim_vec::<Dim<6>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                7 => copy_dynamic_inner::<Dim<7>>(
                    Dim::<7>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<7>>(),
                    dst_strides.to_dim_vec::<Dim<7>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                8 => copy_dynamic_inner::<Dim<8>>(
                    Dim::<8>::vec(shape.len(), |i| shape[i] as u64),
                    src_strides.to_dim_vec::<Dim<8>>(),
                    dst_strides.to_dim_vec::<Dim<8>>(),
                    src,
                    dst,
                    n_continuous_bytes,
                ),
                _ => unimplemented!(),
            }
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
    // `n_continuous == k` and drives each specialized `copy_1d` arm (k in 1/2/4/8/16/32/64, subject
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
            check::<T>(&shape, &src_strides, &dst_strides, misalign);
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
