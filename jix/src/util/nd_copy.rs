use crate::arrayvec::ArrayVec;
use crate::dtype::{Dtype, Itemsize};
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::iter::NdIter;
use crate::{DimArray, DimDyn, Dimension, SliceExt};

/// A reusable, dtype-specialized copier that moves a rectangular n-dimensional region between two
/// raw byte buffers under independent source and destination strides.
///
/// [`new`](Self::new) inspects the dtype once and picks the cheapest copy routine up front: a
/// monomorphized scalar copy for the common power-of-two `(itemsize, alignment)` pairs, a
/// field-by-field copy for structs that decompose into at most four scalar fields, or a generic
/// byte-wise fallback for everything else. Each [`copy`](Self::copy) then moves one region by
/// walking `shape` with an [`NdIter`] and copying the appropriate number of bytes at each element.
pub(crate) struct NdCopier<'a, D: Dimension>(NdCopierInner<'a, D>);
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
    #[inline]
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
    fn create_scalar_fn(dtype: &Dtype) -> Option<fn(NdCopyArgs<D>)> {
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

    fn scalar_fn<T: 'static>(args: NdCopyArgs<D>) {
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

        let shape = shape.as_ref();
        let ndim = shape.len();
        assert_eq!(ndim, src_strides.as_ref().len());
        assert_eq!(ndim, dst_strides.as_ref().len());

        // copy more then itemsize if the last dim(s) is contiguous
        let n_continuous_dims = (0..ndim)
            .rev()
            .scan(size_of::<T>(), |expected_stride, dim| {
                let is_contiguous =
                    src_strides[dim] == *expected_stride && dst_strides[dim] == *expected_stride;
                *expected_stride *= shape[dim];
                Some(is_contiguous)
            })
            .take_while(|&is_contiguous| is_contiguous)
            .count();
        let itemsize = size_of::<T>() * shape[ndim - n_continuous_dims..].iter().product::<usize>();
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
        assert_eq!(ndim, src_strides.as_ref().len());
        assert_eq!(ndim, dst_strides.as_ref().len());
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

    // Run a single copy through `NdCopier<D>` for element type `T` and assert it
    // byte-matches the independent reference.
    fn check<T: Dtyped, D: Dimension>(
        shape: &[usize],
        src_strides: &[usize],
        dst_strides: &[usize],
    ) {
        let dtype = T::DTYPE;
        let itemsize = dtype.itemsize() as usize;
        assert_eq!(src_strides.len(), shape.len());
        assert_eq!(dst_strides.len(), shape.len());

        let src_len = buf_len(shape, src_strides, itemsize);
        let dst_len = buf_len(shape, dst_strides, itemsize);

        let mut src = vec![0u8; src_len];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }

        let expected = reference_copy(&src, dst_len, shape, src_strides, dst_strides, itemsize);

        let mut actual = vec![0u8; dst_len];
        let copier = NdCopier::<D>::new(&dtype);
        let shape_v = D::vec(shape.len(), |i| shape[i]);
        let src_v = D::vec(shape.len(), |i| src_strides[i]);
        let dst_v = D::vec(shape.len(), |i| dst_strides[i]);
        unsafe {
            copier.copy(
                src.as_ptr(),
                actual.as_mut_ptr(),
                &shape_v,
                &src_v,
                &dst_v,
                &dtype,
            );
        }

        assert_eq!(
            actual, expected,
            "itemsize={itemsize} shape={shape:?} src_strides={src_strides:?} dst_strides={dst_strides:?}"
        );
    }

    // For a fixed (element type, dimension type, shape), exercise the four
    // contiguous/strided combinations of source and destination.
    fn check_layouts<T: Dtyped, D: Dimension>(shape: &[usize]) {
        let itemsize = T::DTYPE.itemsize() as usize;
        let ndim = shape.len();
        let ones = vec![1usize; ndim];
        let mult_a: Vec<usize> = (0..ndim).map(|d| if d % 2 == 0 { 2 } else { 1 }).collect();
        let mult_b: Vec<usize> = (0..ndim).map(|d| d + 2).collect();
        let cont = strided_strides(shape, &ones, itemsize);
        let src_strided = strided_strides(shape, &mult_a, itemsize);
        let dst_strided = strided_strides(shape, &mult_b, itemsize);

        check::<T, D>(shape, &cont, &cont); // contiguous src + dst (fully coalesced)
        check::<T, D>(shape, &src_strided, &cont); // strided src, contiguous dst
        check::<T, D>(shape, &cont, &dst_strided); // contiguous src, strided dst
        check::<T, D>(shape, &src_strided, &dst_strided); // strided src + dst
    }

    // The full dimensionality matrix (0D, 1D, 2D, 3D, 4D, and the same shapes as
    // `DimDyn`) for one element type, each with contiguous and strided layouts.
    fn check_all_dims<T: Dtyped>() {
        check_layouts::<T, Dim<0>>(&[]);
        check_layouts::<T, DimDyn>(&[]);

        check_layouts::<T, Dim<1>>(&[7]);
        check_layouts::<T, DimDyn>(&[7]);

        check_layouts::<T, Dim<2>>(&[3, 4]);
        check_layouts::<T, DimDyn>(&[3, 4]);

        check_layouts::<T, Dim<3>>(&[2, 3, 4]);
        check_layouts::<T, DimDyn>(&[2, 3, 4]);

        check_layouts::<T, Dim<4>>(&[2, 2, 3, 3]);
        check_layouts::<T, DimDyn>(&[2, 2, 3, 3]);
    }

    // ---- Scalar element types (the specialized `scalar_fn` paths) ----
    // (1, 1) -> u8: i8, u8, bool. (2, 2) -> u16: i16, u16, f16.
    // (4, 4) -> u32: i32, u32, f32. (8, 8) -> u64: i64, u64, f64.

    #[test]
    fn copy_i8() {
        check_all_dims::<i8>();
    }
    #[test]
    fn copy_u8() {
        check_all_dims::<u8>();
    }
    #[test]
    fn copy_bool() {
        check_all_dims::<bool>();
    }
    #[test]
    fn copy_i16() {
        check_all_dims::<i16>();
    }
    #[test]
    fn copy_u16() {
        check_all_dims::<u16>();
    }
    #[test]
    fn copy_i32() {
        check_all_dims::<i32>();
    }
    #[test]
    fn copy_u32() {
        check_all_dims::<u32>();
    }
    #[test]
    fn copy_f32() {
        check_all_dims::<f32>();
    }
    #[test]
    fn copy_i64() {
        check_all_dims::<i64>();
    }
    #[test]
    fn copy_u64() {
        check_all_dims::<u64>();
    }
    #[test]
    fn copy_f64() {
        check_all_dims::<f64>();
    }

    #[cfg(feature = "half")]
    #[test]
    fn copy_f16() {
        check_all_dims::<crate::scalar::f16>();
    }

    // Complex hits the two (itemsize, alignment) cases no other scalar reaches:
    // Complex<f32> -> (8, 4) via [u32; 2]; Complex<f64> -> (16, 8) via [u64; 2].
    #[cfg(feature = "num-complex")]
    #[test]
    fn copy_complex_f32() {
        check_all_dims::<crate::scalar::Complex<f32>>();
    }
    #[cfg(feature = "num-complex")]
    #[test]
    fn copy_complex_f64() {
        check_all_dims::<crate::scalar::Complex<f64>>();
    }

    // ---- Non-scalar dtypes: the `copy_dynamic` fallback ----
    // Array-of-scalar dtypes carry an inner shape, so they miss every specialized
    // (itemsize, alignment) case and are not decomposed as structs either. `[i32; 3]`
    // gives itemsize 12; `[u8; 3]` gives an odd itemsize (3) with no power-of-two run.

    #[test]
    fn copy_dynamic_itemsize_12() {
        check_all_dims::<[i32; 3]>();
    }
    #[test]
    fn copy_dynamic_odd_itemsize() {
        check_all_dims::<[u8; 3]>();
    }

    // ---- Struct dtype: the `copy_struct` path ----

    #[test]
    fn copy_struct() {
        check_all_dims::<StructNoPad>();
    }
}
