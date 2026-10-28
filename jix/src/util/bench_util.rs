use crate::{dtype::Dtype, NdCopier, PtrMutNoalias, PtrNoalias};

/// Copy `shape` elements from `src` to `dst`, using the given strides and data type.
///
/// # SAFETY
///
/// The caller must ensure that `src`/`dst` are byte slices covering every byte the copy touches for
/// the given shape and strides (the region spans `strided_span_bytes` bytes forward from each slice
/// start), and that the source and destination regions do not overlap.
#[doc(hidden)]
pub unsafe fn nd_copy(
    src: &[u8],
    dst: &mut [u8],
    shape: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
    dtype: &Dtype,
) {
    unsafe {
        NdCopier::new(dtype).copy(
            PtrNoalias::from_slice(src),
            PtrMutNoalias::from_slice(dst),
            shape,
            src_strides,
            dst_strides,
            dtype,
        )
    }
}

/// Write `idx . weights` (a `u64`) at every index of `shape` into the strided `dst`, through
/// `NdIter::for_each`: the per-element loop of [`Array::compact_fn`](crate::Array::compact_fn).
/// A 3-dim `shape` runs the `Dim<3>` instantiation, any other the `DimDyn` one.
///
/// # SAFETY
///
/// `dst` must cover every byte the walk touches: `shape` and `strides` (in bytes, multiples of 8)
/// must keep every element inside it, 8-byte aligned.
#[doc(hidden)]
pub unsafe fn nd_iter_fill(dst: &mut [u8], shape: &[u64], strides: &[usize], weights: &[u64]) {
    use crate::util::iter::NdIter;
    use crate::{Dim, DimDyn, SliceExt};

    #[inline(always)]
    fn fill<D: crate::Dimension>(
        base: *mut u8,
        shape: D::Vec<u64>,
        strides: D::Vec<usize>,
        w: &[u64],
    ) where
        D::Vec<u64>: crate::DimVec<u64, Dimension = D>,
    {
        NdIter::builder(shape)
            .with_strides_offset_ext(strides, 0)
            .build()
            .for_each(|idx, offset| {
                let value = (0..w.len()).map(|d| idx[d] * w[d]).sum::<u64>();
                unsafe { base.add(offset).cast::<u64>().write(value) };
            });
    }

    assert!(shape.len() == strides.len() && shape.len() == weights.len());
    let base = dst.as_mut_ptr();
    if shape.len() == 3 {
        let w: &[u64; 3] = weights.try_into().unwrap();
        fill::<Dim<3>>(
            base,
            [shape[0], shape[1], shape[2]],
            [strides[0], strides[1], strides[2]],
            w,
        );
    } else {
        fill::<DimDyn>(
            base,
            shape.to_dim_vec::<DimDyn>(),
            strides.to_dim_vec::<DimDyn>(),
            weights,
        );
    }
}
