use crate::{dtype::Dtype, NdCopier};

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
    unsafe { NdCopier::new(dtype).copy(src, dst, shape, src_strides, dst_strides, dtype) }
}
