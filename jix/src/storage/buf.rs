use std::ops::Range;

use crate::codec::TmpBuf;
use crate::dtype::Dtype;
use crate::util::default_strides_cast;
use crate::{default_strides_from_iter, Dimension, ReadContext, SliceExt};

/// Destination for a [`read_data`](crate::ArrayStorage::read_data) call.
///
/// One of:
/// - a caller-provided contiguous byte buffer ([`OutBuf::new`]),
/// - a lazily-allocated pooled buffer (`OutBuf::new_lazy`),
/// - a caller-provided *strided* byte buffer (`OutBuf::new_strided`) - a rectangular sub-region of some
///   larger destination, described by per-dimension byte strides.
pub struct OutBuf<'a> {
    data: OutBufInner<'a>,
    strides: Option<&'a [usize]>,
}

/// Internal representation of an [`OutBuf`].
pub(crate) enum OutBufInner<'a> {
    Borrowed(&'a mut [u8]),
    Lazy(&'a ReadContext),
    LazyAllocated(TmpBuf<'a>),
}

impl<'a> OutBuf<'a> {
    /// Write into a caller-provided contiguous buffer.
    #[inline(always)]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self {
            data: OutBufInner::Borrowed(buf),
            strides: None,
        }
    }

    /// Defer allocation: the storage allocates a pooled buffer from `context` on the first demand for
    /// a mutable slice.
    #[inline(always)]
    pub(crate) fn new_lazy(context: &'a ReadContext) -> Self {
        Self {
            data: OutBufInner::Lazy(context),
            strides: None,
        }
    }

    /// Write into a caller-provided *strided* buffer.
    ///
    /// `buf` is the destination byte buffer and `strides` gives the per-dimension byte stride of the
    /// region being written (row-major over the read `index`, but embedded in a larger buffer).
    ///
    /// # Safety
    ///
    /// The length of `buf` is **not** checked against `strides` (nor against the read `index` it will
    /// later be used with). The caller must ensure `buf` is valid for every strided write.
    #[inline(always)]
    pub(crate) unsafe fn new_strided(buf: &'a mut [u8], strides: &'a [usize]) -> Self {
        Self {
            data: OutBufInner::Borrowed(buf),
            strides: Some(strides),
        }
    }

    /// The writable destination bytes plus its byte-strides if it is strided (`None` when contiguous,
    /// in which case the slice is exactly the row-major output). A lazy buffer is materialized first.
    #[inline]
    pub(crate) fn get_mut<'b>(
        &'b mut self,
        nitems: usize,
        dtype: &Dtype,
    ) -> (&'b mut [u8], Option<&'b [usize]>) {
        self.materialize(nitems, dtype);
        let data = match &mut self.data {
            OutBufInner::Borrowed(buf) => buf,
            OutBufInner::LazyAllocated(tmp) => tmp.as_mut_slice(),
            OutBufInner::Lazy(_) => unreachable!(),
        };
        (data, self.strides)
    }

    /// Like [`get_mut`](Self::get_mut) but always returns concrete byte-strides: the buffer's own
    /// strides if it is strided, otherwise the C-order strides over the read shape.
    #[inline]
    pub(crate) fn get_strided_mut<'b, D: Dimension>(
        &'b mut self,
        index: &[Range<u64>],
        dtype: &Dtype,
    ) -> (&'b mut [u8], D::Vec<usize>) {
        let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
        let (buf, strides) = self.get_mut(nitems, dtype);
        let strides = match strides {
            Some(strides) => strides.to_dim_vec::<D>(),
            None => {
                let shape = index.iter().map(|r| (r.end - r.start) as usize);
                default_strides_from_iter::<D, _>(index.len(), shape, dtype.itemsize() as usize)
            }
        };
        (buf, strides)
    }

    /// Returns the buffer contents, or `None` for a not-yet-materialized lazy `OutBuf` or a strided
    /// `OutBuf` (neither has a single contiguous view).
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> Option<&[u8]> {
        let data = match &self.data {
            OutBufInner::Borrowed(buf) => buf,
            OutBufInner::LazyAllocated(tmp) => tmp.as_slice(),
            OutBufInner::Lazy(_) => return None,
        };
        self.strides.is_none().then_some(data)
    }

    #[track_caller]
    #[inline(always)]
    pub(crate) fn unwrap_tmp(self) -> TmpBuf<'a> {
        assert!(self.strides.is_none());
        match self.data {
            OutBufInner::LazyAllocated(tmp) => tmp,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn strides(&self) -> Option<&[usize]> {
        self.strides
    }

    /// The output byte-strides for a read whose output shape is `out_shape`: this buffer's own
    /// strides if it is strided, otherwise the C-order strides over `out_shape` for `itemsize`. Lets
    /// a shape op derive the strides to forward without branching on the buffer variant.
    #[inline]
    pub(crate) fn strides_or_default<D: Dimension>(
        &self,
        out_shape: &D::Vec<u64>,
        itemsize: usize,
    ) -> D::Vec<usize> {
        match self.strides() {
            Some(strides) => strides.to_dim_vec::<D>(),
            None => default_strides_cast(out_shape, itemsize),
        }
    }

    /// Re-wrap this buffer's bytes as a *strided* destination with `new_strides`, for a shape op that
    /// forwards its (single) output buffer to an inner read after remapping axes. A lazy buffer is
    /// first materialized (see [`materialize`](Self::materialize)) so there is always a concrete
    /// buffer to point at; this lets shape ops take one unified path whether or not the incoming
    /// buffer was already strided.
    ///
    /// # Safety
    ///
    /// Same contract as [`new_strided`](Self::new_strided): the length of the underlying buffer is
    /// **not** checked against `new_strides`. The caller must ensure it is valid for every strided
    /// write implied by `new_strides` and the read `index`.
    #[inline(always)]
    pub(crate) unsafe fn with_strides<'b>(
        &'b mut self,
        nitems: usize,
        dtype: &Dtype,
        new_strides: &'b [usize],
    ) -> OutBuf<'b> {
        self.materialize(nitems, dtype);
        OutBuf {
            data: match &mut self.data {
                OutBufInner::Borrowed(buf) => OutBufInner::Borrowed(buf),
                OutBufInner::LazyAllocated(tmp) => OutBufInner::Borrowed(tmp.as_mut_slice()),
                OutBufInner::Lazy(_) => unreachable!(),
            },
            strides: Some(new_strides),
        }
    }

    /// Materialize a lazy buffer into a pooled `LazyAllocated` in place, so it has a concrete backing
    /// slice. No-op for the already-materialized (borrowed / tmp) variants. Uses the lazy variant's
    /// own stored context.
    #[inline]
    pub(crate) fn materialize(&mut self, nitems: usize, dtype: &Dtype) {
        let ctx = match &self.data {
            OutBufInner::Lazy(ctx) => *ctx,
            _ => return,
        };
        self.data = OutBufInner::LazyAllocated(
            ctx.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::OutBuf;
    use crate::dtype::{Dtype, Dtyped};
    use crate::{DimDyn, ReadContext};

    fn i32_dtype() -> Dtype {
        <i32 as Dtyped>::DTYPE
    }

    /// View a mutable `i32` slice as bytes (and thus `i32`-aligned).
    fn as_bytes_mut(v: &mut [i32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v))
        }
    }

    /// A contiguous destination has no strides of its own, so `get_strided_mut` synthesizes the
    /// C-order (row-major) byte strides over the read shape.
    #[test]
    fn get_strided_mut_synthesizes_c_order_strides_for_contiguous() {
        let dtype = i32_dtype();
        let mut backing = vec![0i32; 6];
        let mut ob = OutBuf::new(as_bytes_mut(&mut backing));
        let (buf, strides) = ob.get_strided_mut::<DimDyn>(&[0..2, 0..3], &dtype);
        assert_eq!(buf.len(), 24);
        assert_eq!(strides.as_ref(), [12, 4]); // [3 * 4, 4]
    }

    /// A strided destination reports its own strides verbatim - `read_data` writes each element
    /// straight to its final address rather than staging through a contiguous scratch.
    #[test]
    fn get_strided_mut_reports_own_strides_for_strided() {
        let dtype = i32_dtype();
        // Row stride 16 bytes (one i32 gap per row) instead of the C-order 12.
        let strides = [16usize, 4];
        let mut backing = vec![0i32; 8];
        let mut ob = unsafe { OutBuf::new_strided(as_bytes_mut(&mut backing), &strides) };
        let (_buf, got) = ob.get_strided_mut::<DimDyn>(&[0..2, 0..3], &dtype);
        assert_eq!(got.as_ref(), strides);
    }

    /// A lazy destination is materialized on demand into a pooled buffer sized for the read, and
    /// reports C-order strides like any other contiguous buffer.
    #[test]
    fn get_strided_mut_materializes_lazy() {
        let dtype = i32_dtype();
        let ctx = ReadContext::default();
        let mut ob = OutBuf::new_lazy(&ctx);
        assert!(ob.as_slice().is_none()); // not yet materialized
        let (buf, strides) = ob.get_strided_mut::<DimDyn>(&[0..2, 0..3], &dtype);
        assert_eq!(buf.len(), 24);
        assert_eq!(strides.as_ref(), [12, 4]);
        assert_eq!(ob.as_slice().map(<[u8]>::len), Some(24));
    }

    /// `with_strides` re-points a buffer at new strides (what a shape op does when it forwards its
    /// output buffer to an inner read after remapping axes), whatever variant it started as.
    #[test]
    fn with_strides_repoints_any_variant() {
        let dtype = i32_dtype();
        let remapped = [4usize, 8]; // e.g. a transposed view of a [2, 2] i32 region
        let mut backing = vec![0i32; 4];
        let mut ob = OutBuf::new(as_bytes_mut(&mut backing));
        let inner = unsafe { ob.with_strides(4, &dtype, &remapped) };
        assert_eq!(inner.strides(), Some(remapped.as_slice()));

        let ctx = ReadContext::default();
        let mut lazy = OutBuf::new_lazy(&ctx);
        let inner = unsafe { lazy.with_strides(4, &dtype, &remapped) };
        assert_eq!(inner.strides(), Some(remapped.as_slice()));
    }
}
