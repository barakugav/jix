use std::ops::Range;

use crate::codec::TmpBuf;
use crate::dtype::Dtype;
use crate::error::{bail, check_buffer_aligned, Result};
use crate::util::{default_strides_cast, default_strides_slice, NdCopier};
use crate::{default_strides_from_iter, Dimension, ReadContext, SliceExt};

/// Destination for a [`read_data`](crate::ArrayStorage::read_data) call.
///
/// One of:
/// - a caller-provided contiguous byte buffer ([`OutBuf::new`]),
/// - a lazily-allocated pooled buffer (`OutBuf::new_lazy`),
/// - a caller-provided *strided* byte buffer (`OutBuf::new_strided`) - a rectangular sub-region of some
///   larger destination, described by per-dimension byte strides.
pub struct OutBuf<'a>(pub(crate) OutBufInner<'a>);

/// Internal representation of an [`OutBuf`].
///
/// A backend that can honor arbitrary strides writes straight into the destination via `get_mut` /
/// `get_strided_mut` (a shape op forwarding a read to an inner array remaps the strides with
/// `with_strides`). A backend that can only emit contiguous row-major output instead takes a scratch
/// buffer from `get_contiguous_mut` and calls `OutBufContiguousBuf::finalize` to scatter it into a
/// strided destination - folding what would otherwise be a separate copy at the call site into the
/// single unavoidable one.
pub(crate) enum OutBufInner<'a> {
    Contiguous(&'a mut [u8]),
    ContiguousLazy(&'a ReadContext),
    ContiguousLazyAllocated(TmpBuf<'a>),
    Strided {
        buf: &'a mut [u8],
        strides: &'a [usize],
    },
}

impl<'a> OutBuf<'a> {
    /// Write into a caller-provided contiguous buffer.
    #[inline(always)]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self(OutBufInner::Contiguous(buf))
    }

    /// Defer allocation: the storage allocates a pooled buffer from `context` on the first demand for
    /// a mutable slice.
    #[inline(always)]
    pub(crate) fn new_lazy(context: &'a ReadContext) -> Self {
        Self(OutBufInner::ContiguousLazy(context))
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
        Self(OutBufInner::Strided { buf, strides })
    }

    /// Obtain a contiguous, writable buffer for a `read_data` whose row-major output shape is
    /// `out_shape` (element counts per dimension), for `dtype`.
    ///
    /// The returned [`OutBufContiguousBuf`] exposes a contiguous `&mut [u8]` of exactly
    /// `out_shape.product() * dtype.itemsize()` bytes via
    /// [`as_mut_slice`](OutBufContiguousBuf::as_mut_slice). For the contiguous and lazy variants
    /// that slice IS the real destination and [`finalize`](OutBufContiguousBuf::finalize) is a
    /// no-op.
    ///
    /// For the strided variant there are two cases. When the destination's strides equal the
    /// C-contiguous (row-major) strides for `out_shape` it is really contiguous, so - provided it
    /// is large enough and aligned - the slice IS the destination and `finalize` is a no-op, with
    /// no scratch allocation or scatter. Otherwise the slice is a temporary contiguous scratch
    /// buffer, allocated here from `context`; the caller fills it and then calls `finalize` to
    /// scatter it into the real strided destination. Skipping that final call (e.g. by returning
    /// early on error) leaves the destination untouched.
    ///
    /// `context` is used only to allocate scratch for the (genuinely) strided variant; the lazy
    /// variant keeps using its own stored context.
    #[inline]
    pub(crate) fn get_contiguous_mut<'b>(
        &'b mut self,
        out_shape: &[usize],
        dtype: &Dtype,
        context: &'b ReadContext,
    ) -> Result<OutBufContiguousBuf<'b>> {
        let itemsize = dtype.itemsize() as usize;
        let nbytes = out_shape.iter().product::<usize>() * itemsize;

        // ContiguousLazy: materialize into an owned `ContiguousLazyAllocated` in place, so a later `unwrap_tmp`/`as_slice` sees it.
        let lazy_ctx = match &self.0 {
            OutBufInner::ContiguousLazy(ctx) => Some(*ctx),
            _ => None,
        };
        if let Some(lazy_ctx) = lazy_ctx {
            let tmp = lazy_ctx.tmp_buf(nbytes, dtype.alignment());
            self.0 = OutBufInner::ContiguousLazyAllocated(tmp);
        }

        // Strided: take the destination out (leaving a harmless empty placeholder).
        if matches!(self.0, OutBufInner::Strided { .. }) {
            let taken = std::mem::replace(&mut self.0, OutBufInner::Contiguous(&mut []));
            let OutBufInner::Strided { buf, strides } = taken else {
                unreachable!()
            };
            // Fast path: a strided destination whose strides equal the C-contiguous (row-major)
            // strides for `out_shape` is really contiguous - the write lands in `buf[..nbytes]`
            // with no gaps. Hand that slice out directly, skipping both the scratch allocation and
            // the `finalize` scatter. Falls back to the scatter path if the buffer is too short or
            // insufficiently aligned to be used as a typed destination.
            let is_contiguous = strides == default_strides_slice(out_shape, itemsize).as_ref();
            let aligned = (buf.as_ptr() as usize).is_multiple_of(dtype.alignment().as_usize());
            if is_contiguous && buf.len() >= nbytes && aligned {
                return Ok(OutBufContiguousBuf::Direct(&mut buf[..nbytes]));
            }
            // Fallback: allocate a contiguous scratch buffer to fill; `finalize` scatters it into
            // `strided_buf`.
            let contiguous_buf = context.tmp_buf(nbytes, dtype.alignment());
            return Ok(OutBufContiguousBuf::Strided {
                strided_buf: buf,
                strides,
                contiguous_buf,
            });
        }

        // Contiguous (caller-provided, or the lazy buffer just materialized above): hand out the
        // slice directly.
        let mut buf = match &mut self.0 {
            OutBufInner::Contiguous(buf) => OutBufContiguousBuf::Direct(buf),
            OutBufInner::ContiguousLazyAllocated(tmp) => {
                OutBufContiguousBuf::Direct(tmp.as_mut_slice())
            }
            OutBufInner::ContiguousLazy(_) | OutBufInner::Strided { .. } => unreachable!(),
        };
        if buf.as_mut_slice().len() != nbytes {
            #[inline(never)]
            fn buffer_size_fail(buf_len: usize, nbytes: usize, dtype: &Dtype) -> Result<()> {
                bail!(
                    InvalidBufferSize,
                    "Unexpected buffer size {buf_len} requested for {nbytes} bytes with dtype {dtype}"
                );
            }
            buffer_size_fail(buf.as_mut_slice().len(), nbytes, dtype)?;
        }
        check_buffer_aligned(buf.as_mut_slice().as_ptr(), dtype)?;
        Ok(buf)
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
        match &mut self.0 {
            OutBufInner::Contiguous(buf) => (buf, None),
            OutBufInner::Strided { buf, strides } => (buf, Some(strides)),
            OutBufInner::ContiguousLazyAllocated(tmp) => (tmp.as_mut_slice(), None),
            OutBufInner::ContiguousLazy(_) => unreachable!(),
        }
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
        match &self.0 {
            OutBufInner::Contiguous(buf) => Some(buf),
            OutBufInner::ContiguousLazyAllocated(tmp) => Some(tmp.as_slice()),
            OutBufInner::ContiguousLazy(_) | OutBufInner::Strided { .. } => None,
        }
    }

    #[track_caller]
    #[inline(always)]
    pub(crate) fn unwrap_tmp(self) -> TmpBuf<'a> {
        match self.0 {
            OutBufInner::ContiguousLazyAllocated(tmp) => tmp,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn strides(&self) -> Option<&[usize]> {
        match &self.0 {
            OutBufInner::Strided { strides, .. } => Some(strides),
            _ => None,
        }
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
        let buf = match &mut self.0 {
            OutBufInner::Contiguous(items) => items,
            OutBufInner::Strided { buf, strides: _ } => buf,
            OutBufInner::ContiguousLazyAllocated(tmp_buf) => tmp_buf.as_mut_slice(),
            OutBufInner::ContiguousLazy(_) => unreachable!(),
        };
        unsafe { OutBuf::new_strided(buf, new_strides) }
    }

    /// Materialize a lazy buffer into a pooled `ContiguousLazyAllocated` in place, so it has a concrete backing slice.
    /// No-op for the already-materialized (borrowed / tmp / strided) variants. Uses the lazy
    /// variant's own stored context.
    #[inline]
    pub(crate) fn materialize(&mut self, nitems: usize, dtype: &Dtype) {
        let ctx = match &self.0 {
            OutBufInner::ContiguousLazy(ctx) => *ctx,
            _ => return,
        };
        self.0 = OutBufInner::ContiguousLazyAllocated(
            ctx.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment()),
        );
    }
}

/// A contiguous, writable view obtained from [`OutBuf::get_contiguous_mut`].
///
/// Fill the contiguous `&mut [u8]` from [`as_mut_slice`](Self::as_mut_slice), then call
/// [`finalize`](Self::finalize). For a [`Direct`](Self::Direct) destination the slice IS the
/// destination and `finalize` is a no-op; for a [`Strided`](Self::Strided) destination the slice is a
/// scratch buffer that `finalize` scatters into the real destination. Skipping `finalize` (e.g. by
/// returning early on error) leaves the destination untouched.
pub(crate) enum OutBufContiguousBuf<'b> {
    /// The contiguous slice is the real destination; `finalize` does nothing.
    Direct(&'b mut [u8]),
    /// The write is staged in `contiguous_buf` and scattered into `strided_buf` (using `strides`) by
    /// `finalize`.
    Strided {
        strided_buf: &'b mut [u8],
        strides: &'b [usize],
        contiguous_buf: TmpBuf<'b>,
    },
}
impl OutBufContiguousBuf<'_> {
    /// The contiguous, writable buffer to fill: the real destination for [`Direct`](Self::Direct),
    /// or the scratch buffer for [`Strided`](Self::Strided).
    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            OutBufContiguousBuf::Direct(buf) => buf,
            OutBufContiguousBuf::Strided { contiguous_buf, .. } => contiguous_buf.as_mut_slice(),
        }
    }

    /// Finalize the write. For a [`Strided`](Self::Strided) destination, scatter the filled scratch
    /// buffer into the real destination; `shape` is the row-major output shape the scratch holds. For
    /// [`Direct`](Self::Direct), a no-op. Not calling this leaves a strided destination untouched, so
    /// an early return on error never performs a partial scatter.
    #[inline]
    pub(crate) fn finalize(self, shape: &[usize], dtype: &Dtype) {
        let OutBufContiguousBuf::Strided {
            strided_buf,
            strides,
            contiguous_buf,
        } = self
        else {
            return;
        };
        let itemsize = dtype.itemsize() as usize;
        let src_strides = default_strides_slice(shape, itemsize);
        let copier = NdCopier::new(dtype);
        // SAFETY: `contiguous_buf` is a fresh, disjoint allocation holding `shape` in row-major
        // order; `strided_buf`/`strides` describe the same-shaped region inside the caller's buffer,
        // which `OutBuf::new_strided`'s (unsafe) contract requires to be valid for these writes.
        unsafe {
            copier.copy(
                contiguous_buf.as_slice().as_ptr(),
                strided_buf.as_mut_ptr(),
                shape,
                src_strides.as_ref(),
                strides,
                dtype,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OutBuf, OutBufContiguousBuf};
    use crate::dtype::{Dtype, Dtyped};
    use crate::ReadContext;

    fn i32_dtype() -> Dtype {
        <i32 as Dtyped>::DTYPE
    }

    /// View a mutable `i32` slice as bytes. The buffer is `i32`-aligned, which the
    /// contiguous fast path requires.
    fn as_bytes_mut(v: &mut [i32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v))
        }
    }

    fn write_i32s(dst: &mut [u8], vals: impl IntoIterator<Item = i32>) {
        for (i, v) in vals.into_iter().enumerate() {
            dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
        }
    }

    /// A strided destination whose strides are the C-contiguous defaults for the shape is
    /// written straight through: `get_contiguous_mut` returns `Direct` (no scratch) and
    /// `finalize` performs no scatter.
    #[test]
    fn strided_with_default_strides_writes_directly_no_scatter() {
        let dtype = i32_dtype();
        let ctx = ReadContext::default();
        let shape = [2usize, 3];
        // C-default byte strides for [2, 3] of i32: [3*4, 4] = [12, 4].
        let strides = [12usize, 4];
        let mut backing = vec![0i32; 6]; // exactly the 24-byte region, i32-aligned
        {
            let bytes = as_bytes_mut(&mut backing);
            let mut ob = unsafe { OutBuf::new_strided(bytes, &strides) };
            let mut cbuf = ob.get_contiguous_mut(&shape, &dtype, &ctx).unwrap();
            assert!(matches!(cbuf, OutBufContiguousBuf::Direct(_)));
            write_i32s(cbuf.as_mut_slice(), 1..=6);
            cbuf.finalize(&shape, &dtype); // no-op for Direct
        }
        assert_eq!(backing, [1, 2, 3, 4, 5, 6]);
    }

    /// The fast path uses only the leading `nbytes` of an over-sized destination, matching how
    /// callers hand in `&mut buf[offset..]` (a strided sub-region base). Trailing bytes are
    /// left untouched.
    #[test]
    fn strided_default_into_larger_buffer_uses_prefix() {
        let dtype = i32_dtype();
        let ctx = ReadContext::default();
        let shape = [2usize, 2];
        let strides = [8usize, 4]; // C-default for [2, 2] of i32
        let mut backing = vec![7i32; 6]; // 24 bytes; the region needs only 16
        {
            let bytes = as_bytes_mut(&mut backing);
            let mut ob = unsafe { OutBuf::new_strided(bytes, &strides) };
            let mut cbuf = ob.get_contiguous_mut(&shape, &dtype, &ctx).unwrap();
            assert!(matches!(cbuf, OutBufContiguousBuf::Direct(_)));
            assert_eq!(cbuf.as_mut_slice().len(), 16);
            write_i32s(cbuf.as_mut_slice(), 1..=4);
            cbuf.finalize(&shape, &dtype);
        }
        assert_eq!(backing, [1, 2, 3, 4, 7, 7]);
    }

    /// A genuinely strided destination (gaps between rows) stages through a contiguous scratch
    /// buffer and is scattered into place by `finalize`.
    #[test]
    fn strided_with_gaps_scatters_via_finalize() {
        let dtype = i32_dtype();
        let ctx = ReadContext::default();
        let shape = [2usize, 3];
        // Row stride 16 bytes (one i32 gap per row) instead of the default 12.
        let strides = [16usize, 4];
        let mut backing = vec![0i32; 8]; // 32 bytes; max element offset is 16 + 8 = 24
        {
            let bytes = as_bytes_mut(&mut backing);
            let mut ob = unsafe { OutBuf::new_strided(bytes, &strides) };
            let mut cbuf = ob.get_contiguous_mut(&shape, &dtype, &ctx).unwrap();
            assert!(matches!(cbuf, OutBufContiguousBuf::Strided { .. }));
            assert_eq!(cbuf.as_mut_slice().len(), 24); // contiguous scratch of 6 i32
            write_i32s(cbuf.as_mut_slice(), 1..=6);
            cbuf.finalize(&shape, &dtype);
        }
        // Element (i, j) lands at i*4 + j in i32 units: row 0 -> [0,1,2], gap@3; row 1 -> [4,5,6], gap@7.
        assert_eq!(backing, [1, 2, 3, 0, 4, 5, 6, 0]);
    }

    /// A plain contiguous destination is handed out directly, as before.
    #[test]
    fn contiguous_outbuf_is_direct() {
        let dtype = i32_dtype();
        let ctx = ReadContext::default();
        let shape = [4usize];
        let mut backing = vec![0i32; 4];
        {
            let bytes = as_bytes_mut(&mut backing);
            let mut ob = OutBuf::new(bytes);
            let mut cbuf = ob.get_contiguous_mut(&shape, &dtype, &ctx).unwrap();
            assert!(matches!(cbuf, OutBufContiguousBuf::Direct(_)));
            write_i32s(cbuf.as_mut_slice(), 10..=13);
            cbuf.finalize(&shape, &dtype);
        }
        assert_eq!(backing, [10, 11, 12, 13]);
    }
}
