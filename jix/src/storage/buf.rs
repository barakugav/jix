use std::ops::Range;

use crate::codec::TmpBuf;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::util::{default_strides_slice, dim_arr, DimArray, NdCopier};
use crate::ReadContext;

/// Destination for a [`read_data`](crate::ArrayStorage::read_data) call.
///
/// One of:
/// - a caller-provided contiguous byte buffer ([`OutBuf::new`]),
/// - a lazily-allocated pooled buffer ([`OutBuf::new_lazy`]),
/// - a caller-provided *strided* byte buffer ([`OutBuf::new_strided`]) - a rectangular sub-region of
///   some larger destination, described by per-dimension byte strides.
///
/// Storage backends never write to an `OutBuf` directly; they request a contiguous writable slice
/// via [`get_continuous_mut`](OutBuf::get_continuous_mut). For the contiguous and lazy variants that
/// slice IS the destination; for the strided variant a temporary contiguous scratch buffer is handed
/// out and scattered into the real destination once the write succeeds. This lets every backend keep
/// emitting plain row-major output while still supporting a strided destination, folding what would
/// otherwise be a separate copy at the call site into the single unavoidable one.
///
/// After a successful `read_data` call the contents of a contiguous/lazy `OutBuf` can be accessed via
/// [`as_slice`](OutBuf::as_slice).
pub struct OutBuf<'a>(OutBufInner<'a>);
enum OutBufInner<'a> {
    Lazy(&'a ReadContext),
    Borrowed(&'a mut [u8]),
    Tmp(TmpBuf<'a>),
    /// A rectangular strided sub-region of a larger destination buffer, with per-dimension byte
    /// strides. Writes are staged through a contiguous scratch buffer and scattered on success; see
    /// [`OutBuf::get_continuous_mut`].
    // Dormant until a strided producer (e.g. `to_ndarray_buf`) starts calling `new_strided`.
    #[allow(dead_code)]
    BorrowedStrided {
        dst: &'a mut [u8],
        strides: DimArray<usize>,
    },
}
impl<'a> OutBuf<'a> {
    /// Write into a caller-provided contiguous buffer.
    #[inline(always)]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self(OutBufInner::Borrowed(buf))
    }

    /// Defer allocation: the storage allocates a pooled buffer from `context` on the first demand for
    /// a mutable slice.
    #[inline(always)]
    pub fn new_lazy(context: &'a ReadContext) -> Self {
        Self(OutBufInner::Lazy(context))
    }

    /// Write into a caller-provided *strided* buffer.
    ///
    /// `dst` is the destination byte buffer and `strides` gives the per-dimension byte stride of the
    /// region being written (row-major over the read `index`, but embedded in a larger buffer). The
    /// storage still emits contiguous row-major output; the strided scatter into `dst` happens inside
    /// the guard returned by [`get_continuous_mut`](OutBuf::get_continuous_mut).
    ///
    /// # Safety
    ///
    /// The length of `dst` is **not** checked against `strides` (nor against the read `index` it will
    /// later be used with). The caller must ensure `dst` is valid for every strided write, i.e. it
    /// spans at least `sum((extent[d] - 1) * strides[d]) + itemsize` bytes from its start for the
    /// region that will be read into it.
    // Dormant until a strided producer (e.g. `to_ndarray_buf`) starts calling it.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) unsafe fn new_strided(dst: &'a mut [u8], strides: DimArray<usize>) -> Self {
        Self(OutBufInner::BorrowedStrided { dst, strides })
    }

    /// Obtain a contiguous, writable buffer for a `read_data` of the given `index`/`dtype`.
    ///
    /// The returned [`OutBufContinuousBuf`] guard hands out a contiguous `&mut [u8]` of exactly
    /// `nitems * dtype.itemsize()` bytes via [`edit`](OutBufContinuousBuf::edit). For the contiguous
    /// and lazy variants that slice is the real destination. For the strided variant the guard
    /// allocates a temporary contiguous scratch buffer (from `context`) inside `edit` and scatters it
    /// into the real destination - but only if the `edit` closure returns `Ok`, so a failed read
    /// never performs the strided copy.
    ///
    /// `context` is used only to allocate scratch for the strided variant; the lazy variant keeps
    /// using its own stored context.
    #[inline]
    pub(crate) fn get_continuous_mut<'b>(
        &'b mut self,
        index: &[Range<u64>],
        dtype: &'b Dtype,
        context: &'b ReadContext,
    ) -> OutBufContinuousBuf<'b> {
        // Lazy: materialize into an owned `Tmp` in place, so a later `unwrap_tmp`/`as_slice` sees it.
        // (Unchanged from the old `get_mut`; keeps using the lazy variant's own context.)
        let lazy_ctx = match &self.0 {
            OutBufInner::Lazy(ctx) => Some(*ctx),
            _ => None,
        };
        if let Some(lazy_ctx) = lazy_ctx {
            let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
            let tmp = lazy_ctx.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment());
            self.0 = OutBufInner::Tmp(tmp);
        }

        // Strided: take the destination out (leaving a harmless empty placeholder). The scratch
        // buffer is allocated and scattered into `dst` inside `edit`.
        if matches!(self.0, OutBufInner::BorrowedStrided { .. }) {
            let taken = std::mem::replace(&mut self.0, OutBufInner::Borrowed(&mut []));
            let OutBufInner::BorrowedStrided { dst, strides } = taken else {
                unreachable!()
            };
            let shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
            return OutBufContinuousBuf::Strided {
                dst,
                dst_strides: strides,
                shape,
                dtype,
                context,
            };
        }

        // Contiguous (caller-provided, or the lazy buffer just materialized above): hand out the
        // slice directly.
        match &mut self.0 {
            OutBufInner::Borrowed(buf) => OutBufContinuousBuf::Direct(buf),
            OutBufInner::Tmp(tmp) => OutBufContinuousBuf::Direct(tmp.as_mut_slice()),
            OutBufInner::Lazy(_) | OutBufInner::BorrowedStrided { .. } => unreachable!(),
        }
    }

    /// Returns the buffer contents, or `None` for a not-yet-materialized lazy `OutBuf` or a strided
    /// `OutBuf` (neither has a single contiguous view).
    #[inline(always)]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.0 {
            OutBufInner::Lazy(_) | OutBufInner::BorrowedStrided { .. } => None,
            OutBufInner::Borrowed(buf) => Some(buf),
            OutBufInner::Tmp(tmp) => Some(tmp.as_slice()),
        }
    }

    /// Returns the mutable buffer contents, or `None` for a not-yet-materialized lazy `OutBuf` or a
    /// strided `OutBuf` (neither has a single contiguous view).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match &mut self.0 {
            OutBufInner::Lazy(_) | OutBufInner::BorrowedStrided { .. } => None,
            OutBufInner::Borrowed(buf) => Some(buf),
            OutBufInner::Tmp(tmp) => Some(tmp.as_mut_slice()),
        }
    }

    #[track_caller]
    #[inline(always)]
    pub(crate) fn unwrap_tmp(self) -> TmpBuf<'a> {
        match self.0 {
            OutBufInner::Tmp(tmp) => tmp,
            _ => unreachable!(),
        }
    }
}

/// A contiguous, writable view obtained from [`OutBuf::get_continuous_mut`].
///
/// Call [`edit`](Self::edit) with a closure that fills the contiguous slice. For a strided
/// destination the slice is a temporary scratch buffer that is scattered into the real destination
/// when - and only when - the closure returns `Ok`; a closure that returns `Err`, or a
/// non-strided `OutBuf`, performs no scatter.
pub(crate) enum OutBufContinuousBuf<'b> {
    /// The contiguous slice is the real destination; nothing to do after the write.
    Direct(&'b mut [u8]),
    /// The write is staged in a scratch buffer (allocated from `context` in `edit`) and, on success,
    /// scattered into `dst` using `dst_strides`.
    Strided {
        dst: &'b mut [u8],
        dst_strides: DimArray<usize>,
        shape: DimArray<usize>,
        dtype: &'b Dtype,
        context: &'b ReadContext,
    },
}
impl OutBufContinuousBuf<'_> {
    /// Fill the contiguous buffer via `f`. For a strided destination the staged scratch buffer is
    /// scattered into the real destination iff `f` returns `Ok`, so a failed read leaves the
    /// destination untouched.
    #[inline]
    pub(crate) fn edit(&mut self, f: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<()> {
        match self {
            OutBufContinuousBuf::Direct(buf) => f(buf),
            OutBufContinuousBuf::Strided {
                context,
                dst,
                dst_strides,
                shape,
                dtype,
            } => {
                let itemsize = dtype.itemsize() as usize;
                let nitems = shape.as_ref().iter().product::<usize>();
                let mut tmp = context.tmp_buf(nitems * itemsize, dtype.alignment());
                f(tmp.as_mut_slice())?;
                // Scatter the contiguous scratch buffer into the strided destination. Only reached on
                // success, so a failed read never touches `dst`.
                let src_strides = default_strides_slice(shape.as_ref(), itemsize);
                let copier = NdCopier::new(dtype);
                // SAFETY: `tmp` is a fresh, disjoint allocation holding `shape` in row-major order;
                // `dst`/`dst_strides` describe the same-shaped region inside the caller's buffer,
                // which `OutBuf::new_strided`'s (unsafe) contract requires to be valid for these
                // writes.
                unsafe {
                    copier.copy(
                        tmp.as_slice().as_ptr(),
                        dst.as_mut_ptr(),
                        shape.as_ref(),
                        src_strides.as_ref(),
                        dst_strides.as_ref(),
                        dtype,
                    );
                }
                Ok(())
            }
        }
    }
}
