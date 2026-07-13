use std::ops::Range;

use crate::codec::TmpBuf;
use crate::dtype::Dtype;
use crate::error::{ensure, Result};
use crate::util::{default_strides_cast, default_strides_slice, NdCopier};
use crate::{default_strides_from_iter, Dimension, ReadContext, SliceExt};

/// Destination for a [`read_data`](crate::ArrayStorage::read_data) call.
///
/// One of:
/// - a caller-provided contiguous byte buffer ([`OutBuf::new`]),
/// - a lazily-allocated pooled buffer ([`OutBuf::new_lazy`]),
/// - a caller-provided *strided* byte buffer ([`OutBuf::new_strided`]) - a rectangular sub-region of some
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

    /// Obtain a contiguous, writable buffer for a `read_data` of the given `index`/`dtype`.
    ///
    /// The returned [`OutBufContiguousBuf`] exposes a contiguous `&mut [u8]` of exactly
    /// `nitems * dtype.itemsize()` bytes via [`as_mut_slice`](OutBufContiguousBuf::as_mut_slice).
    /// For the contiguous and lazy variants that slice IS the real destination and
    /// [`finalize`](OutBufContiguousBuf::finalize) is a no-op. For the strided
    /// variant the slice is a temporary contiguous scratch buffer, allocated here from `context`; the
    /// caller fills it and then calls `finalize` to scatter it into the real strided
    /// destination. Skipping that final call (e.g. by returning early on error) leaves the
    /// destination untouched.
    ///
    /// `context` is used only to allocate scratch for the strided variant; the lazy variant keeps
    /// using its own stored context.
    #[inline]
    pub(crate) fn get_contiguous_mut<'b>(
        &'b mut self,
        nitems: usize,
        dtype: &Dtype,
        context: &'b ReadContext,
    ) -> Result<OutBufContiguousBuf<'b>> {
        // ContiguousLazy: materialize into an owned `ContiguousLazyAllocated` in place, so a later `unwrap_tmp`/`as_slice` sees it.
        let lazy_ctx = match &self.0 {
            OutBufInner::ContiguousLazy(ctx) => Some(*ctx),
            _ => None,
        };
        if let Some(lazy_ctx) = lazy_ctx {
            let tmp = lazy_ctx.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment());
            self.0 = OutBufInner::ContiguousLazyAllocated(tmp);
        }

        // Strided: take the destination out (leaving a harmless empty placeholder) and allocate a
        // contiguous scratch buffer to fill; `finalize` scatters it into `strided_buf`.
        if matches!(self.0, OutBufInner::Strided { .. }) {
            let taken = std::mem::replace(&mut self.0, OutBufInner::Contiguous(&mut []));
            let OutBufInner::Strided { buf, strides } = taken else {
                unreachable!()
            };
            let contiguous_buf =
                context.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment());
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
        ensure!(
            buf.as_mut_slice().len() == nitems * dtype.itemsize() as usize,
            InvalidBufferSize,
            "Unexpected buffer size {} requested for {nitems} items with dtype {dtype}",
            buf.as_mut_slice().len()
        );
        ensure!(
            (buf.as_mut_slice().as_ptr() as usize).is_multiple_of(dtype.alignment().as_usize()),
            InvalidArgument,
            "Buffer pointer {:p} is not aligned to dtype {dtype} alignment {}",
            buf.as_mut_slice().as_ptr(),
            dtype.alignment().as_usize()
        );
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
