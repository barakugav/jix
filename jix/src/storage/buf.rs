use std::ops::Range;

use crate::codec::TmpBuf;
use crate::dtype::Dtype;
use crate::ReadContext;

/// Destination for a [`read_data`](ArrayStorage::read_data) call.
///
/// Either a caller-provided byte buffer ([`OutBuf::new`]) or a lazily-allocated pooled buffer
/// ([`OutBuf::new_lazy`]).
/// After a successful `read_data` call, the buffer contents can be accessed via
/// [`as_slice`](OutBuf::as_slice).
pub struct OutBuf<'a>(OutBufInner<'a>);
enum OutBufInner<'a> {
    Lazy(&'a ReadContext),
    Borrowed(&'a mut [u8]),
    Tmp(TmpBuf<'a>),
}
impl<'a> OutBuf<'a> {
    /// Write into a caller-provided buffer.
    #[inline(always)]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self(OutBufInner::Borrowed(buf))
    }

    /// Defer allocation: the storage allocates a pooled buffer from `context` on the first demand
    /// for a mutable slice.
    #[inline(always)]
    pub fn new_lazy(context: &'a ReadContext) -> Self {
        Self(OutBufInner::Lazy(context))
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, index: &[Range<u64>], dtype: &Dtype) -> &mut [u8] {
        if let OutBufInner::Lazy(context) = &self.0 {
            let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
            let tmp = context.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment());
            self.0 = OutBufInner::Tmp(tmp);
        }
        self.as_mut_slice().unwrap()
    }

    /// Returns the buffer contents, or `None` if this is a not-yet-materialized lazy `OutBuf`.
    #[inline(always)]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.0 {
            OutBufInner::Lazy(_) => None,
            OutBufInner::Borrowed(buf) => Some(buf),
            OutBufInner::Tmp(tmp) => Some(tmp.as_slice()),
        }
    }

    /// Returns the mutable buffer contents, or `None` if this is a not-yet-materialized lazy `OutBuf`.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match &mut self.0 {
            OutBufInner::Lazy(_) => None,
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
