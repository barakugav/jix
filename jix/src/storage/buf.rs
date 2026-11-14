use std::sync::Arc;

use crate::buf_pool::{PoolBuf, PoolBufShared};
use crate::dtype::Dtype;
use crate::error::{bail, check_get_range, check_region_size, ensure, Result};
use crate::storage::Index;
use crate::util::{
    default_strides_slice, dim_arr, strided_span_bytes, strides_for_layout_order, DimArray,
    RawAlignedBytes, SliceExt,
};
use crate::{ArrayStorage, DimDyn, DimIdx, NdCopier, PtrMutNoalias, PtrNoalias, ReadContext};

/// A borrowed, strided view over a region of array bytes - the value produced and consumed by
/// [`ArrayStorage::read_data`](crate::ArrayStorage::read_data).
///
/// A `StridedBuf` pairs a block of bytes with one byte stride per dimension. Element
/// `(i0, i1, ..., in)` of the read region lives at byte offset
/// `i0*strides[0] + i1*strides[1] + ... + in*strides[n]` from the start of the buffer. The struct
/// carries no shape, dtype, or length of its own: those come from the read request (the index ranges
/// and the array's dtype), and it is the caller's job to interpret the bytes accordingly.
///
/// The data is NOT guaranteed to be aligned to the element dtype - neither the base pointer nor the
/// strides - so read and write elements through it with unaligned accesses. No method of this crate
/// requires a `StridedBuf` to be aligned.
///
/// # Examples
///
/// Build a strided view over a caller-owned buffer and read it back through the strides:
///
/// ```
/// use jix::storage::StridedBuf;
///
/// // Four contiguous i32 values: 4 elements, one dimension, a 4-byte stride.
/// let values = [10i32, 20, 30, 40];
/// let buf = unsafe { StridedBuf::from_raw_parts(values.as_ptr().cast::<u8>(), &[4], &[4], 4) };
/// assert_eq!(buf.strides(), &[4]);
///
/// // Element `i` lives at `data_ptr() + i * strides[0]`.
/// let base = buf.data_ptr();
/// let stride = buf.strides()[0];
/// for i in 0..4 {
///     let elem = unsafe { base.add(i * stride).cast::<i32>().read_unaligned() };
///     assert_eq!(elem, values[i]);
/// }
/// ```
pub struct StridedBuf<'a> {
    data: StridedBufData<'a>,
    /// Byte stride per dimension of the read region.
    strides: DimArray<usize>,
}

enum StridedBufData<'a> {
    Slice(&'a [u8]),
    SliceMut(&'a mut [u8]),
    PoolBuf {
        buf: PoolBuf<'a>,
        offset: usize,
    },
    PoolBufMut {
        buf: PoolBuf<'a>,
        offset: usize,
    },
    Shared {
        buf: PoolBufShared<'a>,
        offset: usize,
    },
}

impl<'a> StridedBuf<'a> {
    /// Build a read-only [`StridedBuf`] from a raw base pointer and byte `strides`.
    ///
    /// # Safety
    ///
    /// `data` must be valid for reads according to the `shape`, `strides` and `itemsize`, and it
    /// must remain valid for the lifetime `'a`. The memory must not be mutated during `'a`. `data`
    /// must be non-null even for an empty region, e.g. `NonNull::dangling()`.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: *const u8,
        shape: &[usize],
        strides: &[usize],
        itemsize: usize,
    ) -> Self {
        let span = strided_span_bytes(shape, strides, itemsize);
        let slice = unsafe { std::slice::from_raw_parts(data, span) };
        unsafe { Self::from_slice(slice, strides.to_dim_vec::<DimDyn>()) }
    }

    /// Build a writable [`StridedBuf`] from a raw base pointer and byte `strides`.
    ///
    /// # Safety
    ///
    /// `data` must be valid for reads and writes according to the `shape`, `strides` and
    /// `itemsize`, and it must remain valid for the lifetime `'a`. The memory must not be accessed
    /// through any other pointer during `'a`: when used as the `out` of a read, it must not overlap
    /// memory read by any array in the lazy chain (e.g. a [`Plain`](crate::storage::Plain) over the
    /// same buffer). `data` must be non-null even for an empty region, e.g. `NonNull::dangling()`.
    /// The caller should ensure there are no two different indices that result in the same byte
    /// offset (i.e. the strides must not alias, broadcasted dimensions are not allowed).
    #[inline]
    pub unsafe fn from_raw_parts_mut(
        data: *mut u8,
        shape: &[usize],
        strides: &[usize],
        itemsize: usize,
    ) -> Self {
        let span = strided_span_bytes(shape, strides, itemsize);
        let slice = unsafe { std::slice::from_raw_parts_mut(data, span) };
        unsafe { Self::from_slice_mut(slice, strides.to_dim_vec::<DimDyn>()) }
    }

    /// Create a read-only view into `data` with the given (byte) `strides`.
    ///
    /// # Safety
    ///
    /// The caller must ensure every strided access implied by `strides` (and the read shape it will
    /// be used with) stays in bounds of `data`.
    #[inline]
    pub(crate) unsafe fn from_slice(data: &'a [u8], strides: DimArray<usize>) -> Self {
        Self {
            data: StridedBufData::Slice(data),
            strides,
        }
    }

    /// Creates a writable view into `data` with the given (byte) `strides`.
    ///
    /// # Safety
    ///
    /// The caller must ensure every strided access implied by `strides` (and the read shape it will
    /// be used with) stays in bounds of `data`.
    /// The caller should ensure there are no two different indices that result in the same byte
    /// offset (i.e. the strides must not alias, broadcasted dimensions are not allowed).
    #[inline]
    pub(crate) unsafe fn from_slice_mut(data: &'a mut [u8], strides: DimArray<usize>) -> Self {
        Self {
            data: StridedBufData::SliceMut(data),
            strides,
        }
    }

    /// Create a writable view into a pooled buffer with the given (byte) `strides`.
    ///
    /// # Safety
    ///
    /// Same contract as [`from_slice_mut`](Self::from_slice_mut).
    #[inline]
    pub(crate) unsafe fn from_pool(buf: PoolBuf<'a>, strides: DimArray<usize>) -> Self {
        Self {
            data: StridedBufData::PoolBufMut { buf, offset: 0 },
            strides,
        }
    }

    /// Create a read-only view into a shared pooled buffer with the given (byte) `strides`.
    ///
    /// # Safety
    ///
    /// Same contract as [`from_slice`](Self::from_slice).
    #[inline]
    pub(crate) unsafe fn from_shared(buf: PoolBufShared<'a>, strides: DimArray<usize>) -> Self {
        Self {
            data: StridedBufData::Shared { buf, offset: 0 },
            strides,
        }
    }

    #[inline]
    pub(crate) fn is_writable(&self) -> bool {
        matches!(
            self.data,
            StridedBufData::SliceMut(_) | StridedBufData::PoolBufMut { .. }
        )
    }

    #[inline]
    pub(crate) fn make_readonly(&mut self) {
        self.data = match std::mem::replace(&mut self.data, StridedBufData::Slice(&[])) {
            StridedBufData::SliceMut(s) => StridedBufData::Slice(s),
            StridedBufData::PoolBufMut { buf, offset } => StridedBufData::PoolBuf { buf, offset },
            data => data,
        };
    }

    /// The byte stride of each dimension.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.strides.as_ref()
    }

    /// A raw pointer to the first byte of the buffer.
    ///
    /// Access it using the strides.
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data().0.as_ptr()
    }

    /// A mutable raw pointer to the first byte of the buffer.
    ///
    /// `None` if this is a read-only view. Access it using the strides.
    #[inline]
    pub fn data_ptr_mut(&mut self) -> Option<*mut u8> {
        self.is_writable().then(|| self.data_mut().0.as_mut_ptr())
    }

    #[inline]
    pub(crate) fn data(&self) -> (&[u8], &DimArray<usize>) {
        let data = match &self.data {
            StridedBufData::Slice(s) => *s,
            StridedBufData::SliceMut(s) => &**s,
            StridedBufData::PoolBuf { buf, offset }
            | StridedBufData::PoolBufMut { buf, offset } => &buf.as_slice()[*offset..],
            StridedBufData::Shared { buf, offset } => &buf.as_slice()[*offset..],
        };
        (data, &self.strides)
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> (&mut [u8], &DimArray<usize>) {
        let strides = &self.strides;
        let data = match &mut self.data {
            StridedBufData::SliceMut(s) => &mut **s,
            StridedBufData::PoolBufMut { buf, offset } => &mut buf.as_mut_slice()[*offset..],
            StridedBufData::Slice(_)
            | StridedBufData::PoolBuf { .. }
            | StridedBufData::Shared { .. } => {
                panic!("data_mut on a read-only StridedBuf view")
            }
        };
        (data, strides)
    }

    /// Whether the strides are exactly row-major packed for `shape` and `dtype`'s itemsize.
    #[inline]
    pub(crate) fn is_contiguous(&self, shape: &[usize], dtype: &Dtype) -> bool {
        let (_data, strides) = self.data();
        strides == default_strides_slice(shape, dtype.itemsize() as usize).as_ref()
    }

    /// Whether the strides are exactly row-major packed for `shape` and `dtype`'s itemsize, and
    /// the base pointer is aligned to `dtype`'s alignment (and the strides).
    #[inline]
    pub(crate) fn is_contiguous_aligned(&self, shape: &[usize], dtype: &Dtype) -> bool {
        let (data, _strides) = self.data();
        self.is_contiguous(shape, dtype)
            && (data.as_ptr() as usize).is_multiple_of(dtype.alignment().as_usize())
    }

    /// Re-label with `strides` in place, keeping the same backing bytes.
    ///
    /// # Safety
    ///
    /// Same contract as [`from_slice_mut`](Self::from_slice_mut) for a writable buffer. Call
    /// [`make_readonly`](Self::make_readonly) first to relabel with aliasing strides.
    #[inline]
    pub(crate) unsafe fn set_strides(&mut self, strides: DimArray<usize>) {
        self.strides = strides;
    }

    /// Copy `src` - laid out at `src_strides` over `shape` - into this destination, honoring this
    /// buffer's own strides.
    ///
    /// # Safety
    ///
    /// `src_strides` and this buffer's strides must both describe in-bounds `shape`-sized regions of
    /// `src` and the destination bytes respectively, and both must have length `shape.len()`.
    #[inline]
    pub(crate) unsafe fn copy_from(
        &mut self,
        src: &[u8],
        src_strides: &[usize],
        shape: &[usize],
        dtype: &Dtype,
    ) {
        let (dst, dst_strides) = self.data_mut();
        let copier = NdCopier::new(dtype);
        unsafe {
            copier.copy(
                PtrNoalias::from_slice(src),
                PtrMutNoalias::from_slice(dst),
                shape,
                src_strides,
                dst_strides,
                dtype,
            )
        };
    }

    /// Create a new view into the same underlying bytes, offset by `n` bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the resulting view is still valid for the same shape and strides.
    #[inline]
    pub(crate) unsafe fn with_offset(self, n: usize) -> Self {
        let StridedBuf { data, strides } = self;
        let data = match data {
            StridedBufData::Slice(s) => StridedBufData::Slice(&s[n..]),
            StridedBufData::SliceMut(s) => StridedBufData::SliceMut(&mut s[n..]),
            StridedBufData::PoolBuf { buf, offset } => StridedBufData::PoolBuf {
                buf,
                offset: offset + n,
            },
            StridedBufData::PoolBufMut { buf, offset } => StridedBufData::PoolBufMut {
                buf,
                offset: offset + n,
            },
            StridedBufData::Shared { buf, offset } => StridedBufData::Shared {
                buf,
                offset: offset + n,
            },
        };
        StridedBuf { data, strides }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn view(&self) -> StridedBuf<'_> {
        let (data, _) = self.data();
        StridedBuf {
            data: StridedBufData::Slice(data),
            strides: self.strides.clone(),
        }
    }

    #[inline]
    pub(crate) fn view_mut(&mut self) -> StridedBuf<'_> {
        let strides = self.strides.clone();
        let (data, _) = self.data_mut();
        StridedBuf {
            data: StridedBufData::SliceMut(data),
            strides,
        }
    }

    pub(crate) fn into_shared(
        self,
    ) -> Result<(Arc<RawAlignedBytes>, usize, DimArray<usize>), Self> {
        Ok(match self.data {
            StridedBufData::PoolBuf { buf, offset }
            | StridedBufData::PoolBufMut { buf, offset } => {
                (Arc::new(buf.into_inner()), offset, self.strides)
            }
            StridedBufData::Shared { buf, offset } => (buf.into_arc(), offset, self.strides),
            _ => return Err(self),
        })
    }
}

#[inline(always)]
pub(crate) fn check_out_buf(
    out: &OutBuf<'_, '_>,
    array_shape: &[u64],
    index: &Index<'_>,
    itemsize: usize,
) -> Result<()> {
    check_get_range(array_shape, index)?;
    check_region_size(index.size(), itemsize)?;
    match &out.0 {
        OutBufInner::Push(out) => check_out_buf_impl(out, index, itemsize),
        OutBufInner::Pull(_) => Ok(()),
    }
}

#[inline]
pub(crate) fn check_out_buf_dims(out: &StridedBuf, ndim: usize) -> Result<()> {
    ensure!(
        out.strides().len() == ndim,
        InvalidArgument,
        "out buffer has {} strides but array has {} dimensions",
        out.strides().len(),
        ndim
    );
    ensure!(
        out.is_writable(),
        InvalidArgument,
        "out buffer must be a writable destination, not a read-only view"
    );
    Ok(())
}

#[inline(never)]
pub(crate) fn check_out_buf_impl(
    out: &StridedBuf,
    index: &Index<'_>,
    itemsize: usize,
) -> Result<()> {
    check_out_buf_dims(out, index.ndim())?;
    let size = index.size();
    let strides = out.strides();
    let d_len_stride = |d: usize| (size[d] as u64, strides[d] as u64);

    /// The span in bytes of a region with `(len, stride)` dims.
    ///
    /// If every dim steps past the span of the previous dims, meaning no two elements overlap,
    /// returns Some(span), otherwise None.
    fn disjoint_span(dims: impl Iterator<Item = (u64, u64)>, itemsize: usize) -> Option<u64> {
        dims.filter(|&(len, _)| len > 1)
            .try_fold(itemsize as u64, |span, (len, stride)| {
                (stride >= span).then(|| stride.saturating_mul(len - 1).saturating_add(span))
            })
    }

    let mut span = None;
    if span.is_none() {
        // common fast path, strides decrease with the dim
        span = disjoint_span((0..size.len()).rev().map(d_len_stride), itemsize);
    }
    if span.is_none() {
        // general path, sort dims by stride
        let mut dims = dim_arr(size.len(), d_len_stride);
        dims.sort_unstable_by_key(|&(_len, stride)| stride);
        span = disjoint_span(dims.into_iter(), itemsize);
    }

    let Some(span) = span else {
        bail!(
            InvalidArgument,
            "out buffer strides {:?} overlap over the region {index:?}",
            out.strides()
        );
    };
    let len = out.data().0.len();
    ensure!(
        span <= len as u64,
        InvalidArgument,
        "out buffer of {len} bytes is too small for the region {index:?}"
    );
    Ok(())
}

/// A buffer from the pool of `context` for a region of `shape`, laid out in `layout_order`.
#[inline]
pub(crate) fn alloc_out_buf<'a>(
    context: &'a ReadContext,
    shape: &[usize],
    dtype: &Dtype,
    layout_order: &[DimIdx],
) -> StridedBuf<'a> {
    let itemsize = dtype.itemsize() as usize;
    let buf = context.allocate_buf(
        shape.iter().product::<usize>() * itemsize,
        dtype.alignment(),
    );
    let strides = strides_for_layout_order(shape, itemsize, layout_order);
    unsafe { StridedBuf::from_pool(buf, strides) }
}

/// The destination of an [`ArrayStorage::read_data`](crate::ArrayStorage::read_data), in one of
/// two modes mirroring NumPy's `out=`:
/// - **push**: the caller's buffer, which the read writes the region into at its own strides;
/// - **pull**: an empty slot, in which the read puts the cheapest buffer it can - a borrowed view
///   of its own memory (possibly with broadcast 0-stride dims) or a freshly materialized one - at
///   strides of its choosing.
pub struct OutBuf<'s, 'a>(pub(crate) OutBufInner<'s, 'a>);

pub(crate) enum OutBufInner<'s, 'a> {
    Push(&'s mut StridedBuf<'a>),
    Pull(&'s mut Option<StridedBuf<'a>>),
}

impl<'s, 'a> OutBuf<'s, 'a> {
    #[inline(always)]
    pub(crate) fn push(out: &'s mut StridedBuf<'a>) -> Self {
        Self(OutBufInner::Push(out))
    }

    /// A pull destination: the read puts its buffer in `slot`, which must be empty.
    #[inline(always)]
    pub(crate) fn pull(slot: &'s mut Option<StridedBuf<'a>>) -> Self {
        assert!(slot.is_none());
        Self(OutBufInner::Pull(slot))
    }

    #[inline(always)]
    pub(crate) fn is_pull(&self) -> bool {
        matches!(self.0, OutBufInner::Pull(_))
    }

    /// The buffer to write the region of `shape` into: the caller's in a push read, a new one
    /// from [`alloc_out_buf`], put in the slot, in a pull read.
    #[inline]
    pub(crate) fn materialize(
        self,
        context: &'a ReadContext,
        shape: &[usize],
        dtype: &Dtype,
        layout_order: &[DimIdx],
    ) -> &'s mut StridedBuf<'a> {
        match self.0 {
            OutBufInner::Push(out) => out,
            OutBufInner::Pull(slot) => {
                fill_slot(slot, alloc_out_buf(context, shape, dtype, layout_order))
            }
        }
    }

    /// Ends the read with `view` of the region of `shape`: puts it in the slot in a pull read,
    /// copies it into the caller's buffer in a push read.
    #[inline]
    pub(crate) fn put(self, view: StridedBuf<'a>, shape: &[usize], dtype: &Dtype) {
        match self.0 {
            OutBufInner::Pull(slot) => _ = fill_slot(slot, view),
            OutBufInner::Push(out) => {
                let (src, src_strides) = view.data();
                // SAFETY: `view` holds the region at its strides, and the caller checked `out`.
                unsafe { out.copy_from(src, src_strides, shape, dtype) };
            }
        }
    }
}

/// Puts `value` in `slot`, which is empty, and returns it.
///
/// The slot is not checked here but in [`OutBuf::pull`], before `value` is built, which lets
/// `value` be built in place. A value already in the slot would leak, not drop.
#[inline(always)]
pub(crate) fn fill_slot<T>(slot: &mut Option<T>, value: T) -> &mut T {
    std::mem::forget(slot.take());
    slot.insert(value)
}

/// Reads of an [`ArrayStorage`] in each mode.
pub(crate) trait ArrayStorageExt: ArrayStorage {
    /// Reads the region `index` into a buffer of the storage's choosing.
    #[inline(always)]
    fn read_data_pull<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
    ) -> Result<StridedBuf<'a>> {
        let mut slot = None;
        self.read_data_pull_in(index, context, &mut slot)?;
        Ok(slot.unwrap()) // filled by `read_data_pull_in`
    }

    /// Like [`read_data_pull`](Self::read_data_pull), putting the buffer in `slot`, which must be
    /// empty.
    #[inline(always)]
    fn read_data_pull_in<'s, 'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        slot: &'s mut Option<StridedBuf<'a>>,
    ) -> Result<&'s mut StridedBuf<'a>> {
        self.read_data(index, context, OutBuf::pull(slot))?;
        Ok(slot
            .as_mut()
            .expect("a pull read puts its buffer in the slot"))
    }

    /// Reads the region `index` into `out`, at its strides.
    ///
    /// The bytes of `out` are borrowed for as long as `self` and `context`; a buffer borrowing
    /// longer-lived bytes is passed as `&mut out.view_mut()`.
    #[inline(always)]
    fn read_data_push<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        out: &mut StridedBuf<'a>,
    ) -> Result<()> {
        self.read_data(index, context, OutBuf::push(out))
    }
}
impl<S: ArrayStorage + ?Sized> ArrayStorageExt for S {}

/// Read a stride-remapping forwarder's inner region once, honoring the mode of `out`.
///
/// A forwarder like `PermuteDims`/`InsertDim`/`RemoveDim` relates its output dims to its inner
/// dims by a fixed per-dim correspondence. The inner read is the same either way; only the
/// direction of the stride remap differs:
/// - pull: read the inner region into the slot and relabel its strides into output order, in
///   place, with `inner2outer_strides_fn(inner_strides) -> output_strides`.
/// - push: re-stride the destination into inner order with
///   `outer2inner_strides_fn(output_strides) -> inner_strides` and forward it down, so the inner
///   storage writes in place.
///
/// # Safety
///
/// The caller should ensure the created strides are valid.
#[inline]
pub(crate) unsafe fn read_data_and_map_strides<'a>(
    inner: &'a impl ArrayStorage,
    inner_index: &Index<'_>,
    context: &'a ReadContext,
    out: OutBuf<'_, 'a>,
    inner2outer_strides_fn: impl FnOnce(&[usize]) -> DimArray<usize>,
    outer2inner_strides_fn: impl FnOnce(&[usize]) -> DimArray<usize>,
) -> Result<()> {
    match out.0 {
        OutBufInner::Pull(slot) => {
            let buf = inner.read_data_pull_in(inner_index, context, slot)?;
            let strides = inner2outer_strides_fn(buf.strides());
            unsafe { buf.set_strides(strides) };
            Ok(())
        }
        OutBufInner::Push(out) => {
            let (data, strides) = out.data_mut();
            let inner_strides = outer2inner_strides_fn(strides);
            let mut inner_out = unsafe { StridedBuf::from_slice_mut(data, inner_strides) };
            inner.read_data(inner_index, context, OutBuf::push(&mut inner_out))
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::array;

    use crate::Array;
    use crate::ReadContext;

    #[test]
    fn out_smaller_than_the_region_is_rejected() {
        let ctx = ReadContext::default();
        // A 1-byte writable pull result, reused as `out` of a 4-byte read.
        let small = Array::plain_ndarray(array![1u8, 2]).unwrap().flip(0);
        let mut out = small.to_buf_strided(&[0..1], &ctx, None).unwrap();
        let big = || Array::plain_ndarray(array![1u8, 2, 3, 4]).unwrap();
        let ctx2 = ReadContext::default();
        assert!(big()
            .to_buf_strided(&[0..4], &ctx2, Some(&mut out))
            .is_err());
        assert!(big()
            .flip(0)
            .to_buf_strided(&[0..4], &ctx2, Some(&mut out))
            .is_err());
        assert!(big().to_buf_strided(&[1..2], &ctx2, Some(&mut out)).is_ok());
    }

    #[test]
    fn out_overlapping_over_the_region_is_rejected() {
        let ctx = ReadContext::default();
        // A writable pull result with strides [0, 4], reused as `out` of a 3-row read.
        let a = Array::plain_ndarray(array![1i32, 2, 3, 4]).unwrap();
        let a = a.flip(0).insert_dim(0);
        let mut out = a.to_buf_strided(&[0..1, 0..4], &ctx, None).unwrap();
        let big = Array::plain_ndarray(array![[10i32, 11, 12, 13], [20, 21, 22, 23]]).unwrap();
        let ctx2 = ReadContext::default();
        assert!(big
            .to_buf_strided(&[0..2, 0..4], &ctx2, Some(&mut out))
            .is_err());
        assert!(big
            .to_buf_strided(&[1..2, 0..4], &ctx2, Some(&mut out))
            .is_ok());
    }

    #[test]
    fn aliasing_pull_result_is_not_accepted_as_out() {
        let ctx = ReadContext::default();
        let b = Array::plain_ndarray(array![[1.0f32], [2.0]])
            .unwrap()
            .flip(0)
            .broadcast(&[2, 4]);
        let n = -Array::plain_ndarray(array![[1.0f32]])
            .unwrap()
            .broadcast(&[2, 4]);
        let x = Array::plain_ndarray(array![[1.0f32, 2., 3., 4.], [5., 6., 7., 8.]]).unwrap();
        for mut out in [
            b.to_buf_strided(&[0..2, 0..4], &ctx, None).unwrap(),
            n.to_buf_strided(&[0..2, 0..4], &ctx, None).unwrap(),
        ] {
            assert!(out.data_ptr_mut().is_none());
            let ctx2 = ReadContext::default();
            assert!(x
                .view()
                .to_buf_strided(&[0..2, 0..4], &ctx2, Some(&mut out))
                .is_err());
        }
    }
}
