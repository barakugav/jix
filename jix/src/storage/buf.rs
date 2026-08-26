use std::ops::Range;

use crate::buf_pool::PoolBuf;
use crate::dtype::Dtype;
use crate::error::{ensure, Result};
use crate::util::{default_strides_slice, strided_span_bytes, DimArray, SliceExt};
use crate::{ArrayStorage, DimDyn, NdCopier, ReadContext};

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
    PoolBuf { buf: PoolBuf<'a>, offset: usize },
}

impl<'a> StridedBuf<'a> {
    /// Build a read-only [`StridedBuf`] from a raw base pointer and byte `strides`.
    ///
    /// # Safety
    ///
    /// `data` must be valid for reads according to the `shape`, `strides` and `itemsize`, and it
    /// must remain valid for the lifetime `'a`
    #[inline]
    pub unsafe fn from_raw_parts(
        data: *const u8,
        shape: &[usize],
        strides: &[usize],
        itemsize: usize,
    ) -> Self {
        let span = strided_span_bytes(shape, strides, itemsize);
        let slice = unsafe { std::slice::from_raw_parts(data, span) };
        unsafe { Self::from_slice(slice, strides) }
    }

    /// Build a writable [`StridedBuf`] from a raw base pointer and byte `strides`.
    ///
    /// # Safety
    ///
    /// `data` must be valid for reads and writes according to the `shape`, `strides` and
    /// `itemsize`, and it must remain valid for the lifetime `'a`.
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
        unsafe { Self::from_slice_mut(slice, strides) }
    }

    /// Create a read-only view into `data` with the given (byte) `strides`.
    ///
    /// # Safety
    ///
    /// The caller must ensure every strided access implied by `strides` (and the read shape it will
    /// be used with) stays in bounds of `data`.
    #[inline]
    pub(crate) unsafe fn from_slice(data: &'a [u8], strides: &[usize]) -> Self {
        Self {
            data: StridedBufData::Slice(data),
            strides: strides.to_dim_vec::<DimDyn>(),
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
    pub(crate) unsafe fn from_slice_mut(data: &'a mut [u8], strides: &[usize]) -> Self {
        Self {
            data: StridedBufData::SliceMut(data),
            strides: strides.to_dim_vec::<DimDyn>(),
        }
    }

    /// Create a writable view into a pooled buffer with the given (byte) `strides`.
    ///
    /// # Safety
    ///
    /// Same contract as [`from_slice_mut`](Self::from_slice_mut).
    #[inline]
    pub(crate) unsafe fn from_pool(buf: PoolBuf<'a>, strides: &[usize]) -> Self {
        Self {
            data: StridedBufData::PoolBuf { buf, offset: 0 },
            strides: strides.to_dim_vec::<DimDyn>(),
        }
    }

    #[inline]
    pub(crate) fn is_writable(&self) -> bool {
        !matches!(self.data, StridedBufData::Slice(_))
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
    pub(crate) fn data(&self) -> (&[u8], &[usize]) {
        let data = match &self.data {
            StridedBufData::Slice(s) => *s,
            StridedBufData::SliceMut(s) => &**s,
            StridedBufData::PoolBuf { buf, offset } => &buf.as_slice()[*offset..],
        };
        (data, self.strides.as_ref())
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> (&mut [u8], &[usize]) {
        let strides = self.strides.as_ref();
        let data = match &mut self.data {
            StridedBufData::SliceMut(s) => &mut **s,
            StridedBufData::PoolBuf { buf, offset } => &mut buf.as_mut_slice()[*offset..],
            StridedBufData::Slice(_) => panic!("data_mut on a read-only StridedBuf view"),
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

    /// Consume and re-label with `strides`, keeping the same backing bytes.
    ///
    /// # Safety
    ///
    /// Same contract as [`from_slice_mut`](Self::from_slice_mut).
    #[inline]
    pub(crate) unsafe fn with_strides(mut self, strides: &[usize]) -> Self {
        self.strides = strides.to_dim_vec::<DimDyn>();
        self
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
        unsafe { copier.copy(src, dst, shape, src_strides, dst_strides, dtype) };
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
}

pub(crate) fn check_out_buf(out: Option<&StridedBuf>, array_shape: &[u64]) -> Result<()> {
    let Some(out) = out else { return Ok(()) };
    ensure!(
        out.strides().len() == array_shape.len(),
        InvalidArgument,
        "out buffer has {} strides but array has {} dimensions",
        out.strides().len(),
        array_shape.len()
    );
    ensure!(
        out.is_writable(),
        InvalidArgument,
        "out buffer must be a writable destination, not a read-only view"
    );
    Ok(())
}

#[inline]
pub(crate) fn materialize_out_buf<'a>(
    out: Option<&'a mut StridedBuf<'_>>,
    context: &'a ReadContext,
    out_shape: &[usize],
    dtype: &Dtype,
) -> StridedBuf<'a> {
    match out {
        Some(out) => out.view_mut(),
        None => {
            let itemsize = dtype.itemsize() as usize;
            let buf = context.allocate_buf(
                out_shape.iter().product::<usize>() * itemsize,
                dtype.alignment(),
            );
            let strides = default_strides_slice(out_shape, itemsize);
            // SAFETY: C-order strides for a pooled buffer sized to `out_shape`.
            unsafe { StridedBuf::from_pool(buf, strides.as_ref()) }
        }
    }
}

/// Read a stride-remapping forwarder's inner region once, honoring the caller's `out=` mode.
///
/// A forwarder like `PermuteAxes`/`InsertAxis`/`RemoveAxis` relates its output axes to its inner
/// axes by a fixed per-axis correspondence. The inner read is the same either way; only the
/// direction of the stride remap differs:
/// - pull (`out=None`): read the inner as a view and relabel its strides into output order with
///   `inner2outer_strides_fn(inner_strides) -> output_strides`.
/// - push (`out=Some`): re-stride the destination into inner order with
///   `outer2inner_strides_fn(output_strides) -> inner_strides` and forward it down, so the inner
///   storage writes in place.
///
/// # Safety
///
/// The caller should ensure the created strides are valid.
#[inline]
pub(crate) unsafe fn read_data_and_map_strides<'a>(
    inner: &'a impl ArrayStorage,
    inner_index: &[Range<u64>],
    context: &'a ReadContext,
    mut out: Option<&'a mut StridedBuf<'_>>,
    inner2outer_strides_fn: impl FnOnce(&[usize]) -> DimArray<usize>,
    outer2inner_strides_fn: impl FnOnce(&[usize]) -> DimArray<usize>,
) -> Result<StridedBuf<'a>> {
    // This function does an unsafe dance of extending lifetimes, with the goal to have a single
    // call to inner.read_data() for better inlining of ops pipelines.

    let is_push = out.is_some();
    let mut inner_out = out.as_deref_mut().map(|o| {
        let (data, strides) = o.data_mut();
        let inner_strides = outer2inner_strides_fn(strides);
        unsafe { StridedBuf::from_slice_mut(data, inner_strides.as_ref()) }
    });

    let inner_buf = inner.read_data(inner_index, context, inner_out.as_mut())?;

    if is_push {
        // The inner storage wrote into `out` through `inner_out`
        drop(inner_buf);
        drop(inner_out);
        Ok(out.unwrap().view_mut())
    } else {
        // SAFETY: `inner_buf` does not borrow the local `inner_out` (which was `None`) - its data
        // lives for 'a - so extending its lifetime to 'a is sound.
        let inner_buf = unsafe { std::mem::transmute::<StridedBuf<'_>, StridedBuf<'a>>(inner_buf) };
        let out_strides = inner2outer_strides_fn(inner_buf.strides());
        Ok(unsafe { inner_buf.with_strides(out_strides.as_ref()) })
    }
}
