use std::mem::MaybeUninit;
use std::ops::Range;

use crate::error::{check_get_buffer_size, check_ranges, check_region_size, ensure, Result};
use crate::storage::{
    alloc_out_buf, check_out_buf_dims, check_out_buf_impl, split_ranges, ArrayStorageExt,
    ArrayStorageTyped, Index, OutBuf, StridedBuf,
};
use crate::util::iter::NdIter;
use crate::util::{calc_block_end, cast_slice_mut, default_strides_slice, dim_arr, IterExt};
use crate::ReadContext;
use crate::{ArrayStorage, Dim, DimDyn, Dimension, PlainArray};

use super::Array;

impl<S: ArrayStorage> Array<S> {
    /// Read the entire array into a fresh heap-allocated [`ndarray::Array`].
    ///
    /// This is the simplest way to materialize a jix array into a standard in-memory ndarray.
    /// All blocks of the underlying compact storage are decompressed and the elements are
    /// returned in an ndarray with the same shape and element type as the source. The memory
    /// layout of the returned array is an implementation detail and may vary.
    ///
    /// For a lazy view (e.g. `Array<Add<Mul<Compact, _>, _>>`), `to_ndarray` walks the entire
    /// operation pipeline: each composed op is evaluated on the fly as data flows out of the
    /// innermost storage. Materializing the same chain repeatedly will redo the work; if you
    /// plan to read the result more than once, call [`compact`](Array::compact) first to re-compress
    /// the result into a fresh `Array<Compact>` (or [`compact_with`](Array::compact_with) to control
    /// the block shape).
    ///
    /// `to_ndarray` allocates a fresh [`ReadContext`] internally, so callers don't need to
    /// manage one. For repeated reads (e.g. iterating tiles over a large array) prefer
    /// [`to_ndarray_sub`](Array::to_ndarray_sub) with an explicit context obtained from
    /// [`read_ctx`](Array::read_ctx) - that way the codec scratch buffers and decompressor
    /// instance are shared across calls.
    ///
    /// The returned array is not necessarily C-contiguous. The implementation is free to
    /// choose any memory layout it prefers, mostly for performance reasons.
    /// If you need a specific memory layout, use [`to_buf_strided`](Array::to_buf_strided) and
    /// pass a pre-allocated buffer with your specified strides.
    ///
    /// # Errors
    ///
    /// - [`InvalidShapeOperation`](crate::error::ErrorKind::InvalidShapeOperation) - the result
    ///   does not fit in memory.
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
    /// let nd = a.to_ndarray()?;
    /// assert_eq!(nd, array![[1.0f32, 2.0], [3.0, 4.0]]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// Materialize a lazy pipeline. None of the arithmetic runs at construction time - it
    /// only executes when `to_ndarray` walks the chain:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix2};
    ///
    /// let a = Array::compact_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    /// let b = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
    ///
    /// // Build a lazy view: (a + b) * 2 - 1
    /// let lazy = (a + b).map(|x| x * 2.0f32 - 1.0f32);
    /// assert_eq!(lazy.shape(), &[2, 3]);
    ///
    /// let nd = lazy.to_ndarray::<Ix2>()?; // executes the pipeline
    /// assert_eq!(nd[[0, 0]], (1.0 + 1.0) * 2.0 - 1.0);
    /// assert_eq!(nd[[1, 2]], (6.0 + 1.0) * 2.0 - 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline]
    pub fn to_ndarray<D>(&self) -> Result<ndarray::Array<S::Item, D>>
    where
        S: ArrayStorageTyped,
        D: ndarray::Dimension,
    {
        let shape = self.shape();
        let full_range = dim_arr(shape.len(), |dim| 0..shape[dim]);
        self.to_ndarray_sub(full_range.as_ref(), &self.read_ctx())
    }

    /// Read a rectangular sub-region of the array into a fresh heap-allocated [`ndarray::Array`].
    ///
    /// `index` contains one half-open `start..end` range per dimension within
    /// `0..self.shape()[dim]`. The result has shape `[index[0].len(), index[1].len(), ...]`
    /// and contains the corresponding elements from the source.
    ///
    /// The returned array is not necessarily C-contiguous. The implementation is free to
    /// choose any memory layout it prefers, mostly for performance reasons.
    /// If you need a specific memory layout, use [`to_buf_strided`](Array::to_buf_strided) and
    /// pass a pre-allocated buffer with your specified strides.
    ///
    /// # When to use this over [`to_ndarray`](Array::to_ndarray)
    ///
    /// Use `to_ndarray_sub` when you only need a window of a large array, or when you want
    /// to stream over an array tile-by-tile without ever materializing the whole thing in
    /// memory. Use `to_ndarray` when you need the entire array at once and don't plan to
    /// issue further reads.
    ///
    /// # What gets decompressed
    ///
    /// For [`Array<Compact>`](crate::storage::Compact) (and views layered on top of it),
    /// `to_ndarray_sub` only touches the blocks that overlap `index`:
    ///
    /// - **Block-aligned ranges**: if `index` aligns to block boundaries on every dim,
    ///   exactly the covered blocks are decompressed - no wasted work.
    /// - **Unaligned ranges**: the overlapping boundary blocks are decompressed in full and
    ///   the requested slice is copied out. Elements outside `index` within those boundary
    ///   blocks are decoded but then discarded.
    ///
    /// To keep sub-region reads cheap, choose a `block_shape` (via
    /// [`ArrayParams::block_shape`](crate::ArrayParams::block_shape) when creating the array)
    /// that matches your access pattern.
    ///
    /// For lazy views (e.g. after `reshape`, `permute_dims`, `broadcast`, or element-wise
    /// ops), the index range is propagated inward through the chain and only the
    /// corresponding region of the innermost storage is read. A shape-changing op can
    /// scramble the mapping such that a small output range still touches many input blocks -
    /// if this matters for performance, materialize with [`compact`](Array::compact) before
    /// iterating sub-regions.
    ///
    /// # `ReadContext` reuse
    ///
    /// `context` carries the decoder instance and scratch buffers used during decompression.
    /// Allocating these on every read is expensive, especially for small windows. Obtain a
    /// single context via [`read_ctx`](Array::read_ctx) once and pass it to every call.
    /// [`read_ctx`](Array::read_ctx) inherits the array's stored decoder configuration;
    /// `ReadContext::default()` works too but uses default decoder parameters regardless of
    /// the array.
    ///
    /// # Errors
    ///
    /// - [`InvalidIndex`](crate::error::ErrorKind::InvalidIndex) - `index` is out of bounds or
    ///   has a different number of dimensions than the array.
    /// - [`InvalidShapeOperation`](crate::error::ErrorKind::InvalidShapeOperation) - the result
    ///   does not fit in memory.
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// Stream a large array tile-by-tile, reusing a single `ReadContext`. The block shape
    /// is chosen to divide the tile shape, so every tile read decompresses exactly its
    /// covering blocks with no boundary waste:
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    /// use ndarray::Ix2;
    ///
    /// // 256x256 f32 array stored as 32x32 blocks.
    /// let data = ndarray::Array2::<f32>::from_shape_fn((256, 256), |(i, j)| (i + j) as f32);
    /// let a = Array::compact_ndarray_with(&data, ArrayParams::new().block_shape(&[32, 32])?)?;
    ///
    /// // Walk the array as 64x64 tiles. 32 divides 64, so each call decompresses exactly
    /// // 2*2 = 4 blocks - no boundary waste. The same ReadContext is reused across all
    /// // 16 reads, sharing decompressor and scratch buffers.
    /// let ctx = a.read_ctx();
    /// let mut total = 0.0f32;
    /// for tr in 0..4 {
    ///     for tc in 0..4 {
    ///         let tile = a.to_ndarray_sub::<Ix2>(
    ///             &[(tr * 64)..((tr + 1) * 64), (tc * 64)..((tc + 1) * 64)],
    ///             &ctx,
    ///         )?;
    ///         total += tile.sum();
    ///     }
    /// }
    /// assert!(total > 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// Sub-region reads also work on lazy views - the requested range is propagated through
    /// the pipeline so the inner storage only sees the corresponding window:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    /// let scaled = a.map(|x| x * 10i32);
    ///
    /// let ctx = scaled.read_ctx();
    /// assert_eq!(
    ///     scaled.to_ndarray_sub(&[1..3, 1..3], &ctx)?,
    ///     array![[50, 60], [80, 90]],
    /// );
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline]
    pub fn to_ndarray_sub<D>(
        &self,
        index: &[Range<u64>],
        context: &ReadContext,
    ) -> Result<ndarray::Array<S::Item, D>>
    where
        S: ArrayStorageTyped,
        D: ndarray::Dimension,
    {
        check_ranges(self.shape(), index)?;
        let (start, size) = split_ranges(index);
        let ndim = self.ndim();
        if let Some(expected) = D::NDIM {
            ensure!(
                ndim == expected,
                InvalidShapeOperation,
                "Cannot read an array of ndim {ndim} into an ndarray of ndim {expected}"
            );
        }
        let itemsize = self.dtype().itemsize() as usize;
        let alloc_bytes = size
            .iter()
            .map(|&n| n.max(1) as u64)
            .chain([itemsize as u64])
            .try_shape_product();
        ensure!(
            alloc_bytes.is_some_and(|b| b <= isize::MAX as u64),
            InvalidShapeOperation,
            "read region {index:?} with itemsize {itemsize} does not fit in memory"
        );
        let mut out_shape = D::zeros(ndim);
        for dim in 0..ndim {
            out_shape[dim] = size[dim];
        }
        let is_empty = size.contains(&0);

        let mut array = {
            let spec = self.storage.spec();
            let layout_order = spec.read_layout_order();
            if layout_order
                .iter()
                .enumerate()
                .all(|(i, &dim)| i == dim as usize)
            {
                // C-order hint
                ndarray::Array::uninit(out_shape)
            } else {
                // Custom layout hint
                let mut alloc_shape = D::zeros(ndim);
                let mut inv_order = D::zeros(ndim);
                for dim in 0..ndim {
                    alloc_shape[dim] = out_shape[layout_order[dim] as usize];
                    inv_order[layout_order[dim] as usize] = dim;
                }
                ndarray::Array::uninit(alloc_shape).permuted_axes(inv_order)
            }
        };

        if !is_empty {
            let index = unsafe { Index::new_unchecked(&start, &size) };
            let buf_strides = dim_arr(ndim, |dim| array.strides()[dim] as usize * itemsize);
            let buf = unsafe {
                cast_slice_mut::<MaybeUninit<S::Item>, u8>(std::slice::from_raw_parts_mut(
                    array.as_mut_ptr(),
                    array.len(),
                ))
            };
            let mut out = unsafe { StridedBuf::from_slice_mut(buf, buf_strides) };
            self.to_buf_strided_checked(&index, context, OutBuf::push(&mut out))?;
        }
        Ok(unsafe { array.assume_init() })
    }

    /// Read a rectangular sub-region into a caller-supplied byte buffer.
    ///
    /// The raw I/O primitive underlying [`to_ndarray`](Array::to_ndarray) and
    /// [`to_ndarray_sub`](Array::to_ndarray_sub), and the contiguous-destination shorthand for
    /// [`to_buf_strided`](Array::to_buf_strided). `buf` must be exactly
    /// `index.iter().map(|r| r.len()).product() * self.dtype().itemsize()` bytes; it needs no
    /// particular alignment. Elements are written in row-major (C-contiguous) order.
    ///
    /// Use [`to_buf_strided`](Array::to_buf_strided) instead when the destination is strided rather
    /// than packed, or when you would rather let the array hand back a buffer of its own than
    /// allocate one up front.
    ///
    /// # Errors
    ///
    /// - [`InvalidBufferSize`](crate::error::ErrorKind::InvalidBufferSize) - `buf` has the wrong
    ///   length for the requested range and dtype.
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    ///
    /// let context = a.read_ctx();
    /// let mut buf = vec![0u32; 4];
    /// {
    ///     let buf =
    ///         unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4) };
    ///     a.to_buf_slice(&[1..3, 1..3], buf, &context)?;
    /// }
    /// assert_eq!(buf, vec![5, 6, 8, 9]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline]
    pub fn to_buf_slice(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()> {
        let dtype = self.dtype();
        check_ranges(self.shape(), index)?;
        let (start, size) = split_ranges(index);
        let strides = default_strides_slice(&size, dtype.itemsize() as usize);
        check_get_buffer_size(&size, dtype, buf)?;
        if size.contains(&0) {
            return Ok(());
        }
        let index = unsafe { Index::new_unchecked(&start, &size) };
        check_region_size(&size, dtype.itemsize() as usize)?;
        let mut out = unsafe { StridedBuf::from_slice_mut(buf, strides) };
        self.to_buf_strided_checked(&index, context, OutBuf::push(&mut out))?;
        Ok(())
    }

    /// Read a rectangular sub-region and return it as a [`StridedBuf`].
    ///
    /// The raw I/O primitive underlying [`to_ndarray`](Array::to_ndarray) and
    /// [`to_ndarray_sub`](Array::to_ndarray_sub), and [`to_buf_slice`](Array::to_buf_slice).
    ///
    /// The two modes mirror NumPy's `out=`:
    /// - **pull** (`out` is `None`): the impl returns the cheapest buffer it can - a borrowed view
    ///   into its own memory (possibly with broadcast/stride-0 dims), or a freshly materialized one -
    ///   and picks the strides. The bytes are only contiguous if the strides say so.
    /// - **push** (`out` is `Some(dst)`): the region must be written into `dst` at `dst`'s own strides,
    ///   and the returned `StridedBuf` is a view of `dst`. `dst` must be writable and have one stride
    ///   per array dimension.
    ///
    /// For the common case of pushing into a packed row-major buffer, use
    /// [`to_buf_slice`](Array::to_buf_slice), which builds the `StridedBuf` for you.
    ///
    /// # Arguments
    ///
    /// - `index` - one half-open range (`start..end`) per dimension, within the array bounds.
    /// - `context` - read context carrying reusable buffers and decoder state.
    /// - `out` - optional destination, as described above.
    ///
    /// # Alignment
    ///
    /// No alignment is required of `out`, and none is promised of the returned buffer: it is only
    /// as aligned as its backing bytes happen to be. Reinterpret the bytes as typed values with
    /// unaligned reads, or stage the region through a buffer of your own.
    ///
    /// # Errors
    ///
    /// - [`InvalidIndex`](crate::error::ErrorKind::InvalidIndex) - `index` has the wrong number
    ///   of ranges, or a range starts after it ends or reaches past the array shape.
    /// - [`InvalidArgument`](crate::error::ErrorKind::InvalidArgument) - `out` is a read-only
    ///   view, or has a number of strides that does not match the array's dimensions.
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// Pull mode - the array hands back a buffer, and the strides say where each element lives:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    /// let context = a.read_ctx();
    ///
    /// let buf = a.to_buf_strided(&[1..3, 1..3], &context, None)?;
    /// let (base, strides) = (buf.data_ptr(), buf.strides());
    /// let mut values = Vec::new();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         let offset = i * strides[0] + j * strides[1];
    ///         values.push(unsafe { base.add(offset).cast::<i32>().read_unaligned() });
    ///     }
    /// }
    /// assert_eq!(values, vec![5, 6, 8, 9]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// Push mode - write the region into a column-major destination of your own:
    ///
    /// ```
    /// use jix::storage::StridedBuf;
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    /// let context = a.read_ctx();
    ///
    /// // 2x2 destination, column-major: stride 4 bytes down a column, 8 bytes across a row.
    /// let mut dst = [0i32; 4];
    /// {
    ///     let mut out = unsafe {
    ///         StridedBuf::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), &[2, 2], &[4, 8], 4)
    ///     };
    ///     a.to_buf_strided(&[1..3, 1..3], &context, Some(&mut out))?;
    /// }
    /// assert_eq!(dst, [5, 8, 6, 9]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline]
    pub fn to_buf_strided<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_ranges(self.shape(), index)?;
        let (start, size) = split_ranges(index);
        if size.contains(&0) {
            return self.read_empty(&size, context, out);
        }
        let index = unsafe { Index::new_unchecked(&start, &size) };
        let itemsize = self.dtype().itemsize() as usize;
        check_region_size(index.size(), itemsize)?;
        match out {
            None => {
                let mut slot = None;
                self.to_buf_strided_checked(&index, context, OutBuf::pull(&mut slot))?;
                Ok(slot.expect("a pull read puts its buffer in the slot"))
            }
            Some(out) => {
                check_out_buf_impl(out, &index, itemsize)?;
                let mut out = out.view_mut();
                self.to_buf_strided_checked(&index, context, OutBuf::push(&mut out))?;
                Ok(out)
            }
        }
    }

    #[cold]
    fn read_empty<'a>(
        &'a self,
        size: &[usize],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_region_size(size, self.dtype().itemsize() as usize)?;
        Ok(match out {
            Some(out) => {
                check_out_buf_dims(out, size.len())?;
                out.view_mut()
            }
            None => {
                let spec = self.storage.spec();
                alloc_out_buf(context, size, self.dtype(), spec.read_layout_order())
            }
        })
    }

    // Like `to_buf_strided`, for an `index` the caller checked against the shape and the region
    // size, and an `out` that passes `check_out_buf_impl`.
    #[inline]
    fn to_buf_strided_checked<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        out: OutBuf<'_, 'a>,
    ) -> Result<()> {
        let dtype = self.dtype();
        let spec = self.storage.spec();
        let small_read = spec.flags().plain_read() || {
            let nitems = index.size().iter().product::<usize>();
            let (min_nitems, _) = spec.read_size().nitems(dtype.itemsize());
            nitems as u64 <= min_nitems
        };
        let _cache_scope = context.cache_scope(&spec.dynamic().arrays_to_cache);
        if small_read {
            // Fast path for small reads
            self.storage.read_data(index, context, out)
        } else {
            to_buf_strided_slow(&self.storage, index, context, out)
        }
    }
}

#[inline(never)]
fn to_buf_strided_slow<'a, S: ArrayStorage>(
    storage: &'a S,
    index: &Index<'_>,
    context: &'a ReadContext,
    out: OutBuf<'_, 'a>,
) -> Result<()> {
    // Erase the storage so the body is monomorphized once
    let storage: &dyn ArrayStorage = storage;
    to_buf_strided_slow_dyn_dispatch(storage, index, context, out)
}

#[inline(never)]
fn to_buf_strided_slow_dyn_dispatch<'a>(
    storage: &'a dyn ArrayStorage,
    index: &Index<'_>,
    context: &'a ReadContext,
    out: OutBuf<'_, 'a>,
) -> Result<()> {
    let to_ndarray_fn = match storage.shape().len() {
        0 => to_buf_strided_slow_dyn::<Dim<0>>,
        1 => to_buf_strided_slow_dyn::<Dim<1>>,
        2 => to_buf_strided_slow_dyn::<Dim<2>>,
        3 => to_buf_strided_slow_dyn::<Dim<3>>,
        4 => to_buf_strided_slow_dyn::<Dim<4>>,
        5 => to_buf_strided_slow_dyn::<Dim<5>>,
        6 => to_buf_strided_slow_dyn::<Dim<6>>,
        7 => to_buf_strided_slow_dyn::<Dim<7>>,
        8 => to_buf_strided_slow_dyn::<Dim<8>>,
        _ => to_buf_strided_slow_dyn::<DimDyn>,
    };
    to_ndarray_fn(storage, index, context, out)
}

// index range and buffer size are not checked
#[inline(never)]
fn to_buf_strided_slow_dyn<'a, D: Dimension>(
    storage: &'a dyn ArrayStorage,
    index: &Index<'_>,
    context: &'a ReadContext,
    out: OutBuf<'_, 'a>,
) -> Result<()> {
    let shape = storage.shape();
    let ndim = index.ndim();
    let dtype = storage.dtype();

    let spec = storage.spec();
    let out_shape = index.size();
    let out = out.materialize(context, out_shape, dtype, spec.read_layout_order());
    debug_assert!(!out_shape.contains(&0));
    let max_shape = D::vec(ndim, |dim| out_shape[dim] as u64);
    let read_shape = spec.read_shape_heuristic::<D>(max_shape.as_ref(), shape, dtype.itemsize());
    // Block-space begin/end for NdIter.
    let block_begin = D::vec(ndim, |dim| index.start()[dim] / read_shape[dim]);
    let block_end = D::vec(ndim, |dim| {
        calc_block_end(index.start()[dim], index.end(dim), read_shape[dim])
    });
    // NdIter that yields blocks of size <= read_shape
    let block_iter = NdIter::builder_with_begin(block_begin, block_end)
        .with_block_offset_size_ext(
            &D::vec(ndim, |dim| index.start()[dim]),
            &D::vec(ndim, |dim| index.end(dim)),
            D::vec(ndim, |dim| read_shape[dim]),
        )
        .build();

    let (out_buf, out_strides) = out.data_mut();

    for (block_idx, (block_inner_offset, block_size)) in block_iter {
        let block_start = D::vec(ndim, |dim| {
            block_idx[dim] * read_shape[dim] + block_inner_offset[dim]
        });

        let out_offset = (0..ndim)
            .map(|dim| (block_start[dim] - index.start()[dim]) as usize * out_strides[dim])
            .sum::<usize>();

        let mut block_out =
            unsafe { StridedBuf::from_slice_mut(&mut out_buf[out_offset..], out_strides.clone()) };
        let block_size = D::vec(ndim, |dim| block_size[dim] as usize);
        let block_index =
            unsafe { Index::new_unchecked(block_start.as_ref(), block_size.as_ref()) };
        storage.read_data_push(&block_index, context, &mut block_out)?;
    }
    Ok(())
}

impl<S: ArrayStorage> Array<S> {
    /// Read the entire array into a fresh heap-allocated `Array<Plain>`.
    ///
    /// Similar to [`to_ndarray`](Array::to_ndarray), but returns a jix `Array<Plain>` instead of an
    /// `ndarray::Array`.
    /// Equivalent to `Array::plain_ndarray(self.to_ndarray()?)`.
    ///
    /// Like [`compact`](Array::compact), `to_plain` can be used to materialize an intermediate
    /// array: it breaks the lazy storage chain - be it an operation pipeline or a shape view such
    /// as the output of [`reshape`](Array::reshape) - so that later reads consume the stored result
    /// directly instead of re-running the whole pipeline. The difference is that `to_plain` keeps
    /// the result uncompressed, skipping the compression pass [`compact`](Array::compact) performs,
    /// so it is cheaper to produce and to read back at the cost of holding the full data in memory.
    ///
    /// This is especially useful when a deep or expensive view is broadcast into another operation.
    /// Because broadcasting reads the same logical elements many times, each read would otherwise
    /// re-evaluate the entire view; materializing it once with `to_plain` (or
    /// [`compact`](Array::compact)) avoids the repeated work. For example, dividing an array by its
    /// per-row standard deviation (`a / a.std(dim=1)` in NumPy terms; in Rust the reduction must be
    /// broadcast back to `a`'s shape explicitly) recomputes the reduction for every element it is
    /// broadcast against unless the `std` result is materialized first.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.0f64, 3.0], [4.0, 8.0]])?;
    ///
    /// // Dividing `a` by its per-row std broadcasts the reduction across the columns. Without
    /// // `to_plain` the `std` view would be re-evaluated for every column; materializing it once
    /// // computes each row's std a single time.
    /// let std = a.view().std(1, 0.0).insert_dim(1).to_plain()?; // shape [2, 1]
    /// let normalized = (a / std.broadcast(&[2, 2])).to_ndarray()?;
    ///
    /// assert_eq!(normalized, array![[1.0f64, 3.0], [2.0, 4.0]]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn to_plain(&self) -> Result<PlainArray<S::Item>>
    where
        S: ArrayStorageTyped,
    {
        let array = self.to_ndarray::<ndarray::IxDyn>()?;
        Array::plain_ndarray(array)
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::{array, Ix1, Ix2, Ix3, IxDyn};

    use super::Array;

    use crate::error::ErrorKind;

    use crate::storage::StridedBuf;

    use crate::ArrayParams;

    use crate::array::tests::*;

    // -----------------------------------------------------------------------
    // to_ndarray - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], [4], &[4]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![0, 1, 2, 3]);
    }

    #[test]
    fn to_ndarray_1d_two_blocks() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], [6], &[3]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn to_ndarray_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], [8], &[4]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![10, 20, 30, 40, 50, 60, 70, 80]);
    }

    // -----------------------------------------------------------------------
    // to_ndarray - 2D
    // Block-major order: block [r,c] = row-major grid index r*ncols_blocks+c.
    // shape=[4,6], block_shape=[2,3] -> grid 2*2:
    //   block0=[0,0]: rows 0-1, cols 0-2 -> 0,1,2,6,7,8
    //   block1=[0,1]: rows 0-1, cols 3-5 -> 3,4,5,9,10,11
    //   block2=[1,0]: rows 2-3, cols 0-2 -> 12,13,14,18,19,20
    //   block3=[1,1]: rows 2-3, cols 3-5 -> 15,16,17,21,22,23
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_2d() {
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            [4, 6],
            &[2, 3],
        );
        let got = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([4, 6], (0u8..24).collect()).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // to_ndarray_sub - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_1d_full_range() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], [6], &[3]);
        let got = a.to_ndarray_sub(&[0..6], &a.read_ctx()).unwrap();
        assert_eq!(got, array![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn to_ndarray_sub_1d_aligned_second_block() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], [6], &[3]);
        let got = a.to_ndarray_sub(&[3..6], &a.read_ctx()).unwrap();
        assert_eq!(got, array![3, 4, 5]);
    }

    #[test]
    fn to_ndarray_sub_1d_cross_block_boundary() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], [6], &[3]);
        let got = a.to_ndarray_sub(&[1..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1, 2, 3, 4]);
    }

    #[test]
    fn to_ndarray_sub_1d_within_single_block() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], [6], &[3]);
        let got = a.to_ndarray_sub(&[1..2], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1]);
    }

    // -----------------------------------------------------------------------
    // to_ndarray_sub - 2D
    // shape=[4,6], block_shape=[2,3], data as in to_ndarray_2d test.
    // range=[1..3, 2..5] -> output shape [2,3]:
    //   [8,  9,  10]
    //   [14, 15, 16]
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_2d() {
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            [4, 6],
            &[2, 3],
        );
        let got = a.to_ndarray_sub(&[1..3, 2..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![[8u8, 9, 10], [14, 15, 16]]);
    }

    #[test]
    fn to_buf_slice_into_a_misaligned_buf() {
        // A packed destination needs no particular alignment. Run under Miri to check the accesses.
        let src = ndarray::Array2::from_shape_fn((3, 4), |(r, c)| (r * 4 + c) as i32);
        let a = Array::compact_ndarray(&src).unwrap();
        let ctx = a.read_ctx();

        // A byte window starting one byte past a 4-aligned address.
        let mut backing = [0u8; 12 * 4 + 4];
        let off = backing.as_ptr().align_offset(4) + 1;
        a.to_buf_slice(&[0..3, 0..4], &mut backing[off..off + 12 * 4], &ctx)
            .unwrap();

        let got = (0..12)
            .map(|i| unsafe {
                backing
                    .as_ptr()
                    .add(off + i * 4)
                    .cast::<i32>()
                    .read_unaligned()
            })
            .collect::<Vec<_>>();
        assert_eq!(got, (0..12i32).collect::<Vec<_>>());
    }

    // -----------------------------------------------------------------------
    // to_buf_strided - pull and push modes, over the block-splitting slow path
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_empty_range_on_a_hinted_layout() {
        // `to_ndarray_sub` takes its strides from the allocation, and ndarray zeroes *every*
        // stride once any extent is 0 - so an empty read must not depend on them being dense.
        let a = Array::compact_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let t = a.view().permute_dims(&[1, 0]);
        let ctx = t.read_ctx();
        assert_eq!(
            t.to_ndarray_sub::<Ix2>(&[1..1, 0..2], &ctx)
                .unwrap()
                .shape(),
            &[0, 2]
        );
        assert_eq!(
            t.to_ndarray_sub::<Ix2>(&[0..3, 1..1], &ctx)
                .unwrap()
                .shape(),
            &[3, 0]
        );
        assert_eq!(
            t.to_ndarray_sub::<Ix2>(&[2..2, 1..1], &ctx)
                .unwrap()
                .shape(),
            &[0, 0]
        );
    }

    #[test]
    fn to_ndarray_allocates_in_the_hinted_layout() {
        let a = Array::compact_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        // A plain C-order read stays C-contiguous.
        let nd = a.to_ndarray().unwrap();
        assert_eq!(nd, ndarray::array![[1i32, 2, 3], [4, 5, 6]]);
        assert!(nd.as_slice().is_some());

        // Transposed: the destination follows the source's layout, so it comes back F-contiguous
        // and the copy runs straight through instead of transposing every element.
        let t = a.view().permute_dims(&[1, 0]).to_ndarray().unwrap();
        assert_eq!(t, ndarray::array![[1i32, 4], [2, 5], [3, 6]]);
        assert_eq!(t.strides(), &[1, 3]);
        assert!(t.as_slice().is_none());
    }

    #[test]
    fn to_ndarray_empty_region_too_large_errors() {
        let a = Array::compact_ndarray(&ndarray::array![[[7u8]]]).unwrap();
        let a = a.broadcast(&[0, 1 << 62, 3]);
        assert!(a.to_ndarray::<ndarray::IxDyn>().is_err());
    }

    #[test]
    fn pull_mode_allocates_in_the_hinted_layout() {
        let a = Array::compact_ndarray(
            &ndarray::Array::from_shape_vec([4, 5], (0..20i32).collect::<Vec<_>>()).unwrap(),
        )
        .unwrap();
        // Shape [5, 4] with layout order [1, 0]. Flip forwards the layout and allocates its own
        // pull-mode buffer, which must follow it: dim 1 outermost, dim 0 contiguous.
        let f = a.view().permute_dims(&[1, 0]).flip(0);
        let ctx = f.read_ctx();
        let buf = f.to_buf_strided(&[0..5, 0..4], &ctx, None).unwrap();
        assert_eq!(buf.strides(), &[4, 20]);
    }

    #[test]
    fn to_buf_strided_pull_and_push_over_a_split_read() {
        let src = ndarray::Array2::from_shape_fn((12, 10), |(r, c)| (r * 100 + c) as i32);
        // A tiny read-size budget forces the read to be cut into several pieces, so the test
        // exercises the block-splitting path rather than a single forwarded read.
        let params = ArrayParams::new()
            .block_shape(&[4, 5])
            .unwrap()
            .read_size((16, 64));
        let a = Array::compact_ndarray_with(&src, params).unwrap();
        let ctx = a.read_ctx();

        let index = [2..11, 1..9];
        let (rows, cols) = (9usize, 8usize);
        let at = |r: usize, c: usize| src[(r + 2, c + 1)];

        // Pull mode: the array hands back a buffer, and its strides say where each element sits.
        {
            let buf = a.to_buf_strided(&index, &ctx, None).unwrap();
            let (base, strides) = (buf.data_ptr(), buf.strides());
            for r in 0..rows {
                for c in 0..cols {
                    let got = unsafe {
                        base.add(r * strides[0] + c * strides[1])
                            .cast::<i32>()
                            .read_unaligned()
                    };
                    assert_eq!(got, at(r, c), "pull mismatch at ({r}, {c})");
                }
            }
        }

        // Push mode into a column-major destination: every piece of the split read has to land at
        // the destination's own strides, not at row-major ones.
        let itemsize = size_of::<i32>();
        let mut dst = vec![0i32; rows * cols];
        {
            let mut out = unsafe {
                StridedBuf::from_raw_parts_mut(
                    dst.as_mut_ptr().cast::<u8>(),
                    &[rows, cols],
                    &[itemsize, rows * itemsize],
                    itemsize,
                )
            };
            a.to_buf_strided(&index, &ctx, Some(&mut out)).unwrap();
        }
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(dst[c * rows + r], at(r, c), "push mismatch at ({r}, {c})");
            }
        }

        // The same region through the packed row-major shorthand.
        let mut dst = vec![0i32; rows * cols];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len() * itemsize)
            };
            a.to_buf_slice(&index, bytes, &ctx).unwrap();
        }
        let expected = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| at(r, c)))
            .collect::<Vec<_>>();
        assert_eq!(dst, expected);
    }

    #[test]
    fn to_buf_strided_empty_range() {
        let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let ctx = a.read_ctx();
        let buf = a.to_buf_strided(&[1..1, 0..3], &ctx, None).unwrap();
        assert_eq!(buf.strides().len(), 2);
    }

    #[test]
    fn empty_reads_write_nothing() {
        let a = Array::plain_ndarray(array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let sum = a.view() + a.view();
        let ctx = sum.read_ctx();
        for (index, shape) in [([1..1, 0..3], [0, 3]), ([0..2, 3..3], [2, 0])] {
            let read = sum.to_ndarray_sub::<ndarray::Ix2>(&index, &ctx).unwrap();
            assert_eq!(read.shape(), shape);
            assert!(sum.to_buf_strided(&index, &ctx, None).is_ok());
            sum.to_buf_slice(&index, &mut [], &ctx).unwrap();

            let mut dst = [7i32; 6];
            let mut out = unsafe {
                StridedBuf::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), &[2, 3], &[12, 4], 4)
            };
            sum.to_buf_strided(&index, &ctx, Some(&mut out)).unwrap();
            assert_eq!(dst, [7; 6]);
        }
    }

    #[test]
    fn to_buf_strided_rejects_a_bad_destination() {
        let a = Array::compact_ndarray(&array![[1i32, 2], [3, 4]]).unwrap();
        let ctx = a.read_ctx();
        let itemsize = size_of::<i32>();
        let mut dst = [0i32; 4];

        // A read-only view cannot be written into.
        let mut out = unsafe {
            StridedBuf::from_raw_parts(
                dst.as_ptr().cast::<u8>(),
                &[2, 2],
                &[2 * itemsize, itemsize],
                itemsize,
            )
        };
        let err = a
            .to_buf_strided(&[0..2, 0..2], &ctx, Some(&mut out))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);

        // One stride per array dimension, no more and no fewer.
        let mut out = unsafe {
            StridedBuf::from_raw_parts_mut(
                dst.as_mut_ptr().cast::<u8>(),
                &[4],
                &[itemsize],
                itemsize,
            )
        };
        let err = a
            .to_buf_strided(&[0..2, 0..2], &ctx, Some(&mut out))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    }

    #[test]
    fn to_buf_strided_rejects_a_bad_destination_of_a_split_read() {
        // A tiny read-size budget cuts the read into pieces, each read into a view of `out`.
        let params = ArrayParams::new()
            .block_shape(&[4, 4])
            .unwrap()
            .read_size((16, 64));
        let a = Array::compact_ndarray_with(&ndarray::Array2::<i32>::ones((8, 8)), params).unwrap();
        let ctx = a.read_ctx();
        let itemsize = size_of::<i32>();
        // A 4x8 destination for an 8x8 read: large enough for the first pieces only. It is
        // rejected before any piece is written.
        let mut dst = [0i32; 32];
        let mut out = unsafe {
            StridedBuf::from_raw_parts_mut(
                dst.as_mut_ptr().cast::<u8>(),
                &[4, 8],
                &[8 * itemsize, itemsize],
                itemsize,
            )
        };
        let err = a
            .to_buf_strided(&[0..8, 0..8], &ctx, Some(&mut out))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
        assert_eq!(dst, [0; 32]);
    }

    #[test]
    fn to_ndarray_checks_the_requested_ndim() {
        let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let ctx = a.read_ctx();

        // A static ndim must match the array's; a dynamic one always does.
        assert_eq!(a.to_ndarray::<Ix2>().unwrap(), array![[1, 2, 3], [4, 5, 6]]);
        assert_eq!(
            a.to_ndarray::<IxDyn>().unwrap(),
            array![[1, 2, 3], [4, 5, 6]].into_dyn()
        );
        for err in [
            a.to_ndarray::<Ix1>().unwrap_err(),
            a.to_ndarray::<Ix3>().unwrap_err(),
            a.to_ndarray_sub::<Ix1>(&[0..1, 0..2], &ctx).unwrap_err(),
        ] {
            assert_eq!(err.kind(), ErrorKind::InvalidShapeOperation);
        }
        // An invalid index is reported before the ndim mismatch.
        assert_eq!(
            a.to_ndarray_sub::<Ix1>(&[0..1], &ctx).unwrap_err().kind(),
            ErrorKind::InvalidIndex
        );
    }
}
