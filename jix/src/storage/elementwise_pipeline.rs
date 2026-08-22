use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::Range;

use crate::buf_pool::PoolBuf;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{ensure, Result};
use crate::ops::LanesInfo;
use crate::storage::StridedBuf;
use crate::{
    array_from_fn_inline, dim_arr, DimArray, DimDyn, NdCopier, NdIterUnordered, NdIterUnorderedDyn,
    OperandsArray, SliceExt, N_OPERANDS_MAX,
};

/// A lazily-evaluated read of one rectangular region, driven operand-first.
///
/// The pipeline mirrors the storage chain: operands hold the regions the backends produced, inner
/// nodes compute. `to_buf` picks the visitation order, positions the operand cursors and pulls the
/// result out through `read_bulk` - both on the crate-private `ElementwisePipelineImpl` half.
#[allow(private_bounds)]
pub trait ElementwisePipeline<T>: ElementwisePipelineImpl<T> {}

/// The per-node half of a [`ElementwisePipeline`] pipeline: what it reads from, and how it computes.
pub(crate) trait ElementwisePipelineImpl<T> {
    /// How many operands [`operands`](Self::operands) yields, if that is known at compile time.
    ///
    /// `None` when the count is only known at runtime, which happens for sequences of arrays with a
    /// dynamic length (`Vec<Array<_>>`, `&[Array<_>]`). Any node built on top of such a sequence
    /// propagates the `None`.
    const N_OPERANDS: Option<usize>;

    /// Every operand the pipeline draws from, in a fixed order that does not change between calls.
    fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's;

    /// Read the next `N` elements through the operand cursors and advance them.
    ///
    /// A node reads `N` elements from each of its children, combines them, and leaves every operand
    /// cursor `N * inner_stride` bytes further along - so a run of calls walks the run the caller
    /// set up, with no offset threaded through the chain.
    ///
    /// `CONTIGUOUS` promises `inner_stride == dtype.itemsize()` for *every* operand, letting the step
    /// fold into a compile-time constant.
    ///
    /// # Safety
    ///
    /// Every operand's `current_ptr` must be aligned for its dtype and have at least `N` elements at
    /// `inner_stride` in bounds of its `original_data`.
    unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(&self) -> [T; N];

    /// Run the pipeline over the whole read region, into `out` or into a fresh pooled buffer.
    ///
    /// The destination is operand 0 and the pipeline's own operands are 1..; all of them cover the
    /// same logical region, each at its own byte strides and element type. [`NdIterUnordered`]
    /// orders and coalesces the axes over that operand set, and each of its inner 1-d runs is
    /// walked here a chunk at a time.
    ///
    /// `read_bulk` loads whole elements with no unaligned fallback, so any operand whose bytes are
    /// not naturally aligned is staged through an aligned scratch buffer: each chunk is gathered
    /// into it before the run (or scattered back out after it, for the destination), and the
    /// pipeline then reads it contiguously.
    #[inline(never)]
    fn to_buf<'b>(
        self,
        index: &[Range<u64>],
        context: &'b ReadContext,
        out: Option<&'b mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'b>>
    where
        T: Dtyped,
        Self: Sized,
    {
        use crate::storage::materialize_out_buf;

        let shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
        let output_dtype = T::DTYPE;
        // The destination is described by `output_dtype` but written as `T`, so the two layouts
        // have to agree for the aligned writes in `inner_loop` to be sound.
        let mut out = materialize_out_buf(out, context, shape.as_ref(), &output_dtype);
        if shape.contains(&0) {
            return Ok(out); // empty region
        }

        let to_buf_fn = const {
            let n_operands = match Self::N_OPERANDS {
                Some(n_operands) => Some(1 + n_operands), // +1 for output
                None => None,
            };

            let mut to_buf_fn: Option<
                fn(Self, &[usize], &mut StridedBuf<'_>, &ReadContext) -> Result<()>,
            > = None;

            if let Some(n_operands) = n_operands {
                to_buf_fn = match n_operands {
                    1 => Some(to_buf_impl::<_, 1>),
                    2 => Some(to_buf_impl::<_, 2>),
                    3 => Some(to_buf_impl::<_, 3>),
                    4 => Some(to_buf_impl::<_, 4>),
                    5 => Some(to_buf_impl::<_, 5>),
                    6 => Some(to_buf_impl::<_, 6>),
                    7 => Some(to_buf_impl::<_, 7>),
                    8 => Some(to_buf_impl::<_, 8>),
                    9 => Some(to_buf_impl::<_, 9>),
                    10 => Some(to_buf_impl::<_, 10>),
                    11 => Some(to_buf_impl::<_, 11>),
                    12 => Some(to_buf_impl::<_, 12>),
                    _ => None,
                }
            };

            match to_buf_fn {
                Some(to_buf_fn) => to_buf_fn,
                None => to_buf_impl_dyn,
            }
        };

        to_buf_fn(self, shape.as_ref(), &mut out, context)?;

        Ok(out)
    }
}

/// Elements of one operand a scratch buffer holds, i.e. how far a single chunk reaches.
const CHUNK_LEN: usize = 8192;

/// The walk, with the operand count known at compile time.
///
/// Deliberately a near-copy of [`to_buf_impl_dyn`]: same tables, same staging, same order of
/// operations. What differs is that every per-operand table is a plain `[_; N_OPERANDS]` and the
/// iterator is monomorphized over the count, so the loops over operands unroll and the offsets stay
/// in registers. Keep the two in step.
fn to_buf_impl<T, const N_OPERANDS: usize>(
    pipeline: impl ElementwisePipelineImpl<T>,
    shape: &[usize],
    out: &mut StridedBuf<'_>,
    context: &ReadContext,
) -> Result<()>
where
    T: Dtyped,
{
    let output_dtype = T::DTYPE;

    // ---- operand table ---------------------------------------------------------------
    // The destination joins the pipeline's own operands as operand 0: all of them cover the same
    // logical region, each at its own byte strides and element type, and the walk orders and
    // coalesces the axes over the whole set. The only thing that sets the destination apart is
    // that its cursor is written through rather than read, which is what `is_output_operand`
    // marks below.
    //
    // `N_OPERANDS` counts that slot too, so it is one more than the pipeline's own count - which is
    // what `to_buf` instantiated this with, so the iterator below is exhausted exactly. Nothing
    // here caps the count: the tables are sized by the instantiation rather than by
    // `N_OPERANDS_MAX`.
    let out_operand = Operand::destination(out, &output_dtype);
    let is_output_operand = |i: usize| i == 0;
    let mut operand_iter = std::iter::once(&out_operand).chain(pipeline.operands());
    let operands: [_; N_OPERANDS] = array_from_fn_inline(|_| operand_iter.next().unwrap());
    debug_assert!(operand_iter.next().is_none());
    let layouts: [_; N_OPERANDS] = array_from_fn_inline(|i| {
        let dtype = operands[i].dtype;
        (dtype.itemsize() as usize, dtype.alignment().as_usize())
    });
    let strides: [_; N_OPERANDS] = array_from_fn_inline(|i| operands[i].strides());

    let iter = NdIterUnordered::new(shape, strides, layouts);
    let chunk_len_max = CHUNK_LEN.min(iter.inner_len());

    // ---- scratch buffers for the operands that are not naturally aligned ---------------
    // TODO: staging is driven purely by alignment. Gathering a *strided* operand into scratch
    // as well would trade a copy for a contiguous inner loop; worth measuring.
    let mut staging: [_; N_OPERANDS] = array_from_fn_inline(|i| {
        let operand = operands[i];
        let aligned =
            iter.is_aligned()[i] && (operand.base_ptr() as usize).is_multiple_of(layouts[i].1);
        (!aligned).then(|| Staging {
            buf: context.allocate_buf(chunk_len_max * layouts[i].0, operand.dtype.alignment()),
            copier: NdCopier::new(operand.dtype),
        })
    });

    // A staged operand is read (or written) straight out of its scratch buffer, so it is
    // contiguous whatever its own strides say.
    let contiguous = (0..N_OPERANDS).all(|i| iter.is_contiguous()[i] || staging[i].is_some());
    let inner_loop_fn = if contiguous {
        pick_inner_loop::<_, _, true>()
    } else {
        pick_inner_loop::<_, _, false>()
    };

    iter.foreach_inner_1d(|offsets, len, inner_strides| {
        for pos in (0..len).step_by(chunk_len_max) {
            let chunk_len = chunk_len_max.min(len - pos);

            // Point every cursor at this chunk: straight at the operand's own bytes, or at its
            // scratch buffer, which an input is gathered into on the way.
            for (i, operand) in operands.iter().enumerate() {
                // SAFETY: the walk only yields offsets inside every operand's region.
                let chunk_src =
                    unsafe { operand.base_ptr().add(offsets[i] + pos * inner_strides[i]) };
                match &mut staging[i] {
                    None => operand.set_cursor(chunk_src, inner_strides[i]),
                    Some(staging) => {
                        if !is_output_operand(i) {
                            unsafe {
                                staging.gather(
                                    chunk_src,
                                    chunk_len,
                                    inner_strides[i],
                                    operand.dtype,
                                )
                            };
                        }
                        operand.set_cursor(staging.buf.as_mut_slice().as_mut_ptr(), layouts[i].0);
                    }
                }
            }

            // The destination's cursor is the one the pipeline's output goes to. It carries write
            // provenance - `Operand::destination` took it from a writable buffer - and points at
            // `n` elements of `T`, either in the destination itself or in its scratch buffer.
            let dst = out_operand.current_ptr.get().cast_mut().cast::<T>();
            inner_loop_fn(&pipeline, dst, out_operand.inner_stride.get(), chunk_len);

            // Scatter the chunk just written back out of the destination's scratch buffer.
            if let Some(staging) = &staging[0] {
                let offset = offsets[0] + pos * inner_strides[0];
                // SAFETY: the walk only yields offsets inside the destination region, and the
                // scratch buffer holds the `n` elements the inner loop just wrote.
                unsafe {
                    staging.scatter(
                        out_operand.base_ptr().add(offset),
                        chunk_len,
                        inner_strides[0],
                        &output_dtype,
                    )
                };
            }
        }
    });

    Ok(())
}

/// The walk, for a pipeline whose operand count is only known at runtime.
///
/// The twin of [`to_buf_impl`], which runs whenever the count is a compile-time constant. Here the
/// per-operand tables are [`OperandsArray`]s, so the count is capped at [`N_OPERANDS_MAX`].
fn to_buf_impl_dyn<T>(
    pipeline: impl ElementwisePipelineImpl<T>,
    shape: &[usize],
    out: &mut StridedBuf<'_>,
    context: &ReadContext,
) -> Result<()>
where
    T: Dtyped,
{
    let output_dtype = T::DTYPE;

    // ---- operand table ---------------------------------------------------------------
    // The destination joins the pipeline's own operands as operand 0: all of them cover the same
    // logical region, each at its own byte strides and element type, and the walk orders and
    // coalesces the axes over the whole set. The only thing that sets the destination apart is
    // that its cursor is written through rather than read, which is what `is_output_operand`
    // marks below.
    //
    // That one slot leaves `N_OPERANDS_MAX - 1` for the pipeline itself. For a pipeline whose
    // operand count is known statically the check folds away at monomorphization.
    ensure!(
        pipeline.operands().count() < N_OPERANDS_MAX,
        InvalidArgument,
        "a read pipeline is limited to {} operands",
        N_OPERANDS_MAX,
    );
    let out_operand = Operand::destination(out, &output_dtype);
    let is_output_operand = |i: usize| i == 0;
    let operands = std::iter::once(&out_operand)
        .chain(pipeline.operands())
        .collect::<OperandsArray<&Operand<'_>>>();
    let layouts = operands
        .iter()
        .map(|operand| {
            let dtype = operand.dtype;
            (dtype.itemsize() as usize, dtype.alignment().as_usize())
        })
        .collect::<OperandsArray<_>>();
    let strides = operands
        .iter()
        .map(|operand| operand.strides())
        .collect::<OperandsArray<&[usize]>>();

    let iter = NdIterUnorderedDyn::new(shape, &strides, &layouts);
    let chunk_len_max = CHUNK_LEN.min(iter.inner_len());

    // ---- scratch buffers for the operands that are not naturally aligned ---------------
    // TODO: staging is driven purely by alignment. Gathering a *strided* operand into scratch
    // as well would trade a copy for a contiguous inner loop; worth measuring.
    let mut staging = operands
        .iter()
        .enumerate()
        .map(|(i, operand)| {
            let aligned =
                iter.is_aligned()[i] && (operand.base_ptr() as usize).is_multiple_of(layouts[i].1);
            (!aligned).then(|| Staging {
                buf: context.allocate_buf(chunk_len_max * layouts[i].0, operand.dtype.alignment()),
                copier: NdCopier::new(operand.dtype),
            })
        })
        .collect::<OperandsArray<_>>();

    // A staged operand is read (or written) straight out of its scratch buffer, so it is
    // contiguous whatever its own strides say.
    let contiguous = (0..operands.len()).all(|i| iter.is_contiguous()[i] || staging[i].is_some());
    let inner_loop_fn = if contiguous {
        pick_inner_loop::<_, _, true>()
    } else {
        pick_inner_loop::<_, _, false>()
    };

    iter.foreach_inner_1d(|offsets, len, inner_strides| {
        for pos in (0..len).step_by(chunk_len_max) {
            let chunk_len = chunk_len_max.min(len - pos);

            // Point every cursor at this chunk: straight at the operand's own bytes, or at its
            // scratch buffer, which an input is gathered into on the way.
            for (i, operand) in operands.iter().enumerate() {
                // SAFETY: the walk only yields offsets inside every operand's region.
                let chunk_src =
                    unsafe { operand.base_ptr().add(offsets[i] + pos * inner_strides[i]) };
                match &mut staging[i] {
                    None => operand.set_cursor(chunk_src, inner_strides[i]),
                    Some(staging) => {
                        if !is_output_operand(i) {
                            unsafe {
                                staging.gather(
                                    chunk_src,
                                    chunk_len,
                                    inner_strides[i],
                                    operand.dtype,
                                )
                            };
                        }
                        operand.set_cursor(staging.buf.as_mut_slice().as_mut_ptr(), layouts[i].0);
                    }
                }
            }

            // The destination's cursor is the one the pipeline's output goes to. It carries write
            // provenance - `Operand::destination` took it from a writable buffer - and points at
            // `n` elements of `T`, either in the destination itself or in its scratch buffer.
            let dst = out_operand.current_ptr.get().cast_mut().cast::<T>();
            inner_loop_fn(&pipeline, dst, out_operand.inner_stride.get(), chunk_len);

            // Scatter the chunk just written back out of the destination's scratch buffer.
            if let Some(staging) = &staging[0] {
                let offset = offsets[0] + pos * inner_strides[0];
                // SAFETY: the walk only yields offsets inside the destination region, and the
                // scratch buffer holds the `n` elements the inner loop just wrote.
                unsafe {
                    staging.scatter(
                        out_operand.base_ptr().add(offset),
                        chunk_len,
                        inner_strides[0],
                        &output_dtype,
                    )
                };
            }
        }
    });

    Ok(())
}

/// The aligned scratch buffer standing in for one operand whose own bytes are not.
///
/// `read_bulk` loads and stores whole elements with no unaligned fallback, so an operand that would
/// not be aligned is walked through this instead: an input's chunk is gathered in before the run,
/// the destination's is scattered back out after it, and in between the pipeline sees a contiguous,
/// aligned run either way.
struct Staging<'a> {
    buf: PoolBuf<'a>,
    copier: NdCopier<'a>,
}

impl Staging<'_> {
    /// Copy an input's chunk of `n` elements into the scratch buffer.
    ///
    /// # Safety
    ///
    /// `src` must point at the chunk's first element with `n` elements at `stride` in bounds behind
    /// it, in an allocation distinct from the scratch buffer, and the buffer must hold `n` elements
    /// of `dtype`.
    #[inline]
    unsafe fn gather(&mut self, src: *const u8, n: usize, stride: usize, dtype: &Dtype) {
        let itemsize = dtype.itemsize() as usize;
        // SAFETY: the caller vouches for `n` elements at `stride` behind `src`.
        let src = unsafe { std::slice::from_raw_parts(src, chunk_span(n, stride, itemsize)) };
        // SAFETY: the two buffers are distinct allocations, per the caller.
        unsafe {
            self.copier.copy(
                src,
                self.buf.as_mut_slice(),
                &[n],
                &[stride],
                &[itemsize],
                dtype,
            )
        };
    }

    /// Copy the `n` elements the pipeline just wrote out of the scratch buffer and into `dst`.
    ///
    /// # Safety
    ///
    /// Same contract as [`gather`](Self::gather), with `dst` in place of `src`.
    #[inline]
    unsafe fn scatter(&self, dst: *mut u8, n: usize, stride: usize, dtype: &Dtype) {
        let itemsize = dtype.itemsize() as usize;
        // SAFETY: the caller vouches for `n` elements at `stride` behind `dst`.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, chunk_span(n, stride, itemsize)) };
        // SAFETY: the two buffers are distinct allocations, per the caller.
        unsafe {
            self.copier.copy(
                self.buf.as_slice(),
                dst,
                &[n],
                &[itemsize],
                &[stride],
                dtype,
            )
        };
    }
}

/// The bytes a run of `n` elements `stride` apart spans, counting only up to the last one's end.
#[inline]
fn chunk_span(n: usize, stride: usize, itemsize: usize) -> usize {
    n.saturating_sub(1) * stride + itemsize
}

/// One flat run of `len` elements: pull them through the pipeline `LANES` at a time and write them
/// to `dst`, stepping `dst_stride` bytes per element.
///
/// Every operand cursor is already positioned at the run's first element, and both ends are
/// aligned (`to_buf` stages any operand that would not be), so the reads and writes are plain
/// aligned accesses.
#[inline(never)]
fn inner_loop<T, const LANES: usize, const CONTIGUOUS: bool>(
    pipeline: &impl ElementwisePipelineImpl<T>,
    dst: *mut T,
    dst_stride: usize,
    len: usize,
) where
    T: Dtyped,
{
    if CONTIGUOUS {
        debug_assert_eq!(dst_stride, size_of::<T>());
    }
    debug_assert!(dst.is_aligned());
    let mut i = 0;
    while i + LANES <= len {
        let chunk = unsafe { pipeline.read_bulk::<LANES, CONTIGUOUS>() };
        if CONTIGUOUS {
            let elms = unsafe { dst.add(i).cast::<[T; LANES]>() };
            unsafe { elms.write(chunk) };
        } else {
            #[allow(clippy::needless_range_loop)]
            for k in 0..LANES {
                let elm = unsafe { dst.cast::<u8>().add((i + k) * dst_stride).cast::<T>() };
                unsafe { elm.write(chunk[k]) };
            }
        }
        i += LANES;
    }
    while i < len {
        let [val] = unsafe { pipeline.read_bulk::<1, CONTIGUOUS>() };
        let elm = if CONTIGUOUS {
            unsafe { dst.add(i) }
        } else {
            unsafe { dst.cast::<u8>().add(i * dst_stride).cast::<T>() }
        };
        unsafe { elm.write(val) };
        i += 1;
    }
}

type InnerLoopFn<T, P> = fn(&P, *mut T, usize, usize);
fn pick_inner_loop<T, P, const CONTIGUOUS: bool>() -> InnerLoopFn<T, P>
where
    T: Dtyped,
    P: ElementwisePipelineImpl<T>,
{
    match <T as LanesInfo>::LANES {
        1 => inner_loop::<_, 1, CONTIGUOUS>,
        2 => inner_loop::<_, 2, CONTIGUOUS>,
        4 => inner_loop::<_, 4, CONTIGUOUS>,
        8 => inner_loop::<_, 8, CONTIGUOUS>,
        16 => inner_loop::<_, 16, CONTIGUOUS>,
        32 => inner_loop::<_, 32, CONTIGUOUS>,
        64 => inner_loop::<_, 64, CONTIGUOUS>,
        128 => inner_loop::<_, 128, CONTIGUOUS>,
        256 => inner_loop::<_, 256, CONTIGUOUS>,
        512 => inner_loop::<_, 512, CONTIGUOUS>,
        _ => inner_loop::<_, 1024, CONTIGUOUS>,
    }
}

/// One region a pipeline reads - or writes - plus the cursor it is walked through.
///
/// An operand holds the whole region it covers but never decides *where* in it to read:
/// [`to_buf`](ElementwisePipelineImpl::to_buf) owns the iteration order. Before each run of
/// [`read_bulk`](ElementwisePipelineImpl::read_bulk) calls it points `current_ptr` at the run's
/// first element and sets `inner_stride` to the byte step between consecutive elements; the operand
/// then reads straight off those two fields, advancing the cursor as it goes.
///
/// The read destination is an operand too - operand 0 of the walk - and works the same way, except
/// that its cursor is written through rather than read.
pub(crate) struct Operand<'a> {
    data: OperandData<'a>,
    /// The element type of the region, which need not be the pipeline's output type.
    dtype: &'a Dtype,

    /// The next element to read: null until `to_buf` positions it, then advanced by `read_bulk`.
    pub(crate) current_ptr: Cell<*const u8>,
    /// Byte step between consecutive elements of the current run.
    pub(crate) inner_stride: Cell<usize>,
}

/// The region behind an [`Operand`]: bytes to read, or bytes to write.
enum OperandData<'a> {
    /// A region already read out of a storage, owned here so a materialized read stays alive for
    /// as long as the pipeline. Its bytes are reached by reborrowing it, one chunk at a time.
    Read(StridedBuf<'a>),
    /// The caller's destination, as the write pointer to walk it through plus its byte strides.
    ///
    /// Held raw rather than as the [`StridedBuf`] it came from because a reborrow of that buffer -
    /// even a shared one, even just moving it - invalidates the pointer being written through.
    Write {
        base_ptr: *mut u8,
        strides: DimArray<usize>,
    },
}

impl<'a> Operand<'a> {
    /// Wrap an already-read region as an operand the pipeline reads through its cursor.
    #[inline]
    pub(crate) fn new(original_data: StridedBuf<'a>, dtype: &'a Dtype) -> Self {
        Self::from_data(OperandData::Read(original_data), dtype)
    }

    /// Wrap a write destination as the operand the pipeline's output is written through.
    ///
    /// `out` stays borrowed for as long as the operand lives, because from here on the bytes are
    /// only reachable through the pointer taken below.
    #[inline]
    fn destination(out: &'a mut StridedBuf<'_>, dtype: &'a Dtype) -> Self {
        // The layout before the pointer: a later shared reborrow of `out` would invalidate it.
        let strides = out.strides().to_dim_vec::<DimDyn>();
        let base_ptr = out.data_ptr_mut().expect("a read destination is writable");
        Self::from_data(OperandData::Write { base_ptr, strides }, dtype)
    }

    #[inline]
    fn from_data(data: OperandData<'a>, dtype: &'a Dtype) -> Self {
        Self {
            // Both cursor fields are set by `to_buf` before every run; there is nothing to read
            // here until then.
            current_ptr: Cell::new(std::ptr::null()),
            inner_stride: Cell::new(0),
            data,
            dtype,
        }
    }

    /// The byte strides of the whole region, one per dimension of the read region, 0 on a
    /// broadcast axis.
    #[inline]
    fn strides(&self) -> &[usize] {
        match &self.data {
            OperandData::Read(data) => data.strides(),
            OperandData::Write { strides, .. } => strides.as_ref(),
        }
    }

    /// The first byte of the region.
    ///
    /// Only a destination operand's pointer carries write provenance; the one a read operand hands
    /// back must never be written through.
    #[inline]
    fn base_ptr(&self) -> *mut u8 {
        match &self.data {
            OperandData::Read(data) => data.data_ptr().cast_mut(),
            OperandData::Write { base_ptr, .. } => *base_ptr,
        }
    }

    /// Point the cursor at the first of a run of elements `inner_stride` bytes apart.
    #[inline]
    fn set_cursor(&self, ptr: *const u8, inner_stride: usize) {
        self.current_ptr.set(ptr);
        self.inner_stride.set(inner_stride);
    }
}

pub(crate) struct OperandTyped<'a, T> {
    operand: Operand<'a>,
    _phantom: PhantomData<T>,
}

impl<'a, T> OperandTyped<'a, T> {
    /// Present `data` - the region already read out of a storage - as a pipeline.
    #[inline]
    pub(crate) unsafe fn new(data: StridedBuf<'a>, dtype: &'a Dtype) -> Self {
        Self {
            operand: Operand::new(data, dtype),
            _phantom: PhantomData,
        }
    }
}

impl<T: Dtyped> ElementwisePipelineImpl<T> for OperandTyped<'_, T> {
    const N_OPERANDS: Option<usize> = Some(1);

    #[inline]
    fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
        std::iter::once(&self.operand)
    }

    #[inline(always)]
    unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(&self) -> [T; N] {
        let base = self.operand.current_ptr.get().cast::<T>();
        debug_assert!(!base.is_null());
        debug_assert!(base.is_aligned());

        let stride = self.operand.inner_stride.get();
        if CONTIGUOUS {
            debug_assert_eq!(stride, size_of::<T>());
            let vals = unsafe { base.cast::<[T; N]>().read() };

            self.operand
                .current_ptr
                .set(unsafe { base.add(N).cast::<u8>() });
            vals
        } else {
            let vals = array_from_fn_inline(|i| unsafe {
                base.cast::<u8>().add(i * stride).cast::<T>().read()
            });

            self.operand
                .current_ptr
                .set(unsafe { base.cast::<u8>().add(N * stride) });
            vals
        }
    }
}
impl<T: Dtyped> ElementwisePipeline<T> for OperandTyped<'_, T> {}

pub(crate) const fn n_operands_sum(counts: &[Option<usize>]) -> Option<usize> {
    let mut total = 0;
    let mut i = 0;
    while i < counts.len() {
        match counts[i] {
            Some(n) => total += n,
            None => return None,
        }
        i += 1;
    }
    Some(total)
}

pub(crate) const fn n_operands_mul(count: Option<usize>, n: usize) -> Option<usize> {
    match count {
        Some(count) => Some(count * n),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array_from_fn_inline, strided_span_bytes, Array, ArrayStorage, Ty};

    // ---------------------------------------------------------------------------
    // Harness
    // ---------------------------------------------------------------------------

    /// An over-aligned byte buffer whose element 0 sits `misalign` bytes in, so a region can be
    /// forced onto the staged path.
    struct Bytes {
        backing: Vec<u64>,
        misalign: usize,
    }
    impl Bytes {
        fn new(len: usize, misalign: usize) -> Self {
            Self {
                backing: vec![0u64; (len + misalign).div_ceil(8) + 1],
                misalign,
            }
        }
        fn ptr(&self) -> *const u8 {
            unsafe { self.backing.as_ptr().cast::<u8>().add(self.misalign) }
        }
        fn ptr_mut(&mut self) -> *mut u8 {
            unsafe { self.backing.as_mut_ptr().cast::<u8>().add(self.misalign) }
        }
        fn get<T: Copy>(&self, byte_offset: usize) -> T {
            unsafe { self.ptr().add(byte_offset).cast::<T>().read_unaligned() }
        }
        fn set<T: Copy>(&mut self, byte_offset: usize, value: T) {
            unsafe {
                self.ptr_mut()
                    .add(byte_offset)
                    .cast::<T>()
                    .write_unaligned(value)
            };
        }
    }

    /// Byte strides for a row-major array whose backing shape is `shape[d] * mult[d]`, sampling one
    /// logical element every `mult[d]` slots along axis `d`. `mult[d] == 0` broadcasts axis `d`.
    fn strided_strides(shape: &[usize], mult: &[usize], itemsize: usize) -> Vec<usize> {
        let ndim = shape.len();
        let backing = (0..ndim)
            .map(|d| shape[d] * mult[d].max(1))
            .collect::<Vec<_>>();
        let mut cstr = vec![0usize; ndim];
        let mut acc = itemsize;
        for d in (0..ndim).rev() {
            cstr[d] = acc;
            acc *= backing[d];
        }
        (0..ndim).map(|d| cstr[d] * mult[d]).collect()
    }

    /// The byte offset of every logical element of `shape`, in row-major order.
    fn offsets(shape: &[usize], strides: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        let mut out = Vec::new();
        if shape.contains(&0) {
            return out;
        }
        let mut idx = vec![0usize; ndim];
        loop {
            out.push((0..ndim).map(|d| idx[d] * strides[d]).sum::<usize>());
            let mut d = ndim;
            loop {
                if d == 0 {
                    return out;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Run `lhs + rhs` over `shape` through `to_buf` and compare against a plain element-wise
    /// reference. `mults` / `misaligns` are `[dst, lhs, rhs]`; `push` selects the `out=` mode.
    #[track_caller]
    fn check_add(shape: &[usize], mults: [&[usize]; 3], misaligns: [usize; 3], push: bool) {
        type T = u32;
        let itemsize = size_of::<T>();
        let strides: [Vec<usize>; 3] =
            std::array::from_fn(|i| strided_strides(shape, mults[i], itemsize));
        let offs: [Vec<usize>; 3] = std::array::from_fn(|i| offsets(shape, &strides[i]));
        let nitems = offs[0].len();

        let mut bufs: [Bytes; 3] = std::array::from_fn(|i| {
            Bytes::new(
                strided_span_bytes(shape, &strides[i], itemsize),
                misaligns[i],
            )
        });
        // Only the elements a *broadcast* source actually exposes get a value; every logical
        // element then reads whichever backing slot its stride lands on.
        for (k, (&lhs_off, &rhs_off)) in offs[1].iter().zip(&offs[2]).enumerate() {
            bufs[1].set::<T>(lhs_off, k as T * 7 + 1);
            bufs[2].set::<T>(rhs_off, k as T * 13 + 5);
        }
        let expected = (0..nitems)
            .map(|k| bufs[1].get::<T>(offs[1][k]) + bufs[2].get::<T>(offs[2][k]))
            .collect::<Vec<T>>();

        let [dst, lhs, rhs] = &mut bufs;
        // let node = AddNode::<T> {
        //     lhs: OperandTyped::new(
        //         unsafe { StridedBuf::from_raw_parts(lhs.ptr(), shape, &strides[1], itemsize) },
        //         &dtype,
        //     ),
        //     rhs: OperandTyped::new(
        //         unsafe { StridedBuf::from_raw_parts(rhs.ptr(), shape, &strides[2], itemsize) },
        //         &dtype,
        //     ),
        // };

        let shape_u64 = shape.iter().map(|&s| s as u64).collect::<Vec<_>>();
        let lhs_array: Array<crate::storage::Plain<_, Ty<T>, _>> = unsafe {
            Array::plain_ndarray_ptr(
                lhs.ptr(),
                &shape_u64,
                &strides[1],
                T::DTYPE,
                Default::default(),
            )
            .unwrap()
        };
        let rhs_array: Array<crate::storage::Plain<_, Ty<T>, _>> = unsafe {
            Array::plain_ndarray_ptr(
                rhs.ptr(),
                &shape_u64,
                &strides[2],
                T::DTYPE,
                Default::default(),
            )
            .unwrap()
        };

        let context = ReadContext::default();
        let index = shape.iter().map(|&s| 0..s as u64).collect::<Vec<_>>();

        let lhs_pipeline = lhs_array
            .storage
            .read_as_elementwise_pipeline::<T>(&index, &context)
            .unwrap();
        let rhs_pipeline = rhs_array
            .storage
            .read_as_elementwise_pipeline::<T>(&index, &context)
            .unwrap();

        // ---------------------------------------------------------------------------
        // A hand-written binary node over two `OperandTyped` leaves - the shape a real op takes.
        // ---------------------------------------------------------------------------
        struct AddNode<P1, P2> {
            lhs: P1,
            rhs: P2,
        }
        impl<
                T: Dtyped + std::ops::Add<Output = T>,
                P1: ElementwisePipelineImpl<T>,
                P2: ElementwisePipelineImpl<T>,
            > ElementwisePipelineImpl<T> for AddNode<P1, P2>
        {
            const N_OPERANDS: Option<usize> = Some(2);

            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.lhs.operands().chain(self.rhs.operands())
            }
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(&self) -> [T; N] {
                let lhs = unsafe { self.lhs.read_bulk::<N, CONTIGUOUS>() };
                let rhs = unsafe { self.rhs.read_bulk::<N, CONTIGUOUS>() };
                array_from_fn_inline(|i| lhs[i] + rhs[i])
            }
        }
        impl<
                T: Dtyped + std::ops::Add<Output = T>,
                P1: ElementwisePipelineImpl<T>,
                P2: ElementwisePipelineImpl<T>,
            > ElementwisePipeline<T> for AddNode<P1, P2>
        {
        }
        let node = AddNode {
            lhs: lhs_pipeline,
            rhs: rhs_pipeline,
        };

        if push {
            {
                let mut out = unsafe {
                    StridedBuf::from_raw_parts_mut(dst.ptr_mut(), shape, &strides[0], itemsize)
                };
                node.to_buf(&index, &context, Some(&mut out)).unwrap();
            }
            let got = (0..nitems)
                .map(|k| dst.get::<T>(offs[0][k]))
                .collect::<Vec<T>>();
            assert_eq!(got, expected, "shape={shape:?} strides={strides:?}");
        } else {
            let buf = node.to_buf(&index, &context, None).unwrap();
            let c_strides = crate::util::default_strides_slice(shape, itemsize);
            let got = offsets(shape, c_strides.as_ref())
                .into_iter()
                .map(|off| unsafe { buf.data_ptr().add(off).cast::<T>().read_unaligned() })
                .collect::<Vec<T>>();
            assert_eq!(got, expected, "shape={shape:?} strides={strides:?}");
        }
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn contiguous_c_order() {
        for push in [false, true] {
            check_add(&[3, 4], [&[1, 1]; 3], [0; 3], push);
        }
    }

    #[test]
    fn strided_operands_split_the_inner_run() {
        // Gaps along the inner axis stop the coalescing, so every operand keeps its own inner
        // stride and the non-contiguous inner loop runs.
        for push in [false, true] {
            check_add(&[3, 4], [&[1, 2], &[2, 3], &[1, 1]], [0; 3], push);
        }
    }

    #[test]
    fn misaligned_operands_are_staged() {
        // A 1-byte base offset makes u32 reads unaligned, so each of these operands goes through a
        // scratch buffer - one at a time, then all three at once.
        for push in [false, true] {
            check_add(&[3, 4], [&[1, 1]; 3], [1, 0, 0], push);
            check_add(&[3, 4], [&[1, 1]; 3], [0, 1, 0], push);
            check_add(&[3, 4], [&[1, 1]; 3], [0, 0, 1], push);
            check_add(&[3, 4], [&[1, 1]; 3], [1, 1, 1], push);
            // Staged *and* strided: the gather has to honor the source stride.
            check_add(&[3, 4], [&[1, 2], &[2, 3], &[1, 1]], [1, 1, 1], push);
        }
    }

    #[test]
    fn broadcast_operand() {
        // A 0 stride repeats one element along the axis, both for a direct read and for a gather
        // into scratch.
        for push in [false, true] {
            check_add(&[3, 4], [&[1, 1], &[0, 1], &[1, 0]], [0; 3], push);
            check_add(&[3, 4], [&[1, 1], &[0, 1], &[1, 0]], [0, 1, 1], push);
        }
    }

    #[test]
    fn inner_run_longer_than_one_chunk() {
        // Longer than `CHUNK_LEN`, so a run is processed in several chunks and the cursors have to
        // pick up where the previous chunk left off. Miri walks every element by hand, so there it
        // only gets the misaligned case, and only just past the boundary.
        let long = if cfg!(miri) { 8200 } else { 20000 };
        for push in [false, true] {
            check_add(&[long], [&[1]; 3], [1, 1, 1], push);
            if !cfg!(miri) {
                check_add(&[long], [&[1]; 3], [0; 3], push);
                check_add(&[3, long], [&[1, 2]; 3], [1, 1, 1], push);
            }
        }
    }

    #[test]
    fn scalar_and_size_one_axes() {
        for push in [false, true] {
            check_add(&[], [&[]; 3], [0; 3], push);
            check_add(&[], [&[]; 3], [1, 1, 1], push);
            check_add(&[1, 5, 1], [&[1, 1, 1]; 3], [0; 3], push);
        }
    }

    #[test]
    fn empty_region_writes_nothing() {
        for push in [false, true] {
            check_add(&[0], [&[1]; 3], [0; 3], push);
            check_add(&[2, 0, 3], [&[1, 1, 1]; 3], [0; 3], push);
        }
    }

    #[test]
    fn prop_matches_elementwise_reference() {
        // Random ranks, extents, per-operand gaps/broadcasts and base misalignments: whatever axis
        // order, coalescing, staging and chunking `to_buf` picks, the result must be the plain
        // element-wise sum.
        use proptest::prelude::*;
        use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

        let strategy = (0usize..=3).prop_flat_map(|ndim| {
            (
                prop::collection::vec(1usize..=4, ndim),
                prop::collection::vec(1usize..=2, ndim), // dst must not broadcast
                prop::collection::vec(0usize..=2, ndim),
                prop::collection::vec(0usize..=2, ndim),
                prop::collection::vec(0usize..=1, 3),
                any::<bool>(),
            )
        });
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x1E4Fu64.to_le_bytes());
        TestRunner::new_with_rng(
            Config {
                cases: if cfg!(miri) { 16 } else { 256 },
                failure_persistence: None,
                ..Config::default()
            },
            TestRng::from_seed(RngAlgorithm::ChaCha, &seed),
        )
        .run(&strategy, |(shape, m0, m1, m2, misaligns, push)| {
            let misaligns = [misaligns[0], misaligns[1], misaligns[2]];
            check_add(&shape, [&m0, &m1, &m2], misaligns, push);
            Ok(())
        })
        .unwrap();
    }

    // ---------------------------------------------------------------------------
    // End to end: real storage chains driven through the pipeline
    // ---------------------------------------------------------------------------

    /// Run `storage` through the element-wise pipeline over `index` and return the elements in
    /// row-major order, both in pull mode and pushed into a strided destination.
    #[track_caller]
    fn pipeline_elements<S, T>(storage: &S, index: &[Range<u64>]) -> Vec<T>
    where
        S: crate::ArrayStorage,
        T: Dtyped + std::fmt::Debug + PartialEq,
    {
        let context = ReadContext::default();
        let shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<Vec<_>>();
        let nitems = shape.iter().product::<usize>();
        let itemsize = size_of::<T>();

        let pulled = {
            let buf = storage
                .read_as_elementwise_pipeline::<T>(index, &context)
                .unwrap()
                .to_buf(index, &context, None)
                .unwrap();
            let strides = crate::util::default_strides_slice(&shape, itemsize);
            offsets(&shape, strides.as_ref())
                .into_iter()
                .map(|off| unsafe { buf.data_ptr().add(off).cast::<T>().read_unaligned() })
                .collect::<Vec<T>>()
        };

        // Push into a destination with a gap along every axis, so the walk cannot coalesce it into
        // one contiguous run and the strided write path runs too.
        let mults = vec![2usize; shape.len()];
        let dst_strides = strided_strides(&shape, &mults, itemsize);
        let mut dst = Bytes::new(strided_span_bytes(&shape, &dst_strides, itemsize), 0);
        {
            let mut out = unsafe {
                StridedBuf::from_raw_parts_mut(dst.ptr_mut(), &shape, &dst_strides, itemsize)
            };
            storage
                .read_as_elementwise_pipeline::<T>(index, &context)
                .unwrap()
                .to_buf(index, &context, Some(&mut out))
                .unwrap();
        }
        let pushed = offsets(&shape, &dst_strides)
            .into_iter()
            .map(|off| dst.get::<T>(off))
            .collect::<Vec<T>>();

        assert_eq!(pulled.len(), nitems);
        assert_eq!(pulled, pushed, "pull and push modes disagree");
        pulled
    }

    /// The elements `read_data` produces for the same region - the reference every pipeline run is
    /// checked against.
    fn read_data_elements<S, T>(storage: &S, index: &[Range<u64>]) -> Vec<T>
    where
        S: crate::ArrayStorage,
        T: Dtyped,
    {
        let context = ReadContext::default();
        let shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<Vec<_>>();
        let buf = storage.read_data(index, &context, None).unwrap();
        offsets(&shape, buf.strides())
            .into_iter()
            .map(|off| unsafe { buf.data_ptr().add(off).cast::<T>().read_unaligned() })
            .collect::<Vec<T>>()
    }

    fn full_index(shape: &[u64]) -> Vec<Range<u64>> {
        shape.iter().map(|&s| 0..s).collect()
    }

    /// What the pipeline built for `storage` declares as its operand count, and how many operands
    /// it actually walks - one per leaf that stayed separate, so the latter is the direct measure of
    /// how far the chain fused.
    fn operand_counts<S, T>(storage: &S, index: &[Range<u64>]) -> (Option<usize>, usize)
    where
        S: crate::ArrayStorage,
        T: Dtyped,
    {
        fn declared<T, P: ElementwisePipelineImpl<T>>(_pipeline: &P) -> Option<usize> {
            P::N_OPERANDS
        }
        let context = ReadContext::default();
        let pipeline = storage
            .read_as_elementwise_pipeline::<T>(index, &context)
            .unwrap();
        (declared::<T, _>(&pipeline), pipeline.operands().count())
    }

    /// How many operands the pipeline walks, checking `N_OPERANDS` against it along the way - so
    /// every `operand_count` assertion below also pins down the declared constant.
    #[track_caller]
    fn operand_count<S, T>(storage: &S, index: &[Range<u64>]) -> usize
    where
        S: crate::ArrayStorage,
        T: Dtyped,
    {
        let (declared, walked) = operand_counts::<S, T>(storage, index);
        assert!(
            declared.is_none_or(|n| n == walked),
            "N_OPERANDS is {declared:?} but operands() yields {walked}"
        );
        walked
    }

    #[test]
    fn source_default_matches_read_data() {
        // The default `read_as_elementwise_pipeline`: a storage that does not
        // decompose reads its region once and presents itself as a single operand.
        let nd = ndarray::Array2::from_shape_fn((5, 7), |(i, j)| (i * 7 + j) as f32);
        let arr = crate::Array::compact_ndarray(&nd).unwrap();
        let index = full_index(arr.storage.shape());
        assert_eq!(
            pipeline_elements::<_, f32>(&arr.storage, &index),
            read_data_elements::<_, f32>(&arr.storage, &index),
        );
    }

    #[test]
    fn op1_and_op2_chains_match_read_data() {
        use core::ops::{Add, Mul, Neg};

        // A fresh pair of arrays per chain: `Array<Compact<..>>` owns its blocks and is not `Clone`.
        let a = || {
            let nd = ndarray::Array2::from_shape_fn((6, 9), |(i, j)| (i * 9 + j) as f32);
            crate::Array::compact_ndarray(&nd).unwrap()
        };
        let b = || {
            let nd = ndarray::Array2::from_shape_fn((6, 9), |(i, j)| (i as f32) - (j as f32) * 0.5);
            crate::Array::compact_ndarray(&nd).unwrap()
        };

        // Op1 over one leaf, Op2 over two, and Op1/Op2 stacked - three leaves under three nodes,
        // which only fuses into one pass if every node keeps its inputs as separate operands.
        let op1 = a().neg();
        let op2 = a().add(b());
        let nested = a().add(b()).neg().mul(a());

        // The whole point: each op keeps its inputs as separate operands, so a chain of n leaves
        // is walked as n operands in a single pass rather than collapsing into one buffered read.
        let index = full_index(&[6u64, 9]);
        assert_eq!(operand_count::<_, f32>(&op1.storage, &index), 1);
        assert_eq!(operand_count::<_, f32>(&op2.storage, &index), 2);
        assert_eq!(operand_count::<_, f32>(&nested.storage, &index), 3);

        for index in [vec![0..6, 0..9], vec![1..5, 2..8]] {
            assert_eq!(
                pipeline_elements::<_, f32>(&op1.storage, &index),
                read_data_elements::<_, f32>(&op1.storage, &index),
                "op1 index={index:?}"
            );
            assert_eq!(
                pipeline_elements::<_, f32>(&op2.storage, &index),
                read_data_elements::<_, f32>(&op2.storage, &index),
                "op2 index={index:?}"
            );
            assert_eq!(
                pipeline_elements::<_, f32>(&nested.storage, &index),
                read_data_elements::<_, f32>(&nested.storage, &index),
                "nested index={index:?}"
            );
        }
    }

    #[test]
    fn op2_over_a_plain_strided_view_stays_zero_copy() {
        use core::ops::Add;

        // `Plain` lends a view of its own buffer, so each leaf keeps the source strides - here a
        // reversed-axis view, whose operand is strided but never copied.
        let nd_a = ndarray::Array2::from_shape_fn((4, 6), |(i, j)| (i * 6 + j) as i32);
        let nd_b = ndarray::Array2::from_shape_fn((4, 6), |(i, j)| (i as i32) * 100 + j as i32);
        let nd_b_t = nd_b.t();
        let a = crate::Array::plain_ndarray_ref(&nd_a).unwrap();
        let b = crate::Array::plain_ndarray_ref(&nd_b_t).unwrap();
        let sum = a.add(b.permute_axes(&[1, 0]));
        let index = full_index(sum.storage.shape());
        assert_eq!(
            pipeline_elements::<_, i32>(&sum.storage, &index),
            read_data_elements::<_, i32>(&sum.storage, &index),
        );

        // Both operands point straight into the two ndarrays: nothing was copied, and the
        // permuted one contributes its remapped strides rather than a repacked buffer.
        let context = ReadContext::default();
        let pipeline = sum
            .storage
            .read_as_elementwise_pipeline::<i32>(&index, &context)
            .unwrap();
        let sources = [
            nd_a.as_ptr().cast::<u8>() as usize,
            nd_b.as_ptr().cast::<u8>() as usize,
        ];
        let operand_ptrs = pipeline
            .operands()
            .map(|operand| operand.base_ptr() as usize)
            .collect::<Vec<_>>();
        assert_eq!(operand_ptrs.len(), 2);
        for (&ptr, &src) in operand_ptrs.iter().zip(&sources) {
            let span = nd_a.len() * size_of::<i32>();
            assert!(
                (src..src + span).contains(&ptr),
                "operand at {ptr:#x} is a copy, not a view into {src:#x}"
            );
        }
    }

    #[test]
    fn op2_dtype_mismatch_is_rejected() {
        use core::ops::Add;

        let nd = ndarray::Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let sum = crate::Array::compact_ndarray(&nd)
            .unwrap()
            .add(crate::Array::compact_ndarray(&nd).unwrap());
        let index = full_index(sum.storage.shape());
        let context = ReadContext::default();
        assert!(sum
            .storage
            .read_as_elementwise_pipeline::<i64>(&index, &context)
            .is_err());
    }

    /// An `Array` broadcasting one `f32` across a 4x5 region.
    fn scalar_4x5(value: f32) -> crate::Array<crate::storage::scalar::Scalar<f32, crate::Dim<2>>> {
        crate::Array::from_storage(
            crate::storage::scalar::Scalar::new(value, [4u64, 5], crate::ArrayParams::default())
                .unwrap(),
        )
    }

    #[test]
    fn scalar_pipeline_has_no_operands() {
        // A `Scalar` has nothing to walk, so it contributes zero operands and the destination alone
        // drives the walk - both standalone and as one side of an `Op2`.
        use core::ops::Add;

        let scalar = scalar_4x5(7.0);
        let index = full_index(scalar.storage.shape());
        assert_eq!(operand_count::<_, f32>(&scalar.storage, &index), 0);
        assert_eq!(
            pipeline_elements::<_, f32>(&scalar.storage, &index),
            vec![7.0f32; 20],
        );

        let nd = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i * 5 + j) as f32);
        let sum = crate::Array::compact_ndarray(&nd)
            .unwrap()
            .add(scalar_4x5(7.0));
        assert_eq!(operand_count::<_, f32>(&sum.storage, &index), 1);
        assert_eq!(
            pipeline_elements::<_, f32>(&sum.storage, &index),
            read_data_elements::<_, f32>(&sum.storage, &index),
        );
    }

    #[test]
    fn where_pipeline_keeps_three_operands() {
        let nd_c = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i + j) % 2 == 0);
        let nd_x = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i * 5 + j) as i32);
        let nd_y = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| -((i * 5 + j) as i32));
        let selected = crate::ops::where_condition(
            crate::Array::compact_ndarray(&nd_c).unwrap(),
            crate::Array::compact_ndarray(&nd_x).unwrap(),
            crate::Array::compact_ndarray(&nd_y).unwrap(),
        );
        let index = full_index(selected.storage.shape());
        assert_eq!(operand_count::<_, i32>(&selected.storage, &index), 3);
        assert_eq!(
            pipeline_elements::<_, i32>(&selected.storage, &index),
            read_data_elements::<_, i32>(&selected.storage, &index),
        );
    }

    #[test]
    fn into_dim_and_into_type_forward_the_pipeline() {
        // Both are pure re-tags, so they must hand the inner pipeline through untouched rather than
        // fall back to buffering it into one operand.
        use core::ops::Add;

        let nd_a = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i * 5 + j) as f32);
        let nd_b = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (j * 4 + i) as f32);
        let sum = crate::Array::compact_ndarray(&nd_a)
            .unwrap()
            .add(crate::Array::compact_ndarray(&nd_b).unwrap());
        let index = full_index(sum.storage.shape());
        let expected = read_data_elements::<_, f32>(&sum.storage, &index);

        let re_dimmed = crate::ops::IntoDim::<_, crate::DimDyn>::new_array(sum).unwrap();
        assert_eq!(operand_count::<_, f32>(&re_dimmed.storage, &index), 2);
        assert_eq!(
            pipeline_elements::<_, f32>(&re_dimmed.storage, &index),
            expected
        );

        let re_typed = crate::ops::IntoType::<_, crate::TypeDyn>::new_array(re_dimmed).unwrap();
        assert_eq!(operand_count::<_, f32>(&re_typed.storage, &index), 2);
        assert_eq!(
            pipeline_elements::<_, f32>(&re_typed.storage, &index),
            expected
        );
    }

    // ---------------------------------------------------------------------------
    // Array sequences: every array of the sequence keeps its own operands
    // ---------------------------------------------------------------------------

    /// Three 4x5 `Compact` arrays with distinct contents.
    fn three_arrays() -> [crate::Array<crate::storage::Compact<crate::Ty<i32>, crate::Dim<2>>>; 3] {
        std::array::from_fn(|k| {
            let nd = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| {
                (i * 5 + j) as i32 * (k as i32 + 1)
            });
            crate::Array::compact_ndarray(&nd).unwrap()
        })
    }

    #[test]
    fn map_multiple_keeps_one_operand_per_array() {
        // One operand per array in the sequence, for each `ArraySequence` shape: a fixed-length
        // array, a `Vec`, a borrowed slice, and a heterogeneous tuple.
        let index = full_index(&[4u64, 5]);

        let fixed = crate::ops::map_multiple(three_arrays(), |xs: [i32; 3]| xs[0] + xs[1] + xs[2]);
        assert_eq!(operand_count::<_, i32>(&fixed.storage, &index), 3);
        assert_eq!(
            pipeline_elements::<_, i32>(&fixed.storage, &index),
            read_data_elements::<_, i32>(&fixed.storage, &index),
        );

        let owned = crate::ops::map_multiple(
            three_arrays().into_iter().collect::<Vec<_>>(),
            |xs: &[i32]| xs.iter().sum::<i32>(),
        );
        assert_eq!(operand_count::<_, i32>(&owned.storage, &index), 3);
        assert_eq!(
            pipeline_elements::<_, i32>(&owned.storage, &index),
            read_data_elements::<_, i32>(&owned.storage, &index),
        );

        let arrays = three_arrays();
        let borrowed = crate::ops::map_multiple(&arrays[..], |xs: &[i32]| xs.iter().sum::<i32>());
        assert_eq!(operand_count::<_, i32>(&borrowed.storage, &index), 3);
        assert_eq!(
            pipeline_elements::<_, i32>(&borrowed.storage, &index),
            read_data_elements::<_, i32>(&borrowed.storage, &index),
        );

        let fixed_ref = crate::ops::map_multiple(&arrays, |xs: [i32; 3]| xs[0] * 2 - xs[2]);
        assert_eq!(operand_count::<_, i32>(&fixed_ref.storage, &index), 3);
        assert_eq!(
            pipeline_elements::<_, i32>(&fixed_ref.storage, &index),
            read_data_elements::<_, i32>(&fixed_ref.storage, &index),
        );
    }

    #[test]
    fn map_multiple_over_a_tuple_of_mixed_dtypes() {
        let index = full_index(&[4u64, 5]);
        let nd_i = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i * 5 + j) as i32);
        let nd_f = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i as f32) - (j as f32) * 0.25);
        let nd_b = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i + j) % 2 == 0);
        let mixed = crate::ops::map_multiple(
            (
                crate::Array::compact_ndarray(&nd_i).unwrap(),
                crate::Array::compact_ndarray(&nd_f).unwrap(),
                crate::Array::compact_ndarray(&nd_b).unwrap(),
            ),
            |(x, y, flag): (i32, f32, bool)| if flag { x as f32 + y } else { y },
        );
        assert_eq!(operand_count::<_, f32>(&mixed.storage, &index), 3);
        assert_eq!(
            pipeline_elements::<_, f32>(&mixed.storage, &index),
            read_data_elements::<_, f32>(&mixed.storage, &index),
        );
    }

    #[test]
    fn map_multiple_fuses_with_the_ops_below_it() {
        // Each sequence element is itself an `Op2`, so the sequence must hand *both* of each
        // element's operands through: 3 arrays x 2 leaves = 6 operands in one pass.
        use core::ops::Add;

        let index = full_index(&[4u64, 5]);
        let sums = three_arrays().map(|arr| {
            let nd = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (j * 4 + i) as i32);
            arr.add(crate::Array::compact_ndarray(&nd).unwrap())
        });
        let mapped = crate::ops::map_multiple(sums, |xs: [i32; 3]| xs[0] ^ xs[1] ^ xs[2]);
        assert_eq!(operand_count::<_, i32>(&mapped.storage, &index), 6);
        assert_eq!(
            pipeline_elements::<_, i32>(&mapped.storage, &index),
            read_data_elements::<_, i32>(&mapped.storage, &index),
        );
    }

    /// The declared `N_OPERANDS` for a pipeline over `storage`.
    #[track_caller]
    fn declared_n_operands<S, T>(storage: &S, index: &[Range<u64>]) -> Option<usize>
    where
        S: crate::ArrayStorage,
        T: Dtyped,
    {
        operand_counts::<S, T>(storage, index).0
    }

    #[test]
    fn n_operands_is_known_unless_a_sequence_length_is_not() {
        use core::ops::{Add, Mul, Neg};

        let index = full_index(&[4u64, 5]);
        let compact = || {
            let nd = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i * 5 + j) as i32);
            crate::Array::compact_ndarray(&nd).unwrap()
        };
        let flags = || {
            let nd = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| (i + j) % 2 == 0);
            crate::Array::compact_ndarray(&nd).unwrap()
        };

        // Anything built from a fixed set of operands knows its count statically.
        let scalar = scalar_4x5(1.0);
        assert_eq!(
            declared_n_operands::<_, f32>(&scalar.storage, &index),
            Some(0)
        );

        let leaf = compact();
        assert_eq!(
            declared_n_operands::<_, i32>(&leaf.storage, &index),
            Some(1)
        );

        let op1 = compact().neg();
        assert_eq!(declared_n_operands::<_, i32>(&op1.storage, &index), Some(1));

        let op2 = compact().add(compact());
        assert_eq!(declared_n_operands::<_, i32>(&op2.storage, &index), Some(2));

        let nested = compact().add(compact()).neg().mul(compact());
        assert_eq!(
            declared_n_operands::<_, i32>(&nested.storage, &index),
            Some(3)
        );

        let selected = crate::ops::where_condition(flags(), compact(), compact());
        assert_eq!(
            declared_n_operands::<_, i32>(&selected.storage, &index),
            Some(3)
        );

        // `map_multiple` multiplies the per-array count through a fixed-length sequence...
        let fixed = crate::ops::map_multiple(three_arrays(), |xs: [i32; 3]| xs[0] + xs[1] + xs[2]);
        assert_eq!(
            declared_n_operands::<_, i32>(&fixed.storage, &index),
            Some(3)
        );

        let fused = crate::ops::map_multiple(
            three_arrays().map(|arr| arr.add(compact())),
            |xs: [i32; 3]| xs[0] ^ xs[1] ^ xs[2],
        );
        assert_eq!(
            declared_n_operands::<_, i32>(&fused.storage, &index),
            Some(6)
        );

        // ...and sums the per-element counts through a tuple: 1 + 1 + 2.
        let tuple = crate::ops::map_multiple(
            (compact(), compact().neg(), compact().add(compact())),
            |(a, b, c): (i32, i32, i32)| a + b + c,
        );
        assert_eq!(
            declared_n_operands::<_, i32>(&tuple.storage, &index),
            Some(4)
        );

        // A `Vec` or slice sequence only knows its length at runtime, and that `None` propagates
        // up through every node above it.
        let owned = crate::ops::map_multiple(
            three_arrays().into_iter().collect::<Vec<_>>(),
            |xs: &[i32]| xs.iter().sum::<i32>(),
        );
        assert_eq!(operand_counts::<_, i32>(&owned.storage, &index), (None, 3));

        let arrays = three_arrays();
        let borrowed = crate::ops::map_multiple(&arrays[..], |xs: &[i32]| xs.iter().sum::<i32>());
        assert_eq!(
            operand_counts::<_, i32>(&borrowed.storage, &index),
            (None, 3)
        );

        let above = borrowed.neg().add(compact());
        assert_eq!(operand_counts::<_, i32>(&above.storage, &index), (None, 4));
    }

    #[test]
    fn wide_static_sequence() {
        // Ten leaves plus the destination is eleven operands, past `N_OPERANDS_MAX`. A statically
        // counted pipeline is walked by `to_buf_impl`, whose tables are sized by the instantiation
        // rather than by that cap, so a chain this wide goes through in one pass.
        let arrays = std::array::from_fn::<_, 10, _>(|k| {
            let nd = ndarray::Array1::from_shape_fn(6, |i| (i + k) as i32);
            crate::Array::compact_ndarray(&nd).unwrap()
        });
        let index = full_index(&[6u64]);
        assert_eq!(
            operand_counts::<_, i32>(&arrays[0].storage, &index),
            (Some(1), 1)
        );

        let mapped = crate::ops::map_multiple(arrays, |xs: [i32; 10]| xs.iter().sum::<i32>());
        let out = mapped.to_ndarray().unwrap();
        // Array `k` holds `i + k`, so element `i` sums to `10 * i + 45`.
        assert_eq!(
            out.iter().copied().collect::<Vec<i32>>(),
            vec![45, 55, 65, 75, 85, 95]
        );
    }

    // TODO
    // #[test]
    // fn wide_sequence() {
    //     let n = 100usize;
    //     let arrays = (0..n)
    //         .map(|k| {
    //             let nd = ndarray::Array1::from_shape_fn(6, |i| (i + k) as i32);
    //             crate::Array::compact_ndarray(&nd).unwrap()
    //         })
    //         .collect::<Vec<_>>();
    //     let mapped = crate::ops::map_multiple(arrays, |xs: &[i32]| xs.iter().sum::<i32>());
    //     assert!(mapped.to_ndarray().is_ok());
    // }
}
