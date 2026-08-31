use std::cell::Cell;
use std::cmp::Reverse;
use std::marker::PhantomData;
use std::ops::Range;

use crate::buf_pool::PoolBuf;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::Result;
use crate::ops::LanesInfo;
use crate::storage::StridedBuf;
use crate::util::default_strides_slice;
use crate::{
    array_from_fn_inline, dim_arr, strided_span_bytes, ArrayExt, DimArray, DimDyn, DimIdx,
    NdCopier, NdIterUnordered, NdIterUnorderedDyn, SliceExt,
};

/// A lazy pipeline of element-wise operations over a rectangular region.
///
/// Returned by [`ArrayStorage::read_as_elementwise_pipeline`](crate::ArrayStorage::read_as_elementwise_pipeline).
/// Used internally, rarely should be touched by a user.
#[allow(private_bounds)]
pub trait ElementwisePipeline<T>: ElementwisePipelineImpl<T> {}

pub(crate) trait ElementwisePipelineImpl<T> {
    /// The number of leaf operands the pipeline reads from, if known at compile time.
    const N_OPERANDS: Option<usize>;

    /// The leaf operands the pipeline reads from.
    fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's;

    /// Read `N` elements at `offset` from the pipeline.
    ///
    /// A node reads `N` elements from each of its children at the same `offset` and combines them.
    /// Nothing is advanced: every leaf holds the base of the run its cursor was set to, and
    /// derives the address from `offset`, so the whole tree shares the caller's loop counter.
    ///
    /// `CONTIGUOUS` promises `inner_stride == dtype.itemsize()` for every operand the pipeline
    /// reads, letting the step fold into a compile-time constant.
    ///
    /// # Safety
    ///
    /// Every operand's `current_ptr` must be aligned for its dtype, and elements
    /// `offset..offset + N` at `inner_stride` must be in bounds of its `original_data`.
    unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(&self, offset: usize) -> [T; N];

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
        let output_dtype = T::DTYPE;
        let (shape, mut out) =
            materialize_pipeline_out_buf(index, &mut self.operands(), &output_dtype, context, out);
        if shape.contains(&0) {
            return Ok(out); // empty region
        }

        let to_buf_fn = const {
            #[allow(clippy::type_complexity)]
            let mut to_buf_fn: Option<
                fn(Self, &[usize], &mut StridedBuf<'_>, &ReadContext) -> Result<()>,
            > = None;

            if let Some(n_operands) = Self::N_OPERANDS {
                let n_operands = 1 + n_operands; // +1 for output
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
                    13 => Some(to_buf_impl::<_, 13>),
                    14 => Some(to_buf_impl::<_, 14>),
                    15 => Some(to_buf_impl::<_, 15>),
                    16 => Some(to_buf_impl::<_, 16>),
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
impl<P, T> ElementwisePipeline<T> for P where P: ElementwisePipelineImpl<T> {}

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
    let out_operand = Operand::new_output(out, &output_dtype);
    let operands = {
        let mut operand_iter = std::iter::once(&out_operand).chain(pipeline.operands());
        let operands = array_from_fn_inline::<_, N_OPERANDS>(|_| operand_iter.next().unwrap());
        debug_assert!(operand_iter.next().is_none());
        operands
    };

    let loop_cc = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, true, true>()(&pipeline, dst, dst_stride, len)
    };
    let loop_cs = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, true, false>()(&pipeline, dst, dst_stride, len)
    };
    let loop_sc = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, false, true>()(&pipeline, dst, dst_stride, len)
    };
    let loop_ss = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, false, false>()(&pipeline, dst, dst_stride, len)
    };
    let factory = |flags: InnerLoopFlags| {
        let loop_fn: &'_ InnerLoop<'_> = match (flags.inputs_contiguous, flags.output_contiguous) {
            (true, true) => &loop_cc,
            (true, false) => &loop_cs,
            (false, true) => &loop_sc,
            (false, false) => &loop_ss,
        };
        loop_fn
    };
    to_buf_type_erased(operands, &factory, shape, context);

    Ok(())
}

// like `to_buf_impl`, but the number of operands is not known at compile time.
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
    let out_operand = Operand::new_output(out, &output_dtype);
    let operands = std::iter::once(&out_operand)
        .chain(pipeline.operands())
        .collect::<Vec<_>>();

    let loop_cc = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, true, true>()(&pipeline, dst, dst_stride, len)
    };
    let loop_cs = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, true, false>()(&pipeline, dst, dst_stride, len)
    };
    let loop_sc = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, false, true>()(&pipeline, dst, dst_stride, len)
    };
    let loop_ss = |dst: &mut [u8], dst_stride: usize, len: usize| {
        pick_inner_loop::<T, _, false, false>()(&pipeline, dst, dst_stride, len)
    };
    let factory = |flags: InnerLoopFlags| {
        let loop_fn: &'_ InnerLoop<'_> = match (flags.inputs_contiguous, flags.output_contiguous) {
            (true, true) => &loop_cc,
            (true, false) => &loop_cs,
            (false, true) => &loop_sc,
            (false, false) => &loop_ss,
        };
        loop_fn
    };
    to_buf_type_erased_dyn(&operands, &factory, shape, context);
    Ok(())
}

#[inline(never)]
fn to_buf_type_erased<const N_OPERANDS: usize>(
    operands: [&Operand<'_>; N_OPERANDS],
    inner_loop_factory: InnerLoopFactory<'_>,
    shape: &[usize],
    context: &ReadContext,
) {
    let out_operand = operands[0];
    let output_dtype = out_operand.dtype;
    let is_output_operand = |op_i: usize| op_i == 0;
    let layouts = operands.map_inline_ref(|operand| {
        let dtype = operand.dtype;
        (dtype.itemsize(), dtype.alignment())
    });
    let strides = operands.map_inline_ref(|operand| operand.strides());

    let iter = NdIterUnordered::new(shape, strides, layouts);
    let chunk_len_max = iter.inner_len().min(Staging::BUFFER_SIZE);

    let mut staging = array_from_fn_inline::<_, N_OPERANDS>(|i| {
        let operand = operands[i];
        let aligned = iter.is_aligned()[i]
            && (operand.base_ptr() as usize).is_multiple_of(layouts[i].1.as_usize());
        (!aligned).then(|| Staging {
            buf: context.allocate_buf(
                chunk_len_max * layouts[i].0 as usize,
                operand.dtype.alignment(),
            ),
            copier: NdCopier::new(operand.dtype),
        })
    });

    // A staged operand is read (or written) straight out of its scratch buffer, so it is
    // contiguous whatever the original strides say.
    let operand_contiguous = |i: usize| iter.is_contiguous()[i] || staging[i].is_some();
    let inner_loop_flags = InnerLoopFlags {
        inputs_contiguous: (1..N_OPERANDS).all(&operand_contiguous),
        output_contiguous: operand_contiguous(0),
    };
    let inner_loop = inner_loop_factory(inner_loop_flags);

    iter.foreach_inner_1d(|offsets, len, inner_strides| {
        for pos in (0..len).step_by(chunk_len_max) {
            let chunk_len = chunk_len_max.min(len - pos);

            // Point every cursor at this chunk
            for (op_i, operand) in operands.iter().enumerate() {
                let chunk_src = unsafe {
                    operand
                        .base_ptr()
                        .add(offsets[op_i] + pos * inner_strides[op_i])
                };
                match &mut staging[op_i] {
                    None => operand.set_cursor(chunk_src, inner_strides[op_i]),
                    Some(staging) => {
                        if !is_output_operand(op_i) {
                            unsafe {
                                staging.gather(
                                    chunk_src,
                                    chunk_len,
                                    inner_strides[op_i],
                                    operand.dtype,
                                )
                            };
                        }
                        operand.set_cursor(
                            staging.buf.as_mut_slice().as_mut_ptr(),
                            layouts[op_i].0 as usize,
                        );
                    }
                }
            }

            let dst_stride = out_operand.inner_stride.get();
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    out_operand.current_ptr.get().cast_mut(),
                    strided_span_bytes(&[chunk_len], &[dst_stride], layouts[0].0 as usize),
                )
            };
            inner_loop(dst, dst_stride, chunk_len);

            if let Some(staging) = &staging[0] {
                // Scatter the chunk just written back out of the destination's scratch buffer.
                let offset = offsets[0] + pos * inner_strides[0];
                unsafe {
                    staging.scatter(
                        out_operand.base_ptr().add(offset),
                        chunk_len,
                        inner_strides[0],
                        output_dtype,
                    )
                };
            }
        }
    });
}

// Identical to `to_buf_type_erased`, but the number of operands is not known at compile time.
#[inline(never)]
fn to_buf_type_erased_dyn(
    operands: &[&Operand<'_>],
    inner_loop_factory: InnerLoopFactory<'_>,
    shape: &[usize],
    context: &ReadContext,
) {
    let out_operand = operands[0];
    let output_dtype = out_operand.dtype;
    let is_output_operand = |op_i: usize| op_i == 0;
    let layouts = operands
        .iter()
        .map(|operand| {
            let dtype = operand.dtype;
            (dtype.itemsize(), dtype.alignment())
        })
        .collect::<Vec<_>>();
    let strides = operands
        .iter()
        .map(|operand| operand.strides())
        .collect::<Vec<_>>();

    let iter = NdIterUnorderedDyn::new(shape, &strides, &layouts);
    let chunk_len_max = iter.inner_len().min(Staging::BUFFER_SIZE);

    let mut staging = operands
        .iter()
        .enumerate()
        .map(|(i, operand)| {
            let aligned = iter.is_aligned()[i]
                && (operand.base_ptr() as usize).is_multiple_of(layouts[i].1.as_usize());
            (!aligned).then(|| Staging {
                buf: context.allocate_buf(
                    chunk_len_max * layouts[i].0 as usize,
                    operand.dtype.alignment(),
                ),
                copier: NdCopier::new(operand.dtype),
            })
        })
        .collect::<Vec<_>>();

    // A staged operand is read (or written) straight out of its scratch buffer, so it is
    // contiguous whatever the original strides say.
    let operand_contiguous = |i: usize| iter.is_contiguous()[i] || staging[i].is_some();
    let inner_loop_flags = InnerLoopFlags {
        inputs_contiguous: (1..operands.len()).all(&operand_contiguous),
        output_contiguous: operand_contiguous(0),
    };
    let inner_loop = inner_loop_factory(inner_loop_flags);

    iter.foreach_inner_1d(|offsets, len, inner_strides| {
        for pos in (0..len).step_by(chunk_len_max) {
            let chunk_len = chunk_len_max.min(len - pos);

            // Point every cursor at this chunk
            for (op_i, operand) in operands.iter().enumerate() {
                let chunk_src = unsafe {
                    operand
                        .base_ptr()
                        .add(offsets[op_i] + pos * inner_strides[op_i])
                };
                match &mut staging[op_i] {
                    None => operand.set_cursor(chunk_src, inner_strides[op_i]),
                    Some(staging) => {
                        if !is_output_operand(op_i) {
                            unsafe {
                                staging.gather(
                                    chunk_src,
                                    chunk_len,
                                    inner_strides[op_i],
                                    operand.dtype,
                                )
                            };
                        }
                        operand.set_cursor(
                            staging.buf.as_mut_slice().as_mut_ptr(),
                            layouts[op_i].0 as usize,
                        );
                    }
                }
            }

            let dst_stride = out_operand.inner_stride.get();
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    out_operand.current_ptr.get().cast_mut(),
                    strided_span_bytes(&[chunk_len], &[dst_stride], layouts[0].0 as usize),
                )
            };
            inner_loop(dst, dst_stride, chunk_len);

            if let Some(staging) = &staging[0] {
                // Scatter the chunk just written back out of the destination's scratch buffer.
                let offset = offsets[0] + pos * inner_strides[0];
                unsafe {
                    staging.scatter(
                        out_operand.base_ptr().add(offset),
                        chunk_len,
                        inner_strides[0],
                        output_dtype,
                    )
                };
            }
        }
    });
}

/// A temporary buffer used to stage a chunk of an operand data during a pipeline run.
struct Staging<'a> {
    buf: PoolBuf<'a>,
    copier: NdCopier<'a>,
}

impl Staging<'_> {
    const BUFFER_SIZE: usize = 8192;

    /// Copy `n` elements from `src` into the scratch buffer, stepping `stride` bytes per element.
    ///
    /// # Safety
    ///
    /// `src` must point at the chunk's first element with `n` elements at `stride` in bounds behind
    /// it, in an allocation distinct from the scratch buffer, and the buffer must hold `n` elements
    /// of `dtype`.
    #[inline]
    unsafe fn gather(&mut self, src: *const u8, n: usize, stride: usize, dtype: &Dtype) {
        let itemsize = dtype.itemsize() as usize;
        let src_span = n.saturating_sub(1) * stride + itemsize;
        // SAFETY: the caller vouches for `n` elements at `stride` behind `src`.
        let src = unsafe { std::slice::from_raw_parts(src, src_span) };
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

    /// Copy `n` elements from the scratch buffer into `dst`, stepping `stride` bytes per element.
    ///
    /// # Safety
    ///
    /// `dst` must point at the chunk's first element with `n` elements at `stride` in bounds behind
    /// it, in an allocation distinct from the scratch buffer, and the buffer must hold `n` elements
    /// of `dtype`.
    #[inline]
    unsafe fn scatter(&self, dst: *mut u8, n: usize, stride: usize, dtype: &Dtype) {
        let itemsize = dtype.itemsize() as usize;
        let dst_span = n.saturating_sub(1) * stride + itemsize;
        // SAFETY: the caller vouches for `n` elements at `stride` behind `dst`.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, dst_span) };
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

#[inline(never)]
fn inner_loop<T, const LANES: usize, const IN_CONTIGUOUS: bool, const OUT_CONTIGUOUS: bool>(
    pipeline: &impl ElementwisePipelineImpl<T>,
    dst: &mut [u8],
    dst_stride: usize,
    len: usize,
) where
    T: Dtyped,
{
    if OUT_CONTIGUOUS {
        debug_assert_eq!(dst_stride, size_of::<T>());
    }
    let dst = dst.as_mut_ptr().cast::<T>();
    debug_assert!(dst.is_aligned());

    let body_limit = len - len % LANES;
    let mut i = 0;
    while i < body_limit {
        let chunk = unsafe { pipeline.read_bulk::<LANES, IN_CONTIGUOUS>(i) };
        if OUT_CONTIGUOUS {
            unsafe { dst.add(i).cast::<[T; LANES]>().write(chunk) };
        } else {
            #[allow(clippy::needless_range_loop)]
            for k in 0..LANES {
                unsafe {
                    dst.cast::<u8>()
                        .add((i + k) * dst_stride)
                        .cast::<T>()
                        .write(chunk[k])
                };
            }
        }
        i += LANES;
    }
    while i < len {
        let [val] = unsafe { pipeline.read_bulk::<1, IN_CONTIGUOUS>(i) };
        if OUT_CONTIGUOUS {
            unsafe { dst.add(i).write(val) };
        } else {
            unsafe { dst.cast::<u8>().add(i * dst_stride).cast::<T>().write(val) };
        }
        i += 1;
    }
}

#[derive(Clone, Copy)]
struct InnerLoopFlags {
    inputs_contiguous: bool,
    output_contiguous: bool,
}
type InnerLoop<'a> = dyn Fn(&mut [u8], usize, usize) + 'a;
type InnerLoopFactory<'a> = &'a dyn Fn(InnerLoopFlags) -> &'a InnerLoop<'a>;

fn pick_inner_loop<T, P, const IN_CONTIGUOUS: bool, const OUT_CONTIGUOUS: bool>(
) -> fn(&P, &mut [u8], usize, usize)
where
    T: Dtyped,
    P: ElementwisePipelineImpl<T>,
{
    const {
        let default_lanes = <T as LanesInfo>::LANES;
        const STRIDED_LANES: usize = 4;
        let lanes = if IN_CONTIGUOUS && OUT_CONTIGUOUS || default_lanes < STRIDED_LANES {
            default_lanes
        } else {
            STRIDED_LANES
        };
        match lanes {
            1 => inner_loop::<_, 1, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            2 => inner_loop::<_, 2, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            4 => inner_loop::<_, 4, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            8 => inner_loop::<_, 8, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            16 => inner_loop::<_, 16, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            32 => inner_loop::<_, 32, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            64 => inner_loop::<_, 64, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            128 => inner_loop::<_, 128, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            256 => inner_loop::<_, 256, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            512 => inner_loop::<_, 512, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
            _ => inner_loop::<_, 1024, IN_CONTIGUOUS, OUT_CONTIGUOUS>,
        }
    }
}

/// One region a pipeline reads - or writes - plus the cursor it is walked through.
///
/// An operand holds the whole region it covers but never decides *where* in it to read:
/// [`to_buf`](ElementwisePipelineImpl::to_buf) owns the iteration order. Before each run of
/// [`read_bulk`](ElementwisePipelineImpl::read_bulk) calls it points `current_ptr` at the run's
/// first element and sets `inner_stride` to the byte step between consecutive elements; the operand
/// then reads straight off those two fields, advancing the cursor as it goes.
pub(crate) struct Operand<'a> {
    data: OperandData<'a>,
    dtype: &'a Dtype,

    /// The next element to read: null until `to_buf` positions it, then advanced by `read_bulk`.
    pub(crate) current_ptr: Cell<*const u8>,
    /// Byte step between consecutive elements of the current run.
    pub(crate) inner_stride: Cell<usize>,
}
enum OperandData<'a> {
    Input(StridedBuf<'a>),
    Output {
        // we hold here a raw pointer rather than `StridedBuf` due to some Miri restrictions
        base_ptr: *mut u8,
        strides: DimArray<usize>,
    },
}

impl<'a> Operand<'a> {
    #[inline]
    pub(crate) fn new_input(original_data: StridedBuf<'a>, dtype: &'a Dtype) -> Self {
        Self::new_impl(OperandData::Input(original_data), dtype)
    }

    #[inline]
    pub(crate) fn new_output(out: &'a mut StridedBuf<'_>, dtype: &'a Dtype) -> Self {
        let strides = out.strides().to_dim_vec::<DimDyn>();
        let base_ptr = out.data_ptr_mut().unwrap();
        Self::new_impl(OperandData::Output { base_ptr, strides }, dtype)
    }

    #[inline]
    fn new_impl(data: OperandData<'a>, dtype: &'a Dtype) -> Self {
        Self {
            // Both cursor fields are set by `to_buf` before every run; there is nothing to read
            // here until then.
            current_ptr: Cell::new(std::ptr::null()),
            inner_stride: Cell::new(0),
            data,
            dtype,
        }
    }

    #[inline]
    fn strides(&self) -> &[usize] {
        match &self.data {
            OperandData::Input(data) => data.strides(),
            OperandData::Output { strides, .. } => strides.as_ref(),
        }
    }

    #[inline]
    fn base_ptr(&self) -> *mut u8 {
        match &self.data {
            OperandData::Input(data) => data.data_ptr().cast_mut(),
            OperandData::Output { base_ptr, .. } => *base_ptr,
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
    /// Create a new input operand from a region of bytes.
    ///
    /// # Safety
    ///
    /// The `data` must be a valid view of `dtype` elements, and the `dtype` must match `T::DTYPE`.
    #[inline]
    pub(crate) unsafe fn new_input(data: StridedBuf<'a>, dtype: &'a Dtype) -> Self {
        Self {
            operand: Operand::new_input(data, dtype),
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
    unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(&self, offset: usize) -> [T; N] {
        let base = self.operand.current_ptr.get().cast::<T>();
        debug_assert!(!base.is_null());
        debug_assert!(base.is_aligned());

        if CONTIGUOUS {
            debug_assert_eq!(self.operand.inner_stride.get(), size_of::<T>());
            unsafe { base.add(offset).cast::<[T; N]>().read() }
        } else {
            let stride = self.operand.inner_stride.get();
            let base = unsafe { base.cast::<u8>().add(offset * stride) };
            array_from_fn_inline(|k| unsafe { base.add(k * stride).cast::<T>().read() })
        }
    }
}

#[inline(never)]
fn materialize_pipeline_out_buf<'b, 's>(
    index: &[Range<u64>],
    operands: &mut dyn Iterator<Item = &'s Operand<'s>>,
    output_dtype: &Dtype,
    context: &'b ReadContext,
    out: Option<&'b mut StridedBuf<'_>>,
) -> (DimArray<usize>, StridedBuf<'b>) {
    let shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
    let out = match out {
        Some(out) => out.view_mut(),
        None => {
            let itemsize = output_dtype.itemsize() as usize;
            let strides = pick_output_layout(operands, shape.as_ref(), itemsize);
            let buf = context.allocate_buf(
                shape.iter().product::<usize>() * itemsize,
                output_dtype.alignment(),
            );
            unsafe { StridedBuf::from_pool(buf, strides.as_ref()) }
        }
    };
    (shape, out)
}

#[inline(never)]
fn pick_output_layout<'s>(
    operands: &mut dyn Iterator<Item = &'s Operand<'s>>,
    shape: &[usize],
    itemsize: usize,
) -> DimArray<usize> {
    fn pick_axis_order<'s>(
        operands: &mut dyn Iterator<Item = &'s Operand<'s>>,
        shape: &[usize],
        itemsize: usize,
    ) -> Option<DimArray<DimIdx>> {
        let ndim = shape.len();
        if ndim <= 1 || shape.iter().product::<usize>() * itemsize <= 4096 {
            return None;
        }

        let mut axis_order = None;
        for operand in operands {
            let strides = operand.strides();
            debug_assert_eq!(strides.len(), ndim);
            let mut operand_order = dim_arr(ndim, |d| d as DimIdx);
            operand_order.sort_by_key(|&d| {
                let d = d as usize;
                let ignore_axis = shape[d] <= 1 || strides[d] == 0;
                Reverse(if ignore_axis { usize::MAX } else { strides[d] })
            });
            match &axis_order {
                None => axis_order = Some(operand_order),
                Some(axis_order) if *axis_order == operand_order => {}
                Some(_) => return None,
            }
        }
        axis_order
    }

    let axis_order = pick_axis_order(operands, shape, itemsize);
    match axis_order {
        Some(axis_order) => {
            let mut strides = dim_arr(shape.len(), |_| itemsize);
            let mut stride = itemsize;
            for &d in axis_order.iter().rev() {
                let d = d as usize;
                strides[d] = stride;
                stride *= shape[d];
            }
            strides
        }
        None => default_strides_slice(shape, itemsize),
    }
}

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
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(
                &self,
                offset: usize,
            ) -> [T; N] {
                let lhs = unsafe { self.lhs.read_bulk::<N, CONTIGUOUS>(offset) };
                let rhs = unsafe { self.rhs.read_bulk::<N, CONTIGUOUS>(offset) };
                array_from_fn_inline(|i| lhs[i] + rhs[i])
            }
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
            // The buffer picks its own layout, so walk it by its own strides.
            let got = offsets(shape, buf.strides())
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
        // Longer than `Staging::BUFFER_SIZE`, so a run is processed in several chunks and the cursors have to
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
    fn out_layout_follows_the_operands() {
        type T = u32;
        let itemsize = size_of::<T>();
        let dtype = T::DTYPE;

        // 64 * 64 * 4 = 16 KiB, comfortably over 4096.
        let big = [64usize, 64];
        let c_strides = crate::util::default_strides_slice(&big, itemsize);
        let f_strides = [itemsize, itemsize * big[0]];
        let data = vec![0u8; big[0] * big[1] * itemsize];
        let operand = |strides: &[usize]| {
            // SAFETY: both layouts are dense over `big`, and `data` is sized for exactly that.
            let buf = unsafe { StridedBuf::from_raw_parts(data.as_ptr(), &big, strides, itemsize) };
            Operand::new_input(buf, &dtype)
        };

        let pick = |ops: &[Operand<'_>], shape: &[usize]| {
            pick_output_layout(&mut ops.iter(), shape, itemsize)
        };

        // Every operand F-ordered -> so is the output.
        let f_ops = [operand(&f_strides), operand(&f_strides)];
        assert_eq!(pick(&f_ops, &big).as_ref(), &f_strides);

        // Every operand C-ordered -> C order, same as before the layout matching.
        let c_ops = [operand(c_strides.as_ref()), operand(c_strides.as_ref())];
        assert_eq!(pick(&c_ops, &big).as_ref(), c_strides.as_ref());

        // Operands that disagree -> fall back to C order.
        let mixed = [operand(c_strides.as_ref()), operand(&f_strides)];
        assert_eq!(pick(&mixed, &big).as_ref(), c_strides.as_ref());

        // No operands at all -> C order.
        assert_eq!(pick(&[], &big).as_ref(), c_strides.as_ref());

        // Small outputs stay C-ordered whatever the operands look like: 8 * 8 * 4 = 256 bytes.
        let small = [8usize, 8];
        let small_c = crate::util::default_strides_slice(&small, itemsize);
        let small_f = [itemsize, itemsize * small[0]];
        let small_data = vec![0u8; small[0] * small[1] * itemsize];
        let small_op = |strides: &[usize]| {
            // SAFETY: both layouts are dense over `small`, and `small_data` is sized for that.
            let buf = unsafe {
                StridedBuf::from_raw_parts(small_data.as_ptr(), &small, strides, itemsize)
            };
            Operand::new_input(buf, &dtype)
        };
        let small_ops = [small_op(&small_f), small_op(&small_f)];
        assert_eq!(pick(&small_ops, &small).as_ref(), small_c.as_ref());
    }

    #[test]
    fn out_layout_ignores_axes_the_operand_does_not_walk() {
        type T = u32;
        let itemsize = size_of::<T>();
        let dtype = T::DTYPE;

        let shape = [64usize, 64];
        let c_strides = crate::util::default_strides_slice(&shape, itemsize);
        let f_strides = [itemsize, itemsize * shape[0]];
        let data = vec![0u8; shape[0] * shape[1] * itemsize];
        let pick = |strides: &[usize]| {
            // SAFETY: every layout used here stays inside `data`, sized for the full shape.
            let buf =
                unsafe { StridedBuf::from_raw_parts(data.as_ptr(), &shape, strides, itemsize) };
            let ops = [Operand::new_input(buf, &dtype)];
            pick_output_layout(&mut ops.iter(), &shape, itemsize)
        };

        // Broadcast along axis 0. Axis 1 is the only one actually walked, so it belongs innermost -
        // C order, not the F order a raw sort by stride would pick for a 0.
        assert_eq!(pick(&[0, itemsize]).as_ref(), c_strides.as_ref());
        // Broadcast along axis 1 instead: now axis 0 is the walked one, so it goes innermost.
        assert_eq!(pick(&[itemsize, 0]).as_ref(), &f_strides);

        // An extent-1 axis carries no preference either, whatever stride it happens to hold, so a
        // junk stride there must not flip the layout of the axes that do matter.
        let shape1 = [1usize, 64, 64];
        let c_strides1 = crate::util::default_strides_slice(&shape1, itemsize);
        // SAFETY: axis 0 has extent 1, so its stride is never stepped; the rest is dense in `data`.
        let buf = unsafe {
            StridedBuf::from_raw_parts(
                data.as_ptr(),
                &shape1,
                &[1, itemsize * 64, itemsize],
                itemsize,
            )
        };
        let ops = [Operand::new_input(buf, &dtype)];
        assert_eq!(
            pick_output_layout(&mut ops.iter(), &shape1, itemsize).as_ref(),
            c_strides1.as_ref()
        );
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
        // `to_buf`'s dispatch table instantiates `to_buf_impl::<_, N>` for operand counts up to
        // sixteen (the destination included) and hands anything wider to `to_buf_impl_dyn`. Both
        // sides of that boundary are reachable with a count known at compile time, so walk each:
        // fifteen leaves plus the destination is the last const-generic arm, sixteen overflows it.
        fn leaves<const N: usize>(
        ) -> [crate::Array<crate::storage::Compact<crate::Ty<i32>, crate::Dim<1>>>; N] {
            std::array::from_fn(|k| {
                let nd = ndarray::Array1::from_shape_fn(6, |i| (i + k) as i32);
                crate::Array::compact_ndarray(&nd).unwrap()
            })
        }
        // Leaf `k` holds `i + k`, so element `i` of an `n`-leaf sum is `n * i + n * (n - 1) / 2`.
        let expected = |n: usize| {
            (0..6)
                .map(|i| (n * i + n * (n - 1) / 2) as i32)
                .collect::<Vec<i32>>()
        };
        let index = full_index(&[6u64]);

        // 15 + 1 output = 16: the widest count with its own instantiation.
        let mapped =
            crate::ops::map_multiple(leaves::<15>(), |xs: [i32; 15]| xs.iter().sum::<i32>());
        assert_eq!(
            operand_counts::<_, i32>(&mapped.storage, &index),
            (Some(15), 15)
        );
        let out = mapped.to_ndarray().unwrap();
        assert_eq!(out.iter().copied().collect::<Vec<i32>>(), expected(15));

        // 16 + 1 output = 17: past the table, so the runtime-count walk runs even though
        // `N_OPERANDS` is `Some`.
        let mapped =
            crate::ops::map_multiple(leaves::<16>(), |xs: [i32; 16]| xs.iter().sum::<i32>());
        assert_eq!(
            operand_counts::<_, i32>(&mapped.storage, &index),
            (Some(16), 16)
        );
        let out = mapped.to_ndarray().unwrap();
        assert_eq!(out.iter().copied().collect::<Vec<i32>>(), expected(16));
    }

    #[test]
    fn wide_sequence() {
        // A `Vec` of arrays has no compile-time count, so the chain is walked by `to_buf_impl_dyn`,
        // which is unbounded: a hundred leaves plus the destination is a hundred and one operands.
        let n = 100usize;
        let arrays = (0..n)
            .map(|k| {
                let nd = ndarray::Array1::from_shape_fn(6, |i| (i + k) as i32);
                crate::Array::compact_ndarray(&nd).unwrap()
            })
            .collect::<Vec<_>>();
        let mapped = crate::ops::map_multiple(arrays, |xs: &[i32]| xs.iter().sum::<i32>());
        let index = full_index(mapped.storage.shape());
        assert_eq!(operand_counts::<_, i32>(&mapped.storage, &index), (None, n));

        let out = mapped.to_ndarray().unwrap();
        // Array `k` holds `i + k`, so element `i` sums to `100 * i + 4950`.
        assert_eq!(
            out.iter().copied().collect::<Vec<i32>>(),
            (0..6).map(|i| 100 * i + 4950).collect::<Vec<i32>>()
        );
    }
}
