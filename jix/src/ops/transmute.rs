use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{ensure, Result};
use crate::storage::{ArraySpec, ArrayStorageInfo, OutBuf};
use crate::{Array, ArrayStorage, ElementType, Ty, TypeDyn};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Reinterprets each element as the type `T` without converting the bytes. See [`Transmute`] for
    /// details and examples.
    ///
    /// # Safety
    ///
    /// Every element must be a valid bit pattern for `T` (see [`Transmute`]).
    #[track_caller]
    pub unsafe fn transmute_elements<T>(self) -> Array<Transmute<S, Ty<T>>>
    where
        T: Dtyped,
    {
        unsafe { Transmute::new_array(self, Ty::new()).unwrap() }
    }

    /// Reinterprets each element as the runtime dtype `dtype` without converting the bytes; recover a
    /// typed array with [`into_typed`](Array::into_typed). See [`Transmute`] for details.
    ///
    /// # Safety
    ///
    /// Every element must be a valid bit pattern for `dtype` (see [`Transmute`]).
    #[track_caller]
    pub unsafe fn transmute_elements_dyn(self, dtype: Dtype) -> Array<Transmute<S, TypeDyn>> {
        unsafe { Transmute::new_array(self, TypeDyn::from_dtype(dtype).unwrap()).unwrap() }
    }
}

/// Storage that reinterprets each element of the inner array as a different dtype of the same
/// itemsize, without converting or copying any bytes.
///
/// This is the array-level analogue of transmuting a slice: the stored bytes are unchanged and only
/// the element type is relabelled, so an `f32` array can be viewed as its raw `u32` bit patterns, or
/// a `#[derive(Dtyped)]` struct as an equally-sized `[u8; N]`. To numerically *convert* values
/// instead (e.g. round `f32` to `i32`), use [`Array::cast()`](crate::Array::cast). The source and
/// destination dtypes must have the same itemsize but may differ in alignment (e.g. `u32` align 4 vs
/// `[u8; 4]` align 1), which reads handle transparently. Output dtype is the new dtype; output shape
/// equals the input shape.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::transmute_elements()`](crate::Array::transmute_elements) and
/// [`Array::transmute_elements_dyn()`](crate::Array::transmute_elements_dyn).
///
/// # Safety
///
/// Constructing a `Transmute` is `unsafe`: the stored bytes are later read back as the new dtype, so
/// every element must be a valid bit pattern for it. This holds for any integer or float type (all
/// bit patterns are valid), but not for types with restricted representations such as `bool`.
///
/// # Examples
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1.0f32, 2.0, -0.5])?;
/// // View the f32 bits as u32 (both 4-byte elements) - no conversion.
/// let bits = unsafe { a.transmute_elements::<u32>() };
/// assert_eq!(bits.to_ndarray()?[0], 1.0f32.to_bits());
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Transmute<S, ET> {
    array: S,
    element_type: ET,
}
impl<S, ET> Transmute<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    /// Constructs a [`Transmute`] storage. See the struct docs for semantics and examples.
    ///
    /// Errors if `new_type`'s itemsize differs from `array`'s dtype itemsize.
    ///
    /// # Safety
    ///
    /// Every element of `array` must be a valid bit pattern for `new_type` (see [`Transmute`]).
    pub unsafe fn new(array: S, new_type: ET) -> Result<Self> {
        let src_dtype = array.dtype();
        let dst_dtype = new_type.dtype();
        ensure!(
            src_dtype.itemsize() == dst_dtype.itemsize(),
            UnsupportedDtype,
            "Cannot transmute between dtypes with different sizes: {src_dtype} vs {dst_dtype}"
        );
        Ok(Self {
            element_type: new_type,
            array,
        })
    }

    /// Constructs an array with [`Transmute`] storage. See the storage struct docs for semantics and examples.
    ///
    /// # Safety
    ///
    /// Every element of `array` must be a valid bit pattern for `new_type` (see [`Transmute`]).
    pub unsafe fn new_array(array: Array<S>, new_type: ET) -> Result<Array<Self>> {
        unsafe { Self::new(array.into_storage(), new_type).map(Array::from_storage) }
    }
}
impl<S, ET> ArrayStorage for Transmute<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    type ElementType = ET;
    type Dimension = S::Dimension;

    #[inline]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        let src_dtype = self.array.dtype();
        let dst_dtype = self.element_type.dtype();
        let (src_align, dst_align) = (
            src_dtype.alignment().as_usize(),
            dst_dtype.alignment().as_usize(),
        );
        let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;

        // `buf` is aligned to the destination dtype, but the inner read expects the source dtype's alignment.
        // That is a problem only for a *contiguous* caller buffer whose source alignment exceeds its
        // (weaker) destination alignment - a strided buffer carries no alignment precondition, and a
        // lazy buffer is pool-allocated to a cache line (over-aligned for any dtype). In that one case,
        // read through a source-aligned scratch buffer and copy the bytes across afterwards.
        let mut scratch = (src_align > dst_align
            && buf.strides().is_none()
            && matches!(buf.as_slice(), Some(s) if !(s.as_ptr() as usize).is_multiple_of(src_align)))
        .then(|| context.tmp_buf(nitems * src_dtype.itemsize() as usize, src_dtype.alignment()));
        let (bytes, strides) = match scratch.as_mut() {
            Some(scratch) => (scratch.as_mut_slice(), None),
            None => buf.get_mut(nitems, dst_dtype),
        };

        {
            let mut dst = match strides {
                Some(strides) => unsafe { OutBuf::new_strided(bytes, strides) },
                None => OutBuf::new(bytes),
            };
            self.array.read_data(index, &mut dst, context)?;
        }

        // If we staged, copy the identically-sized bytes back into the caller's buffer.
        if let Some(scratch) = scratch {
            let (dst, _) = buf.get_mut(nitems, dst_dtype);
            dst.copy_from_slice(scratch.as_slice());
        }
        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.array.shape()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.element_type.dtype()
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array.spec().with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Transmute", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Transmute<S::DimensionChange<NewD>, ET>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Transmute {
            array: self.array.dimension_change()?,
            element_type: self.element_type,
        })
    }

    type ElementTypeChange<NewET: ElementType> = Transmute<S, NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        Ok(Transmute {
            array: self.array,
            element_type: NewET::from_dtype(self.element_type.dtype().clone())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Transmute;
    use crate::dtype::Dtyped;
    use crate::storage::{OutBuf, ReadData};
    use crate::{Array, ArrayParams, ArrayStorage, Ty};

    /// Build a single-block `Compact` array from a 1-D `u32` ndarray, so a full read hits the
    /// contiguous single-block fast path (the one that enforces source-dtype buffer alignment).
    fn compact_u32_single_block(data: &[u32]) -> Array<impl ArrayStorage<ElementType = Ty<u32>>> {
        let mut params = ArrayParams::new();
        params.block_shape(&[data.len() as u32]);
        Array::compact_ndarray_with(&ndarray::Array1::from(data.to_vec()), params).unwrap()
    }

    /// A byte window of length `len` whose start is deliberately *not* a multiple of `align`, so it
    /// under-aligns any dtype whose alignment is `align`. Returns the backing buffer (keep it alive)
    /// and the offset of the window within it.
    fn misaligned_backing(len: usize, align: usize) -> (Vec<u8>, usize) {
        let backing = vec![0u8; len + align];
        let off = backing.as_ptr().align_offset(align) + 1;
        assert!(!(backing.as_ptr() as usize + off).is_multiple_of(align));
        assert!(off + len <= backing.len());
        (backing, off)
    }

    /// `Transmute::new` requires the source and destination dtypes to have the same itemsize.
    #[test]
    fn new_rejects_size_mismatch() {
        let arr = compact_u32_single_block(&[1, 2, 3]);
        // u32 (4 bytes) -> u16 (2 bytes): different itemsize, must be rejected.
        let result = unsafe { Transmute::new(arr.into_storage(), Ty::<u16>::new()) };
        assert!(result.is_err());
    }

    /// Same itemsize, same alignment (`f32` <-> `u32`): the bytes are reinterpreted unchanged, so a
    /// transmuted read yields each element's raw bit pattern.
    #[test]
    fn transmute_reinterprets_bits_when_alignment_matches() {
        let arr = Array::compact_ndarray(&ndarray::array![1.0f32, -2.0, 0.5]).unwrap();
        let bits = unsafe { arr.transmute_elements::<u32>() }
            .to_ndarray()
            .unwrap();
        assert_eq!(bits[0], 1.0f32.to_bits());
        assert_eq!(bits[1], (-2.0f32).to_bits());
        assert_eq!(bits[2], 0.5f32.to_bits());
    }

    /// `transmute_dyn` sets the runtime dtype while leaving shape and bytes untouched; the result can
    /// be recovered as a typed array with the new dtype.
    #[test]
    fn transmute_dyn_sets_runtime_dtype() {
        let arr = Array::compact_ndarray(&ndarray::array![1i32, 2, 3]).unwrap();
        let t = unsafe { arr.transmute_elements_dyn(u32::DTYPE) };
        assert_eq!(t.dtype(), &u32::DTYPE);
        assert_eq!(t.shape(), &[3]);
        let vals = t.into_typed::<u32>().unwrap().to_ndarray().unwrap();
        assert_eq!(vals.as_slice().unwrap(), &[1u32, 2, 3]);
    }

    /// Transmuting to a *weaker*-aligned dtype (`u32` align 4 -> `[u8; 4]` align 1): the caller's
    /// contiguous buffer only needs the destination alignment (1), so it can be under-aligned for the
    /// source read. The single-block fast path of `Compact` rejects an under-aligned buffer, so the
    /// transmute must stage the read through an aligned scratch buffer. Without that, this read fails.
    #[test]
    fn transmute_into_weaker_alignment_stages_misaligned_buffer() {
        let arr = compact_u32_single_block(&[0x0102_0304, 0x0506_0708, 0x090a_0b0c, 0x0d0e_0f10]);
        let t = unsafe { arr.transmute_elements::<[u8; 4]>() };
        let ctx = t.read_ctx();

        let (mut backing, off) = misaligned_backing(16, 4);
        let dst = &mut backing[off..off + 16];
        t.storage()
            .read_data(&[0..4], &mut OutBuf::new(dst), &ctx)
            .unwrap();

        // Little-endian: u32 0x01020304 -> bytes [04, 03, 02, 01].
        assert_eq!(&dst[0..4], &[0x04, 0x03, 0x02, 0x01]);
        assert_eq!(&dst[4..8], &[0x08, 0x07, 0x06, 0x05]);
        assert_eq!(&dst[8..12], &[0x0c, 0x0b, 0x0a, 0x09]);
        assert_eq!(&dst[12..16], &[0x10, 0x0f, 0x0e, 0x0d]);
    }

    /// Lazy read through a transmute to a *stronger*-aligned dtype (`[u8; 4]` align 1 -> `u32`
    /// align 4). The lazy output buffer is allocated from the pool, which over-aligns to a cache line
    /// (larger than any dtype alignment), so the consumer's typed (aligned) `u32` read lands on a
    /// well-aligned buffer. Checks the bytes reinterpret correctly. (`Plain` storage keeps this
    /// runnable under Miri, which independently verifies the read alignment.)
    #[test]
    fn transmute_into_stronger_alignment_lazy_read() {
        let data: [[u8; 4]; 2] = [[0x04, 0x03, 0x02, 0x01], [0x08, 0x07, 0x06, 0x05]];
        let arr = Array::plain_ndarray(ndarray::arr1(&data)).unwrap();
        let t = unsafe { arr.transmute_elements::<u32>() };
        let ctx = t.read_ctx();

        let mut read = t.storage().read_data_typed::<u32>(&[0..2], &ctx).unwrap();
        assert_eq!(read.len(), 2);
        let vals = read.read_bulk::<2>(0);
        assert_eq!(vals, [0x0102_0304u32, 0x0506_0708]);
    }

    /// Lazy read through a transmute to a *weaker*-aligned dtype (`u32` align 4 -> `[u8; 4]` align 1).
    /// A lazy buffer has no materialized slice, so it must be forwarded to the inner read rather than
    /// routed through the contiguous-buffer staging path (which inspects the buffer's slice).
    #[test]
    fn transmute_into_weaker_alignment_lazy_read() {
        let arr = compact_u32_single_block(&[0x0102_0304, 0x0506_0708]);
        let t = unsafe { arr.transmute_elements::<[u8; 4]>() };
        let ctx = t.read_ctx();

        let mut read = t
            .storage()
            .read_data_typed::<[u8; 4]>(&[0..2], &ctx)
            .unwrap();
        assert_eq!(read.len(), 2);
        let vals = read.read_bulk::<2>(0);
        assert_eq!(vals, [[0x04, 0x03, 0x02, 0x01], [0x08, 0x07, 0x06, 0x05]]);
    }
}
