use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{ensure, Result};
use crate::storage::{ArraySpec, ArrayStorageInfo, StridedBuf};
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

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self.array.read_data(index, context, out)
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
    use crate::codec::ReadContext;
    use crate::dtype::Dtyped;
    use crate::storage::StridedBuf;
    use crate::util::assert_array_matches;
    use crate::{Array, ArrayStorage, Ty};

    /// An 8-byte struct dtype, to check that a transmute handles a composite element type and not
    /// just scalars.
    #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
    #[repr(C)]
    struct Pair {
        x: i32,
        y: i32,
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// The one thing `Transmute::new` rejects: dtypes of different itemsize. The output element has
    /// to occupy exactly the bytes the input element did, since nothing is copied or converted.
    #[test]
    fn new_rejects_a_different_itemsize() {
        let arr = Array::compact_ndarray(&ndarray::array![1u32, 2, 3]).unwrap();
        // u32 is 4 bytes, u16 is 2.
        assert!(unsafe { Transmute::new(arr.into_storage(), Ty::<u16>::new()) }.is_err());
    }

    /// Equal itemsize is the only requirement - the kinds may differ freely, and so may the
    /// alignments (`u32` is 4-aligned, `[u8; 4]` is 1-aligned).
    #[test]
    fn new_accepts_any_dtype_of_the_same_itemsize() {
        let arr = Array::compact_ndarray(&ndarray::array![1u32, 2, 3]).unwrap();
        assert!(unsafe { Transmute::new(arr.as_ref().into_storage(), Ty::<i32>::new()) }.is_ok());
        assert!(unsafe { Transmute::new(arr.as_ref().into_storage(), Ty::<f32>::new()) }.is_ok());
        assert!(
            unsafe { Transmute::new(arr.as_ref().into_storage(), Ty::<[u8; 4]>::new()) }.is_ok()
        );
    }

    // -----------------------------------------------------------------------
    // The bytes are relabelled, not converted
    // -----------------------------------------------------------------------

    /// `f32 -> u32` hands back each element's raw bit pattern rather than its numeric value - the
    /// difference from [`Array::cast`], which would round 0.5 to 0.
    #[test]
    fn float_elements_read_back_as_their_bit_patterns() {
        let src = ndarray::array![1.0f32, -2.0, 0.5, f32::INFINITY, f32::NAN];
        let arr = Array::compact_ndarray(&src).unwrap();
        let bits = unsafe { arr.transmute_elements::<u32>() };

        let expected = src.mapv(f32::to_bits);
        assert_array_matches(&bits, &expected);
    }

    /// Two's-complement bits survive `i32 -> u32`, including the sign bit.
    #[test]
    fn negative_integers_keep_their_two_s_complement_bits() {
        let src = ndarray::array![-1i32, i32::MIN, 0, 7];
        let arr = Array::compact_ndarray(&src).unwrap();
        let bits = unsafe { arr.transmute_elements::<u32>() };

        assert_array_matches(&bits, &ndarray::array![u32::MAX, 1 << 31, 0, 7]);
    }

    /// A struct dtype viewed as an equally-sized byte array exposes its in-memory layout: two
    /// little-endian `i32` fields back to back.
    #[test]
    fn struct_elements_view_as_their_layout_bytes() {
        let src = ndarray::array![Pair { x: 1, y: 2 }, Pair { x: -1, y: 256 }];
        let arr = Array::compact_ndarray(&src).unwrap();
        let bytes = unsafe { arr.transmute_elements::<[u8; 8]>() };

        assert_array_matches(
            &bytes,
            &ndarray::arr1(&[
                [1u8, 0, 0, 0, 2, 0, 0, 0],
                [0xff, 0xff, 0xff, 0xff, 0, 1, 0, 0],
            ]),
        );
    }

    /// Transmuting there and back is the identity - the bytes were never touched, so the second
    /// relabel restores the original element type and values.
    #[test]
    fn transmuting_back_restores_the_original_values() {
        let src = ndarray::array![[1.5f64, -0.25], [1e300, 0.0]];
        let arr = Array::compact_ndarray(&src).unwrap();
        let there_and_back = unsafe {
            arr.transmute_elements::<[u8; 8]>()
                .transmute_elements::<f64>()
        };

        assert_array_matches(&there_and_back, &src);
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    /// Only the element type changes: shape, ndim and element count are the inner array's.
    #[test]
    fn shape_is_untouched_and_dtype_is_the_new_one() {
        let arr = Array::compact_ndarray(&ndarray::Array3::<f32>::zeros((2, 3, 4))).unwrap();
        let t = unsafe { arr.transmute_elements::<u32>() };

        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.nitems(), 24);
        assert_eq!(t.dtype(), &u32::DTYPE);
    }

    /// The `_dyn` variant takes the dtype at runtime and produces a type-erased array;
    /// `into_typed` asserts the element type to get element-wise ops back.
    #[test]
    fn dyn_variant_sets_the_runtime_dtype() {
        let arr = Array::compact_ndarray(&ndarray::array![1i32, 2, 3]).unwrap();
        let t = unsafe { arr.transmute_elements_dyn(u32::DTYPE) };
        assert_eq!(t.dtype(), &u32::DTYPE);

        // Recovering it as the wrong type is refused, as the right one succeeds.
        assert!(t.as_ref().into_typed::<i32>().is_err());
        assert_array_matches(
            &t.into_typed::<u32>().unwrap(),
            &ndarray::array![1u32, 2, 3],
        );
    }

    // -----------------------------------------------------------------------
    // Reads
    // -----------------------------------------------------------------------

    /// A pull read is forwarded straight to the inner storage, so it lends the inner bytes rather
    /// than materializing anything: over a `Plain` array the returned buffer points at the source
    /// ndarray's own allocation.
    #[test]
    fn pull_read_lends_the_inner_bytes() {
        let src = ndarray::array![1.0f32, 2.0, 3.0];
        let arr = Array::plain_ndarray_ref(&src).unwrap();
        let t = unsafe { arr.transmute_elements::<u32>() };
        let ctx = ReadContext::default();

        let view = t.storage().read_data(&[0..3], &ctx, None).unwrap();
        assert_eq!(view.data_ptr(), src.as_ptr().cast::<u8>());
        assert_eq!(view.strides(), &[size_of::<f32>()]);
    }

    /// A push read hands the caller's destination down unchanged, so the inner storage writes at
    /// the caller's own strides - here a column-major layout, not the row-major one a copy would
    /// produce.
    #[test]
    fn push_read_honors_the_destination_strides() {
        let arr =
            Array::compact_ndarray(&ndarray::array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        let t = unsafe { arr.transmute_elements::<u32>() };
        let ctx = t.read_ctx();

        // 4 bytes down a column, 8 bytes across a row, plus one slot of slack to catch a write
        // that spills past the region.
        const SENTINEL: u32 = 0xdead_beef;
        let itemsize = size_of::<u32>();
        let mut dst = [SENTINEL; 7];
        {
            let mut out = unsafe {
                StridedBuf::from_raw_parts_mut(
                    dst.as_mut_ptr().cast::<u8>(),
                    &[2, 3],
                    &[itemsize, 2 * itemsize],
                    itemsize,
                )
            };
            t.storage()
                .read_data(&[0..2, 0..3], &ctx, Some(&mut out))
                .unwrap();
        }
        assert_eq!(
            &dst[..6],
            &[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0].map(f32::to_bits),
            "elements should land column by column"
        );
        assert_eq!(dst[6], SENTINEL, "wrote past the region");
    }

    /// The read index is forwarded unchanged, so a windowed read of the transmuted array covers
    /// exactly the corresponding window of the inner one.
    #[test]
    fn sub_region_reads_map_one_to_one() {
        let src = ndarray::Array2::from_shape_fn((4, 5), |(r, c)| (r * 5 + c) as i32);
        let arr = Array::compact_ndarray(&src).unwrap();
        let t = unsafe { arr.transmute_elements::<u32>() };
        let ctx = t.read_ctx();

        let window = t.to_ndarray_sub(&[1..3, 2..5], &ctx).unwrap();
        assert_eq!(
            window,
            src.slice(ndarray::s![1..3, 2..5]).mapv(|v| v as u32)
        );
    }

    // -----------------------------------------------------------------------
    // Composition
    // -----------------------------------------------------------------------

    /// A transmute is a lazy view like any other op, so it composes on both sides: reading the
    /// exponent-and-sign half of an `f32` here means masking the transmuted bits, and the whole
    /// chain still runs in one pass.
    #[test]
    fn composes_with_element_wise_ops() {
        let src = ndarray::array![1.0f32, -1.0, 2.0, -2.0];
        let arr = Array::compact_ndarray(&src).unwrap();
        let sign_bits = unsafe { arr.transmute_elements::<u32>() }.map(|bits| bits >> 31);

        assert_array_matches(&sign_bits, &ndarray::array![0u32, 1, 0, 1]);
    }

    /// A transmute of a shape view reads through both layers - the slice narrows the region and the
    /// transmute relabels the bytes it yields.
    #[test]
    fn composes_with_shape_ops() {
        let src = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let arr = Array::compact_ndarray(&src).unwrap();
        let column = arr.slice((.., 1..2));
        let bits = unsafe { column.transmute_elements::<u32>() };

        assert_array_matches(&bits, &ndarray::array![[2u32], [5]]);
    }

    /// An empty read is a no-op, not a special case: zero elements out of a non-empty array.
    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn empty_reads_are_fine() {
        let arr = Array::compact_ndarray(&ndarray::array![1u32, 2, 3]).unwrap();
        let t = unsafe { arr.transmute_elements::<f32>() };
        let ctx = t.read_ctx();

        assert_eq!(t.to_ndarray_sub(&[1..1], &ctx).unwrap().len(), 0);
    }
}
