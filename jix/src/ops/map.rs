use std::ops::Range;

use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, check_dtype_size_nonzero, ensure, Result};
use crate::ops::{Op1, Op2};
use crate::storage::params::{combine_block_layout, combine_elementwise_hints, ArraySpecDynamic};
use crate::storage::{
    check_out_buf, ArrayStorageInfo, ArrayStorageTyped, ElementwisePipeline,
    ElementwisePipelineImpl, Operand, StridedBuf,
};
use crate::{
    array_from_fn_inline, Array, ArraySequence, ArraySequenceDimension, ArraySequenceTyped,
    ArrayStorage, ElementwisePipelineTuple, ReadContext, Ty,
};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Applies `map_fn` to each element, returning an array with dtype `R`. See [`Map`] for
    /// details and examples.
    ///
    /// # Panics
    ///
    /// Panics if the array's dtype does not match `T::DTYPE`.
    #[track_caller]
    pub fn map<R, F>(self, map_fn: F) -> Array<Map<S, F>>
    where
        S: ArrayStorageTyped,
        R: Dtyped,
        F: Fn(S::Item) -> R,
    {
        Map::new_array(self, map_fn).unwrap()
    }
}

/// Applies a function element-wise to an array.
///
/// The output array has the same shape as the input, and dtype determined by the output of
/// `F: Fn(S::Item) -> O`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::map()`](crate::Array::map).
///
/// # Examples
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
/// let result = a.map(|x: i32| x * x).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[1, 4, 9, 16]);
///
/// // Change element type in the mapping function
/// let b = Array::compact_ndarray(&array![0.0f32, 1.5, -2.0])?;
/// let result = b.map(|x: f32| x > 0.0).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Map<S, F>(Op1<S, F>);
impl<S, F> Map<S, F> {
    /// Constructs a [`Map`] storage. See the struct docs for semantics and examples.
    pub fn new<O>(array: S, map_fn: F) -> Result<Self>
    where
        S: ArrayStorageTyped,
        F: Fn(S::Item) -> O,
        O: Dtyped,
    {
        Ok(Self(Op1::new(array, map_fn)?))
    }

    /// Constructs an array with [`Map`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<O>(array: Array<S>, map_fn: F) -> Result<Array<Self>>
    where
        S: ArrayStorageTyped,
        F: Fn(S::Item) -> O,
        O: Dtyped,
    {
        Self::new(array.into_storage(), map_fn).map(Array::from_storage)
    }
}
impl<S, O, F> ArrayStorage for Map<S, F>
where
    S: ArrayStorageTyped,
    O: Dtyped,
    F: Fn(S::Item) -> O,
{
    type ElementType = Ty<O>;
    type Dimension = S::Dimension;
    crate::storage::impl_array_storage_forward!(<S, O, F>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Map", [&self.0])
    }

    type DimensionChange<NewD: crate::Dimension> = Map<S::DimensionChange<NewD>, F>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Map(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
}

/// Applies a binary function element-wise to two arrays.
///
/// The two input arrays must have the same shape. The output array has the same shape, and dtype
/// determined by the output of `F: Fn(S1::Item, S2::Item) -> O`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as [`map2`].
///
/// # Examples
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
/// let b = Array::compact_ndarray(&array![10i32, 20, 30, 40])?;
///
/// let result = jix::ops::map2(a, b, |x, y| x + y).to_ndarray()?;
/// assert_eq!(result, array![11, 22, 33, 44]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Map2<S1, S2, F>(Op2<S1, S2, F>);
impl<S1, S2, F> Map2<S1, S2, F> {
    /// Constructs a [`Map2`] storage. See the struct docs for semantics and examples.
    pub fn new<O>(a: S1, b: S2, map_fn: F) -> Result<Self>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped<Dimension = S1::Dimension>,
        F: Fn(S1::Item, S2::Item) -> O,
        O: Dtyped,
    {
        Ok(Self(Op2::new(a, b, map_fn)?))
    }

    /// Constructs an array with [`Map2`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<O>(a: Array<S1>, b: Array<S2>, map_fn: F) -> Result<Array<Self>>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped<Dimension = S1::Dimension>,
        F: Fn(S1::Item, S2::Item) -> O,
        O: Dtyped,
    {
        Self::new(a.into_storage(), b.into_storage(), map_fn).map(Array::from_storage)
    }
}
impl<S1, S2, O, F> ArrayStorage for Map2<S1, S2, F>
where
    S1: ArrayStorageTyped,
    S2: ArrayStorageTyped<Dimension = S1::Dimension>,
    O: Dtyped,
    F: Fn(S1::Item, S2::Item) -> O,
{
    type ElementType = Ty<O>;
    type Dimension = S1::Dimension;
    crate::storage::impl_array_storage_forward!(<S1, S2, O, F>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Map2", [&self.0.a, &self.0.b])
    }

    type DimensionChange<NewD: crate::Dimension> =
        Map2<S1::DimensionChange<NewD>, S2::DimensionChange<NewD>, F>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Map2(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
}

/// Applies a binary function element-wise to two arrays. See [`Map2`] for details and examples.
#[track_caller]
pub fn map2<S1, S2, O, F>(a: Array<S1>, b: Array<S2>, map_fn: F) -> Array<Map2<S1, S2, F>>
where
    S1: ArrayStorageTyped,
    S2: ArrayStorageTyped<Dimension = S1::Dimension>,
    O: Dtyped,
    F: Fn(S1::Item, S2::Item) -> O,
{
    Map2::new_array(a, b, map_fn).unwrap()
}

/// Applies a function element-wise across a sequence of arrays.
///
/// All input arrays must have the same shape. For each element position the corresponding element
/// of every input array is gathered into an
/// [`ItemSequence`](crate::util::ArraySequenceTyped::ItemSequence) and passed to `map_fn`. The
/// output array has that same shape, and dtype determined by the output of
/// `F: Fn(ArraysT::ItemSequence<'_>) -> O`. This generalizes [`Map`] (one input) and [`Map2`] (two
/// inputs) to an arbitrary number of arrays.
///
/// The shape of the value handed to `map_fn` depends on the [`ArraySequence`](crate::util::ArraySequence)
/// type:
///
/// | Sequence type | `ItemSequence` passed to `map_fn` |
/// |---------------|-----------------------------------|
/// | `[Array<S>; N]` or `&[Array<S>; N]` | `[S::Item; N]` - fixed-length array |
/// | `Vec<Array<S>>` or `&[Array<S>]` | `&[S::Item]` - slice, one entry per array |
/// | `(Array<S0>, Array<S1>, ...)` | `(S0::Item, S1::Item, ...)` - tuple, may be heterogeneous |
///
/// Use a tuple when the inputs have different element types; use a fixed-length array, slice, or
/// `Vec` when they are homogeneous.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`map_multiple`].
///
/// # Examples
///
/// Combine a homogeneous fixed-length array of inputs (the closure receives `[i32; 3]`):
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![10i32, 20, 30])?;
/// let c = Array::compact_ndarray(&array![100i32, 200, 300])?;
///
/// let result = jix::ops::map_multiple([a, b, c], |xs: [i32; 3]| xs.iter().sum::<i32>())
///     .to_ndarray()?;
/// assert_eq!(result, array![111, 222, 333]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Combine a heterogeneous tuple of inputs (the closure receives `(i32, f32, bool)`):
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![0.5f32, 1.5, 2.5])?;
/// let c = Array::compact_ndarray(&array![true, false, true])?;
///
/// let result = jix::ops::map_multiple((a, b, c), |(x, y, flag): (i32, f32, bool)| {
///     if flag { x as f32 + y } else { x as f32 * y }
/// })
///     .to_ndarray()?;
/// assert_eq!(result, array![1.5f32, 3.0, 5.5]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct MapMultiple<ArraysT, F> {
    arrays: ArraysT,
    map_fn: F,
    spec: ArraySpecDynamic,
}
impl<ArraysT, F> MapMultiple<ArraysT, F> {
    /// Constructs a [`MapMultiple`] storage. See the struct docs for semantics and examples.
    pub fn new<O>(arrays: ArraysT, map_fn: F) -> Result<Self>
    where
        ArraysT: ArraySequence + ArraySequenceDimension + ArraySequenceTyped,
        F: Fn(ArraysT::ItemSequence<'_>) -> O,
        O: Dtyped,
    {
        check_dtype_size_nonzero(&O::DTYPE)?;
        let narrays = arrays.narrays();
        if narrays > 1 {
            let shape = arrays.shape(0);
            ensure!(
                (1..narrays).all(|i| arrays.shape(i) == shape),
                InvalidArgument,
                "MapMultiple shape mismatch: {:?}",
                (0..narrays).map(|i| arrays.shape(i)).collect::<Vec<_>>()
            );
        }
        let (element_cost, read_shape_scale_order) = {
            let inputs = (0..narrays)
                .map(|i| {
                    let sp = arrays.spec(i);
                    (sp.element_cost(), sp.read_shape_scale_order().as_slice())
                })
                .collect::<Vec<_>>();
            combine_elementwise_hints(&inputs)
        };
        let (block_shape, block_shape_fixed_dims) = {
            let inputs = (0..narrays)
                .map(|i| {
                    let sp = arrays.spec(i);
                    (sp.block_shape().as_slice(), sp.block_shape_fixed_dims())
                })
                .collect::<Vec<_>>();
            combine_block_layout(&inputs)
        };
        let mut spec = arrays.spec(0).dynamic().clone();
        spec.block_shape = block_shape;
        spec.block_shape_fixed_dims = block_shape_fixed_dims;
        spec.element_cost = element_cost;
        spec.read_shape_scale_order = read_shape_scale_order;
        Ok(Self {
            arrays,
            map_fn,
            spec,
        })
    }
}
impl<ArraysT, O, F> ArrayStorage for MapMultiple<ArraysT, F>
where
    ArraysT: ArraySequence + ArraySequenceDimension + ArraySequenceTyped,
    F: Fn(ArraysT::ItemSequence<'_>) -> O,
    O: Dtyped,
{
    type ElementType = Ty<O>;
    type Dimension = ArraysT::Dimension;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(out.as_deref(), self.shape())?;
        self.read_as_elementwise_pipeline::<O>(index, context)?
            .to_buf(index, context, out)
    }

    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, ArraysT, O, F>>
    where
        T: Dtyped,
    {
        check_dtype(&T::DTYPE, &O::DTYPE)?;
        let inner = self.arrays.read_as_elementwise_pipeline(index, context)?;

        struct MapMultiplePipeline<'a, ArraysT, D, F> {
            inner: D,
            f: &'a F,
            phantom: std::marker::PhantomData<ArraysT>,
        }
        impl<ArraysT, F, O, T, D> ElementwisePipelineImpl<T> for MapMultiplePipeline<'_, ArraysT, D, F>
        where
            ArraysT: ArraySequence + ArraySequenceDimension + ArraySequenceTyped,
            F: Fn(ArraysT::ItemSequence<'_>) -> O,
            O: Dtyped,
            T: Dtyped,
            D: ElementwisePipelineTuple<ArraysT>,
        {
            const N_OPERANDS: Option<usize> = D::N_OPERANDS;

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inner.operands()
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(&self) -> [T; N] {
                // The iterator is consumed here and dropped before the next call, as
                // `read_bulk_as_iter` requires.
                let mut items = unsafe { self.inner.read_bulk_as_iter::<N, CONTIGUOUS>() };
                array_from_fn_inline(|_| {
                    let x = (self.f)(items.next().unwrap());

                    const { assert!(size_of::<O>() == size_of::<T>()) };
                    // SAFETY: the caller checked `T` and `O` are the same dtype.
                    unsafe { std::mem::transmute_copy::<O, T>(&x) }
                })
            }
        }
        impl<ArraysT, F, O, T, D> ElementwisePipeline<T> for MapMultiplePipeline<'_, ArraysT, D, F>
        where
            ArraysT: ArraySequence + ArraySequenceDimension + ArraySequenceTyped,
            F: Fn(ArraysT::ItemSequence<'_>) -> O,
            O: Dtyped,
            T: Dtyped,
            D: ElementwisePipelineTuple<ArraysT>,
        {
        }

        Ok(MapMultiplePipeline {
            inner,
            f: &self.map_fn,
            phantom: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.arrays.shape(0)
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        const { &O::DTYPE }
    }

    #[inline]
    fn spec(&self) -> crate::storage::ArraySpec<'_> {
        self.arrays
            .spec(0)
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        let deps = (0..self.arrays.narrays())
            .map(|i| self.arrays.as_array_storage(i))
            .collect::<Vec<_>>();
        ArrayStorageInfo::new_deps_dyn("MapMultiple", deps)
    }

    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}

/// Applies a function element-wise across a sequence of arrays. See [`MapMultiple`] for details
/// and examples.
///
/// # Panics
///
/// Panics if the input arrays do not all have the same shape.
#[track_caller]
pub fn map_multiple<ArraysT, O, F>(arrays: ArraysT, map_fn: F) -> Array<MapMultiple<ArraysT, F>>
where
    ArraysT: ArraySequence + ArraySequenceDimension + ArraySequenceTyped,
    O: Dtyped,
    F: Fn(ArraysT::ItemSequence<'_>) -> O,
{
    Array::from_storage(MapMultiple::new(arrays, map_fn).unwrap())
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::array::Array;
    use crate::util::arr_params;

    #[test]
    fn map_same_type_1d() {
        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_ndarray_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: i32| x * 2).to_ndarray().unwrap();
        assert_eq!(actual, a.mapv(|x| x * 2));
    }

    #[test]
    fn map_same_type_multi_block() {
        let a = array![1i32, 2, 3, 4, 5, 6];
        let za = Array::compact_ndarray_with(&a, arr_params(&[2])).unwrap();
        let actual = za.map(|x: i32| x + 10).to_ndarray().unwrap();
        assert_eq!(actual, a.mapv(|x| x + 10));
    }

    #[test]
    fn map_type_change_i32_to_f64() {
        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_ndarray_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: i32| x as f64 * 0.5).to_ndarray().unwrap();
        let expected = a.mapv(|x| x as f64 * 0.5);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_type_change_f32_to_bool() {
        let a = array![0.0f32, 1.0, -1.0, 0.0];
        let za = Array::compact_ndarray_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: f32| x != 0.0).to_ndarray().unwrap();
        let expected = a.mapv(|x| x != 0.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_2d_multi_block() {
        let a = ndarray::Array::from_shape_fn((3, 4), |idx| (idx.0 * 4 + idx.1) as i32);
        let za = Array::compact_ndarray_with(&a, arr_params(&[2, 2])).unwrap();
        let actual = za.map(|x: i32| x * x).to_ndarray().unwrap();
        let expected = a.mapv(|x| x * x);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_output_dtype_is_r() {
        let a = array![1i32, 2, 3];
        let za = Array::compact_ndarray_with(&a, arr_params(&[3])).unwrap();
        let mapped = za.map(|x: i32| x as f64);
        use crate::dtype::Dtyped;
        assert_eq!(mapped.dtype(), &f64::DTYPE);
    }

    #[test]
    fn map_integer_to_struct() {
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct Point {
            x: i32,
            y: i32,
        }

        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_ndarray_with(&a, arr_params(&[4])).unwrap();
        let actual = za
            .map(|v: i32| Point { x: v, y: v * 10 })
            .to_ndarray()
            .unwrap();
        let expected = array![
            Point { x: 1, y: 10 },
            Point { x: 2, y: 20 },
            Point { x: 3, y: 30 },
            Point { x: 4, y: 40 },
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_chain_integer_to_struct_to_bigger_struct() {
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct Small {
            x: i32,
            y: i32,
        }

        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct Big {
            x: i32,
            y: i32,
            norm_sq: i64,
        }

        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_ndarray_with(&a, arr_params(&[2])).unwrap();
        let actual = za
            .map(|v: i32| Small { x: v, y: v + 1 })
            .map(|s: Small| Big {
                x: s.x,
                y: s.y,
                norm_sq: (s.x as i64) * (s.x as i64) + (s.y as i64) * (s.y as i64),
            })
            .to_ndarray()
            .unwrap();
        let expected = array![
            Big {
                x: 1,
                y: 2,
                norm_sq: 5
            },
            Big {
                x: 2,
                y: 3,
                norm_sq: 13
            },
            Big {
                x: 3,
                y: 4,
                norm_sq: 25
            },
            Big {
                x: 4,
                y: 5,
                norm_sq: 41
            },
        ];
        assert_eq!(actual, expected);
    }

    // -----------------------------------------------------------------------
    // map_multiple - one test per ArraySequence variant, plus dtype/shape cases
    // -----------------------------------------------------------------------

    #[test]
    fn map_multiple_fixed_len_array() {
        // `[Array<S>; N]` variant: the closure receives `[i32; 3]`.
        let a = Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap();
        let b = Array::compact_ndarray_with(&array![10i32, 20, 30], arr_params(&[3])).unwrap();
        let c = Array::compact_ndarray_with(&array![100i32, 200, 300], arr_params(&[3])).unwrap();
        let actual = crate::ops::map_multiple([a, b, c], |xs: [i32; 3]| xs.iter().sum::<i32>())
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![111, 222, 333]);
    }

    #[test]
    fn map_multiple_ref_fixed_len_array() {
        // `&[Array<S>; N]` variant: still yields `[i32; 2]`, but borrows the inputs.
        let arrays = [
            Array::compact_ndarray_with(&array![1i32, 2, 3, 4], arr_params(&[4])).unwrap(),
            Array::compact_ndarray_with(&array![5i32, 6, 7, 8], arr_params(&[4])).unwrap(),
        ];
        let actual = crate::ops::map_multiple(&arrays, |xs: [i32; 2]| xs[0] * xs[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![5, 12, 21, 32]);
    }

    #[test]
    fn map_multiple_vec() {
        // `Vec<Array<S>>` variant: the closure receives `&[i32]`, one entry per array.
        let arrays = vec![
            Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap(),
            Array::compact_ndarray_with(&array![4i32, 5, 6], arr_params(&[3])).unwrap(),
            Array::compact_ndarray_with(&array![7i32, 8, 9], arr_params(&[3])).unwrap(),
        ];
        let actual = crate::ops::map_multiple(arrays, |xs: &[i32]| xs.iter().sum::<i32>())
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![12, 15, 18]);
    }

    #[test]
    fn map_multiple_slice() {
        // `&[Array<S>]` variant: also yields `&[i32]`.
        let arrays = vec![
            Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap(),
            Array::compact_ndarray_with(&array![4i32, 5, 6], arr_params(&[3])).unwrap(),
        ];
        let actual =
            crate::ops::map_multiple(arrays.as_slice(), |xs: &[i32]| xs.iter().product::<i32>())
                .to_ndarray()
                .unwrap();
        assert_eq!(actual, array![4, 10, 18]);
    }

    #[test]
    fn map_multiple_tuple_homogeneous() {
        // Tuple variant with matching element types - mirrors `map2`.
        let a = Array::compact_ndarray_with(&array![1i32, 2, 3, 4], arr_params(&[2])).unwrap();
        let b = Array::compact_ndarray_with(&array![10i32, 20, 30, 40], arr_params(&[2])).unwrap();
        let actual = crate::ops::map_multiple((a, b), |(x, y): (i32, i32)| x + y)
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![11, 22, 33, 44]);
    }

    #[test]
    fn map_multiple_tuple_heterogeneous() {
        // Tuple variant with three different dtypes - the case a homogeneous sequence cannot express.
        let a = Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap();
        let b = Array::compact_ndarray_with(&array![0.5f32, 1.5, 2.5], arr_params(&[3])).unwrap();
        let c = Array::compact_ndarray_with(&array![true, false, true], arr_params(&[3])).unwrap();
        let actual =
            crate::ops::map_multiple(
                (a, b, c),
                |(x, y, flag): (i32, f32, bool)| {
                    if flag {
                        x as f32 + y
                    } else {
                        x as f32 - y
                    }
                },
            )
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![1.5f32, 0.5, 5.5]);
    }

    #[test]
    fn map_multiple_multi_block() {
        // Force small blocks so the read loop crosses block boundaries.
        let a =
            Array::compact_ndarray_with(&array![1i32, 2, 3, 4, 5, 6], arr_params(&[2])).unwrap();
        let b =
            Array::compact_ndarray_with(&array![6i32, 5, 4, 3, 2, 1], arr_params(&[2])).unwrap();
        let actual = crate::ops::map_multiple([a, b], |xs: [i32; 2]| xs[0] + xs[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![7, 7, 7, 7, 7, 7]);
    }

    #[test]
    fn map_multiple_2d_multi_block() {
        let a = ndarray::Array::from_shape_fn((3, 4), |idx| (idx.0 * 4 + idx.1) as i32);
        let b = ndarray::Array::from_shape_fn((3, 4), |idx| (idx.0 + idx.1) as i32);
        let za = Array::compact_ndarray_with(&a, arr_params(&[2, 2])).unwrap();
        let zb = Array::compact_ndarray_with(&b, arr_params(&[2, 2])).unwrap();
        let actual = crate::ops::map_multiple((za, zb), |(x, y): (i32, i32)| x * 10 + y)
            .to_ndarray()
            .unwrap();
        let expected = ndarray::Array::from_shape_fn((3, 4), |idx| {
            let x = (idx.0 * 4 + idx.1) as i32;
            let y = (idx.0 + idx.1) as i32;
            x * 10 + y
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_multiple_output_dtype_change() {
        // i32 inputs -> f64 output; the result dtype follows the closure's return type.
        let a = Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap();
        let b = Array::compact_ndarray_with(&array![4i32, 5, 6], arr_params(&[3])).unwrap();
        let mapped = crate::ops::map_multiple([a, b], |xs: [i32; 2]| (xs[0] + xs[1]) as f64 / 2.0);
        use crate::dtype::Dtyped;
        assert_eq!(mapped.dtype(), &f64::DTYPE);
        let actual = mapped.to_ndarray().unwrap();
        assert_eq!(actual, array![2.5f64, 3.5, 4.5]);
    }

    #[test]
    fn map_multiple_struct_output() {
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct Point {
            x: i32,
            y: i32,
        }

        let a = Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap();
        let b = Array::compact_ndarray_with(&array![4i32, 5, 6], arr_params(&[3])).unwrap();
        let actual = crate::ops::map_multiple((a, b), |(x, y): (i32, i32)| Point { x, y })
            .to_ndarray()
            .unwrap();
        assert_eq!(
            actual,
            array![
                Point { x: 1, y: 4 },
                Point { x: 2, y: 5 },
                Point { x: 3, y: 6 },
            ]
        );
    }

    #[test]
    fn map_multiple_single_array() {
        // narrays == 1: the shape-equality check is skipped and it behaves like `map`.
        let a = Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap();
        let actual = crate::ops::map_multiple([a], |xs: [i32; 1]| xs[0] * 2)
            .to_ndarray()
            .unwrap();
        assert_eq!(actual, array![2, 4, 6]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn map_multiple_shape_mismatch_panics() {
        let a = Array::compact_ndarray_with(&array![1i32, 2, 3], arr_params(&[3])).unwrap();
        let b = Array::compact_ndarray_with(&array![1i32, 2, 3, 4], arr_params(&[4])).unwrap();
        let _ = crate::ops::map_multiple([a, b], |xs: [i32; 2]| xs[0] + xs[1]);
    }

    proptest::proptest! {
        #[test]
        fn proptest_map_i32(
            (nd, za) in crate::util::carray_strategy_from_shape::<i32>(
                crate::util::shape_strategy(),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
        ) {
            let expected = nd.mapv(|x| x.wrapping_mul(2).wrapping_add(1));
            crate::util::assert_array_matches(
                &za.map(|x: i32| x.wrapping_mul(2).wrapping_add(1)),
                &expected,
            );
        }

        #[test]
        fn proptest_map_i32_to_f64(
            (nd, za) in crate::util::carray_strategy_from_shape::<i32>(
                crate::util::shape_strategy(),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
        ) {
            let expected = nd.mapv(|x| x as f64 * 0.5);
            crate::util::assert_array_matches(&za.map(|x: i32| x as f64 * 0.5), &expected);
        }
    }

    // ---- Zero-itemsize kernel output ----
    //
    // A mapping op is the one place an array's dtype comes from a caller-chosen type rather than
    // from an input array, so the output dtype has to be checked here. Leaf storages are gated by
    // `ArrayParams::tune`, but a lazy view never re-tunes. `[i32; 0]` is a `Dtyped` type with
    // itemsize 0.

    #[test]
    fn map_rejects_zero_itemsize_output() {
        use crate::ops::Map;

        let za = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let err = Map::new_array(za, |_: i32| -> [i32; 0] { [] })
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::UnsupportedDtype);
    }

    #[test]
    fn map2_rejects_zero_itemsize_output() {
        use crate::ops::Map2;

        let a = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let b = Array::compact_ndarray(&array![4i32, 5, 6]).unwrap();
        let err = Map2::new_array(a, b, |_: i32, _: i32| -> [i32; 0] { [] })
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::UnsupportedDtype);
    }

    #[test]
    fn map_multiple_rejects_zero_itemsize_output() {
        use crate::ops::MapMultiple;

        let a = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let b = Array::compact_ndarray(&array![4i32, 5, 6]).unwrap();
        let err = MapMultiple::new((a, b), |_| -> [i32; 0] { [] })
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::UnsupportedDtype);
    }
}
