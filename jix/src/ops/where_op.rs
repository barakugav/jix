use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_range, ensure, Result};
use crate::storage::params::{combine_block_layout, combine_elementwise_hints, ArraySpecDynamic};
use crate::storage::{
    ArraySpec, ArrayStorageInfo, ArrayStorageTyped, OutBuf, ReadData, ReadDataExt,
};
use crate::util::cast_slice;
use crate::{default_strides, Array, ArrayStorage, Dimension, NdIterUnordered};

/// Element-wise selection from `x` or `y` based on `condition`. See [`Where`] for details and
/// examples.
///
/// # Panics
///
/// Panics if `condition` is not `bool`, `x` and `y` differ in dtype, or any two arrays differ
/// in shape.
#[track_caller]
pub fn where_condition<SC, SX, SY>(
    condition: Array<SC>,
    x: Array<SX>,
    y: Array<SY>,
) -> Array<Where<SC, SX, SY>>
where
    SC: ArrayStorageTyped<Item = bool>,
    SX: ArrayStorage<Dimension = SC::Dimension>,
    SY: ArrayStorage<ElementType = SX::ElementType, Dimension = SC::Dimension>,
{
    Where::new_array(condition, x, y).unwrap()
}

/// Selects elements element-wise from `x` or `y` depending on `condition`
///
/// For each index `i`, the output is `x[i]` if `condition[i]` is `true`, otherwise `y[i]`.
/// Semantics match `numpy.where(condition, x, y)`.
///
/// `condition` must have dtype `bool`. `x` and `y` must have the same dtype. All three arrays
/// must have the same shape. Output dtype equals the dtype of `x` and `y`. Output shape equals
/// the input shape.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`where_condition()`].
///
/// # Examples
/// ```
/// use jix::ops::where_condition;
/// use jix::Array;
/// use ndarray::array;
///
/// let cond = Array::compact_ndarray(&array![true, false, true, false])?;
/// let x = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
/// let y = Array::compact_ndarray(&array![10i32, 20, 30, 40])?;
/// let result = where_condition(cond, x, y).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[1, 20, 3, 40]);
///
/// // 2-D arrays
/// let cond = Array::compact_ndarray(&array![[true, false], [false, true]])?;
/// let x = Array::compact_ndarray(&array![[1.0f64, 2.0], [3.0, 4.0]])?;
/// let y = Array::compact_ndarray(&array![[10.0f64, 20.0], [30.0, 40.0]])?;
/// let result = where_condition(cond, x, y).to_ndarray()?;
/// assert_eq!(result[[0, 0]], 1.0);
/// assert_eq!(result[[0, 1]], 20.0);
/// assert_eq!(result[[1, 0]], 30.0);
/// assert_eq!(result[[1, 1]], 4.0);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Where<SC, SX, SY> {
    condition: SC,
    x: SX,
    y: SY,
    spec: ArraySpecDynamic,
}
impl<SC, SX, SY> Where<SC, SX, SY>
where
    SC: ArrayStorageTyped<Item = bool>,
    SX: ArrayStorage<Dimension = SC::Dimension>,
    SY: ArrayStorage<ElementType = SX::ElementType, Dimension = SC::Dimension>,
{
    /// Constructs a [`Where`] storage. See the struct docs for semantics and examples.
    pub fn new(condition: SC, x: SX, y: SY) -> Result<Self> {
        ensure!(
            condition.dtype() == &bool::DTYPE,
            UnsupportedDtype,
            "where condition must have boolean dtype, got {}",
            condition.dtype()
        );
        ensure!(
            x.dtype() == y.dtype(),
            UnsupportedDtype,
            "x and y arrays must have the same dtype, got {} and {}",
            x.dtype(),
            y.dtype()
        );
        let shape = condition.shape();
        ensure!(
            x.shape() == shape && y.shape() == shape,
            InvalidArgument,
            "condition, x, and y arrays must have the same shape, got {:?}, {:?}, and {:?}",
            shape,
            x.shape(),
            y.shape()
        );

        let c_spec = condition.spec();
        let x_spec = x.spec();
        let y_spec = y.spec();
        let (element_cost, read_shape_scale_order) = combine_elementwise_hints(&[
            (c_spec.element_cost(), c_spec.read_shape_scale_order()),
            (x_spec.element_cost(), x_spec.read_shape_scale_order()),
            (y_spec.element_cost(), y_spec.read_shape_scale_order()),
        ]);
        let (block_shape, block_shape_fixed_dims) = combine_block_layout(&[
            (c_spec.block_shape(), c_spec.block_shape_fixed_dims()),
            (x_spec.block_shape(), x_spec.block_shape_fixed_dims()),
            (y_spec.block_shape(), y_spec.block_shape_fixed_dims()),
        ]);
        let mut spec = x_spec.dynamic().clone();
        spec.block_shape = block_shape;
        spec.block_shape_fixed_dims = block_shape_fixed_dims;
        spec.element_cost = element_cost;
        spec.read_shape_scale_order = read_shape_scale_order;
        Ok(Self {
            condition,
            x,
            y,
            spec,
        })
    }

    /// Constructs an array with [`Where`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(condition: Array<SC>, x: Array<SX>, y: Array<SY>) -> Result<Array<Self>> {
        Self::new(condition.into_storage(), x.into_storage(), y.into_storage())
            .map(Array::from_storage)
    }
}
impl<SC, SX, SY> ArrayStorage for Where<SC, SX, SY>
where
    SC: ArrayStorageTyped<Item = bool>,
    SX: ArrayStorage<Dimension = SC::Dimension>,
    SY: ArrayStorage<ElementType = SX::ElementType, Dimension = SC::Dimension>,
{
    type ElementType = SX::ElementType;
    type Dimension = SC::Dimension;

    #[inline]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        check_get_range(self.shape(), index)?;
        let out_shape = <Self::Dimension as Dimension>::vec(index.len(), |d| {
            (index[d].end - index[d].start) as usize
        });
        let dtype = self.dtype();
        let (itemsize, alignment) = (dtype.itemsize() as usize, dtype.alignment().as_usize());

        let mut condition_buf = OutBuf::new_lazy(context);
        let mut y_buf = OutBuf::new_lazy(context);
        self.condition
            .read_data(index, &mut condition_buf, context)?;

        let (out_buf, out_strides) = buf.get_strided_mut::<Self::Dimension>(index, dtype);
        // read x directly into the output buffer
        {
            let mut x_buf = unsafe { OutBuf::new_strided(&mut *out_buf, out_strides.as_ref()) };
            self.x.read_data(index, &mut x_buf, context)?;
        }
        self.y.read_data(index, &mut y_buf, context)?;

        let condition = unsafe { cast_slice::<_, bool>(condition_buf.as_slice().unwrap()) };
        let y_buf = y_buf.as_slice().unwrap();

        // Operand 0 is the output buffer, `x`, operand 1 the condition mask and operand 2 `y`.
        // The latter two were just read into contiguous buffers, so they have default strides.
        let condition_strides = default_strides(&out_shape, size_of::<bool>());
        let y_strides = default_strides(&out_shape, itemsize);
        let iter = NdIterUnordered::new(
            out_shape.as_ref(),
            [
                out_strides.as_ref(),
                condition_strides.as_ref(),
                y_strides.as_ref(),
            ],
            [(itemsize, alignment), (1, 1), (itemsize, alignment)],
        );
        let [x_aligned, _cond_aligned, y_aligned] = iter.is_aligned();
        let aligned = x_aligned
            && y_aligned
            && (out_buf.as_ptr() as usize).is_multiple_of(alignment)
            && (y_buf.as_ptr() as usize).is_multiple_of(alignment);
        let contiguous = iter.is_contiguous().iter().all(|&c| c);

        type InnerLoopFn = unsafe fn(&mut [u8], &[bool], &[u8], usize, [usize; 3], usize);
        let mut inner_loop_fn: InnerLoopFn = inner_loop_generic;
        if aligned {
            fn create_inner_loop_fn<T: Copy, const LANES: usize>(contiguous: bool) -> InnerLoopFn {
                match contiguous {
                    true => inner_loop::<T, LANES, true>,
                    false => inner_loop::<T, LANES, false>,
                }
            }
            let typed_inner_loop_fn = match (itemsize, alignment) {
                (1, 1) => Some(create_inner_loop_fn::<u8, 16>(contiguous)),
                (2, 2) => Some(create_inner_loop_fn::<u16, 16>(contiguous)),
                (4, 2) => Some(create_inner_loop_fn::<[u16; 2], 16>(contiguous)),
                (4, 4) => Some(create_inner_loop_fn::<u32, 16>(contiguous)),
                (8, 4) => Some(create_inner_loop_fn::<[u32; 2], 8>(contiguous)),
                (8, 8) => Some(create_inner_loop_fn::<u64, 8>(contiguous)),
                (16, 8 | 16) => Some(create_inner_loop_fn::<[u64; 2], 4>(contiguous)),
                _ => None,
            };
            if let Some(typed_inner_loop_fn) = typed_inner_loop_fn {
                inner_loop_fn = typed_inner_loop_fn;
            }
        }

        iter.foreach_inner_1d(|[out_offset, cond_offset, y_offset], len, strides| unsafe {
            inner_loop_fn(
                out_buf.get_unchecked_mut(out_offset..),
                condition.get_unchecked(cond_offset..),
                y_buf.get_unchecked(y_offset..),
                len,
                strides,
                itemsize,
            )
        });

        #[inline(never)]
        unsafe fn inner_loop<T: Copy, const LANES: usize, const CONTIGUOUS: bool>(
            x: &mut [u8],
            condition: &[bool],
            y: &[u8],
            len: usize,
            strides: [usize; 3],
            itemsize: usize,
        ) {
            let [x_stride, cond_stride, y_stride] = strides;
            if CONTIGUOUS {
                debug_assert_eq!(x_stride, size_of::<T>());
                debug_assert_eq!(cond_stride, size_of::<bool>());
                debug_assert_eq!(y_stride, size_of::<T>());
            }
            assert_eq!(itemsize, size_of::<T>());
            let x = x.as_mut_ptr().cast::<T>();
            let condition = condition.as_ptr();
            let y = y.as_ptr().cast::<T>();
            let mut i = 0;
            unsafe {
                if CONTIGUOUS {
                    while i + LANES <= len {
                        let x_chunk_ptr = x.add(i).cast::<[T; LANES]>();
                        let x_chunk = x_chunk_ptr.read();
                        let y_chunk = y.add(i).cast::<[T; LANES]>().read();
                        let cond_chunk = condition.add(i).cast::<[bool; LANES]>().read();
                        let mut out = x_chunk;
                        for k in 0..LANES {
                            out[k] = if cond_chunk[k] {
                                x_chunk[k]
                            } else {
                                y_chunk[k]
                            };
                        }
                        x_chunk_ptr.write(out);
                        i += LANES;
                    }
                }
                while i < len {
                    let (x, cond, y) = if CONTIGUOUS {
                        (x.add(i), condition.add(i), y.add(i))
                    } else {
                        (
                            x.byte_add(i * x_stride),
                            condition.add(i * cond_stride),
                            y.byte_add(i * y_stride),
                        )
                    };
                    let x_val = x.read();
                    let y_val = y.read();
                    let val = if cond.read() { x_val } else { y_val };
                    x.write(val);
                    i += 1;
                }
            }
        }

        #[inline(never)]
        unsafe fn inner_loop_generic(
            x: &mut [u8],
            condition: &[bool],
            y: &[u8],
            len: usize,
            strides: [usize; 3],
            itemsize: usize,
        ) {
            let [x_stride, cond_stride, y_stride] = strides;
            let x = x.as_mut_ptr();
            let condition = condition.as_ptr();
            let y = y.as_ptr();
            unsafe {
                for i in 0..len {
                    let cond = condition.add(i * cond_stride).read();
                    if !cond {
                        let x = x.add(i * x_stride);
                        let y = y.add(i * y_stride);
                        x.copy_from_nonoverlapping(y, itemsize);
                    }
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, SC, SX, SY>>
    where
        T: Dtyped,
    {
        let condition = self.condition.read_data_typed::<bool>(index, context)?;
        let x = self.x.read_data_typed::<T>(index, context)?;
        let y = self.y.read_data_typed::<T>(index, context)?;

        Ok(condition
            .zip_items(x.zip_items(y))
            .map_items(|(cond, (x, y))| if cond { x } else { y }))
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.condition.shape()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.x.dtype()
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.x
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Where", [&self.condition, &self.x, &self.y])
    }

    type DimensionChange<NewD: crate::Dimension> =
        Where<SC::DimensionChange<NewD>, SX::DimensionChange<NewD>, SY::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Where {
            condition: self.condition.dimension_change()?,
            x: self.x.dimension_change()?,
            y: self.y.dimension_change()?,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> =
        Where<SC, SX::ElementTypeChange<NewET>, SY::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> Result<Self::ElementTypeChange<NewET>>
    where
        Self: Sized,
    {
        Ok(Where {
            condition: self.condition,
            x: self.x.element_type_change()?,
            y: self.y.element_type_change()?,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use ndarray::array;
    use proptest::prelude::*;

    use super::{where_condition, Where};
    use crate::array::Array;
    use crate::dtype::Dtyped;
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::storage::Compact;
    use crate::util::ScalarStrategy;
    use crate::{DimDyn, Ty};
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    #[allow(clippy::type_complexity)]
    fn strategy2<T>() -> impl Strategy<
        Value = (
            ndarray::ArrayD<bool>,
            ndarray::ArrayD<T>,
            ndarray::ArrayD<T>,
            Array<Compact<Ty<bool>, DimDyn>>,
            Array<Compact<Ty<T>, DimDyn>>,
            Array<Compact<Ty<T>, DimDyn>>,
        ),
    >
    where
        T: ScalarStrategy + Debug,
    {
        crate::util::shape_strategy()
            .prop_flat_map(|shape| {
                (
                    crate::util::carray_strategy_from_shape::<bool>(
                        Just(shape.clone()),
                        <bool as ScalarStrategy>::any_strategy(),
                    ),
                    crate::util::carray_strategy_from_shape::<T>(
                        Just(shape.clone()),
                        <T as ScalarStrategy>::any_strategy(),
                    ),
                    crate::util::carray_strategy_from_shape::<T>(
                        Just(shape),
                        <T as ScalarStrategy>::any_strategy(),
                    ),
                )
            })
            .prop_map(|((nd_cond, za_cond), (nd_x, za_x), (nd_y, za_y))| {
                (nd_cond, nd_x, nd_y, za_cond, za_x, za_y)
            })
    }

    /// Compresses `cond`/`x`/`y` (with `block_shape`, if given, applied to all three), runs
    /// `where_condition` on them, and asserts the result matches the elementwise `if c { x }
    /// else { y }` reference. Shared by the concrete per-byte-width tests below, which only
    /// differ in the fixed input arrays.
    fn check_where_concrete<T, InD>(
        cond: &ndarray::Array<bool, InD>,
        x: &ndarray::Array<T, InD>,
        y: &ndarray::Array<T, InD>,
        block_shape: Option<&[usize]>,
    ) where
        T: Dtyped + Copy + Debug + PartialEq,
        InD: ndarray::Dimension + crate::IntoDimension,
    {
        let za_cond = match block_shape {
            Some(bs) => Array::compact_ndarray_with(cond, crate::util::arr_params(bs)).unwrap(),
            None => Array::compact_ndarray(cond).unwrap(),
        };
        let za_x = match block_shape {
            Some(bs) => Array::compact_ndarray_with(x, crate::util::arr_params(bs)).unwrap(),
            None => Array::compact_ndarray(x).unwrap(),
        };
        let za_y = match block_shape {
            Some(bs) => Array::compact_ndarray_with(y, crate::util::arr_params(bs)).unwrap(),
            None => Array::compact_ndarray(y).unwrap(),
        };
        let expected = ndarray::Zip::from(cond)
            .and(x)
            .and(y)
            .map_collect(|&c, &x, &y| if c { x } else { y });
        crate::util::assert_array_matches(&where_condition(za_cond, za_x, za_y), &expected);
    }

    // Proptest macro: one test per dtype covering random shapes, random block shapes,
    // full reads, and random sub-range reads via assert_array_matches.
    // The condition, x, and y arrays all share the same random shape.
    macro_rules! test_where_dtype {
        ($dtype:ty) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<where_op_ $dtype>](
                        (nd_cond, nd_x, nd_y, za_cond, za_x, za_y) in strategy2::<$dtype>()
                    ) {
                        let expected_vals: Vec<$dtype> = nd_cond
                            .iter()
                            .zip(nd_x.iter().zip(nd_y.iter()))
                            .map(|(&c, (&xi, &yi))| if c { xi } else { yi })
                            .collect();
                        let expected = ndarray::ArrayD::from_shape_vec(
                            nd_cond.shape().to_vec(),
                            expected_vals,
                        )
                        .unwrap();
                        let result = where_condition(za_cond, za_x, za_y);
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };
    }

    // `read_data`'s `match (itemsize, alignment)` picks the inner loop by byte width/alignment,
    // not by the specific dtype (see `read_data` above), so one proptest dtype is kept per
    // distinct branch and same-branch dtypes are covered by a concrete test:
    //   (1,1)  u8-width  -> kept `i8`  proptest; u8/bool     -> where_1byte_concrete
    //   (2,2)  u16-width -> kept `i16` proptest; u16/f16     -> where_2byte_concrete
    //   (4,4)  u32-width -> kept `i32` proptest; u32/f32     -> where_4byte_concrete
    //   (8,8)  u64-width -> kept `i64` proptest; u64/f64     -> where_8byte_concrete
    //   (8,4)  mixed     -> where_complex_f32_concrete (Complex<f32>: 8 bytes, 4-byte align)
    //   (16,8) mixed     -> where_complex_f64_concrete (Complex<f64>: 16 bytes, 8-byte align)
    test_where_dtype!(i8);
    test_where_dtype!(i16);
    test_where_dtype!(i32);
    test_where_dtype!(i64);

    // --- concrete tests for the remaining dtypes ---
    //
    // Each dtype here shares a byte-branch with one of the proptests kept above (or, for
    // the two complex dtypes, hits its own mixed-alignment branch), so a fixed input is enough
    // to keep the branch covered. Every condition mask below runs an all-true segment, then an
    // all-false segment, then an alternating (mixed) tail, so a single array exercises all three
    // selection patterns.

    #[test]
    fn where_1byte_concrete() {
        // (1,1) branch, shared with the kept `i8` proptest: u8 and bool.
        let cond = array![true, true, true, false, false, false, true, false];

        let x_u8 = array![0u8, 1, 100, u8::MAX, 0xAA, 5, 6, 7];
        let y_u8 = array![u8::MAX, 0, 0xAA, 1, 100, 50, 60, 70];
        check_where_concrete(&cond, &x_u8, &y_u8, None);

        let x_bool = array![true, true, false, false, true, false, true, false];
        let y_bool = array![false, false, true, true, false, true, false, true];
        check_where_concrete(&cond, &x_bool, &y_bool, None);
    }

    #[test]
    fn where_2byte_concrete() {
        // (2,2) branch, shared with the kept `i16` proptest: u16 and (feature-gated) f16.
        let cond = array![true, true, true, false, false, false, true, false];

        let x_u16 = array![0u16, 1, 100, u16::MAX, 0xAAAA, 5, 6, 7];
        let y_u16 = array![u16::MAX, 0, 0xAAAA, 1, 100, 50, 60, 70];
        check_where_concrete(&cond, &x_u16, &y_u16, None);

        #[cfg(feature = "half")]
        {
            let x_f16 = array![
                f16::from_f32(0.0),
                f16::from_f32(-1.5),
                f16::from_f32(1.5),
                f16::MAX,
                f16::MIN,
                f16::from_f32(2.0),
                f16::from_f32(-2.0),
                f16::from_f32(3.0),
            ];
            let y_f16 = array![
                f16::MIN,
                f16::MAX,
                f16::from_f32(0.0),
                f16::from_f32(-1.5),
                f16::from_f32(1.5),
                f16::from_f32(30.0),
                f16::from_f32(40.0),
                f16::from_f32(50.0),
            ];
            check_where_concrete(&cond, &x_f16, &y_f16, None);
        }
    }

    #[test]
    fn where_4byte_concrete() {
        // (4,4) branch, shared with the kept `i32` proptest: u32 and f32. Also exercises a
        // non-default 2-D block shape to cross block boundaries.
        let cond = ndarray::array![[true, true, false], [false, true, false]];

        let x_u32 = ndarray::array![[0u32, 1, u32::MAX], [0xAAAAAAAAu32, 100, 200]];
        let y_u32 = ndarray::array![[u32::MAX, 0, 1], [100, 0xAAAAAAAAu32, 300]];
        check_where_concrete(&cond, &x_u32, &y_u32, Some(&[1, 2]));

        let x_f32 = ndarray::array![
            [0.0f32, -1.5, f32::MAX],
            [f32::MIN, f32::INFINITY, f32::NEG_INFINITY]
        ];
        let y_f32 = ndarray::array![
            [f32::MIN, f32::MAX, 0.0],
            [f32::NEG_INFINITY, 1.5, f32::INFINITY]
        ];
        check_where_concrete(&cond, &x_f32, &y_f32, None);
    }

    #[test]
    fn where_8byte_concrete() {
        // (8,8) branch, shared with the kept `i64` proptest: u64 and f64.
        let cond = array![true, true, true, false, false, false, true, false];

        let x_u64 = array![0u64, 1, 100, u64::MAX, 0xAAAAAAAAAAAAAAAA, 5, 6, 7];
        let y_u64 = array![u64::MAX, 0, 0xAAAAAAAAAAAAAAAA, 1, 100, 50, 60, 70];
        check_where_concrete(&cond, &x_u64, &y_u64, None);

        let x_f64 = array![
            0.0f64,
            -1.5,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.5,
            2.5
        ];
        let y_f64 = array![
            f64::MIN,
            f64::MAX,
            0.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            1.5,
            2.5,
            3.5
        ];
        check_where_concrete(&cond, &x_f64, &y_f64, None);
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn where_complex_f32_concrete() {
        // (8,4) mixed branch: Complex<f32> is 8 bytes wide but only 4-byte aligned, distinct
        // from the (8,8) branch that i64/u64/f64 hit above.
        let cond = array![true, true, false, false, true, false];
        let x = array![
            complex_f32 { re: 0.0, im: 0.0 },
            complex_f32 { re: -1.5, im: 2.5 },
            complex_f32 {
                re: f32::MAX,
                im: f32::MIN
            },
            complex_f32 { re: 1.0, im: -1.0 },
            complex_f32 {
                re: 100.0,
                im: -100.0
            },
            complex_f32 { re: 3.0, im: 4.0 },
        ];
        let y = array![
            complex_f32 {
                re: f32::MIN,
                im: f32::MAX
            },
            complex_f32 { re: 1.0, im: -1.0 },
            complex_f32 { re: 0.0, im: 0.0 },
            complex_f32 { re: -1.5, im: 2.5 },
            complex_f32 { re: 5.0, im: 6.0 },
            complex_f32 { re: 7.0, im: 8.0 },
        ];
        check_where_concrete(&cond, &x, &y, None);
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn where_complex_f64_concrete() {
        // (16,8) mixed branch: Complex<f64> is 16 bytes wide but only 8-byte aligned.
        let cond = array![true, true, false, false, true, false];
        let x = array![
            complex_f64 { re: 0.0, im: 0.0 },
            complex_f64 { re: -1.5, im: 2.5 },
            complex_f64 {
                re: f64::MAX,
                im: f64::MIN
            },
            complex_f64 { re: 1.0, im: -1.0 },
            complex_f64 {
                re: 100.0,
                im: -100.0
            },
            complex_f64 { re: 3.0, im: 4.0 },
        ];
        let y = array![
            complex_f64 {
                re: f64::MIN,
                im: f64::MAX
            },
            complex_f64 { re: 1.0, im: -1.0 },
            complex_f64 { re: 0.0, im: 0.0 },
            complex_f64 { re: -1.5, im: 2.5 },
            complex_f64 { re: 5.0, im: 6.0 },
            complex_f64 { re: 7.0, im: 8.0 },
        ];
        check_where_concrete(&cond, &x, &y, None);
    }

    // --- strided output buffer ---

    /// Read a `where_condition` view into a *strided* destination and assert every logical output
    /// element lands in its own slot with the bytes between slots left untouched - i.e. `read_data`
    /// selects straight into the caller's byte strides instead of staging through a contiguous
    /// scratch and scattering at the end.
    ///
    /// `inner_gap` is how many element slots each consecutive innermost element advances (1 is a
    /// row-major-contiguous destination; 2 leaves one untouched gap element between slots), and
    /// `base_offset` shifts the destination base that many bytes off an 8-aligned address - so a
    /// non-multiple of the dtype's alignment drives the unaligned inner loop.
    #[track_caller]
    fn check_where_strided<T, InD>(
        cond: &ndarray::Array<bool, InD>,
        x: &ndarray::Array<T, InD>,
        y: &ndarray::Array<T, InD>,
        inner_gap: usize,
        base_offset: usize,
    ) where
        T: Dtyped + Copy + Debug + PartialEq,
        InD: ndarray::Dimension + crate::IntoDimension,
    {
        use crate::storage::OutBuf;
        use crate::ArrayStorage;

        const SENTINEL: u8 = 0xA5;
        let itemsize = size_of::<T>();
        let out_shape = cond.shape().to_vec();
        let ndim = out_shape.len();
        let index: Vec<std::ops::Range<u64>> = out_shape.iter().map(|&s| 0..s as u64).collect();

        // A block shape of 2 per axis makes every read span several blocks.
        let params = || crate::util::arr_params(&vec![2usize; ndim]);
        let view = where_condition(
            Array::compact_ndarray_with(cond, params()).unwrap(),
            Array::compact_ndarray_with(x, params()).unwrap(),
            Array::compact_ndarray_with(y, params()).unwrap(),
        );
        let ctx = view.read_ctx();
        let storage = view.into_storage();

        // Destination byte strides: `inner_gap` slots on the innermost axis, propagated outward so
        // the region is gap-free apart from that inner spacing.
        let mut byte_strides = vec![itemsize; ndim];
        if ndim > 0 {
            byte_strides[ndim - 1] = inner_gap * itemsize;
            for d in (0..ndim - 1).rev() {
                byte_strides[d] = byte_strides[d + 1] * out_shape[d + 1];
            }
        }
        let span = (0..ndim)
            .map(|d| out_shape[d].saturating_sub(1) * byte_strides[d])
            .sum::<usize>()
            + itemsize;

        // `Vec<u64>` backing so the base is 8-aligned and `base_offset` alone decides alignment.
        let mut backing = vec![0u64; (base_offset + span).div_ceil(8) + 1];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), backing.len() * 8)
        };
        bytes.fill(SENTINEL);
        {
            let dst = &mut bytes[base_offset..];
            let mut out = unsafe { OutBuf::new_strided(dst, &byte_strides) };
            storage.read_data(&index, &mut out, &ctx).unwrap();
        }

        // Every logical element must sit in its slot, and every byte outside a slot must still hold
        // the sentinel.
        let mut is_slot_byte = vec![false; bytes.len()];
        let mut coord = vec![0usize; ndim];
        for (flat, ((&c, &xi), &yi)) in cond.iter().zip(x.iter()).zip(y.iter()).enumerate() {
            let mut rem = flat;
            for d in (0..ndim).rev() {
                coord[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            let off = base_offset + (0..ndim).map(|d| coord[d] * byte_strides[d]).sum::<usize>();
            let got = unsafe { bytes.as_ptr().add(off).cast::<T>().read_unaligned() };
            assert_eq!(got, if c { xi } else { yi }, "coord {coord:?}");
            is_slot_byte[off..off + itemsize].fill(true);
        }
        for (i, &b) in bytes.iter().enumerate() {
            if !is_slot_byte[i] {
                assert_eq!(b, SENTINEL, "gap byte {i} was overwritten");
            }
        }
    }

    /// `cond`/`x`/`y` for a `[3, 4]` array, with `x`/`y` built from `f` so one body covers every
    /// element width. The mask runs true/false segments plus an alternating tail.
    #[allow(clippy::type_complexity)]
    fn strided_inputs<T: Clone>(
        f: impl Fn(i64) -> T,
    ) -> (
        ndarray::Array2<bool>,
        ndarray::Array2<T>,
        ndarray::Array2<T>,
    ) {
        let cond = ndarray::Array2::from_shape_fn((3, 4), |(r, c)| match r {
            0 => true,
            1 => false,
            _ => c % 2 == 0,
        });
        let x = ndarray::Array2::from_shape_fn((3, 4), |(r, c)| f((r * 4 + c) as i64));
        let y = ndarray::Array2::from_shape_fn((3, 4), |(r, c)| f(-1 - (r * 4 + c) as i64));
        (cond, x, y)
    }

    // The inner loop is picked by (element width, alignment) and then specialized on
    // (all-operands-contiguous, aligned), so each width is checked in all three interesting
    // destination layouts: contiguous+aligned, strided (gap) and contiguous-but-unaligned.
    macro_rules! test_where_strided_dtype {
        ($dtype:ty, $f:expr) => {
            paste::paste! {
                #[test]
                fn [<where_strided_output_ $dtype>]() {
                    let (cond, x, y) = strided_inputs($f);
                    check_where_strided(&cond, &x, &y, 1, 0); // contiguous, aligned
                    check_where_strided(&cond, &x, &y, 2, 0); // gap between slots
                    check_where_strided(&cond, &x, &y, 3, 0);
                    if size_of::<$dtype>() > 1 {
                        // Base one byte off an 8-aligned address: the strides stay
                        // itemsize-multiples but no `$dtype` access is aligned.
                        check_where_strided(&cond, &x, &y, 1, 1);
                        check_where_strided(&cond, &x, &y, 2, 1);
                    }
                }
            }
        };
    }

    test_where_strided_dtype!(i8, |v| v as i8);
    test_where_strided_dtype!(i16, |v| v as i16);
    test_where_strided_dtype!(i32, |v| v as i32);
    test_where_strided_dtype!(i64, |v| v);
    #[cfg(feature = "num-complex")]
    test_where_strided_dtype!(complex_f32, |v| complex_f32 {
        re: v as f32,
        im: -(v as f32)
    });
    #[cfg(feature = "num-complex")]
    test_where_strided_dtype!(complex_f64, |v| complex_f64 {
        re: v as f64,
        im: -(v as f64)
    });

    #[test]
    fn where_strided_output_4byte_2aligned() {
        // The (4, 2) width branch: 4 bytes wide but only 2-byte aligned, so it is served by
        // `[u16; 2]` rather than `u32`. No scalar dtype lands here.
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct U16Pair {
            a: u16,
            b: u16,
        }
        assert_eq!((size_of::<U16Pair>(), align_of::<U16Pair>()), (4, 2));

        let (cond, x, y) = strided_inputs(|v| U16Pair {
            a: v as u16,
            b: (v * 7) as u16,
        });
        check_where_strided(&cond, &x, &y, 1, 0);
        check_where_strided(&cond, &x, &y, 2, 0);
        check_where_strided(&cond, &x, &y, 1, 1);
        check_where_strided(&cond, &x, &y, 2, 1);
    }

    #[test]
    fn where_strided_output_struct_dtype() {
        // A 12-byte / 4-aligned dtype matches none of the scalar-width branches, so this drives the
        // byte-wise `inner_loop_generic` fallback into a strided destination.
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct Triple {
            a: i32,
            b: i32,
            c: i32,
        }
        assert_eq!(size_of::<Triple>(), 12);

        let (cond, x, y) = strided_inputs(|v| Triple {
            a: v as i32,
            b: (v * 3) as i32,
            c: (v * 5) as i32,
        });
        check_where_strided(&cond, &x, &y, 1, 0);
        check_where_strided(&cond, &x, &y, 2, 0);
        check_where_strided(&cond, &x, &y, 1, 1);
    }

    #[test]
    fn where_strided_output_3d() {
        let cond = ndarray::Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i + j + k) % 3 != 0);
        let x = ndarray::Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let y =
            ndarray::Array3::from_shape_fn((2, 3, 4), |(i, j, k)| -1 - (i * 12 + j * 4 + k) as i32);
        check_where_strided(&cond, &x, &y, 1, 0);
        check_where_strided(&cond, &x, &y, 2, 0);
        check_where_strided(&cond, &x, &y, 2, 1);
    }

    // --- error cases ---

    #[test]
    fn x_y_dtype_mismatch_fails() {
        let cond = Array::compact_ndarray(&array![true, false, true]).unwrap();
        let x = Array::compact_ndarray(&array![1i32, 2, 3])
            .unwrap()
            .into_type_dyn();
        let y = Array::compact_ndarray(&array![4.0f64, 5.0, 6.0])
            .unwrap()
            .into_type_dyn();
        assert!(Where::new_array(cond, x, y).is_err());
    }

    #[test]
    fn shape_mismatch_condition_vs_x_fails() {
        let cond = Array::compact_ndarray(&array![true, false]).unwrap();
        let x = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let y = Array::compact_ndarray(&array![4i32, 5, 6]).unwrap();
        assert!(Where::new_array(cond, x, y).is_err());
    }

    #[test]
    fn shape_mismatch_x_vs_y_fails() {
        let cond = Array::compact_ndarray(&array![true, false, true]).unwrap();
        let x = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let y = Array::compact_ndarray(&array![4i32, 5]).unwrap();
        assert!(Where::new_array(cond, x, y).is_err());
    }

    // --- edge cases ---

    #[test]
    fn all_true_selects_x() {
        let cond = Array::compact_ndarray(&array![true, true, true, true]).unwrap();
        let x = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
        let y = Array::compact_ndarray(&array![10i32, 20, 30, 40]).unwrap();
        let result = where_condition(cond, x, y).to_ndarray().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn all_false_selects_y() {
        let cond = Array::compact_ndarray(&array![false, false, false, false]).unwrap();
        let x = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
        let y = Array::compact_ndarray(&array![10i32, 20, 30, 40]).unwrap();
        let result = where_condition(cond, x, y).to_ndarray().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10, 20, 30, 40]);
    }

    // Exercises the fallback byte-copy path in where_impl for struct dtypes.
    #[test]
    fn struct_dtype_fallback_path() {
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[repr(C)]
        struct Pair {
            a: i32,
            b: i32,
        }

        let cond = Array::compact_ndarray(&array![true, false, true]).unwrap();
        let x = Array::compact_ndarray(&array![
            Pair { a: 1, b: 2 },
            Pair { a: 3, b: 4 },
            Pair { a: 5, b: 6 }
        ])
        .unwrap();
        let y = Array::compact_ndarray(&array![
            Pair { a: 10, b: 20 },
            Pair { a: 30, b: 40 },
            Pair { a: 50, b: 60 }
        ])
        .unwrap();
        let result = where_condition(cond, x, y).to_ndarray().unwrap();
        assert_eq!(result[0], Pair { a: 1, b: 2 }); // cond=true  -> x
        assert_eq!(result[1], Pair { a: 30, b: 40 }); // cond=false -> y
        assert_eq!(result[2], Pair { a: 5, b: 6 }); // cond=true  -> x
    }
}
