use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_range, ensure, Result};
use crate::storage::params::{combine_block_layout, combine_elementwise_hints, ArraySpecDynamic};
use crate::storage::{
    ArraySpec, ArrayStorageInfo, ArrayStorageTyped, OutBuf, ReadData, ReadDataExt,
};
use crate::util::{cast_slice, cast_slice_mut};
use crate::{Array, ArrayStorage, Dimension};

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
        let (element_cost, dim_scale_weights) = combine_elementwise_hints(&[
            (c_spec.element_cost(), &c_spec.dim_scale_weights()),
            (x_spec.element_cost(), &x_spec.dim_scale_weights()),
            (y_spec.element_cost(), &y_spec.dim_scale_weights()),
        ]);
        let (block_shape, block_shape_fixed_dims) = combine_block_layout(&[
            (&c_spec.block_shape(), c_spec.block_shape_fixed_dims()),
            (&x_spec.block_shape(), x_spec.block_shape_fixed_dims()),
            (&y_spec.block_shape(), y_spec.block_shape_fixed_dims()),
        ]);
        let mut spec = x_spec.dynamic().clone();
        spec.block_shape = block_shape;
        spec.block_shape_fixed_dims = block_shape_fixed_dims;
        spec.element_cost = element_cost;
        spec.dim_scale_weights = dim_scale_weights;
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

        let mut condition_buf = OutBuf::new_lazy(context);
        let mut y_buf = OutBuf::new_lazy(context);
        self.condition
            .read_data(index, &mut condition_buf, context)?;

        let mut cbuf = buf.get_contiguous_mut(out_shape.as_ref(), dtype, context)?;
        let buf = cbuf.as_mut_slice();
        // read x directly into the output buffer
        self.x
            .read_data(index, &mut OutBuf::new(&mut *buf), context)?;
        self.y.read_data(index, &mut y_buf, context)?;

        let condition = unsafe { cast_slice::<_, bool>(condition_buf.as_slice().unwrap()) };
        let y_buf = y_buf.as_slice().unwrap();

        unsafe fn where_impl<T>(condition: &[bool], buf: &mut [u8], y_buf: &[u8])
        where
            T: Copy,
        {
            let x = unsafe { cast_slice_mut::<_, T>(buf) };
            let y = unsafe { cast_slice::<_, T>(y_buf) };
            for (cond, (x, y)) in condition.iter().zip(x.iter_mut().zip(y)) {
                *x = if *cond { *x } else { *y };
            }
        }

        match (dtype.itemsize(), dtype.alignment().as_usize()) {
            (1, 1) => unsafe { where_impl::<u8>(condition, buf, y_buf) },
            (2, 2) => unsafe { where_impl::<u16>(condition, buf, y_buf) },
            (4, 2) => unsafe { where_impl::<[u16; 2]>(condition, buf, y_buf) },
            (4, 4) => unsafe { where_impl::<u32>(condition, buf, y_buf) },
            (8, 4) => unsafe { where_impl::<[u32; 2]>(condition, buf, y_buf) },
            (8, 8) => unsafe { where_impl::<u64>(condition, buf, y_buf) },
            (16, 8 | 16) => unsafe { where_impl::<[u64; 2]>(condition, buf, y_buf) },
            (itemsize, _) => {
                let x = buf.chunks_exact_mut(itemsize as usize);
                let y = y_buf.chunks_exact(itemsize as usize);
                for (cond, (x, y)) in condition.iter().zip(x.zip(y)) {
                    if !cond {
                        x.copy_from_slice(y);
                    }
                }
            }
        };
        cbuf.finalize(out_shape.as_ref(), dtype);
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

    // where_impl's `match (dtype.itemsize(), dtype.alignment())` picks the copy strategy by
    // byte width/alignment, not by the specific dtype (see `read_data` above), so one proptest
    // dtype is kept per distinct branch and same-branch dtypes are covered by a concrete test:
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
