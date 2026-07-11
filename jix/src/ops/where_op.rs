use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_range, ensure, Result};
use crate::storage::{
    ArraySpec, ArrayStorageInfo, ArrayStorageTyped, OutBuf, ReadData, ReadDataExt,
};
use crate::util::{cast_slice, cast_slice_mut};
use crate::{Array, ArrayStorage};

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

        Ok(Self { condition, x, y })
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
        let dtype = self.dtype();

        let mut condition_buf = OutBuf::new_lazy(context);
        let mut y_buf = OutBuf::new_lazy(context);
        self.condition
            .read_data(index, &mut condition_buf, context)?;

        let mut buf = buf.get_contiguous_mut(index, dtype, context);
        buf.edit(|buf| {
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
            Ok(())
        })
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
        self.x.spec().with_cleared_flags()
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

    // Covers where_impl branches: (1,1)=u8, (2,2)=u16, (4,4)=u32, (8,8)=u64
    test_where_dtype!(i8);
    test_where_dtype!(u8);
    test_where_dtype!(bool);
    test_where_dtype!(i16);
    test_where_dtype!(u16);
    test_where_dtype!(i32);
    test_where_dtype!(u32);
    test_where_dtype!(f32);
    test_where_dtype!(i64);
    test_where_dtype!(u64);
    test_where_dtype!(f64);

    #[cfg(feature = "half")]
    test_where_dtype!(f16);

    #[cfg(feature = "num-complex")]
    test_where_dtype!(complex_f32); // (8, 4) branch
    #[cfg(feature = "num-complex")]
    test_where_dtype!(complex_f64); // (16, 8) branch

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
