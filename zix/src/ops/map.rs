use crate::dtype::Dtyped;
use crate::error::Result;
use crate::ops::{Op1, Op2};
use crate::storage::ArrayStorageTyped;
use crate::{Array, ArrayStorage, Ty};

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
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![1i32, 2, 3, 4])?;
/// let result = a.map(|x: i32| x * x).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[1, 4, 9, 16]);
///
/// // Change element type in the mapping function
/// let b = Array::compact_array(&array![0.0f32, 1.5, -2.0])?;
/// let result = b.map(|x: f32| x > 0.0).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
/// # Ok::<(), zix::Error>(())
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
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![1i32, 2, 3, 4])?;
/// let b = Array::compact_array(&array![10i32, 20, 30, 40])?;
///
/// let result = zix::ops::map2(a, b, |x, y| x + y).to_ndarray()?;
/// assert_eq!(result, array![11, 22, 33, 44]);
/// # Ok::<(), zix::Error>(())
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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::array::Array;
    use crate::util::arr_params;

    #[test]
    fn map_same_type_1d() {
        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_array_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: i32| x * 2).to_ndarray().unwrap();
        assert_eq!(actual, a.mapv(|x| x * 2));
    }

    #[test]
    fn map_same_type_multi_block() {
        let a = array![1i32, 2, 3, 4, 5, 6];
        let za = Array::compact_array_with(&a, arr_params(&[2])).unwrap();
        let actual = za.map(|x: i32| x + 10).to_ndarray().unwrap();
        assert_eq!(actual, a.mapv(|x| x + 10));
    }

    #[test]
    fn map_type_change_i32_to_f64() {
        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_array_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: i32| x as f64 * 0.5).to_ndarray().unwrap();
        let expected = a.mapv(|x| x as f64 * 0.5);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_type_change_f32_to_bool() {
        let a = array![0.0f32, 1.0, -1.0, 0.0];
        let za = Array::compact_array_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: f32| x != 0.0).to_ndarray().unwrap();
        let expected = a.mapv(|x| x != 0.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_2d_multi_block() {
        let a = ndarray::Array::from_shape_fn((3, 4), |idx| (idx.0 * 4 + idx.1) as i32);
        let za = Array::compact_array_with(&a, arr_params(&[2, 2])).unwrap();
        let actual = za.map(|x: i32| x * x).to_ndarray().unwrap();
        let expected = a.mapv(|x| x * x);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_output_dtype_is_r() {
        let a = array![1i32, 2, 3];
        let za = Array::compact_array_with(&a, arr_params(&[3])).unwrap();
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
        let za = Array::compact_array_with(&a, arr_params(&[4])).unwrap();
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
        let za = Array::compact_array_with(&a, arr_params(&[2])).unwrap();
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
}
