use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec};

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
    pub fn map<T, R, F>(self, map_fn: F) -> Array<Map<S, T, R, F>>
    where
        T: Dtyped,
        R: Dtyped,
        F: Fn(T) -> R,
    {
        Array::from_storage(Map::new(self, map_fn).unwrap())
    }
}

/// Applies a function element-wise to an array.
///
/// `T` must match the array's element dtype at runtime; each element is passed to
/// `F: Fn(T) -> R` and the result written as `R`. The output dtype is `R::DTYPE` and the output
/// shape equals the input shape.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, but the operation is also available as
/// [`Array::map()`](crate::Array::map).
///
/// # Examples
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![1i32, 2, 3, 4])?;
/// let result = a.map(|x: i32| x * x).to_ndarray::<i32>()?;
/// assert_eq!(result.as_slice().unwrap(), &[1, 4, 9, 16]);
///
/// // Change element type in the mapping function
/// let b = Array::compact_array(&array![0.0f32, 1.5, -2.0])?;
/// let result = b.map(|x: f32| x > 0.0).to_ndarray::<bool>()?;
/// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct Map<S, I, O, F> {
    array: Array<S>,

    map_fn: F,
    output_dtype: Dtype,
    _phantom: std::marker::PhantomData<(I, O)>,
}
impl<S, I, O, F> Map<S, I, O, F> {
    /// Constructs a `Map` storage. See [`Map`] for semantics and examples.
    pub fn new(array: Array<S>, map_fn: F) -> Result<Self>
    where
        S: ArrayStorage,
        I: Dtyped,
        O: Dtyped,
        F: Fn(I) -> O,
    {
        let src_dtype = I::DTYPE;
        ensure!(
            src_dtype == *array.dtype(),
            UnsupportedDtype,
            "map input dtype mismatch: array has {:#?} but input generic (I) is {src_dtype:#?}",
            array.dtype()
        );

        Ok(Self {
            map_fn,
            output_dtype: O::DTYPE,
            _phantom: std::marker::PhantomData,
            array,
        })
    }
}
impl<S, I, O, F> ArrayStorage for Map<S, I, O, F>
where
    S: ArrayStorage,
    I: Dtyped,
    O: Dtyped,
    F: Fn(I) -> O,
{
    type Dimension = S::Dimension;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape(), index)?;
        let (src_dtype, dst_dtype) = (self.array.dtype(), O::DTYPE);
        let nitems = check_get_buffer_size(index, &dst_dtype, buf)?;

        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);

        let in_place = src_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(src_dtype.alignment().as_usize());
        let mut tmp_buf;
        let (read_buf, dst) = if in_place {
            let ptr = buf.as_mut_ptr();
            ((ptr, buf.len()), ptr)
        } else {
            tmp_buf = context.tmp_buf(nitems * src_itemsize, src_dtype.alignment());
            let tmp_buf = tmp_buf.as_mut_slice();
            ((tmp_buf.as_mut_ptr(), tmp_buf.len()), buf.as_mut_ptr())
        };
        let read_buf = unsafe { std::slice::from_raw_parts_mut(read_buf.0, read_buf.1) };
        self.array.storage.read_data(index, read_buf, context)?;
        let src = read_buf.as_ptr();

        for i in 0..nitems {
            unsafe {
                let value = src.cast::<I>().add(i).read();
                let value = (self.map_fn)(value);
                dst.cast::<O>().add(i).write(value);
            }
        }
        Ok(())
    }

    fn shape(&self) -> &[u64] {
        self.array.shape()
    }
    fn dtype(&self) -> &Dtype {
        &self.output_dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        self.array.storage._spec()
    }
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
        let actual = za.map(|x: i32| x * 2).to_ndarray::<i32>().unwrap();
        assert_eq!(actual, a.mapv(|x| x * 2).into_dyn());
    }

    #[test]
    fn map_same_type_multi_block() {
        let a = array![1i32, 2, 3, 4, 5, 6];
        let za = Array::compact_array_with(&a, arr_params(&[2])).unwrap();
        let actual = za.map(|x: i32| x + 10).to_ndarray::<i32>().unwrap();
        assert_eq!(actual, a.mapv(|x| x + 10).into_dyn());
    }

    #[test]
    fn map_type_change_i32_to_f64() {
        let a = array![1i32, 2, 3, 4];
        let za = Array::compact_array_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: i32| x as f64 * 0.5).to_ndarray::<f64>().unwrap();
        let expected = a.mapv(|x| x as f64 * 0.5);
        assert_eq!(actual, expected.into_dyn());
    }

    #[test]
    fn map_type_change_f32_to_bool() {
        let a = array![0.0f32, 1.0, -1.0, 0.0];
        let za = Array::compact_array_with(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: f32| x != 0.0).to_ndarray::<bool>().unwrap();
        let expected = a.mapv(|x| x != 0.0);
        assert_eq!(actual, expected.into_dyn());
    }

    #[test]
    fn map_2d_multi_block() {
        let a = ndarray::Array::from_shape_fn((3, 4), |idx| (idx.0 * 4 + idx.1) as i32);
        let za = Array::compact_array_with(&a, arr_params(&[2, 2])).unwrap();
        let actual = za.map(|x: i32| x * x).to_ndarray::<i32>().unwrap();
        let expected = a.mapv(|x| x * x);
        assert_eq!(actual, expected.into_dyn());
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
            .to_ndarray::<Point>()
            .unwrap();
        let expected = array![
            Point { x: 1, y: 10 },
            Point { x: 2, y: 20 },
            Point { x: 3, y: 30 },
            Point { x: 4, y: 40 },
        ];
        assert_eq!(actual, expected.into_dyn());
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
            .to_ndarray::<Big>()
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
        assert_eq!(actual, expected.into_dyn());
    }

    #[test]
    fn map_wrong_dtype_panics() {
        let a = array![1i32, 2, 3];
        let za = Array::compact_array_with(&a, arr_params(&[3])).unwrap();
        // Constructing directly with wrong T should return Err
        let result = super::Map::new(za, |x: f32| x + 1.0);
        assert!(result.is_err());
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
