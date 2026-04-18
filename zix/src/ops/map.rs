use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::DimArray;

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Applies `map_fn` element-wise, producing an array with dtype `R`.
    ///
    /// `T` must match the array's element type at runtime; the mapping function
    /// receives each element as `T` and produces an `R`. The output array has
    /// the same shape and block layout as the input.
    ///
    /// # Panics
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

/// Lazy element-wise mapping over an array.
///
/// Stores the source array and a function `F: Fn(T) -> R`. On read, each
/// element is interpreted as `T`, passed through the function, and the result
/// written as `R`. No allocation beyond the read buffer occurs.
///
/// Construct via [`Array::map`]; use [`Map::new`] for fallible construction.
pub struct Map<S, I, O, F> {
    array: Array<S>,

    map_fn: F,
    dtype: Dtype,
    _phantom: std::marker::PhantomData<(I, O)>,

    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<S, I, O, F> Map<S, I, O, F> {
    pub fn new(array: Array<S>, map_fn: F) -> io::Result<Self>
    where
        S: ArrayStorage,
        I: Dtyped,
        O: Dtyped,
        F: Fn(I) -> O,
    {
        let src_dtype = I::DTYPE;
        if src_dtype != *array.dtype() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "map input dtype mismatch: array has {:?} but input generic (I) is {:?}",
                    array.dtype(),
                    src_dtype
                ),
            ));
        }

        Ok(Self {
            map_fn,
            dtype: O::DTYPE,
            _phantom: std::marker::PhantomData,
            shape: array.shape().try_into().unwrap(),
            blocks_layout: array.blocks_layout().clone(),
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
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()> {
        let (src_dtype, dst_dtype) = (self.array.dtype(), O::DTYPE);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);
        let nitems = buf.len() / dst_itemsize;

        let in_place = src_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(src_dtype.alignment() as usize);
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
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage.spec()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{Array, ArrayParams};
    use crate::storage::block::BlockSize;

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    #[test]
    fn map_same_type_1d() {
        let a = ndarray::array![1i32, 2, 3, 4].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[4])).unwrap();
        let actual = za.map(|x: i32| x * 2).data().to_ndarray::<i32>().unwrap();
        assert_eq!(actual, a.mapv(|x| x * 2));
    }

    #[test]
    fn map_same_type_multi_block() {
        let a = ndarray::array![1i32, 2, 3, 4, 5, 6].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[2])).unwrap();
        let actual = za.map(|x: i32| x + 10).data().to_ndarray::<i32>().unwrap();
        assert_eq!(actual, a.mapv(|x| x + 10));
    }

    #[test]
    fn map_type_change_i32_to_f64() {
        let a = ndarray::array![1i32, 2, 3, 4].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[4])).unwrap();
        let actual = za
            .map(|x: i32| x as f64 * 0.5)
            .data()
            .to_ndarray::<f64>()
            .unwrap();
        let expected = a.mapv(|x| x as f64 * 0.5);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_type_change_f32_to_bool() {
        let a = ndarray::array![0.0f32, 1.0, -1.0, 0.0].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[4])).unwrap();
        let actual = za
            .map(|x: f32| x != 0.0)
            .data()
            .to_ndarray::<bool>()
            .unwrap();
        let expected = a.mapv(|x| x != 0.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_2d_multi_block() {
        let a = ndarray::Array::from_shape_fn(ndarray::IxDyn(&[3, 4]), |idx| {
            (idx[0] * 4 + idx[1]) as i32
        });
        let za = Array::from_ndarray(&a, arr_params(&[2, 2])).unwrap();
        let actual = za.map(|x: i32| x * x).data().to_ndarray::<i32>().unwrap();
        let expected = a.mapv(|x| x * x);
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_output_dtype_is_r() {
        let a = ndarray::array![1i32, 2, 3].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[3])).unwrap();
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

        let a = ndarray::array![1i32, 2, 3, 4].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[4])).unwrap();
        let actual = za
            .map(|v: i32| Point { x: v, y: v * 10 })
            .data()
            .to_ndarray::<Point>()
            .unwrap();
        let expected = ndarray::array![
            Point { x: 1, y: 10 },
            Point { x: 2, y: 20 },
            Point { x: 3, y: 30 },
            Point { x: 4, y: 40 },
        ]
        .into_dyn();
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

        let a = ndarray::array![1i32, 2, 3, 4].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[2])).unwrap();
        let actual = za
            .map(|v: i32| Small { x: v, y: v + 1 })
            .map(|s: Small| Big {
                x: s.x,
                y: s.y,
                norm_sq: (s.x as i64) * (s.x as i64) + (s.y as i64) * (s.y as i64),
            })
            .data()
            .to_ndarray::<Big>()
            .unwrap();
        let expected = ndarray::array![
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
        ]
        .into_dyn();
        assert_eq!(actual, expected);
    }

    #[test]
    fn map_wrong_dtype_panics() {
        let a = ndarray::array![1i32, 2, 3].into_dyn();
        let za = Array::from_ndarray(&a, arr_params(&[3])).unwrap();
        // Constructing directly with wrong T should return Err
        let result = super::Map::new(za, |x: f32| x + 1.0);
        assert!(result.is_err());
    }
}
