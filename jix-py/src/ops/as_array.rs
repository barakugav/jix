use jix_core::ArrayAny;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::array::{resolve_array_params, Array};
use crate::dtype::{dtype_from_numpy, numpy_descr_from_any};
use crate::ops::astype_impl;
use crate::ops::common::Operand;

/// Convert any array-like object to a [`jix.Array`][jix.Array].
///
/// This function differ from [`jix.compact()`][jix.compact] in that it does not compress the data -
/// it always produces a view of the input data. In Some cases a copy may be necessary, for example
/// to convert from a raw python list to a typed buffer, but in general this function tries to avoid
/// copying data when possible, and it never compresses the data.
///
/// Accepts Python scalars, lists, tuples, NumPy arrays, and any other object accepted by
/// [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
///
/// Args:
///     value: The array-like to convert. Accepts Python scalars, lists, tuples, NumPy arrays,
///         or any object accepted by `numpy.asarray`.
///     dtype: Optional dtype to cast the array to. Accepts a numpy dtype object, a dtype string
///         (e.g. `'float32'`), or a Python type like `np.float32` - anything `numpy.dtype()`
///         accepts. When omitted, the input dtype is preserved. The cast is a lazy view, so no
///         computation happens until the result is read (see [`jix.astype()`][jix.astype]).
///     params: Block layout and codec configuration. See [`jix.compact()`][jix.compact] for
///         details. The returned array is never compressed, so these mostly affect only arrays
///         later created from it (e.g. via `compact()` or `write_array()`), which inherit them.
///         Only relevant when `value` is not already a jix array; an existing jix array is
///         returned as-is.
///
/// Note:
///     - If `value` is already an `Array` and `dtype` is `None`, it is returned as-is with no copy.
///       When `dtype` is given, the result is a lazy cast view over the input instead.
///     - 0-dimensional inputs produce a scalar array backed by a single value (no buffer).
///     - All other inputs share the underlying buffer with the intermediate NumPy array (zero-copy);
///       the NumPy array is kept alive for as long as the returned `Array` is alive.
///
/// Raises:
///     ValueError: If the input cannot be converted by `numpy.asarray`.
///     ValueError: If the array has more dimensions than jix supports.
///     ValueError: If the array has negative strides (e.g. a reversed slice `a[::-1]`).
///     TypeError: If `dtype` requests an unsupported cast (e.g. from a complex type to an int or
///         float type, or a cast involving struct dtypes).
///
/// Returns:
///     A [`jix.Array`][jix.Array] view of the input. If `value` is already a [`jix.Array`][jix.Array]
///         and no `dtype` is given, it is returned unchanged with no copy.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (value, *, dtype=None, params=None))]
#[inline]
pub fn asarray<'py>(
    value: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, Array>> {
    let py = value.py();
    let params = resolve_array_params(py, params)?;
    let mut py_arr = Operand::from_any_with_params(value, params, false)?.into_py_array(py)?;

    if let Some(dtype) = dtype {
        let np_dtype = &numpy_descr_from_any(py, dtype)?;
        let dtype = dtype_from_numpy(np_dtype)?;
        let array = astype_impl(py_arr.get().arr.clone(), &dtype)?;
        py_arr = Bound::new(
            py,
            Array::from_core_with_np_dtype(array, np_dtype.clone().unbind()),
        )?;
    }

    Ok(py_arr)
}

#[inline]
pub(crate) fn asarray_simple<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    asarray(value, None, None)
}

#[inline]
pub(crate) fn any_to_core_array<'py>(value: &Bound<'py, PyAny>) -> PyResult<ArrayAny> {
    Ok(asarray_simple(value)?.get().to_core())
}

#[cfg(test)]
mod tests {
    use jix_core::dtype::Dtyped;
    use jix_core::scalar::Complex;
    use ndarray::{array, Array0, ArrayD};
    use numpy::PyArray;
    use pyo3::prelude::*;

    use super::asarray;

    /// Call `asarray` and read back the data as an ndarray.
    fn collect<T: Dtyped>(val: &Bound<'_, PyAny>) -> ArrayD<T> {
        asarray(val, None, None)
            .unwrap()
            .get()
            .arr
            .view()
            .into_typed::<T>()
            .unwrap()
            .to_ndarray()
            .unwrap()
    }

    // -- 0-D: Python scalars --------------------------------------------------

    #[test]
    fn test_python_int() {
        Python::attach(|py| {
            let val = 42i64.into_pyobject(py).unwrap();
            let data = collect::<i64>(val.as_any());
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 42);
        });
    }

    #[test]
    fn test_python_float() {
        Python::attach(|py| {
            let val = 2.5f64.into_pyobject(py).unwrap();
            let data = collect::<f64>(val.as_any());
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 2.5);
        });
    }

    #[test]
    fn test_python_bool() {
        Python::attach(|py| {
            let val = true.into_pyobject(py).unwrap();
            let data = collect::<bool>(val.as_any());
            assert_eq!(data.ndim(), 0);
            assert!(*data.first().unwrap());
        });
    }

    // -- 0-D: typed numpy scalars ---------------------------------------------

    fn np0<T>(py: Python<'_>, v: T) -> Bound<'_, PyAny>
    where
        T: numpy::Element + Copy,
    {
        PyArray::<T, _>::from_array(py, &Array0::from_elem((), v)).into_any()
    }

    #[test]
    fn test_0d_i8() {
        Python::attach(|py| {
            let data = collect::<i8>(&np0(py, -5i8));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), -5i8);
        });
    }

    #[test]
    fn test_0d_i32() {
        Python::attach(|py| {
            let data = collect::<i32>(&np0(py, -7i32));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), -7i32);
        });
    }

    #[test]
    fn test_0d_u8() {
        Python::attach(|py| {
            let data = collect::<u8>(&np0(py, 255u8));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 255u8);
        });
    }

    #[test]
    fn test_0d_f32() {
        Python::attach(|py| {
            let data = collect::<f32>(&np0(py, 1.5f32));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 1.5f32);
        });
    }

    #[test]
    fn test_0d_f64() {
        Python::attach(|py| {
            let data = collect::<f64>(&np0(py, -2.5f64));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), -2.5f64);
        });
    }

    #[test]
    fn test_0d_bool() {
        Python::attach(|py| {
            let data = collect::<bool>(&np0(py, false));
            assert_eq!(data.ndim(), 0);
            assert!(!*data.first().unwrap());
        });
    }

    // -- 0-D: complex scalars -------------------------------------------------

    #[test]
    fn test_0d_complex_f32() {
        Python::attach(|py| {
            let val = py
                .eval(
                    cr#"__import__('numpy').array(complex(1, -2), dtype='complex64')"#,
                    None,
                    None,
                )
                .unwrap();
            let data = collect::<Complex<f32>>(&val);
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), Complex::new(1.0f32, -2.0f32));
        });
    }

    #[test]
    fn test_0d_complex_f64() {
        Python::attach(|py| {
            let val = py
                .eval(cr#"__import__('numpy').array(complex(3, 4))"#, None, None)
                .unwrap();
            let data = collect::<Complex<f64>>(&val);
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), Complex::new(3.0f64, 4.0f64));
        });
    }

    // -- multi-dimensional arrays ---------------------------------------------

    fn npd<T, D>(py: Python<'_>, arr: ndarray::Array<T, D>) -> Bound<'_, PyAny>
    where
        T: numpy::Element,
        D: ndarray::Dimension,
    {
        PyArray::<T, D>::from_array(py, &arr).into_any()
    }

    #[test]
    fn test_1d_f32() {
        Python::attach(|py| {
            let orig = array![1.0f32, 2.0, 3.0, 4.0];
            let data = collect::<f32>(&npd(py, orig.clone()));
            assert_eq!(data, orig.into_dyn());
        });
    }

    #[test]
    fn test_1d_i64() {
        Python::attach(|py| {
            let orig = array![10i64, 20, 30];
            let data = collect::<i64>(&npd(py, orig.clone()));
            assert_eq!(data, orig.into_dyn());
        });
    }

    #[test]
    fn test_2d_f64() {
        Python::attach(|py| {
            let orig = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
            let data = collect::<f64>(&npd(py, orig.clone()));
            assert_eq!(data, orig.into_dyn());
        });
    }

    #[test]
    fn test_2d_u8() {
        Python::attach(|py| {
            let orig = ArrayD::from_shape_vec(vec![3, 4], (0u8..12).collect()).unwrap();
            let data = collect::<u8>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
        });
    }

    #[test]
    fn test_3d_i32() {
        Python::attach(|py| {
            let orig = ArrayD::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
            let data = collect::<i32>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
        });
    }

    #[test]
    fn test_shape_preserved() {
        Python::attach(|py| {
            let orig =
                ArrayD::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f32).collect()).unwrap();
            let arr = asarray(&npd(py, orig), None, None).unwrap();
            assert_eq!(arr.get().arr.shape(), &[2u64, 3, 4]);
        });
    }

    // -- passthrough ----------------------------------------------------------

    #[test]
    fn test_passthrough() {
        Python::attach(|py| {
            let orig = array![1.0f32, 2.0];
            let arr1 = asarray(&npd(py, orig), None, None).unwrap();
            let ptr1 = arr1.as_ptr();
            let arr2 = asarray(arr1.as_any(), None, None).unwrap();
            assert_eq!(arr2.as_ptr(), ptr1);
        });
    }

    // -- error cases ----------------------------------------------------------

    #[test]
    fn test_negative_stride_error() {
        Python::attach(|py| {
            let val = py
                .eval(cr#"__import__('numpy').array([1, 2, 3])[::-1]"#, None, None)
                .unwrap();
            assert!(asarray(&val, None, None)
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyOverflowError>(py));
        });
    }
}
