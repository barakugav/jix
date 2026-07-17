mod common;

mod as_array;
pub use as_array::*;

mod bitwise;
pub use bitwise::*;

mod cmp;
pub use cmp::*;

mod shape_ops;
use numpy::PyArrayDescr;
pub use shape_ops::*;

mod where_op;
pub use where_op::*;

mod op1;
pub use op1::*;

mod op2;
pub use op2::*;

mod logical1;
pub use logical1::*;

mod reduction;
pub use reduction::*;

mod sub_dtype;
pub use sub_dtype::*;

mod complex;
pub use complex::*;

use jix_core::dtype::{Dtype, ScalarKind};
use jix_core::ArrayAny;
use pyo3::prelude::*;

use crate::array::Array;
use crate::dtype::dtype_from_numpy;

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
/// Casts each element of `array` to a new dtype.
///
/// Supported casts:
/// - Between any two non-complex scalar types: `int8`, `int16`, `int32`, `int64`, `uint8`,
///   `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`, `bool`.
/// - From any non-complex scalar type to a complex type (`complex64`, `complex128`): the value
///   becomes the real part and the imaginary part is zero.
/// - Between the two complex types (`complex64` <-> `complex128`), and from a complex type to
///   `bool`.
///
/// `bool` conversions follow C semantics: zero -> `False`, any non-zero value -> `True`.
/// Casting from a complex type to a real numeric type (int, uint, or float), or any cast
/// involving struct dtypes, is not supported.
///
/// Output dtype is the target dtype. Output shape equals the input shape.
///
/// `dtype` may be a numpy dtype object, a dtype string (e.g. `'float32'`), or a Python
/// type like `np.float32`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Array to cast.
///     dtype: The target dtype. Accepts a numpy dtype object, a dtype string (e.g.
///         `'float32'`), or a Python type like `np.float32`.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the new dtype. No computation occurs until the result
///         is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1, 2, 3, 4], dtype=np.int32)
///     result = jix.astype(a, np.float64)
///     assert np.array_equal(result.numpy(), [1.0, 2.0, 3.0, 4.0])
///
///     # Zero -> False, non-zero -> True
///     b = jix.compact([0, 1, -2, 0], dtype=np.int32)
///     result = jix.astype(b, bool)
///     assert np.array_equal(result.numpy(), [False, True, True, False])
///     ```
pub fn astype<'py>(
    array: &Bound<'py, PyAny>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = crate::ops::asarray_simple(array)?;
    let array = &py_arr.get().arr;
    let np_dtype = &PyArrayDescr::new(dtype.py(), dtype)?;
    let dtype = dtype_from_numpy(np_dtype)?;

    let array = astype_impl(array.clone(), &dtype)?;

    Bound::new(
        py_arr.py(),
        Array::from_core_with_np_dtype(array, np_dtype.clone().unbind()),
    )
}
#[inline(never)]
pub(crate) fn astype_impl(array: ArrayAny, dtype: &Dtype) -> PyResult<ArrayAny> {
    if array.dtype() == dtype {
        return Ok(array); // no-op, same dtype
    }
    if let Some((src, dst)) = array.dtype().try_to_scalar().zip(dtype.try_to_scalar()) {
        use jix_core::scalar::{f16, Complex};

        macro_rules! cast_impl {
            ($src_type:ty, $dst_type:ty) => {{
                let array = array.into_typed::<$src_type>().unwrap();
                let array = array.cast::<$dst_type>();
                return Ok(array.into_any());
            }};
        }
        macro_rules! cast_num {
            ($src_type:ty) => {
                match dst {
                    ScalarKind::I8 => cast_impl!($src_type, i8),
                    ScalarKind::I16 => cast_impl!($src_type, i16),
                    ScalarKind::I32 => cast_impl!($src_type, i32),
                    ScalarKind::I64 => cast_impl!($src_type, i64),
                    ScalarKind::U8 => cast_impl!($src_type, u8),
                    ScalarKind::U16 => cast_impl!($src_type, u16),
                    ScalarKind::U32 => cast_impl!($src_type, u32),
                    ScalarKind::U64 => cast_impl!($src_type, u64),
                    ScalarKind::F16 => cast_impl!($src_type, f16),
                    ScalarKind::F32 => cast_impl!($src_type, f32),
                    ScalarKind::F64 => cast_impl!($src_type, f64),
                    ScalarKind::ComplexF32 => cast_impl!($src_type, Complex<f32>),
                    ScalarKind::ComplexF64 => cast_impl!($src_type, Complex<f64>),
                    ScalarKind::Bool => cast_impl!($src_type, bool),
                }
            };
        }
        macro_rules! cast_complex {
            ($src_type:ty) => {
                match dst {
                    ScalarKind::I8 => {}
                    ScalarKind::I16 => {}
                    ScalarKind::I32 => {}
                    ScalarKind::I64 => {}
                    ScalarKind::U8 => {}
                    ScalarKind::U16 => {}
                    ScalarKind::U32 => {}
                    ScalarKind::U64 => {}
                    ScalarKind::F16 => {}
                    ScalarKind::F32 => {}
                    ScalarKind::F64 => {}
                    ScalarKind::ComplexF32 => cast_impl!($src_type, Complex<f32>),
                    ScalarKind::ComplexF64 => cast_impl!($src_type, Complex<f64>),
                    ScalarKind::Bool => cast_impl!($src_type, bool),
                }
            };
        }
        match src {
            ScalarKind::I8 => cast_num!(i8),
            ScalarKind::I16 => cast_num!(i16),
            ScalarKind::I32 => cast_num!(i32),
            ScalarKind::I64 => cast_num!(i64),
            ScalarKind::U8 => cast_num!(u8),
            ScalarKind::U16 => cast_num!(u16),
            ScalarKind::U32 => cast_num!(u32),
            ScalarKind::U64 => cast_num!(u64),
            ScalarKind::F16 => cast_num!(f16),
            ScalarKind::F32 => cast_num!(f32),
            ScalarKind::F64 => cast_num!(f64),
            ScalarKind::ComplexF32 => cast_complex!(Complex<f32>),
            ScalarKind::ComplexF64 => cast_complex!(Complex<f64>),
            ScalarKind::Bool => cast_num!(bool),
        };
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Unsupported cast from {} to {}",
        array.dtype(),
        dtype
    )))
}
