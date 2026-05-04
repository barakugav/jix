mod common;
use common::{define_op1, define_op2};

mod as_array;
pub use as_array::*;

mod bitwise;
pub use bitwise::*;

mod cmp;
pub use cmp::*;

mod shape_ops;
pub use shape_ops::*;

mod where_op;
pub use where_op::*;

use pyo3::prelude::*;

use crate::array::Array;
use crate::dtype::dtype_from_any;
use crate::util::IntoPyResult;

mod op1;
pub use op1::*;

mod op2;
pub use op2::*;

mod logical1;
pub use logical1::*;

mod reduction;
pub use reduction::*;

pub mod copy_op;
pub use copy_op::*;

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
/// Casts each element of `array` to a new dtype.
///
/// Supported casts:
/// - Between any two scalar types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
///   `uint32`, `uint64`, `float16`, `float32`, `float64`, `bool`.
/// - Between the two complex types: `complex64` ↔ `complex128`.
///
/// `bool` conversions follow C semantics: zero → `False`, any non-zero value → `True`.
/// Casting between complex and non-complex types, or involving struct dtypes, is not
/// supported.
///
/// Output dtype is the target dtype. Output shape equals the input shape.
///
/// `dtype` may be a numpy dtype object, a dtype string (e.g. `'float32'`), or a Python
/// type like `np.float32`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// a = zix.compact([1, 2, 3, 4], dtype=np.int32)
/// result = zix.astype(a, np.float64)
/// assert np.array_equal(result.numpy(), [1.0, 2.0, 3.0, 4.0])
///
/// # Zero → False, non-zero → True
/// b = zix.compact([0, 1, -2, 0], dtype=np.int32)
/// result = zix.astype(b, bool)
/// assert np.array_equal(result.numpy(), [False, True, True, False])
/// ```
pub fn astype<'py>(
    array: &Bound<'py, Array>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = array;
    let array = &py_arr.get().arr;
    let dtype = dtype_from_any(dtype)?;
    if dtype == *array.dtype() {
        return Ok(py_arr.clone()); // no-op, same dtype
    }
    let ret = zix_core::ops::AsType::new(array.clone(), dtype).into_py_result()?;
    Bound::new(py_arr.py(), Array::from_core_storage(ret))
}
