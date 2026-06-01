mod common;

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

mod op1;
pub use op1::*;

mod op2;
pub use op2::*;

mod logical1;
pub use logical1::*;

mod reduction;
pub use reduction::*;

mod copy_op;
pub use copy_op::*;

mod sub_dtype;
pub use sub_dtype::*;

use pyo3::prelude::*;
use zix_core::dtype::{Dtype, DtypeScalarKind};
use zix_core::ArrayAny;

use crate::array::Array;
use crate::dtype::dtype_from_any;

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
/// Casts each element of `array` to a new dtype.
///
/// Supported casts:
/// - Between any two scalar types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
///   `uint32`, `uint64`, `float16`, `float32`, `float64`, `bool`.
/// - Between the two complex types: `complex64` <-> `complex128`.
///
/// `bool` conversions follow C semantics: zero -> `False`, any non-zero value -> `True`.
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
/// # Zero -> False, non-zero -> True
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
    let array = astype_impl(array.clone(), &dtype)?;
    Bound::new(py_arr.py(), Array::from_core(array))
}
#[inline(never)]
pub(crate) fn astype_impl(array: ArrayAny, dtype: &Dtype) -> PyResult<ArrayAny> {
    if array.dtype() == dtype {
        return Ok(array); // no-op, same dtype
    }
    if let Some((src, dst)) = array.dtype().try_to_scalar().zip(dtype.try_to_scalar()) {
        use zix_core::scalar::{f16, Complex};

        macro_rules! cast_impl {
            ($src_type:ty, $dst_type:ty) => {{
                let array = array.to_typed::<$src_type>().unwrap();
                let array = array.cast::<$dst_type>();
                return Ok(array.to_type_dyn().into_any());
            }};
        }
        macro_rules! cast_num {
            ($src_type:ty) => {
                match dst {
                    DtypeScalarKind::I8 => cast_impl!($src_type, i8),
                    DtypeScalarKind::I16 => cast_impl!($src_type, i16),
                    DtypeScalarKind::I32 => cast_impl!($src_type, i32),
                    DtypeScalarKind::I64 => cast_impl!($src_type, i64),
                    DtypeScalarKind::U8 => cast_impl!($src_type, u8),
                    DtypeScalarKind::U16 => cast_impl!($src_type, u16),
                    DtypeScalarKind::U32 => cast_impl!($src_type, u32),
                    DtypeScalarKind::U64 => cast_impl!($src_type, u64),
                    DtypeScalarKind::F16 => cast_impl!($src_type, f16),
                    DtypeScalarKind::F32 => cast_impl!($src_type, f32),
                    DtypeScalarKind::F64 => cast_impl!($src_type, f64),
                    DtypeScalarKind::ComplexF32 => cast_impl!($src_type, Complex<f32>),
                    DtypeScalarKind::ComplexF64 => cast_impl!($src_type, Complex<f64>),
                    DtypeScalarKind::Bool => cast_impl!($src_type, bool),
                }
            };
        }
        macro_rules! cast_complex {
            ($src_type:ty) => {
                match dst {
                    DtypeScalarKind::I8 => {}
                    DtypeScalarKind::I16 => {}
                    DtypeScalarKind::I32 => {}
                    DtypeScalarKind::I64 => {}
                    DtypeScalarKind::U8 => {}
                    DtypeScalarKind::U16 => {}
                    DtypeScalarKind::U32 => {}
                    DtypeScalarKind::U64 => {}
                    DtypeScalarKind::F16 => {}
                    DtypeScalarKind::F32 => {}
                    DtypeScalarKind::F64 => {}
                    DtypeScalarKind::ComplexF32 => cast_impl!($src_type, Complex<f32>),
                    DtypeScalarKind::ComplexF64 => cast_impl!($src_type, Complex<f64>),
                    DtypeScalarKind::Bool => cast_impl!($src_type, bool),
                }
            };
        }
        match src {
            DtypeScalarKind::I8 => cast_num!(i8),
            DtypeScalarKind::I16 => cast_num!(i16),
            DtypeScalarKind::I32 => cast_num!(i32),
            DtypeScalarKind::I64 => cast_num!(i64),
            DtypeScalarKind::U8 => cast_num!(u8),
            DtypeScalarKind::U16 => cast_num!(u16),
            DtypeScalarKind::U32 => cast_num!(u32),
            DtypeScalarKind::U64 => cast_num!(u64),
            DtypeScalarKind::F16 => cast_num!(f16),
            DtypeScalarKind::F32 => cast_num!(f32),
            DtypeScalarKind::F64 => cast_num!(f64),
            DtypeScalarKind::ComplexF32 => cast_complex!(Complex<f32>),
            DtypeScalarKind::ComplexF64 => cast_complex!(Complex<f64>),
            DtypeScalarKind::Bool => cast_num!(bool),
        };
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Unsupported cast from {} to {}",
        array.dtype(),
        dtype
    )))
}
