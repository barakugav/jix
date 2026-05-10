use pyo3::prelude::*;
use zix_core::dtype::DtypeScalarKind;

use crate::ops::common::Precision;
use crate::ops::{define_op2, promote, Operand, Scalar};
use crate::util::IntoPyResult;
use crate::Array;

define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Both arrays must have
    /// the same dtype. Output dtype and shape equal the input.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types each component is added independently:
    /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
    ///
    /// Available via the `+` operator on arrays.
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape. The scalar must have the
    /// same dtype as the array (use `np.int32(5)`, not bare `5`, for `int32` arrays).
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 2, 3], dtype=np.int32)
    /// b = zix.compact([10, 20, 30], dtype=np.int32)
    /// result = zix.add(a, b)  # same as `a + b`
    /// assert np.array_equal(result.numpy(), [11, 22, 33])
    ///
    /// # A typed scalar can be used as either operand.
    /// result2 = zix.add(a, np.int32(10))
    /// assert np.array_equal(result2.numpy(), [11, 12, 13])
    /// ```
    add,
    Add
);
define_op2!(
    /// Element-wise subtraction of two arrays (`a - b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Both arrays must have
    /// the same dtype. Output dtype and shape equal the input.
    ///
    /// For **integer** types the result wraps on underflow (two's complement).
    /// For **complex** types each component is subtracted independently:
    /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
    ///
    /// Available via the `-` operator on arrays.
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape. The scalar must have the
    /// same dtype as the array (use `np.int32(5)`, not bare `5`, for `int32` arrays).
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([10, 20, 30], dtype=np.int32)
    /// b = zix.compact([1, 2, 3], dtype=np.int32)
    /// result = zix.subtract(a, b)  # same as `a - b`
    /// assert np.array_equal(result.numpy(), [9, 18, 27])
    ///
    /// # A typed scalar can be used as either operand.
    /// result2 = zix.subtract(a, np.int32(1))
    /// assert np.array_equal(result2.numpy(), [9, 19, 29])
    /// ```
    subtract,
    Sub
);
define_op2!(
    /// Element-wise multiplication of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Both arrays must have
    /// the same dtype. Output dtype and shape equal the input.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types this is full complex multiplication:
    /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
    ///
    /// Available via the `*` operator on arrays.
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape. The scalar must have the
    /// same dtype as the array (use `np.int32(5)`, not bare `5`, for `int32` arrays).
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 2, 3], dtype=np.int32)
    /// b = zix.compact([4, 5, 6], dtype=np.int32)
    /// result = zix.multiply(a, b)  # same as `a * b`
    /// assert np.array_equal(result.numpy(), [4, 10, 18])
    ///
    /// # A typed scalar can be used as either operand.
    /// result2 = zix.multiply(a, np.int32(3))
    /// assert np.array_equal(result2.numpy(), [3, 6, 9])
    /// ```
    multiply,
    Mul
);
define_op2!(
    /// Element-wise division of two arrays (`a / b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Both arrays must have
    /// the same dtype. Output dtype and shape equal the input.
    ///
    /// For **integer** types the result is truncating (rounds towards zero); dividing by
    /// zero raises an error.
    /// For **float** types, division by zero produces `+/-inf` or `NaN`.
    /// For **complex** types this is full complex division.
    ///
    /// Available via the `/` operator on arrays.
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape. The scalar must have the
    /// same dtype as the array (use `np.int32(5)`, not bare `5`, for `int32` arrays).
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    /// - for integer types, division truncates towards zero (same as numpy)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([10, 20, 30], dtype=np.int32)
    /// b = zix.compact([2, 4, 5], dtype=np.int32)
    /// result = zix.divide(a, b)  # same as `a / b`
    /// assert np.array_equal(result.numpy(), [5, 5, 6])
    ///
    /// # A typed scalar can be used as either operand.
    /// result2 = zix.divide(a, np.int32(2))
    /// assert np.array_equal(result2.numpy(), [5, 10, 15])
    /// ```
    divide,
    Div
);
define_op2!(
    /// Element-wise exponentiation (`a` raised to the power `b`).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Negative base with a non-integer exponent produces `NaN`.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape. The scalar must have the
    /// same dtype as the array (use `np.float32(2.0)`, not bare `2.0`, for `float32` arrays).
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    /// - only `f32` and `f64` are supported (numpy supports integer power as well)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([2.0, 3.0, 4.0], dtype=np.float32)
    /// b = zix.compact([3.0, 2.0, 0.5], dtype=np.float32)
    /// result = zix.power(a, b)
    /// assert np.array_equal(result.numpy(), [8.0, 9.0, 2.0])
    ///
    /// # Raise each element to a fixed exponent using a typed scalar.
    /// result2 = zix.power(a, np.float32(2.0))
    /// assert np.array_equal(result2.numpy(), [4.0, 9.0, 16.0])
    /// ```
    power,
    Power
);

pub(crate) fn asarray22<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, Array>, Bound<'py, Array>)> {
    let py = a.py();
    let mut a = Operand::from_any(a)?;
    let mut b = Operand::from_any(b)?;

    // returns None for scalars
    fn extract_shape<'a>(asarray: &'a Operand) -> Option<&'a [u64]> {
        match asarray {
            Operand::Zix(array) => Some(array.get().arr.shape()),
            Operand::Numpy(array) => Some(array.shape()),
            Operand::Scalar { .. } => None,
        }
    }
    let shape = [&a, &b]
        .into_iter()
        .map(extract_shape)
        .filter_map(|s| s.map(|s| s.to_vec()))
        .next();

    let dtype = promote(&[&a, &b]);

    fn operand_cast_if_scalar(
        value: &mut Operand,
        target_dtype: DtypeScalarKind,
    ) -> Result<(), zix_core::Error> {
        let Operand::Scalar {
            value, precision, ..
        } = value
        else {
            return Ok(());
        };
        macro_rules! do_cast {
            ($value:expr) => {
                match target_dtype {
                    DtypeScalarKind::I8
                    | DtypeScalarKind::I16
                    | DtypeScalarKind::I32
                    | DtypeScalarKind::I64 => Scalar::Int(zix_core::ops::__private::cast($value)),
                    DtypeScalarKind::U8
                    | DtypeScalarKind::U16
                    | DtypeScalarKind::U32
                    | DtypeScalarKind::U64 => Scalar::UInt(zix_core::ops::__private::cast($value)),
                    DtypeScalarKind::F16 | DtypeScalarKind::F32 | DtypeScalarKind::F64 => {
                        Scalar::Float(zix_core::ops::__private::cast($value))
                    }
                    DtypeScalarKind::ComplexF32 | DtypeScalarKind::ComplexF64 => {
                        Scalar::Complex(zix_core::ops::__private::cast($value))
                    }
                    DtypeScalarKind::Bool => Scalar::Bool(zix_core::ops::__private::cast($value)),
                }
            };
        }
        *value = match value {
            Scalar::Bool(value) => do_cast!(*value),
            Scalar::UInt(value) => do_cast!(*value),
            Scalar::Int(value) => do_cast!(*value),
            Scalar::Float(value) => do_cast!(*value),
            Scalar::Complex(value) => match target_dtype {
                DtypeScalarKind::ComplexF32 | DtypeScalarKind::ComplexF64 => {
                    Scalar::Complex(zix_core::ops::__private::cast(*value))
                }
                _ => unreachable!(),
            },
        };
        *precision = Some(match target_dtype {
            DtypeScalarKind::I8 => Precision::P1,
            DtypeScalarKind::I16 => Precision::P2,
            DtypeScalarKind::I32 => Precision::P4,
            DtypeScalarKind::I64 => Precision::P8,
            DtypeScalarKind::U8 => Precision::P1,
            DtypeScalarKind::U16 => Precision::P2,
            DtypeScalarKind::U32 => Precision::P4,
            DtypeScalarKind::U64 => Precision::P8,
            DtypeScalarKind::F16 => Precision::P2,
            DtypeScalarKind::F32 => Precision::P4,
            DtypeScalarKind::F64 => Precision::P8,
            DtypeScalarKind::ComplexF32 => Precision::P4,
            DtypeScalarKind::ComplexF64 => Precision::P8,
            DtypeScalarKind::Bool => Precision::P1,
        });
        Ok(())
    }

    fn asarray_broadcast_if_scalar(
        value: &mut Operand,
        broadcast: &[u64],
    ) -> Result<(), zix_core::Error> {
        if let Operand::Scalar { shape, .. } = value {
            *shape = broadcast.to_vec();
        }
        Ok(())
    }

    if let Some(dtype) = dtype {
        operand_cast_if_scalar(&mut a, dtype).into_py_result()?;
        operand_cast_if_scalar(&mut b, dtype).into_py_result()?;
    }

    if let Some(shape) = shape {
        asarray_broadcast_if_scalar(&mut a, &shape).into_py_result()?;
        asarray_broadcast_if_scalar(&mut b, &shape).into_py_result()?;
    }

    let a = a.into_py_array(py)?;
    let b = b.into_py_array(py)?;
    Ok((a, b))
}
