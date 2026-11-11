use std::sync::LazyLock;

use jix_core::dtype::Dtyped;
use jix_core::scalar::bool8;
use jix_core::scalar::{f16, Complex};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::ops::common::{
    broadcast_operands, define_op2, CastKind, OpDescriptor, OpFnDescriptor, Operand, Scalar,
};
use crate::util::IntoPyResult;
use crate::Array;

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    ///
    /// For **float** types, `NaN == NaN` is `False` (IEEE 754). For **complex** types, both the
    /// real and imaginary components must be equal.
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common type (Safe
    /// casting rules), except that `u64` and signed integer dtypes are compared exactly.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([1, 0, 3], dtype=np.int32)
    ///     result = jix.equal(a, b)
    ///     assert np.array_equal(result.numpy(), [True, False, True])
    ///
    ///     # NaN != NaN per IEEE 754.
    ///     c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    ///     d = jix.compact([float('nan'), 1.0], dtype=np.float32)
    ///     result = jix.equal(c, d)
    ///     assert np.array_equal(result.numpy(), [False, True])
    ///     ```
    equal,
    Equal,
    dispatch = {
        [
            u8, i8, u16, i16, u32, i32, u64, i64, (u64, i64), (i64, u64), f16, f32, f64,
            Complex<f32>, Complex<f64>, bool8
        ],
        Safe
    },
    cast_untyped_float_to_f64 = true
);

define_op2!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    ///
    /// For **float** types, `NaN != NaN` returns `True` per IEEE 754.
    /// For **complex** types, returns `True` if either the real or imaginary component differs.
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common type (Safe
    /// casting rules), except that `u64` and signed integer dtypes are compared exactly.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([1, 0, 3], dtype=np.int32)
    ///     result = jix.not_equal(a, b)
    ///     assert np.array_equal(result.numpy(), [False, True, False])
    ///     ```
    not_equal,
    NotEqual,
    dispatch = {
        [
            u8, i8, u16, i16, u32, i32, u64, i64, (u64, i64), (i64, u64), f16, f32, f64,
            Complex<f32>, Complex<f64>, bool8
        ],
        Safe
    },
    cast_untyped_float_to_f64 = true
);

define_op2!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True > False`.
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common type (Safe
    /// casting rules), except that `u64` and signed integer dtypes are compared exactly.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([3, 1, 2], dtype=np.int32)
    ///     b = jix.compact([1, 1, 3], dtype=np.int32)
    ///     result = jix.greater(a, b)
    ///     assert np.array_equal(result.numpy(), [True, False, False])
    ///     ```
    greater,
    Greater,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, (u64, i64), (i64, u64), f16, f32, f64, bool8],
        Safe
    },
    cast_untyped_float_to_f64 = true
);

define_op2!(
    /// Element-wise greater-than-or-equal test (`a >= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True > False`.
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common type (Safe
    /// casting rules), except that `u64` and signed integer dtypes are compared exactly.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([3, 1, 2], dtype=np.int32)
    ///     b = jix.compact([1, 1, 3], dtype=np.int32)
    ///     result = jix.greater_equal(a, b)
    ///     assert np.array_equal(result.numpy(), [True, True, False])
    ///     ```
    greater_equal,
    GreaterEqual,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, (u64, i64), (i64, u64), f16, f32, f64, bool8],
        Safe
    },
    cast_untyped_float_to_f64 = true
);

define_op2!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False < True`.
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common type (Safe
    /// casting rules), except that `u64` and signed integer dtypes are compared exactly.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 1, 3], dtype=np.int32)
    ///     b = jix.compact([3, 1, 2], dtype=np.int32)
    ///     result = jix.less(a, b)
    ///     assert np.array_equal(result.numpy(), [True, False, False])
    ///     ```
    less,
    Less,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, (u64, i64), (i64, u64), f16, f32, f64, bool8],
        Safe
    },
    cast_untyped_float_to_f64 = true
);

define_op2!(
    /// Element-wise less-than-or-equal test (`a <= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False < True`.
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common type (Safe
    /// casting rules), except that `u64` and signed integer dtypes are compared exactly.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 1, 3], dtype=np.int32)
    ///     b = jix.compact([3, 1, 2], dtype=np.int32)
    ///     result = jix.less_equal(a, b)
    ///     assert np.array_equal(result.numpy(), [True, True, False])
    ///     ```
    less_equal,
    LessEqual,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, (u64, i64), (i64, u64), f16, f32, f64, bool8],
        Safe
    },
    cast_untyped_float_to_f64 = true
);

/// Element-wise approximate equality test, analogous to `numpy.isclose`.
///
/// Returns a `bool` array that is `True` at each position where the corresponding elements
/// of `a` and `b` are within tolerance:
/// `|a - b| <= max(atol, rtol * max(|a|, |b|))`
///
/// This is not NumPy's formula: `numpy.isclose` tests `|a - b| <= atol + rtol * |b|`, which
/// depends on the operand order.
///
/// `NaN` is close to nothing, and an infinity only to the same infinity. For complex arrays,
/// both the real and the imaginary parts must pass the check.
///
/// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`; integers and `bool` are
/// cast to a float first.
///
/// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a common type
/// (Safe casting rules) before comparison.
///
/// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
///
/// Args:
///     a: First operand.
///     b: Second operand.
///     rtol: Relative tolerance, a real Python or NumPy scalar, shared by both parts of a
///         complex value.
///     atol: Absolute tolerance, a real Python or NumPy scalar. For complex arrays it applies
///         to both parts, or may be a complex scalar setting one tolerance per part.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape.
///
/// Raises:
///     TypeError: If `rtol` is complex, or if `atol` is complex for a non-complex array dtype.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.0, 2.0, 3.0], dtype=np.float32)
///     b = jix.compact([1.0, 2.01, 4.0], dtype=np.float32)
///     result = jix.isclose(a, b, rtol=0.0, atol=0.02)
///     assert np.array_equal(result.numpy(), [True, True, False])
///
///     # NaN inputs always produce False.
///     c = jix.compact([float('nan'), 1.0], dtype=np.float32)
///     d = jix.compact([float('nan'), 1.0], dtype=np.float32)
///     result = jix.isclose(c, d, rtol=0.0, atol=0.0)
///     assert np.array_equal(result.numpy(), [False, True])
///
///     # Same infinities are equal; opposite signs are not.
///     e = jix.compact([float('inf'), float('-inf')], dtype=np.float32)
///     f = jix.compact([float('inf'), float('inf')], dtype=np.float32)
///     result = jix.isclose(e, f, rtol=0.0, atol=0.0)
///     assert np.array_equal(result.numpy(), [True, False])
///
///     # Complex array: scalar atol applies to both components; complex atol splits them.
///     g = jix.compact([1.0 + 2.0j], dtype=np.complex64)
///     h = jix.compact([1.01 + 2.1j], dtype=np.complex64)
///     result = jix.isclose(g, h, rtol=0.0, atol=complex(0.02, 0.2))
///     assert np.array_equal(result.numpy(), [True])
///     ```
#[pyo3::pyfunction(signature = (a, b, rtol, atol))]
pub fn isclose<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rtol: &Bound<'py, PyAny>,
    atol: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, Array>> {
    let py = a.py();

    struct ApproxEqualArgs {
        rtol: Scalar,
        atol: Scalar,
    }

    fn cast_tol<T>(scalar: Scalar) -> PyResult<T>
    where
        bool8: jix_core::scalar::Cast<T>,
        u64: jix_core::scalar::Cast<T>,
        i64: jix_core::scalar::Cast<T>,
        f64: jix_core::scalar::Cast<T>,
    {
        Ok(match scalar {
            Scalar::Bool(v) => <bool8 as jix_core::scalar::Cast<T>>::cast(bool8::from(v)),
            Scalar::UInt(v) => <u64 as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::Int(v) => <i64 as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::Float(v) => <f64 as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::Complex(_) => {
                return Err(PyTypeError::new_err(
                    "a complex can't be cast to a real type for rtol or atol",
                ))
            }
        })
    }

    fn float_descriptor<T>() -> OpFnDescriptor<2, ApproxEqualArgs>
    where
        T: Dtyped + jix_core::scalar::ApproxEq<AbsoluteTolerance = T, RelativeTolerance = T>,
        bool8: jix_core::scalar::Cast<T>,
        u64: jix_core::scalar::Cast<T>,
        i64: jix_core::scalar::Cast<T>,
        f64: jix_core::scalar::Cast<T>,
    {
        OpFnDescriptor::new2_args::<T>(CastKind::Safe, |a, b, args: ApproxEqualArgs| {
            let rtol = cast_tol::<T>(args.rtol)?;
            let atol = cast_tol::<T>(args.atol)?;
            let res = jix_core::ops::ApproxEq::new_array(a, b, rtol, atol).into_py_result()?;
            Ok(res.into_any())
        })
    }

    fn complex_descriptor<F>() -> OpFnDescriptor<2, ApproxEqualArgs>
    where
        F: Copy + Send + Sync + 'static,
        Complex<F>: Dtyped
            + jix_core::scalar::ApproxEq<AbsoluteTolerance = Complex<F>, RelativeTolerance = F>,
        bool8: jix_core::scalar::Cast<F>,
        u64: jix_core::scalar::Cast<F>,
        i64: jix_core::scalar::Cast<F>,
        f64: jix_core::scalar::Cast<F>,
    {
        OpFnDescriptor::new2_args::<Complex<F>>(CastKind::Safe, |a, b, args: ApproxEqualArgs| {
            let rtol = cast_tol::<F>(args.rtol)?;
            let atol = match args.atol {
                Scalar::Bool(_) | Scalar::UInt(_) | Scalar::Int(_) | Scalar::Float(_) => {
                    let atol = cast_tol::<F>(args.atol)?;
                    Complex { re: atol, im: atol }
                }
                Scalar::Complex(complex) => Complex {
                    re: <f64 as jix_core::scalar::Cast<F>>::cast(complex.re),
                    im: <f64 as jix_core::scalar::Cast<F>>::cast(complex.im),
                },
            };
            let res = jix_core::ops::ApproxEq::new_array(a, b, rtol, atol).into_py_result()?;
            Ok(res.into_any())
        })
    }

    static DISPATCH_TABLE: LazyLock<OpDescriptor<2, ApproxEqualArgs>> = LazyLock::new(|| {
        OpDescriptor::new(
            "isclose",
            vec![
                float_descriptor::<f16>(),
                float_descriptor::<f32>(),
                float_descriptor::<f64>(),
                complex_descriptor::<f32>(),
                complex_descriptor::<f64>(),
            ],
        )
        .cast_untyped_float_to_f64(true)
    });

    let rtol_scalar = Scalar::from_any(rtol)?;
    let atol_scalar = Scalar::from_any(atol)?;
    let a = Operand::from_any(a)?;
    let b = Operand::from_any(b)?;
    let [a, b] = broadcast_operands([a, b])?;
    let res = DISPATCH_TABLE.dispatch_args(
        [a, b],
        ApproxEqualArgs {
            rtol: rtol_scalar,
            atol: atol_scalar,
        },
    )?;
    Bound::new(py, Array::from_core(res))
}
