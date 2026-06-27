use std::sync::LazyLock;

use jix_core::dtype::Dtyped;
use jix_core::scalar::{f16, Complex};
use pyo3::prelude::*;

use crate::ops::common::{define_op1, CastKind, OpDescriptor, OpFnDescriptor, Operand};
use crate::util::IntoPyResult;

define_op1!(
    /// Arithmetic negation applied element-wise (`-array`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `f16`, `f32`, `f64`,
    /// `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result is the two's-complement negation. Negating the
    /// minimum representable value (e.g. `int8` `-128`) overflows and wraps.
    /// For **complex** types both components are negated independently.
    ///
    /// Available via the unary `-` operator on arrays: `-arr`.
    ///
    /// Args:
    ///     array: Input array. Unsigned integer inputs are automatically cast to the next
    ///         larger signed integer type before negation (Safe casting rules): `u8 -> i16`,
    ///         `u16 -> i32`, `u32 -> i64`. This differs from numpy, which overflow for unsigned
    ///         negation.
    ///         A `bool` input is cast to `i8` (False -> 0, True -> -1).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.0, -2.5, 3.0], dtype=np.float32)
    ///     result = jix.negative(a)
    ///     assert np.array_equal(result.numpy(), [-1.0, 2.5, -3.0])
    ///
    ///     # Unsigned integers are auto-cast to the next larger signed type.
    ///     b = jix.compact([1, 2, 3], dtype=np.uint8)
    ///     result = jix.negative(b)
    ///     assert result.dtype == np.int16
    ///     assert np.array_equal(result.numpy(), [-1, -2, -3])
    ///     ```
    negative,
    Neg,
    dispatch = {
        [i8, i16, i32, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
);

define_op1!(
    /// Rounds each element down to the nearest integer (towards -inf).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.1, 2.9, -1.1, -2.9], dtype=np.float32)
    ///     result = jix.floor(a)
    ///     assert np.array_equal(result.numpy(), [1.0, 2.0, -2.0, -3.0])
    ///     ```
    floor,
    Floor,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +inf).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.1, 2.0, -1.7, -0.1], dtype=np.float32)
    ///     result = jix.ceil(a)
    ///     assert np.array_equal(result.numpy(), [2.0, 2.0, -1.0, 0.0])
    ///     ```
    ceil,
    Ceil,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Ties (values exactly halfway between two integers) are broken by rounding away from
    /// zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from "round-half-to-even"
    /// (banker's rounding) used by Python's built-in `round()` and `numpy.round_`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.4, 1.6, 2.5, -0.5], dtype=np.float32)
    ///     result = jix.round(a)
    ///     assert np.array_equal(result.numpy(), [1.0, 2.0, 3.0, -1.0])
    ///     ```
    round,
    Round,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
define_op1!(
    /// Computes the square root of each element.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Negative inputs produce `NaN`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([4.0, 9.0, 16.0], dtype=np.float32)
    ///     result = jix.sqrt(a)
    ///     assert np.array_equal(result.numpy(), [2.0, 3.0, 4.0])
    ///
    ///     # Negative input produces NaN.
    ///     b = jix.compact([-1.0], dtype=np.float32)
    ///     assert np.isnan(jix.sqrt(b).numpy()[0])
    ///     ```
    sqrt,
    Sqrt,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Squares each element (`x * x`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`,
    /// `c32`, `c64`. The output dtype always matches the input dtype.
    ///
    /// For **integer** types squaring can overflow, following two's-complement `*` semantics
    /// (the result wraps).
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape and dtype as `array`. No
    ///         computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.0, -2.0, 3.0], dtype=np.float32)
    ///     result = jix.square(a)
    ///     assert np.array_equal(result.numpy(), [1.0, 4.0, 9.0])
    ///
    ///     # Integer dtypes are preserved.
    ///     b = jix.compact([2, -3, 4], dtype=np.int32)
    ///     result = jix.square(b)
    ///     assert result.dtype == np.int32
    ///     assert np.array_equal(result.numpy(), [4, 9, 16])
    ///     ```
    square,
    Square,
    dispatch = {
        [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    }
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, 1.0], dtype=np.float32)
    ///     result = jix.exp(a)
    ///     assert result.numpy()[0] == 1.0
    ///     assert abs(result.numpy()[1] - np.e) < 1e-5
    ///     ```
    exp,
    Exp,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
/// Computes the logarithm of each element.
///
/// When `base` is `None` (the default), computes the natural logarithm (`ln x`).
/// When `base` is provided, computes `log_base(x) = ln(x) / ln(base)`.
///
/// Supported dtypes: `f16`, `f32`, `f64`.
///
/// Negative inputs produce `NaN`; zero produces `-inf`.
///
/// Args:
///     array: Input array.
///     base: Optional logarithm base. When omitted or `None`, the natural logarithm is computed.
///         Must be a positive number other than 1.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
///         the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # Natural logarithm (default).
///     a = jix.compact([1.0, np.e], dtype=np.float32)
///     result = jix.log(a)
///     assert abs(result.numpy()[0]) < 1e-5         # ln(1) = 0
///     assert abs(result.numpy()[1] - 1.0) < 1e-5   # ln(e) = 1
///
///     # Base-2 logarithm.
///     c = jix.compact([1.0, 2.0, 8.0], dtype=np.float32)
///     result_c = jix.log(c, base=2)
///     assert abs(result_c.numpy()[0]) < 1e-5        # log2(1) = 0
///     assert abs(result_c.numpy()[1] - 1.0) < 1e-5  # log2(2) = 1
///     assert abs(result_c.numpy()[2] - 3.0) < 1e-5  # log2(8) = 3
///
///     # Base-10 logarithm.
///     d = jix.compact([1.0, 10.0, 100.0], dtype=np.float64)
///     result_d = jix.log(d, base=10)
///     assert abs(result_d.numpy()[0]) < 1e-10       # log10(1) = 0
///     assert abs(result_d.numpy()[1] - 1.0) < 1e-10 # log10(10) = 1
///     assert abs(result_d.numpy()[2] - 2.0) < 1e-10 # log10(100) = 2
///
///     # Zero produces -inf; negative input produces NaN.
///     b = jix.compact([0.0, -1.0], dtype=np.float32)
///     result_b = jix.log(b)
///     assert np.isneginf(result_b.numpy()[0])
///     assert np.isnan(result_b.numpy()[1])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyo3::pyfunction(signature = (array, base=None))]
pub fn log<'py>(array: &Bound<'py, PyAny>, base: Option<f64>) -> pyo3::PyResult<crate::Array> {
    struct LogArgs {
        multiplier: Option<f64>,
    }
    fn log_op_descriptor<T>() -> OpFnDescriptor<1, LogArgs>
    where
        T: Dtyped + num_traits::Float,
        f64: jix_core::scalar::Cast<T>,
    {
        OpFnDescriptor::new1_args::<T>(CastKind::Safe, |a, args: LogArgs| {
            let res = jix_core::ops::Ln::new_array(a).into_py_result()?;
            let Some(multiplier) = args.multiplier else {
                return Ok(res.into_type_dyn().into_any());
            };
            let multiplier = <f64 as jix_core::scalar::Cast<T>>::cast(multiplier);
            let res = res.map(move |x| x * multiplier);
            Ok(res.into_type_dyn().into_any())
        })
    }
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, LogArgs>> = LazyLock::new(|| {
        OpDescriptor::new(
            "log",
            vec![
                log_op_descriptor::<f16>(),
                log_op_descriptor::<f32>(),
                log_op_descriptor::<f64>(),
            ],
        )
    });
    let array = Operand::from_any(array)?;
    let args = LogArgs {
        multiplier: base.map(|b| 1.0 / b.ln()),
    };
    let res = DISPATCH_TABLE.dispatch_args([array], args)?;
    Ok(crate::Array::from_core(res))
}
define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, np.pi / 2], dtype=np.float32)
    ///     result = jix.sin(a)
    ///     assert abs(result.numpy()[0]) < 1e-5   # sin(0) = 0
    ///     assert abs(result.numpy()[1] - 1.0) < 1e-5  # sin(pi/2) = 1
    ///     ```
    sin,
    Sin,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, np.pi], dtype=np.float32)
    ///     result = jix.cos(a)
    ///     assert abs(result.numpy()[0] - 1.0) < 1e-5   # cos(0) = 1
    ///     assert abs(result.numpy()[1] - (-1.0)) < 1e-5  # cos(pi) = -1
    ///     ```
    cos,
    Cos,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, np.pi / 4], dtype=np.float32)
    ///     result = jix.tan(a)
    ///     assert abs(result.numpy()[0]) < 1e-5        # tan(0) = 0
    ///     assert abs(result.numpy()[1] - 1.0) < 1e-5  # tan(pi/4) = 1
    ///     ```
    tan,
    Tan,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, 1.0, -1.0], dtype=np.float32)
    ///     result = jix.asin(a)
    ///     assert abs(result.numpy()[0]) < 1e-5
    ///     assert abs(result.numpy()[1] - np.pi / 2) < 1e-5
    ///     ```
    asin,
    Asin,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, pi]`.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.0, 0.0, -1.0], dtype=np.float32)
    ///     result = jix.acos(a)
    ///     assert abs(result.numpy()[0]) < 1e-5
    ///     assert abs(result.numpy()[1] - np.pi / 2) < 1e-5
    ///     assert abs(result.numpy()[2] - np.pi) < 1e-5
    ///     ```
    acos,
    Acos,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, 1.0], dtype=np.float32)
    ///     result = jix.atan(a)
    ///     assert abs(result.numpy()[0]) < 1e-5
    ///     assert abs(result.numpy()[1] - np.pi / 4) < 1e-5
    ///     ```
    atan,
    Atan,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Returns the sign of each element.
    ///
    /// Supported dtypes: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`,
    /// `uint64`, `float16`, `float32`, `float64`.
    ///
    /// For **signed integer** types: returns `-1`, `0`, or `+1` of the same dtype.
    ///
    /// For **unsigned integer** types: returns `0` or `1` of the same dtype (unsigned values
    /// cannot be negative).
    ///
    /// For **float** types: returns `+1.0` for positive values and `-1.0` for negative values.
    /// Zero is signed: `+0.0` returns `+1.0` and `-0.0` returns `-1.0`.
    ///
    /// **Auto-casting**: `bool` inputs are cast to `int8` before the operation.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape and dtype as `array` (after
    ///         any auto-cast). No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([3, -5, 0], dtype=np.int32)
    ///     result = jix.sign(a)
    ///     assert np.array_equal(result.numpy(), [1, -1, 0])
    ///
    ///     b = jix.compact([3.0, -5.0, -0.1], dtype=np.float32)
    ///     result = jix.sign(b)
    ///     assert np.array_equal(result.numpy(), [1.0, -1.0, -1.0])
    ///
    ///     c = jix.compact([1, 0, 5], dtype=np.uint8)
    ///     result = jix.sign(c)
    ///     assert result.dtype == np.uint8
    ///     assert np.array_equal(result.numpy(), [1, 0, 1])
    ///
    ///     # bool auto-casts to int8.
    ///     d = jix.compact([True, False, True], dtype=np.bool_)
    ///     assert jix.sign(d).dtype == np.int8
    ///     ```
    sign,
    Sign,
    dispatch = {
        [i8, u8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);
define_op1!(
    /// Computes the absolute value of each element.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | same |
    /// | `f16`, `f32`, `f64` | same |
    /// | `Complex<f32>` | `f32` |
    /// | `Complex<f64>` | `f64` |
    ///
    /// For **complex** types the result is the modulus `sqrt(re^2 + im^2)` computed via
    /// `hypot` for numerical stability. The output dtype is the real component type.
    ///
    /// For **signed integer** types, the minimum value overflows: `abs(-128)` on `int8`
    /// wraps back to `-128`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///         the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([-3, 0, 5, -7], dtype=np.int32)
    ///     result = jix.absolute(a)
    ///     assert np.array_equal(result.numpy(), [3, 0, 5, 7])
    ///     ```
    absolute,
    Abs,
    dispatch = {
        [i8, i16, i32, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    }
);
