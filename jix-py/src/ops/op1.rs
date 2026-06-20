use jix_core::scalar::{f16, Complex};

use crate::ops::common::define_op1;

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
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Negative inputs produce `NaN`; zero produces `-inf`.
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
    ///     a = jix.compact([1.0, np.e], dtype=np.float32)
    ///     result = jix.log(a)
    ///     assert abs(result.numpy()[0]) < 1e-5   # ln(1) = 0
    ///     assert abs(result.numpy()[1] - 1.0) < 1e-5  # ln(e) = 1
    ///
    ///     # Zero produces -inf; negative input produces NaN.
    ///     b = jix.compact([0.0, -1.0], dtype=np.float32)
    ///     result_b = jix.log(b)
    ///     assert np.isneginf(result_b.numpy()[0])
    ///     assert np.isnan(result_b.numpy()[1])
    ///     ```
    log,
    Ln,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
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
    /// Returns the sign of each element as a floating-point value.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Returns `+1.0` for positive values and `-1.0` for negative values.
    /// Zero is signed: `+0.0` returns `+1.0` and `-0.0` returns `-1.0`.
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
    ///     a = jix.compact([3.0, -5.0, -0.1], dtype=np.float32)
    ///     result = jix.signum(a)
    ///     assert np.array_equal(result.numpy(), [1.0, -1.0, -1.0])
    ///     ```
    signum,
    Signum,
    dispatch = {
        [f16, f32, f64], // TODO: signed integers
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
