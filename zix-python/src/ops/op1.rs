use crate::ops::common::define_op1;

define_op1!(
    /// Arithmetic negation applied element-wise (`-array`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `f16`, `f32`, `f64`,
    /// `Complex<f32>`, `Complex<f64>`. Output dtype and shape equal the input.
    ///
    /// For **integer** types the result is the two's-complement negation. Negating the
    /// minimum representable value (e.g. `int8` `-128`) overflows and wraps.
    /// For **complex** types both components are negated independently.
    ///
    /// Available via the unary `-` operator on arrays: `-arr`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1.0, -2.5, 3.0], dtype=np.float32)
    /// result = zix.negative(a)
    /// assert np.array_equal(result.numpy(), [-1.0, 2.5, -3.0])
    /// ```
    negative,
    Neg
);
define_op1!(
    /// Rounds each element down to the nearest integer (towards −∞).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1.1, 2.9, -1.1, -2.9], dtype=np.float32)
    /// result = zix.floor(a)
    /// assert np.array_equal(result.numpy(), [1.0, 2.0, -2.0, -3.0])
    /// ```
    floor,
    Floor
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +∞).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1.1, 2.0, -1.7, -0.1], dtype=np.float32)
    /// result = zix.ceil(a)
    /// assert np.array_equal(result.numpy(), [2.0, 2.0, -1.0, 0.0])
    /// ```
    ceil,
    Ceil
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Ties (values exactly halfway between two integers) are broken by rounding away from
    /// zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from "round-half-to-even"
    /// (banker's rounding) used by Python's built-in `round()` and `numpy.round_`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1.4, 1.6, 2.5, -0.5], dtype=np.float32)
    /// result = zix.round(a)
    /// assert np.array_equal(result.numpy(), [1.0, 2.0, 3.0, -1.0])
    /// ```
    round,
    Round
);
define_op1!(
    /// Computes the square root of each element.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Negative inputs produce `NaN`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([4.0, 9.0, 16.0], dtype=np.float32)
    /// result = zix.sqrt(a)
    /// assert np.array_equal(result.numpy(), [2.0, 3.0, 4.0])
    ///
    /// # Negative input produces NaN.
    /// b = zix.compact([-1.0], dtype=np.float32)
    /// assert np.isnan(zix.sqrt(b).numpy()[0])
    /// ```
    sqrt,
    Sqrt
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0.0, 1.0], dtype=np.float32)
    /// result = zix.exp(a)
    /// assert result.numpy()[0] == 1.0
    /// assert abs(result.numpy()[1] - np.e) < 1e-5
    /// ```
    exp,
    Exp
);
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Negative inputs produce `NaN`; zero produces `-inf`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1.0, np.e], dtype=np.float32)
    /// result = zix.log(a)
    /// assert abs(result.numpy()[0]) < 1e-5   # ln(1) = 0
    /// assert abs(result.numpy()[1] - 1.0) < 1e-5  # ln(e) = 1
    ///
    /// # Zero produces -inf; negative input produces NaN.
    /// b = zix.compact([0.0, -1.0], dtype=np.float32)
    /// result_b = zix.log(b)
    /// assert np.isneginf(result_b.numpy()[0])
    /// assert np.isnan(result_b.numpy()[1])
    /// ```
    log,
    Ln
);
define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0.0, np.pi / 2], dtype=np.float32)
    /// result = zix.sin(a)
    /// assert abs(result.numpy()[0]) < 1e-5   # sin(0) = 0
    /// assert abs(result.numpy()[1] - 1.0) < 1e-5  # sin(π/2) = 1
    /// ```
    sin,
    Sin
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0.0, np.pi], dtype=np.float32)
    /// result = zix.cos(a)
    /// assert abs(result.numpy()[0] - 1.0) < 1e-5   # cos(0) = 1
    /// assert abs(result.numpy()[1] - (-1.0)) < 1e-5  # cos(π) = -1
    /// ```
    cos,
    Cos
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0.0, np.pi / 4], dtype=np.float32)
    /// result = zix.tan(a)
    /// assert abs(result.numpy()[0]) < 1e-5        # tan(0) = 0
    /// assert abs(result.numpy()[1] - 1.0) < 1e-5  # tan(π/4) = 1
    /// ```
    tan,
    Tan
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-π/2, π/2]`.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0.0, 1.0, -1.0], dtype=np.float32)
    /// result = zix.asin(a)
    /// assert abs(result.numpy()[0]) < 1e-5
    /// assert abs(result.numpy()[1] - np.pi / 2) < 1e-5
    /// ```
    asin,
    Asin
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, π]`.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1.0, 0.0, -1.0], dtype=np.float32)
    /// result = zix.acos(a)
    /// assert abs(result.numpy()[0]) < 1e-5
    /// assert abs(result.numpy()[1] - np.pi / 2) < 1e-5
    /// assert abs(result.numpy()[2] - np.pi) < 1e-5
    /// ```
    acos,
    Acos
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-π/2, π/2)`.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0.0, 1.0], dtype=np.float32)
    /// result = zix.atan(a)
    /// assert abs(result.numpy()[0]) < 1e-5
    /// assert abs(result.numpy()[1] - np.pi / 4) < 1e-5
    /// ```
    atan,
    Atan
);
define_op1!(
    /// Returns the sign of each element as a floating-point value.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype and shape equal the input.
    ///
    /// Returns `+1.0` for positive values and `-1.0` for negative values.
    /// Zero is signed: `+0.0` returns `+1.0` and `-0.0` returns `-1.0`.
    ///
    /// This function deviates from `numpy.sign` in that it only supports float types.
    /// Use `numpy.sign` for integer inputs.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([3.0, -5.0, -0.1], dtype=np.float32)
    /// result = zix.signum(a)
    /// assert np.array_equal(result.numpy(), [1.0, -1.0, -1.0])
    /// ```
    signum,
    Signum
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
    /// For **complex** types the result is the modulus `sqrt(re² + im²)` computed via
    /// `hypot` for numerical stability. The output dtype is the real component type.
    ///
    /// For **signed integer** types, the minimum value overflows: `abs(-128)` on `int8`
    /// wraps back to `-128`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([-3, 0, 5, -7], dtype=np.int32)
    /// result = zix.absolute(a)
    /// assert np.array_equal(result.numpy(), [3, 0, 5, 7])
    /// ```
    absolute,
    Abs
);
