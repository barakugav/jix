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

// op1
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
    /// a = zix.asarray(np.array([1.0, -2.5, 3.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([1.1, 2.9, -1.1, -2.9], dtype=np.float32))
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
    /// a = zix.asarray(np.array([1.1, 2.0, -1.7, -0.1], dtype=np.float32))
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
    /// a = zix.asarray(np.array([1.4, 1.6, 2.5, -0.5], dtype=np.float32))
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
    /// a = zix.asarray(np.array([4.0, 9.0, 16.0], dtype=np.float32))
    /// result = zix.sqrt(a)
    /// assert np.array_equal(result.numpy(), [2.0, 3.0, 4.0])
    ///
    /// # Negative input produces NaN.
    /// b = zix.asarray(np.array([-1.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([0.0, 1.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([1.0, np.e], dtype=np.float32))
    /// result = zix.log(a)
    /// assert abs(result.numpy()[0]) < 1e-5   # ln(1) = 0
    /// assert abs(result.numpy()[1] - 1.0) < 1e-5  # ln(e) = 1
    ///
    /// # Zero produces -inf; negative input produces NaN.
    /// b = zix.asarray(np.array([0.0, -1.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([0.0, np.pi / 2], dtype=np.float32))
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
    /// a = zix.asarray(np.array([0.0, np.pi], dtype=np.float32))
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
    /// a = zix.asarray(np.array([0.0, np.pi / 4], dtype=np.float32))
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
    /// a = zix.asarray(np.array([0.0, 1.0, -1.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([1.0, 0.0, -1.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([0.0, 1.0], dtype=np.float32))
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
    /// a = zix.asarray(np.array([3.0, -5.0, -0.1], dtype=np.float32))
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
    /// a = zix.asarray(np.array([-3, 0, 5, -7], dtype=np.int32))
    /// result = zix.absolute(a)
    /// assert np.array_equal(result.numpy(), [3, 0, 5, 7])
    /// ```
    absolute,
    Abs
);

// op2
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
    /// a = zix.asarray(np.array([1, 2, 3], dtype=np.int32))
    /// b = zix.asarray(np.array([10, 20, 30], dtype=np.int32))
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
    /// a = zix.asarray(np.array([10, 20, 30], dtype=np.int32))
    /// b = zix.asarray(np.array([1, 2, 3], dtype=np.int32))
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
    /// a = zix.asarray(np.array([1, 2, 3], dtype=np.int32))
    /// b = zix.asarray(np.array([4, 5, 6], dtype=np.int32))
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
    /// For **float** types, division by zero produces `±inf` or `NaN`.
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
    /// a = zix.asarray(np.array([10, 20, 30], dtype=np.int32))
    /// b = zix.asarray(np.array([2, 4, 5], dtype=np.int32))
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
    /// a = zix.asarray(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    /// b = zix.asarray(np.array([3.0, 2.0, 0.5], dtype=np.float32))
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

// logical1
define_op1!(
    /// Tests whether each element is `NaN` (not a number).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `True` if the element is `NaN`, `False` otherwise.
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
    /// a = zix.asarray(np.array([float('nan'), 1.0, float('inf'), -1.0], dtype=np.float32))
    /// result = zix.is_nan(a)
    /// assert np.array_equal(result.numpy(), [True, False, False, False])
    /// ```
    is_nan,
    IsNan
);
define_op1!(
    /// Tests whether each element is finite (not `±inf` and not `NaN`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `True` if the element is a finite number, `False` for `±inf` and `NaN`.
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
    /// a = zix.asarray(np.array([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float32))
    /// result = zix.is_finite(a)
    /// assert np.array_equal(result.numpy(), [True, False, False, False])
    /// ```
    is_finite,
    IsFinite
);
define_op1!(
    /// Tests whether each element is infinite (`+inf` or `-inf`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `True` only for `+inf` and `-inf`; returns `False` for finite values and `NaN`.
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
    /// a = zix.asarray(np.array([float('inf'), float('-inf'), float('nan'), 1.0], dtype=np.float32))
    /// result = zix.is_infinite(a)
    /// assert np.array_equal(result.numpy(), [True, True, False, False])
    /// ```
    is_infinite,
    IsInfinite
);

// reduction
macro_rules! define_reduction_op {
    ($(#[$meta:meta])* $name:ident, $core_op:ident $(, extra_args = ($($extra_arg:ident : $extra_ty:ty = $extra_default:expr),+))?) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axes=None,
            keepdims=false,
            $($($extra_arg=$extra_default,)+)?
        ))]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axes: Option<Vec<i32>>,
            keepdims: bool,
            $($($extra_arg: $extra_ty),+)?
        ) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::as_core_array(array)?;
            let axes = match axes {
                Some(axes) => crate::util::normalize_axes(axes, array.ndim())?,
                None => (0..array.ndim()).collect(),
            };
            let res = zix_core::ops::$core_op::new(array, &axes, keepdims $($(, $extra_arg)+)?);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
    ($(#[$meta:meta])* $name:ident, $core_op:ident, single_axis = true) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axis=None,
            keepdims=false,
        ))]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axis: Option<i32>,
            keepdims: bool,
        ) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::as_core_array(array)?;
            let axis = match axis {
                Some(axis) => crate::util::normalize_axis(axis, array.ndim())?,
                None => {
                    if array.ndim() != 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "axis must be specified for arrays with ndim != 1",
                        ));
                    }
                    0
                },
            };
            let res = zix_core::ops::$core_op::new(array, axis, keepdims);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
}
define_reduction_op!(
    /// Reduces one or more axes by taking the maximum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the input dtype.
    ///
    /// For **float** types, `NaN` values are ignored: the result is the maximum of all
    /// non-`NaN` values. If all elements are `NaN`, the result is `NaN`.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    /// # Reduce all axes → scalar
    /// assert zix.max(a).numpy()[()] == 6
    /// # Reduce axis 0 → shape [3]
    /// assert np.array_equal(zix.max(a, axes=[0]).numpy(), [4, 5, 6])
    /// # Reduce axis 0, keepdims=True → shape [1, 3]
    /// assert zix.max(a, axes=[0], keepdims=True).numpy().shape == (1, 3)
    /// ```
    max,
    Max
);
define_reduction_op!(
    /// Reduces one or more axes by taking the minimum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the input dtype.
    ///
    /// For **float** types, `NaN` values are ignored: the result is the minimum of all
    /// non-`NaN` values. If all elements are `NaN`, the result is `NaN`.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    /// # Reduce all axes → scalar
    /// assert zix.min(a).numpy()[()] == 1
    /// # Reduce axis 0 → shape [3]
    /// assert np.array_equal(zix.min(a, axes=[0]).numpy(), [1, 2, 3])
    /// ```
    min,
    Min
);
define_reduction_op!(
    /// Returns the index of the maximum element along a single axis.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype is `u64`.
    ///
    /// If multiple elements share the maximum value, the index of the first occurrence is
    /// returned. For **float** types, `NaN` values are treated as less than any non-`NaN`
    /// value, so they are never selected unless all elements are `NaN`.
    ///
    /// `axis` accepts negative values (e.g. `-1` for the last axis). For 1-D arrays,
    /// `axis=None` is equivalent to `axis=0`.
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
    /// a = zix.asarray(np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int32))
    /// # Index of max along axis 1 (per row)
    /// assert np.array_equal(zix.argmax(a, axis=1).numpy(), [1, 2])
    /// # Index of max along axis 0 (per column)
    /// assert np.array_equal(zix.argmax(a, axis=0).numpy(), [1, 0, 1])
    /// ```
    argmax,
    ArgMax,
    single_axis = true
);
define_reduction_op!(
    /// Returns the index of the minimum element along a single axis.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype is `u64`.
    ///
    /// If multiple elements share the minimum value, the index of the first occurrence is
    /// returned. For **float** types, `NaN` values are treated as greater than any non-`NaN`
    /// value, so they are never selected unless all elements are `NaN`.
    ///
    /// `axis` accepts negative values (e.g. `-1` for the last axis). For 1-D arrays,
    /// `axis=None` is equivalent to `axis=0`.
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
    /// a = zix.asarray(np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int32))
    /// # Index of min along axis 1 (per row)
    /// assert np.array_equal(zix.argmin(a, axis=1).numpy(), [0, 1])
    /// ```
    argmin,
    ArgMin,
    single_axis = true
);
define_reduction_op!(
    /// Reduces one or more axes by summing all elements.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Output dtype equals the input
    /// dtype.
    ///
    /// For **integer** types, the result wraps on overflow (two's complement). For
    /// large sums, consider casting to a wider type first.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    /// assert zix.sum(a).numpy()[()] == 21
    /// assert np.array_equal(zix.sum(a, axes=[0]).numpy(), [5, 7, 9])
    /// ```
    sum,
    Sum
);
define_reduction_op!(
    /// Reduces one or more axes by multiplying all elements.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Output dtype equals the input
    /// dtype.
    ///
    /// For **integer** types, the result wraps on overflow. For large products, consider
    /// casting to a wider type first.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    /// assert zix.product(a).numpy()[()] == 720
    /// assert np.array_equal(zix.product(a, axes=[0]).numpy(), [4, 10, 18])
    /// ```
    product,
    Product
);
define_reduction_op!(
    /// Computes the arithmetic mean along one or more axes.
    ///
    /// Supported dtypes: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Output dtype equals
    /// the input dtype.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// This function deviates from numpy in that only float and complex types are supported.
    /// For integer inputs, cast to `f64` first with `zix.astype(array, 'float64')`.
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
    /// a = zix.asarray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    /// assert zix.mean(a).numpy()[()] == 3.5
    /// assert np.allclose(zix.mean(a, axes=[0]).numpy(), [2.5, 3.5, 4.5])
    /// ```
    mean,
    Mean
);
define_reduction_op!(
    /// Computes the variance along one or more axes.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype equals the input dtype.
    ///
    /// `ddof` (delta degrees of freedom) defaults to `0.0` (population variance). Use
    /// `ddof=1.0` for the sample (Bessel-corrected) variance.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32))
    /// assert abs(zix.var(a).numpy()[()] - 4.0) < 1e-5   # population variance
    /// assert abs(zix.var(a, ddof=1.0).numpy()[()] - np.var(a.numpy(), ddof=1)) < 1e-3
    /// ```
    var,
    Variance,
    extra_args = (ddof: f64 = 0.0)
);
define_reduction_op!(
    /// Computes the standard deviation along one or more axes.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype equals the input dtype.
    ///
    /// `ddof` (delta degrees of freedom) defaults to `0.0` (population standard deviation).
    /// Use `ddof=1.0` for the sample (Bessel-corrected) standard deviation.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32))
    /// assert abs(zix.std(a).numpy()[()] - 2.0) < 1e-5   # population std dev
    /// ```
    std,
    StandardDeviation,
    extra_args = (ddof: f64 = 0.0)
);
define_reduction_op!(
    /// Reduces one or more axes with logical AND: returns `True` if all elements are truthy.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero → `False`, non-zero → `True`), then the
    /// AND reduction is applied. Returns `True` only if every element in the reduced
    /// dimensions is truthy; returns `True` for empty reductions.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([[True, True], [True, False]]))
    /// assert zix.all(a).numpy()[()] == False
    /// assert np.array_equal(zix.all(a, axes=[1]).numpy(), [True, False])
    /// ```
    all,
    All
);
define_reduction_op!(
    /// Reduces one or more axes with logical OR: returns `True` if any element is truthy.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero → `False`, non-zero → `True`), then the
    /// OR reduction is applied. Returns `True` if at least one element in the reduced
    /// dimensions is truthy; returns `False` for empty reductions.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
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
    /// a = zix.asarray(np.array([[False, False], [False, True]]))
    /// assert zix.any(a).numpy()[()] == True
    /// assert np.array_equal(zix.any(a, axes=[1]).numpy(), [False, True])
    /// ```
    any,
    Any
);

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
/// a = zix.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
/// result = zix.astype(a, np.float64)
/// assert np.array_equal(result.numpy(), [1.0, 2.0, 3.0, 4.0])
///
/// # Zero → False, non-zero → True
/// b = zix.asarray(np.array([0, 1, -2, 0], dtype=np.int32))
/// result = zix.astype(b, bool)
/// assert np.array_equal(result.numpy(), [False, True, True, False])
/// ```
pub fn astype<'py>(array: &Bound<'py, Array>, dtype: &Bound<'py, PyAny>) -> PyResult<Array> {
    let array = array.get().to_core_array();
    let dtype = dtype_from_any(dtype)?;
    let ret = zix_core::ops::AsType::new(array, dtype).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
