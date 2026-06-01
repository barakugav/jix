use zix_core::scalar::{f16, Complex};

use crate::ops::common::define_op2;

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
    Add,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
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
    Sub,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
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
    Mul,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
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
    Div,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
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
    Pow,
    dispatch = {
        // [u8, i8, u16, i16, u32, i32, u64, i64, f32, f64], // TODO
        [f32, f64],
        Safe
    }
);
