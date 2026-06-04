use jix_core::scalar::{f16, Complex};

use crate::ops::common::define_op2;

define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types each component is added independently:
    /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
    ///
    /// Available via the `+` operator on arrays.
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 + i32 -> i32`, `i32 + f32 -> f64`. This is similar to
    /// numpy's type promotion but may pick a different common type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly (prepend missing leading dimensions as 1, then expand size-1 dims).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 2, 3], dtype=np.int32)
    /// b = jix.compact([10, 20, 30], dtype=np.int32)
    /// result = jix.add(a, b)  # same as `a + b`
    /// assert np.array_equal(result.numpy(), [11, 22, 33])
    ///
    /// # Broadcasting: (3, 1) + (1, 4) -> (3, 4)
    /// x = jix.compact(np.arange(3, dtype=np.int32).reshape(3, 1))
    /// y = jix.compact(np.arange(4, dtype=np.int32).reshape(1, 4))
    /// assert jix.add(x, y).shape == (3, 4)
    ///
    /// # Mixed types: u8 + i32 -> i32
    /// a8 = jix.compact([1, 2, 3], dtype=np.uint8)
    /// b32 = jix.compact([100, 200, 300], dtype=np.int32)
    /// assert jix.add(a8, b32).dtype == np.int32
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
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result wraps on underflow (two's complement).
    /// For **complex** types each component is subtracted independently:
    /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
    ///
    /// Available via the `-` operator on arrays.
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 + i32 -> i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([10, 20, 30], dtype=np.int32)
    /// b = jix.compact([1, 2, 3], dtype=np.int32)
    /// result = jix.subtract(a, b)  # same as `a - b`
    /// assert np.array_equal(result.numpy(), [9, 18, 27])
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
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types this is full complex multiplication:
    /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
    ///
    /// Available via the `*` operator on arrays.
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 2, 3], dtype=np.int32)
    /// b = jix.compact([4, 5, 6], dtype=np.int32)
    /// result = jix.multiply(a, b)  # same as `a * b`
    /// assert np.array_equal(result.numpy(), [4, 10, 18])
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
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result truncates toward zero; dividing by zero raises
    /// an error. For **float** types, division by zero produces `+/-inf` or `NaN`.
    /// For **complex** types this is full complex division.
    ///
    /// Available via the `/` operator on arrays.
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([10, 20, 30], dtype=np.int32)
    /// b = jix.compact([2, 4, 5], dtype=np.int32)
    /// result = jix.divide(a, b)  # same as `a / b`
    /// assert np.array_equal(result.numpy(), [5, 5, 6])
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
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common float type (Safe casting rules; `f32 + f64 -> f64`).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([2.0, 3.0, 4.0], dtype=np.float32)
    /// b = jix.compact([3.0, 2.0, 0.5], dtype=np.float32)
    /// result = jix.power(a, b)
    /// assert np.array_equal(result.numpy(), [8.0, 9.0, 2.0])
    ///
    /// # Raise each element to a fixed exponent using a typed scalar.
    /// result2 = jix.power(a, np.float32(2.0))
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
