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
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 + i32 -> i32`, `i32 + f32 -> f64`. This is similar to
    /// numpy's type promotion but may pick a different common type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly (prepend missing leading dimensions as 1, then expand size-1 dims).
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([10, 20, 30], dtype=np.int32)
    ///     result = jix.add(a, b)  # same as `a + b`
    ///     assert np.array_equal(result.numpy(), [11, 22, 33])
    ///
    ///     # Broadcasting: (3, 1) + (1, 4) -> (3, 4)
    ///     x = jix.compact(np.arange(3, dtype=np.int32).reshape(3, 1))
    ///     y = jix.compact(np.arange(4, dtype=np.int32).reshape(1, 4))
    ///     assert jix.add(x, y).shape == (3, 4)
    ///
    ///     # Mixed types: u8 + i32 -> i32
    ///     a8 = jix.compact([1, 2, 3], dtype=np.uint8)
    ///     b32 = jix.compact([100, 200, 300], dtype=np.int32)
    ///     assert jix.add(a8, b32).dtype == np.int32
    ///     ```
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
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 + i32 -> i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([10, 20, 30], dtype=np.int32)
    ///     b = jix.compact([1, 2, 3], dtype=np.int32)
    ///     result = jix.subtract(a, b)  # same as `a - b`
    ///     assert np.array_equal(result.numpy(), [9, 18, 27])
    ///     ```
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
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([4, 5, 6], dtype=np.int32)
    ///     result = jix.multiply(a, b)  # same as `a * b`
    ///     assert np.array_equal(result.numpy(), [4, 10, 18])
    ///     ```
    multiply,
    Mul,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
);
define_op2!(
    /// Element-wise (true) division of two arrays (`a / b`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    /// For integer dtypes use [`jix.floor_divide()`][jix.floor_divide] (or the `//`
    /// operator).
    ///
    /// For **float** types, division by zero produces `+/-inf` or `NaN`.
    /// For **complex** types this is full complex division.
    ///
    /// Available via the `/` operator on arrays.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([10.0, 20.0, 30.0], dtype=np.float32)
    ///     b = jix.compact([4.0, 5.0, 8.0], dtype=np.float32)
    ///     result = jix.divide(a, b)  # same as `a / b`
    ///     assert np.array_equal(result.numpy(), [2.5, 4.0, 3.75])
    ///     ```
    divide,
    Div,
    dispatch = {
        [f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
);
define_op2!(
    /// Element-wise integer division of two arrays (`a // b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// For float and complex dtypes use [`jix.divide()`][jix.divide] (or the `/`
    /// operator).
    ///
    /// The result truncates toward zero (matching Rust's `/` on integers), so for
    /// signed operands with a negative quotient this differs from numpy's
    /// `floor_divide`, which rounds toward negative infinity (e.g. `-7 // 2` is `-3`
    /// here, but `-4` in numpy). For unsigned operands the two agree. Dividing by
    /// zero raises an error.
    ///
    /// Available via the `//` operator on arrays.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 // i32 -> i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([10, 20, 30], dtype=np.int32)
    ///     b = jix.compact([3, 4, 7], dtype=np.int32)
    ///     result = jix.floor_divide(a, b)  # same as `a // b`
    ///     assert np.array_equal(result.numpy(), [3, 5, 4])
    ///     ```
    floor_divide,
    Div,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);
define_op2!(
    /// Element-wise exponentiation (`a` raised to the power `b`).
    ///
    /// Supported base dtypes: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
    /// `f32`, `f64`. For an **integer** base the exponent must be a non-negative
    /// integer (it is cast to an unsigned type); for a **float** base the exponent
    /// may be an integer or a float. Complex dtypes are not supported yet.
    ///
    /// For a directly supported `(base, exponent)` pair the output dtype equals the
    /// base dtype - e.g. `u8 ** u8 -> u8`, `i32 ** u32 -> i32`, `f32 ** i32 -> f32`,
    /// `f64 ** f64 -> f64`. Integer results wrap on overflow (two's complement).
    ///
    /// Negative base with a non-integer exponent produces `NaN`.
    ///
    /// **Type promotion**: when `a` and `b` do not directly match a supported pair,
    /// both operands are cast under Safe casting rules and the first matching pair
    /// is used. Two consequences are worth noting:
    /// - A **signed** exponent cannot fill the unsigned-exponent slot, so an integer
    ///   base with a signed exponent promotes to float: `i8 ** i8 -> f32`,
    ///   `i32 ** i32 -> f64`. (A plain Python `int` exponent is signed, so
    ///   `power(int_array, 3)` is float; pass `np.uint32(3)` to keep an integer result.)
    /// - The widest exponent slot is 32-bit, so a 64-bit exponent also promotes the
    ///   result to float: `u8 ** u64 -> f64`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the base).
    ///     b: Second operand (the exponent).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast
    ///         shape. No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([2.0, 3.0, 4.0], dtype=np.float32)
    ///     b = jix.compact([3.0, 2.0, 0.5], dtype=np.float32)
    ///     result = jix.power(a, b)
    ///     assert np.array_equal(result.numpy(), [8.0, 9.0, 2.0])
    ///
    ///     # Raise each element to a fixed exponent using a typed scalar.
    ///     result2 = jix.power(a, np.float32(2.0))
    ///     assert np.array_equal(result2.numpy(), [4.0, 9.0, 16.0])
    ///
    ///     # Integer base with an unsigned integer exponent keeps the base dtype.
    ///     ai = jix.compact([2, 3, 4], dtype=np.int32)
    ///     result3 = jix.power(ai, np.uint32(3))
    ///     assert result3.dtype == np.dtype("int32")
    ///     assert np.array_equal(result3.numpy(), [8, 27, 64])
    ///     ```
    power,
    Pow,
    dispatch = {
        [
            // integers, exponent must be unsigned
            (u8, u8),
            (i8, u8),
            (u16, u16),
            (i16, u16),
            (u32, u32),
            (i32, u32),
            (u64, u32),
            (i64, u32),
            // floats
            (f32, i32),
            (f32, f32),
            (f64, i32),
            (f64, f64)
            // complex TODO
            // (Complex<f32>, i32),
            // (Complex<f32>, f32),
            // (Complex<f64>, i64),
            // (Complex<f64>, f64)
        ],
        Safe
    }
);
