use jix_core::scalar::bool8;

use crate::ops::common::{define_op1, define_op2};

define_op2!(
    /// Element-wise logical AND of two arrays.
    ///
    /// Accepted input dtypes: any integer, float or complex dtype, and `bool`.
    ///
    /// Each operand is cast to `bool` (zero -> `False`, any non-zero value -> `True`), so mixed
    /// dtypes are allowed.
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
    ///     a = jix.compact([0, 1, 0, 5], dtype=np.int32)
    ///     b = jix.compact([1, 1, 0, 0], dtype=np.int32)
    ///     result = jix.logical_and(a, b)
    ///     assert np.array_equal(result.numpy(), [False, True, False, False])
    ///
    ///     # Mixed dtypes: float and int both cast to bool
    ///     c = jix.compact([0.0, 1.5], dtype=np.float32)
    ///     d = jix.compact([1, 0], dtype=np.int32)
    ///     assert np.array_equal(jix.logical_and(c, d).numpy(), [False, False])
    ///     ```
    logical_and,
    And,
    dispatch = {
        [bool8],
        Unsafe
    }
);

define_op2!(
    /// Element-wise logical OR of two arrays.
    ///
    /// Accepted input dtypes: any integer, float or complex dtype, and `bool`.
    ///
    /// Each operand is cast to `bool` (zero -> `False`, any non-zero value -> `True`).
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
    ///     a = jix.compact([0, 1, 0, 5], dtype=np.int32)
    ///     b = jix.compact([0, 0, 0, 0], dtype=np.int32)
    ///     result = jix.logical_or(a, b)
    ///     assert np.array_equal(result.numpy(), [False, True, False, True])
    ///     ```
    logical_or,
    Or,
    dispatch = {
        [bool8],
        Unsafe
    }
);

define_op2!(
    /// Element-wise logical XOR of two arrays.
    ///
    /// Accepted input dtypes: any integer, float or complex dtype, and `bool`.
    ///
    /// Each operand is cast to `bool` (zero -> `False`, any non-zero value -> `True`).
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
    ///     a = jix.compact([0, 1, 0, 5], dtype=np.int32)
    ///     b = jix.compact([0, 1, 1, 0], dtype=np.int32)
    ///     result = jix.logical_xor(a, b)
    ///     assert np.array_equal(result.numpy(), [False, False, True, True])
    ///     ```
    logical_xor,
    Xor,
    dispatch = {
        [bool8],
        Unsafe
    }
);

define_op1!(
    /// Element-wise logical NOT.
    ///
    /// Accepted input dtypes: any integer, float or complex dtype, and `bool`.
    ///
    /// Each element is cast to `bool` (zero -> `False`, any non-zero value -> `True`), then
    /// negated.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0, 1, -3, 0], dtype=np.int32)
    ///     result = jix.logical_not(a)
    ///     assert np.array_equal(result.numpy(), [True, False, False, True])
    ///     ```
    logical_not,
    Not,
    dispatch = {
        [bool8],
        Unsafe
    }
);

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype equals the promoted input type.
    ///
    /// For `bool` this is logical AND.
    ///
    /// **Type promotion**: if `a` and `b` have different integer/bool dtypes, both are cast
    /// to the smallest integer type that can represent both (Safe casting rules).
    /// For example `u8 & u16 -> u16`, `i8 & u16 -> i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b1100, 0b1010, 0b1111], dtype=np.uint8)
    ///     b = jix.compact([0b1010, 0b0101, 0b0000], dtype=np.uint8)
    ///     result = jix.bitwise_and(a, b)
    ///     assert np.array_equal(result.numpy(), [0b1000, 0b0000, 0b0000])
    ///     ```
    bitwise_and,
    And,
    dispatch = {
        [bool8, u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op2!(
    /// Element-wise bitwise OR of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype equals the promoted input type.
    ///
    /// For `bool` this is logical OR.
    ///
    /// **Type promotion**: if `a` and `b` have different integer/bool dtypes, both are cast
    /// to the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b1100, 0b1010, 0b0000], dtype=np.uint8)
    ///     b = jix.compact([0b1010, 0b0101, 0b1111], dtype=np.uint8)
    ///     result = jix.bitwise_or(a, b)
    ///     assert np.array_equal(result.numpy(), [0b1110, 0b1111, 0b1111])
    ///     ```
    bitwise_or,
    Or,
    dispatch = {
        [bool8, u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op2!(
    /// Element-wise bitwise XOR of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype equals the promoted input type.
    ///
    /// For `bool` this is logical XOR.
    ///
    /// **Type promotion**: if `a` and `b` have different integer/bool dtypes, both are cast
    /// to the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b1100, 0b1010, 0b1111], dtype=np.uint8)
    ///     b = jix.compact([0b1010, 0b1010, 0b1111], dtype=np.uint8)
    ///     result = jix.bitwise_xor(a, b)
    ///     assert np.array_equal(result.numpy(), [0b0110, 0b0000, 0b0000])
    ///     ```
    bitwise_xor,
    Xor,
    dispatch = {
        [bool8, u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op1!(
    /// Element-wise bitwise NOT (one's complement).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype and shape equal the input.
    ///
    /// For `bool` this is logical NOT. For signed integers the result is `-(x + 1)`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b00001111, 0b11110000, 0], dtype=np.uint8)
    ///     result = jix.bitwise_not(a)
    ///     assert np.array_equal(result.numpy(), [0b11110000, 0b00001111, 0xFF])
    ///     ```
    bitwise_not,
    Not,
    dispatch = {
        [bool8, u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`; `bool` is cast to
    /// `u8`.
    /// Output dtype equals the promoted input type.
    ///
    /// Vacated bits are filled with zeros. As in numpy, shifting by the bit width of the type
    /// or more, or by a negative value, gives `0`.
    ///
    /// **Type promotion**: if `a` and `b` have different integer dtypes, both are cast to
    /// the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b00000001, 0b00000010, 0b00000100], dtype=np.uint8)
    ///     b = jix.compact([1, 2, 3], dtype=np.uint8)
    ///     result = jix.bitwise_left_shift(a, b)
    ///     assert np.array_equal(result.numpy(), [0b00000010, 0b00001000, 0b00100000])
    ///     ```
    bitwise_left_shift,
    BitwiseShiftLeft,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op2!(
    /// Element-wise right shift (`a >> b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`; `bool` is cast to
    /// `u8`.
    /// Output dtype equals the promoted input type.
    ///
    /// For **unsigned** types this is a logical shift: vacated bits are filled with zeros.
    /// For **signed** types this is an arithmetic shift: vacated bits are filled with the
    /// sign bit. As in numpy, shifting by the bit width of the type or more, or by a negative
    /// value, shifts all the bits out: `0`, or `-1` for a negative signed value.
    ///
    /// **Type promotion**: if `a` and `b` have different integer dtypes, both are cast to
    /// the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b10000000, 0b00100000, 0b00001000], dtype=np.uint8)
    ///     b = jix.compact([1, 2, 3], dtype=np.uint8)
    ///     result = jix.bitwise_right_shift(a, b)
    ///     assert np.array_equal(result.numpy(), [0b01000000, 0b00001000, 0b00000001])
    ///     ```
    bitwise_right_shift,
    BitwiseShiftRight,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op2!(
    /// Element-wise bitwise left rotation.
    ///
    /// Supported value dtypes (`a`): `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// The rotation amount (`b`) must be `u32` or a Python int. No operand is cast, and the output
    /// dtype equals `a`'s dtype.
    ///
    /// Bits shifted out of the most-significant end wrap around to the least-significant end.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: Value to rotate.
    ///     b: Rotation amount.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b10000001, 0b00000001, 0b11110000], dtype=np.uint8)
    ///     b = jix.compact([1, 3, 4], dtype=np.uint32)
    ///     result = jix.bitwise_rotate_left(a, b)
    ///     assert np.array_equal(result.numpy(), [0b00000011, 0b00001000, 0b00001111])
    ///     ```
    bitwise_rotate_left,
    BitwiseRotateLeft,
    dispatch = {
        [
            (u8, u32),
            (i8, u32),
            (u16, u32),
            (i16, u32),
            (u32, u32),
            (i32, u32),
            (u64, u32),
            (i64, u32)
        ],
        None
    }
);

define_op2!(
    /// Element-wise bitwise right rotation.
    ///
    /// Supported value dtypes (`a`): `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// The rotation amount (`b`) must be `u32` or a Python int. No operand is cast, and the output
    /// dtype equals `a`'s dtype.
    ///
    /// Bits shifted out of the least-significant end wrap around to the most-significant end.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: Value to rotate.
    ///     b: Rotation amount.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b10000001, 0b00001000, 0b00001111], dtype=np.uint8)
    ///     b = jix.compact([1, 3, 4], dtype=np.uint32)
    ///     result = jix.bitwise_rotate_right(a, b)
    ///     assert np.array_equal(result.numpy(), [0b11000000, 0b00000001, 0b11110000])
    ///     ```
    bitwise_rotate_right,
    BitwiseRotateRight,
    dispatch = {
        [
            (u8, u32),
            (i8, u32),
            (u16, u32),
            (i16, u32),
            (u32, u32),
            (i32, u32),
            (u64, u32),
            (i64, u32)
        ],
        None
    }
);
