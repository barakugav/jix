use crate::ops::common::{define_op1, define_op2};

define_op2!(
    /// Element-wise logical AND of two arrays.
    ///
    /// Accepted input dtypes: any integer (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`), `f16`, `f32`, `f64`,
    /// and `bool`. Output dtype is always `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero value -> `True`;
    /// for `bool` this is the identity), then the logical AND is applied.
    ///
    /// Mixed dtypes are allowed: each operand is independently cast to `bool` before the
    /// operation.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// Args:
    ///     a: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     b: May be anything that [`jix.asarray()`][jix.asarray] accepts.
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
        [bool],
        Unsafe
    }
);

define_op2!(
    /// Element-wise logical OR of two arrays.
    ///
    /// Accepted input dtypes: any integer (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`), `f16`, `f32`, `f64`,
    /// and `bool`. Output dtype is always `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero -> `True`), then the
    /// logical OR is applied. Returns `True` when at least one element is truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// Args:
    ///     a: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     b: May be anything that [`jix.asarray()`][jix.asarray] accepts.
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
        [bool],
        Unsafe
    }
);

define_op2!(
    /// Element-wise logical XOR of two arrays.
    ///
    /// Accepted input dtypes: any integer (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`), `f16`, `f32`, `f64`,
    /// and `bool`. Output dtype is always `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero -> `True`), then the
    /// logical XOR is applied. Returns `True` when exactly one element is truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// Args:
    ///     a: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     b: May be anything that [`jix.asarray()`][jix.asarray] accepts.
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
        [bool],
        Unsafe
    }
);

define_op1!(
    /// Element-wise logical NOT.
    ///
    /// Accepted input dtypes: any integer (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`), `f16`, `f32`, `f64`,
    /// and `bool`. Output dtype is always `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero value -> `True`),
    /// then negated. Returns `True` for zero (falsy) elements and `False` for non-zero
    /// (truthy) elements.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
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
        [bool],
        Unsafe
    }
);

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype equals the promoted input type.
    ///
    /// Applies the bitwise AND to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical AND.
    ///
    /// **Type promotion**: if `a` and `b` have different integer/bool dtypes, both are cast
    /// to the smallest integer type that can represent both (Safe casting rules).
    /// For example `u8 & u16 -> u16`, `i8 & u16 -> i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     b: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
        [bool, u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op2!(
    /// Element-wise bitwise OR of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype equals the promoted input type.
    ///
    /// Applies the bitwise OR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical OR.
    ///
    /// **Type promotion**: if `a` and `b` have different integer/bool dtypes, both are cast
    /// to the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     b: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
        [bool, u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op2!(
    /// Element-wise bitwise XOR of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype equals the promoted input type.
    ///
    /// Applies the bitwise XOR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical XOR.
    ///
    /// **Type promotion**: if `a` and `b` have different integer/bool dtypes, both are cast
    /// to the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     b: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
        [bool, u8, i8, u16, i16, u32, i32, u64, i64],
        Safe
    }
);

define_op1!(
    /// Element-wise bitwise NOT (one's complement).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype and shape equal the input.
    ///
    /// Flips every bit. For `bool` this is equivalent to logical NOT.
    /// For signed integers the result is `-(x + 1)` (e.g. `~0` on `int32` gives `-1`).
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///     the result is read.
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
        [bool, u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype equals the promoted input type.
    ///
    /// Shifts the bits of each element of `a` left by the corresponding value in `b`.
    /// Vacated bits are filled with zeros. Shifting by a value greater than or equal to the
    /// bit width of the type produces zero.
    ///
    /// **Type promotion**: if `a` and `b` have different integer dtypes, both are cast to
    /// the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that jix.asarray() accepts.
    ///     b: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype equals the promoted input type.
    ///
    /// For **unsigned** types this is a logical shift: vacated bits are filled with zeros.
    /// For **signed** types this is an arithmetic shift: vacated bits are filled with the
    /// sign bit (the result preserves the sign). Shifting by a value greater than or equal
    /// to the bit width produces zero (unsigned) or the sign-extended value (signed).
    ///
    /// **Type promotion**: if `a` and `b` have different integer dtypes, both are cast to
    /// the smallest integer type that can represent both (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that jix.asarray() accepts.
    ///     b: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
    /// The rotation amount (`b`) must be castable to `u32`. Output dtype equals `a`'s dtype.
    ///
    /// Rotates the bits of each element of `a` left by the corresponding value in `b`
    /// (interpreted as `u32`). Unlike a left shift, bits shifted out of the most-significant
    /// position wrap around to the least-significant position, so no bits are lost.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that jix.asarray() accepts.
    ///     b: May be anything that jix.asarray() accepts. Must be castable to `u32`.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
            (i8, u32),
            (u8, u32),
            (i16, u32),
            (u16, u32),
            (i32, u32),
            (u32, u32),
            (i64, u32),
            (u64, u32)
        ],
        Safe
    }
);

define_op2!(
    /// Element-wise bitwise right rotation.
    ///
    /// Supported value dtypes (`a`): `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// The rotation amount (`b`) must be castable to `u32`. Output dtype equals `a`'s dtype.
    ///
    /// Rotates the bits of each element of `a` right by the corresponding value in `b`
    /// (interpreted as `u32`). Unlike a right shift, bits shifted out of the least-significant
    /// position wrap around to the most-significant position, so no bits are lost.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: May be anything that jix.asarray() accepts.
    ///     b: May be anything that jix.asarray() accepts. Must be castable to `u32`.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same dtype and broadcast shape. No computation
    ///     occurs until the result is read.
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
            (i8, u32),
            (u8, u32),
            (i16, u32),
            (u16, u32),
            (i32, u32),
            (u32, u32),
            (i64, u32),
            (u64, u32)
        ],
        Safe
    }
);

define_op1!(
    /// Counts the number of set bits (`1`s) in each element (population count).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Also known as the Hamming weight. For signed integers the bit representation
    /// (including the sign bit) is used.
    ///
    /// Args:
    ///     array: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///     dtype. No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b00001111, 0b11001100, 0b11111111], dtype=np.uint8)
    ///     result = jix.count_ones(a)
    ///     assert np.array_equal(result.numpy(), [4, 4, 8])
    ///     ```
    count_ones,
    CountOnes,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op1!(
    /// Counts the number of unset bits (`0`s) in each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Equivalent to `bit_width - count_ones`. For signed integers the full bit
    /// representation (including the sign bit) is used.
    ///
    /// Args:
    ///     array: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///     dtype. No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b11110000, 0b00001111, 0b11111111], dtype=np.uint8)
    ///     result = jix.count_zeros(a)
    ///     assert np.array_equal(result.numpy(), [4, 4, 0])
    ///
    ///     # Zero has all bits unset: count_zeros == bit width.
    ///     b = jix.compact([0], dtype=np.uint8)
    ///     assert jix.count_zeros(b).numpy()[0] == 8  # u8 has 8 bits
    ///     ```
    count_zeros,
    CountZeros,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op1!(
    /// Counts the number of leading zero bits in each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Counts zeros from the most-significant bit down to (but not including) the first
    /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for a
    /// `uint32` zero).
    ///
    /// Args:
    ///     array: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///     dtype. No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0x00010000, 0x80000000, 0x00000001], dtype=np.uint32)
    ///     result = jix.leading_zeros(a)
    ///     assert np.array_equal(result.numpy(), [15, 0, 31])
    ///
    ///     # Zero returns the bit width of the type (32 for uint32).
    ///     b = jix.compact([0], dtype=np.uint32)
    ///     assert jix.leading_zeros(b).numpy()[0] == 32
    ///     ```
    leading_zeros,
    LeadingZeros,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op1!(
    /// Counts the number of trailing zero bits in each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Counts zeros from the least-significant bit up to (but not including) the first
    /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for a
    /// `uint32` zero).
    ///
    /// Args:
    ///     array: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///     dtype. No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0x00010000, 0x80000000, 0x00000001], dtype=np.uint32)
    ///     result = jix.trailing_zeros(a)
    ///     assert np.array_equal(result.numpy(), [16, 31, 0])
    ///
    ///     # Zero returns the bit width of the type (32 for uint32).
    ///     b = jix.compact([0], dtype=np.uint32)
    ///     assert jix.trailing_zeros(b).numpy()[0] == 32
    ///     ```
    trailing_zeros,
    TrailingZeros,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op1!(
    /// Reverses the byte order of each element.
    ///
    /// Supported dtypes: `i16`, `i32`, `i64`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// Swaps the bytes of each element (e.g. converts between big-endian and little-endian
    /// representation). Single-byte types (`i8`, `u8`) are not supported since swapping
    /// one byte is a no-op.
    ///
    /// Args:
    ///     array: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///     the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0x12345678], dtype=np.uint32)
    ///     result = jix.swap_bytes(a)
    ///     assert result.numpy()[0] == 0x78563412
    ///     ```
    swap_bytes,
    SwapBytes,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);

define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// The most-significant bit becomes the least-significant and vice versa.
    ///
    /// Args:
    ///     array: May be anything that jix.asarray() accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
    ///     the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0b00000001, 0b10000000, 0b10101010], dtype=np.uint8)
    ///     result = jix.reverse_bits(a)
    ///     assert np.array_equal(result.numpy(), [0b10000000, 0b00000001, 0b01010101])
    ///     ```
    reverse_bits,
    ReverseBits,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64],
        None
    }
);
