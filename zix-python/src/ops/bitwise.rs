use crate::ops::{define_op1, define_op2};

define_op2!(
    /// Element-wise logical AND of two arrays.
    ///
    /// Supported dtypes: all integer types (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`,
    /// `u64`), `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, and `bool`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero value -> `True`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical AND is applied.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0, 1, 0, 5], dtype=np.int32)
    /// b = zix.compact([1, 1, 0, 0], dtype=np.int32)
    /// result = zix.logical_and(a, b)
    /// assert np.array_equal(result.numpy(), [False, True, False, False])
    /// ```
    logical_and,
    LogicalAnd
);

define_op2!(
    /// Element-wise logical OR of two arrays.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`. The output shape equals the input
    /// shape.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero value -> `True`),
    /// then the logical OR is applied. Returns `True` when at least one element is truthy.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0, 1, 0, 5], dtype=np.int32)
    /// b = zix.compact([0, 0, 0, 0], dtype=np.int32)
    /// result = zix.logical_or(a, b)
    /// assert np.array_equal(result.numpy(), [False, True, False, True])
    /// ```
    logical_or,
    LogicalOr
);

define_op2!(
    /// Element-wise logical XOR of two arrays.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`. The output shape equals the input
    /// shape.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero value -> `True`),
    /// then the logical XOR is applied. Returns `True` when exactly one element is truthy.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0, 1, 0, 5], dtype=np.int32)
    /// b = zix.compact([0, 1, 1, 0], dtype=np.int32)
    /// result = zix.logical_xor(a, b)
    /// assert np.array_equal(result.numpy(), [False, False, True, True])
    /// ```
    logical_xor,
    LogicalXor
);

define_op1!(
    /// Element-wise logical NOT.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`. The output shape equals the input
    /// shape.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, any non-zero value -> `True`),
    /// then negated. Returns `True` for zero (falsy) elements and `False` for non-zero
    /// (truthy) elements.
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
    /// a = zix.compact([0, 1, -3, 0], dtype=np.int32)
    /// result = zix.logical_not(a)
    /// assert np.array_equal(result.numpy(), [True, False, False, True])
    /// ```
    logical_not,
    LogicalNot
);

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype and shape equal the input.
    ///
    /// Applies the bitwise AND to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical AND.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0b1100, 0b1010, 0b1111], dtype=np.uint8)
    /// b = zix.compact([0b1010, 0b0101, 0b0000], dtype=np.uint8)
    /// result = zix.bitwise_and(a, b)
    /// assert np.array_equal(result.numpy(), [0b1000, 0b0000, 0b0000])
    /// ```
    bitwise_and,
    BitwiseAnd
);

define_op2!(
    /// Element-wise bitwise OR of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype and shape equal the input.
    ///
    /// Applies the bitwise OR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical OR.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0b1100, 0b1010, 0b0000], dtype=np.uint8)
    /// b = zix.compact([0b1010, 0b0101, 0b1111], dtype=np.uint8)
    /// result = zix.bitwise_or(a, b)
    /// assert np.array_equal(result.numpy(), [0b1110, 0b1111, 0b1111])
    /// ```
    bitwise_or,
    BitwiseOr
);

define_op2!(
    /// Element-wise bitwise XOR of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype and shape equal the input.
    ///
    /// Applies the bitwise XOR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical XOR.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0b1100, 0b1010, 0b1111], dtype=np.uint8)
    /// b = zix.compact([0b1010, 0b1010, 0b1111], dtype=np.uint8)
    /// result = zix.bitwise_xor(a, b)
    /// assert np.array_equal(result.numpy(), [0b0110, 0b0000, 0b0000])
    /// ```
    bitwise_xor,
    BitwiseXor
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
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0b00001111, 0b11110000, 0], dtype=np.uint8)
    /// result = zix.bitwise_not(a)
    /// assert np.array_equal(result.numpy(), [0b11110000, 0b00001111, 0xFF])
    /// ```
    bitwise_not,
    BitwiseNot
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// Shifts the bits of each element of `a` left by the corresponding value in `b`.
    /// Vacated bits are filled with zeros. Shifting by a value greater than or equal to the
    /// bit width of the type produces zero.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0b00000001, 0b00000010, 0b00000100], dtype=np.uint8)
    /// b = zix.compact([1, 2, 3], dtype=np.uint8)
    /// result = zix.bitwise_shift_left(a, b)
    /// assert np.array_equal(result.numpy(), [0b00000010, 0b00001000, 0b00100000])
    /// ```
    bitwise_shift_left,
    BitwiseShiftLeft
);

define_op2!(
    /// Element-wise right shift (`a >> b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// For **unsigned** types this is a logical shift: vacated bits are filled with zeros.
    /// For **signed** types this is an arithmetic shift: vacated bits are filled with the
    /// sign bit (the result preserves the sign). Shifting by a value greater than or equal
    /// to the bit width produces zero (unsigned) or the sign-extended value (signed).
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
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
    /// a = zix.compact([0b10000000, 0b00100000, 0b00001000], dtype=np.uint8)
    /// b = zix.compact([1, 2, 3], dtype=np.uint8)
    /// result = zix.bitwise_shift_right(a, b)
    /// assert np.array_equal(result.numpy(), [0b01000000, 0b00001000, 0b00000001])
    /// ```
    bitwise_shift_right,
    BitwiseShiftRight
);

define_op2!(
    /// Element-wise bitwise left rotation.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// Rotates the bits of each element of `a` left by the corresponding value in `b`
    /// (interpreted as `u32`). Unlike a left shift, bits shifted out of the most-significant
    /// position wrap around to the least-significant position, so no bits are lost.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts.
    ///
    /// This function deviates from numpy (which has no equivalent) in that both inputs
    /// must have the same dtype and shape.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0b10000001, 0b00000001, 0b11110000], dtype=np.uint8)
    /// b = zix.compact([1, 3, 4], dtype=np.uint8)
    /// result = zix.bitwise_rotate_left(a, b)
    /// assert np.array_equal(result.numpy(), [0b00000011, 0b00001000, 0b00001111])
    /// ```
    bitwise_rotate_left,
    BitwiseRotateLeft
);

define_op2!(
    /// Element-wise bitwise right rotation.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// Rotates the bits of each element of `a` right by the corresponding value in `b`
    /// (interpreted as `u32`). Unlike a right shift, bits shifted out of the least-significant
    /// position wrap around to the most-significant position, so no bits are lost.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts.
    ///
    /// This function deviates from numpy (which has no equivalent) in that both inputs
    /// must have the same dtype and shape.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0b10000001, 0b00001000, 0b00001111], dtype=np.uint8)
    /// b = zix.compact([1, 3, 4], dtype=np.uint8)
    /// result = zix.bitwise_rotate_right(a, b)
    /// assert np.array_equal(result.numpy(), [0b11000000, 0b00000001, 0b11110000])
    /// ```
    bitwise_rotate_right,
    BitwiseRotateRight
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
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0b00001111, 0b11001100, 0b11111111], dtype=np.uint8)
    /// result = zix.count_ones(a)
    /// assert np.array_equal(result.numpy(), [4, 4, 8])
    /// ```
    count_ones,
    CountOnes
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
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0b11110000, 0b00001111, 0b11111111], dtype=np.uint8)
    /// result = zix.count_zeros(a)
    /// assert np.array_equal(result.numpy(), [4, 4, 0])
    ///
    /// # Zero has all bits unset: count_zeros == bit width.
    /// b = zix.compact([0], dtype=np.uint8)
    /// assert zix.count_zeros(b).numpy()[0] == 8  # u8 has 8 bits
    /// ```
    count_zeros,
    CountZeros
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
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0x00010000, 0x80000000, 0x00000001], dtype=np.uint32)
    /// result = zix.leading_zeros(a)
    /// assert np.array_equal(result.numpy(), [15, 0, 31])
    ///
    /// # Zero returns the bit width of the type (32 for uint32).
    /// b = zix.compact([0], dtype=np.uint32)
    /// assert zix.leading_zeros(b).numpy()[0] == 32
    /// ```
    leading_zeros,
    LeadingZeros
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
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0x00010000, 0x80000000, 0x00000001], dtype=np.uint32)
    /// result = zix.trailing_zeros(a)
    /// assert np.array_equal(result.numpy(), [16, 31, 0])
    ///
    /// # Zero returns the bit width of the type (32 for uint32).
    /// b = zix.compact([0], dtype=np.uint32)
    /// assert zix.trailing_zeros(b).numpy()[0] == 32
    /// ```
    trailing_zeros,
    TrailingZeros
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
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([0x12345678], dtype=np.uint32)
    /// result = zix.swap_bytes(a)
    /// assert result.numpy()[0] == 0x78563412
    /// ```
    swap_bytes,
    SwapBytes
);

define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// The most-significant bit becomes the least-significant and vice versa.
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
    /// a = zix.compact([0b00000001, 0b10000000, 0b10101010], dtype=np.uint8)
    /// result = zix.reverse_bits(a)
    /// assert np.array_equal(result.numpy(), [0b10000000, 0b00000001, 0b01010101])
    /// ```
    reverse_bits,
    ReverseBits
);
