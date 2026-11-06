use pyo3::prelude::*;

use crate::ops::common::{define_op1, define_op1_table, Operand, Precision, Rank};
use crate::Array;

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
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///         dtype. No computation occurs until the result is read.
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
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///         dtype. No computation occurs until the result is read.
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
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///         dtype. No computation occurs until the result is read.
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
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array` and an unsigned integer output
    ///         dtype. No computation occurs until the result is read.
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

/// Reverses the byte order of each element.
///
/// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
/// Output dtype and shape equal the input.
///
/// Swaps the bytes of each element (e.g. converts between big-endian and little-endian
/// representation). For single-byte types (`i8`, `u8`, `bool`) this is a no-op: a
/// [`jix.Array`][jix.Array] input is returned as is (the same object).
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
///     a = jix.compact([0x12345678], dtype=np.uint32)
///     result = jix.byteswap(a)
///     assert result.numpy()[0] == 0x78563412
///     ```
#[pyo3::pyfunction]
pub fn byteswap<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let array = Operand::from_any(array)?;

    // swapping the bytes of a single-byte element is a no-op
    if matches!(
        array.rank_precision(),
        Some((Rank::UInt | Rank::Int, Some(Precision::P1)) | (Rank::Bool, _))
    ) {
        return array.into_py_array(py);
    }

    let dispatch_table = define_op1_table!(
        byteswap,
        SwapBytes,
        dispatch = { [u16, i16, u32, i32, u64, i64], None }
    );
    let res = dispatch_table.dispatch1(array)?;
    Bound::new(py, Array::from_core(res))
}

define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// The most-significant bit becomes the least-significant and vice versa.
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
