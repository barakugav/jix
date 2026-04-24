#[allow(unused_imports)]
use crate::dtype::f16;
use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::op1::define_op1;
use crate::ops::op2::define_op2;
use crate::storage::ArrayStorage;
use crate::Array;

define_op2!(
    /// Element-wise logical AND of two arrays.
    ///
    /// Supported dtypes: all numeric types, `bool`, and `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// Each element is first cast to `bool` (zero → `false`, any non-zero value → `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical AND is applied. Returns `true` only when both elements
    /// are truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0i32, 1, 0, 5];
    /// let b = ndarray::array![1i32, 1, 0, 0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.logical_and(zb).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false, false]);
    ///
    /// // Works on bool arrays directly.
    /// let c = ndarray::array![true, false, true];
    /// let d = ndarray::array![true, true, false];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.logical_and(zd).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    LogicalAnd,
    LogicalAndKernel,
    |a, b| crate::ops::astype::cast::<_, bool>(a) && crate::ops::astype::cast::<_, bool>(b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
    output_type = bool
);
define_op2!(
    /// Element-wise logical OR of two arrays.
    ///
    /// Supported dtypes: all numeric types, `bool`, and `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// Each element is first cast to `bool` (zero → `false`, any non-zero value → `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical OR is applied. Returns `true` when at least one element
    /// is truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0i32, 1, 0, 5];
    /// let b = ndarray::array![0i32, 0, 0, 0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.logical_or(zb).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false, true]);
    ///
    /// // Works on bool arrays directly.
    /// let c = ndarray::array![true, false, false];
    /// let d = ndarray::array![false, true, false];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.logical_or(zd).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    LogicalOr,
    LogicalOrKernel,
    |a, b| crate::ops::astype::cast::<_, bool>(a) || crate::ops::astype::cast::<_, bool>(b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
    output_type = bool
);
define_op2!(
    /// Element-wise logical XOR of two arrays.
    ///
    /// Supported dtypes: all numeric types, `bool`, and `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// Each element is first cast to `bool` (zero → `false`, any non-zero value → `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical XOR is applied. Returns `true` when exactly one element
    /// is truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0i32, 1, 0, 5];
    /// let b = ndarray::array![0i32, 1, 1, 0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.logical_xor(zb).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, false, true, true]);
    ///
    /// // Works on bool arrays directly.
    /// let c = ndarray::array![true, false, true];
    /// let d = ndarray::array![true, false, false];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.logical_xor(zd).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, false, true]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    LogicalXor,
    LogicalXorKernel,
    |a, b| crate::ops::astype::cast::<_, bool>(a) ^ crate::ops::astype::cast::<_, bool>(b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
    output_type = bool
);
define_op1!(
    /// Element-wise logical NOT.
    ///
    /// Supported dtypes: all numeric types, `bool`, and `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// Each element is first cast to `bool` (zero → `false`, any non-zero value → `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then negated. Returns `true` for zero (falsy) elements and `false` for
    /// non-zero (truthy) elements.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0i32, 1, -3, 0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.logical_not().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, true]);
    ///
    /// // Works on bool arrays directly.
    /// let b = ndarray::array![true, false, true];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.logical_not().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    LogicalNot,
    LogicalNotKernel,
    |a| !crate::ops::astype::cast::<_, bool>(a),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
    output_type = bool
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
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b1100u8, 0b1010, 0b1111];
    /// let b = ndarray::array![0b1010u8, 0b0101, 0b0000];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.bitwise_and(zb).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b1000, 0b0000, 0b0000]);
    ///
    /// // Mask out the lower nibble.
    /// let c = ndarray::array![0xABu8, 0xCDu8];
    /// let d = ndarray::array![0xF0u8, 0xF0u8];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.bitwise_and(zd).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xA0, 0xC0]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    BitwiseAnd,
    BitwiseAndKernel,
    |a, b| a & b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
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
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b1100u8, 0b1010, 0b0000];
    /// let b = ndarray::array![0b1010u8, 0b0101, 0b1111];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.bitwise_or(zb).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b1110, 0b1111, 0b1111]);
    ///
    /// // Set a specific bit pattern.
    /// let c = ndarray::array![0x0Fu8, 0x00u8];
    /// let d = ndarray::array![0xF0u8, 0xF0u8];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.bitwise_or(zd).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xFF, 0xF0]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    BitwiseOr,
    BitwiseOrKernel,
    |a, b| a | b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
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
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b1100u8, 0b1010, 0b1111];
    /// let b = ndarray::array![0b1010u8, 0b1010, 0b1111];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.bitwise_xor(zb).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b0110, 0b0000, 0b0000]);
    ///
    /// // Toggle bits using a mask.
    /// let c = ndarray::array![0xFFu8, 0x0Fu8];
    /// let d = ndarray::array![0x0Fu8, 0x0Fu8];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.bitwise_xor(zd).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xF0, 0x00]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    BitwiseXor,
    BitwiseXorKernel,
    |a, b| a ^ b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
);
define_op1!(
    /// Element-wise bitwise NOT (one's complement).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`.
    /// Output dtype and shape equal the input.
    ///
    /// Flips every bit. For `bool` this is equivalent to logical NOT.
    /// For signed integers the result is `-(x + 1)` (e.g. `!0i32 == -1`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b00001111u8, 0b11110000u8, 0u8];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.bitwise_not().to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b11110000, 0b00001111, 0xFF]);
    ///
    /// // For bool arrays, bitwise NOT is equivalent to logical NOT.
    /// let b = ndarray::array![true, false];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.bitwise_not().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    BitwiseNot,
    BitwiseNotKernel,
    |a| !a,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// Shifts the bits of each element of `a` left by the corresponding value in `b`.
    /// Vacated bits are filled with zeros. Shifting by a value greater than or equal to
    /// the bit width of the type is a panic in debug builds and implementation-defined
    /// in release builds.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b00000001u8, 0b00000010u8, 0b00000100u8];
    /// let b = ndarray::array![1u8, 2, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.bitwise_shift_left(zb).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b00000010, 0b00001000, 0b00100000]);
    ///
    /// // Signed arithmetic left shift: sign bit is lost if shifted out.
    /// let c = ndarray::array![1i8, -1i8];
    /// let d = ndarray::array![3i8, 1i8];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.bitwise_shift_left(zd).to_ndarray::<i8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[8, -2]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    BitwiseShiftLeft,
    BitwiseShiftLeftKernel,
    |a, b| a << b,
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = "same"
);
define_op2!(
    /// Element-wise right shift (`a >> b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// For **unsigned** types this is a logical shift: vacated bits are filled with zeros.
    /// For **signed** types this is an arithmetic shift: vacated bits are filled with the
    /// sign bit (the result preserves the sign of the value).
    /// Shifting by a value greater than or equal to the bit width of the type is a panic
    /// in debug builds and implementation-defined in release builds.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b10000000u8, 0b00100000u8, 0b00001000u8];
    /// let b = ndarray::array![1u8, 2, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.bitwise_shift_right(zb).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b01000000, 0b00001000, 0b00000001]);
    ///
    /// // Signed arithmetic right shift: vacated bits are filled with the sign bit.
    /// let c = ndarray::array![-8i8, -1i8];
    /// let d = ndarray::array![2i8, 1i8];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.bitwise_shift_right(zd).to_ndarray::<i8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-2, -1]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    BitwiseShiftRight,
    BitwiseShiftRightKernel,
    |a, b| a >> b,
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = "same"
);
define_op1!(
    /// Counts the number of set bits (`1`s) in each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Also known as the population count or Hamming weight. For signed integers the
    /// bit representation (including the sign bit) is used.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b00001111u8, 0b11001100u8, 0b11111111u8];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.count_ones().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 4, 8]);
    ///
    /// // Zero has no set bits.
    /// let b = ndarray::array![0u8, 0u8];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.count_ones().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0, 0]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    CountOnes,
    CountOnesKernel,
    |a| a.count_ones(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
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
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b11110000u8, 0b00001111u8, 0b11111111u8];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.count_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 4, 0]);
    ///
    /// // Zero has all bits unset: count_zeros == bit width.
    /// let b = ndarray::array![0u8];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.count_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 8); // u8 has 8 bits
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    CountZeros,
    CountZerosKernel,
    |a| a.count_zeros(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    /// Counts the number of leading zero bits in each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Counts zeros from the most-significant bit down to (but not including) the first
    /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
    /// `0u32`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0x00010000u32, 0x80000000u32, 0x00000001u32];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.leading_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[15, 0, 31]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = ndarray::array![0u32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.leading_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    LeadingZeros,
    LeadingZerosKernel,
    |a| a.leading_zeros(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    /// Counts the number of trailing zero bits in each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype is `u32`. The output shape equals the input shape.
    ///
    /// Counts zeros from the least-significant bit up to (but not including) the first
    /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
    /// `0u32`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0x00010000u32, 0x80000000u32, 0x00000001u32];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.trailing_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[16, 31, 0]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = ndarray::array![0u32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.trailing_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    TrailingZeros,
    TrailingZerosKernel,
    |a| a.trailing_zeros(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    /// Reverses the byte order of each element.
    ///
    /// Supported dtypes: `i16`, `i32`, `i64`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// Swaps the bytes of each element in-place (e.g. converts between big-endian and
    /// little-endian representation). Single-byte types (`i8`, `u8`) are not supported
    /// since swapping one byte is a no-op.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0x00FF0000u32, 0x0000FF00u32];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.swap_bytes().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0x0000FF00, 0x00FF0000]);
    ///
    /// // Classic endian-swap example.
    /// let b = ndarray::array![0x12345678u32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.swap_bytes().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 0x78563412u32);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    SwapBytes,
    SwapBytesKernel,
    |a| a.swap_bytes(),
    [i16, i32, i64, u16, u32, u64],
    output_type = "same"
);
define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// Output dtype and shape equal the input.
    ///
    /// The most-significant bit becomes the least-significant and vice versa.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0b00000001u8, 0b10000000u8, 0b10101010u8];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.reverse_bits().to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b10000000, 0b00000001, 0b01010101]);
    ///
    /// // Reversing bits of 0 gives 0.
    /// let b = ndarray::array![0u8];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.reverse_bits().to_ndarray::<u8>()?;
    /// assert_eq!(result[[0]], 0u8);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    ReverseBits,
    ReverseBitsKernel,
    |a| a.reverse_bits(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = "same"
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(logical_and: LogicalAnd);
    define_array_op2_method!(logical_or: LogicalOr);
    define_array_op2_method!(logical_xor: LogicalXor);
    define_array_op1_method!(logical_not: LogicalNot);
    define_array_op2_method!(bitwise_and: BitwiseAnd);
    define_array_op2_method!(bitwise_or: BitwiseOr);
    define_array_op2_method!(bitwise_xor: BitwiseXor);
    define_array_op1_method!(bitwise_not: BitwiseNot);
    define_array_op2_method!(bitwise_shift_left: BitwiseShiftLeft);
    define_array_op2_method!(bitwise_shift_right: BitwiseShiftRight);
    define_array_op1_method!(count_ones: CountOnes);
    define_array_op1_method!(count_zeros: CountZeros);
    define_array_op1_method!(leading_zeros: LeadingZeros);
    define_array_op1_method!(trailing_zeros: TrailingZeros);
    define_array_op1_method!(swap_bytes: SwapBytes);
    define_array_op1_method!(reverse_bits: ReverseBits);
}
