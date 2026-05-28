use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::op2::define_op2;
use crate::ops::{define_op1, define_op2_rhs_fixed};
#[allow(unused_imports)]
use crate::scalar::{f16, Complex};
use crate::{Array, ArrayStorage};

pub(crate) mod _traits {
    use crate::scalar::traits_util::{define_op1_trait, define_op2_trait};
    #[allow(unused_imports)]
    use crate::scalar::{f16, Complex};

    define_op2_trait!(
        LogicalAnd,
        logical_and,
        |a, b| <_ as crate::scalar::Cast<bool>>::cast(a) && <_ as crate::scalar::Cast<bool>>::cast(b),
        [pairs_of[i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex<f32>, Complex<f64>, bool] => bool]
        // TODO: f16
    );

    define_op2_trait!(
        LogicalOr,
        logical_or,
        |a, b| <_ as crate::scalar::Cast<bool>>::cast(a) || <_ as crate::scalar::Cast<bool>>::cast(b),
        [pairs_of[i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex<f32>, Complex<f64>, bool] => bool]
        // TODO: f16
    );

    define_op2_trait!(
        LogicalXor,
        logical_xor,
        |a, b| <_ as crate::scalar::Cast<bool>>::cast(a) ^ <_ as crate::scalar::Cast<bool>>::cast(b),
        [pairs_of[i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex<f32>, Complex<f64>, bool] => bool]
        // TODO: f16
    );

    define_op1_trait!(
        LogicalNot,
        logical_not,
        |a| !<_ as crate::scalar::Cast<bool>>::cast(a),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex<f32>, Complex<f64>, bool] => bool
        // TODO: f16
    );
}

define_op2!(
    /// Element-wise logical AND of two arrays.
    ///
    /// Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `false`, any non-zero value -> `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical AND is applied. Returns `true` only when both elements
    /// are truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::logical_and()`](crate::Array::logical_and).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0i32, 1, 0, 5])?;
    /// let b = Array::compact_array(&array![1i32, 1, 0, 0])?;
    /// let result = a.logical_and(b).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false, false]);
    ///
    /// // Works on bool arrays directly.
    /// let c = Array::compact_array(&array![true, false, true])?;
    /// let d = Array::compact_array(&array![true, true, false])?;
    /// let result = c.logical_and(d).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    LogicalAnd,
    LogicalAndKernel,
    <crate::scalar::LogicalAnd>::logical_and(a, b),
);

define_op2!(
    /// Element-wise logical OR of two arrays.
    ///
    /// Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `false`, any non-zero value -> `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical OR is applied. Returns `true` when at least one element
    /// is truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::logical_or()`](crate::Array::logical_or).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0i32, 1, 0, 5])?;
    /// let b = Array::compact_array(&array![0i32, 0, 0, 0])?;
    /// let result = a.logical_or(b).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false, true]);
    ///
    /// // Works on bool arrays directly.
    /// let c = Array::compact_array(&array![true, false, false])?;
    /// let d = Array::compact_array(&array![false, true, false])?;
    /// let result = c.logical_or(d).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    LogicalOr,
    LogicalOrKernel,
    <crate::scalar::LogicalOr>::logical_or(a, b),
);

define_op2!(
    /// Element-wise logical XOR of two arrays.
    ///
    /// Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `false`, any non-zero value -> `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then the logical XOR is applied. Returns `true` when exactly one element
    /// is truthy.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::logical_xor()`](crate::Array::logical_xor).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0i32, 1, 0, 5])?;
    /// let b = Array::compact_array(&array![0i32, 1, 1, 0])?;
    /// let result = a.logical_xor(b).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, false, true, true]);
    ///
    /// // Works on bool arrays directly.
    /// let c = Array::compact_array(&array![true, false, true])?;
    /// let d = Array::compact_array(&array![true, false, false])?;
    /// let result = c.logical_xor(d).to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, false, true]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    LogicalXor,
    LogicalXorKernel,
    <crate::scalar::LogicalXor>::logical_xor(a, b),
);

define_op1!(
    /// Element-wise logical NOT.
    ///
    /// Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `false`, any non-zero value -> `true`;
    /// for `bool` this is the identity; for complex, non-zero means at least one component
    /// is non-zero), then negated. Returns `true` for zero (falsy) elements and `false` for
    /// non-zero (truthy) elements.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::logical_not()`](crate::Array::logical_not).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0i32, 1, -3, 0])?;
    /// let result = a.logical_not().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, true]);
    ///
    /// // Works on bool arrays directly.
    /// let b = Array::compact_array(&array![true, false, true])?;
    /// let result = b.logical_not().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    LogicalNot,
    LogicalNotKernel,
    <crate::ops::bitwise::_traits::LogicalNot>::logical_not,
);

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// Applies the bitwise AND to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical AND.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_and()`](crate::Array::bitwise_and).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b1100u8, 0b1010, 0b1111])?;
    /// let b = Array::compact_array(&array![0b1010u8, 0b0101, 0b0000])?;
    /// let result = a.bitwise_and(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b1000, 0b0000, 0b0000]);
    ///
    /// // Mask out the lower nibble.
    /// let c = Array::compact_array(&array![0xABu8, 0xCDu8])?;
    /// let d = Array::compact_array(&array![0xF0u8, 0xF0u8])?;
    /// let result = c.bitwise_and(d).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xA0, 0xC0]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseAnd,
    BitwiseAndKernel,
    <core::ops::BitAnd>::bitand(a, b),
);
define_op2!(
    /// Element-wise bitwise OR of two arrays.
    ///
    /// Applies the bitwise OR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical OR.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_or()`](crate::Array::bitwise_or).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b1100u8, 0b1010, 0b0000])?;
    /// let b = Array::compact_array(&array![0b1010u8, 0b0101, 0b1111])?;
    /// let result = a.bitwise_or(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b1110, 0b1111, 0b1111]);
    ///
    /// // Set a specific bit pattern.
    /// let c = Array::compact_array(&array![0x0Fu8, 0x00u8])?;
    /// let d = Array::compact_array(&array![0xF0u8, 0xF0u8])?;
    /// let result = c.bitwise_or(d).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xFF, 0xF0]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseOr,
    BitwiseOrKernel,
    <core::ops::BitOr>::bitor(a, b),
);
define_op2!(
    /// Element-wise bitwise XOR of two arrays.
    ///
    /// Applies the bitwise XOR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical XOR.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_xor()`](crate::Array::bitwise_xor).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b1100u8, 0b1010, 0b1111])?;
    /// let b = Array::compact_array(&array![0b1010u8, 0b1010, 0b1111])?;
    /// let result = a.bitwise_xor(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b0110, 0b0000, 0b0000]);
    ///
    /// // Toggle bits using a mask.
    /// let c = Array::compact_array(&array![0xFFu8, 0x0Fu8])?;
    /// let d = Array::compact_array(&array![0x0Fu8, 0x0Fu8])?;
    /// let result = c.bitwise_xor(d).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xF0, 0x00]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseXor,
    BitwiseXorKernel,
    <core::ops::BitXor>::bitxor(a, b),
);

define_op1!(
    /// Element-wise bitwise NOT (one's complement).
    ///
    /// Flips every bit. For `bool` this is equivalent to logical NOT.
    /// For signed integers the result is `-(x + 1)` (e.g. `!0i32 == -1`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_not()`](crate::Array::bitwise_not).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b00001111u8, 0b11110000u8, 0u8])?;
    /// let result = a.bitwise_not().to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b11110000, 0b00001111, 0xFF]);
    ///
    /// // For bool arrays, bitwise NOT is equivalent to logical NOT.
    /// let b = Array::compact_array(&array![true, false])?;
    /// let result = b.bitwise_not().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseNot,
    BitwiseNotKernel,
    <core::ops::Not>::not,
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// Shifts the bits of each element of `a` left by the corresponding value in `b`.
    /// Vacated bits are filled with zeros. Shifting by a value greater than or equal to
    /// the bit width of the type is defined to produce zero, matching `u32::unbounded_shl`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_shift_left()`](crate::Array::bitwise_shift_left).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b00000001u8, 0b00000010u8, 0b00000100u8])?;
    /// let b = Array::compact_array(&array![1u8, 2, 3])?;
    /// let result = a.bitwise_shift_left(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b00000010, 0b00001000, 0b00100000]);
    ///
    /// // Signed arithmetic left shift: sign bit is lost if shifted out.
    /// let c = Array::compact_array(&array![1i8, -1i8])?;
    /// let d = Array::compact_array(&array![3i8, 1i8])?;
    /// let result = c.bitwise_shift_left(d).to_ndarray::<i8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[8, -2]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseShiftLeft,
    BitwiseShiftLeftKernel,
    <core::ops::Shl>::shl(a, b),
);

define_op2!(
    /// Element-wise right shift (`a >> b`).
    ///
    /// For **unsigned** types this is a logical shift: vacated bits are filled with zeros.
    /// For **signed** types this is an arithmetic shift: vacated bits are filled with the
    /// sign bit (the result preserves the sign of the value).
    /// Shifting by a value greater than or equal to the bit width of the type is
    /// defined to produce zero, matching `u32::unbounded_shr`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_shift_right()`](crate::Array::bitwise_shift_right).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b10000000u8, 0b00100000u8, 0b00001000u8])?;
    /// let b = Array::compact_array(&array![1u8, 2, 3])?;
    /// let result = a.bitwise_shift_right(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b01000000, 0b00001000, 0b00000001]);
    ///
    /// // Signed arithmetic right shift: vacated bits are filled with the sign bit.
    /// let c = Array::compact_array(&array![-8i8, -1i8])?;
    /// let d = Array::compact_array(&array![2i8, 1i8])?;
    /// let result = c.bitwise_shift_right(d).to_ndarray::<i8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-2, -1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseShiftRight,
    BitwiseShiftRightKernel,
    <core::ops::Shr>::shr(a, b),
);
define_op2_rhs_fixed!(
    /// Element-wise bitwise left rotation (`a.rotate_left(b as u32)`).
    ///
    /// Rotates the bits of each element of `a` left by the corresponding value in `b`
    /// cast to `u32`. Unlike a left shift, bits shifted out of the most-significant
    /// position wrap around to the least-significant position, so no bits are lost.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_rotate_left()`](crate::Array::bitwise_rotate_left).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b10000001u8, 0b00000001u8, 0b11110000u8])?;
    /// let b = Array::compact_array(&array![1u32, 3, 4])?;
    /// let result = a.bitwise_rotate_left(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b00000011, 0b00001000, 0b00001111]);
    ///
    /// // Rotating by 0 is a no-op.
    /// let c = Array::compact_array(&array![0xABu8])?;
    /// let d = Array::compact_array(&array![0u32])?;
    /// let result = c.bitwise_rotate_left(d).to_ndarray::<u8>()?;
    /// assert_eq!(result[[0]], 0xABu8);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseRotateLeft,
    BitwiseRotateLeftKernel,
    <num_traits::PrimInt>::rotate_left(a, b),
    rhs = u32,
    type Output<T1> = T1,
    type Output<S1> = S1::Item,
);

define_op2_rhs_fixed!(
    /// Element-wise bitwise right rotation (`a.rotate_right(b as u32)`).
    ///
    /// Rotates the bits of each element of `a` right by the corresponding value in `b`
    /// cast to `u32`. Unlike a right shift, bits shifted out of the least-significant
    /// position wrap around to the most-significant position, so no bits are lost.
    /// The rotation amount is taken modulo the bit width of the type.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_rotate_right()`](crate::Array::bitwise_rotate_right).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b10000001u8, 0b00001000u8, 0b00001111u8])?;
    /// let b = Array::compact_array(&array![1u32, 3, 4])?;
    /// let result = a.bitwise_rotate_right(b).to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b11000000, 0b00000001, 0b11110000]);
    ///
    /// // Rotating by 0 is a no-op.
    /// let c = Array::compact_array(&array![0xABu8])?;
    /// let d = Array::compact_array(&array![0u32])?;
    /// let result = c.bitwise_rotate_right(d).to_ndarray::<u8>()?;
    /// assert_eq!(result[[0]], 0xABu8);
    /// # Ok::<(), zix::Error>(())
    /// ```
    BitwiseRotateRight,
    BitwiseRotateRightKernel,
    <num_traits::PrimInt>::rotate_right(a, b),
    rhs = u32,
    type Output<T1> = T1,
    type Output<S1> = S1::Item,
);
define_op1!(
    /// Counts the number of set bits (`1`s) in each element.
    ///
    /// Output dtype is `u32`.
    ///
    /// Also known as the population count or Hamming weight. For signed integers the
    /// bit representation (including the sign bit) is used.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::count_ones()`](crate::Array::count_ones).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b00001111u8, 0b11001100u8, 0b11111111u8])?;
    /// let result = a.count_ones().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 4, 8]);
    ///
    /// // Zero has no set bits.
    /// let b = Array::compact_array(&array![0u8, 0u8])?;
    /// let result = b.count_ones().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0, 0]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    CountOnes,
    CountOnesKernel,
    <num_traits::PrimInt>::count_ones,
    type Output = u32,
);
define_op1!(
    /// Counts the number of unset bits (`0`s) in each element.
    ///
    /// Output dtype is `u32`.
    ///
    /// Equivalent to `bit_width - count_ones`. For signed integers the full bit
    /// representation (including the sign bit) is used.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::count_zeros()`](crate::Array::count_zeros).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b11110000u8, 0b00001111u8, 0b11111111u8])?;
    /// let result = a.count_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 4, 0]);
    ///
    /// // Zero has all bits unset: count_zeros == bit width.
    /// let b = Array::compact_array(&array![0u8])?;
    /// let result = b.count_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 8); // u8 has 8 bits
    /// # Ok::<(), zix::Error>(())
    /// ```
    CountZeros,
    CountZerosKernel,
    <num_traits::PrimInt>::count_zeros,
    type Output = u32,
);
define_op1!(
    /// Counts the number of leading zero bits in each element.
    ///
    /// Output dtype is `u32`.
    ///
    /// Counts zeros from the most-significant bit down to (but not including) the first
    /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
    /// `0u32`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::leading_zeros()`](crate::Array::leading_zeros).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0x00010000u32, 0x80000000u32, 0x00000001u32])?;
    /// let result = a.leading_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[15, 0, 31]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = Array::compact_array(&array![0u32])?;
    /// let result = b.leading_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), zix::Error>(())
    /// ```
    LeadingZeros,
    LeadingZerosKernel,
    <num_traits::PrimInt>::leading_zeros,
    type Output = u32,
);
define_op1!(
    /// Counts the number of trailing zero bits in each element.
    ///
    /// Output dtype is `u32`.
    ///
    /// Counts zeros from the least-significant bit up to (but not including) the first
    /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
    /// `0u32`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::trailing_zeros()`](crate::Array::trailing_zeros).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0x00010000u32, 0x80000000u32, 0x00000001u32])?;
    /// let result = a.trailing_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[16, 31, 0]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = Array::compact_array(&array![0u32])?;
    /// let result = b.trailing_zeros().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), zix::Error>(())
    /// ```
    TrailingZeros,
    TrailingZerosKernel,
    <num_traits::PrimInt>::trailing_zeros,
    type Output = u32,
);
define_op1!(
    /// Reverses the byte order of each element.
    ///
    /// Swaps the bytes of each element in-place (e.g. converts between big-endian and
    /// little-endian representation). Single-byte types (`i8`, `u8`) are not supported
    /// since swapping one byte is a no-op.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::swap_bytes()`](crate::Array::swap_bytes).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0x00FF0000u32, 0x0000FF00u32])?;
    /// let result = a.swap_bytes().to_ndarray::<u32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0x0000FF00, 0x00FF0000]);
    ///
    /// // Classic endian-swap example.
    /// let b = Array::compact_array(&array![0x12345678u32])?;
    /// let result = b.swap_bytes().to_ndarray::<u32>()?;
    /// assert_eq!(result[[0]], 0x78563412u32);
    /// # Ok::<(), zix::Error>(())
    /// ```
    SwapBytes,
    SwapBytesKernel,
    <num_traits::PrimInt>::swap_bytes,
    type Output<T> = T,
);
define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// The most-significant bit becomes the least-significant and vice versa.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::reverse_bits()`](crate::Array::reverse_bits).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0b00000001u8, 0b10000000u8, 0b10101010u8])?;
    /// let result = a.reverse_bits().to_ndarray::<u8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b10000000, 0b00000001, 0b01010101]);
    ///
    /// // Reversing bits of 0 gives 0.
    /// let b = Array::compact_array(&array![0u8])?;
    /// let result = b.reverse_bits().to_ndarray::<u8>()?;
    /// assert_eq!(result[[0]], 0u8);
    /// # Ok::<(), zix::Error>(())
    /// ```
    ReverseBits,
    ReverseBitsKernel,
    <num_traits::PrimInt>::reverse_bits,
    type Output<T> = T,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(logical_and: LogicalAnd, crate::scalar::LogicalAnd);
    define_array_op2_method!(logical_or: LogicalOr, crate::scalar::LogicalOr);
    define_array_op2_method!(logical_xor: LogicalXor, crate::scalar::LogicalXor);
    define_array_op1_method!(logical_not: LogicalNot, crate::scalar::LogicalNot);
    define_array_op2_method!(bitwise_and: BitwiseAnd, core::ops::BitAnd);
    define_array_op2_method!(bitwise_or: BitwiseOr, core::ops::BitOr);
    define_array_op2_method!(bitwise_xor: BitwiseXor, core::ops::BitXor);
    define_array_op1_method!(bitwise_not: BitwiseNot, core::ops::Not);
    define_array_op2_method!(bitwise_shift_left: BitwiseShiftLeft, core::ops::Shl);
    define_array_op2_method!(bitwise_shift_right: BitwiseShiftRight, core::ops::Shr);
    define_array_op2_method!(bitwise_rotate_left: BitwiseRotateLeft, num_traits::PrimInt, fixed_lhs_type = u32);
    define_array_op2_method!(bitwise_rotate_right: BitwiseRotateRight, num_traits::PrimInt, fixed_lhs_type = u32);
    define_array_op1_method!(count_ones: CountOnes, num_traits::PrimInt, fixed_output_type = true);
    define_array_op1_method!(count_zeros: CountZeros, num_traits::PrimInt, fixed_output_type = true);
    define_array_op1_method!(leading_zeros: LeadingZeros, num_traits::PrimInt, fixed_output_type = true);
    define_array_op1_method!(trailing_zeros: TrailingZeros, num_traits::PrimInt, fixed_output_type = true);
    define_array_op1_method!(swap_bytes: SwapBytes, num_traits::PrimInt, fixed_output_type = true);
    define_array_op1_method!(reverse_bits: ReverseBits, num_traits::PrimInt, fixed_output_type = true);
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;
    use crate::ops::op1::tests::test_op1;
    use crate::ops::op2::tests::test_op2;

    // any_strategy: need zeros in the sample to exercise the true branch of logical_not.
    // Reference: a == Default::default() is equivalent to !cast::<_, bool>(a) for all types.
    test_op1!(
        logical_not,
        |a| a == Default::default(),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        logical_op_strategy,
        // #[cfg(feature = "half")]
        // [f16], // TODO
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    // bitwise_not: !a, full range is valid (no overflow for bitwise complement)
    test_op1!(
        bitwise_not,
        |a| !a,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    // bit-counting ops: full range is valid, output is u32
    test_op1!(
        count_ones,
        |a| a.count_ones(),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op1!(
        count_zeros,
        |a| a.count_zeros(),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op1!(
        leading_zeros,
        |a| a.leading_zeros(),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op1!(
        trailing_zeros,
        |a| a.trailing_zeros(),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    // byte/bit permutation ops: same output type, full range is valid
    test_op1!(
        swap_bytes,
        |a| a.swap_bytes(),
        [i16, i32, i64, u16, u32, u64],
        any_strategy
    );
    test_op1!(
        reverse_bits,
        |a| a.reverse_bits(),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );

    // bitwise_and/or/xor: same output type, full range valid
    test_op2!(
        bitwise_and,
        |a, b| a & b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        bitwise_or,
        |a, b| a | b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        bitwise_xor,
        |a, b| a ^ b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );

    // shift ops: shift amount b must be in [0, bit_width) to avoid debug panic
    test_op2!(
        bitwise_shift_left,
        |a, b| a.unbounded_shl(b as u32),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        shift_safe_strategy
    );
    test_op2!(
        bitwise_shift_right,
        |a, b| a.unbounded_shr(b as u32),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        shift_safe_strategy
    );

    // TODO
    // rotate ops: rotation wraps modulo bit width, so any value of b is valid
    // test_op2!(
    //     bitwise_rotate_left,
    //     |a, b| a.rotate_left(b as u32),
    //     [i8, i16, i32, i64, u8, u16, u32, u64],
    //     any_strategy
    // );
    // test_op2!(
    //     bitwise_rotate_right,
    //     |a, b| a.rotate_right(b as u32),
    //     [i8, i16, i32, i64, u8, u16, u32, u64],
    //     any_strategy
    // );

    // logical ops: bool output; reference uses != Default::default() to match cast::<T, bool>
    test_op2!(
        logical_and,
        |a, b| (a != Default::default()) && (b != Default::default()),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        logical_op_strategy,
        // #[cfg(feature = "half")]
        // [f16], // TODO
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_op2!(
        logical_or,
        |a, b| (a != Default::default()) || (b != Default::default()),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        logical_op_strategy,
        // #[cfg(feature = "half")]
        // [f16], // TODO
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_op2!(
        logical_xor,
        |a, b| (a != Default::default()) ^ (b != Default::default()),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        logical_op_strategy,
        // #[cfg(feature = "half")]
        // [f16], // TODO
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
}
