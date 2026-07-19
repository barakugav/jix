use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::op2::define_op2;
use crate::ops::{define_op1, define_op2_rhs_fixed};
use crate::{Array, ArrayStorage};

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// Applies the bitwise AND to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical AND (`&&`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// as the `&` operator or [`Array::bitand()`](core::ops::BitAnd::bitand).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b1100u8, 0b1010, 0b1111])?;
    /// let b = Array::compact_ndarray(&array![0b1010u8, 0b0101, 0b0000])?;
    /// let result = (a & b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b1000, 0b0000, 0b0000]);
    ///
    /// // Mask out the lower nibble.
    /// let c = Array::compact_ndarray(&array![0xABu8, 0xCDu8])?;
    /// let d = Array::compact_ndarray(&array![0xF0u8, 0xF0u8])?;
    /// let result = (c & d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xA0, 0xC0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    And,
    AndKernel,
    <core::ops::BitAnd>::bitand(a, b),
    core_op = BitAnd::bitand,
);
define_op2!(
    /// Element-wise bitwise OR of two arrays.
    ///
    /// Applies the bitwise OR to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical OR (`||`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// the `|` operator or [`Array::bitor()`](core::ops::BitOr::bitor).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b1100u8, 0b1010, 0b0000])?;
    /// let b = Array::compact_ndarray(&array![0b1010u8, 0b0101, 0b1111])?;
    /// let result = (a | b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b1110, 0b1111, 0b1111]);
    ///
    /// // Set a specific bit pattern.
    /// let c = Array::compact_ndarray(&array![0x0Fu8, 0x00u8])?;
    /// let d = Array::compact_ndarray(&array![0xF0u8, 0xF0u8])?;
    /// let result = (c | d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xFF, 0xF0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Or,
    OrKernel,
    <core::ops::BitOr>::bitor(a, b),
    core_op = BitOr::bitor,
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
    /// the `^` operator or [`Array::bitxor()`](core::ops::BitXor::bitxor).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b1100u8, 0b1010, 0b1111])?;
    /// let b = Array::compact_ndarray(&array![0b1010u8, 0b1010, 0b1111])?;
    /// let result = (a ^ b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b0110, 0b0000, 0b0000]);
    ///
    /// // Toggle bits using a mask.
    /// let c = Array::compact_ndarray(&array![0xFFu8, 0x0Fu8])?;
    /// let d = Array::compact_ndarray(&array![0x0Fu8, 0x0Fu8])?;
    /// let result = (c ^ d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0xF0, 0x00]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Xor,
    XorKernel,
    <core::ops::BitXor>::bitxor(a, b),
    core_op = BitXor::bitxor,
);

define_op1!(
    /// Element-wise bitwise NOT.
    ///
    /// Flips every bit. For `bool` this is equivalent to logical NOT.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// the `!` operator or [`Array::not()`](core::ops::Not::not).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b00001111u8, 0b11110000u8, 0u8])?;
    /// let result = (!a).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b11110000, 0b00001111, 0xFF]);
    ///
    /// // For bool arrays, bitwise NOT is equivalent to logical NOT.
    /// let b = Array::compact_ndarray(&array![true, false])?;
    /// let result = (!b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Not,
    NotKernel,
    <core::ops::Not>::not,
    core_op = Not::not,
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// Shifts the bits of each element of `a` left by the corresponding value in `b`.
    /// Vacated bits are filled with zeros. The shift uses Rust's `<<` operator: shifting by
    /// a value greater than or equal to the bit width of the type panics in debug builds and
    /// masks the shift amount modulo the bit width in release builds (it does NOT produce zero).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_shift_left()`](crate::Array::bitwise_shift_left).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b00000001u8, 0b00000010u8, 0b00000100u8])?;
    /// let b = Array::compact_ndarray(&array![1u8, 2, 3])?;
    /// let result = a.bitwise_shift_left(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b00000010, 0b00001000, 0b00100000]);
    ///
    /// // Signed arithmetic left shift: sign bit is lost if shifted out.
    /// let c = Array::compact_ndarray(&array![1i8, -1i8])?;
    /// let d = Array::compact_ndarray(&array![3i8, 1i8])?;
    /// let result = c.bitwise_shift_left(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[8, -2]);
    /// # Ok::<(), jix::Error>(())
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
    /// The shift uses Rust's `>>` operator: shifting by a value greater than or equal to the
    /// bit width of the type panics in debug builds and masks the shift amount modulo the bit
    /// width in release builds (it does NOT produce zero).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::bitwise_shift_right()`](crate::Array::bitwise_shift_right).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b10000000u8, 0b00100000u8, 0b00001000u8])?;
    /// let b = Array::compact_ndarray(&array![1u8, 2, 3])?;
    /// let result = a.bitwise_shift_right(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b01000000, 0b00001000, 0b00000001]);
    ///
    /// // Signed arithmetic right shift: vacated bits are filled with the sign bit.
    /// let c = Array::compact_ndarray(&array![-8i8, -1i8])?;
    /// let d = Array::compact_ndarray(&array![2i8, 1i8])?;
    /// let result = c.bitwise_shift_right(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-2, -1]);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b10000001u8, 0b00000001u8, 0b11110000u8])?;
    /// let b = Array::compact_ndarray(&array![1u32, 3, 4])?;
    /// let result = a.bitwise_rotate_left(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b00000011, 0b00001000, 0b00001111]);
    ///
    /// // Rotating by 0 is a no-op.
    /// let c = Array::compact_ndarray(&array![0xABu8])?;
    /// let d = Array::compact_ndarray(&array![0u32])?;
    /// let result = c.bitwise_rotate_left(d).to_ndarray()?;
    /// assert_eq!(result[[0]], 0xABu8);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b10000001u8, 0b00001000u8, 0b00001111u8])?;
    /// let b = Array::compact_ndarray(&array![1u32, 3, 4])?;
    /// let result = a.bitwise_rotate_right(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b11000000, 0b00000001, 0b11110000]);
    ///
    /// // Rotating by 0 is a no-op.
    /// let c = Array::compact_ndarray(&array![0xABu8])?;
    /// let d = Array::compact_ndarray(&array![0u32])?;
    /// let result = c.bitwise_rotate_right(d).to_ndarray()?;
    /// assert_eq!(result[[0]], 0xABu8);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b00001111u8, 0b11001100u8, 0b11111111u8])?;
    /// let result = a.count_ones().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 4, 8]);
    ///
    /// // Zero has no set bits.
    /// let b = Array::compact_ndarray(&array![0u8, 0u8])?;
    /// let result = b.count_ones().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0, 0]);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b11110000u8, 0b00001111u8, 0b11111111u8])?;
    /// let result = a.count_zeros().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 4, 0]);
    ///
    /// // Zero has all bits unset: count_zeros == bit width.
    /// let b = Array::compact_ndarray(&array![0u8])?;
    /// let result = b.count_zeros().to_ndarray()?;
    /// assert_eq!(result[[0]], 8); // u8 has 8 bits
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0x00010000u32, 0x80000000u32, 0x00000001u32])?;
    /// let result = a.leading_zeros().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[15, 0, 31]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = Array::compact_ndarray(&array![0u32])?;
    /// let result = b.leading_zeros().to_ndarray()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0x00010000u32, 0x80000000u32, 0x00000001u32])?;
    /// let result = a.trailing_zeros().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[16, 31, 0]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = Array::compact_ndarray(&array![0u32])?;
    /// let result = b.trailing_zeros().to_ndarray()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0x00FF0000u32, 0x0000FF00u32])?;
    /// let result = a.swap_bytes().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0x0000FF00, 0x00FF0000]);
    ///
    /// // Classic endian-swap example.
    /// let b = Array::compact_ndarray(&array![0x12345678u32])?;
    /// let result = b.swap_bytes().to_ndarray()?;
    /// assert_eq!(result[[0]], 0x78563412u32);
    /// # Ok::<(), jix::Error>(())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b00000001u8, 0b10000000u8, 0b10101010u8])?;
    /// let result = a.reverse_bits().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0b10000000, 0b00000001, 0b01010101]);
    ///
    /// // Reversing bits of 0 gives 0.
    /// let b = Array::compact_ndarray(&array![0u8])?;
    /// let result = b.reverse_bits().to_ndarray()?;
    /// assert_eq!(result[[0]], 0u8);
    /// # Ok::<(), jix::Error>(())
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
    use std::ops::{BitAnd, BitOr, BitXor, Not};

    use crate::ops::op1::tests::test_op1;
    use crate::ops::op2::tests::test_op2;

    test_op1!(count_ones, |a| a.count_ones(), [u8, u32], any_strategy);
    test_op2!(bitand, |a, b| a & b, [u8, u32], any_strategy);
    test_op2!(
        bitwise_shift_left,
        |a, b| a.unbounded_shl(b as u32),
        [u8, u32],
        shift_safe_strategy
    );

    // Surplus widths (u16, u64) of the three ops kept as property tests above: u8/u32 are
    // already covered there, so only the remaining two byte widths need concrete coverage.

    // Edge-value inputs shared by the `*_concrete` tests below: 0, MAX (all bits set), a
    // single set bit, and an alternating 0xAA bit pattern, one array per byte width. The `_2`
    // arrays pair each value with a different one at the same index, so binary ops
    // (bitand/bitor/bitxor) get every "one side is the edge value" combination exercised.
    const EDGE_U8: [u8; 4] = [0, u8::MAX, 1, 0xAA];
    const EDGE_U8_2: [u8; 4] = [u8::MAX, 0, 0xAA, 1];
    const EDGE_U16: [u16; 4] = [0, u16::MAX, 1, 0xAAAA];
    const EDGE_U16_2: [u16; 4] = [u16::MAX, 0, 0xAAAA, 1];
    const EDGE_U32: [u32; 4] = [0, u32::MAX, 1, 0xAAAA_AAAA];
    const EDGE_U32_2: [u32; 4] = [u32::MAX, 0, 0xAAAA_AAAA, 1];
    const EDGE_U64: [u64; 4] = [0, u64::MAX, 1, 0xAAAA_AAAA_AAAA_AAAA];
    const EDGE_U64_2: [u64; 4] = [u64::MAX, 0, 0xAAAA_AAAA_AAAA_AAAA, 1];

    // Shift-amount arrays for the shift ops: 0 and width - 1 alongside 1 and half the width,
    // matching the `EDGE_*` array of the same width.
    const SHIFT_U8: [u8; 4] = [0, 7, 1, 4];
    const SHIFT_U16: [u16; 4] = [0, 15, 1, 8];
    const SHIFT_U32: [u32; 4] = [0, 31, 1, 16];
    const SHIFT_U64: [u64; 4] = [0, 63, 1, 32];

    // A single unary-op assertion for one input array: compact it, apply `$method`, and check
    // against the `ndarray`-computed reference `$body`.
    macro_rules! concrete_op1_case {
        ($method:ident, |$a:ident| $body:expr, $arr:expr) => {{
            use crate::Array;
            let nd = ndarray::arr1(&$arr);
            let za = Array::compact_ndarray(&nd).unwrap();
            let expected = nd.mapv(|$a| $body);
            crate::util::assert_array_matches(&za.as_ref().$method(), &expected);
        }};
    }

    // A `#[test]` fn running `concrete_op1_case!` for each input array in `[...]`.
    macro_rules! concrete_op1 {
        ($name:ident, $method:ident, |$a:ident| $body:expr, [$($arr:expr),+ $(,)?]) => {
            #[test]
            fn $name() {
                $(concrete_op1_case!($method, |$a| $body, $arr);)+
            }
        };
    }

    // A single binary-op assertion for one pair of input arrays (the second array is the
    // shift amount for the shift ops).
    macro_rules! concrete_op2_case {
        ($method:ident, |$a:ident, $b:ident| $body:expr, $arr_a:expr, $arr_b:expr) => {{
            use crate::Array;
            let nd_a = ndarray::arr1(&$arr_a);
            let nd_b = ndarray::arr1(&$arr_b);
            let za = Array::compact_ndarray(&nd_a).unwrap();
            let zb = Array::compact_ndarray(&nd_b).unwrap();
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|&$a, &$b| $body);
            crate::util::assert_array_matches(&za.as_ref().$method(zb.as_ref()), &expected);
        }};
    }

    // A `#[test]` fn running `concrete_op2_case!` for each array pair in `[...]`.
    macro_rules! concrete_op2 {
        (
            $name:ident, $method:ident, |$a:ident, $b:ident| $body:expr,
            [$(($arr_a:expr, $arr_b:expr)),+ $(,)?]
        ) => {
            #[test]
            fn $name() {
                $(concrete_op2_case!($method, |$a, $b| $body, $arr_a, $arr_b);)+
            }
        };
    }

    concrete_op1!(
        count_ones_concrete,
        count_ones,
        |a| a.count_ones(),
        [EDGE_U16, EDGE_U64]
    );

    concrete_op2!(
        bitand_concrete,
        bitand,
        |a, b| a & b,
        [(EDGE_U16, EDGE_U16_2), (EDGE_U64, EDGE_U64_2)]
    );

    // shift amounts include 0 and width - 1, the values kept in `SHIFT_*` above.
    concrete_op2!(
        bitwise_shift_left_concrete,
        bitwise_shift_left,
        |a, b| a.unbounded_shl(b as u32),
        [(EDGE_U16, SHIFT_U16), (EDGE_U64, SHIFT_U64)]
    );

    #[test]
    fn not_concrete() {
        use crate::Array;

        // Edge inputs per width: 0, MAX (all ones), a single set bit, and an alternating
        // 0xAA pattern.
        concrete_op1_case!(not, |a| !a, EDGE_U8);
        concrete_op1_case!(not, |a| !a, EDGE_U16);
        concrete_op1_case!(not, |a| !a, EDGE_U32);

        // Non-default block shape (2 blocks of 2 elements) so a multi-block read exercises
        // this op family too.
        let a64 = ndarray::arr1(&EDGE_U64);
        let za64 = Array::compact_ndarray_with(&a64, crate::util::arr_params(&[2])).unwrap();
        let expected64 = a64.mapv(|a: u64| !a);
        crate::util::assert_array_matches(&za64.as_ref().not(), &expected64);
    }

    concrete_op1!(
        count_zeros_concrete,
        count_zeros,
        |a| a.count_zeros(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    concrete_op1!(
        leading_zeros_concrete,
        leading_zeros,
        |a| a.leading_zeros(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    concrete_op1!(
        trailing_zeros_concrete,
        trailing_zeros,
        |a| a.trailing_zeros(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    // No u8 case: single-byte types don't support swap_bytes (swapping one byte is a no-op),
    // see the doc comment on `SwapBytes` above.
    concrete_op1!(
        swap_bytes_concrete,
        swap_bytes,
        |a| a.swap_bytes(),
        [EDGE_U16, EDGE_U32, EDGE_U64]
    );

    concrete_op1!(
        reverse_bits_concrete,
        reverse_bits,
        |a| a.reverse_bits(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    // Each operand pairs an edge value against its permuted partner (`EDGE_*_2`) so every
    // "one side is the edge value" combination is exercised.
    concrete_op2!(
        bitor_concrete,
        bitor,
        |a, b| a | b,
        [
            (EDGE_U8, EDGE_U8_2),
            (EDGE_U16, EDGE_U16_2),
            (EDGE_U32, EDGE_U32_2),
            (EDGE_U64, EDGE_U64_2),
        ]
    );

    concrete_op2!(
        bitxor_concrete,
        bitxor,
        |a, b| a ^ b,
        [
            (EDGE_U8, EDGE_U8_2),
            (EDGE_U16, EDGE_U16_2),
            (EDGE_U32, EDGE_U32_2),
            (EDGE_U64, EDGE_U64_2),
        ]
    );

    // shift amounts include 0 and width - 1 alongside the edge values being shifted.
    concrete_op2!(
        bitwise_shift_right_concrete,
        bitwise_shift_right,
        |a, b| a.unbounded_shr(b as u32),
        [
            (EDGE_U8, SHIFT_U8),
            (EDGE_U16, SHIFT_U16),
            (EDGE_U32, SHIFT_U32),
            (EDGE_U64, SHIFT_U64),
        ]
    );
}
