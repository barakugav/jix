use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::op2::define_op2;
use crate::ops::prelude::*;
use crate::ops::{define_op1, define_op2_rhs_fixed};

pub(crate) mod _traits {
    use crate::scalar::traits_util::{
        define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
    };

    define_scalar_op2_trait!(
        /// Scalar kernel of [`And`](crate::ops::And), as [`core::ops::BitAnd`].
        BitAnd,
        bitand,
        bitand_bulk
    );
    define_scalar_op2_trait!(
        /// Scalar kernel of [`Or`](crate::ops::Or), as [`core::ops::BitOr`].
        BitOr,
        bitor,
        bitor_bulk
    );
    define_scalar_op2_trait!(
        /// Scalar kernel of [`Xor`](crate::ops::Xor), as [`core::ops::BitXor`].
        BitXor,
        bitxor,
        bitxor_bulk
    );
    define_scalar_op1_trait!(
        /// Scalar kernel of [`Not`](crate::ops::Not), as [`core::ops::Not`].
        Not,
        not,
        not_bulk
    );
    /// Implement the bitwise `$Trait` by the operator `$op`: with SIMD for the integers.
    macro_rules! impl_bitwise2 {
        ($Trait:ident::$f:ident / $f_bulk:ident, $op:tt) => {
            impl_scalar_op2!(
                $Trait::$f / $f_bulk, |a, b| a $op b, simd: |a, b| a $op b,
                [i8, i16, i32, i64, u8, u16, u32, u64]
            );
            impl_scalar_op2!($Trait::$f, |a, b| a $op b, [bool]);
        };
    }
    impl_bitwise2!(BitAnd::bitand / bitand_bulk, &);
    impl_bitwise2!(BitOr::bitor / bitor_bulk, |);
    impl_bitwise2!(BitXor::bitxor / bitxor_bulk, ^);
    impl_scalar_op1!(
        Not::not / not_bulk, |x| !x, simd: |x| !x, [i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_scalar_op1!(Not::not, |x| !x, [bool]);

    define_scalar_op2_trait!(
        /// Scalar kernel of [`BitwiseShiftRight`](crate::ops::BitwiseShiftRight), as
        /// [`core::ops::Shr`].
        Shr,
        shr,
        shr_bulk
    );
    impl_scalar_op2!(
        Shr::shr, |a, b| a >> b,
        cross [i8, i16, i32, i64, u8, u16, u32, u64] x [i8, i16, i64, u8, u16, u64]
    );
    impl_scalar_op2!(
        Shr::shr, |a, b| a >> b,
        cross [i8, i16, i64, u8, u16, u64] x [i32, u32]
    );
    // SIMD for 32-bit values only: the others are not faster than the auto-vectorized scalar
    // kernel (static analysis). The amount modulo the bit width, as in release builds (debug
    // builds panic).
    impl_scalar_op2!(
        Shr::shr / shr_bulk, |a, b| a >> b, simd: |a, b| a >> (b & 31),
        pairs [(i32, i32), (u32, u32)]
    );
    impl_scalar_op2!(
        Shr::shr / shr_bulk, |a, b| a >> b, simd: |a, b| a >> (b & 31).bitcast::<S::i32s>(),
        pairs [(i32, u32)]
    );
    impl_scalar_op2!(
        Shr::shr / shr_bulk, |a, b| a >> b, simd: |a, b| a >> (b & 31).bitcast::<S::u32s>(),
        pairs [(u32, i32)]
    );

    define_scalar_op1_trait!(
        /// Scalar kernel of [`CountZeros`](crate::ops::CountZeros), as [`u32::count_zeros`].
        CountZeros,
        count_zeros,
        count_zeros_bulk
    );
    // SIMD for 32-bit inputs: the only ones with as many lanes as the `u32` output.
    impl_scalar_op1!(
        CountZeros::count_zeros / count_zeros_bulk, |x| x.count_zeros(),
        simd: |x| (!x).count_ones(), [u32 => u32]
    );
    impl_scalar_op1!(
        CountZeros::count_zeros / count_zeros_bulk, |x| x.count_zeros(),
        simd: |x| (!x).count_ones().bitcast::<S::u32s>(), [i32 => u32]
    );
    impl_scalar_op1!(
        CountZeros::count_zeros, |x| x.count_zeros(),
        [i8 => u32, i16 => u32, i64 => u32, u8 => u32, u16 => u32, u64 => u32]
    );

    /// Define the integer scalar trait `$Trait` as `u32`'s `$f`, of output `$Out`.
    macro_rules! int_op1 {
        ($(#[$meta:meta])* $Trait:ident, $f:ident, $f_bulk:ident, $Out:ty) => {
            define_scalar_op1_trait!($(#[$meta])* $Trait, $f, $f_bulk);
            impl_scalar_op1!(
                $Trait::$f, |x| x.$f(),
                [
                    i8 => $Out, i16 => $Out, i32 => $Out, i64 => $Out,
                    u8 => $Out, u16 => $Out, u32 => $Out, u64 => $Out,
                ]
            );
        };
    }
    int_op1!(
        /// Scalar kernel of [`CountOnes`](crate::ops::CountOnes), as [`u32::count_ones`].
        CountOnes,
        count_ones,
        count_ones_bulk,
        u32
    );
    int_op1!(
        /// Scalar kernel of [`LeadingZeros`](crate::ops::LeadingZeros), as
        /// [`u32::leading_zeros`].
        LeadingZeros,
        leading_zeros,
        leading_zeros_bulk,
        u32
    );
    int_op1!(
        /// Scalar kernel of [`TrailingZeros`](crate::ops::TrailingZeros), as
        /// [`u32::trailing_zeros`].
        TrailingZeros,
        trailing_zeros,
        trailing_zeros_bulk,
        u32
    );
    int_op1!(
        /// Scalar kernel of [`SwapBytes`](crate::ops::SwapBytes), as [`u32::swap_bytes`].
        SwapBytes,
        swap_bytes,
        swap_bytes_bulk,
        Self
    );
    int_op1!(
        /// Scalar kernel of [`ReverseBits`](crate::ops::ReverseBits), as [`u32::reverse_bits`].
        ReverseBits,
        reverse_bits,
        reverse_bits_bulk,
        Self
    );

    define_scalar_op2_trait!(
        /// Scalar kernel of [`BitwiseRotateLeft`](crate::ops::BitwiseRotateLeft), as
        /// [`u32::rotate_left`].
        RotateLeft<Rhs = u32>, rotate_left, rotate_left_bulk
    );
    define_scalar_op2_trait!(
        /// Scalar kernel of [`BitwiseRotateRight`](crate::ops::BitwiseRotateRight), as
        /// [`u32::rotate_right`].
        RotateRight<Rhs = u32>, rotate_right, rotate_right_bulk
    );
    impl_scalar_op2!(
        RotateLeft::rotate_left, |a, b| a.rotate_left(b),
        cross [i8, i16, i32, i64, u8, u16, u32, u64] x [u32]
    );
    impl_scalar_op2!(
        RotateRight::rotate_right, |a, b| a.rotate_right(b),
        cross [i8, i16, i32, i64, u8, u16, u32, u64] x [u32]
    );

    define_scalar_op2_trait!(
        /// Scalar kernel of [`BitwiseShiftLeft`](crate::ops::BitwiseShiftLeft), as
        /// [`core::ops::Shl`].
        Shl,
        shl,
        shl_bulk
    );
    // No SIMD: LLVM vectorizes the scalar kernel's 32-bit shift as a multiply by `2^b` where
    // fearless_simd extracts the lanes (SSE4.2), so no body is faster (static analysis).
    impl_scalar_op2!(
        Shl::shl, |a, b| a << b,
        cross [i8, i16, i32, i64, u8, u16, u32, u64] x [i8, i16, i32, i64, u8, u16, u32, u64]
    );
}

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// Applies the bitwise AND to each pair of corresponding bits. For `bool` this is
    /// equivalent to logical AND (`&&`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// the `&` operator or [`Array::bitand()`](core::ops::BitAnd::bitand).
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
    <crate::scalar::BitAnd>::bitand(a, b),
    core_op = BitAnd::bitand,
    simd: bitand_bulk,
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
    <crate::scalar::BitOr>::bitor(a, b),
    core_op = BitOr::bitor,
    simd: bitor_bulk,
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
    <crate::scalar::BitXor>::bitxor(a, b),
    core_op = BitXor::bitxor,
    simd: bitxor_bulk,
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
    <crate::scalar::Not>::not,
    core_op = Not::not,
    simd: not_bulk,
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
    <crate::scalar::Shl>::shl(a, b),
    simd: shl_bulk,
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
    <crate::scalar::Shr>::shr(a, b),
    simd: shr_bulk,
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
    <crate::scalar::RotateLeft>::rotate_left(a, b),
    rhs = u32,
    simd: rotate_left_bulk,
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
    <crate::scalar::RotateRight>::rotate_right(a, b),
    rhs = u32,
    simd: rotate_right_bulk,
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
    <crate::scalar::CountOnes>::count_ones,
    simd: count_ones_bulk,
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
    <crate::scalar::CountZeros>::count_zeros,
    simd: count_zeros_bulk,
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
    <crate::scalar::LeadingZeros>::leading_zeros,
    simd: leading_zeros_bulk,
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
    <crate::scalar::TrailingZeros>::trailing_zeros,
    simd: trailing_zeros_bulk,
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
    <crate::scalar::SwapBytes>::swap_bytes,
    simd: swap_bytes_bulk,
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
    <crate::scalar::ReverseBits>::reverse_bits,
    simd: reverse_bits_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(bitwise_shift_left: BitwiseShiftLeft, crate::scalar::Shl);
    define_array_op2_method!(bitwise_shift_right: BitwiseShiftRight, crate::scalar::Shr);
    define_array_op2_method!(bitwise_rotate_left: BitwiseRotateLeft, crate::scalar::RotateLeft, fixed_lhs_type = u32);
    define_array_op2_method!(bitwise_rotate_right: BitwiseRotateRight, crate::scalar::RotateRight, fixed_lhs_type = u32);
    define_array_op1_method!(count_ones: CountOnes, crate::scalar::CountOnes);
    define_array_op1_method!(count_zeros: CountZeros, crate::scalar::CountZeros);
    define_array_op1_method!(leading_zeros: LeadingZeros, crate::scalar::LeadingZeros);
    define_array_op1_method!(trailing_zeros: TrailingZeros, crate::scalar::TrailingZeros);
    define_array_op1_method!(swap_bytes: SwapBytes, crate::scalar::SwapBytes);
    define_array_op1_method!(reverse_bits: ReverseBits, crate::scalar::ReverseBits);
}

#[cfg(test)]
mod tests {
    use std::ops::{BitAnd, BitOr, BitXor, Not};

    use crate::ops::op1::tests::test_op1;
    use crate::ops::op2::tests::test_op2;

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: shift amounts modulo the bit width), on every SIMD
    /// level of the CPU, over edge cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::{
            AndKernel, BitwiseShiftRightKernel, CountOnesKernel, CountZerosKernel, NotKernel,
            OrKernel, XorKernel,
        };
        use crate::ops::op1::Op1Kernel;
        use crate::ops::op2::Op2Kernel;
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $ta:ty, $tb:ty, $f:expr) => {
                    let (a, b) = (<$ta>::simd_test_values(0), <$tb>::simd_test_values(5));
                    let what = concat!(
                        stringify!($kernel),
                        " ",
                        stringify!($ta),
                        " ",
                        stringify!($tb)
                    );
                    assert_same_elements($kernel.apply_bulk(simd, a, b), |i| $f(a[i], b[i]), what);
                };
            }
            macro_rules! case1 {
                ($kernel:ident, $t:ty, $f:expr) => {
                    let xs = <$t>::simd_test_values(0);
                    let what = concat!(stringify!($kernel), " ", stringify!($t));
                    assert_same_elements($kernel.apply_bulk(simd, xs), |i| $f(xs[i]), what);
                };
            }
            macro_rules! ints {
                ($m:ident!($kernel:ident, $f:expr)) => {
                    $m!($kernel, i8, i8, $f);
                    $m!($kernel, i16, i16, $f);
                    $m!($kernel, i32, i32, $f);
                    $m!($kernel, i64, i64, $f);
                    $m!($kernel, u8, u8, $f);
                    $m!($kernel, u16, u16, $f);
                    $m!($kernel, u32, u32, $f);
                    $m!($kernel, u64, u64, $f);
                };
            }
            // Every 32-bit (value, amount) pair of signedness.
            macro_rules! shifts {
                ($kernel:ident, $f:ident) => {
                    shifts!(@width $kernel, $f, i32, u32);
                };
                (@width $kernel:ident, $f:ident, $i:ty, $u:ty) => {
                    case!($kernel, $i, $i, |x: $i, y: $i| x.$f(y as u32));
                    case!($kernel, $i, $u, |x: $i, y: $u| x.$f(y as u32));
                    case!($kernel, $u, $u, |x: $u, y: $u| x.$f(y as u32));
                    case!($kernel, $u, $i, |x: $u, y: $i| x.$f(y as u32));
                };
            }
            simd.vectorize(|| {
                ints!(case!(AndKernel, |x, y| x & y));
                ints!(case!(OrKernel, |x, y| x | y));
                ints!(case!(XorKernel, |x, y| x ^ y));
                macro_rules! not {
                    ($kernel:ident, $t:ty, $_t:ty, $f:expr) => {
                        case1!($kernel, $t, |x: $t| !x)
                    };
                }
                ints!(not!(NotKernel, ()));
                shifts!(BitwiseShiftRightKernel, wrapping_shr);
                case1!(CountOnesKernel, i32, i32::count_ones);
                case1!(CountOnesKernel, u32, u32::count_ones);
                case1!(CountZerosKernel, i32, i32::count_zeros);
                case1!(CountZerosKernel, u32, u32::count_zeros);
            });
        }
        for_each_simd_level!(check);
    }

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
            crate::util::assert_array_matches(&za.view().$method(), &expected);
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
            crate::util::assert_array_matches(&za.view().$method(zb.view()), &expected);
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
        crate::util::assert_array_matches(&za64.view().not(), &expected64);
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
