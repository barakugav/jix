use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::define_op1;
use crate::ops::op2::define_op2;
use crate::ops::prelude::*;

pub(crate) mod _traits {
    use crate::scalar::traits_util::{
        define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
    };

    define_scalar_op2_trait!(
        /// Bitwise AND operation, like [`core::ops::BitAnd`].
        ///
        /// Applies the bitwise AND to each pair of corresponding bits. For `bool` this is
        /// equivalent to logical AND (`&&`).
        BitAnd,
        /// Bitwise AND with another value.
        bitand,
    );
    define_scalar_op2_trait!(
        /// Bitwise OR operation, like [`core::ops::BitOr`].
        ///
        /// Applies the bitwise OR to each pair of corresponding bits. For `bool` this is
        /// equivalent to logical OR (`||`).
        BitOr,
        /// Bitwise OR with another value.
        bitor,
    );
    define_scalar_op2_trait!(
        /// Bitwise XOR operation, like [`core::ops::BitXor`].
        ///
        /// Applies the bitwise XOR to each pair of corresponding bits. For `bool` this is
        /// equivalent to logical XOR.
        BitXor,
        /// Bitwise XOR with another value.
        bitxor,
    );
    define_scalar_op1_trait!(
        /// Bitwise NOT operation, like [`core::ops::Not`].
        ///
        /// Flips every bit. For `bool` this is equivalent to logical NOT.
        Not,
        /// Bitwise NOT of the value.
        not,
    );

    macro_rules! impl_bitwise2 {
        ($Trait:ident::$f:ident, $op:tt) => {
            impl_scalar_op2!(
                impl $Trait for [i8, i16, i32, i64, u8, u16, u32, u64] {
                    fn $f = |a, b| a $op b,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [bool] {
                    fn $f = |a, b| a $op b,
                }
            );
        };
    }
    impl_bitwise2!(BitAnd::bitand, &);
    impl_bitwise2!(BitOr::bitor, |);
    impl_bitwise2!(BitXor::bitxor, ^);

    impl_scalar_op1!(
        impl Not for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn not = |x| !x,
        }
    );
    impl_scalar_op1!(
        impl Not for [bool] {
            fn not = |x| !x,
        }
    );

    define_scalar_op2_trait!(
        /// Right shift operation, like [`core::ops::Shr`].
        ///
        /// For **unsigned** types this is a logical shift: vacated bits are filled with zeros.
        /// For **signed** types this is an arithmetic shift: vacated bits are filled with the
        /// sign bit (the result preserves the sign of the value).
        /// The shift uses Rust's `>>` operator: shifting by a value greater than or equal to the
        /// bit width of the type panics in debug builds and masks the shift amount modulo the bit
        /// width in release builds (it does NOT produce zero).
        Shr,
        /// Shift the bits right by another value.
        shr,
    );
    impl_scalar_op2!(
        impl Shr for [i8, i16, i32, i64, u8, u16, u32, u64] x [i8, i16, i64, u8, u16, u64] {
            fn shr = |a, b| a >> b,
        }
    );
    impl_scalar_op2!(
        impl Shr for [i8, i16, i64, u8, u16, u64] x [i32, u32] {
            fn shr = |a, b| a >> b,
        }
    );
    impl_scalar_op2!(
        impl Shr for [i32, u32] x [i32, u32] {
            fn shr = |a, b| a >> b,
        }
    );

    define_scalar_op1_trait!(
        /// Zero-bit count operation, as [`u32::count_zeros`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Equivalent to `bit_width - count_ones`. For signed integers the full bit
        /// representation (including the sign bit) is used.
        CountZeros,
        /// Count the unset bits (`0`s).
        count_zeros,
    );
    impl_scalar_op1!(
        impl CountZeros for [i8, i16, i32, i64, u8, u16, u32, u64] {
            type Output = u32;
            fn count_zeros = |x| x.count_zeros(),
        }
    );

    macro_rules! int_op1 {
        (
            $(#[$trait_meta:meta])* $Trait:ident,
            $(#[$fn_meta:meta])* $f:ident,
            $Out:ty
        ) => {
            define_scalar_op1_trait!(
                $(#[$trait_meta])* $Trait,
                $(#[$fn_meta])* $f,
            );
            impl_scalar_op1!(
                impl $Trait for [i8, i16, i32, i64, u8, u16, u32, u64] {
                    type Output = $Out;
                    fn $f = |x| x.$f(),
                }
            );
        };
    }
    int_op1!(
        /// Set-bit count operation, as [`u32::count_ones`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Also known as the population count or Hamming weight. For signed integers the
        /// bit representation (including the sign bit) is used.
        CountOnes,
        /// Count the set bits (`1`s).
        count_ones,
        u32
    );
    int_op1!(
        /// Leading-zero count operation, as [`u32::leading_zeros`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Counts zeros from the most-significant bit down to (but not including) the first
        /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
        /// `0u32`).
        LeadingZeros,
        /// Count the leading zero bits.
        leading_zeros,
        u32
    );
    int_op1!(
        /// Trailing-zero count operation, as [`u32::trailing_zeros`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Counts zeros from the least-significant bit up to (but not including) the first
        /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
        /// `0u32`).
        TrailingZeros,
        /// Count the trailing zero bits.
        trailing_zeros,
        u32
    );
    int_op1!(
        /// Byte-order reversal operation, as [`u32::swap_bytes`].
        ///
        /// Swaps the bytes of the value (e.g. converts between big-endian and little-endian
        /// representation). Single-byte types (`i8`, `u8`) are not supported since swapping one
        /// byte is a no-op.
        SwapBytes,
        /// Reverse the byte order.
        swap_bytes,
        Self
    );
    int_op1!(
        /// Bit-order reversal operation, as [`u32::reverse_bits`].
        ///
        /// The most-significant bit becomes the least-significant and vice versa.
        ReverseBits,
        /// Reverse the bit order.
        reverse_bits,
        Self
    );

    define_scalar_op2_trait!(
        /// Left rotation operation, as [`u32::rotate_left`].
        ///
        /// Rotates the bits left by `rhs`. Unlike a left shift, bits shifted out of the
        /// most-significant position wrap around to the least-significant position, so no bits
        /// are lost. The rotation amount is taken modulo the bit width of the type.
        RotateLeft<Rhs = u32>,
        /// Rotate the bits left by another value.
        rotate_left,
    );
    define_scalar_op2_trait!(
        /// Right rotation operation, as [`u32::rotate_right`].
        ///
        /// Rotates the bits right by `rhs`. Unlike a right shift, bits shifted out of the
        /// least-significant position wrap around to the most-significant position, so no bits
        /// are lost. The rotation amount is taken modulo the bit width of the type.
        RotateRight<Rhs = u32>,
        /// Rotate the bits right by another value.
        rotate_right,
    );
    impl_scalar_op2!(
        impl RotateLeft for [i8, i16, i32, i64, u8, u16, u32, u64] x [u32] {
            fn rotate_left = |a, b| a.rotate_left(b),
        }
    );
    impl_scalar_op2!(
        impl RotateRight for [i8, i16, i32, i64, u8, u16, u32, u64] x [u32] {
            fn rotate_right = |a, b| a.rotate_right(b),
        }
    );

    define_scalar_op2_trait!(
        /// Left shift operation, like [`core::ops::Shl`].
        ///
        /// Shifts the bits left by `rhs`. Vacated bits are filled with zeros. The shift uses
        /// Rust's `<<` operator: shifting by a value greater than or equal to the bit width of
        /// the type panics in debug builds and masks the shift amount modulo the bit width in
        /// release builds (it does NOT produce zero).
        Shl,
        /// Shift the bits left by another value.
        shl,
    );
    impl_scalar_op2!(
        impl Shl for
            [i8, i16, i32, i64, u8, u16, u32, u64] x [i8, i16, i32, i64, u8, u16, u32, u64]
        {
            fn shl = |a, b| a << b,
        }
    );
}

define_op2!(
    /// Element-wise bitwise AND of two arrays.
    ///
    /// See [`jix::scalar::BitAnd`](crate::scalar::BitAnd) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::BitAnd::bitand,
    core_op: core::ops::BitAnd::bitand,
);
define_op2!(
    /// Element-wise bitwise OR of two arrays.
    ///
    /// See [`jix::scalar::BitOr`](crate::scalar::BitOr) scalar trait for the per-element semantics.
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
    crate::scalar::BitOr::bitor,
    core_op: core::ops::BitOr::bitor,
);
define_op2!(
    /// Element-wise bitwise XOR of two arrays.
    ///
    /// See [`jix::scalar::BitXor`](crate::scalar::BitXor) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::BitXor::bitxor,
    core_op: core::ops::BitXor::bitxor,
);

define_op1!(
    /// Element-wise bitwise NOT.
    ///
    /// See [`jix::scalar::Not`](crate::scalar::Not) scalar trait for the per-element semantics.
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
    crate::scalar::Not::not,
    core_op: core::ops::Not::not,
);

define_op2!(
    /// Element-wise left shift (`a << b`).
    ///
    /// See [`jix::scalar::Shl`](crate::scalar::Shl) scalar trait for the per-element semantics.
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
    crate::scalar::Shl::shl,
);

define_op2!(
    /// Element-wise right shift (`a >> b`).
    ///
    /// See [`jix::scalar::Shr`](crate::scalar::Shr) scalar trait for the per-element semantics.
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
    crate::scalar::Shr::shr,
);
define_op2!(
    /// Element-wise bitwise left rotation (`a.rotate_left(b as u32)`).
    ///
    /// See [`jix::scalar::RotateLeft`](crate::scalar::RotateLeft) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::RotateLeft::rotate_left,
);

define_op2!(
    /// Element-wise bitwise right rotation (`a.rotate_right(b as u32)`).
    ///
    /// See [`jix::scalar::RotateRight`](crate::scalar::RotateRight) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::RotateRight::rotate_right,
);
define_op1!(
    /// Counts the number of set bits (`1`s) in each element.
    ///
    /// See [`jix::scalar::CountOnes`](crate::scalar::CountOnes) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::CountOnes::count_ones,
);
define_op1!(
    /// Counts the number of unset bits (`0`s) in each element.
    ///
    /// See [`jix::scalar::CountZeros`](crate::scalar::CountZeros) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::CountZeros::count_zeros,
);
define_op1!(
    /// Counts the number of leading zero bits in each element.
    ///
    /// See [`jix::scalar::LeadingZeros`](crate::scalar::LeadingZeros) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::LeadingZeros::leading_zeros,
);
define_op1!(
    /// Counts the number of trailing zero bits in each element.
    ///
    /// See [`jix::scalar::TrailingZeros`](crate::scalar::TrailingZeros) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::TrailingZeros::trailing_zeros,
);
define_op1!(
    /// Reverses the byte order of each element.
    ///
    /// See [`jix::scalar::SwapBytes`](crate::scalar::SwapBytes) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::SwapBytes::swap_bytes,
);
define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// See [`jix::scalar::ReverseBits`](crate::scalar::ReverseBits) scalar trait for the
    /// per-element semantics.
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
    crate::scalar::ReverseBits::reverse_bits,
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
