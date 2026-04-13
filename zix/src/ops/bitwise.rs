use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::define_logical1_op;
use crate::ops::math1::{define_math1_op, define_math1_op_kernel};
use crate::ops::math2::define_math2_op;

define_math2_op!(
    BitwiseAnd,
    BitwiseAndKernel,
    |a, b| a & b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool]
);
define_math2_op!(
    BitwiseOr,
    BitwiseOrKernel,
    |a, b| a | b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool]
);
define_math2_op!(
    BitwiseXor,
    BitwiseXorKernel,
    |a, b| a ^ b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool]
);
define_math1_op!(
    BitwiseNot,
    BitwiseNotKernel,
    |a| !a,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool]
);

define_math2_op!(
    BitwiseShiftLeft,
    BitwiseShiftLeftKernel,
    |a, b| a << b,
    [i8, i16, i32, i64, u8, u16, u32, u64]
);
define_math2_op!(
    BitwiseShiftRight,
    BitwiseShiftRightKernel,
    |a, b| a >> b,
    [i8, i16, i32, i64, u8, u16, u32, u64]
);
define_logical1_op!(
    CountOnes,
    CountOnesKernel,
    |a| -> u32 { a.count_ones() },
    [i8, i16, i32, i64, u8, u16, u32, u64]
);
define_logical1_op!(
    CountZeros,
    CountZerosKernel,
    |a| -> u32 { a.count_zeros() },
    [i8, i16, i32, i64, u8, u16, u32, u64]
);
define_logical1_op!(
    LeadingZeros,
    LeadingZerosKernel,
    |a| -> u32 { a.leading_zeros() },
    [i8, i16, i32, i64, u8, u16, u32, u64]
);
define_logical1_op!(
    TrailingZeros,
    TrailingZerosKernel,
    |a| -> u32 { a.trailing_zeros() },
    [i8, i16, i32, i64, u8, u16, u32, u64]
);
define_math1_op!(
    SwapBytes,
    SwapBytesKernel,
    |a| a.swap_bytes(),
    [i16, i32, i64, u16, u32, u64]
);
define_math1_op!(
    ReverseBits,
    ReverseBitsKernel,
    |a| a.reverse_bits(),
    [i8, i16, i32, i64, u8, u16, u32, u64]
);

impl<S> crate::Array<S>
where
    S: crate::storage::ArrayStorage,
{
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
