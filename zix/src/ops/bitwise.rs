#[allow(unused_imports)]
use crate::dtype::f16;
use crate::ops::common::{define_array_op1_method, define_array_op2_method};
use crate::ops::op1::define_op1;
use crate::ops::op2::define_op2;
use crate::storage::ArrayStorage;
use crate::Array;

define_op2!(
    LogicalAnd,
    LogicalAndKernel,
    |a, b| crate::ops::astype::cast::<_, bool>(a) && crate::ops::astype::cast::<_, bool>(b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = bool
);
define_op2!(
    LogicalOr,
    LogicalOrKernel,
    |a, b| crate::ops::astype::cast::<_, bool>(a) || crate::ops::astype::cast::<_, bool>(b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = bool
);
define_op2!(
    LogicalXor,
    LogicalXorKernel,
    |a, b| crate::ops::astype::cast::<_, bool>(a) ^ crate::ops::astype::cast::<_, bool>(b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = bool
);
define_op1!(
    LogicalNot,
    LogicalNotKernel,
    |a| !crate::ops::astype::cast::<_, bool>(a),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = bool
);

define_op2!(
    BitwiseAnd,
    BitwiseAndKernel,
    |a, b| a & b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
);
define_op2!(
    BitwiseOr,
    BitwiseOrKernel,
    |a, b| a | b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
);
define_op2!(
    BitwiseXor,
    BitwiseXorKernel,
    |a, b| a ^ b,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
);
define_op1!(
    BitwiseNot,
    BitwiseNotKernel,
    |a| !a,
    [i8, i16, i32, i64, u8, u16, u32, u64, bool],
    output_type = "same"
);

define_op2!(
    BitwiseShiftLeft,
    BitwiseShiftLeftKernel,
    |a, b| a << b,
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = "same"
);
define_op2!(
    BitwiseShiftRight,
    BitwiseShiftRightKernel,
    |a, b| a >> b,
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = "same"
);
define_op1!(
    CountOnes,
    CountOnesKernel,
    |a| a.count_ones(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    CountZeros,
    CountZerosKernel,
    |a| a.count_zeros(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    LeadingZeros,
    LeadingZerosKernel,
    |a| a.leading_zeros(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    TrailingZeros,
    TrailingZerosKernel,
    |a| a.trailing_zeros(),
    [i8, i16, i32, i64, u8, u16, u32, u64],
    output_type = u32
);
define_op1!(
    SwapBytes,
    SwapBytesKernel,
    |a| a.swap_bytes(),
    [i16, i32, i64, u16, u32, u64],
    output_type = "same"
);
define_op1!(
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
