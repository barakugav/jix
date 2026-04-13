use crate::ops::common::{define_array_op1_method, define_array_op2_method};
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

impl<S> crate::Array<S>
where
    S: crate::storage::ArrayStorage,
{
    define_array_op2_method!(bitwise_and: BitwiseAnd);
    define_array_op2_method!(bitwise_or: BitwiseOr);
    define_array_op2_method!(bitwise_xor: BitwiseXor);
    define_array_op1_method!(bitwise_not: BitwiseNot);
}
