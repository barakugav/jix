use crate::ops::{define_op1, define_op2};

define_op2!(logical_and, LogicalAnd);
define_op2!(logical_or, LogicalOr);
define_op2!(logical_xor, LogicalXor);
define_op1!(logical_not, LogicalNot);
define_op2!(bitwise_and, BitwiseAnd);
define_op2!(bitwise_or, BitwiseOr);
define_op2!(bitwise_xor, BitwiseXor);
define_op1!(bitwise_not, BitwiseNot);
define_op2!(bitwise_shift_left, BitwiseShiftLeft);
define_op2!(bitwise_shift_right, BitwiseShiftRight);
define_op2!(bitwise_rotate_left, BitwiseRotateLeft);
define_op2!(bitwise_rotate_right, BitwiseRotateRight);
define_op1!(count_ones, CountOnes);
define_op1!(leading_zeros, LeadingZeros);
define_op1!(trailing_zeros, TrailingZeros);
define_op1!(swap_bytes, SwapBytes);
define_op1!(reverse_bits, ReverseBits);
