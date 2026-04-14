use crate::dtype::{Complex, f16};
use crate::ops::common::define_array_op2_method;
use crate::ops::define_math2_op;
use crate::ops::logical2::define_logical2_op;

define_logical2_op!(
    Equal,
    EqualKernel,
    |a, b| -> bool { a == b },
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16, (Complex<f32>), (Complex<f64>)]
);
define_logical2_op!(
    NotEqual,
    NotEqualKernel,
    |a, b| -> bool { a != b },
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16, (Complex<f32>), (Complex<f64>)]
);
define_logical2_op!(
    Greater,
    GreaterKernel,
    |a, b| -> bool { a > b },
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);
define_logical2_op!(
    GreaterEqual,
    GreaterEqualKernel,
    |a, b| -> bool { a >= b },
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);
define_logical2_op!(
    Less,
    LessKernel,
    |a, b| -> bool { a < b },
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);
define_logical2_op!(
    LessEqual,
    LessEqualKernel,
    |a, b| -> bool { a <= b },
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);

define_math2_op!(
    Maximum,
    MaximumKernel,
    |a, b| MaximumTrait::maximum(a, b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool]
);
define_math2_op!(
    Minimum,
    MinimumKernel,
    |a, b| MinimumTrait::minimum(a, b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool]
);

trait MaximumTrait {
    fn maximum(self, other: Self) -> Self;
}
macro_rules! impl_integer_maximum {
    ($($t:ty),* $(,)?) => {
        $(impl MaximumTrait for $t {
            fn maximum(self, other: Self) -> Self {
                std::cmp::max(self, other)
            }
        })*
    };
}
macro_rules! impl_float_maximum {
    ($($t:ty),* $(,)?) => {
        $(impl MaximumTrait for $t {
            fn maximum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.max(other)
                }
            }
        })*
    };
}
impl_integer_maximum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
impl_float_maximum!(f32, f64);
#[cfg(feature = "half")]
impl_float_maximum!(f16);

trait MinimumTrait {
    fn minimum(self, other: Self) -> Self;
}
macro_rules! impl_integer_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl MinimumTrait for $t {
            fn minimum(self, other: Self) -> Self {
                std::cmp::min(self, other)
            }
        })*
    };
}
macro_rules! impl_float_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl MinimumTrait for $t {
            fn minimum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.min(other)
                }
            }
        })*
    };
}
impl_integer_minimum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
impl_float_minimum!(f32, f64);
#[cfg(feature = "half")]
impl_float_minimum!(f16);

impl<S> crate::Array<S>
where
    S: crate::storage::ArrayStorage,
{
    define_array_op2_method!(equal: Equal);
    define_array_op2_method!(not_equal: NotEqual);
    define_array_op2_method!(greater: Greater);
    define_array_op2_method!(greater_equal: GreaterEqual);
    define_array_op2_method!(less: Less);
    define_array_op2_method!(less_equal: LessEqual);
    define_array_op2_method!(maximum: Maximum);
    define_array_op2_method!(minimum: Minimum);
}
