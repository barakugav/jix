use crate::array::Array;
#[allow(unused_imports)]
use crate::dtype::f16;
use crate::ops::common::define_array_op1_method;
use crate::ops::define_op1;
use crate::storage::ArrayStorage;

define_op1!(
    IsNan,
    IsNanKernel,
    |a| a.is_nan(),
    [f16, f32, f64],
    output_type = bool
);
define_op1!(
    IsFinite,
    IsFiniteKernel,
    |a| a.is_finite(),
    [f16, f32, f64],
    output_type = bool
);
define_op1!(
    IsInfinite,
    IsInfiniteKernel,
    |a| a.is_infinite(),
    [f16, f32, f64],
    output_type = bool
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(is_nan: IsNan);
    define_array_op1_method!(is_finite: IsFinite);
    define_array_op1_method!(is_infinite: IsInfinite);
}
