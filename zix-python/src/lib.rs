use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod array;
pub use array::Array;

mod codec;
pub use codec::ReadContext;

mod params;
pub use params::ArrayParams;

mod dtype;
pub mod ops;
mod storage;
mod util;

#[pymodule]
mod zix {
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::{Array, ArrayParams, ReadContext};

    #[pymodule_export]
    use crate::array::compact;

    #[pymodule_export]
    use crate::ops::{asarray, astype};

    #[pymodule_export]
    use crate::ops::copy;

    #[pymodule_export]
    use crate::ops::{add, divide, multiply, power, subtract};

    #[pymodule_export]
    use crate::ops::{
        bitwise_and, bitwise_not, bitwise_or, bitwise_rotate_left, bitwise_rotate_right,
        bitwise_shift_left, bitwise_shift_right, bitwise_xor, count_ones, count_zeros,
        leading_zeros, logical_and, logical_not, logical_or, logical_xor, reverse_bits, swap_bytes,
        trailing_zeros,
    };

    #[pymodule_export]
    use crate::ops::r#where;

    #[pymodule_export]
    use crate::ops::{
        equal, greater, greater_equal, less, less_equal, maximum, minimum, not_equal,
    };

    #[pymodule_export]
    use crate::ops::{
        broadcast, concatenate, flatten, insert_axes, permute_axes, remove_axes, reshape, squeeze,
        stack, unsqueeze,
    };

    #[pymodule_export]
    use crate::ops::{
        absolute, acos, asin, atan, ceil, cos, exp, floor, log, negative, round, signum, sin, sqrt,
        tan,
    };

    #[pymodule_export]
    use crate::ops::{is_finite, is_infinite, is_nan};

    #[pymodule_export]
    use crate::ops::{all, any, argmax, argmin, max, mean, min, product, std, sum, var};
}

define_stub_info_gatherer!(gen_pyi);
