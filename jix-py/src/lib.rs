#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg))]
//
#![doc = include_str!("../docs/module.md")]
//! # Disclaimer
//!
//! This project would not exist without the work of several upstream authors and communities.
//! Specifically, this project was greatly inspired by the [C-Blosc2](https://github.com/Blosc/c-blosc2) library.
//! This crate can almost be seen as a port of ideas and natural Rust evolution of C-Blosc2.
//! See the `THANKS.md` at the repository root for a more complete list of contributors and inspirations,
//! and the `NOTICE` file for full attribution and license text.

use pyo3::prelude::*;

mod archive;
mod array;
mod codec;
mod dtype;
mod ops;
mod util;

#[doc = include_str!("../docs/module.md")]
#[pymodule]
mod jix {
    /// The version of the jix library.
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    pub const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    pub use crate::{array::Array, codec::ReadContext};

    #[pymodule_export]
    pub use crate::array::compact;

    #[pymodule_export]
    pub use crate::archive::{read_array, write_array};

    #[pymodule_export]
    pub use crate::ops::{asarray, astype};

    #[pymodule_export]
    pub use crate::ops::{add, divide, floor_divide, multiply, power, subtract};

    #[pymodule_export]
    pub use crate::ops::{
        bitwise_and, bitwise_left_shift, bitwise_not, bitwise_or, bitwise_right_shift,
        bitwise_rotate_left, bitwise_rotate_right, bitwise_xor, count_ones, count_zeros,
        leading_zeros, logical_and, logical_not, logical_or, logical_xor, reverse_bits, swap_bytes,
        trailing_zeros,
    };

    #[pymodule_export]
    pub use crate::ops::r#where;

    #[pymodule_export]
    pub use crate::ops::{
        equal, greater, greater_equal, less, less_equal, maximum, minimum, not_equal,
    };

    #[pymodule_export]
    pub use crate::ops::{
        broadcast, concatenate, flatten, flip, insert_axis, permute_axes, remove_axis, repeat,
        reshape, roll, slice, squeeze, stack, tile, unsqueeze,
    };

    #[pymodule_export]
    pub use crate::ops::{
        absolute, acos, asin, atan, ceil, cos, exp, floor, log, negative, round, sign, sin, sqrt,
        tan,
    };

    #[pymodule_export]
    pub use crate::ops::{is_finite, is_infinite, is_nan};

    #[pymodule_export]
    pub use crate::ops::{all, any, argmax, argmin, max, mean, min, product, std, sum, var};

    #[pymodule_export]
    pub use crate::ops::dtype_sub_field;

    #[pymodule_export]
    pub use crate::ops::{imag, real};

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        use pyo3::prelude::*;

        // aliases
        m.add("pow", m.getattr("power")?)?;
        m.add("abs", m.getattr("absolute")?)?;
        m.add("concat", m.getattr("concatenate")?)?;

        Ok(())
    }
}
pub use crate::jix::*;

pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::type_info::ModuleDocInfo {
        module: "jix",
        doc: {
            fn _fmt() -> String {
                include_str!("../docs/module.md").to_string()
            }
            _fmt
        }
    }
}

// TODO: pyo3 stub doesn't generate a docstring for constants.
pyo3_stub_gen::module_variable!("jix", "__version__", String);

#[doc(hidden)]
pub mod __private {
    use pyo3_stub_gen::define_stub_info_gatherer;

    define_stub_info_gatherer!(generate_pyi);
}
