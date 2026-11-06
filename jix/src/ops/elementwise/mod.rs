//! Element-wise operations: the `Op1` / `Op2` engines and the op families built on them.

mod op1;
pub(crate) use op1::*;

mod op2;
pub(crate) use op2::*;

mod arith;
pub use arith::*;

mod div_rem;
pub use div_rem::*;

mod rounding;
pub use rounding::*;

mod exp_log;
pub use exp_log::*;

mod trig;
pub use trig::*;

mod cast;
pub use cast::Cast;

mod float_special;
pub use float_special::*;

mod cmp;
pub use cmp::*;

mod min_max;
pub use min_max::*;

mod bitwise;
pub use bitwise::*;

mod bit_count;
pub use bit_count::*;

mod map;
pub use map::*;

mod select;
pub use select::*;

#[cfg(feature = "num-complex")]
mod complex;
#[cfg(feature = "num-complex")]
pub use complex::*;

pub(crate) mod traits {
    pub use super::arith::traits::*;
    pub use super::bit_count::traits::*;
    pub use super::bitwise::traits::*;
    pub use super::cast::traits::*;
    pub use super::cmp::traits::*;
    pub use super::div_rem::traits::*;
    pub use super::exp_log::traits::*;
    pub use super::float_special::traits::*;
    pub use super::min_max::traits::*;
    pub use super::rounding::traits::*;
    pub use super::trig::traits::*;
}
