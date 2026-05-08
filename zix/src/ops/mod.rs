//! Array operations.
//!
//! Every operation returns a new [`Array<S>`](crate::Array) where `S` is a struct that implements
//! [`ArrayStorage`](crate::storage::ArrayStorage). The storage wraps the input array(s) and
//! transforms read requests on demand - no data is copied at construction time. An operation only
//! executes when its result is materialized, e.g. via [`.to_ndarray()`](crate::Array::to_ndarray)
//! or [`.copy()`](crate::Array::copy).
//!
//! # Operation chains
//!
//! Because each operation is just a generic wrapper, chains compose at the type level:
//!
//! ```text
//! let result = array           // Array<Compact>
//!     .astype::<f32>()         // Array<AsType<Compact>>
//!     .floor()                 // Array<Floor<AsType<Compact>>>
//!     .exp()                   // Array<Exp<Floor<AsType<Compact>>>>
//!     .copy();                 // Array<Compact> - materialize the pipeline
//! ```
//!
//! The compiler sees through all the wrappers and can inline the entire pipeline into a single
//! read loop, with no intermediate heap allocations beyond the final output buffer. There is no
//! runtime evaluation graph or scheduler - the type system *is* the execution plan.
//! The idea is to let the compiler be a competitive alternative to complex evaluation engines.
//!
//! # Shape-changing operations and performance
//!
//! Operations such as [`Reshape`] and [`Broadcast`] remap how output indices translate to
//! positions in the underlying blocks. When the new layout crosses block boundaries that the
//! original layout respected, a single read request may decompress many more blocks than necessary,
//! each discarding most of their data.
//!
//! To avoid this, materialize the array after a shape change:
//!
//! * [`.copy()`](crate::Array::copy) - re-encodes with a block shape derived automatically from
//!   the new shape and the original block shape.
//! * [`.copy_with(params, ...)`](crate::Array::copy_with) - re-encodes with an explicit
//!   [`ArrayParams`](crate::ArrayParams), giving full control over the new block shape. Use this
//!   to guarantee your access pattern is well-aligned.
//!
//! The eager variants ([`Array::reshape`](crate::Array::reshape),
//! [`Array::broadcast`](crate::Array::broadcast)) call `.copy()` internally.
//! Use the `_view` variants with care.
//!
//! # Multi-array operations
//!
//! Operations that accept a variable number of input arrays - [`stack`] and [`concatenate`] - take
//! an [`ArraySequence`](crate::util::ArraySequence) argument. This covers `Vec<Array<S>>`,
//! `&[Array<S>]`, fixed-length arrays `[Array<S>; N]`, and heterogeneous tuples of up to ten
//! arrays. See [`ArraySequence`](crate::util::ArraySequence) for details.

mod into_compact;
pub use into_compact::*;

mod astype;
#[allow(unused_imports)]
pub(crate) use astype::cast;
pub use astype::AsType;

mod shape_ops;
pub use shape_ops::*;

mod op1;
pub use op1::*;

mod op2;
pub use op2::*;

mod logical1;
pub use logical1::*;

mod cmp;
pub use cmp::*;

mod bitwise;
pub use bitwise::*;

mod reduction;
pub use reduction::*;

mod map;
pub use map::*;

mod where_op;
pub use where_op::*;

mod sub_dtype;
pub use sub_dtype::*;

mod common;

#[doc(hidden)]
pub mod __private {
    pub use super::astype::{cast, Cast};
}
