//! Array operations.
//!
//! Every operation returns a new [`Array<S>`](crate::Array) where `S` is a struct that implements
//! [`ArrayStorage`](crate::ArrayStorage). The storage wraps the input array(s) and
//! transforms read requests on demand - no data is copied at construction time. An operation only
//! executes when its result is materialized, e.g. via [`.to_ndarray()`](crate::Array::to_ndarray)
//! or [`.compact()`](crate::Array::compact).
//!
//! # Operation chains
//!
//! Because each operation is just a generic wrapper, chains compose at the type level:
//!
//! ```text
//! let result = array           // Array<Compact>
//!     .cast::<f32>()           // Array<Cast<Compact>>
//!     .floor()                 // Array<Floor<Cast<Compact>>>
//!     .exp()                   // Array<Exp<Floor<Cast<Compact>>>>
//!     .compact();              // Array<Compact> - materialize the pipeline
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
//! * [`.compact()`](crate::Array::compact) - re-encodes with a block shape derived automatically from
//!   the new shape and the original block shape.
//! * [`.compact_with(params, ...)`](crate::Array::compact_with) - re-encodes with an explicit
//!   [`ArrayParams`](crate::ArrayParams), giving full control over the new block shape. Use this
//!   to guarantee your access pattern is well-aligned.
//!
//! The eager variant ([`Array::reshape`](crate::Array::reshape)) calls `.compact()` internally.
//! Use the `_view` variants with care.
//!
//! # Typed element requirements
//!
//! Element-wise operations - arithmetic, comparisons, reductions, bitwise ops, type casting - all
//! require the input storage to be *typed*: the element type must be known at compile time, not
//! just at runtime. Concretely, the input must satisfy
//! [`ArrayStorageTyped`](crate::storage::ArrayStorageTyped), which is a shorthand for
//! `ArrayStorage<ElementType = Ty<T>>` for some concrete `T: Dtyped`.
//!
//! Arrays constructed from typed sources are typed automatically:
//!
//! ```
//! use jix::Array;
//! use ndarray::array;
//!
//! // compact_ndarray returns Array<Compact<Ty<f32>, ...>>: automatically typed.
//! let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
//! let b = a.exp();        // fine: f32: Exp
//! let c = b.cast::<i32>(); // fine: f32: Cast<i32>
//! # Ok::<(), jix::Error>(())
//! ```
//!
//! Arrays loaded from disk carry [`TypeDyn`](crate::TypeDyn) because the element type
//! comes from the file header. Use [`Array::to_typed`](crate::Array::to_typed) to assert the
//! expected element type and recover compile-time tracking:
//!
//! ```no_run
//! use std::path::Path;
//!
//! use jix::{Array, ArrayParams};
//!
//! let src = Array::read_from_file(Path::new("data.jix"), ArrayParams::default())?;
//! // src is Array<Compact<TypeDyn, DimDyn>> - ops not yet available
//!
//! let typed = src.into_typed::<f32>()?; // validates dtype at runtime
//! let result = typed.exp().cast::<f64>().compact()?;
//! # Ok::<(), jix::Error>(())
//! ```
//!
//! Each op additionally requires the input element type to implement the relevant scalar trait,
//! such as traits in [`core::ops`], [`num_traits`] or [`jix::scalar`](crate::scalar).
//! For example, `add()` requires `core::ops::Add`, `exp()` requires `num_traits::Float`, and
//! `sum()` requires `crate::scalar::Sum`.
//!
//! # Multi-array operations
//!
//! Operations that accept a variable number of input arrays - [`stack`] and [`concatenate`] - take
//! an [`ArraySequence`](crate::util::ArraySequence) argument. This covers `Vec<Array<S>>`,
//! `&[Array<S>]`, fixed-length arrays `[Array<S>; N]`, and heterogeneous tuples of up to ten
//! arrays. See [`ArraySequence`](crate::util::ArraySequence) for details.

mod maybe_compact;
pub use maybe_compact::*;

mod cast;
pub use cast::Cast;

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
pub use common::AxesArg;
pub(crate) use common::BulkInfo;

pub(crate) mod _traits {
    pub use super::cast::_traits::*;
    pub use super::cmp::_traits::*;
    pub use super::op1::_traits::*;
    pub use super::reduction::_traits::*;
}

mod to_type;
pub use to_type::*;

mod to_dim;
pub use to_dim::*;
