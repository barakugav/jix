//! Array operations.
//!
//! Every operation returns a new [`Array<S>`](crate::Array) where `S` is a struct that implements
//! [`StorageBase`](crate::StorageBase). The storage wraps the input array(s) and
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
//!     .compact()?;             // Array<Compact> - materialize the pipeline
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
//! To avoid this, consider materializing the array after a shape change:
//!
//! * [`.compact()`](crate::Array::compact) - re-encodes with a block shape derived automatically
//!   from the new shape and the original block shape.
//! * [`.compact_with(params, ...)`](crate::Array::compact_with) - re-encodes with an explicit
//!   [`CompactParams`](crate::storage::CompactParams), giving full control over the new block
//!   shape. Use this to guarantee your access pattern is well-aligned.
//! * [`.to_plain()`](crate::Array::to_plain) - materializes into an uncompressed in-memory buffer.
//!
//! [`Array::reshape`](crate::Array::reshape) is the operation most prone to this cost. Being a
//! lazy view, it pays the cost on every read, so call `.compact()` on the result when it will be
//! read more than once.
//!
//! # Typed element requirements
//!
//! Element-wise operations - arithmetic, comparisons, reductions, bitwise ops, type casting - all
//! require the input storage to be *typed*: the element type must be known at compile time, not
//! just at runtime. Concretely, the input must satisfy
//! [`Storage`](crate::Storage), which is a shorthand for
//! `StorageBase<ElementType = T>` for some concrete `T: Dtyped`.
//!
//! Arrays constructed from typed sources are typed automatically:
//!
//! ```
//! use jix::Array;
//! use ndarray::array;
//!
//! // compact_ndarray returns Array<Compact<f32>>: automatically typed.
//! let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
//! let b = a.exp();        // fine: f32: Exp
//! let c = b.cast::<i32>(); // fine: f32: Cast<i32>
//! # Ok::<(), jix::Error>(())
//! ```
//!
//! Arrays loaded from disk carry [`TypeDyn`](crate::TypeDyn) because the element type is only
//! known at runtime. Use [`Array::into_typed`](crate::Array::into_typed) to assert the expected
//! element type and recover compile-time tracking:
//!
//! ```no_run
//! use std::path::Path;
//!
//! use jix::Array;
//!
//! let src = Array::read_from_file(Path::new("data.jix"))?;
//! // src is Array<Compact<TypeDyn>> - element-wise ops not yet available
//!
//! // validates the dtype at runtime
//! let typed = src.into_typed::<f32>()?;
//! // typed is Array<Compact<f32>> - ops available
//! let result = typed.exp().cast::<f64>().compact()?;
//! # Ok::<(), jix::Error>(())
//! ```
//!
//! Each op additionally requires the input element type to implement the relevant scalar trait
//! of [`jix::scalar`](crate::scalar). For example, `a + b` requires
//! [`jix::scalar::Add`](crate::scalar::Add), `exp()` requires
//! [`jix::scalar::Exp`](crate::scalar::Exp), and `sum()` requires
//! [`jix::scalar::Sum`](crate::scalar::Sum).
//!
//! # Multi-array operations
//!
//! Operations that accept a variable number of input arrays - [`stack`] and [`concatenate`] - take
//! an [`ArraySequence`](crate::storage::ArraySequence) argument. This covers `Vec<Array<S>>`,
//! `&[Array<S>]`, fixed-length arrays `[Array<S>; N]`, and heterogeneous tuples of up to 12
//! arrays. See [`ArraySequence`](crate::storage::ArraySequence) for details.

// Internal only, with no caller yet outside its tests.
#[allow(dead_code)]
mod maybe_compact;

mod shape_ops;
pub use shape_ops::*;

mod elementwise;
pub use elementwise::*;

mod reduction;
pub use reduction::*;

mod sub_dtype;
pub use sub_dtype::*;

mod transmute;
pub use transmute::*;

mod dims_arg;
pub use dims_arg::DimsArg;

pub(crate) mod traits {
    pub use super::elementwise::traits::*;
    pub use super::reduction::traits::*;
}

mod into_type;
pub use into_type::*;

pub(crate) mod prelude {
    pub(crate) use std::ops::Range;

    pub(crate) use super::{
        DimsArg, ElementwisePipeline, ElementwisePipelineImpl, InnerLoops, Operand, max_itemsize,
        n_operands_sum,
    };
    pub(crate) use crate::context::{CacheScope, ReadContext};
    pub(crate) use crate::dtype::{Alignment, Dtype, Dtyped, Itemsize};
    pub(crate) use crate::error::*;
    pub(crate) use crate::scalar::bool8;
    pub(crate) use crate::storage::cache_id::ArrayCacheId;
    pub(crate) use crate::storage::params::{
        ArraySpecDynamic, ArraySpecFlags, ReadTiling, combine_block_layout,
        combine_elementwise_hints, combine_select_hints, read_layout_order_insert_dont_care_dim,
    };
    pub(crate) use crate::storage::{
        ArraySpec, ArrayStorageExt, ArrayStorageInfo, BlockSize, Index, OutBuf, OutBufInner,
        StridedBuf, check_out_buf, read_data_and_map_strides, write_out_buf_slot,
    };
    pub(crate) use crate::util::iter::NdIter;
    pub(crate) use crate::util::{
        ArrayExt, DimArray, DimIdx, IterExt, NdCopier, NdIterUnordered, PtrExt, PtrMutExt,
        PtrMutNoalias, PtrNoalias, REQUIRE_ALIGNED, ScaleWeight, SliceExt, array_from_fn_inline,
        calc_block_end, default_logical_strides, default_strides, default_strides_from_iter,
        dim_arr, try_dim_arr,
    };
    pub(crate) use crate::{
        Array, DimDyn, DimVec, Dimension, ElementType, NDIM_MAX, Storage, StorageBase, TypeDyn,
    };
}
