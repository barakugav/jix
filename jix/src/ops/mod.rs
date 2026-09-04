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
//! To avoid this, materialize the array after a shape change:
//!
//! * [`.compact()`](crate::Array::compact) - re-encodes with a block shape derived automatically from
//!   the new shape and the original block shape.
//! * [`.compact_with(params, ...)`](crate::Array::compact_with) - re-encodes with an explicit
//!   [`ArrayParams`](crate::ArrayParams), giving full control over the new block shape. Use this
//!   to guarantee your access pattern is well-aligned.
//!
//! Like every other operation, [`Array::reshape`](crate::Array::reshape) is a lazy view, so this
//! cost is paid only when the view is read. Reshape is the operation most prone to it - call
//! `.compact()` on the result when it will be read more than once.
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
//! // compact_ndarray returns Array<Compact<Ty<f32>>>: automatically typed.
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
//! // src is Array<Compact<TypeDyn>> - ops not yet available
//!
//! // validates the dtype at runtime
//! let typed = src.into_typed::<f32>()?;
//! // typed is Array<Compact<Ty<f32>>> - ops available
//! let result = typed.exp().cast::<f64>().compact()?;
//! # Ok::<(), jix::Error>(())
//! ```
//!
//! Each op additionally requires the input element type to implement the relevant scalar trait,
//! of [`jix::scalar`](crate::scalar). For example, `a + b` requires [`jix::scalar::Add`](crate::scalar::Add), `exp()`
//! requires [`jix::scalar::Exp`](crate::scalar::Exp), and `sum()` requires [`jix::scalar::Sum`](crate::scalar::Sum).
//!
//! # Multi-array operations
//!
//! Operations that accept a variable number of input arrays - [`stack`] and [`concatenate`] - take
//! an [`ArraySequence`](crate::util::ArraySequence) argument. This covers `Vec<Array<S>>`,
//! `&[Array<S>]`, fixed-length arrays `[Array<S>; N]`, and heterogeneous tuples of up to ten
//! arrays. See [`ArraySequence`](crate::util::ArraySequence) for details.

mod maybe_compact;
pub use maybe_compact::*;

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

    pub(crate) use super::DimsArg;
    pub(crate) use super::{
        max_itemsize, n_operands_sum, ElementwisePipeline, ElementwisePipelineImpl, Operand,
    };
    pub(crate) use crate::dtype::{Alignment, Dtype, Dtyped, Itemsize};
    pub(crate) use crate::error::*;
    pub(crate) use crate::scalar::bool8;
    pub(crate) use crate::storage::params::{
        combine_block_layout, combine_elementwise_hints, combine_select_hints,
        read_layout_order_insert_dont_care_dim, ArraySpecDynamic,
    };
    pub(crate) use crate::storage::{
        check_out_buf, materialize_out_buf, read_data_and_map_strides, ArraySpec, ArrayStorageInfo,
        ArrayStorageTyped, BlockSize, StridedBuf,
    };
    pub(crate) use crate::util::{
        array_from_fn_inline, calc_block_end, default_logical_strides, default_strides,
        default_strides_from_iter, dim_arr, iter::NdIter, try_dim_arr, ArrayExt, DimArray, DimIdx,
        IterExt, NdCopier, NdIterUnordered, PtrExt, PtrMutExt, PtrMutNoalias, PtrNoalias,
        ScaleWeight, SliceExt, REQUIRE_ALIGNED,
    };
    pub(crate) use crate::ReadContext;
    pub(crate) use crate::{
        Array, ArrayStorage, DimDyn, DimVec, Dimension, ElementType, Ty, TypeDyn, NDIM_MAX,
    };
}

#[cfg(test)]
mod storage_meta_tests {
    //! Every op repeats the same `ArrayStorage` boilerplate - `info`, `spec` and
    //! `element_type_change` - and an op's own tests never reach it, because they only read data
    //! back. One array per op is built here and run through the same checks.

    use crate::ops::{concatenate, map2, map_multiple, stack};
    use crate::scalar::bool8;
    use crate::storage::Compact;
    use crate::ReadContext;
    use crate::{Array, ArrayStorage, Ty};

    fn i32_2d() -> Array<Compact<Ty<i32>>> {
        Array::compact_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]]).unwrap()
    }

    fn f64_2d() -> Array<Compact<Ty<f64>>> {
        Array::compact_ndarray(&ndarray::array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap()
    }

    fn bool_2d() -> Array<Compact<Ty<bool8>>> {
        let (t, f) = (bool8::TRUE, bool8::FALSE);
        Array::compact_ndarray(&ndarray::array![[t, f, t], [f, t, f]]).unwrap()
    }

    /// Names of `storage` and, depth first, of everything it reads from.
    fn info_tree(storage: &dyn ArrayStorage, out: &mut Vec<String>) {
        let info = storage.info();
        out.push(info.name().to_string());
        for dep in info.dependencies() {
            info_tree(*dep, out);
        }
    }

    /// Assert the op's reported name and that its tree bottoms out at a leaf storage, then re-tag
    /// the element type and check the shape and dtype survived.
    fn check<S: ArrayStorage>(array: Array<S>, name: &str) {
        let mut names = Vec::new();
        info_tree(array.storage(), &mut names);
        assert_eq!(names[0], name);
        let leaf = names.last().unwrap().as_str();
        assert!(matches!(leaf, "Compact" | "Plain" | "Scalar"), "{names:?}");

        let shape = array.shape().to_vec();
        let dtype = array.dtype().clone();

        // Pull mode. The op's own tests always materialize into a packed buffer, which is the
        // push path; pull lets the storage hand back its own strides instead.
        let index: Vec<_> = shape.iter().map(|&n| 0..n).collect();
        let context = ReadContext::default();
        {
            let pulled = array.storage().read_data(&index, &context, None).unwrap();
            assert_eq!(pulled.strides().len(), shape.len());
        }

        let compacted = array.compact().unwrap();
        assert_eq!(compacted.shape(), &shape[..]);

        let retagged = array.into_type_dyn();
        assert_eq!(retagged.shape(), &shape[..]);
        assert_eq!(retagged.dtype(), &dtype);
    }

    #[test]
    fn shape_ops() {
        check(i32_2d().reshape(&[3, 2]), "Reshape");
        check(i32_2d().slice((0..1, 0..2)), "Slice");
        check(i32_2d().permute_dims(&[1, 0]), "PermuteDims");
        check(i32_2d().repeat(2, 0), "Repeat");
        check(i32_2d().flip(0), "Flip");
        check(i32_2d().roll(1, 0), "Roll");
        check(i32_2d().tile(2, 0), "Tile");
        check(i32_2d().reshape(&[1, 6]).broadcast(&[4, 6]), "Broadcast");
        check(i32_2d().insert_dim(0), "InsertDim");
        check(i32_2d().reshape(&[1, 2, 3]).remove_dim(0), "RemoveDim");
        check(concatenate([i32_2d(), i32_2d()], 0), "Concatenate");
        check(stack([i32_2d(), i32_2d()], 0), "Stack");
    }

    #[test]
    fn elementwise_ops() {
        check(f64_2d().floor(), "Floor");
        check(i32_2d().square(), "Square");
        check(i32_2d() + i32_2d(), "Add");
        check(i32_2d() - i32_2d(), "Sub");
        check(i32_2d() * i32_2d(), "Mul");
        check(i32_2d() / i32_2d(), "Div");
        check(f64_2d().pow(f64_2d()), "Pow");
        check(i32_2d().abs(), "Abs");
        check(i32_2d().equal(i32_2d()), "Equal");
        let shifts = Array::compact_ndarray(&ndarray::array![[1u32, 2, 3], [4, 5, 6]]).unwrap();
        check(i32_2d().bitwise_rotate_left(shifts), "BitwiseRotateLeft");
        check(i32_2d().cast::<i64>(), "Cast");
        check(i32_2d().map(|x| x as f32), "Map");
        check(map2(i32_2d(), i32_2d(), |a, b| a + b), "Map2");
        check(
            map_multiple([i32_2d(), i32_2d(), i32_2d()], |xs: [i32; 3]| {
                xs[0] + xs[1] + xs[2]
            }),
            "MapMultiple",
        );
        check(bool_2d().select(i32_2d(), i32_2d()), "Select");
        check(f64_2d().approx_equal(f64_2d(), 1e-9, 1e-9), "ApproxEq");
    }

    #[test]
    fn reductions() {
        check(i32_2d().sum(0), "Sum");
        check(i32_2d().reduce_unordered(0, |a, b| a + b), "Reduce");
    }

    #[test]
    fn wrapping_storages() {
        use crate::ops::IntoType;

        let base = i32_2d();
        let view = crate::storage::View::new(base.storage());
        check(Array::from_storage(view), "View");
        check(i32_2d().into_any(), "Any");
        check(i32_2d().maybe_compact().unwrap(), "MaybeCompact");
        check(unsafe { i32_2d().transmute_elements::<u32>() }, "Transmute");
        check(
            IntoType::<_, Ty<i32>>::new_array(i32_2d()).unwrap(),
            "IntoType",
        );
    }

    #[test]
    fn leaf_storages() {
        let plain = Array::plain_ndarray(ndarray::array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        check(plain, "Plain");
        let scalar =
            crate::__private::Scalar::new(7i32, &[2, 3], crate::ArrayParams::default()).unwrap();
        assert_eq!(*scalar.data(), 7);
        check(Array::from_storage(scalar), "Scalar");
    }

    #[test]
    fn struct_dtype_field() {
        #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
        #[dtyped(crate = "crate")]
        #[repr(C)]
        struct Pair {
            x: i32,
            y: i32,
        }

        let pts = ndarray::array![Pair { x: 1, y: 10 }, Pair { x: 2, y: 20 }];
        let array = Array::compact_ndarray(&pts).unwrap();
        check(array.dtype_sub_field::<i32>("x"), "SubDtype");
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn complex_ops() {
        use crate::scalar::Complex;

        let data = ndarray::array![Complex::new(1.0f32, 2.0), Complex::new(3.0, 4.0)];
        check(Array::compact_ndarray(&data).unwrap().real(), "Real");
        check(Array::compact_ndarray(&data).unwrap().imag(), "Imaginary");
    }
}
