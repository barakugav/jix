use std::marker::PhantomData;

use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};

/// Compile-time element-type tracking: the bound on the `ElementType` associated type of
/// [`ArrayStorage`](crate::ArrayStorage). Its two implementors:
///
/// - [`Ty<T>`] - the concrete element type `T` is known at compile time. All element-wise
///   operations (arithmetic, comparisons, reductions, cast) are available.
/// - [`TypeDyn`] - the element type is only available at runtime. Arrays loaded from disk
///   start with this. Call [`Array::into_typed::<T>()`](crate::Array::into_typed) to assert
///   the expected element type and recover compile-time tracking.
///
/// This trait is sealed: it cannot be implemented outside this crate.
pub trait ElementType: Clone + Send + Sync + crate::util::Sealed {
    /// `Some(dtype)` when the element type is statically known ([`Ty<T>`]),
    /// `None` for [`TypeDyn`].
    const DTYPE: Option<Dtype>;
}

/// Check that `dtype` matches `ET`: always true for [`TypeDyn`].
pub(crate) fn check_element_type<ET: ElementType>(dtype: &Dtype) -> Result<()> {
    match &ET::DTYPE {
        Some(expected) => check_dtype(dtype, expected),
        None => Ok(()),
    }
}

/// Runtime-only element type tag. `S::ElementType = TypeDyn` when the element type is not
/// known at compile time (e.g. arrays loaded from a `.jix` file).
///
/// Arrays with `TypeDyn` do not support most element-wise operations directly; call
/// [`Array::into_typed::<T>()`](crate::Array::into_typed) first to assert the expected
/// element type and recover [`ArrayStorageTyped`](crate::storage::ArrayStorageTyped).
#[derive(Clone)]
pub enum TypeDyn {}
impl crate::util::Sealed for TypeDyn {}
impl ElementType for TypeDyn {
    const DTYPE: Option<Dtype> = None;
}

/// Compile-time element type tag. `S::ElementType = Ty<T>` when the scalar element type
/// `T` is statically known.
///
/// `Ty<T>` enables all element-wise operations: arithmetic, comparisons, reductions, and
/// type casts. Arrays constructed from typed sources (e.g.
/// [`Array::compact_ndarray`](crate::Array::compact_ndarray)) automatically carry `Ty<T>`.
#[derive(Clone)]
pub struct Ty<T>(PhantomData<T>);
impl<T> Ty<T> {
    /// Construct the element type marker.
    #[allow(clippy::new_without_default)]
    #[inline(always)]
    pub fn new() -> Self
    where
        T: Dtyped,
    {
        Self(PhantomData)
    }
}
impl<T> crate::util::Sealed for Ty<T> {}
impl<T> ElementType for Ty<T>
where
    T: Dtyped,
{
    const DTYPE: Option<Dtype> = Some(T::DTYPE);
}
