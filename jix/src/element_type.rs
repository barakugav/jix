use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};

/// Compile-time element-type tracking: the bound on the `ElementType` associated type of
/// [`ArrayStorage`](crate::ArrayStorage). Its implementors:
///
/// - every [`Dtyped`] type `T` - the element type is known at compile time. All element-wise
///   operations (arithmetic, comparisons, reductions, cast) are available.
/// - [`TypeDyn`] - the element type is only available at runtime. Arrays loaded from disk
///   start with this. Call [`Array::into_typed::<T>()`](crate::Array::into_typed) to assert
///   the expected element type and recover compile-time tracking.
///
/// This trait is sealed: it cannot be implemented for other types.
pub trait ElementType: Clone + Send + Sync + crate::util::Sealed {
    /// `Some(T::DTYPE)` for a [`Dtyped`] type `T`, `None` for [`TypeDyn`].
    const STATIC_DTYPE: Option<Dtype>;
}

impl<T: Dtyped> crate::util::Sealed for T {}
impl<T: Dtyped> ElementType for T {
    const STATIC_DTYPE: Option<Dtype> = Some(T::DTYPE);
}

/// Check that `dtype` matches `ET`: always true for [`TypeDyn`].
pub(crate) fn check_element_type<ET: ElementType>(dtype: &Dtype) -> Result<()> {
    match &ET::STATIC_DTYPE {
        Some(expected) => check_dtype(dtype, expected),
        None => Ok(()),
    }
}

/// Runtime-only element type marker. `S::ElementType = TypeDyn` when the element type is not
/// known at compile time (e.g. arrays loaded from a `.jix` file). It is only used as a type: the
/// dtype itself is kept by the storage.
///
/// Arrays with `TypeDyn` do not support most element-wise operations directly; call
/// [`Array::into_typed::<T>()`](crate::Array::into_typed) first to assert the expected
/// element type and recover [`ArrayStorageTyped`](crate::storage::ArrayStorageTyped).
#[derive(Clone)]
#[non_exhaustive]
pub struct TypeDyn;
impl crate::util::Sealed for TypeDyn {}
impl ElementType for TypeDyn {
    const STATIC_DTYPE: Option<Dtype> = None;
}
