use std::marker::PhantomData;

use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};
use crate::util::assert_unchecked_eq;

/// Compile-time element-type tracking for [`ArrayStorage`](crate::ArrayStorage).
///
/// Every [`ArrayStorage`](crate::ArrayStorage) has an associated `type ElementType: ElementType`. There are two
/// implementors:
///
/// - [`Ty<T>`] - the concrete element type `T` is known at compile time. All element-wise
///   operations (arithmetic, comparisons, reductions, cast) are available.
/// - [`TypeDyn`] - the element type is only available at runtime. Arrays loaded from disk
///   start with this. Call [`Array::to_typed::<T>()`](crate::Array::to_typed) to assert
///   the expected element type and recover compile-time tracking.
pub trait ElementType: Clone + Send + Sync {
    /// `Some(dtype)` when the element type is statically known ([`Ty<T>`]),
    /// `None` for [`TypeDyn`].
    const DTYPE: Option<Dtype>;

    /// Construct from a runtime `Dtype`, validating it against `DTYPE`.
    ///
    /// Returns an error if `Self::DTYPE = Some(d)` and `dtype != d`.
    /// Always succeeds for `TypeDyn`.
    fn from_dtype(dtype: Dtype) -> Result<Self>
    where
        Self: Sized;

    /// Returns the element dtype, either the fixed dtype or the runtime one.
    fn dtype(&self) -> &Dtype;
}

/// Runtime-only element type tag. `S::ElementType = TypeDyn` when the element type is not
/// known at compile time (e.g. arrays loaded from a `.jix` file).
///
/// Arrays with `TypeDyn` do not support most element-wise operations directly; call
/// [`Array::to_typed::<T>()`](crate::Array::to_typed) first to assert the expected
/// element type and recover [`ArrayStorageTyped`](crate::ArrayStorage).
#[derive(Clone)]
pub struct TypeDyn(Dtype);
impl ElementType for TypeDyn {
    const DTYPE: Option<Dtype> = None;

    #[inline(always)]
    fn from_dtype(dtype: Dtype) -> Result<Self> {
        Ok(Self(dtype))
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        &self.0
    }
}

/// Compile-time element type tag. `S::ElementType = Ty<T>` when the scalar element type
/// `T` is statically known.
///
/// `Ty<T>` enables all element-wise operations: arithmetic, comparisons, reductions, and
/// type casts. Arrays constructed from typed sources (e.g.
/// [`Array::compact_array`](crate::Array::compact_array)) automatically carry `Ty<T>`.
#[derive(Clone)]
pub struct Ty<T>(Dtype, PhantomData<T>);
impl<T> Ty<T> {
    /// Construct the element type marker.
    #[allow(clippy::new_without_default)]
    #[inline(always)]
    pub fn new() -> Self
    where
        T: Dtyped,
    {
        Self(T::DTYPE, PhantomData)
    }
}
impl<T> ElementType for Ty<T>
where
    T: Dtyped,
{
    const DTYPE: Option<Dtype> = Some(T::DTYPE);

    #[inline(always)]
    fn from_dtype(dtype: Dtype) -> Result<Self> {
        check_dtype(&dtype, &T::DTYPE)?;
        Ok(Self::new())
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        unsafe { assert_unchecked_eq!(self.0, T::DTYPE) };
        &self.0
    }
}
