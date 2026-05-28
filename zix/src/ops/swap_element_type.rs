use std::marker::PhantomData;

use crate::error::Result;
use crate::storage::{ArrayStorage, ElementType, TypeDyn};
use crate::util::assert_unchecked_eq;
use crate::{Array, Error, ErrorKind};

/// A storage adapter that re-tags an existing array with a specific [`ElementType`].
///
/// `SwapType<S, ET>` wraps `Array<S>` and declares `type ElementType = ET`, overriding whatever
/// element type `S` reported. This is a zero-overhead wrapper: all reads are forwarded directly to
/// the inner storage without any data transformation.
///
/// The only work done at construction time is a dtype check: if `ET::DTYPE = Some(dtype)`,
/// `SwapType::new` verifies that the array's actual dtype matches and returns an error if not.
/// If `ET = TypeDyn` (`ET::DTYPE = None`), the check is skipped and construction always succeeds.
///
/// # Purpose
///
/// Arrays loaded from files carry [`TypeDyn`] as their element type because the concrete dtype is
/// not known until the archive header is parsed. `SwapType` lets callers assert "I know this array
/// holds `f32`" and recover static tracking so that subsequent operations — which require
/// [`ArrayStorageTyped`](crate::storage::ArrayStorageTyped) — become available.
///
/// The primary entry points on [`Array`] are:
/// - [`Array::into_typed::<T>()`](crate::Array::into_typed) — validates and upgrades to `Ty<T>`
/// - [`Array::into_type_dyn()`](crate::Array::into_type_dyn) — downgrades to `TypeDyn`
///
/// For concrete block-compressed storages ([`Compact`](crate::storage::Compact),
/// [`CompactMmap`](crate::storage::CompactMmap)) consider
/// [`Array::swap_element_type`](crate::Array::swap_element_type) instead — it replaces the
/// storage's `ET` parameter in-place and avoids adding this wrapper layer.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use zix::{Array, ArrayParams};
///
/// // read_from_file returns Array<Compact<TypeDyn, DimDyn>>: element type unknown at compile time.
/// let a = Array::read_from_file(Path::new("data.zix"), ArrayParams::default())?;
///
/// // into_typed wraps the storage in SwapType, asserting the element type is f32 at runtime.
/// let typed = a.into_typed::<f32>()?;
///
/// // Subsequent operations requiring ArrayStorageTyped are now available.
/// let result = typed.exp().cast::<f64>().copy()?;
/// # Ok::<(), zix::Error>(())
/// ```
pub struct SwapType<S, ET> {
    inner: Array<S>,
    element_type: PhantomData<ET>,
}
impl<S, ET> SwapType<S, ET> {
    /// Wrap `array` in a `SwapType<S, ET>`, validating the dtype against `ET`.
    ///
    /// If `ET::DTYPE = Some(dtype)` and the array's actual dtype does not match, returns an error.
    /// If `ET = TypeDyn` (`ET::DTYPE = None`), the check is skipped and construction always succeeds.
    pub fn new(array: Array<S>) -> Result<Self>
    where
        S: ArrayStorage,
        ET: ElementType,
    {
        if let Some(expected_dtype) = ET::DTYPE {
            let actual_dtype = array.dtype();
            if actual_dtype != &expected_dtype {
                return Err(Error::new(
                    ErrorKind::InvalidShapeOperation,
                    format!(
                        "Cannot convert array with dtype {actual_dtype:?} to expected dtype {expected_dtype:?}"
                    ),
                ));
            }
        }
        Ok(Self {
            inner: array,
            element_type: PhantomData,
        })
    }
}
impl<S, ET> ArrayStorage for SwapType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    type ElementType = ET;
    type Dimension = S::Dimension;

    fn read_data(
        &self,
        index: &[core::ops::Range<u64>],
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> crate::error::Result<()> {
        if let Some(dtype) = ET::DTYPE {
            unsafe { assert_unchecked_eq!(self.inner.storage.dtype(), &dtype) };
        }
        self.inner.storage.read_data(index, buf, context)
    }
    fn shape(&self) -> &[u64] {
        self.inner.storage.shape()
    }
    fn dtype(&self) -> &crate::dtype::Dtype {
        let dtype = self.inner.storage.dtype();
        if let Some(expected_dtype) = ET::DTYPE {
            unsafe { assert_unchecked_eq!(dtype, &expected_dtype) };
        }
        dtype
    }
    fn _spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
        self.inner.storage._spec()
    }
}

/// In-place element-type re-tagging for concrete storage backends.
///
/// This trait is the element-type analogue of `SwapDimInplace`. It allows concrete storage types
/// ([`Compact`](crate::storage::Compact), [`Plain`](crate::storage::Plain), etc.) to change their
/// `Ty` type parameter directly, without wrapping in an outer [`SwapType`] layer. The result is the
/// same storage type with a different element-type tag rather than a new wrapper type.
///
/// The primary entry points are [`Array::swap_element_type`] and [`Array::swap_element_type_dyn`].
pub trait SwapElementTypeInplace: ArrayStorage {
    type SwapElementType<NewET: ElementType>: ArrayStorage<ElementType = NewET>;

    fn swap_element_type<NewET: ElementType>(self) -> Result<Self::SwapElementType<NewET>>;
}
impl<S> Array<S> {
    /// Re-tag the array's element type to `NewET`, validating the dtype at runtime.
    ///
    /// This calls [`SwapElementTypeInplace::swap_element_type`] on the inner storage, changing its
    /// `ElementType` parameter in-place. If `NewET::DTYPE = Some(dtype)` and the actual dtype does
    /// not match, an error is returned. If `NewET = TypeDyn`, the check is skipped.
    ///
    /// Prefer [`Array::into_typed`](crate::Array::into_typed) for the common case of upgrading a
    /// `TypeDyn` array to a specific `Ty<T>`.
    pub fn swap_element_type<NewET: ElementType>(self) -> Result<Array<S::SwapElementType<NewET>>>
    where
        S: SwapElementTypeInplace,
    {
        Ok(Array::from_storage(
            self.into_storage().swap_element_type()?,
        ))
    }

    /// Re-tag the array's element type to [`TypeDyn`], discarding compile-time element tracking.
    ///
    /// This is the infallible downgrade path: the dtype value is preserved but the static
    /// `Ty<T>` marker is replaced with `TypeDyn`. Always succeeds.
    pub fn swap_element_type_dyn(self) -> Array<S::SwapElementType<TypeDyn>>
    where
        S: SwapElementTypeInplace,
    {
        self.swap_element_type::<TypeDyn>().unwrap()
    }
}
