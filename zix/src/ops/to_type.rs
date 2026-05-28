use std::marker::PhantomData;

use crate::error::Result;
use crate::ArrayStorage;
use crate::util::assert_unchecked_eq;
use crate::{Array, ElementType, Error, ErrorKind};

/// A lazy storage adapter that re-tags an array's element-type parameter without copying data.
///
/// `ToType<S, ET>` wraps an [`Array<S>`](crate::Array) and presents it as having element type
/// `ET`, without touching the underlying bytes. At construction time, if `ET` is a concrete type
/// ([`Ty<T>`](crate::Ty)), the runtime [`Dtype`](crate::dtype::Dtype) is validated to
/// match `T`; if it is [`TypeDyn`](crate::TypeDyn) the conversion always succeeds.
///
/// The typical entry points are [`Array::to_type`](crate::Array::to_type),
/// [`Array::to_typed`](crate::Array::to_typed), and
/// [`Array::to_type_dyn`](crate::Array::to_type_dyn).
///
/// For concrete block-compressed or plain storages that implement [`ElementTypeChange`], prefer
/// [`Array::into_type`](crate::Array::into_type) instead — it re-tags the element type in-place
/// without adding this wrapper layer.
///
/// # Examples
///
/// Erase and recover the static element type:
///
/// ```
/// use zix::Array;
/// use zix::dtype::Dtyped;
/// use ndarray::array;
///
/// // compact_array infers Ty<f32> from the ndarray input type.
/// let a = Array::compact_array(&array![1.0f32, 2.0, 3.0])?;
/// // a: Array<Compact<Ty<f32>, Dim<1>>>
///
/// // Erase the static element type — always succeeds.
/// let dyn_a = a.to_type_dyn();
/// // dyn_a: Array<ToType<Compact<Ty<f32>, Dim<1>>, TypeDyn>>
/// // dyn_a: Array<Storage::ElementType = TypeDyn>
/// assert_eq!(dyn_a.dtype(), &f32::DTYPE);
///
/// // Recover the concrete type — validated at runtime.
/// let typed_a = dyn_a.to_typed::<f32>()?;
/// // typed_a: Array<ToType<ToType<..., TypeDyn>, Ty<f32>>>
/// // typed_a: Array<Storage::ElementType = Ty<f32>>
///
/// // Element-wise operations are available again.
/// let result = typed_a.exp().to_ndarray::<f32>()?;
/// # Ok::<(), zix::Error>(())
/// ```
pub struct ToType<S, ET> {
    inner: Array<S>,
    element_type: PhantomData<ET>,
}
impl<S, ET> ToType<S, ET> {
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
impl<S, ET> ArrayStorage for ToType<S, ET>
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

/// Opt-in trait for storage types that can swap their element-type parameter in-place.
///
/// Storages that implement this trait can change their [`ElementType`] without wrapping
/// themselves in a [`ToType`] adapter. The result is a leaner type: e.g. `Compact<NewET, D>`
/// instead of `ToType<Compact<ET, D>, NewET>`.
///
/// Implementors: [`Compact`](crate::storage::Compact),
/// [`CompactMmap`](crate::storage::CompactMmap),
/// [`CompactBorrowed`](crate::storage::CompactBorrowed),
/// [`Plain`](crate::storage::Plain).
///
/// The typical entry points are [`Array::into_type`](crate::Array::into_type),
/// [`Array::into_typed`](crate::Array::into_typed), and
/// [`Array::into_type_dyn`](crate::Array::into_type_dyn). Use
/// [`Array::to_type`](crate::Array::to_type) / [`Array::to_typed`](crate::Array::to_typed) for
/// storage types that do not implement this trait.
pub trait ElementTypeChange {
    /// The concrete storage type after swapping the element type to `NewET`.
    type ElementTypeChange<NewET: ElementType>: ArrayStorage<ElementType = NewET>
    where
        Self: ArrayStorage;

    /// Consume `self`, validate the new element type against the runtime dtype, and return the
    /// re-tagged storage.
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `NewET = Ty<T>` and the runtime dtype does not equal `T::DTYPE`. Always succeeds for
    /// `NewET = TypeDyn`.
    fn change_type<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>>
    where
        Self: ArrayStorage;
}
