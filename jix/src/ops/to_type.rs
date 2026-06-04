use std::marker::PhantomData;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtyped;
use crate::error::{check_dtype, Result};
use crate::storage::ReadData;
use crate::util::assert_unchecked_eq;
use crate::{Array, ArrayStorage, ElementType};

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
/// use jix::Array;
/// use jix::dtype::Dtyped;
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
/// let result = typed_a.exp().to_ndarray()?;
/// # Ok::<(), jix::Error>(())
/// ```
pub struct ToType<S, ET> {
    inner: S,
    element_type: PhantomData<ET>,
}
impl<S, ET> ToType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    /// Constructs a [`ToType`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> Result<Self> {
        if let Some(expected_dtype) = ET::DTYPE {
            check_dtype(array.dtype(), &expected_dtype)?;
        }
        Ok(Self {
            inner: array,
            element_type: PhantomData,
        })
    }

    /// Constructs an array with [`ToType`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, ET> ArrayStorage for ToType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    type ElementType = ET;
    type Dimension = S::Dimension;

    #[inline(always)]
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        if let Some(dtype) = ET::DTYPE {
            unsafe { assert_unchecked_eq!(self.inner.dtype(), &dtype) };
        }
        self.inner.read_data(index, buf, context)
    }

    #[inline(always)]
    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, S, ET>>
    where
        T: Dtyped,
    {
        if let Some(dtype) = ET::DTYPE {
            unsafe { assert_unchecked_eq!(self.inner.dtype(), &dtype) };
        }
        self.inner.read_data_typed(index, context)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
    #[inline(always)]
    fn dtype(&self) -> &crate::dtype::Dtype {
        let dtype = self.inner.dtype();
        if let Some(expected_dtype) = ET::DTYPE {
            unsafe { assert_unchecked_eq!(dtype, &expected_dtype) };
        }
        dtype
    }
    fn spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
        self.inner.spec()
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
