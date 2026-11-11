use std::marker::PhantomData;

use crate::ops::prelude::*;

/// A lazy storage adapter that re-tags an array's element-type parameter without copying data.
///
/// `IntoType<S, ET>` wraps an [`Array<S>`](crate::Array) and presents it as having element type
/// `ET`, without touching the underlying bytes. At construction time, if `ET` is a concrete type
/// ([`Ty<T>`](crate::Ty)), the runtime [`Dtype`](crate::dtype::Dtype) is validated to
/// match `T`; if it is [`TypeDyn`](crate::TypeDyn) the conversion always succeeds.
///
/// The typical entry points are [`Array::into_type`](crate::Array::into_type),
/// [`Array::into_typed`](crate::Array::into_typed), and
/// [`Array::into_type_dyn`](crate::Array::into_type_dyn), which either wrap the array in `IntoType`
/// or, for some storages, swap the element-type parameter in-place. For example,
/// a borrowed `&S` (from [`Array::view`](crate::Array::view)) can not be re-tagged in-place, but
/// [`Compact`](crate::storage::Compact) can.
///
/// # Examples
///
/// Erase and recover the static element type:
///
/// ```
/// use jix::dtype::Dtyped;
/// use jix::Array;
/// use ndarray::{array, Ix1};
///
/// // compact_ndarray infers Ty<f32> from the ndarray input type.
/// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
/// // a: Array<Compact<Ty<f32>>>
///
/// // Erase the static element type - always succeeds.
/// let dyn_a = a.view().into_type_dyn();
/// // dyn_a: Array<IntoType<&Compact<Ty<f32>>, TypeDyn>>
/// // dyn_a: Array<Storage::ElementType = TypeDyn>
/// assert_eq!(dyn_a.dtype(), &f32::DTYPE);
///
/// // Recover the concrete type - validated at runtime.
/// let typed_a = dyn_a.into_typed::<f32>()?;
/// // typed_a: Array<IntoType<..., Ty<f32>>>
/// // typed_a: Array<Storage::ElementType = Ty<f32>>
///
/// // Element-wise operations are available again.
/// let result = typed_a.exp().to_ndarray::<Ix1>()?;
/// # Ok::<(), jix::Error>(())
/// ```
pub struct IntoType<S, ET> {
    inner: S,
    element_type: PhantomData<ET>,
}
impl<S, ET> IntoType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    /// Constructs a [`IntoType`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> Result<Self> {
        if let Some(expected_dtype) = ET::DTYPE {
            check_dtype(array.dtype(), &expected_dtype)?;
        }
        Ok(Self {
            inner: array,
            element_type: PhantomData,
        })
    }

    /// Constructs an array with [`IntoType`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, ET> crate::storage::ArrayStorageImpl for IntoType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
}
impl<S, ET> ArrayStorage for IntoType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    type ElementType = ET;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        if let Some(dtype) = ET::DTYPE {
            debug_assert_eq!(self.inner.dtype(), &dtype);
        }
        self.inner.read_data(index, context, out)
    }

    #[inline(always)]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S, ET>>
    where
        T: Dtyped,
    {
        if let Some(dtype) = ET::DTYPE {
            debug_assert_eq!(self.inner.dtype(), &dtype);
        }
        self.inner.read_as_elementwise_pipeline(index, context)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
    #[inline(always)]
    fn dtype(&self) -> &crate::dtype::Dtype {
        const { &ET::DTYPE }
            .as_ref()
            .unwrap_or_else(|| self.inner.dtype())
    }
    #[inline]
    fn spec(&self) -> crate::storage::ArraySpec<'_> {
        self.inner.spec()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("IntoType", [&self.inner])
    }
    #[inline]
    fn as_compact(&self) -> Option<crate::storage::CompactBorrowed<'_, TypeDyn>> {
        self.inner.as_compact()
    }

    type ElementTypeChange<NewET: ElementType> = IntoType<S, NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        IntoType::new(self.inner)
    }
}

macro_rules! impl_element_type_change_default {
    () => {
        type ElementTypeChange<NewET: crate::ElementType> = crate::ops::IntoType<Self, NewET>;

        #[inline]
        fn element_type_change<NewET: crate::ElementType>(
            self,
        ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
            crate::ops::IntoType::new(self)
        }
    };
}
pub(crate) use impl_element_type_change_default;

/// Methods for converting an array to a different element type.
impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Re-tag this array's element type as `NewET`, returning an error if the runtime dtype does
    /// not match.
    ///
    /// This is the bridge between dynamic and static element-type tracking. Arrays loaded from
    /// files carry [`TypeDyn`] as their element type because the compiler cannot know the dtype
    /// at that point. After you have confirmed the dtype (e.g. by reading `array.dtype()` or
    /// knowing the data ahead of time), call `into_type::<Ty<T>>()` (or the
    /// [`into_typed::<T>`](Self::into_typed) sugar) to recover a statically-typed element type.
    /// Subsequent operations on the result will propagate the static `Ty<T>` through the type
    /// system.
    ///
    /// Most element-wise operations require a static element type (`ArrayStorageTyped`).
    ///
    /// This method replaces the inner storage with `S::ElementTypeChange<NewET>`, which is
    /// implemented by the simpler [`IntoType<S, NewET>`](crate::ops::IntoType) adaptor for some
    /// storages, but may be implemented as an in-place replacement for others.
    ///
    /// See [`into_type_dyn`](Self::into_type_dyn).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::error::ErrorKind::UnsupportedDtype) if
    /// `NewET = Ty<T>` and `self.dtype() != T::DTYPE`. Always succeeds for `NewET = TypeDyn`.
    #[inline]
    pub fn into_type<NewET>(self) -> Result<Array<S::ElementTypeChange<NewET>>>
    where
        NewET: ElementType,
    {
        Ok(Array::from_storage(
            self.into_storage().element_type_change()?,
        ))
    }

    /// Re-tag this array's element type as [`Ty<T>`](crate::Ty), asserting a concrete scalar type.
    ///
    /// Sugar for [`into_type::<Ty<T>>()`](Self::into_type).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::error::ErrorKind::UnsupportedDtype) if
    /// `self.dtype() != T::DTYPE`.
    #[inline]
    pub fn into_typed<T>(self) -> Result<Array<S::ElementTypeChange<Ty<T>>>>
    where
        T: Dtyped,
    {
        self.into_type()
    }

    /// Re-tag this array's element type as [`TypeDyn`], erasing static element-type information.
    ///
    /// Infallible sugar for [`into_type::<TypeDyn>()`](Self::into_type).
    #[inline]
    pub fn into_type_dyn(self) -> Array<S::ElementTypeChange<TypeDyn>> {
        self.into_type().unwrap()
    }
}
