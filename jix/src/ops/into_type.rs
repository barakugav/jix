use std::marker::PhantomData;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtyped;
use crate::error::{check_dtype, Result};
use crate::storage::{ArrayStorageInfo, OutBuf, ReadData};
use crate::util::assert_unchecked_eq;
use crate::{Array, ArrayStorage, ElementType};

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
/// [`Ref`](crate::storage::Ref) can not be re-tagged in-place, but
/// [`Compact`](crate::storage::Compact) can.
///
/// # Examples
///
/// Erase and recover the static element type:
///
/// ```
/// use jix::dtype::Dtyped;
/// use jix::Array;
/// use ndarray::array;
///
/// // compact_ndarray infers Ty<f32> from the ndarray input type.
/// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
/// // a: Array<Compact<Ty<f32>, Dim<1>>>
///
/// // Erase the static element type - always succeeds.
/// let dyn_a = a.as_ref().into_type_dyn();
/// // dyn_a: Array<IntoType<AsRef<Compact<Ty<f32>, Dim<1>>>, TypeDyn>>
/// // dyn_a: Array<Storage::ElementType = TypeDyn>
/// assert_eq!(dyn_a.dtype(), &f32::DTYPE);
///
/// // Recover the concrete type - validated at runtime.
/// let typed_a = dyn_a.into_typed::<f32>()?;
/// // typed_a: Array<IntoType<..., Ty<f32>>>
/// // typed_a: Array<Storage::ElementType = Ty<f32>>
///
/// // Element-wise operations are available again.
/// let result = typed_a.exp().to_ndarray()?;
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
impl<S, ET> ArrayStorage for IntoType<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    type ElementType = ET;
    type Dimension = S::Dimension;

    #[inline(always)]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
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
        const { &ET::DTYPE }
            .as_ref()
            .unwrap_or_else(|| self.inner.dtype())
    }
    #[inline]
    fn spec(&self) -> crate::storage::ArraySpec<'_> {
        self.inner.spec()
    }
    #[inline]
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("IntoType", [&self.inner])
    }

    type DimensionChange<NewD: crate::Dimension> = IntoType<S::DimensionChange<NewD>, ET>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(IntoType {
            inner: self.inner.dimension_change()?,
            element_type: PhantomData,
        })
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
