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
/// The typical entry points are [`Array::into_type`](crate::Array::into_type),
/// [`Array::into_typed`](crate::Array::into_typed), and
/// [`Array::into_type_dyn`](crate::Array::into_type_dyn), which either wrap the array in `ToType`
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
/// // dyn_a: Array<ToType<AsRef<Compact<Ty<f32>, Dim<1>>>, TypeDyn>>
/// // dyn_a: Array<Storage::ElementType = TypeDyn>
/// assert_eq!(dyn_a.dtype(), &f32::DTYPE);
///
/// // Recover the concrete type - validated at runtime.
/// let typed_a = dyn_a.into_typed::<f32>()?;
/// // typed_a: Array<ToType<..., Ty<f32>>>
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

    type DimensionChange<NewD: crate::Dimension> = ToType<S::DimensionChange<NewD>, ET>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(ToType {
            inner: self.inner.dimension_change()?,
            element_type: PhantomData,
        })
    }

    type ElementTypeChange<NewET: ElementType> = ToType<S, NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        ToType::new(self.inner)
    }
}

macro_rules! impl_element_type_change_default {
    () => {
        type ElementTypeChange<NewET: crate::ElementType> = crate::ops::ToType<Self, NewET>;

        #[inline]
        fn element_type_change<NewET: crate::ElementType>(
            self,
        ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
            crate::ops::ToType::new(self)
        }
    };
}
pub(crate) use impl_element_type_change_default;
