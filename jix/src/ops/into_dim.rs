use std::marker::PhantomData;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtyped;
use crate::error::{ensure, Result};
use crate::storage::{ArrayStorageInfo, ReadData, StridedBuf};
use crate::util::assert_unchecked_eq;
use crate::{Array, ArrayStorage, Dimension};

/// A lazy storage adapter that re-tags an array's dimension parameter without copying data.
///
/// `IntoDim<S, D>` wraps an [`Array<S>`](crate::Array) and presents it as having dimension type
/// `D`, without touching the underlying bytes. At construction time, if `D` is a concrete static
/// dimension ([`Dim<N>`](crate::Dim)), the runtime ndim is validated to equal `N`; if it is
/// [`DimDyn`](crate::DimDyn) the conversion always succeeds.
///
/// The typical entry points are [`Array::into_dim`](crate::Array::into_dim) and
/// [`Array::into_dim_dyn`](crate::Array::into_dim_dyn), which either wrap the array in `IntoDim`
/// or, for some storages, swap the dimension parameter in-place. For example,
/// [`Ref`](crate::storage::Ref) can not be re-tagged in-place, but
/// [`Compact`](crate::storage::Compact) can.
///
/// # Examples
///
/// Assert the ndim of a dynamically-shaped array:
///
/// ```
/// use jix::{Array, Dim};
///
/// // Arrays loaded from files (or built from ndarray::IxDyn) carry DimDyn.
/// let a = Array::compact_ndarray(&ndarray::ArrayD::<i32>::zeros(ndarray::IxDyn(&[2, 3, 4])))?;
/// // a: Array<Compact<TypeDyn, DimDyn>>
/// // a: Array<Storage::Dimension = DimDyn>
///
/// // Assert it is 3-D; returns an error if the ndim does not match.
/// let a3d = a.as_ref().into_dim::<Dim<3>>()?;
/// // a3d: Array<IntoDim<AsRef<Compact<TypeDyn, DimDyn>>, Dim<3>>>
/// // a3d: Array<Storage::Dimension = Dim<3>>
///
/// // Subsequent operations propagate Dim<3> through the type system.
/// let a4d = a3d.insert_axis(0); // Array<InsertAxis<..., Dim<4>>>
/// assert_eq!(a4d.shape(), &[1, 2, 3, 4]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct IntoDim<S, D> {
    inner: S,
    dim: PhantomData<D>,
}
impl<S, D> IntoDim<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    /// Constructs a [`IntoDim`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> Result<Self> {
        if let Some(ndim) = D::NDIM {
            let shape = array.shape();
            ensure!(
                shape.len() == ndim,
                InvalidShapeOperation,
                "Cannot convert array with shape {shape:?} to dimension with ndim={ndim}"
            );
        }
        Ok(Self {
            inner: array,
            dim: PhantomData,
        })
    }

    /// Constructs an array with [`IntoDim`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, D> ArrayStorage for IntoDim<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    type ElementType = S::ElementType;
    type Dimension = D;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        if let Some(ndim) = D::NDIM {
            unsafe { assert_unchecked_eq!(ndim, self.inner.shape().len()) };
        }
        self.inner.read_data(index, context, out)
    }

    #[inline(always)]
    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, S, D>>
    where
        T: Dtyped,
    {
        if let Some(ndim) = D::NDIM {
            unsafe { assert_unchecked_eq!(ndim, self.inner.shape().len()) };
        }
        self.inner.read_data_typed(index, context)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        let shape = self.inner.shape();
        if let Some(ndim) = D::NDIM {
            unsafe { assert_unchecked_eq!(ndim, shape.len()) };
        }
        shape
    }
    #[inline(always)]
    fn dtype(&self) -> &crate::dtype::Dtype {
        self.inner.dtype()
    }
    #[inline]
    fn spec(&self) -> crate::storage::ArraySpec<'_> {
        self.inner.spec()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("IntoDim", [&self.inner])
    }

    type DimensionChange<NewD: Dimension> = IntoDim<S, NewD>;
    #[inline]
    fn dimension_change<NewD: Dimension>(self) -> Result<Self::DimensionChange<NewD>> {
        IntoDim::new(self.inner)
    }

    type ElementTypeChange<NewET: crate::ElementType> = IntoDim<S::ElementTypeChange<NewET>, D>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> Result<Self::ElementTypeChange<NewET>> {
        Ok(IntoDim {
            inner: self.inner.element_type_change()?,
            dim: PhantomData,
        })
    }
}

macro_rules! impl_dimension_change_default {
    () => {
        type DimensionChange<NewD: crate::Dimension> = crate::ops::IntoDim<Self, NewD>;

        #[inline]
        fn dimension_change<NewD: crate::Dimension>(
            self,
        ) -> crate::error::Result<Self::DimensionChange<NewD>> {
            crate::ops::IntoDim::new(self)
        }
    };
}
pub(crate) use impl_dimension_change_default;
