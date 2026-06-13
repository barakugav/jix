use std::marker::PhantomData;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtyped;
use crate::error::Result;
use crate::storage::ReadData;
use crate::util::assert_unchecked_eq;
use crate::{Array, ArrayStorage, Dimension, Error, ErrorKind};

/// A lazy storage adapter that re-tags an array's dimension parameter without copying data.
///
/// `ToDim<S, D>` wraps an [`Array<S>`](crate::Array) and presents it as having dimension type
/// `D`, without touching the underlying bytes. At construction time, if `D` is a concrete static
/// dimension ([`Dim<N>`](crate::Dim)), the runtime ndim is validated to equal `N`; if it is
/// [`DimDyn`](crate::DimDyn) the conversion always succeeds.
///
/// The typical entry points are [`Array::to_dim`](crate::Array::to_dim) and
/// [`Array::to_dim_dyn`](crate::Array::to_dim_dyn).
///
/// For concrete block-compressed or plain storages that implement [`DimensionChange`], prefer
/// [`Array::into_dim`](crate::Array::into_dim) instead - it re-tags the dimension in-place
/// without adding this wrapper layer.
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
/// let a3d = a.to_dim::<Dim<3>>()?;
/// // a3d: Array<ToDim<Compact<TypeDyn, DimDyn>, Dim<3>>>
/// // a3d: Array<Storage::Dimension = Dim<3>>
///
/// // Subsequent operations propagate Dim<3> through the type system.
/// let a4d = a3d.insert_axis(0); // Array<InsertAxis<..., Dim<4>>>
/// assert_eq!(a4d.shape(), &[1, 2, 3, 4]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct ToDim<S, D> {
    inner: S,
    dim: PhantomData<D>,
}
impl<S, D> ToDim<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    /// Constructs a [`ToDim`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> Result<Self> {
        if let Some(ndim) = D::NDIM {
            let shape = array.shape();
            if shape.len() != ndim {
                return Err(Error::new(
                    ErrorKind::InvalidShapeOperation,
                    format!(
                        "Cannot convert array with shape {shape:?} to dimension with ndim={ndim}"
                    ),
                ));
            }
        }
        Ok(Self {
            inner: array,
            dim: PhantomData,
        })
    }

    /// Constructs an array with [`ToDim`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, D> ArrayStorage for ToDim<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    type ElementType = S::ElementType;
    type Dimension = D;

    #[inline(always)]
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        if let Some(ndim) = D::NDIM {
            unsafe { assert_unchecked_eq!(ndim, self.inner.shape().len()) };
        }
        self.inner.read_data(index, buf, context)
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
    fn spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
        self.inner.spec()
    }
}

/// Opt-in trait for storage types that can swap their dimension parameter in-place.
///
/// Storages that implement this trait can change their [`Dimension`] without wrapping
/// themselves in a [`ToDim`] adapter. The result is a leaner type: e.g. `Compact<ET, NewD>`
/// instead of `ToDim<Compact<ET, D>, NewD>`.
///
/// Implementors: [`Compact`](crate::storage::Compact),
/// [`CompactMmap`](crate::storage::CompactMmap),
/// [`CompactBorrowed`](crate::storage::CompactBorrowed),
/// [`Plain`](crate::storage::Plain).
///
/// The typical entry points are [`Array::into_dim`](crate::Array::into_dim) and
/// [`Array::into_dim_dyn`](crate::Array::into_dim_dyn). Use
/// [`Array::to_dim`](crate::Array::to_dim) / [`Array::to_dim_dyn`](crate::Array::to_dim_dyn)
/// for storage types that do not implement this trait.
pub trait DimensionChange {
    /// The concrete storage type after swapping the dimension to `NewD`.
    type DimensionChange<NewD: Dimension>: ArrayStorage<
        ElementType = Self::ElementType,
        Dimension = NewD,
    >
    where
        Self: ArrayStorage;

    /// Consume `self`, validate the new dimension against the runtime ndim, and return the
    /// re-tagged storage.
    ///
    /// Returns [`ErrorKind::InvalidShapeOperation`](crate::ErrorKind::InvalidShapeOperation) if
    /// `NewD = Dim<N>` and the runtime ndim does not equal `N`. Always succeeds for
    /// `NewD = DimDyn`.
    fn dimension_change<NewD: Dimension>(self) -> Result<Self::DimensionChange<NewD>>
    where
        Self: ArrayStorage;
}
