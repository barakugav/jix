use std::hint::assert_unchecked;
use std::marker::PhantomData;

use crate::error::Result;
use crate::storage::ArrayStorage;
use crate::{Array, Dimension, Error, ErrorKind};

/// A storage adapter that re-tags an existing array with a specific [`Dimension`] type.
///
/// `IntoDim<S, D>` wraps `Array<S>` and declares `type Dimension = D`, overriding whatever
/// dimension `S` reported. This is a zero-overhead wrapper: all reads are forwarded directly to
/// the inner storage without any data transformation.
///
/// The only work done at construction time is a ndim check: if `D::NDIM = Some(n)`,
/// `IntoDim::new` verifies that `array.ndim() == n` and returns an error if not.
///
/// # Purpose
///
/// Most concrete storages (e.g. `Compact`, `CompactMmap`) use `DimDyn` as their dimension type
/// because the ndim is determined by runtime data (the array file or buffer). `IntoDim` lets
/// callers assert "I know this array is N-dimensional" and recover static tracking so that
/// subsequent operations benefit from compile-time dimension arithmetic.
///
/// The primary entry point is [`Array::into_dim`](crate::Array::into_dim). For the common
/// direction (any dimension to `DimDyn`), use [`Array::into_dim_dyn`](crate::Array::into_dim_dyn).
///
/// # Example
///
/// ```rust,ignore
/// use zix::{Array, ArrayParams, Dim};
///
/// let a = Array::read_from_file("data.zix", ArrayParams::default())?; // DimDyn
/// let a2d = a.into_dim::<Dim<2>>()?;  // Dim<2> — fails if ndim != 2
/// let a3d = a2d.insert_axis(0);  // Dim<3> — compiler-verified
/// ```
pub struct IntoDim<S, D> {
    inner: Array<S>,
    dim: PhantomData<D>,
}
impl<S, D> IntoDim<S, D> {
    /// Wrap `array` in an `IntoDim<S, D>`, validating the ndim against `D`.
    ///
    /// If `D::NDIM = Some(n)` and `array.ndim() != n`, returns
    /// [`ErrorKind::InvalidShapeOperation`](crate::ErrorKind::InvalidShapeOperation).
    /// If `D = DimDyn` (`D::NDIM = None`), the check is skipped and the construction always
    /// succeeds regardless of the actual ndim.
    pub fn new(array: Array<S>) -> Result<Self>
    where
        S: ArrayStorage,
        D: Dimension,
    {
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
}
impl<S, D> ArrayStorage for IntoDim<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    type Dimension = D;

    fn read_data(
        &self,
        index: &[core::ops::Range<u64>],
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> crate::error::Result<()> {
        if let Some(ndim) = D::NDIM {
            let shape = self.inner.storage.shape();
            debug_assert_eq!(ndim, shape.len());
            unsafe { assert_unchecked(ndim == shape.len()) };
        }
        self.inner.storage.read_data(index, buf, context)
    }
    fn shape(&self) -> &[u64] {
        let shape = self.inner.storage.shape();
        if let Some(ndim) = D::NDIM {
            debug_assert_eq!(ndim, shape.len());
            unsafe { assert_unchecked(ndim == shape.len()) };
        }
        shape
    }
    fn dtype(&self) -> &crate::dtype::Dtype {
        self.inner.storage.dtype()
    }
    fn _spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
        self.inner.storage._spec()
    }
}
