use std::hint::assert_unchecked;
use std::marker::PhantomData;

use crate::error::Result;
use crate::storage::ArrayStorage;
use crate::{Array, DimDyn, Dimension, Error, ErrorKind};

/// A storage adapter that re-tags an existing array with a specific [`Dimension`] type.
///
/// `SwapDim<S, D>` wraps `Array<S>` and declares `type Dimension = D`, overriding whatever
/// dimension `S` reported. This is a zero-overhead wrapper: all reads are forwarded directly to
/// the inner storage without any data transformation.
///
/// The only work done at construction time is a ndim check: if `D::NDIM = Some(n)`,
/// `SwapDim::new` verifies that `array.ndim() == n` and returns an error if not.
///
/// # Purpose
///
/// Arrays loaded from files or memory-mapped sources carry [`DimDyn`] as their storage dimension
/// because the ndim is not known until the file header is read. `SwapDim` lets callers assert
/// "I know this array is N-dimensional" and recover static tracking so that subsequent operations
/// benefit from compile-time dimension arithmetic.
///
/// The primary entry point is [`Array::into_dim`](crate::Array::into_dim), which works for any
/// storage. For concrete block-compressed storages ([`Compact`](crate::storage::Compact),
/// [`CompactMmap`](crate::storage::CompactMmap)) consider
/// [`Array::swap_dim`](crate::Array::swap_dim) instead — it replaces the storage's `D` type
/// parameter in-place and avoids adding this wrapper layer.
///
/// # Example
///
/// ```
/// use zix::{Array, Dim};
///
/// // Passing a dynamically-dimensioned ndarray produces Array<Compact<DimDyn>>.
/// // (Array::read_from_file also carries DimDyn — ndim from file header.)
/// let a = Array::compact_array(&ndarray::ArrayD::<i32>::zeros(ndarray::IxDyn(&[2, 3])))?;
///
/// // into_dim wraps the storage in SwapDim, asserting ndim == 2 at runtime.
/// let a2d = a.into_dim::<Dim<2>>()?;  // Array<SwapDim<Compact<DimDyn>, Dim<2>>>
///
/// // Subsequent operations propagate the static Dim<2>.
/// let a3d = a2d.insert_axis(0usize);  // Array<InsertAxis<..., Dim<3>>>
/// assert_eq!(a3d.shape(), &[1, 2, 3]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct SwapDim<S, D> {
    inner: Array<S>,
    dim: PhantomData<D>,
}
impl<S, D> SwapDim<S, D> {
    /// Wrap `array` in an `SwapDim<S, D>`, validating the ndim against `D`.
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
impl<S, D> ArrayStorage for SwapDim<S, D>
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

/// Re-tags the storage's dimension type parameter in-place, without adding a wrapper layer.
///
/// Implemented by concrete block-compressed storages — [`Compact<D>`](crate::storage::Compact),
/// [`CompactBorrowed<'a, D>`](crate::storage::CompactBorrowed), and
/// [`CompactMmap<D>`](crate::storage::CompactMmap) — and allows changing the `D` type parameter
/// directly. The result type is the same concrete storage with a new dimension, e.g.
/// `Compact<DimDyn>` → `Compact<Dim<2>>`.
///
/// # Compared to [`into_dim`](crate::Array::into_dim)
///
/// [`Array::into_dim`](crate::Array::into_dim) works for *any* `S: ArrayStorage` and wraps the
/// storage in a [`SwapDim<S, D>`] adaptor:
/// ```text
/// x: Array<Compact<DimDyn>>
/// x.into_dim::<Dim<2>>()  →  Array<SwapDim<Compact<DimDyn>, Dim<2>>>
/// ```
///
/// `swap_dim` replaces the dimension type in the underlying storage instead:
/// ```text
/// x: Array<Compact<DimDyn>>
/// x.swap_dim::<Dim<2>>()  →  Array<Compact<Dim<2>>>
/// ```
///
/// Use `swap_dim` (via [`Array::swap_dim`](crate::Array::swap_dim)) when the storage type is
/// known to be a concrete backend, to keep types clean and avoid the extra wrapper.
pub trait SwapDimInplace: ArrayStorage {
    /// The same concrete storage type with its dimension parameter replaced by `NewD`.
    type SwapDimension<NewD: Dimension>: ArrayStorage<Dimension = NewD>;

    /// Replace the storage's dimension type with `NewD`.
    ///
    /// Returns `None` if `NewD` has a static ndim (`NewD::NDIM = Some(n)`) and the actual
    /// runtime ndim does not match. Returns `Some` for `DimDyn` (`NewD::NDIM = None`) regardless
    /// of the actual ndim.
    fn swap_dim<NewD: Dimension>(self) -> Option<Self::SwapDimension<NewD>>;
}
impl<S> Array<S>
where
    S: SwapDimInplace,
{
    /// Re-tag this array's storage dimension in-place, returning `None` if the ndim doesn't match.
    ///
    /// Unlike [`into_dim`](crate::Array::into_dim), which wraps the storage in a [`SwapDim`]
    /// adaptor and works for any storage, `swap_dim` directly replaces the `D` type parameter of
    /// the underlying storage. The result is `Array<S::SwapDimension<NewD>>` — for
    /// `Array<Compact<D>>` that is `Array<Compact<NewD>>`, without any extra wrapper.
    ///
    /// Requires `S: SwapDimInplace`, which is implemented for [`Compact<D>`](crate::storage::Compact),
    /// [`CompactMmap<D>`](crate::storage::CompactMmap), and
    /// [`CompactBorrowed<'a, D>`](crate::storage::CompactBorrowed).
    ///
    /// Returns `None` if `NewD::NDIM = Some(n)` and `self.ndim() != n`. For `NewD = DimDyn`,
    /// always returns `Some`.
    ///
    /// See [`swap_dim_dyn`](Self::swap_dim_dyn) for the infallible variant that erases static
    /// dimension info.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, Dim, DimDyn};
    /// use zix::storage::Compact;
    /// use ndarray::array;
    ///
    /// // compact_array infers Dim<1> from the ndarray type.
    /// let x: Array<Compact<Dim<1>>> = Array::compact_array(&array![1i32, 2]).unwrap();
    ///
    /// // Erase static tracking to DimDyn — always succeeds.
    /// let dyn_x: Array<Compact<DimDyn>> = x.swap_dim_dyn();
    ///
    /// // Recover static tracking: succeeds when ndim matches.
    /// let restored: Array<Compact<Dim<1>>> = dyn_x.swap_dim::<Dim<1>>().unwrap();
    /// assert_eq!(restored.shape(), &[2]);
    /// ```
    pub fn swap_dim<NewD: Dimension>(self) -> Option<Array<S::SwapDimension<NewD>>> {
        Some(Array::from_storage(self.into_storage().swap_dim()?))
    }

    /// Re-tag this array's storage dimension as [`DimDyn`], erasing static dimension information.
    ///
    /// This is the infallible counterpart to [`swap_dim`](Self::swap_dim). Every array has a
    /// runtime ndim regardless of its static type, so the conversion always succeeds.
    ///
    /// Unlike [`into_dim_dyn`](crate::Array::into_dim_dyn), which wraps the storage in a
    /// [`SwapDim`] adaptor, `swap_dim_dyn` replaces the dimension type parameter directly in the
    /// underlying storage. For `Array<Compact<D>>` the result is `Array<Compact<DimDyn>>`, not
    /// `Array<SwapDim<Compact<D>, DimDyn>>`.
    pub fn swap_dim_dyn(self) -> Array<S::SwapDimension<DimDyn>> {
        self.swap_dim::<DimDyn>().unwrap()
    }
}
