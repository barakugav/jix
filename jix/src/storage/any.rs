use std::hint::assert_unchecked;
use std::ops::Range;
use std::sync::Arc;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::storage::ArrayStorageSpec;
use crate::{ArrayStorage, DimDyn, Dimension, ElementType, TypeDyn, NDIM_MAX};

/// A type-erased array storage backend that wraps any dynamically-typed storage via `Arc<dyn ArrayStorage>`.
///
/// `ArrayStorageAny` holds a dynamically-dispatched storage backend and implements `ArrayStorage` by forwarding to it.
/// It is `Clone` (cloning the `Arc`), which makes it easy to store heterogeneous arrays in a collection such
/// as `Vec<ArrayAny>`.
///
/// Use [`ArrayAny`](crate::ArrayAny) (the alias `Array<ArrayStorageAny>`) when you need to
/// hold arrays of different concrete storage types behind a uniform handle, or convert an
/// existing array with [`Array::into_any`](crate::Array::into_any).
#[derive(Clone)]
pub struct ArrayStorageAny {
    inner: Arc<dyn ArrayStorage<ElementType = TypeDyn, Dimension = DimDyn> + Send + Sync>,
    shape: DimDyn,
    element_type: TypeDyn,
}
impl ArrayStorageAny {
    /// Wrap an existing `Arc`-boxed storage as an `ArrayStorageAny`.
    pub fn new(
        storage: Arc<dyn ArrayStorage<ElementType = TypeDyn, Dimension = DimDyn> + Send + Sync>,
    ) -> Self {
        Self {
            shape: DimDyn::from_slice(storage.shape()).unwrap(),
            element_type: TypeDyn::from_dtype(storage.dtype().clone()).unwrap(),
            inner: storage,
        }
    }
}
impl ArrayStorage for ArrayStorageAny {
    type ElementType = TypeDyn;
    type Dimension = DimDyn;

    #[inline(always)]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<(), crate::Error> {
        self.inner.read_data(index, buf, context)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        let s = self.shape.as_slice();
        debug_assert!(s.len() <= NDIM_MAX);
        unsafe { assert_unchecked(s.len() <= NDIM_MAX) };
        s
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.element_type.dtype()
    }

    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.inner.spec()
    }
}
