use std::hint::assert_unchecked;
use std::ops::Range;
use std::sync::Arc;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::storage::params::ArraySpecPtr;
use crate::storage::{ArraySpec, ArrayStorageInfo, StridedBuf};
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
    inner: Arc<dyn AnyInner>,
}
trait AnyInner: Send + Sync {
    fn storage(&self) -> &dyn ArrayStorage;
    fn meta(&self) -> &AnyMeta;
}
struct AnyMeta {
    shape: DimDyn,
    element_type: TypeDyn,
    spec: ArraySpecPtr,
}
impl ArrayStorageAny {
    pub(crate) fn new(storage: impl ArrayStorage + Send + Sync + 'static) -> Self {
        let meta = AnyMeta::new(&storage);
        Self::from_inner_box(Box::new(AnyInnerImpl { storage, meta }))
    }

    #[inline(never)]
    fn from_inner_box(inner: Box<dyn AnyInner>) -> Self {
        Self {
            inner: Arc::from(inner),
        }
    }
}
impl ArrayStorage for ArrayStorageAny {
    type ElementType = TypeDyn;
    type Dimension = DimDyn;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>, crate::Error> {
        self.inner.storage().read_data(index, context, out)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        let meta = self.inner.meta();
        let s = meta.shape.as_slice();
        debug_assert!(s.len() <= NDIM_MAX);
        unsafe { assert_unchecked(s.len() <= NDIM_MAX) };
        s
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        let meta = self.inner.meta();
        meta.element_type.dtype()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        let meta = self.inner.meta();
        unsafe { meta.spec.as_ref(|| self.inner.storage().spec()) }
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Any", [self.inner.storage()])
    }

    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}

struct AnyInnerImpl<S> {
    storage: S,
    meta: AnyMeta,
}
impl<S: ArrayStorage + Send + Sync + 'static> AnyInner for AnyInnerImpl<S> {
    fn storage(&self) -> &dyn ArrayStorage {
        &self.storage
    }

    fn meta(&self) -> &AnyMeta {
        &self.meta
    }
}

impl AnyMeta {
    fn new(storage: &dyn ArrayStorage) -> Self {
        Self {
            shape: DimDyn::from_slice(storage.shape()),
            element_type: TypeDyn::from_dtype(storage.dtype().clone()).unwrap(),
            spec: ArraySpecPtr::new(storage.spec()),
        }
    }
}
