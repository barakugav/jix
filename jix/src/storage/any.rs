use std::hint::assert_unchecked;
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::params::ArraySpecOwned;
use crate::storage::{ArraySpec, ArrayStorageInfo, CompactView, Index, OutBuf};
use crate::ReadContext;
use crate::{DimDyn, Dimension, StorageBase, TypeDyn, NDIM_MAX};

/// A type-erased storage: any storage behind an `Arc`, with reads forwarded to it dynamically.
///
/// Cloning clones the `Arc`. Use [`ArrayAny`](crate::ArrayAny) (the alias
/// `Array<StorageAny>`) to hold arrays of different storage types in one collection, such
/// as a `Vec<ArrayAny>`; convert an array with [`Array::into_any`](crate::Array::into_any).
#[derive(Clone)]
pub struct StorageAny {
    inner: Arc<dyn AnyInner>,
}
trait AnyInner: Send + Sync {
    fn storage(&self) -> &dyn StorageBase;
    fn meta(&self) -> &AnyMeta;
}
struct AnyMeta {
    shape: DimDyn,
    dtype: Dtype,
    spec: ArraySpecOwned,
}
impl StorageAny {
    pub(crate) fn new(storage: impl StorageBase + Send + Sync + 'static) -> Self {
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
impl crate::storage::ArrayStorageImpl for StorageAny {}
impl StorageBase for StorageAny {
    type ElementType = TypeDyn;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        out: OutBuf<'_, 'a>,
    ) -> Result<()> {
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
        &self.inner.meta().dtype
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.inner.meta().spec.as_ref()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Any", [self.inner.storage()])
    }

    #[inline]
    fn as_compact(&self) -> Option<CompactView<'_, TypeDyn>> {
        self.inner.storage().as_compact()
    }

    crate::ops::impl_element_type_change_default!();
}

struct AnyInnerImpl<S> {
    storage: S,
    meta: AnyMeta,
}
impl<S: StorageBase + Send + Sync + 'static> AnyInner for AnyInnerImpl<S> {
    fn storage(&self) -> &dyn StorageBase {
        &self.storage
    }

    fn meta(&self) -> &AnyMeta {
        &self.meta
    }
}

impl AnyMeta {
    fn new(storage: &dyn StorageBase) -> Self {
        let mut spec = storage.spec().to_owned();

        // A parent can't fuse through the erased storage, it reads it with `read_data`
        let flags = &mut spec.dynamic_mut().flags;
        *flags = flags.with_read_tiling(flags.read_tiling().forwarded());

        Self {
            shape: DimDyn::from_slice(storage.shape()),
            dtype: storage.dtype().clone(),
            spec,
        }
    }
}
