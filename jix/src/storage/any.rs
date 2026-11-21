use std::hint::assert_unchecked;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::element_type::check_element_type;
use crate::error::Result;
use crate::storage::params::ArraySpecOwned;
use crate::storage::{ArraySpec, ArrayStorageInfo, CompactView, Index, OutBuf};
use crate::{DimDyn, Dimension, ElementType, NDIM_MAX, ReadContext, StorageBase, TypeDyn};

/// A type-erased storage: any storage behind an `Arc`, with reads forwarded to it dynamically.
///
/// Only the storage type is erased, the element type `ET` is kept. Cloning clones the `Arc`. Use
/// [`ArrayAny`](crate::ArrayAny) (the alias `Array<StorageAny<ET>>`) to hold arrays of different
/// storage types in one collection, such as a `Vec<ArrayAny<f32>>`; convert an array with
/// [`Array::into_any`](crate::Array::into_any).
pub struct StorageAny<ET = TypeDyn> {
    inner: Arc<dyn AnyInner>,
    element_type: PhantomData<ET>,
}
impl<ET> Clone for StorageAny<ET> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            element_type: PhantomData,
        }
    }
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
impl<ET> StorageAny<ET> {
    pub(crate) fn new<S>(storage: S) -> Self
    where
        S: StorageBase<ElementType = ET> + std::any::Any + Send + Sync + 'static,
    {
        let meta = AnyMeta::new(&storage);
        Self::from_inner_box(Box::new(AnyInnerImpl { storage, meta }))
    }

    #[inline(never)]
    fn from_inner_box(inner: Box<dyn AnyInner>) -> Self {
        Self {
            inner: Arc::from(inner),
            element_type: PhantomData,
        }
    }
}
impl<ET> crate::storage::ArrayStorageImpl for StorageAny<ET> {}
impl<ET: ElementType> StorageBase for StorageAny<ET> {
    type ElementType = ET;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        context: &ReadContext,
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

    type ElementTypeChange<NewET: ElementType> = StorageAny<NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        check_element_type::<NewET>(self.dtype())?;
        Ok(StorageAny {
            inner: self.inner,
            element_type: PhantomData,
        })
    }
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
