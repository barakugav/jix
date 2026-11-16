use std::hint::assert_unchecked;
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::params::ArraySpecOwned;
use crate::storage::{ArraySpec, ArrayStorageInfo, CompactBorrowed, Index, OutBuf};
use crate::ReadContext;
use crate::{ArrayStorage, DimDyn, Dimension, ElementType, TypeDyn, NDIM_MAX};

/// A type-erased storage: any storage behind an `Arc`, with reads forwarded to it dynamically.
///
/// Cloning clones the `Arc`. Use [`ArrayAny`](crate::ArrayAny) (the alias
/// `Array<ArrayStorageAny>`) to hold arrays of different storage types in one collection, such
/// as a `Vec<ArrayAny>`; convert an array with [`Array::into_any`](crate::Array::into_any).
///
/// EXPERIMENT (api-simplify): `ET` keeps the element type of the erased storage, so an erased
/// `f32` pipeline is still an `f32` array and keeps its element-wise ops.
#[derive(Clone)]
pub struct ArrayStorageAny<ET = TypeDyn> {
    inner: Arc<dyn AnyInner>,
    element_type: ET,
}
trait AnyInner: Send + Sync {
    fn storage(&self) -> &dyn ArrayStorage;
    fn meta(&self) -> &AnyMeta;
}
struct AnyMeta {
    shape: DimDyn,
    spec: ArraySpecOwned,
}
impl<ET: ElementType> ArrayStorageAny<ET> {
    pub(crate) fn new<S>(storage: S) -> Self
    where
        S: ArrayStorage<ElementType = ET> + Send + Sync + 'static,
    {
        let element_type = ET::from_dtype(storage.dtype().clone()).expect("storage dtype");
        let meta = AnyMeta::new(&storage);
        Self {
            inner: from_inner_box(Box::new(AnyInnerImpl { storage, meta })),
            element_type,
        }
    }
}

#[inline(never)]
fn from_inner_box(inner: Box<dyn AnyInner>) -> Arc<dyn AnyInner> {
    Arc::from(inner)
}
impl<ET: ElementType> crate::storage::ArrayStorageImpl for ArrayStorageAny<ET> {}
impl<ET: ElementType> ArrayStorage for ArrayStorageAny<ET> {
    type ElementType = ET;

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
        self.element_type.dtype()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.inner.meta().spec.as_ref()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Any", [self.inner.storage()])
    }

    #[inline]
    fn as_compact(&self) -> Option<CompactBorrowed<'_, TypeDyn>> {
        self.inner.storage().as_compact()
    }

    // Re-tag in place: no `IntoType` wrapper around an erased storage.
    type ElementTypeChange<NewET: ElementType> = ArrayStorageAny<NewET>;

    fn element_type_change<NewET: ElementType>(self) -> Result<ArrayStorageAny<NewET>> {
        let element_type = NewET::from_dtype(self.dtype().clone())?;
        Ok(ArrayStorageAny {
            inner: self.inner,
            element_type,
        })
    }
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
        let mut spec = storage.spec().to_owned();

        // A parent can't fuse through the erased storage, it reads it with `read_data`
        let flags = &mut spec.dynamic_mut().flags;
        *flags = flags.with_read_tiling(flags.read_tiling().forwarded());

        Self {
            shape: DimDyn::from_slice(storage.shape()),
            spec,
        }
    }
}
