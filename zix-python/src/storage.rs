use std::hint::assert_unchecked;
use std::ops::Range;
use std::sync::Arc;

use zix_core::codec::ReadContext;
use zix_core::dtype::Dtype;
use zix_core::storage::ArrayStorageSpec;
use zix_core::{ArrayStorage, DimDyn, TypeDyn, NDIM_MAX};

use crate::util::DimArray;

#[derive(Clone)]
pub(crate) struct DynStorage {
    inner: Arc<dyn ArrayStorage<ElementType = TypeDyn, Dimension = DimDyn> + Send + Sync>,
    shape: DimArray<u64>,
    dtype: Dtype,
}
impl DynStorage {
    pub(crate) fn new(
        storage: Arc<dyn ArrayStorage<ElementType = TypeDyn, Dimension = DimDyn> + Send + Sync>,
    ) -> Self {
        Self {
            shape: storage.shape().try_into().unwrap(),
            dtype: storage.dtype().clone(),
            inner: storage,
        }
    }
}
impl ArrayStorage for DynStorage {
    type ElementType = TypeDyn;
    type Dimension = DimDyn;

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<(), zix_core::Error> {
        self.inner.read_data(index, buf, context)
    }

    fn shape(&self) -> &[u64] {
        let s = self.shape.as_slice();
        debug_assert!(s.len() <= NDIM_MAX);
        unsafe { assert_unchecked(s.len() <= NDIM_MAX) };
        s
    }

    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.inner.spec()
    }
}
