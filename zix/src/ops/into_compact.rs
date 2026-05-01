use crate::codec::ReadContext;
use crate::error::Result;
use crate::storage::{ArrayStorage, Compact};
use crate::{Array, ArrayParams};

pub struct IntoCompact<S>(pub(crate) ToCompactInner<S>);
pub(crate) enum ToCompactInner<S> {
    Original(S),
    Compact(Compact),
}
impl<S> IntoCompact<S>
where
    S: ArrayStorage,
{
    pub fn new(array: Array<S>, params: ArrayParams, context: &ReadContext) -> Result<Self> {
        Ok(Self(if array.storage.as_compact().is_some() {
            ToCompactInner::Original(array.into_storage())
        } else {
            ToCompactInner::Compact(array.copy_with(params, context)?.into_storage())
        }))
    }
}
impl<S> ArrayStorage for IntoCompact<S>
where
    S: ArrayStorage,
{
    fn read_data(
        &self,
        index: &[core::ops::Range<u64>],
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> crate::error::Result<()> {
        match &self.0 {
            ToCompactInner::Original(s) => s.read_data(index, buf, context),
            ToCompactInner::Compact(c) => c.read_data(index, buf, context),
        }
    }
    fn shape(&self) -> &[u64] {
        match &self.0 {
            ToCompactInner::Original(s) => s.shape(),
            ToCompactInner::Compact(c) => c.shape(),
        }
    }
    fn dtype(&self) -> &crate::dtype::Dtype {
        match &self.0 {
            ToCompactInner::Original(s) => s.dtype(),
            ToCompactInner::Compact(c) => c.dtype(),
        }
    }
    fn _spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
        match &self.0 {
            ToCompactInner::Original(s) => s._spec(),
            ToCompactInner::Compact(c) => c._spec(),
        }
    }
    fn as_compact(&self) -> Option<crate::storage::CompactBorrowed<'_>> {
        Some(match &self.0 {
            ToCompactInner::Original(s) => s.as_compact().unwrap(),
            ToCompactInner::Compact(c) => c.as_compact().unwrap(),
        })
    }
}
