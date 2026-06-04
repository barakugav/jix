use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::Compact;
use crate::{Array, ArrayParams, ArrayStorage};

/// Storage adaptor that guarantees the wrapped array is always in compact
/// block-compressed form.
///
/// Returned by [`Array::maybe_compact`] and [`Array::maybe_compact_with`]. The
/// adaptor handles two cases transparently:
///
/// - **Already compact**: the original storage is kept as is — no copy or re-compression.
/// - **Not compact** (lazy views, op chains, etc.): the array is materialized
///   via `copy_with` into a new [`Compact`] block-table.
///
/// In both cases all [`ArrayStorage`] methods delegate to the inner variant,
/// and the storage is guaranteed to be a materialized compact storage, not a view.
/// The dimension type `S::Dimension` is preserved in both paths.
pub struct MaybeCompact<S: ArrayStorage>(pub(crate) ToCompactInner<S>);

/// The two internal states of an [`MaybeCompact<S>`] storage.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ToCompactInner<S: ArrayStorage> {
    /// The source was already compact; the original storage is kept as-is.
    Original(S),
    /// The source was not compact; it was materialized into a new `Compact`.
    Compact(Compact<S::ElementType, S::Dimension>),
}
impl<S> MaybeCompact<S>
where
    S: ArrayStorage,
{
    /// Constructs a [`MaybeCompact`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, params: ArrayParams, context: &ReadContext) -> Result<Self> {
        Ok(Self(if array.as_compact().is_some() {
            ToCompactInner::Original(array)
        } else {
            ToCompactInner::Compact(
                Array::from_storage(array)
                    .copy_with(params, context)?
                    .into_storage(),
            )
        }))
    }

    /// Constructs an array with [`MaybeCompact`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(
        array: Array<S>,
        params: ArrayParams,
        context: &ReadContext,
    ) -> Result<Array<Self>> {
        Self::new(array.into_storage(), params, context).map(Array::from_storage)
    }
}
impl<S> ArrayStorage for MaybeCompact<S>
where
    S: ArrayStorage,
{
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    #[inline]
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

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        match &self.0 {
            ToCompactInner::Original(s) => s.shape(),
            ToCompactInner::Compact(c) => c.shape(),
        }
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        match &self.0 {
            ToCompactInner::Original(s) => s.dtype(),
            ToCompactInner::Compact(c) => c.dtype(),
        }
    }
    fn spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
        match &self.0 {
            ToCompactInner::Original(s) => s.spec(),
            ToCompactInner::Compact(c) => c.spec(),
        }
    }
    fn as_compact(
        &self,
    ) -> Option<crate::storage::CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
        Some(match &self.0 {
            ToCompactInner::Original(s) => s.as_compact().unwrap(),
            ToCompactInner::Compact(c) => c.as_compact().unwrap(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ndarray::ArrayD;

    use crate::dtype::Dtyped;
    use crate::storage::Compact;
    use crate::util::{arr_params, carray_strategy_any};
    use crate::{Array, ArrayParams, ArrayStorage, DimDyn, Dimension, ElementType, Ty};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn compact<T: Dtyped>(
        vals: Vec<T>,
        shape: &[usize],
        block_shape: &[usize],
    ) -> Array<Compact<Ty<T>, DimDyn>> {
        let src = ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
        Array::compact_array_with(&src, arr_params(block_shape)).unwrap()
    }

    fn to_bytes<ET: ElementType, D: Dimension>(a: &Array<Compact<ET, D>>) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        a.write_to(&mut buf).unwrap();
        buf.into_inner()
    }

    // -----------------------------------------------------------------------
    // Proptest: maybe_compact on already-compact arrays
    //
    // For any compact array, maybe_compact must:
    //   - return the same values
    //   - always produce as_compact() == Some
    //   - produce byte-for-byte identical output (no re-compression)
    // -----------------------------------------------------------------------

    macro_rules! test_maybe_compact_passthrough_dtype {
        ($dtype:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<passthrough_compact_ $dtype>](
                        (src, a) in carray_strategy_any::<$dtype>()
                    ) {
                        let original_bytes = to_bytes(&a);
                        let result = a.maybe_compact().unwrap();
                        proptest::prop_assert!(result.storage().as_compact().is_some());
                        proptest::prop_assert_eq!(
                            result.to_ndarray().unwrap(),
                            src
                        );
                        // No re-compression: serialized bytes must be identical.
                        let mut result_bytes = Cursor::new(Vec::new());
                        result.write_to(&mut result_bytes).unwrap();
                        proptest::prop_assert_eq!(result_bytes.into_inner(), original_bytes);
                    }
                }
            }
        };
    }

    test_maybe_compact_passthrough_dtype!(u8);
    test_maybe_compact_passthrough_dtype!(i32);
    test_maybe_compact_passthrough_dtype!(i64);
    test_maybe_compact_passthrough_dtype!(f32);
    test_maybe_compact_passthrough_dtype!(f64);

    // -----------------------------------------------------------------------
    // Proptest: maybe_compact_with on lazy neg views
    //
    // For any compact array of a signed dtype, wrapping it in a neg view and
    // calling maybe_compact_with must produce the negated values and always
    // yield as_compact() == Some.
    // -----------------------------------------------------------------------

    macro_rules! test_maybe_compact_neg_view_dtype {
        ($dtype:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<neg_view_ $dtype>](
                        (src, a) in carray_strategy_any::<$dtype>()
                    ) {
                        let expected = -&src;
                        let ctx = a.read_ctx();
                        let result = (-a.as_ref())
                            .maybe_compact_with(ArrayParams::default(), &ctx)
                            .unwrap();
                        proptest::prop_assert!(result.storage().as_compact().is_some());
                        proptest::prop_assert_eq!(
                            result.to_ndarray().unwrap(),
                            expected
                        );
                    }
                }
            }
        };
    }

    test_maybe_compact_neg_view_dtype!(i32);
    test_maybe_compact_neg_view_dtype!(i64);
    test_maybe_compact_neg_view_dtype!(f32);
    test_maybe_compact_neg_view_dtype!(f64);

    // -----------------------------------------------------------------------
    // Explicit: maybe_compact_with respects block shape for non-compact sources
    // -----------------------------------------------------------------------

    #[test]
    fn maybe_compact_with_block_shape_respected() {
        // Build a lazy neg view, then compact it with a known block shape.
        // Verify the resulting block_shape matches the requested one.
        let vals: Vec<i32> = (1..=12i32).collect();
        let src = ArrayD::from_shape_vec(vec![3, 4], vals.clone()).unwrap();
        let a = compact::<i32>(vals, &[3, 4], &[3, 4]);
        let ctx = a.read_ctx();

        let result = (-a.as_ref())
            .maybe_compact_with(arr_params(&[2, 2]), &ctx)
            .unwrap();

        // Values must be negated.
        assert_eq!(result.to_ndarray().unwrap(), -&src);
        // The compacted storage must use block shape [2, 2].
        let bs = result
            .storage()
            .as_compact()
            .unwrap()
            .0
            .block_shape()
            .to_vec();
        assert_eq!(bs, &[2u32, 2]);
    }

    // -----------------------------------------------------------------------
    // Explicit: maybe_compact_with for compact source ignores params
    // -----------------------------------------------------------------------

    #[test]
    fn maybe_compact_with_compact_source_ignores_params() {
        // A compact source with block shape [4] must be kept as-is even when
        // maybe_compact_with is called with a different block shape.
        let vals: Vec<i32> = (0..16i32).collect();
        let a = compact::<i32>(vals, &[16], &[4]);
        let ctx = a.read_ctx();
        let original_bytes = to_bytes(&a);

        // Pass a different block shape - it must be ignored.
        let result = a.maybe_compact_with(arr_params(&[8]), &ctx).unwrap();
        let mut result_bytes = Cursor::new(Vec::new());
        result.write_to(&mut result_bytes).unwrap();

        assert_eq!(result_bytes.into_inner(), original_bytes);
    }

    // -----------------------------------------------------------------------
    // Explicit: 3-D array with op chain
    // -----------------------------------------------------------------------

    #[test]
    fn maybe_compact_3d_add_chain_i32() {
        // (a + a) over a 3-D array compressed via maybe_compact_with.
        let vals: Vec<i32> = (1..=60i32).collect();
        let src = ArrayD::from_shape_vec(vec![3, 4, 5], vals.clone()).unwrap();
        let expected = &src + &src;
        let a = compact::<i32>(vals, &[3, 4, 5], &[2, 2, 3]);
        let ctx = a.read_ctx();

        let result = (a.as_ref() + a.as_ref())
            .maybe_compact_with(arr_params(&[2, 2, 3]), &ctx)
            .unwrap();

        assert!(result.storage().as_compact().is_some());
        assert_eq!(result.to_ndarray().unwrap(), expected);
    }
}
