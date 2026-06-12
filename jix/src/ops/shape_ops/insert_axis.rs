use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_range, check_ndim, ensure, Result};
use crate::ops::AxesArg;
use crate::storage::{ArrayStorageSpec, BlockShapeTag, BlocksLayout, ReadData};
use crate::util::DimArray;
use crate::{dim_arr, Array, ArrayStorage, Dimension};

/// Inserts new length-1 dimensions at specified positions in an array's shape,
/// returned by [`Array::insert_axis`](crate::Array::insert_axis).
///
/// Each element of `axis` is a **gap index** that identifies a position *between* (or outside)
/// the input dimensions:
///
/// ```text
/// gap:   0      1      2   orig_ndim
///        |  d0  |  d1  |  d2  |
/// ```
///
/// * Gap `0` - before the first input dimension.
/// * Gap `k` - between input dimensions `k-1` and `k`.
/// * Gap `orig_ndim` - after the last input dimension.
///
/// Each occurrence of a gap index inserts one new length-1 dimension at that position. Duplicate
/// gap indices are allowed and each adds another dimension at the same gap. The order of values in
/// `axis` does not matter - only the multiset of gap indices matters. Valid gap indices are
/// `0..=orig_ndim`.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Dimension tracking
///
/// `InsertAxis<S, D>` is generic over `D: Dimension`, determined by the axis argument type.
/// Statically-sized arguments encode the output ndim in the type:
///
/// | Argument type | Output `D` |
/// |---|---|
/// | `usize` | `S::Dimension::Larger` |
/// | `[usize; N]` / `(usize, ...)` N-tuple | `Larger` applied N times |
/// | `[usize; 0]` / `()` | `S::Dimension` (unchanged) |
/// | `&[usize]` / `&Vec<usize>` | `DimDyn` |
///
/// # Examples
///
/// ```text
/// [N]       axis: [0]     -> [1, N]      (insert before first dim)
/// [N]       axis: [1]     -> [N, 1]      (append after last dim)
/// [N, M]    axis: [1]     -> [N, 1, M]   (insert between dims)
/// [N, M]    axis: [0, 2]  -> [1, N, M, 1]
/// ```
///
/// Different argument types select both the insertion positions and the output dimension type:
///
/// ```
/// use jix::{Array, Dim};
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?; // shape [3], Dim<1>
///
/// // usize -> output D = Dim<2> (one more than input Dim<1>)
/// assert_eq!(a.as_ref().insert_axis(0).shape(), &[1, 3]);
/// assert_eq!(a.as_ref().insert_axis(1).shape(), &[3, 1]);
///
/// // [usize; 2] -> output D = Dim<3> (two more than input Dim<1>)
/// assert_eq!(a.as_ref().insert_axis([0, 1]).shape(), &[1, 3, 1]);
///
/// // &[usize] -> output D = DimDyn
/// let gaps = vec![0, 1];
/// assert_eq!(a.insert_axis(gaps.as_slice()).shape(), &[1, 3, 1]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct InsertAxis<S, D> {
    array: S,
    /// `is_inserted[output_dim]` is `true` for every output dimension that was inserted
    /// (length 1, no corresponding input dimension).
    is_inserted: DimArray<bool>,

    shape: D,
    blocks_layout: BlocksLayout,
}

impl<S, D> InsertAxis<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    /// Constructs a [`InsertAxis`] storage. See the struct docs for semantics and examples.
    pub fn new<Ax>(array: S, axis: Ax) -> Result<Self>
    where
        Ax: AxesArg<ExpandedDimension<S::Dimension> = D>,
    {
        let orig_ndim = array.shape().len();
        let new_ndim = orig_ndim + axis.len();
        check_ndim(new_ndim)?;
        let mut axes = dim_arr(axis.len(), |i| axis.get(i));

        // Each value in `axes` is a gap index in the *input* shape: 0 means "before input dim 0",
        // 1 means "before input dim 1" (i.e. between dims 0 and 1), ..., orig_ndim means "after
        // the last input dim". Duplicates are allowed - each occurrence inserts one additional
        // dim at that gap. Valid range: 0..=orig_ndim.
        for &ax in &axes {
            ensure!(
                ax <= orig_ndim,
                InvalidShapeOperation,
                "axis {ax} out of bounds for array of ndim {orig_ndim} \
                     (gap indices must be in 0..={orig_ndim})"
            );
        }

        // Sort a local copy so we can walk input dims and inserted gaps together in one pass.
        axes.sort_unstable();
        let mut sorted_axes = axes.iter().peekable();

        // Build is_inserted and shape by interleaving: for each gap `g` (0..=orig_ndim),
        // first emit all inserted dims that belong at gap `g`, then emit input dim `g`
        // (if g < orig_ndim).
        let mut is_inserted = DimArray::new();
        let mut shape = DimArray::new();

        for input_dim in 0..orig_ndim {
            while sorted_axes.peek() == Some(&&input_dim) {
                is_inserted.push(true);
                shape.push(1u64);
                sorted_axes.next();
            }
            is_inserted.push(false);
            shape.push(array.shape()[input_dim]);
        }
        // Remaining axes sit at gap orig_ndim (after the last input dim).
        for _ in sorted_axes {
            is_inserted.push(true);
            shape.push(1u64);
        }
        let shape = D::from_slice(&shape).unwrap();

        // Build blocks_layout: inserted dims get block_shape = 1 (Any); non-inserted dims
        // inherit the corresponding input dim's layout unchanged.
        let inner_layout = array.spec().blocks_layout;
        let mut hint = DimArray::new();
        let mut tag = DimArray::new();
        let mut preferred = DimArray::new();
        let mut input_dim = 0usize;
        for &inserted in is_inserted.iter() {
            if inserted {
                hint.push(1);
                tag.push(BlockShapeTag::Any);
                preferred.push(1);
            } else {
                hint.push(inner_layout.block_shape_hint[input_dim]);
                tag.push(inner_layout.block_shape_tag[input_dim]);
                preferred.push(inner_layout.preferred_read_shape[input_dim]);
                input_dim += 1;
            }
        }
        let mut b_layout = inner_layout.clone();
        b_layout.block_shape_hint = hint;
        b_layout.block_shape_tag = tag;
        b_layout.preferred_read_shape = preferred;

        Ok(Self {
            array,
            is_inserted,
            shape,
            blocks_layout: b_layout,
        })
    }

    /// Constructs an array with [`InsertAxis`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<Ax>(array: Array<S>, axis: Ax) -> Result<Array<Self>>
    where
        Ax: AxesArg<ExpandedDimension<S::Dimension> = D>,
    {
        Self::new(array.into_storage(), axis).map(Array::from_storage)
    }

    #[inline(always)]
    fn transform_index(&self, index: &[Range<u64>]) -> Result<Option<DimArray<Range<u64>>>> {
        check_get_range(self.shape(), index)?;

        // Inserted dimensions have size 1 and do not affect the element sequence.
        // Because the output is always C-contiguous, a size-1 dimension is a no-op
        // in the memory layout: its stride equals the stride of the next dimension,
        // and there is exactly one step along it, so the elements appear in the same
        // order as without it.
        //
        // Therefore we simply strip all inserted dims from `index` and forward the
        // remaining ranges to the inner storage unchanged. No temporary buffer or
        // element rearrangement is needed.
        let mut inner_index = DimArray::new();
        for (dim, index) in index.iter().enumerate() {
            if !self.is_inserted[dim] {
                inner_index.push(index.clone());
            } else if index.start == index.end {
                return Ok(None);
            }
        }
        Ok(Some(inner_index))
    }
}

impl<S, D> ArrayStorage for InsertAxis<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    type ElementType = S::ElementType;
    type Dimension = D;

    #[inline]
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        if let Some(inner_index) = self.transform_index(index)? {
            self.array.read_data(&inner_index, buf, context)
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, S, D>>
    where
        T: Dtyped,
    {
        let data = self
            .transform_index(index)?
            .map(|inner_index| self.array.read_data_typed(&inner_index, context))
            .transpose()?;
        struct ReadDataOptional<R>(Option<R>);
        impl<T, R> ReadData<T> for ReadDataOptional<R>
        where
            R: ReadData<T>,
        {
            fn len(&self) -> usize {
                self.0.as_ref().map_or(0, |r| r.len())
            }

            fn read_bulk<const N: usize>(&mut self, offset: usize) -> [T; N] {
                if let Some(r) = &mut self.0 {
                    r.read_bulk(offset)
                } else {
                    unimplemented!()
                }
            }
        }
        Ok(ReadDataOptional(data))
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.spec()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::codec::ReadContext;
    use crate::ops::InsertAxis;
    use crate::storage::Compact;
    use crate::util::{arr_params, shape_strategy, ScalarStrategy};
    use crate::{Array, Dim, DimDyn, Ty, NDIM_MAX};

    fn make1d(vals: Vec<i32>, block_size: usize) -> Array<Compact<Ty<i32>, Dim<1>>> {
        let nd = ndarray::Array::from_shape_vec([vals.len()], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[block_size])).unwrap()
    }

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<Compact<Ty<i32>, Dim<2>>> {
        let nd = ndarray::Array::from_shape_vec([rows, cols], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<Compact<Ty<i32>, Dim<3>>> {
        let nd = ndarray::Array::from_shape_vec([d0, d1, d2], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[d0, d1, d2])).unwrap()
    }

    fn arange(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_insert_before_first_dim() {
        // gap 0 on [6] -> [1, 6]
        assert_eq!(make1d(arange(6), 6).insert_axis(&[0]).shape(), &[1, 6]);
    }

    #[test]
    fn shape_insert_after_last_dim() {
        // gap 1 (=orig_ndim) on [6] -> [6, 1]
        assert_eq!(make1d(arange(6), 6).insert_axis(&[1]).shape(), &[6, 1]);
    }

    #[test]
    fn shape_insert_between_dims() {
        // gap 1 on [3, 4] -> [3, 1, 4]
        assert_eq!(
            make2d(arange(12), 3, 4).insert_axis(&[1]).shape(),
            &[3, 1, 4]
        );
    }

    #[test]
    fn shape_insert_front_and_back() {
        // gaps 0 and 1 on [6] -> [1, 6, 1]
        assert_eq!(
            make1d(arange(6), 6).insert_axis(&[0, 1]).shape(),
            &[1, 6, 1]
        );
    }

    #[test]
    fn shape_insert_duplicates_same_gap() {
        // gaps 0, 0 on [3, 4] -> [1, 1, 3, 4]
        assert_eq!(
            make2d(arange(12), 3, 4).insert_axis(&[0, 0]).shape(),
            &[1, 1, 3, 4]
        );
    }

    #[test]
    fn shape_insert_user_example() {
        // axes=(0,1,1,1,3) on (N=2, M=3, K=4) -> (1, 2, 1, 1, 1, 3, 4, 1)
        let a = make3d(arange(24), 2, 3, 4);
        assert_eq!(
            a.insert_axis(&[0, 1, 1, 1, 3]).shape(),
            &[1, 2, 1, 1, 1, 3, 4, 1]
        );
    }

    #[test]
    fn shape_insert_unsorted_axes_same_result() {
        // Order of axes values should not matter; only the multiset matters.
        let a1 = make3d(arange(24), 2, 3, 4).insert_axis(&[0, 1, 1, 1, 3]);
        let a2 = make3d(arange(24), 2, 3, 4).insert_axis(&[3, 1, 0, 1, 1]);
        assert_eq!(a1.shape(), a2.shape());
    }

    #[test]
    fn shape_empty_axes_is_identity() {
        assert_eq!(make2d(arange(12), 3, 4).insert_axis(&[]).shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_insert_before_first() {
        let got = make1d(arange(6), 6).insert_axis(&[0]).to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([1, 6], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_after_last() {
        let got = make1d(arange(6), 6).insert_axis(&[1]).to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([6, 1], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_between_dims() {
        let got = make2d(arange(12), 3, 4)
            .insert_axis(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 1, 4], arange(12)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_front_and_back() {
        let got = make1d(arange(6), 6)
            .insert_axis(&[0, 1])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([1, 6, 1], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_user_example() {
        // axes=(0,1,1,1,3) on (2,3,4) -> (1,2,1,1,1,3,4,1), elements unchanged
        let got = make3d(arange(24), 2, 3, 4)
            .insert_axis(&[0, 1, 1, 1, 3])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::ArrayD::from_shape_vec(vec![1, 2, 1, 1, 1, 3, 4, 1], arange(24)).unwrap()
        );
    }

    #[test]
    fn full_read_identity_empty_axes() {
        let got = make2d(arange(12), 3, 4)
            .insert_axis(&[])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], arange(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_inserted_dim_is_stripped() {
        // [1, 6]: read [0..1, 2..5] -> same as reading [2..5] from the 1D inner
        let got = make1d(arange(6), 6)
            .insert_axis(&[0])
            .to_ndarray_sub(&[0..1, 2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[2, 3, 4]]);
    }

    #[test]
    fn sub_read_2d_with_inserted_middle() {
        // [3, 1, 4]: read rows 1..3, inserted dim 0..1, cols 0..2
        let got = make2d(arange(12), 3, 4)
            .insert_axis(&[1])
            .to_ndarray_sub(&[1..3, 0..1, 0..2], &ReadContext::default())
            .unwrap();
        // row1=[4,5], row2=[8,9]
        assert_eq!(got, array![[[4, 5]], [[8, 9]]]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make1d(arange(4), 4);
        // orig_ndim=1, valid gaps are 0..=1; axis 2 is out of bounds
        assert!(InsertAxis::new_array(a, &[2]).is_err());
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary shape, arbitrary gap multiset, order-independent
    // -----------------------------------------------------------------------

    fn insert_axis_strategy<T>() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<T>,
            Array<Compact<Ty<T>, DimDyn>>,
            Vec<usize>,
        ),
    >
    where
        T: ScalarStrategy,
    {
        shape_strategy()
            .prop_flat_map(|shape| {
                let max_to_insert = NDIM_MAX - shape.len();
                (Just(shape), 0..=max_to_insert)
            })
            .prop_flat_map(|(shape, n_insert)| {
                let ndim = shape.len();
                let gaps = prop::collection::vec(0..=ndim, n_insert);
                (Just(shape), gaps)
            })
            .prop_flat_map(|(shape, axes)| {
                let array_strat =
                    crate::util::carray_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(axes).prop_shuffle())
            })
            .prop_map(|((nd, za), axes)| (nd, za, axes))
    }

    proptest::proptest! {
        #[test]
        fn proptest_insert_axis((nd, za, axes) in insert_axis_strategy::<i32>()) {
            // Oracle: inserting size-1 axes is a pure reshape - flat order is unchanged.
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable();
            let mut expected_shape: Vec<usize> = nd.shape().to_vec();
            for (i, &gap) in sorted_axes.iter().enumerate() {
                expected_shape.insert(gap + i, 1);
            }
            let expected = ndarray::ArrayD::from_shape_vec(
                expected_shape,
                nd.iter().cloned().collect::<Vec<_>>(),
            )
            .unwrap();
            crate::util::assert_array_matches(&za.insert_axis(&axes), &expected);
        }
    }
}
