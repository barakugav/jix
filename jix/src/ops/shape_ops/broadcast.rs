use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{
    bail, check_get_buffer_size, check_get_range, check_ndim, check_shape_overflow, ensure, Result,
};
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{ArraySpec, ArrayStorageInfo, BlockShapeTag, BlockSize, OutBuf};
use crate::util::{default_strides, dim_arr, DimArray, NdCopier};
use crate::{Array, ArrayStorage, Dimension};

/// Expands an array to a larger shape by repeating elements along length-1 dimensions,
/// returned by [`Array::broadcast`](crate::Array::broadcast).
///
/// The new shape (`shape`) must have the same number of dimensions as the input. For each dimension `d`,
/// either `shape[d] == input_shape[d]` (kept as-is) or `input_shape[d] == 1` (broadcast:
/// the single element is repeated `shape[d]` times). Any other combination is an error.
///
/// Output dtype equals the input dtype. Output shape equals `shape`. `Broadcast<S>` carries
/// `type Dimension = S::Dimension` - broadcasting does not change the number of axes so the
/// dimension type is preserved unchanged.
///
/// `Broadcast` is the lazy zero-cost case of replication restricted to length-1 axes; for
/// general element replication along an axis of any length use [`Repeat`](crate::ops::Repeat)
/// (each element duplicated in place) or [`Tile`](crate::ops::Tile) (the whole sequence
/// duplicated).
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // Row vector [1, 3] -> matrix [2, 3]: every row is identical
/// let a = Array::compact_ndarray(&array![[1i32, 2, 3]])?;
/// let result = a.broadcast(&[2, 3]).to_ndarray()?;
/// assert_eq!(result[[0, 0]], result[[1, 0]]);
/// assert_eq!(result[[0, 2]], result[[1, 2]]);
///
/// // Column vector [3, 1] -> matrix [3, 2]: every column is identical
/// let b = Array::compact_ndarray(&array![[10i32], [20], [30]])?;
/// let result = b.broadcast(&[3, 2]).to_ndarray()?;
/// assert_eq!(result[[0, 0]], 10);
/// assert_eq!(result[[0, 1]], 10);
/// assert_eq!(result[[2, 0]], 30);
/// assert_eq!(result[[2, 1]], 30);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Broadcast<S: ArrayStorage> {
    array: S,
    /// `is_broadcast[d]` is `true` when output dim `d` was expanded from length 1.
    is_broadcast: DimArray<bool>,
    /// `true` when `new_shape == input_shape` - no dimension was actually broadcast.
    /// In this case `read_data` forwards directly to the inner storage with no extra work.
    is_identity: bool,

    new_shape: S::Dimension,
    spec: ArraySpecDynamic,
}

impl<S> Broadcast<S>
where
    S: ArrayStorage,
{
    /// Constructs a [`Broadcast`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, shape: &[u64]) -> Result<Self> {
        let new_shape = shape;
        let input_shape = array.shape();
        let ndim = input_shape.len();

        ensure!(
            new_shape.len() == ndim,
            InvalidShapeOperation,
            "broadcast new_shape has {} dims but array has {ndim} dims",
            new_shape.len()
        );
        check_shape_overflow(new_shape, array.dtype().itemsize() as _)?;

        let mut is_broadcast = DimArray::new();
        for dim in 0..ndim {
            if new_shape[dim] == input_shape[dim] {
                is_broadcast.push(false);
            } else if input_shape[dim] == 1 {
                is_broadcast.push(true);
            } else {
                bail!(
                    InvalidShapeOperation,
                    "cannot broadcast dim {dim} from length {} to length {}",
                    input_shape[dim],
                    new_shape[dim]
                );
            }
        }
        let is_identity = is_broadcast.iter().all(|&b| !b);

        let new_shape = S::Dimension::from_slice(new_shape);

        // For broadcast dims: Any tag, block_shape=max. For unchanged dims: inherit from inner.
        let inner_spec = array.spec();
        let block_shape = dim_arr(ndim, |dim| {
            if is_broadcast[dim] {
                (new_shape[dim].min(BlockSize::MAX as u64) as BlockSize).max(1)
            } else {
                inner_spec.block_shape()[dim]
            }
        });
        let block_shape_tag = dim_arr(ndim, |dim| {
            if is_broadcast[dim] {
                BlockShapeTag::Any
            } else {
                inner_spec.block_shape_tag()[dim]
            }
        });
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_tag,
        };

        Ok(Self {
            array,
            is_broadcast,
            is_identity,
            new_shape,
            spec,
        })
    }

    /// Constructs an array with [`Broadcast`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, shape: &[u64]) -> Result<Array<Self>> {
        Self::new(array.into_storage(), shape).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for Broadcast<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    #[inline]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        // Fast path: no dimension was actually broadcast - forward directly.
        if self.is_identity {
            return self.array.read_data(index, buf, context);
        }

        let dtype = self.dtype();
        check_get_range(self.shape(), index)?;

        let ndim = self.is_broadcast.len();
        let itemsize = dtype.itemsize() as usize;

        // Read from inner with broadcast dims collapsed to 0..1.
        // tmp_buf is C-contiguous over inner_read_shape.
        let inner_index = S::Dimension::vec(ndim, |dim| {
            if self.is_broadcast[dim] {
                0..1
            } else {
                index[dim].clone()
            }
        });
        let inner_read_shape = S::Dimension::vec(ndim, |dim| {
            (inner_index[dim].end - inner_index[dim].start) as usize
        });
        let mut tmp_buf = OutBuf::new_lazy(context);
        self.array
            .read_data(inner_index.as_ref(), &mut tmp_buf, context)?;
        let tmp_buf = tmp_buf.as_slice().unwrap();
        let mut buf = buf.get_continuous_mut(index, dtype, context);
        buf.edit(|buf| {
            check_get_buffer_size(index, dtype, buf)?;

            // Source strides over tmp_buf, with broadcast dims set to 0.
            // A zero stride means advancing along that output axis always reads the same src byte,
            // which is exactly the repeat-element semantics of broadcasting.
            let mut src_strides = default_strides(&inner_read_shape, itemsize);
            for dim in 0..ndim {
                if self.is_broadcast[dim] {
                    src_strides[dim] = 0;
                }
            }

            // Destination strides: C-contiguous over the requested output sub-shape.
            let out_shape =
                S::Dimension::vec(ndim, |dim| (index[dim].end - index[dim].start) as usize);
            let dst_strides = default_strides(&out_shape, itemsize);

            let copier = NdCopier::new(dtype);
            unsafe {
                copier.copy(
                    tmp_buf.as_ptr(),
                    buf.as_mut_ptr(),
                    out_shape.as_ref(),
                    src_strides.as_ref(),
                    dst_strides.as_ref(),
                    dtype,
                )
            };
            Ok(())
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.new_shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Broadcast", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Broadcast<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        check_ndim::<NewD>(self.shape().len())?;
        let new_shape = NewD::from_slice(self.shape());

        Ok(Broadcast {
            array: self.array.dimension_change()?,
            is_broadcast: self.is_broadcast,
            is_identity: self.is_identity,
            new_shape,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = Broadcast<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Broadcast {
            array: self.array.element_type_change()?,
            is_broadcast: self.is_broadcast,
            is_identity: self.is_identity,
            new_shape: self.new_shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::codec::ReadContext;
    use crate::storage::Compact;
    use crate::util::{shape_strategy, ScalarStrategy};
    use crate::{Array, DimDyn, IntoDimension, Ty, NDIM_MAX};

    fn make<Sh>(vals: Vec<i32>, shape: Sh) -> Array<Compact<Ty<i32>, Sh::Dimension>>
    where
        Sh: IntoDimension,
    {
        let shape = shape.into_dimension().unwrap();
        let nd = ndarray::Array::from_shape_vec(shape, vals).unwrap();
        Array::compact_ndarray(&nd).unwrap().into_dim().unwrap()
    }

    fn arange(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_broadcast_axis0() {
        // [1, 4] -> [3, 4]
        assert_eq!(make(arange(4), &[1, 4]).broadcast(&[3, 4]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_broadcast_axis1() {
        // [3, 1] -> [3, 4]
        assert_eq!(make(arange(3), &[3, 1]).broadcast(&[3, 4]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_broadcast_both_axes() {
        // [1, 1] -> [3, 4]
        assert_eq!(make(vec![7], &[1, 1]).broadcast(&[3, 4]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_no_broadcast_is_identity() {
        assert_eq!(
            make(arange(12), &[3, 4]).broadcast(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_3d_middle() {
        // [2, 1, 4] -> [2, 3, 4]
        assert_eq!(
            make(arange(8), &[2, 1, 4]).broadcast(&[2, 3, 4]).shape(),
            &[2, 3, 4]
        );
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_broadcast_axis0() {
        // [1, 4] -> [3, 4]: each row is [0,1,2,3]
        let got = make(arange(4), &[1, 4])
            .broadcast(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]);
    }

    #[test]
    fn full_read_broadcast_axis1() {
        // [3, 1] -> [3, 4]: each col is [0,1,2]
        let got = make(arange(3), &[3, 1])
            .broadcast(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]);
    }

    #[test]
    fn full_read_broadcast_both() {
        // [1, 1] -> [2, 3]: all elements == 7
        let got = make(vec![7], &[1, 1])
            .broadcast(&[2, 3])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[7, 7, 7], [7, 7, 7]]);
    }

    #[test]
    fn full_read_no_broadcast() {
        let got = make(arange(12), &[3, 4])
            .broadcast(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], arange(12)).unwrap()
        );
    }

    #[test]
    fn full_read_broadcast_3d_middle() {
        // [2, 1, 3] -> [2, 4, 3]: axis 1 repeats 4 times
        let got = make(arange(6), &[2, 1, 3])
            .broadcast(&[2, 4, 3])
            .to_ndarray()
            .unwrap();
        // row 0 of inner: [0,1,2], row 1: [3,4,5], each repeated 4 times along axis 1
        assert_eq!(
            got,
            array![
                [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
                [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_broadcast_axis0() {
        // [1, 4] -> [3, 4]: read rows 1..3, cols 1..3
        let got = make(arange(4), &[1, 4])
            .broadcast(&[3, 4])
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        // each row is [1, 2]
        assert_eq!(got, array![[1, 2], [1, 2]]);
    }

    #[test]
    fn sub_read_broadcast_axis1() {
        // [3, 1] -> [3, 5]: read rows 0..2, cols 2..5 (all same element per row)
        let got = make(arange(3), &[3, 1])
            .broadcast(&[3, 5])
            .to_ndarray_sub(&[0..2, 2..5], &ReadContext::default())
            .unwrap();
        // row 0: [0,0,0], row 1: [1,1,1]
        assert_eq!(got, array![[0, 0, 0], [1, 1, 1]]);
    }

    // -----------------------------------------------------------------------
    // Identity fast path
    // -----------------------------------------------------------------------

    #[test]
    fn identity_flag_set_when_no_broadcast() {
        let a = make(arange(12), &[3, 4]);
        let b = super::Broadcast::new_array(a.as_ref(), &[3, 4])
            .unwrap()
            .into_storage();
        assert!(b.is_identity);
    }

    #[test]
    fn identity_flag_not_set_when_broadcast() {
        let a = make(arange(4), &[1, 4]);
        let b = super::Broadcast::new_array(a.as_ref(), &[3, 4])
            .unwrap()
            .into_storage();
        assert!(!b.is_identity);
    }

    #[test]
    fn identity_full_read_correct() {
        let got = make(arange(12), &[3, 4])
            .broadcast(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], arange(12)).unwrap()
        );
    }

    #[test]
    fn identity_sub_read_correct() {
        let got = make(arange(12), &[3, 4])
            .broadcast(&[3, 4])
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        // rows 1..3, cols 1..3 of [[0,1,2,3],[4,5,6,7],[8,9,10,11]] = [[5,6],[9,10]]
        assert_eq!(got, array![[5, 6], [9, 10]]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_ndim_mismatch() {
        let a = make(arange(6), &[2, 3]);
        assert!(super::Broadcast::new_array(a.as_ref(), &[2, 3, 1]).is_err());
    }

    #[test]
    fn error_non_unit_dim_broadcast() {
        let a = make(arange(6), &[2, 3]);
        // axis 0 has length 2, cannot broadcast to 5
        assert!(super::Broadcast::new_array(a.as_ref(), &[5, 3]).is_err());
    }

    // -----------------------------------------------------------------------
    // Proptest: random data, representative broadcast patterns
    // -----------------------------------------------------------------------

    fn broadcast_2d_axis0_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            Array<Compact<Ty<i32>, DimDyn>>,
            usize,
            usize,
        ),
    > {
        use proptest::prelude::*;
        (1usize..=15, 1usize..=15).prop_flat_map(|(n, m)| {
            crate::util::carray_strategy_from_shape::<i32>(
                proptest::strategy::Just(vec![1, m]),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
            .prop_map(move |(nd, za)| (nd, za, n, m))
        })
    }

    fn broadcast_2d_axis1_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            Array<Compact<Ty<i32>, DimDyn>>,
            usize,
            usize,
        ),
    > {
        use proptest::prelude::*;
        (1usize..=15, 1usize..=15).prop_flat_map(|(n, m)| {
            crate::util::carray_strategy_from_shape::<i32>(
                proptest::strategy::Just(vec![n, 1]),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
            .prop_map(move |(nd, za)| (nd, za, n, m))
        })
    }

    proptest::proptest! {
        // [1] -> [N]
        #[test]
        fn proptest_broadcast_1d(
            n in 1usize..=30,
            (nd, za) in crate::util::carray_strategy_from_shape::<i32>(
                proptest::strategy::Just(vec![1]),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
        ) {
            let expected = nd.broadcast(vec![n]).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast(&[n as u64]), &expected);
        }

        // [1, M] -> [N, M]: broadcast axis 0
        #[test]
        fn proptest_broadcast_2d_axis0(
            (nd, za, n, m) in broadcast_2d_axis0_strategy()
        ) {
            let expected = nd.broadcast(vec![n, m]).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast(&[n as u64, m as u64]), &expected);
        }

        // [N, 1] -> [N, M]: broadcast axis 1
        #[test]
        fn proptest_broadcast_2d_axis1(
            (nd, za, n, m) in broadcast_2d_axis1_strategy()
        ) {
            let expected = nd.broadcast(vec![n, m]).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast(&[n as u64, m as u64]), &expected);
        }

        // [N, M] -> [N, M]: identity (no broadcast)
        #[test]
        fn proptest_broadcast_identity(
            (nd, za) in crate::util::carray_strategy_from_shape::<i32>(
                proptest::collection::vec(1usize..=15, 2usize),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
        ) {
            let shape: Vec<u64> = nd.shape().iter().map(|&s| s as u64).collect();
            let expected = nd.clone();
            crate::util::assert_array_matches(&za.broadcast(&shape), &expected);
        }

        #[test]
        fn broadcast_generic(
            (nd, za, broadcast_shape) in broadcast_axes_strategy::<i32>()
        ) {
            let expected = nd.broadcast(broadcast_shape.clone()).unwrap().to_owned();
            let broadcast_shape = broadcast_shape.iter().map(|&s| s as u64).collect::<Vec<_>>();
            let actual = za.broadcast(&broadcast_shape);
            crate::util::assert_array_matches(&actual, &expected);
        }
    }

    use proptest::prelude::*;

    fn broadcast_axes_strategy<T>() -> impl proptest::strategy::Strategy<
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
                let max_dims_to_broadcast = NDIM_MAX - shape.len();
                (Just(shape), 0..=max_dims_to_broadcast)
            })
            .prop_flat_map(|(shape, ndims_to_broadcast)| {
                let dims_to_broadcast = prop::collection::vec(0..=shape.len(), ndims_to_broadcast);
                (Just(shape), dims_to_broadcast)
            })
            .prop_flat_map(|(mut shape, mut dims_to_broadcast)| {
                dims_to_broadcast.sort_unstable();
                for (i, dim) in dims_to_broadcast.iter_mut().enumerate() {
                    let shift = i;
                    shape.insert(shift + *dim, 1);
                    *dim += shift;
                }

                let broadcasted_dims_sizes =
                    prop::collection::vec(1usize..=5, dims_to_broadcast.len());
                let broadcast_shape = shape.clone();
                let broadcast_shape =
                    broadcasted_dims_sizes.prop_map(move |broadcasted_dims_sizes| {
                        let mut broadcast_shape = broadcast_shape.clone();
                        for (&dim, &size) in
                            dims_to_broadcast.iter().zip(broadcasted_dims_sizes.iter())
                        {
                            broadcast_shape[dim] = size;
                        }
                        broadcast_shape
                    });

                (Just(shape), broadcast_shape)
            })
            .prop_flat_map(|(shape, broadcast_shape)| {
                let array_strat =
                    crate::util::carray_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(broadcast_shape))
            })
            .prop_map(|((nd, za), broadcast_shape)| (nd, za, broadcast_shape))
    }
}
