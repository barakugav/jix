use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{bail, check_get_range, check_ndim, check_shape_overflow, ensure, Result};
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{check_out_buf, ArraySpec, ArrayStorageInfo, BlockSize, StridedBuf};
use crate::util::{dim_arr, DimArray};
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
    is_broadcast: <S::Dimension as Dimension>::Vec<bool>,

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

        let mut is_broadcast = S::Dimension::vec(ndim, |_| false);
        for dim in 0..ndim {
            if new_shape[dim] == input_shape[dim] {
                is_broadcast[dim] = false;
            } else if input_shape[dim] == 1 {
                is_broadcast[dim] = true;
            } else {
                bail!(
                    InvalidShapeOperation,
                    "cannot broadcast dim {dim} from length {} to length {}",
                    input_shape[dim],
                    new_shape[dim]
                );
            }
        }

        let new_shape = S::Dimension::from_slice(new_shape);

        // For broadcast dims: not fixed, block_shape=max. For unchanged dims: inherit from inner.
        let inner_spec = array.spec();
        let block_shape = dim_arr(ndim, |dim| {
            if is_broadcast[dim] {
                (new_shape[dim].min(BlockSize::MAX as u64) as BlockSize).max(1)
            } else {
                inner_spec.block_shape()[dim]
            }
        });
        let mut block_shape_fixed_dims = inner_spec.block_shape_fixed_dims();
        for dim in 0..ndim {
            if is_broadcast[dim] {
                block_shape_fixed_dims.set(dim, false);
            }
        }
        let read_shape_scale_order = {
            let in_order = inner_spec.read_shape_scale_order();
            let broadcasted_dims = in_order
                .iter()
                .copied()
                .filter(|&d| is_broadcast[d as usize]);
            let non_broadcasted_dims = in_order
                .iter()
                .copied()
                .filter(|&d| !is_broadcast[d as usize]);
            // A broadcast dim re-reads the same inner element `new_shape[dim]` times, so covering each
            // in full with one read avoids that redundant work: move the broadcast dims to the front of
            // the scaling order (highest priority), keeping the inner relative order within both groups.
            broadcasted_dims
                .chain(non_broadcasted_dims)
                .collect::<DimArray<_>>()
        };
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims,
            element_cost: inner_spec.element_cost(),
            read_shape_scale_order,
            read_layout_order: inner_spec.read_layout_order().clone(),
        };

        Ok(Self {
            array,
            is_broadcast,
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
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_get_range(self.shape(), index)?;
        check_out_buf(out.as_deref(), self.shape())?;
        let ndim = self.is_broadcast.as_ref().len();

        let inner_index = S::Dimension::vec(ndim, |dim| {
            if self.is_broadcast[dim] {
                0..1
            } else {
                index[dim].clone()
            }
        });
        let inner = self.array.read_data(inner_index.as_ref(), context, None)?;

        let inner_strides = inner.strides();
        let broadcasted_strides = S::Dimension::vec(ndim, |dim| {
            if self.is_broadcast[dim] {
                0
            } else {
                inner_strides[dim]
            }
        });

        match out {
            None => Ok(unsafe { inner.with_strides(broadcasted_strides.as_ref()) }),
            Some(out) => {
                let out_shape =
                    S::Dimension::vec(ndim, |dim| (index[dim].end - index[dim].start) as usize);
                let (src, _) = inner.data();
                // SAFETY: `src_strides` are the inner view's strides with broadcast axes zeroed
                unsafe {
                    out.copy_from(
                        src,
                        broadcasted_strides.as_ref(),
                        out_shape.as_ref(),
                        self.dtype(),
                    )
                };
                Ok(out.view_mut())
            }
        }
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
        let ndim = self.shape().len();
        check_ndim::<NewD>(ndim)?;
        let new_shape = NewD::from_slice(self.shape());
        let is_broadcast = NewD::vec(ndim, |dim| self.is_broadcast[dim]);

        Ok(Broadcast {
            array: self.array.dimension_change()?,
            is_broadcast,
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
    use crate::{Array, IntoDimension, Ty, NDIM_MAX};

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
        assert_eq!(make(arange(4), &[1, 4]).broadcast(&[3, 4]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_broadcast_axis1() {
        assert_eq!(make(arange(3), &[3, 1]).broadcast(&[3, 4]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_broadcast_both_axes() {
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
        let got = make(arange(4), &[1, 4])
            .broadcast(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]);
    }

    #[test]
    fn full_read_broadcast_axis1() {
        let got = make(arange(3), &[3, 1])
            .broadcast(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]);
    }

    #[test]
    fn full_read_broadcast_both() {
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
        let got = make(arange(6), &[2, 1, 3])
            .broadcast(&[2, 4, 3])
            .to_ndarray()
            .unwrap();
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
        let got = make(arange(4), &[1, 4])
            .broadcast(&[3, 4])
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[1, 2], [1, 2]]);
    }

    #[test]
    fn sub_read_broadcast_axis1() {
        let got = make(arange(3), &[3, 1])
            .broadcast(&[3, 5])
            .to_ndarray_sub(&[0..2, 2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[0, 0, 0], [1, 1, 1]]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_ndim_mismatch() {
        let a = make(arange(6), &[2, 3]);
        assert!(super::Broadcast::new_array(a.view(), &[2, 3, 1]).is_err());
    }

    #[test]
    fn error_non_unit_dim_broadcast() {
        let a = make(arange(6), &[2, 3]);
        // axis 0 has length 2, cannot broadcast to 5
        assert!(super::Broadcast::new_array(a.view(), &[5, 3]).is_err());
    }

    // -----------------------------------------------------------------------
    // Proptest: random data, representative broadcast patterns
    // -----------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn broadcast_2d_axis0_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            crate::util::TestArray<i32>,
            usize,
            usize,
        ),
    > {
        use proptest::prelude::*;
        (1usize..=15, 1usize..=15).prop_flat_map(|(n, m)| {
            crate::util::array_strategy_from_shape::<i32>(
                proptest::strategy::Just(vec![1, m]),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
            .prop_map(move |(nd, za)| (nd, za, n, m))
        })
    }

    proptest::proptest! {
        #[test]
        fn proptest_broadcast_1d(
            n in 1usize..=30,
            (nd, za) in crate::util::array_strategy_from_shape::<i32>(
                proptest::strategy::Just(vec![1]),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            )
        ) {
            let expected = nd.broadcast(vec![n]).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast(&[n as u64]), &expected);
        }

        #[test]
        fn proptest_broadcast_2d_axis0(
            (nd, za, n, m) in broadcast_2d_axis0_strategy()
        ) {
            let expected = nd.broadcast(vec![n, m]).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast(&[n as u64, m as u64]), &expected);
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

    // -----------------------------------------------------------------------
    // Concrete: [N, 1] -> [N, M] (axis 1) and [N, M] -> [N, M] (identity).
    // Fixed inputs cover the shape edges: a trivial 1x1 array and a
    // multi-element, non-square shape, plus dtype min/max among the values.
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_2d_axis1_concrete() {
        for (n, m, vals) in [
            (1usize, 1usize, vec![i32::MAX]),
            (4usize, 5usize, vec![i32::MIN, -7, 0, i32::MAX]),
        ] {
            let nd = ndarray::Array::from_shape_vec([n, 1], vals.clone()).unwrap();
            let za = make(vals, &[n as u64, 1]);
            let expected = nd.broadcast(vec![n, m]).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast(&[n as u64, m as u64]), &expected);
        }
    }

    #[test]
    fn broadcast_identity_concrete() {
        // Identity fast path (no dimension actually broadcast).
        for (n, m, vals) in [
            (1usize, 1usize, vec![i32::MAX]),
            (3usize, 4usize, {
                let mut v: Vec<i32> = (0..12i32).collect();
                v[0] = i32::MIN;
                v[11] = i32::MAX;
                v
            }),
        ] {
            let nd = ndarray::Array::from_shape_vec([n, m], vals.clone()).unwrap();
            let za = make(vals, &[n as u64, m as u64]);
            let shape: Vec<u64> = nd.shape().iter().map(|&s| s as u64).collect();
            let expected = nd.clone();
            crate::util::assert_array_matches(&za.broadcast(&shape), &expected);
        }
    }

    use proptest::prelude::*;

    #[allow(clippy::type_complexity)]
    fn broadcast_axes_strategy<T>() -> impl proptest::strategy::Strategy<
        Value = (ndarray::ArrayD<T>, crate::util::TestArray<T>, Vec<usize>),
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
                    crate::util::array_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(broadcast_shape))
            })
            .prop_map(|((nd, za), broadcast_shape)| (nd, za, broadcast_shape))
    }
}
