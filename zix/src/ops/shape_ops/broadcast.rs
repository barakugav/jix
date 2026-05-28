use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::{default_strides, dim_arr, nd_copy, DimArray};
use crate::{Array, Dimension};

/// Expands an array to a larger shape by repeating elements along length-1 dimensions,
/// returned by [`Array::broadcast_view`](crate::Array::broadcast_view).
///
/// The new shape (`shape`) must have the same number of dimensions as the input. For each dimension `d`,
/// either `shape[d] == input_shape[d]` (kept as-is) or `input_shape[d] == 1` (broadcast:
/// the single element is repeated `shape[d]` times). Any other combination is an error.
///
/// Output dtype equals the input dtype. Output shape equals `shape`. `Broadcast<S>` carries
/// `type Dimension = S::Dimension` — broadcasting does not change the number of axes so the
/// dimension type is preserved unchanged.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// // Row vector [1, 3] -> matrix [2, 3]: every row is identical
/// let a = Array::compact_array(&array![[1i32, 2, 3]])?;
/// let result = a.broadcast_view(&[2, 3]).to_ndarray::<i32>()?;
/// assert_eq!(result[[0, 0]], result[[1, 0]]);
/// assert_eq!(result[[0, 2]], result[[1, 2]]);
///
/// // Column vector [3, 1] -> matrix [3, 2]: every column is identical
/// let b = Array::compact_array(&array![[10i32], [20], [30]])?;
/// let result = b.broadcast_view(&[3, 2]).to_ndarray::<i32>()?;
/// assert_eq!(result[[0, 0]], 10);
/// assert_eq!(result[[0, 1]], 10);
/// assert_eq!(result[[2, 0]], 30);
/// assert_eq!(result[[2, 1]], 30);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct Broadcast<S: ArrayStorage> {
    array: Array<S>,
    /// `is_broadcast[d]` is `true` when output dim `d` was expanded from length 1.
    is_broadcast: DimArray<bool>,
    /// `true` when `new_shape == input_shape` - no dimension was actually broadcast.
    /// In this case `read_data` forwards directly to the inner storage with no extra work.
    is_identity: bool,

    new_shape: S::Dimension,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> Broadcast<S> {
    /// Constructs a `Broadcast` storage. See [`Broadcast`] for semantics and examples.
    pub fn new(array: Array<S>, shape: &[u64]) -> Result<Self> {
        let new_shape = shape;
        let input_shape = array.shape();
        let ndim = input_shape.len();

        ensure!(
            new_shape.len() == ndim,
            InvalidShapeOperation,
            "broadcast new_shape has {} dims but array has {ndim} dims",
            new_shape.len()
        );

        let mut is_broadcast = DimArray::new();
        for d in 0..ndim {
            if new_shape[d] == input_shape[d] {
                is_broadcast.push(false);
            } else if input_shape[d] == 1 {
                is_broadcast.push(true);
            } else {
                bail!(
                    InvalidShapeOperation,
                    "cannot broadcast dim {d} from length {} to length {}",
                    input_shape[d],
                    new_shape[d]
                );
            }
        }
        let is_identity = is_broadcast.iter().all(|&b| !b);

        let new_shape = S::Dimension::from_slice(new_shape).unwrap();
        let new_shape_slice = new_shape.as_slice();

        // For broadcast dims: Any tag, hint=1, preferred=new size (full extent reads are free).
        // For unchanged dims: inherit from inner.
        let mut b_layout = array.blocks_layout().clone();
        b_layout.block_shape_hint = dim_arr(ndim, |d| {
            if is_broadcast[d] {
                1
            } else {
                b_layout.block_shape_hint[d]
            }
        });
        b_layout.block_shape_tag = dim_arr(ndim, |d| {
            if is_broadcast[d] {
                BlockShapeTag::Any
            } else {
                b_layout.block_shape_tag[d]
            }
        });
        b_layout.preferred_read_shape = dim_arr(ndim, |d| {
            if is_broadcast[d] {
                new_shape_slice[d] as u32
            } else {
                b_layout.preferred_read_shape[d]
            }
        });

        Ok(Self {
            array,
            is_broadcast,
            is_identity,
            new_shape,
            blocks_layout: b_layout,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for Broadcast<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        // Fast path: no dimension was actually broadcast - forward directly.
        if self.is_identity {
            return self.array.storage.read_data(index, buf, context);
        }

        let dtype = self.dtype();
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, dtype, buf)?;

        let ndim = self.is_broadcast.len();
        let itemsize = dtype.itemsize() as usize;

        // Read from inner with broadcast dims collapsed to 0..1.
        // tmp_buf is C-contiguous over inner_read_shape.
        let inner_index = dim_arr(ndim, |d| {
            if self.is_broadcast[d] {
                0..1
            } else {
                index[d].clone()
            }
        });
        let inner_read_shape = dim_arr(ndim, |d| {
            (inner_index[d].end - inner_index[d].start) as usize
        });
        let n_bytes = inner_read_shape.iter().product::<usize>() * itemsize;
        let mut tmp_buf = context.tmp_buf(n_bytes, dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array
            .storage
            .read_data(&inner_index, tmp_buf, context)?;

        // Source strides over tmp_buf, with broadcast dims set to 0.
        // A zero stride means advancing along that output axis always reads the same src byte,
        // which is exactly the repeat-element semantics of broadcasting.
        let mut src_strides = default_strides(&inner_read_shape, itemsize);
        for d in 0..ndim {
            if self.is_broadcast[d] {
                src_strides[d] = 0;
            }
        }

        // Destination strides: C-contiguous over the requested output sub-shape.
        let out_shape = dim_arr(ndim, |d| (index[d].end - index[d].start) as usize);
        let dst_strides = default_strides(&out_shape, itemsize);

        unsafe {
            nd_copy(
                tmp_buf.as_ptr(),
                buf.as_mut_ptr(),
                &out_shape,
                &src_strides,
                &dst_strides,
                itemsize,
            )
        };
        Ok(())
    }

    fn shape(&self) -> &[u64] {
        self.new_shape.as_slice()
    }
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage._spec()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::array::Array;
    use crate::codec::ReadContext;
    use crate::storage::{Compact, Ty};
    use crate::util::{shape_strategy, ScalarStrategy};
    use crate::{DimDyn, NDIM_MAX};

    fn make(vals: Vec<i32>, shape: &[usize]) -> Array<Compact<Ty<i32>, DimDyn>> {
        let nd = ndarray::ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
        Array::compact_array(&nd).unwrap()
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
        assert_eq!(
            make(arange(4), &[1, 4]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_axis1() {
        // [3, 1] -> [3, 4]
        assert_eq!(
            make(arange(3), &[3, 1]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_both_axes() {
        // [1, 1] -> [3, 4]
        assert_eq!(
            make(vec![7], &[1, 1]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_no_broadcast_is_identity() {
        assert_eq!(
            make(arange(12), &[3, 4]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_3d_middle() {
        // [2, 1, 4] -> [2, 3, 4]
        assert_eq!(
            make(arange(8), &[2, 1, 4])
                .broadcast_view(&[2, 3, 4])
                .shape(),
            &[2, 3, 4]
        );
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_broadcast_axis0() {
        // [1, 4] -> [3, 4]: each row is [0,1,2,3]
        let got: ArrayD<i32> = make(arange(4), &[1, 4])
            .broadcast_view(&[3, 4])
            .to_ndarray()
            .unwrap();
        let expected =
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn full_read_broadcast_axis1() {
        // [3, 1] -> [3, 4]: each col is [0,1,2]
        let got: ArrayD<i32> = make(arange(3), &[3, 1])
            .broadcast_view(&[3, 4])
            .to_ndarray()
            .unwrap();
        let expected =
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn full_read_broadcast_both() {
        // [1, 1] -> [2, 3]: all elements == 7
        let got: ArrayD<i32> = make(vec![7], &[1, 1])
            .broadcast_view(&[2, 3])
            .to_ndarray()
            .unwrap();
        let expected = ArrayD::from_shape_vec(vec![2, 3], vec![7; 6]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn full_read_no_broadcast() {
        let got: ArrayD<i32> = make(arange(12), &[3, 4])
            .broadcast_view(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], arange(12)).unwrap());
    }

    #[test]
    fn full_read_broadcast_3d_middle() {
        // [2, 1, 3] -> [2, 4, 3]: axis 1 repeats 4 times
        let got: ArrayD<i32> = make(arange(6), &[2, 1, 3])
            .broadcast_view(&[2, 4, 3])
            .to_ndarray()
            .unwrap();
        // row 0 of inner: [0,1,2], row 1: [3,4,5], each repeated 4 times along axis 1
        let expected = ArrayD::from_shape_vec(
            vec![2, 4, 3],
            vec![
                0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5,
            ],
        )
        .unwrap();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_broadcast_axis0() {
        // [1, 4] -> [3, 4]: read rows 1..3, cols 1..3
        let got: ArrayD<i32> = make(arange(4), &[1, 4])
            .broadcast_view(&[3, 4])
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        // each row is [1, 2]
        let expected = ArrayD::from_shape_vec(vec![2, 2], vec![1, 2, 1, 2]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn sub_read_broadcast_axis1() {
        // [3, 1] -> [3, 5]: read rows 0..2, cols 2..5 (all same element per row)
        let got: ArrayD<i32> = make(arange(3), &[3, 1])
            .broadcast_view(&[3, 5])
            .to_ndarray_sub(&[0..2, 2..5], &ReadContext::default())
            .unwrap();
        // row 0: [0,0,0], row 1: [1,1,1]
        let expected = ArrayD::from_shape_vec(vec![2, 3], vec![0, 0, 0, 1, 1, 1]).unwrap();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Identity fast path
    // -----------------------------------------------------------------------

    #[test]
    fn identity_flag_set_when_no_broadcast() {
        let a = make(arange(12), &[3, 4]);
        let b = super::Broadcast::new(a.as_ref(), &[3, 4]).unwrap();
        assert!(b.is_identity);
    }

    #[test]
    fn identity_flag_not_set_when_broadcast() {
        let a = make(arange(4), &[1, 4]);
        let b = super::Broadcast::new(a.as_ref(), &[3, 4]).unwrap();
        assert!(!b.is_identity);
    }

    #[test]
    fn identity_full_read_correct() {
        let got: ArrayD<i32> = make(arange(12), &[3, 4])
            .broadcast_view(&[3, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], arange(12)).unwrap());
    }

    #[test]
    fn identity_sub_read_correct() {
        let got: ArrayD<i32> = make(arange(12), &[3, 4])
            .broadcast_view(&[3, 4])
            .to_ndarray_sub(&[1..3, 1..3], &ReadContext::default())
            .unwrap();
        // rows 1..3, cols 1..3 of [[0,1,2,3],[4,5,6,7],[8,9,10,11]] = [[5,6],[9,10]]
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![5, 6, 9, 10]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_ndim_mismatch() {
        let a = make(arange(6), &[2, 3]);
        assert!(super::Broadcast::new(a.as_ref(), &[2, 3, 1]).is_err());
    }

    #[test]
    fn error_non_unit_dim_broadcast() {
        let a = make(arange(6), &[2, 3]);
        // axis 0 has length 2, cannot broadcast to 5
        assert!(super::Broadcast::new(a.as_ref(), &[5, 3]).is_err());
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
            let expected = nd.broadcast(ndarray::IxDyn(&[n])).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast_view(&[n as u64]), &expected);
        }

        // [1, M] -> [N, M]: broadcast axis 0
        #[test]
        fn proptest_broadcast_2d_axis0(
            (nd, za, n, m) in broadcast_2d_axis0_strategy()
        ) {
            let expected = nd.broadcast(ndarray::IxDyn(&[n, m])).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast_view(&[n as u64, m as u64]), &expected);
        }

        // [N, 1] -> [N, M]: broadcast axis 1
        #[test]
        fn proptest_broadcast_2d_axis1(
            (nd, za, n, m) in broadcast_2d_axis1_strategy()
        ) {
            let expected = nd.broadcast(ndarray::IxDyn(&[n, m])).unwrap().to_owned();
            crate::util::assert_array_matches(&za.broadcast_view(&[n as u64, m as u64]), &expected);
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
            crate::util::assert_array_matches(&za.broadcast_view(&shape), &expected);
        }

        #[test]
        fn broadcast_generic(
            (nd, za, broadcast_shape) in broadcast_axes_strategy::<i32>()
        ) {
            let expected = nd.broadcast(ndarray::IxDyn(&broadcast_shape)).unwrap().to_owned();
            let broadcast_shape = broadcast_shape.iter().map(|&s| s as u64).collect::<Vec<_>>();
            let actual = za.broadcast_view(&broadcast_shape);
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
