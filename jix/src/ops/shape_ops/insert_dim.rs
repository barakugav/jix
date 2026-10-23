use crate::ops::prelude::*;

/// Inserts new length-1 dimensions at specified positions in an array's shape,
/// returned by [`Array::insert_dim`](crate::Array::insert_dim). The inverse operation
/// is [`RemoveDim`](crate::ops::RemoveDim).
///
/// Like `numpy.expand_dims`, each element of `dim` is a dim index in the *output* shape, whose
/// ndim is `orig_ndim + dim.len()`. Each named output dim is a new length-1 dim; the input dims
/// fill the remaining positions in order. Valid indices are `0..orig_ndim + dim.len()`.
/// Duplicates are not allowed. The order of values in `dim` does not matter.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```text
/// [N]       dim: [0]     -> [1, N]      (insert before first dim)
/// [N]       dim: [1]     -> [N, 1]      (append after last dim)
/// [N, M]    dim: [1]     -> [N, 1, M]   (insert between dims)
/// [N, M]    dim: [0, 3]  -> [1, N, M, 1]
/// [N, M]    dim: [0, 1]  -> [1, 1, N, M]
/// ```
///
/// The dims can be given as any [`DimsArg`](crate::ops::DimsArg): a single `usize`, an array or tuple of `usize`, or a
/// slice:
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?; // shape [3]
///
/// assert_eq!(a.view().insert_dim(0).shape(), &[1, 3]);
/// assert_eq!(a.view().insert_dim(1).shape(), &[3, 1]);
/// assert_eq!(a.view().insert_dim([0, 2]).shape(), &[1, 3, 1]);
///
/// let dims = vec![0, 2];
/// assert_eq!(a.view().insert_dim(dims.as_slice()).shape(), &[1, 3, 1]);
///
/// assert_eq!(a.view().insert_dim([0, 1, 3, 4]).shape(), &[1, 1, 3, 1, 1]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct InsertDim<S: ArrayStorage> {
    array: S,
    original_dims: DimArray<DimIdx>,

    shape: DimDyn,
    spec: ArraySpecDynamic,
}

impl<S> InsertDim<S>
where
    S: ArrayStorage,
{
    /// Constructs an [`InsertDim`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, dim: impl DimsArg) -> Result<Self> {
        let orig_ndim = array.shape().len();
        let new_ndim = orig_ndim + dim.len();
        check_ndim(new_ndim)?;

        // Each value in `dim` is a position in the *output* shape that becomes a new length-1
        // dim; the input dims fill the remaining positions in order.
        let mut is_inserted = dim_arr(new_ndim, |_| false);
        for i in 0..dim.len() {
            let d = dim.get(i);
            ensure!(
                d < new_ndim,
                InvalidShapeOperation,
                "dim {d} out of bounds for output ndim {new_ndim} (must be in 0..{new_ndim})"
            );
            ensure!(
                !is_inserted[d],
                InvalidShapeOperation,
                "repeated dim {d} in insert_dim"
            );
            is_inserted[d] = true;
        }

        let mut shape = DimArray::from_slice(array.shape()).unwrap();
        let orig_spec = array.spec();
        let mut block_shape = <_ as Clone>::clone(orig_spec.block_shape());
        let mut block_shape_fixed_dims = orig_spec.block_shape_fixed_dims();
        // Inserting in ascending order puts each new dim at its final output position.
        for d in (0..new_ndim).filter(|&d| is_inserted[d]) {
            shape.insert(d, 1);
            block_shape.insert(d, 1);
            block_shape_fixed_dims.insert(d, false);
        }
        let shape = DimDyn::from_slice(&shape);
        let original_dims = is_inserted
            .iter()
            .enumerate()
            .filter_map(|(d, inserted)| (!inserted).then_some(d as DimIdx))
            .collect_dim_vec::<DimDyn>(array.shape().len());
        let mut read_shape_scale_weight =
            orig_spec.read_shape_scale_weight().to_dim_vec::<DimDyn>();
        for (d, inserted) in is_inserted.iter().enumerate() {
            if *inserted {
                read_shape_scale_weight.insert(d, ScaleWeight::NONE);
            }
        }
        let mut read_layout_order = orig_spec
            .read_layout_order()
            .iter()
            .map(|&d| original_dims[d as usize])
            .collect::<DimArray<_>>();
        for (d, inserted) in is_inserted.iter().enumerate() {
            if *inserted {
                read_layout_order_insert_dont_care_dim(&mut read_layout_order, d);
            }
        }
        let spec = ArraySpecDynamic::new(
            block_shape,
            block_shape_fixed_dims,
            orig_spec.element_cost(),
            read_shape_scale_weight,
            read_layout_order,
        );

        Ok(Self {
            array,
            original_dims,
            shape,
            spec,
        })
    }

    /// Constructs an array with [`InsertDim`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, dim: impl DimsArg) -> Result<Array<Self>> {
        Self::new(array.into_storage(), dim).map(Array::from_storage)
    }

    #[inline(always)]
    fn transform_index(&self, index: &[Range<u64>]) -> Result<Option<DimArray<Range<u64>>>> {
        check_get_range(self.shape(), index)?;

        for index in index.iter() {
            if index.start == index.end {
                return Ok(None);
            }
        }
        Ok(Some(dim_arr(self.original_dims.as_ref().len(), |dim| {
            index[self.original_dims[dim] as usize].clone()
        })))
    }
}

impl<S> ArrayStorage for InsertDim<S>
where
    S: ArrayStorage,
{
    type ElementType = S::ElementType;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(
            out.as_deref(),
            self.shape(),
            index,
            self.dtype().itemsize() as usize,
        )?;
        let dtype = self.dtype();
        let Some(inner_index) = self.transform_index(index)? else {
            // Empty read
            let out_shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
            return Ok(materialize_out_buf(
                out,
                context,
                out_shape.as_ref(),
                dtype,
                self.spec().read_layout_order(),
            ));
        };
        unsafe {
            read_data_and_map_strides(
                &self.array,
                inner_index.as_ref(),
                context,
                out,
                |inner_strides| {
                    let mut s = dim_arr(index.len(), |_| 0usize);
                    for (d, &o) in self.original_dims.as_ref().iter().enumerate() {
                        s[o as usize] = inner_strides[d];
                    }
                    s
                },
                |out_strides| {
                    dim_arr(self.original_dims.as_ref().len(), |d| {
                        out_strides[self.original_dims[d] as usize]
                    })
                },
            )
        }
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
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
            .map_flags(|flags| flags.clear_compact())
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("InsertDim", [&self.array])
    }

    type ElementTypeChange<NewET: crate::ElementType> = InsertDim<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(InsertDim {
            array: self.array.element_type_change()?,
            original_dims: self.original_dims,
            shape: self.shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::codec::ReadContext;
    use crate::ops::InsertDim;
    use crate::storage::Compact;
    use crate::util::{arr_params, shape_strategy, ScalarStrategy};
    use crate::{Array, Ty, NDIM_MAX};

    fn make1d(vals: Vec<i32>, block_size: usize) -> Array<Compact<Ty<i32>>> {
        let nd = ndarray::Array::from_shape_vec([vals.len()], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[block_size])).unwrap()
    }

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<Compact<Ty<i32>>> {
        let nd = ndarray::Array::from_shape_vec([rows, cols], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<Compact<Ty<i32>>> {
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
        assert_eq!(make1d(arange(6), 6).insert_dim([0]).shape(), &[1, 6]);
    }

    #[test]
    fn shape_insert_after_last_dim() {
        assert_eq!(make1d(arange(6), 6).insert_dim([1]).shape(), &[6, 1]);
    }

    #[test]
    fn shape_insert_between_dims() {
        assert_eq!(make2d(arange(12), 3, 4).insert_dim([1]).shape(), &[3, 1, 4]);
    }

    #[test]
    fn shape_insert_front_and_back() {
        assert_eq!(make1d(arange(6), 6).insert_dim([0, 2]).shape(), &[1, 6, 1]);
    }

    #[test]
    fn shape_insert_adjacent() {
        assert_eq!(
            make2d(arange(12), 3, 4).insert_dim([0, 1]).shape(),
            &[1, 1, 3, 4]
        );
    }

    #[test]
    fn shape_insert_user_example() {
        let a = make3d(arange(24), 2, 3, 4);
        assert_eq!(
            a.insert_dim([0, 2, 3, 4, 7]).shape(),
            &[1, 2, 1, 1, 1, 3, 4, 1]
        );
    }

    #[test]
    fn shape_insert_unsorted_dims_same_result() {
        // Order of dims values should not matter; only the set matters.
        let a1 = make3d(arange(24), 2, 3, 4).insert_dim([0, 2, 3, 4, 7]);
        let a2 = make3d(arange(24), 2, 3, 4).insert_dim([7, 3, 0, 4, 2]);
        assert_eq!(a1.shape(), a2.shape());
    }

    #[test]
    fn shape_empty_dims_is_identity() {
        assert_eq!(make2d(arange(12), 3, 4).insert_dim([]).shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_insert_before_first() {
        let got = make1d(arange(6), 6).insert_dim([0]).to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([1, 6], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_after_last() {
        let got = make1d(arange(6), 6).insert_dim([1]).to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([6, 1], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_between_dims() {
        let got = make2d(arange(12), 3, 4)
            .insert_dim([1])
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
            .insert_dim([0, 2])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([1, 6, 1], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_insert_user_example() {
        let got = make3d(arange(24), 2, 3, 4)
            .insert_dim([0, 2, 3, 4, 7])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::ArrayD::from_shape_vec(vec![1, 2, 1, 1, 1, 3, 4, 1], arange(24)).unwrap()
        );
    }

    #[test]
    fn full_read_identity_empty_dims() {
        let got = make2d(arange(12), 3, 4)
            .insert_dim([])
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
        let got = make1d(arange(6), 6)
            .insert_dim([0])
            .to_ndarray_sub(&[0..1, 2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[2, 3, 4]]);
    }

    #[test]
    fn sub_read_2d_with_inserted_middle() {
        let got = make2d(arange(12), 3, 4)
            .insert_dim([1])
            .to_ndarray_sub(&[1..3, 0..1, 0..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[[4, 5]], [[8, 9]]]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_dim_out_of_bounds() {
        let a = make1d(arange(4), 4);
        // output ndim is 2, so valid dims are 0..2; dim 2 is out of bounds
        assert!(InsertDim::new_array(a, [2]).is_err());
    }

    #[test]
    fn error_dim_out_of_bounds_multiple() {
        let a = make1d(arange(4), 4);
        // output ndim is 3, so valid dims are 0..3
        assert!(InsertDim::new_array(a, [0, 3]).is_err());
    }

    #[test]
    fn error_dim_repeated() {
        let a = make1d(arange(4), 4);
        assert!(InsertDim::new_array(a, [0, 0]).is_err());
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary shape, arbitrary set of output dims, order-independent
    // -----------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn insert_dim_strategy<T>() -> impl proptest::strategy::Strategy<
        Value = (ndarray::ArrayD<T>, crate::util::TestArray<T>, Vec<usize>),
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
                let out_ndim = shape.len() + n_insert;
                let dims = Just((0..out_ndim).collect::<Vec<_>>())
                    .prop_shuffle()
                    .prop_map(move |mut dims| {
                        dims.truncate(n_insert);
                        dims
                    });
                (Just(shape), dims)
            })
            .prop_flat_map(|(shape, dims)| {
                let array_strat =
                    crate::util::array_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(dims).prop_shuffle())
            })
            .prop_map(|((nd, za), dims)| (nd, za, dims))
    }

    proptest::proptest! {
        #[test]
        fn proptest_insert_dim((nd, za, dims) in insert_dim_strategy::<i32>()) {
            // Oracle: inserting size-1 dims is a pure reshape - flat order is unchanged.
            let mut sorted_dims = dims.clone();
            sorted_dims.sort_unstable();
            let mut expected_shape= nd.shape().to_vec();
            for &d in &sorted_dims {
                expected_shape.insert(d, 1);
            }
            let expected = ndarray::ArrayD::from_shape_vec(
                expected_shape,
                nd.iter().cloned().collect::<Vec<_>>(),
            )
            .unwrap();
            crate::util::assert_array_matches(&za.insert_dim(&dims), &expected);
        }
    }
}
