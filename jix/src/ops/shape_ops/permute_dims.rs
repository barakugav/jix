use crate::ops::prelude::*;

/// Reorders the dims of an array, returned by [`Array::permute_dims`](crate::Array::permute_dims).
///
/// The `i`-th output dim corresponds to dim `dims[i]` of the input - identical to the
/// convention used by NumPy's `numpy.transpose`. No data is copied at construction time;
/// elements are rearranged on demand when the result is read.
///
/// `dims` must be a permutation of `0..ndim`: correct length, all values in range, no
/// duplicates.
///
/// Output dtype equals the input dtype. `PermuteDims<S>` carries `type Dimension = S::Dimension`
/// - permutation does not change the number of dims so the dimension type is preserved unchanged.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // 2-D transpose: [2, 3] -> [3, 2]
/// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]])?;
/// let t = a.permute_dims(&[1, 0]);
/// assert_eq!(t.shape(), &[3, 2]);
/// let result = t.to_ndarray()?;
/// assert_eq!(result[[0, 0]], 1);
/// assert_eq!(result[[0, 1]], 4);
/// assert_eq!(result[[2, 1]], 6);
///
/// // 3-D cyclic permutation [2, 3, 4] -> [4, 2, 3]
/// let b = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
/// let zb = Array::compact_ndarray(&b)?;
/// let p = zb.permute_dims(&[2, 0, 1]);
/// assert_eq!(p.shape(), &[4, 2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct PermuteDims<S: ArrayStorage> {
    array: S,
    /// `dims[i]` = index of the input dimension that maps to output dimension `i`.
    dims: <S::Dimension as Dimension>::Vec<DimIdx>,
    /// `inv_dims[d]` = index of the output dimension that maps from input dimension `d`.
    inv_dims: <S::Dimension as Dimension>::Vec<DimIdx>,

    shape: S::Dimension,
    spec: ArraySpecDynamic,
}

impl<S: ArrayStorage> PermuteDims<S> {
    /// Constructs a [`PermuteDims`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, dims: &[usize]) -> Result<Self> {
        let ndim = array.shape().len();
        ensure!(
            dims.len() == ndim,
            InvalidShapeOperation,
            "dims length {} does not match array ndim {ndim}",
            dims.len()
        );
        let mut seen = S::Dimension::vec(ndim, |_| false);
        for &d in dims.as_ref().iter() {
            ensure!(
                d < ndim,
                InvalidShapeOperation,
                "dim {d} out of bounds for array of ndim {ndim}"
            );
            ensure!(
                !seen[d],
                InvalidShapeOperation,
                "duplicate dim {d} in dims {dims:?}"
            );
            seen[d] = true;
        }
        let mut inv_dims = S::Dimension::vec(ndim, |_| DimIdx::default());
        for (i, &d) in dims.as_ref().iter().enumerate() {
            inv_dims[d] = i as DimIdx;
        }

        let input_shape = array.shape();
        let shape = S::Dimension::from_fn(ndim, |i| input_shape[dims[i]]);

        let inner_spec = array.spec();
        let block_shape = dim_arr(ndim, |i| inner_spec.block_shape()[dims[i]]);
        let inner_block_shape_fixed_dims = inner_spec.block_shape_fixed_dims();
        let spec = ArraySpecDynamic::new(
            block_shape,
            (0..ndim)
                .map(|i| inner_block_shape_fixed_dims.get(dims[i]))
                .collect(),
            inner_spec.element_cost(),
            dim_arr(ndim, |d| inner_spec.read_shape_scale_weight()[dims[d]]),
            inner_spec
                .read_layout_order()
                .iter()
                .map(|&old| inv_dims[old as usize])
                .collect(),
        );
        let dims = S::Dimension::vec(ndim, |i| dims[i] as DimIdx);
        Ok(Self {
            shape,
            spec,
            array,
            dims,
            inv_dims,
        })
    }

    /// Constructs an array with [`PermuteDims`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, dims: &[usize]) -> Result<Array<Self>> {
        Self::new(array.into_storage(), dims).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for PermuteDims<S> {
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
        let ndim = self.dims.as_ref().len();
        unsafe {
            read_data_and_map_strides(
                &self.array,
                S::Dimension::vec(ndim, |d| index[self.inv_dims[d] as usize].clone()).as_ref(),
                context,
                out,
                |inner_strides| dim_arr(ndim, |i| inner_strides[self.dims[i] as usize]),
                |out_strides| dim_arr(ndim, |d| out_strides[self.inv_dims[d] as usize]),
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
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("PermuteDims", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = PermuteDims<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let ndim = self.shape().len();
        check_ndim::<NewD>(ndim)?;
        let shape = NewD::from_slice(self.shape());
        let dims = NewD::vec(ndim, |i| self.dims[i]);
        let inv_dims = NewD::vec(ndim, |i| self.inv_dims[i]);
        let array = self.array.dimension_change::<NewD>()?;
        Ok(PermuteDims {
            shape,
            array,
            dims,
            inv_dims,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = PermuteDims<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(PermuteDims {
            array: self.array.element_type_change()?,
            dims: self.dims,
            inv_dims: self.inv_dims,
            shape: self.shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::util::{shape_strategy, ScalarStrategy};
    use crate::Array;

    #[test]
    fn test_i32_2d_transpose() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_dims(&[1, 0]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([1, 0])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_f32_2d_transpose() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_dims(&[1, 0]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([1, 0])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_3d_dims_2_0_1() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_dims(&[2, 0, 1]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_3d_dims_0_2_1() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_dims(&[0, 2, 1]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_3d_identity() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_dims(&[0, 1, 2]).to_ndarray().unwrap();
        assert_eq!(actual, a);
    }

    #[test]
    #[should_panic]
    fn test_wrong_dims_length_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_dims(&[0, 1, 2]);
    }

    #[test]
    #[should_panic]
    fn test_dim_out_of_bounds_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_dims(&[0, 5]);
    }

    #[test]
    #[should_panic]
    fn test_duplicate_dim_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_dims(&[0, 0]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.transpose().to_ndarray().unwrap();
        let expected = a.t().as_standard_layout().into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_transpose_3d_reverses_all_dims() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.transpose().to_ndarray().unwrap();
        let expected = a.t().as_standard_layout().into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_transpose_1d_is_identity() {
        let a = array![1i32, 2, 3];
        let za = Array::compact_ndarray(&a).unwrap();
        assert_eq!(za.transpose().to_ndarray().unwrap(), a);
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary ndim, arbitrary permutation, verified against ndarray
    // -----------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn permute_dims_strategy<T>(
    ) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::util::TestArray<T>, Vec<usize>)>
    where
        T: ScalarStrategy,
    {
        shape_strategy()
            .prop_flat_map(|shape| {
                let ndim = shape.len();
                let perm = Just((0..ndim).collect::<Vec<_>>()).prop_shuffle();
                (Just(shape), perm)
            })
            .prop_flat_map(|(shape, perm)| {
                let array_strat =
                    crate::util::array_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(perm))
            })
            .prop_map(|((nd, za), perm)| (nd, za, perm))
    }

    proptest::proptest! {
        #[test]
        fn proptest_permute_dims((nd, za, perm) in permute_dims_strategy::<i32>()) {
            let expected = nd
                .view()
                .permuted_axes(perm.clone())
                .as_standard_layout()
                .into_owned();
            crate::util::assert_array_matches(&za.permute_dims(&perm), &expected);
        }
    }
}
