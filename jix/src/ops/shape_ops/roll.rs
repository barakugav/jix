use crate::ops::prelude::*;

/// Rolls elements along a dim, wrapping around at the boundary, returned by
/// [`Array::roll`](crate::Array::roll).
///
/// `output[..., i, ...] = input[..., (i - shift) mod L, ...]` on the rolled dim, where
/// `L = shape[dim]`. A positive `shift` moves elements toward larger indices (elements
/// that fall off the end re-enter at the beginning); a negative `shift` moves them toward
/// smaller indices. `shift` is reduced modulo `L`, so any signed integer is accepted.
///
/// See also [`Flip`](crate::ops::Flip), which reverses element order along a dim
/// without wrapping.
///
/// Output shape and dtype equal the input. The result is a lazy view; no computation
/// occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // 1-D positive shift wraps tail to head.
/// let a = Array::compact_ndarray(&array![0i32, 1, 2, 3, 4])?;
/// assert_eq!(a.view().roll(2, 0).to_ndarray()?, array![3, 4, 0, 1, 2]);
/// // Negative shift wraps head to tail.
/// assert_eq!(a.view().roll(-1, 0).to_ndarray()?, array![1, 2, 3, 4, 0]);
/// // Shift mod L; shift == L is identity.
/// assert_eq!(a.roll(5, 0).to_ndarray()?, array![0, 1, 2, 3, 4]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Roll<S: ArrayStorage> {
    array: S,
    dim: DimIdx,
    /// `shift` normalized to `[0, shape[dim])`. Zero means the op is a pass-through.
    shift: u64,
    cache_scope: CacheScope,
    spec: ArraySpecDynamic,
}

impl<S: ArrayStorage> Roll<S> {
    /// Constructs a [`Roll`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, shift: i64, dim: usize) -> Result<Self> {
        let input_shape = array.shape();
        let ndim = input_shape.len();
        ensure!(
            dim < ndim,
            InvalidShapeOperation,
            "roll dim {dim} is out of bounds for array with ndim {ndim}"
        );

        let len = input_shape[dim];
        let shift = if len == 0 {
            0
        } else {
            shift.rem_euclid(len as i64) as u64
        };

        let mut spec = array.spec().dynamic().clone();
        let cache_scope = spec.with_cache_scope();
        Ok(Self {
            array,
            dim: dim as DimIdx,
            shift,
            cache_scope,
            spec,
        })
    }

    /// Constructs an array with [`Roll`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, shift: i64, dim: usize) -> Result<Array<Self>> {
        Self::new(array.into_storage(), shift, dim).map(Array::from_storage)
    }

    fn is_identity(&self) -> bool {
        self.shift == 0
    }
}

impl<S: ArrayStorage> Roll<S> {
    /// The inner start of the region `index` if it does not wrap around: the rolled output range
    /// on dim k then maps to a single contiguous input range, of the same shape.
    #[inline(always)]
    fn unwrapped_start(&self, index: &Index<'_>) -> Option<DimArray<u64>> {
        let (k, shift) = (self.dim as usize, self.shift);
        let (s, e) = (index.start()[k], index.end(k));
        if s < shift && e > shift {
            return None;
        }
        let j_start = if s >= shift {
            s - shift
        } else {
            s + self.shape()[k] - shift
        };
        Some(dim_arr(index.ndim(), |d| {
            if d == k {
                j_start
            } else {
                index.start()[d]
            }
        }))
    }

    /// Reads a region that wraps around into `out`: splits the output along dim k into two
    /// regions and gathers each into the destination.
    ///   Region 1 (output dim-k [0, len1)):   input dim-k [s + L - shift, L), length len1.
    ///   Region 2 (output dim-k [len1, end)): input dim-k [0, e - shift),     length len2.
    fn read_wrapped_into(
        &self,
        index: &Index<'_>,
        context: &ReadContext,
        out: &mut StridedBuf<'_>,
    ) -> Result<()> {
        let (k, shift) = (self.dim as usize, self.shift);
        let l = self.shape()[k];
        let (s, e) = (index.start()[k], index.end(k));
        let len1 = shift - s;
        let len2 = e - shift;
        debug_assert!(!index.size().contains(&0));
        let (out_buf, out_strides) = out.data_mut();
        let mut read_region = |start_k: u64, size_k: u64, dst_dim_k_offset: u64| -> Result<()> {
            let ndim = index.ndim();
            let inner_start = dim_arr(ndim, |d| if d == k { start_k } else { index.start()[d] });
            let inner_size = dim_arr(ndim, |d| {
                if d == k {
                    size_k as usize
                } else {
                    index.size()[d]
                }
            });
            let dst_byte_offset = dst_dim_k_offset as usize * out_strides[k];
            let mut sub = unsafe {
                StridedBuf::from_slice_mut(
                    &mut out_buf[dst_byte_offset..],
                    Clone::clone(out_strides),
                )
            };
            let inner_index = unsafe { Index::new_unchecked(&inner_start, &inner_size) };
            self.array.read_data_into(&inner_index, context, &mut sub)
        };
        read_region(s + l - shift, len1, 0)?;
        read_region(0, len2, len1)
    }
}

impl<S: ArrayStorage> crate::storage::ArrayStorageImpl for Roll<S> {}
impl<S: ArrayStorage> ArrayStorage for Roll<S> {
    type ElementType = S::ElementType;

    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
    ) -> Result<StridedBuf<'a>> {
        if self.is_identity() {
            return self.array.read_data(index, context);
        }
        check_read(self.shape(), index, self.dtype().itemsize() as usize)?;
        let _cache_scope = self.cache_scope.enter(context);
        if let Some(inner_start) = self.unwrapped_start(index) {
            let inner_index = unsafe { Index::new_unchecked(&inner_start, index.size()) };
            return self.array.read_data(&inner_index, context);
        }
        read_via_push(
            index,
            context,
            self.dtype(),
            self.spec().read_layout_order(),
            |out| self.read_wrapped_into(index, context, out),
        )
    }

    #[inline]
    fn read_data_into(
        &self,
        index: &Index<'_>,
        context: &ReadContext,
        out: &mut StridedBuf<'_>,
    ) -> Result<()> {
        if self.is_identity() {
            return self.array.read_data_into(index, context, out);
        }
        check_read_into(out, self.shape(), index, self.dtype().itemsize() as usize)?;
        let _cache_scope = self.cache_scope.enter(context);
        if let Some(inner_start) = self.unwrapped_start(index) {
            let inner_index = unsafe { Index::new_unchecked(&inner_start, index.size()) };
            return self.array.read_data_into(&inner_index, context, out);
        }
        self.read_wrapped_into(index, context, out)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }

    fn spec(&self) -> ArraySpec<'_> {
        if self.is_identity() {
            return self.array.spec();
        }
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Roll", [&self.array])
    }
    fn as_compact(&self) -> Option<crate::storage::CompactBorrowed<'_, TypeDyn>> {
        if self.is_identity() {
            self.array.as_compact()
        } else {
            None
        }
    }

    type ElementTypeChange<NewET: crate::ElementType> = Roll<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(Roll {
            array: self.array.element_type_change()?,
            dim: self.dim,
            shift: self.shift,
            cache_scope: self.cache_scope,
            spec: self.spec,
        })
    }
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Returns a lazy view of the array with elements rolled along the given dim.
    /// See [`Roll`] for details and examples.
    ///
    /// `shift` is reduced modulo `shape[dim]`. Positive shifts move elements toward
    /// larger indices (wrapping around at the end); negative shifts move them the other
    /// way.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= self.ndim()`.
    #[track_caller]
    pub fn roll(self, shift: i64, dim: usize) -> Array<Roll<S>> {
        Roll::new_array(self, shift, dim).unwrap()
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::{array, Ix2};

    use crate::ReadContext;

    use crate::{Array, ArrayStorage, CompactArray, IntoDimension};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make(vals: Vec<i32>, shape: impl IntoDimension) -> CompactArray<i32> {
        let shape = shape.into_dimension().unwrap();
        let nd = ndarray::Array::from_shape_vec(shape, vals).unwrap();
        Array::compact_ndarray(&nd).unwrap()
    }

    fn arange(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata (roll preserves shape)
    // -----------------------------------------------------------------------

    #[test]
    fn shape_preserved_positive_shift() {
        assert_eq!(make(arange(12), [3u64, 4]).roll(1, 0).shape(), &[3, 4]);
    }

    #[test]
    fn shape_preserved_negative_shift() {
        assert_eq!(make(arange(12), [3u64, 4]).roll(-2, 1).shape(), &[3, 4]);
    }

    #[test]
    fn shape_preserved_large_shift() {
        // shift larger than the dim length is reduced mod L.
        assert_eq!(make(arange(12), [3u64, 4]).roll(100, 0).shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_dim_out_of_bounds() {
        let a = make(arange(12), [3u64, 4]);
        let err = super::Roll::new_array(a.view(), 1, 2).unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidShapeOperation);
    }

    // -----------------------------------------------------------------------
    // Fast paths: identity (shift mod L == 0) and empty
    // -----------------------------------------------------------------------

    #[test]
    fn identity_zero_shift_full_read() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), [3u64, 4]).roll(0, 0).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_shift_equals_dim_len() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), [3u64, 4]).roll(3, 0).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_shift_negative_multiple_of_len() {
        let nd = ndarray::Array::from_shape_vec((3, 4), arange(12)).unwrap();
        let got = make(arange(12), [3u64, 4])
            .roll(-6, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_keeps_compact() {
        let a = make(arange(12), [3u64, 4]).roll(3, 0);
        assert!(a.is_compact());
        assert!(a.storage().as_compact().is_some());
        assert!(!make(arange(12), [3u64, 4]).roll(1, 0).is_compact());
    }

    #[test]
    fn identity_size_one_dim() {
        // Rolling a size-1 dim is a no-op for any shift.
        let nd = ndarray::Array::from_shape_vec((1, 4), arange(4)).unwrap();
        let got = make(arange(4), [1u64, 4]).roll(7, 0).to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn identity_empty_array() {
        // Rolling a dim of size 0 is a no-op (no data to move).
        let got = make(vec![], [0u64, 4])
            .roll(3, 0)
            .to_ndarray::<Ix2>()
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_positive_shift() {
        let got = make(arange(5), [5u64]).roll(2, 0).to_ndarray().unwrap();
        assert_eq!(got, array![3, 4, 0, 1, 2]);
    }

    #[test]
    fn full_read_1d_negative_shift() {
        let got = make(arange(5), [5u64]).roll(-1, 0).to_ndarray().unwrap();
        assert_eq!(got, array![1, 2, 3, 4, 0]);
    }

    #[test]
    fn full_read_2d_dim0() {
        let got = make(arange(8), [2u64, 4]).roll(1, 0).to_ndarray().unwrap();
        assert_eq!(got, array![[4, 5, 6, 7], [0, 1, 2, 3]]);
    }

    #[test]
    fn full_read_2d_dim1() {
        let got = make(arange(6), [2u64, 3]).roll(1, 1).to_ndarray().unwrap();
        assert_eq!(got, array![[2, 0, 1], [5, 3, 4]]);
    }

    #[test]
    fn full_read_3d_middle_dim() {
        // Compare against a hand-rolled reference along dim 1 by +1.
        let arr = ndarray::Array::from_shape_vec((2, 3, 2), arange(12)).unwrap();
        let mut expected = arr.clone();
        for i in 0..2 {
            for j in 0..3 {
                for k_dim in 0..2 {
                    let src_j = (j + 3 - 1) % 3;
                    expected[(i, j, k_dim)] = arr[(i, src_j, k_dim)];
                }
            }
        }
        let got = make(arange(12), [2u64, 3, 2])
            .roll(1, 1)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Sub-region reads: non-wrap and wrap variants
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_1d_nonwrap_after_shift() {
        // [0,1,2,3,4] roll +2 -> [3, 4, 0, 1, 2]. Output [2..5) = [0, 1, 2] (no wrap).
        let got = make(arange(5), [5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![0, 1, 2]);
    }

    #[test]
    fn sub_read_1d_nonwrap_before_shift() {
        // [0,1,2,3,4] roll +2 -> [3, 4, 0, 1, 2]. Output [0..2) = [3, 4] (no wrap).
        let got = make(arange(5), [5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[0..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![3, 4]);
    }

    #[test]
    fn sub_read_1d_wrap_across_shift() {
        // [0,1,2,3,4] roll +2 -> [3, 4, 0, 1, 2]. Output [1..4) = [4, 0, 1] (wraps at i=2).
        let got = make(arange(5), [5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[1..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![4, 0, 1]);
    }

    #[test]
    fn sub_read_2d_dim0_wrap() {
        // [[0,1,2],[3,4,5],[6,7,8]] roll +1 dim 0 -> [[6,7,8],[0,1,2],[3,4,5]]
        // Sub rows [0..2), cols [0..3) -> [[6,7,8],[0,1,2]]. Wraps at i=1.
        let got = make(arange(9), [3u64, 3])
            .roll(1, 0)
            .to_ndarray_sub(&[0..2, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[6, 7, 8], [0, 1, 2]]);
    }

    #[test]
    fn sub_read_2d_dim1_wrap() {
        // [[0,1,2,3],[4,5,6,7]] roll +1 dim 1 -> [[3,0,1,2],[7,4,5,6]]
        // Sub rows [0..2), cols [0..3) -> [[3,0,1],[7,4,5]]. Wraps at i=1 on dim 1.
        let got = make(arange(8), [2u64, 4])
            .roll(1, 1)
            .to_ndarray_sub(&[0..2, 0..3], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[3, 0, 1], [7, 4, 5]]);
    }

    #[test]
    fn sub_read_empty_range() {
        let got = make(arange(8), [2u64, 4])
            .roll(1, 0)
            .to_ndarray_sub::<Ix2>(&[1..1, 0..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got.shape(), &[0, 4]);
    }

    #[test]
    fn sub_read_single_element_wrap_point() {
        // [10,20,30,40,50] roll +2 -> [40, 50, 10, 20, 30]. Position 1 == 50.
        let got = make(vec![10, 20, 30, 40, 50], [5u64])
            .roll(2, 0)
            .to_ndarray_sub(&[1..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![50]);
    }

    // -----------------------------------------------------------------------
    // Composition with other ops
    // -----------------------------------------------------------------------

    #[test]
    fn compose_roll_then_compact() {
        let got = make(arange(6), [2u64, 3])
            .roll(1, 0)
            .compact()
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[3, 4, 5], [0, 1, 2]]);
    }

    #[test]
    fn compose_roll_then_slice() {
        let got = make(arange(9), [3u64, 3])
            .roll(1, 0)
            .slice((1..3, ..))
            .to_ndarray()
            .unwrap();
        // Rolled: [[6,7,8],[0,1,2],[3,4,5]]; rows 1..3 = [[0,1,2],[3,4,5]]
        assert_eq!(got, array![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn compose_roll_then_cast() {
        let got = make(arange(3), [3u64])
            .roll(1, 0)
            .cast::<f32>()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![2.0f32, 0.0, 1.0]);
    }

    #[test]
    fn compose_permute_then_roll() {
        // [[0,1,2],[3,4,5]] permute -> [[0,3],[1,4],[2,5]]; roll +1 dim 0 -> [[2,5],[0,3],[1,4]]
        let got = make(arange(6), [2u64, 3])
            .permute_dims(&[1, 0])
            .roll(1, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![[2, 5], [0, 3], [1, 4]]);
    }

    #[test]
    fn compose_roll_then_roll_same_dim_combines() {
        // Two rolls on the same dim are equivalent to one roll with the sum of shifts.
        let nd1 = make(arange(5), [5u64])
            .roll(2, 0)
            .roll(1, 0)
            .to_ndarray()
            .unwrap();
        let nd2 = make(arange(5), [5u64]).roll(3, 0).to_ndarray().unwrap();
        assert_eq!(nd1, nd2);
        assert_eq!(nd1, array![2, 3, 4, 0, 1]);
    }

    #[test]
    fn compose_roll_inverse_is_identity() {
        let nd = ndarray::Array::from_shape_vec((5,), arange(5)).unwrap();
        let got = make(arange(5), [5u64])
            .roll(3, 0)
            .roll(-3, 0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    // -----------------------------------------------------------------------
    // Element type preservation
    // -----------------------------------------------------------------------

    #[test]
    fn element_type_change_into_type_dyn() {
        let a = make(arange(6), [2u64, 3]);
        let r = a.roll(1, 1);
        let dyn_et = r.into_type_dyn();
        assert_eq!(dyn_et.dtype(), &<i32 as crate::dtype::Dtyped>::DTYPE);
    }

    // -----------------------------------------------------------------------
    // Proptest: random shape + dim + shift vs hand-rolled reference
    // -----------------------------------------------------------------------

    fn roll_reference<T: Clone + Default>(
        nd: &ndarray::ArrayD<T>,
        shift: i64,
        dim: usize,
    ) -> ndarray::ArrayD<T> {
        let l = nd.shape()[dim];
        let mut out = ndarray::ArrayD::<T>::from_elem(nd.shape(), T::default());
        if l == 0 {
            return out;
        }
        let l_i128 = l as i128;
        let s_prime = ((shift as i128).rem_euclid(l_i128)) as usize;
        for (i, src_slice) in nd.axis_iter(ndarray::Axis(dim)).enumerate() {
            let dst_i = (i + s_prime) % l;
            let mut dst = out.index_axis_mut(ndarray::Axis(dim), dst_i);
            dst.assign(&src_slice);
        }
        out
    }

    #[allow(clippy::type_complexity)]
    fn roll_strategy() -> impl proptest::strategy::Strategy<
        Value = (
            ndarray::ArrayD<i32>,
            crate::util::TestArray<i32>,
            usize,
            i64,
        ),
    > {
        use proptest::prelude::*;

        let shape = crate::util::shape_strategy().prop_filter("non-empty ndim", |s| !s.is_empty());

        shape.prop_flat_map(|shape| {
            let ndim = shape.len();
            let array_strat = crate::util::array_strategy_from_shape::<i32>(
                Just(shape.clone()),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            );
            let dim = 0..ndim;
            // Mix of small positive, small negative, zero, and large-magnitude shifts.
            let shift = -20i64..=20i64;
            (array_strat, dim, shift).prop_map(|((nd, za), dim, shift)| (nd, za, dim, shift))
        })
    }

    proptest::proptest! {
        #[test]
        fn proptest_roll_generic(
            (nd, za, dim, shift) in roll_strategy()
        ) {
            let expected = roll_reference(&nd, shift, dim);
            let actual = za.roll(shift, dim);
            crate::util::assert_array_matches(&actual, &expected);
        }
    }
}
