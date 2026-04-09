use std::io;
use std::ops::Range;

use crate::NDIM_MAX;
use crate::array::{Array, BlocksLayout};
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::storage::{ArrayStorage, Ref};
use crate::util::{DimArray, default_strides, dim_arr};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    #[track_caller]
    pub fn reshape_view(&self, new_shape: &[usize]) -> Array<Reshape<Ref<'_, S>>> {
        let a = Array::new(Ref(&self.storage));
        Array::new(Reshape::new(a, new_shape).unwrap())
    }
}
pub struct Reshape<S> {
    a: Array<S>,

    dtype: Dtype,
    new_shape: DimArray<usize>,
    blocks_layout: BlocksLayout,
}
impl<S> Reshape<S> {
    pub(crate) fn new(a: Array<S>, new_shape: &[usize]) -> io::Result<Self>
    where
        S: ArrayStorage,
    {
        if new_shape.len() > NDIM_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cannot reshape array to have {} dimensions (max {NDIM_MAX})",
                    new_shape.len()
                ),
            ));
        }
        let orig_shape: DimArray<usize> = a.shape().try_into().unwrap();
        let nitems = orig_shape.iter().product::<usize>();
        let new_nitems = new_shape.iter().product::<usize>();
        if nitems != new_nitems {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cannot reshape array of shape {:?} into shape {:?}",
                    orig_shape, new_shape
                ),
            ));
        }

        let dtype = a.dtype();
        Ok(Self {
            dtype: dtype.clone(),
            new_shape: new_shape.try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            a,
        })
    }
}
impl<S> ArrayStorage for Reshape<S>
where
    S: ArrayStorage,
{
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.new_shape
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }

    fn read_data(
        &self,
        index: &[Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()> {
        assert_eq!(index.len(), self.new_shape.len());
        let ndim = self.new_shape.len();
        let itemsize = self.dtype.itemsize() as usize;

        // 0-D: single element, forward directly
        if ndim == 0 {
            return self.a.storage.read_data(&[], buf, context);
        }

        let new_strides = default_strides(&self.new_shape, 1usize);
        let orig_shape = self.a.storage.shape();
        let orig_strides = default_strides(orig_shape, 1usize);

        let lead_dims = ndim - 1;
        let last_dim = ndim - 1;

        // Pre-allocate dim_ranges for the original shape (reused across calls)
        let orig_ndim = orig_shape.len();
        let mut dim_ranges: DimArray<Range<usize>> = dim_arr(orig_ndim, |_| 0..0);

        let mut write_pos = 0usize;

        // Iterate over all combinations of leading N-1 dimension indices
        let mut lead_idx: DimArray<usize> = (0..lead_dims).map(|i| index[i].start).collect();

        loop {
            let flat_start: usize = lead_idx
                .iter()
                .enumerate()
                .map(|(d, &i)| i * new_strides[d])
                .sum::<usize>()
                + index[last_dim].start;
            let flat_len = index[last_dim].len();

            if flat_len > 0 {
                read_flat_range(
                    flat_start,
                    flat_len,
                    orig_shape,
                    &orig_strides,
                    0,
                    &mut dim_ranges,
                    itemsize,
                    &self.a.storage,
                    buf,
                    &mut write_pos,
                    context,
                )?;
            }

            // Advance leading index counter (odometer-style)
            if lead_dims == 0 {
                break;
            }
            let mut carry = true;
            for d in (0..lead_dims).rev() {
                lead_idx[d] += 1;
                if lead_idx[d] < index[d].end {
                    carry = false;
                    break;
                }
                lead_idx[d] = index[d].start;
            }
            if carry {
                break;
            }
        }

        Ok(())
    }
}

/// Recursively decomposes a flat element range `[flat_start, flat_start + flat_len)` (expressed
/// in the original array's flat C-order) into at most 3 rectangular reads per dimension level,
/// writing results sequentially into `buf` starting at `*write_pos`.
fn read_flat_range<S: ArrayStorage>(
    flat_start: usize,
    flat_len: usize,
    orig_shape: &[usize],
    orig_strides: &[usize],
    dim: usize,
    dim_ranges: &mut DimArray<Range<usize>>,
    itemsize: usize,
    storage: &S,
    buf: &mut [u8],
    write_pos: &mut usize,
    context: &ReadContext,
) -> io::Result<()> {
    let ndim = orig_shape.len();

    if dim == ndim - 1 {
        // Base case: last dimension — the flat range maps directly to a contiguous index range
        dim_ranges[dim] = flat_start..flat_start + flat_len;
        let chunk_bytes = flat_len * itemsize;
        storage.read_data(
            &dim_ranges[..ndim],
            &mut buf[*write_pos..*write_pos + chunk_bytes],
            context,
        )?;
        *write_pos += chunk_bytes;
        return Ok(());
    }

    let stride = orig_strides[dim]; // number of elements per unit of `dim`
    let mut pos = flat_start;
    let mut remaining = flat_len;

    // Partial first unit: handle the tail of the unit that `pos` falls into
    let start_offset = pos % stride;
    if start_offset != 0 {
        let first_len = remaining.min(stride - start_offset);
        dim_ranges[dim] = (pos / stride)..(pos / stride) + 1;
        read_flat_range(
            start_offset,
            first_len,
            orig_shape,
            orig_strides,
            dim + 1,
            dim_ranges,
            itemsize,
            storage,
            buf,
            write_pos,
            context,
        )?;
        pos += first_len;
        remaining -= first_len;
        if remaining == 0 {
            return Ok(());
        }
    }

    // Complete units: `pos` is now stride-aligned; batch as many full units as possible
    let complete_units = remaining / stride;
    if complete_units > 0 {
        let start_major = pos / stride;
        dim_ranges[dim] = start_major..start_major + complete_units;
        for d in dim + 1..ndim {
            dim_ranges[d] = 0..orig_shape[d];
        }
        let chunk_bytes = complete_units * stride * itemsize;
        storage.read_data(
            &dim_ranges[..ndim],
            &mut buf[*write_pos..*write_pos + chunk_bytes],
            context,
        )?;
        *write_pos += chunk_bytes;
        pos += complete_units * stride;
        remaining -= complete_units * stride;
        if remaining == 0 {
            return Ok(());
        }
    }

    // Partial last unit: the remaining elements start at the beginning of a new unit
    let start_major = pos / stride;
    dim_ranges[dim] = start_major..start_major + 1;
    read_flat_range(
        0,
        remaining,
        orig_shape,
        orig_strides,
        dim + 1,
        dim_ranges,
        itemsize,
        storage,
        buf,
        write_pos,
        context,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::array::Array;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Create a 1-D Array<Owned> from `vals` with the given block size.
    fn make1d<T: crate::dtype::Dtyped>(
        vals: Vec<T>,
        block_size: usize,
    ) -> Array<crate::storage::Owned> {
        let nd = ArrayD::from_shape_vec(vec![vals.len()], vals).unwrap();
        Array::from_ndarray(&nd, &[block_size]).unwrap()
    }

    /// Create a 2-D Array<Owned>.
    fn make2d<T: crate::dtype::Dtyped>(
        vals: Vec<T>,
        rows: usize,
        cols: usize,
        block_shape: &[usize],
    ) -> Array<crate::storage::Owned> {
        let nd = ArrayD::from_shape_vec(vec![rows, cols], vals).unwrap();
        Array::from_ndarray(&nd, block_shape).unwrap()
    }

    /// Create a 3-D Array<Owned>.
    fn make3d<T: crate::dtype::Dtyped>(
        vals: Vec<T>,
        d0: usize,
        d1: usize,
        d2: usize,
        block_shape: &[usize],
    ) -> Array<crate::storage::Owned> {
        let nd = ArrayD::from_shape_vec(vec![d0, d1, d2], vals).unwrap();
        Array::from_ndarray(&nd, block_shape).unwrap()
    }

    fn u8s(n: usize) -> Vec<u8> {
        (0..n).map(|i| i as u8).collect()
    }
    fn i32s(n: usize) -> Vec<i32> {
        (0..n).map(|i| i as i32).collect()
    }
    fn f32s(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }
    fn f64s(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_after_reshape_1d_to_2d() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        assert_eq!(r.shape(), &[3, 4]);
    }

    #[test]
    fn shape_after_reshape_2d_to_1d() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        assert_eq!(r.shape(), &[12]);
    }

    #[test]
    fn shape_after_reshape_2d_to_3d() {
        let a = make2d(u8s(24), 4, 6, &[4, 6]);
        let r = a.reshape_view(&[2, 3, 4]);
        assert_eq!(r.shape(), &[2, 3, 4]);
    }

    #[test]
    fn dtype_preserved_after_reshape() {
        use crate::dtype::Dtyped;
        let a = make1d(i32s(6), 6);
        let r = a.reshape_view(&[2, 3]);
        assert_eq!(r.dtype(), &i32::dtype());
    }

    #[test]
    fn reshape_wrong_size_errors() {
        use crate::storage::Ref;
        let a = make1d(u8s(12), 12);
        let a_ref = Array::new(Ref(&a.storage));
        assert!(super::Reshape::new(a_ref, &[3, 5]).is_err());
    }

    // -----------------------------------------------------------------------
    // Full read: 1-D → 2-D
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_to_2d_3x4_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], u8s(12)).unwrap());
    }

    #[test]
    fn full_read_1d_to_2d_4x3_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[4, 3]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4, 3], u8s(12)).unwrap());
    }

    #[test]
    fn full_read_1d_to_2d_2x6_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 6], u8s(12)).unwrap());
    }

    #[test]
    fn full_read_1d_to_2d_6x2_u8() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[6, 2]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![6, 2], u8s(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: 2-D → 1-D (flatten)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_2d_to_1d_flatten() {
        let a = make2d(i32s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        let got: ArrayD<i32> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![12], i32s(12)).unwrap());
    }

    #[test]
    fn full_read_2d_to_1d_flatten_non_square() {
        let a = make2d(u8s(20), 4, 5, &[4, 5]);
        let r = a.reshape_view(&[20]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![20], u8s(20)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: 2-D → 2-D (re-partition)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_2d_to_2d_repartition() {
        // [3, 4] → [2, 6]: rows of 3 in orig map to interleaved rows of 2 in new
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 6], u8s(12)).unwrap());
    }

    #[test]
    fn full_read_2d_to_2d_repartition_asymmetric() {
        // [4, 3] → [3, 4]
        let a = make2d(u8s(12), 4, 3, &[4, 3]);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], u8s(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: higher dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_1d_to_3d() {
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 3, 4], u8s(24)).unwrap());
    }

    #[test]
    fn full_read_3d_to_1d_flatten() {
        let a = make3d(i32s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape_view(&[24]);
        let got: ArrayD<i32> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![24], i32s(24)).unwrap());
    }

    #[test]
    fn full_read_3d_to_2d() {
        // [2, 3, 4] → [6, 4]
        let a = make3d(u8s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape_view(&[6, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![6, 4], u8s(24)).unwrap());
    }

    #[test]
    fn full_read_2d_to_3d() {
        // [6, 4] → [2, 3, 4]
        let a = make2d(u8s(24), 6, 4, &[6, 4]);
        let r = a.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 3, 4], u8s(24)).unwrap());
    }

    #[test]
    fn full_read_3d_repartition() {
        // [2, 3, 4] → [2, 12]
        let a = make3d(u8s(24), 2, 3, 4, &[2, 3, 4]);
        let r = a.reshape_view(&[2, 12]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 12], u8s(24)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: identity reshape (same shape)
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_same_shape_1d() {
        let a = make1d(i32s(8), 8);
        let r = a.reshape_view(&[8]);
        let got: ArrayD<i32> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![8], i32s(8)).unwrap());
    }

    #[test]
    fn full_read_same_shape_2d() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], u8s(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: single-element array
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_single_element_1d_to_1d() {
        let a = make1d(vec![42u8], 1);
        let r = a.reshape_view(&[1]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![42u8]).unwrap());
    }

    // -----------------------------------------------------------------------
    // Full read: various dtypes
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_dtype_i32() {
        let a = make1d(i32s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<i32> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], i32s(12)).unwrap());
    }

    #[test]
    fn full_read_dtype_f32() {
        let a = make1d(f32s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<f32> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], f32s(12)).unwrap());
    }

    #[test]
    fn full_read_dtype_f64() {
        let a = make1d(f64s(12), 12);
        let r = a.reshape_view(&[4, 3]);
        let got: ArrayD<f64> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4, 3], f64s(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Subregion reads: 1-D → 2-D [3, 4], values 0..12
    // -----------------------------------------------------------------------
    //
    // Layout after reshape to [3, 4]:
    //   row 0: [0,  1,  2,  3]
    //   row 1: [4,  5,  6,  7]
    //   row 2: [8,  9, 10, 11]

    #[test]
    fn sub_read_first_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..1, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 4], vec![0, 1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn sub_read_middle_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[1..2, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 4], vec![4, 5, 6, 7]).unwrap()
        );
    }

    #[test]
    fn sub_read_last_row() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[2..3, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 4], vec![8, 9, 10, 11]).unwrap()
        );
    }

    #[test]
    fn sub_read_first_two_rows() {
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..2, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 4], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap()
        );
    }

    #[test]
    fn sub_read_first_two_columns() {
        // [0..3, 0..2] → rows 0-2, cols 0-1
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..3, 0..2]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 2], vec![0, 1, 4, 5, 8, 9]).unwrap()
        );
    }

    #[test]
    fn sub_read_last_two_columns() {
        // [0..3, 2..4] → rows 0-2, cols 2-3
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..3, 2..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 2], vec![2, 3, 6, 7, 10, 11]).unwrap()
        );
    }

    #[test]
    fn sub_read_inner_2x2() {
        // [1..3, 1..3]
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[1..3, 1..3]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![5, 6, 9, 10]).unwrap()
        );
    }

    #[test]
    fn sub_read_single_element_center() {
        // [1..2, 2..3] → element at (1,2) = 6
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[1..2, 2..3]).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1, 1], vec![6]).unwrap());
    }

    #[test]
    fn sub_read_single_element_corner() {
        // [2..3, 3..4] → element at (2,3) = 11
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[2..3, 3..4]).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1, 1], vec![11]).unwrap());
    }

    // -----------------------------------------------------------------------
    // Subregion reads: 2-D → 1-D
    // reshape [3, 4] → [12], sub-read [3..9]
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_flatten_middle_range() {
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[3..9]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], vec![3, 4, 5, 6, 7, 8]).unwrap()
        );
    }

    #[test]
    fn sub_read_flatten_partial_first_row() {
        // Flat [2..6) spans the last 2 of row-0 and first 2 of row-1 (in orig [3,4])
        let a = make2d(u8s(12), 3, 4, &[3, 4]);
        let r = a.reshape_view(&[12]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[2..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![2, 3, 4, 5]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Subregion reads: [2, 6] ← reshape of 1-D [12]
    //
    // Layout:
    //   row 0: [0,  1,  2,  3,  4,  5]
    //   row 1: [6,  7,  8,  9, 10, 11]
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_2x6_row0_partial() {
        // [0..1, 1..4] → [1, 3] = [1, 2, 3]
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..1, 1..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 3], vec![1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn sub_read_2x6_both_rows_partial_cols() {
        // [0..2, 2..5] → rows 0-1, cols 2-4
        // row 0: [2, 3, 4]; row 1: [8, 9, 10]
        let a = make1d(u8s(12), 12);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..2, 2..5]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3], vec![2, 3, 4, 8, 9, 10]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Subregion reads: 3-D [2, 3, 4]
    //
    // Layout (flat indices):
    //   (0,0,*): 0-3    (0,1,*): 4-7    (0,2,*): 8-11
    //   (1,0,*): 12-15  (1,1,*): 16-19  (1,2,*): 20-23
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_3d_single_row() {
        // [0..1, 1..2, 0..4] → (0,1,*) = [4, 5, 6, 7]
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..1, 1..2, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 1, 4], vec![4, 5, 6, 7]).unwrap()
        );
    }

    #[test]
    fn sub_read_3d_inner_block() {
        // [0..2, 1..3, 1..3]:
        //   (0,1,1)=5  (0,1,2)=6
        //   (0,2,1)=9  (0,2,2)=10
        //   (1,1,1)=17 (1,1,2)=18
        //   (1,2,1)=21 (1,2,2)=22
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..2, 1..3, 1..3]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2, 2], vec![5, 6, 9, 10, 17, 18, 21, 22]).unwrap()
        );
    }

    #[test]
    fn sub_read_3d_second_slab() {
        // [1..2, 0..3, 0..4] → all of the second "slab" = [12..24]
        let a = make1d(u8s(24), 24);
        let r = a.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[1..2, 0..3, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 3, 4], (12u8..24).collect()).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Multi-block: block boundaries in the original array
    // -----------------------------------------------------------------------

    #[test]
    fn multiblock_1d_full_read_reshape_to_2d() {
        // 12 elements, block_size=4 → 3 blocks; reshape to [3, 4]
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], u8s(12)).unwrap());
    }

    #[test]
    fn multiblock_1d_sub_read_crosses_block_boundary() {
        // block_size=4, reshape to [3, 4]; read row 0 of new shape
        // flat [0..4) = one full original block → [0, 1, 2, 3]
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..1, 0..4]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 4], vec![0, 1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn multiblock_1d_to_2x6_sub_read_crosses_block_boundary() {
        // block_size=4, reshape to [2, 6]:
        //   row 0: [0..6) spans block0 (0-3) and part of block1 (4-5)
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[0..1, 0..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 6], vec![0, 1, 2, 3, 4, 5]).unwrap()
        );
    }

    #[test]
    fn multiblock_1d_to_2x6_full_read() {
        let a = make1d(u8s(12), 4);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 6], u8s(12)).unwrap());
    }

    #[test]
    fn multiblock_2d_orig_reshape_to_1d() {
        // orig [3, 4] with block_shape [2, 2], flatten to [12]
        let a = make2d(u8s(12), 3, 4, &[2, 2]);
        let r = a.reshape_view(&[12]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![12], u8s(12)).unwrap());
    }

    #[test]
    fn multiblock_2d_orig_reshape_sub_read() {
        // orig [3, 4] with block_shape [2, 2], reshape to [2, 6]
        // sub-read row 1: flat [6..12) → [6, 7, 8, 9, 10, 11]
        let a = make2d(u8s(12), 3, 4, &[2, 2]);
        let r = a.reshape_view(&[2, 6]);
        let got: ArrayD<u8> = r.data().to_ndarray_sub(&[1..2, 0..6]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 6], vec![6, 7, 8, 9, 10, 11]).unwrap()
        );
    }

    #[test]
    fn multiblock_small_blocks_reshape_3d() {
        // 24 elements, block_size=3 → 8 blocks; reshape to [2, 3, 4]
        let a = make1d(u8s(24), 3);
        let r = a.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 3, 4], u8s(24)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Chained: reshape → reshape
    // -----------------------------------------------------------------------

    #[test]
    fn chained_reshape_1d_to_2d_to_3d() {
        let a = make1d(u8s(24), 24);
        let r1 = a.reshape_view(&[4, 6]);
        let r2 = r1.reshape_view(&[2, 3, 4]);
        let got: ArrayD<u8> = r2.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 3, 4], u8s(24)).unwrap());
    }

    #[test]
    fn chained_reshape_then_flatten() {
        let a = make1d(i32s(12), 12);
        let r1 = a.reshape_view(&[3, 4]);
        let r2 = r1.reshape_view(&[12]);
        let got: ArrayD<i32> = r2.data().to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![12], i32s(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Verify flat element order is preserved
    // (reshape cannot change the value at flat index i)
    // -----------------------------------------------------------------------

    #[test]
    fn flat_order_preserved_4x3_vs_3x4() {
        // Both reshape [12] → [4,3] and [3,4] must yield same flat sequence
        let a12 = make1d(u8s(12), 12);
        let r43 = a12.reshape_view(&[4, 3]);
        let r34 = a12.reshape_view(&[3, 4]);

        let flat_43: ArrayD<u8> = r43.reshape_view(&[12]).data().to_ndarray().unwrap();
        let flat_34: ArrayD<u8> = r34.reshape_view(&[12]).data().to_ndarray().unwrap();
        assert_eq!(flat_43, flat_34);
        assert_eq!(flat_43, ArrayD::from_shape_vec(vec![12], u8s(12)).unwrap());
    }
}
