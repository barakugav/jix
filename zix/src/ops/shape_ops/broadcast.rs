use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{Result, bail, check_get_buffer_size, check_get_range, ensure};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::{DimArray, default_strides, dim_arr, nd_copy};

/// Lazy storage type returned by [`Array::broadcast_view`](crate::Array::broadcast_view).
///
/// Presents the underlying array expanded to a larger shape by repeating elements along dimensions
/// that had length 1, without copying any data at construction time.
///
/// # Shape rules
///
/// The `new_shape` must have the same number of dimensions as the input array.  For each
/// dimension `d`:
///
/// * If `input_shape[d] == new_shape[d]` — the dimension is kept as-is.
/// * If `input_shape[d] == 1` and `new_shape[d] >= 1` — the dimension is **broadcast**: the
///   single element is repeated `new_shape[d]` times along that axis.
/// * Any other combination is an error (e.g. trying to broadcast a dim of size 3 to size 5).
///
/// # Examples
///
/// **Broadcast a row vector to a matrix (repeat along axis 0):**
/// ```text
/// input shape: [1, N]    new_shape: [M, N]    output shape: [M, N]
/// ```
/// Every row of the output is identical — it contains the same N elements as the input.
///
/// **Broadcast a column vector to a matrix (repeat along axis 1):**
/// ```text
/// input shape: [M, 1]    new_shape: [M, N]    output shape: [M, N]
/// ```
/// Every column of the output is identical — it contains the same M elements as the input.
///
/// **Broadcast both axes:**
/// ```text
/// input shape: [1, 1]    new_shape: [M, N]    output shape: [M, N]
/// ```
/// All M×N output elements are equal to the single input element.
///
/// **No-op (no broadcast dims):**
/// ```text
/// input shape: [M, N]    new_shape: [M, N]    output shape: [M, N]
/// ```
///
/// # Read behaviour
///
/// Reading a sub-region of the output works by collapsing all broadcast dims in the requested
/// index ranges to `0..1`, reading from the inner storage, then replicating the data into the
/// output buffer using zero strides (advancing along a broadcast output axis always re-reads the
/// same inner element).  No data is copied at construction time.
pub struct Broadcast<S> {
    array: Array<S>,
    /// `is_broadcast[d]` is `true` when output dim `d` was expanded from length 1.
    is_broadcast: DimArray<bool>,
    /// `true` when `new_shape == input_shape` — no dimension was actually broadcast.
    /// In this case `read_data` forwards directly to the inner storage with no extra work.
    is_identity: bool,

    dtype: Dtype,
    new_shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> Broadcast<S> {
    pub fn new(array: Array<S>, new_shape: &[u64]) -> Result<Self> {
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

        let new_shape: DimArray<_> = new_shape.try_into().unwrap();

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
        b_layout.preferred_read_block_shape = dim_arr(ndim, |d| {
            if is_broadcast[d] {
                new_shape[d] as u32
            } else {
                b_layout.preferred_read_block_shape[d]
            }
        });

        let dtype = array.dtype().clone();
        Ok(Self {
            array,
            is_broadcast,
            is_identity,
            dtype,
            new_shape,
            blocks_layout: b_layout,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for Broadcast<S> {
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        // Fast path: no dimension was actually broadcast — forward directly.
        if self.is_identity {
            return self.array.storage.read_data(index, buf, context);
        }

        check_get_range(&self.new_shape, index)?;
        check_get_buffer_size(index, &self.dtype, buf)?;

        let ndim = self.is_broadcast.len();
        let itemsize = self.dtype.itemsize() as usize;

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
        let mut tmp_buf = context.tmp_buf(n_bytes, self.dtype.alignment());
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
        &self.new_shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage.spec()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::array::{Array, ArrayParams};
    use crate::storage::block::BlockSize;

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    fn make(vals: Vec<i32>, shape: &[usize]) -> Array<crate::storage::Owned> {
        let nd = ndarray::ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
        Array::from_ndarray(&nd, arr_params(shape)).unwrap()
    }

    fn seq(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_broadcast_axis0() {
        // [1, 4] → [3, 4]
        assert_eq!(
            make(seq(4), &[1, 4]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_axis1() {
        // [3, 1] → [3, 4]
        assert_eq!(
            make(seq(3), &[3, 1]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_both_axes() {
        // [1, 1] → [3, 4]
        assert_eq!(
            make(vec![7], &[1, 1]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_no_broadcast_is_identity() {
        assert_eq!(
            make(seq(12), &[3, 4]).broadcast_view(&[3, 4]).shape(),
            &[3, 4]
        );
    }

    #[test]
    fn shape_broadcast_3d_middle() {
        // [2, 1, 4] → [2, 3, 4]
        assert_eq!(
            make(seq(8), &[2, 1, 4]).broadcast_view(&[2, 3, 4]).shape(),
            &[2, 3, 4]
        );
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_broadcast_axis0() {
        // [1, 4] → [3, 4]: each row is [0,1,2,3]
        let got: ArrayD<i32> = make(seq(4), &[1, 4])
            .broadcast_view(&[3, 4])
            .data()
            .to_ndarray()
            .unwrap();
        let expected =
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn full_read_broadcast_axis1() {
        // [3, 1] → [3, 4]: each col is [0,1,2]
        let got: ArrayD<i32> = make(seq(3), &[3, 1])
            .broadcast_view(&[3, 4])
            .data()
            .to_ndarray()
            .unwrap();
        let expected =
            ArrayD::from_shape_vec(vec![3, 4], vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn full_read_broadcast_both() {
        // [1, 1] → [2, 3]: all elements == 7
        let got: ArrayD<i32> = make(vec![7], &[1, 1])
            .broadcast_view(&[2, 3])
            .data()
            .to_ndarray()
            .unwrap();
        let expected = ArrayD::from_shape_vec(vec![2, 3], vec![7; 6]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn full_read_no_broadcast() {
        let got: ArrayD<i32> = make(seq(12), &[3, 4])
            .broadcast_view(&[3, 4])
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], seq(12)).unwrap());
    }

    #[test]
    fn full_read_broadcast_3d_middle() {
        // [2, 1, 3] → [2, 4, 3]: axis 1 repeats 4 times
        let got: ArrayD<i32> = make(seq(6), &[2, 1, 3])
            .broadcast_view(&[2, 4, 3])
            .data()
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
        // [1, 4] → [3, 4]: read rows 1..3, cols 1..3
        let got: ArrayD<i32> = make(seq(4), &[1, 4])
            .broadcast_view(&[3, 4])
            .data()
            .to_ndarray_sub(&[1..3, 1..3])
            .unwrap();
        // each row is [1, 2]
        let expected = ArrayD::from_shape_vec(vec![2, 2], vec![1, 2, 1, 2]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn sub_read_broadcast_axis1() {
        // [3, 1] → [3, 5]: read rows 0..2, cols 2..5 (all same element per row)
        let got: ArrayD<i32> = make(seq(3), &[3, 1])
            .broadcast_view(&[3, 5])
            .data()
            .to_ndarray_sub(&[0..2, 2..5])
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
        let a = make(seq(12), &[3, 4]);
        let b = super::Broadcast::new(a.as_ref(), &[3, 4]).unwrap();
        assert!(b.is_identity);
    }

    #[test]
    fn identity_flag_not_set_when_broadcast() {
        let a = make(seq(4), &[1, 4]);
        let b = super::Broadcast::new(a.as_ref(), &[3, 4]).unwrap();
        assert!(!b.is_identity);
    }

    #[test]
    fn identity_full_read_correct() {
        let got: ArrayD<i32> = make(seq(12), &[3, 4])
            .broadcast_view(&[3, 4])
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], seq(12)).unwrap());
    }

    #[test]
    fn identity_sub_read_correct() {
        let got: ArrayD<i32> = make(seq(12), &[3, 4])
            .broadcast_view(&[3, 4])
            .data()
            .to_ndarray_sub(&[1..3, 1..3])
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
        let a = make(seq(6), &[2, 3]);
        assert!(super::Broadcast::new(a.as_ref(), &[2, 3, 1]).is_err());
    }

    #[test]
    fn error_non_unit_dim_broadcast() {
        let a = make(seq(6), &[2, 3]);
        // axis 0 has length 2, cannot broadcast to 5
        assert!(super::Broadcast::new(a.as_ref(), &[5, 3]).is_err());
    }
}
