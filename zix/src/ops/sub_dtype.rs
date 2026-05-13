use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Itemsize};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Returns a view of one named field of a struct dtype. See [`SubDtype`] for details and
    /// examples.
    ///
    /// # Panics
    ///
    /// Panics if the array dtype is not a struct dtype or has no field with the given name.
    #[track_caller]
    pub fn dtype_sub_field(self, sub_field: &str) -> Array<SubDtype<S>> {
        Array::from_storage(SubDtype::new(self, sub_field).unwrap())
    }
}
/// Extracts one named field from a struct dtype array.
///
/// The input array must have a struct dtype with a field named `sub_field`. The output has the
/// dtype of that field and the same shape as the input. Field bytes are copied out of each
/// element on demand.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, but the operation is also available as
/// [`Array::dtype_sub_field()`](crate::Array::dtype_sub_field).
///
/// # Examples
/// ```rust,ignore
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// #[derive(Copy, Clone, zix::dtype::Dtyped)]
/// #[repr(C)]
/// struct Point { x: i32, y: i32 }
///
/// let pts = array![
///     Point { x: 1, y: 10 },
///     Point { x: 2, y: 20 },
///     Point { x: 3, y: 30 },
/// ];
/// let za = Array::compact_array(&pts)?;
/// let xs = za.dtype_sub_field("x").to_ndarray::<i32>()?;
/// assert_eq!(xs.as_slice().unwrap(), &[1, 2, 3]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct SubDtype<S> {
    array: Array<S>,
    dst_dtype: Dtype,
    sub_field_offset: Itemsize,
}
impl<S> SubDtype<S> {
    /// Constructs a `SubDtype` storage. See [`SubDtype`] for semantics and examples.
    pub fn new(array: Array<S>, sub_field: &str) -> Result<Self>
    where
        S: ArrayStorage,
    {
        let src_dtype = array.dtype();
        ensure!(
            src_dtype.shape().is_empty(),
            UnsupportedDtype,
            "Can only take sub-field of a struct dtype with non-custom shape, but got dtype {src_dtype:?}"
        );
        let sub_field_spec = src_dtype
            .fields()
            .and_then(|fields| fields.iter().find(|f| f.0 == sub_field))
            .map(|(_f_name, offset, sub_dtype)| (*offset, sub_dtype.clone()));
        ensure!(
            sub_field_spec.is_some(),
            UnsupportedDtype,
            "dtype {src_dtype:?} does not have a field named '{sub_field}'"
        );
        let (offset, dtype) = sub_field_spec.unwrap();

        Ok(Self {
            dst_dtype: dtype,
            sub_field_offset: offset,
            array,
        })
    }
}
impl<S> ArrayStorage for SubDtype<S>
where
    S: ArrayStorage,
{
    type Dimension = S::Dimension;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;
        let nitems = check_get_buffer_size(index, &self.dst_dtype, buf)?;
        if nitems == 0 {
            return Ok(());
        }
        let (src_dtype, dst_dtype) = (self.array.dtype(), &self.dst_dtype);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);

        let mut tmp_buf = context.tmp_buf(nitems * src_itemsize, src_dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array.storage.read_data(index, tmp_buf, context)?;

        let src_items = tmp_buf.chunks_exact(src_itemsize);
        let dst_items = buf.chunks_exact_mut(dst_itemsize);
        let sub_field_offset = self.sub_field_offset as usize;
        for (src, dst) in src_items.zip(dst_items) {
            let src_field = &src[sub_field_offset..sub_field_offset + dst_itemsize];
            dst.copy_from_slice(src_field);
        }
        Ok(())
    }

    fn shape(&self) -> &[u64] {
        self.array.shape()
    }
    fn dtype(&self) -> &Dtype {
        &self.dst_dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        self.array.storage._spec()
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;

    #[derive(Copy, Clone, PartialEq, Debug, crate::dtype::Dtyped)]
    #[repr(C)]
    struct Pair {
        x: i32,
        y: i32,
    }

    #[test]
    fn basic_field_extraction() {
        let pts = ndarray::array![
            Pair { x: 1, y: 10 },
            Pair { x: 2, y: 20 },
            Pair { x: 3, y: 30 },
        ];
        let za = Array::compact_array(&pts).unwrap();
        let xs = za
            .as_ref()
            .dtype_sub_field("x")
            .to_ndarray::<i32>()
            .unwrap();
        let ys = za
            .as_ref()
            .dtype_sub_field("y")
            .to_ndarray::<i32>()
            .unwrap();
        assert_eq!(xs.as_slice().unwrap(), &[1, 2, 3]);
        assert_eq!(ys.as_slice().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn error_not_struct_dtype() {
        let a = Array::compact_array(&ndarray::array![1i32, 2, 3]).unwrap();
        assert!(super::SubDtype::new(a, "x").is_err());
    }

    #[test]
    fn error_field_not_found() {
        let pts = ndarray::array![Pair { x: 1, y: 10 }];
        let za = Array::compact_array(&pts).unwrap();
        assert!(super::SubDtype::new(za, "z").is_err());
    }

    proptest::proptest! {
        #[test]
        fn proptest_sub_fields(
            pairs in proptest::collection::vec(
                (proptest::num::i32::ANY, proptest::num::i32::ANY),
                1usize..=100,
            )
        ) {
            let pair_structs: Vec<Pair> = pairs.iter().map(|&(x, y)| Pair { x, y }).collect();
            let n = pair_structs.len();
            let nd = ndarray::ArrayD::from_shape_vec(vec![n], pair_structs).unwrap();
            let za = Array::compact_array(&nd).unwrap();
            let expected_x = ndarray::ArrayD::from_shape_vec(
                vec![n],
                pairs.iter().map(|&(x, _)| x).collect::<Vec<_>>(),
            ).unwrap();
            let expected_y = ndarray::ArrayD::from_shape_vec(
                vec![n],
                pairs.iter().map(|&(_, y)| y).collect::<Vec<_>>(),
            ).unwrap();
            crate::util::assert_array_matches(&za.as_ref().dtype_sub_field("x"), &expected_x);
            crate::util::assert_array_matches(&za.as_ref().dtype_sub_field("y"), &expected_y);
        }
    }
}
