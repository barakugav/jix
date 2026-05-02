use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Itemsize};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec};
use crate::util::DimArray;

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
/// Extracts one named field from a struct dtype array, returned by [`Array::dtype_sub_field`].
///
/// The input array must have a struct dtype with a field named `sub_field`. The output has the
/// dtype of that field and the same shape as the input. Field bytes are copied out of each
/// element on demand.
///
/// The result is a lazy view; no computation occurs until the array is read.
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
    shape: DimArray<u64>,
}
impl<S> SubDtype<S> {
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
            shape: array.shape().try_into().unwrap(),
            array,
        })
    }
}
impl<S> ArrayStorage for SubDtype<S>
where
    S: ArrayStorage,
{
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape, index)?;
        let nitems = check_get_buffer_size(index, &self.dst_dtype, buf)?;
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
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dst_dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        self.array.storage._spec()
    }
}
