use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped, Itemsize};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::ArrayStorageSpec;
use crate::{Array, ArrayStorage, ElementType, Ty, TypeDyn};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Returns a view of one named field of a struct dtype. See [`SubDtype`] for details and
    /// examples.
    ///
    /// The method is generic over the output element type, which must match the field dtype.
    /// If the field dtype is not statically known, use [`dtype_sub_field_dyn()`](Self::dtype_sub_field_dyn)
    /// instead.
    ///
    /// # Panics
    ///
    /// Panics if the array dtype is not a struct dtype or has no field with the given name.
    #[track_caller]
    pub fn dtype_sub_field<T>(self, sub_field: &str) -> Array<SubDtype<S, Ty<T>>>
    where
        T: Dtyped,
    {
        SubDtype::new_array(self, sub_field).unwrap()
    }

    /// Returns a view of one named field of a struct dtype, for dynamically-typed fields. See
    /// [`SubDtype`] for details and examples.
    ///
    /// If the type of the field is statically known, use [`dtype_sub_field()`](Self::dtype_sub_field)
    /// instead to get better ergonomics and performance.
    #[track_caller]
    pub fn dtype_sub_field_dyn(self, sub_field: &str) -> Array<SubDtype<S, TypeDyn>> {
        SubDtype::new_array(self, sub_field).unwrap()
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
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::dtype_sub_field()`](crate::Array::dtype_sub_field).
///
/// # Examples
/// ```rust,ignore
/// use jix::Array;
/// use ndarray::array;
///
/// #[derive(Copy, Clone, jix::dtype::Dtyped)]
/// #[repr(C)]
/// struct Point { x: i32, y: i32 }
///
/// let pts = array![
///     Point { x: 1, y: 10 },
///     Point { x: 2, y: 20 },
///     Point { x: 3, y: 30 },
/// ];
/// let za = Array::compact_ndarray(&pts)?;
/// let xs = za.dtype_sub_field::<i32>("x").to_ndarray()?;
/// assert_eq!(xs.as_slice().unwrap(), &[1, 2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct SubDtype<S, ET> {
    array: S,
    dst_type: ET,
    sub_field_offset: Itemsize,
}
impl<S, ET> SubDtype<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    /// Constructs a [`SubDtype`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, sub_field: &str) -> Result<Self> {
        let src_dtype = array.dtype();
        ensure!(
            src_dtype.shape().is_empty(),
            UnsupportedDtype,
            "Can only take sub-field of a struct dtype with non-custom shape, but got dtype {src_dtype}"
        );
        let sub_field_spec = src_dtype
            .fields()
            .and_then(|fields| fields.iter().find(|f| f.0 == sub_field))
            .map(|(_f_name, offset, sub_dtype)| (*offset, sub_dtype));
        ensure!(
            sub_field_spec.is_some(),
            UnsupportedDtype,
            "dtype {src_dtype} does not have a field named '{sub_field}'"
        );
        let (offset, dtype) = sub_field_spec.unwrap();

        let dst_dtype = ET::from_dtype(dtype.clone())?;
        Ok(Self {
            dst_type: dst_dtype,
            sub_field_offset: offset,
            array,
        })
    }

    /// Constructs an array with [`SubDtype`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, sub_field: &str) -> Result<Array<Self>> {
        Self::new(array.into_storage(), sub_field).map(Array::from_storage)
    }
}
impl<S, ET> ArrayStorage for SubDtype<S, ET>
where
    S: ArrayStorage,
    ET: ElementType,
{
    type ElementType = ET;
    type Dimension = S::Dimension;

    #[inline]
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;
        let dst_dtype = self.dtype();
        let nitems = check_get_buffer_size(index, dst_dtype, buf)?;
        if nitems == 0 {
            return Ok(());
        }
        let (src_dtype, dst_dtype) = (self.array.dtype(), dst_dtype);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);

        let mut tmp_buf = context.tmp_buf(nitems * src_itemsize, src_dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array.read_data(index, tmp_buf, context)?;

        let src_items = tmp_buf.chunks_exact(src_itemsize);
        let dst_items = buf.chunks_exact_mut(dst_itemsize);
        let sub_field_offset = self.sub_field_offset as usize;
        for (src, dst) in src_items.zip(dst_items) {
            let src_field = &src[sub_field_offset..sub_field_offset + dst_itemsize];
            dst.copy_from_slice(src_field);
        }
        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.array.shape()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.dst_type.dtype()
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.array.spec()
    }

    type DimensionChange<NewD: crate::Dimension> = SubDtype<S::DimensionChange<NewD>, ET>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(SubDtype {
            array: self.array.dimension_change()?,
            dst_type: self.dst_type,
            sub_field_offset: self.sub_field_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Array, TypeDyn};

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
        let za = Array::compact_ndarray(&pts).unwrap();
        let xs = za
            .as_ref()
            .dtype_sub_field::<i32>("x")
            .to_ndarray()
            .unwrap();
        let ys = za
            .as_ref()
            .dtype_sub_field::<i32>("y")
            .to_ndarray()
            .unwrap();
        assert_eq!(xs.as_slice().unwrap(), &[1, 2, 3]);
        assert_eq!(ys.as_slice().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn error_not_struct_dtype() {
        let a = Array::compact_ndarray(&ndarray::array![1i32, 2, 3]).unwrap();
        assert!(super::SubDtype::<_, TypeDyn>::new_array(a, "x").is_err());
    }

    #[test]
    fn error_field_not_found() {
        let pts = ndarray::array![Pair { x: 1, y: 10 }];
        let za = Array::compact_ndarray(&pts).unwrap();
        assert!(super::SubDtype::<_, TypeDyn>::new_array(za, "z").is_err());
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
            let nd = ndarray::Array::from_shape_vec([n], pair_structs).unwrap();
            let za = Array::compact_ndarray(&nd).unwrap();
            let expected_x = ndarray::Array::from_shape_vec([n],
                pairs.iter().map(|&(x, _)| x).collect::<Vec<_>>(),
            ).unwrap();
            let expected_y = ndarray::Array::from_shape_vec([n],
                pairs.iter().map(|&(_, y)| y).collect::<Vec<_>>(),
            ).unwrap();
            crate::util::assert_array_matches(&za.as_ref().dtype_sub_field::<i32>("x"), &expected_x);
            crate::util::assert_array_matches(&za.as_ref().dtype_sub_field::<i32>("y"), &expected_y);
        }
    }
}
