use std::io;
use std::ops::{Not, Range};

use crate::Array;
use crate::codec::{DecoderCodecConfig, DecoderParams, EncoderParams, ReadContext};
#[allow(unused_imports)]
use crate::dtype::{Complex, f16};
use crate::dtype::{Dtype, Dtyped};
use crate::iter::NdIter;
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::storage::{ArrayStorage, BlocksLayout};
use crate::util::{DimArray, default_strides, dim_arr};

pub(crate) trait ReductionOpKernel {
    fn reduce<'a>(
        &self,
        data: impl Iterator<Item = &'a [u8]> + Clone,
        out: &mut [u8],
        input_dtype: &Dtype,
    ) -> std::io::Result<()>;

    fn output_dtype(&self, input_dtype: &Dtype) -> io::Result<Dtype>;
}

pub(crate) struct ReductionOp<Op, S> {
    op: Op,

    array: Array<S>,
    is_reduced: DimArray<bool>,
    keepdims: bool,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<Op, S> ReductionOp<Op, S> {
    pub(crate) fn new(op: Op, array: Array<S>, axes: &[usize], keepdims: bool) -> io::Result<Self>
    where
        Op: ReductionOpKernel,
        S: ArrayStorage,
    {
        let output_dtype = op.output_dtype(array.dtype())?;

        let input_ndim = array.shape().len();
        let mut is_reduced = dim_arr(input_ndim, |_| false);
        for &ax in axes {
            if ax >= input_ndim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("axis {ax} out of bounds for array of ndim {input_ndim}"),
                ));
            }
            if is_reduced[ax] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("duplicate axis {ax}"),
                ));
            }
            is_reduced[ax] = true;
        }

        let shape: DimArray<u64> = array
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if !is_reduced[i] { Some(s) } else { None })
            .collect();

        let inner_layout = array.blocks_layout();
        let hint: DimArray<_> = (0..input_ndim)
            .filter(|&d| !is_reduced[d])
            .map(|d| inner_layout.block_shape_hint[d])
            .collect();
        let tag: DimArray<_> = (0..input_ndim)
            .filter(|&d| !is_reduced[d])
            .map(|d| inner_layout.block_shape_tag[d])
            .collect();
        let preferred: DimArray<_> = (0..input_ndim)
            .filter(|&d| !is_reduced[d])
            .map(|d| inner_layout.preferred_read_block_shape[d])
            .collect();
        let mut b_layout = inner_layout.clone();
        b_layout.block_shape_hint = hint;
        b_layout.block_shape_tag = tag;
        b_layout.preferred_read_block_shape = preferred;

        Ok(Self {
            op,
            dtype: output_dtype,
            shape,
            blocks_layout: b_layout,
            array,
            is_reduced,
            keepdims,
        })
    }
}
impl<Op, S> ArrayStorage for ReductionOp<Op, S>
where
    Op: ReductionOpKernel,
    S: ArrayStorage,
{
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()> {
        assert!(!self.keepdims); // TODO

        let orig_shape = self.array.shape();
        assert_eq!(index.len(), self.shape().len());
        let orig_ndim = orig_shape.len();

        // Build inner_index: reduced dims span the full original range,
        // non-reduced dims forward the requested output range.
        let mut out_dim = 0usize;
        let inner_index: DimArray<Range<u64>> = (0..orig_ndim)
            .map(|in_d| {
                if self.is_reduced[in_d] {
                    0..orig_shape[in_d]
                } else {
                    let r = index[out_dim].clone();
                    out_dim += 1;
                    r
                }
            })
            .collect();

        let src_dtype = self.array.dtype();
        let dst_dtype = self.dtype();

        let inner_read_shape: DimArray<usize> = inner_index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect();
        let n_inner: usize = inner_read_shape.iter().product();

        let out_shape: DimArray<usize> = index.iter().map(|r| (r.end - r.start) as usize).collect();

        // Read the full inner block into a temp buffer.
        let tmp_buf_size = n_inner * src_dtype.itemsize() as usize;
        let mut tmp_buf = context.tmp_buf(tmp_buf_size, src_dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array
            .storage
            .read_data(&inner_index, tmp_buf, context)?;

        // C-contiguous byte strides for inner and output layouts.
        let inner_strides = default_strides(&inner_read_shape, src_dtype.itemsize() as usize);
        let out_strides = default_strides(&out_shape, dst_dtype.itemsize() as usize);

        let mut out_iter = NdIter::new(
            &out_shape,
            (
                NdIterExtStridesPtr::new(
                    &inner_strides
                        .iter()
                        .zip(&self.is_reduced)
                        .filter_map(|(stride, is_reduced)| is_reduced.not().then_some(*stride))
                        .collect::<DimArray<_>>(),
                    tmp_buf.as_ptr(),
                ),
                NdIterExtStridesPtrMut::new(&out_strides, buf.as_mut_ptr()),
            ),
        );
        let reduction_shape = dim_arr(orig_ndim, |d| {
            if self.is_reduced[d] { orig_shape[d] } else { 1 }
        });

        while let Some((_out_idx, (base_ptr, out_ptr))) = out_iter.next() {
            let reduction_iter = NdIter::new(
                &reduction_shape,
                NdIterExtStridesPtr::new(&inner_strides, base_ptr),
            );
            let reduction_iter = reduction_iter.map(|(_idx, in_ptr)| unsafe {
                std::slice::from_raw_parts(in_ptr, src_dtype.itemsize() as usize)
            });
            let out_entry =
                unsafe { std::slice::from_raw_parts_mut(out_ptr, dst_dtype.itemsize() as usize) };
            self.op.reduce(reduction_iter, out_entry, src_dtype)?;
        }
        Ok(())
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }

    fn codec_params(&self) -> (&EncoderParams, &DecoderParams, &DecoderCodecConfig) {
        self.array.storage.codec_params()
    }
}

macro_rules! define_reduction_op {
    (
        $Name:ident,
        $NameKernel:ident,
        |$arg_acc:ident, $arg_x:ident| $body:expr,
        [$(($($scalar:tt)*) => ($($reduction_type:tt)*)),* $(,)?]) => {
        pub struct $Name<S>(crate::ops::reduction::ReductionOp<$NameKernel, S>);
        impl<S> $Name<S> {
            pub fn new(array: crate::Array<S>, axes: &[usize], keepdims: bool) -> std::io::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                Ok(Self(crate::ops::reduction::ReductionOp::new($NameKernel, array, axes, keepdims)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        define_reduction_op_kernel!($NameKernel, |$arg_acc, $arg_x| $body, [$(($($scalar)*) => ($($reduction_type)*)),*]);
    };
}

macro_rules! define_reduction_op_kernel {
    (
        $NameKernel:ident,
        |$arg_acc:ident, $arg_x:ident| $body:expr,
        [$(($($scalar:tt)*) => ($($reduction_type:tt)*)),* $(,)?]) => {
        struct $NameKernel;
        impl crate::ops::reduction::ReductionOpKernel for $NameKernel {
            fn reduce<'a>(
                &self,
                data: impl Iterator<Item = &'a [u8]> + Clone,
                out: &mut [u8],
                input_dtype: &Dtype,
            ) -> std::io::Result<()> {
                macro_rules! apply_loop_impl {
                    ($scalar2:ty, $reduction_type2:ty) => {{
                        let mut data = data.map(|x| unsafe { x.as_ptr().cast::<$scalar2>().read() });
                        let mut acc: $reduction_type2 = crate::ops::astype::cast(data.next().unwrap());
                        while let Some(x) = data.next() {
                            acc = {
                                let ($arg_acc, $arg_x) = (acc, x);
                                $body
                            };
                        }
                        unsafe { out.as_mut_ptr().cast::<$reduction_type2>().write(acc) };
                        return Ok(())
                    }};
                }
                macro_rules! apply_loop {
                    (f16, $reduction_type2:ty) => {
                        #[cfg(feature = "half")]
                        apply_loop_impl!(f16, $reduction_type2)
                    };
                    (Complex<f32>, $reduction_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(Complex<f32>, $reduction_type2)
                    };
                    (Complex<f64>, $reduction_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(Complex<f64>, $reduction_type2)
                    };
                    ($scalar2:ty, $reduction_type2:ty) => {
                        apply_loop_impl!($scalar2, $reduction_type2)
                    };
                }
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($($scalar)*)) => {
                        apply_loop!($($scalar)*, $($reduction_type)*)
                    },)*
                    _ => {}
                }
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Reduction op not supported for dtype {input_dtype:#?}"),
                ))
            }

            fn output_dtype(&self, input_dtype: &crate::dtype::Dtype) -> std::io::Result<crate::dtype::Dtype> {
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($($scalar)*)) => {
                        return Ok($($reduction_type)*::DTYPE);
                    },)*
                    _ => {},

                };
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Max reduction not supported for dtype {input_dtype:#?}"),
                ))
            }
        }
    };
}
// pub(crate) use {define_reduction_op, define_reduction_op_kernel};

define_reduction_op!(
    Max,
    MaxKernel,
    |m, x| m.max(crate::ops::astype::cast(x)),
    [
        (i8) => (i64),
        (i16) => (i64),
        (i32) => (i64),
        (i64) => (i64),
        (u8) => (u64),
        (u16) => (u64),
        (u32) => (u64),
        (u64) => (u64),
        (f16) => (f64),
        (f32) => (f64),
        (f64) => (f64),
        // TODO
        // (Complex<f32>) => (Complex<f64>),
        // (Complex<f64>) => (Complex<f64>),
        (bool) => (u64),
    ]
);

macro_rules! define_array_reduction_method {
    ($op:ident : $Name:ident) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op(self, axes: &[usize], keepdims: bool) -> crate::Array<$Name<S>> {
            let op = $Name::new(self, axes, keepdims).unwrap();
            crate::Array::from_storage(op)
        }
    };
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(max: Max);
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::array::{Array, ArrayParams};
    use crate::block::BlockSize;

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    fn make(vals: Vec<i32>, shape: &[usize]) -> Array<crate::storage::Owned> {
        let nd = ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
        Array::from_ndarray(&nd, arr_params(shape)).unwrap()
    }

    fn seq(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_reduce_axis0() {
        // [3, 4] reduce axis 0 → [4]
        assert_eq!(make(seq(12), &[3, 4]).max(&[0], false).shape(), &[4]);
    }

    #[test]
    fn shape_reduce_axis1() {
        // [3, 4] reduce axis 1 → [3]
        assert_eq!(make(seq(12), &[3, 4]).max(&[1], false).shape(), &[3]);
    }

    #[test]
    fn shape_reduce_both_axes() {
        // [3, 4] reduce axes [0, 1] → []
        assert_eq!(
            make(seq(12), &[3, 4]).max(&[0, 1], false).shape(),
            &[] as &[u64]
        );
    }

    #[test]
    fn shape_reduce_middle_axis() {
        // [2, 3, 4] reduce axis 1 → [2, 4]
        let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
        let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
        assert_eq!(a.max(&[1], false).shape(), &[2, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_reduce_axis0() {
        // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] max over rows → [8,9,10,11]
        let got: ArrayD<i32> = make(seq(12), &[3, 4])
            .max(&[0], false)
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![8, 9, 10, 11]).unwrap()
        );
    }

    #[test]
    fn full_read_reduce_axis1() {
        // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] max over cols → [3,7,11]
        let got: ArrayD<i32> = make(seq(12), &[3, 4])
            .max(&[1], false)
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3], vec![3, 7, 11]).unwrap()
        );
    }

    #[test]
    fn full_read_reduce_all_axes() {
        // max of [0..12] = 11
        let got: ArrayD<i32> = make(seq(12), &[3, 4])
            .max(&[0, 1], false)
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![11]).unwrap());
    }

    #[test]
    fn full_read_reduce_3d_middle() {
        // shape [2, 3, 4], reduce axis 1
        // For each (i, k): max over j of a[i, j, k]
        // a[i, j, k] = i*12 + j*4 + k
        // max_j a[i, j, k] = i*12 + 2*4 + k = i*12 + k + 8  (j=2 is max)
        let vals: Vec<i32> = (0..24).collect();
        let nd = ArrayD::from_shape_vec(vec![2, 3, 4], vals).unwrap();
        let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
        let got: ArrayD<i32> = a.max(&[1], false).data().to_ndarray().unwrap();
        // expected[i, k] = i*12 + 8 + k
        let expected: Vec<i32> = (0..2)
            .flat_map(|i| (0..4).map(move |k| i * 12 + 8 + k))
            .collect();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 4], expected).unwrap());
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic]
    fn error_axis_out_of_bounds() {
        make(seq(6), &[2, 3]).max(&[2], false);
    }

    #[test]
    #[should_panic]
    fn error_duplicate_axis() {
        make(seq(6), &[2, 3]).max(&[0, 0], false);
    }
}
