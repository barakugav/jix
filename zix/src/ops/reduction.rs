use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
#[allow(unused_imports)]
use crate::dtype::{f16, Complex};
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::util::iter::NdIter;
use crate::util::{default_strides, dim_arr, DimArray};
use crate::Array;

pub(crate) trait ReductionOpKernel {
    fn reduce<'a>(
        &self,
        slice_iter: impl Iterator<Item = (impl Iterator<Item = &'a [u8]> + Clone, &'a mut [u8])>,
        input_dtype: &Dtype,
    ) -> Result<()>;

    fn output_dtype(&self, input_dtype: &Dtype) -> Result<Dtype>;
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
    pub(crate) fn new(op: Op, array: Array<S>, axes: &[usize], keepdims: bool) -> Result<Self>
    where
        Op: ReductionOpKernel,
        S: ArrayStorage,
    {
        let output_dtype = op.output_dtype(array.dtype())?;

        let input_ndim = array.shape().len();
        let mut is_reduced = dim_arr(input_ndim, |_| false);
        for &ax in axes {
            ensure!(
                ax < input_ndim,
                InvalidArgument,
                "axis {ax} out of bounds for array of ndim {input_ndim}"
            );

            ensure!(!is_reduced[ax], InvalidArgument, "duplicate axis {ax}");
            is_reduced[ax] = true;
        }

        let shape = array
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                if is_reduced[i] {
                    keepdims.then_some(1)
                } else {
                    Some(s)
                }
            })
            .collect::<DimArray<_>>();

        let mut b_layout = array.blocks_layout().clone();
        b_layout.block_shape_hint = (0..input_ndim)
            .filter_map(|d| {
                if is_reduced[d] {
                    keepdims.then_some(1)
                } else {
                    Some(b_layout.block_shape_hint[d])
                }
            })
            .collect();
        b_layout.block_shape_tag = (0..input_ndim)
            .filter_map(|d| {
                if is_reduced[d] {
                    keepdims.then_some(crate::storage::BlockShapeTag::Any)
                } else {
                    Some(b_layout.block_shape_tag[d])
                }
            })
            .collect();
        b_layout.preferred_read_block_shape = (0..input_ndim)
            .filter_map(|d| {
                if is_reduced[d] {
                    keepdims.then_some(1)
                } else {
                    Some(b_layout.preferred_read_block_shape[d])
                }
            })
            .collect();

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
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, &self.dtype, buf)?;

        let orig_shape = self.array.shape();
        let orig_ndim = orig_shape.len();

        // Build inner_index: reduced dims span the full original range,
        // non-reduced dims forward the requested output range.
        //
        // With keepdims=false the output has fewer dims than the input, so we
        // use `out_dim` to step through `index`.  With keepdims=true the output
        // has the same number of dims (reduced ones are size-1), so `index[d]`
        // maps directly to input dim `d`.
        let mut out_dim = 0usize;
        let inner_index = (0..orig_ndim)
            .map(|in_d| {
                if self.is_reduced[in_d] {
                    if self.keepdims {
                        out_dim += 1; // skip the size-1 keepdim slot
                    }
                    // TODO: we could read it in chunks
                    0..orig_shape[in_d]
                } else {
                    let r = index[out_dim].clone();
                    out_dim += 1;
                    r
                }
            })
            .collect::<DimArray<_>>();

        let src_dtype = self.array.dtype();
        let dst_dtype = self.dtype();

        let inner_read_shape = inner_index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<DimArray<_>>();
        let n_inner: usize = inner_read_shape.iter().product();

        let out_shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<DimArray<_>>();

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

        // Strides used by out_iter to advance the `base_ptr` into tmp_buf.
        // For reduced dims the outer loop visits exactly one position (size 1
        // when keepdims=true, or the dim is absent when keepdims=false), so
        // their stride contribution to base_ptr is 0.
        let tmp_buf_strides = inner_strides
            .iter()
            .zip(&self.is_reduced)
            .filter_map(|(&s, &reduced)| {
                if reduced {
                    self.keepdims.then_some(0)
                } else {
                    Some(s)
                }
            })
            .collect::<DimArray<_>>();

        let out_iter = NdIter::new(
            &out_shape,
            (
                NdIterExtStridesPtr::new(&tmp_buf_strides, tmp_buf.as_ptr()),
                NdIterExtStridesPtrMut::new(&out_strides, buf.as_mut_ptr()),
            ),
        );
        let reduction_shape = dim_arr(orig_ndim, |d| {
            if self.is_reduced[d] {
                orig_shape[d]
            } else {
                1
            }
        });

        let slice_iter = out_iter.map(|(_out_idx, (base_ptr, out_ptr))| {
            let reduction_iter = NdIter::new(
                &reduction_shape,
                NdIterExtStridesPtr::new(&inner_strides, base_ptr),
            );
            let reduction_iter = reduction_iter.map(|(_idx, in_ptr)| unsafe {
                std::slice::from_raw_parts(in_ptr, src_dtype.itemsize() as usize)
            });
            let out_entry =
                unsafe { std::slice::from_raw_parts_mut(out_ptr, dst_dtype.itemsize() as usize) };
            (reduction_iter, out_entry)
        });
        self.op.reduce(slice_iter, src_dtype)
    }

    fn shape(&self) -> &[u64] {
        &self.shape
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

macro_rules! define_reduction_op {
    (
        $Name:ident,
        $NameKernel:ident,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = $types:tt,
        single_axis = "true"
    ) => {
        pub struct $Name<S>(crate::ops::reduction::ReductionOp<$NameKernel, S>);
        impl<S> $Name<S> {
            pub fn new(array: crate::Array<S>, axis: usize, keepdims: bool $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                let kernel = $NameKernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(kernel, array, &[axis], keepdims)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        define_reduction_op_kernel!(
            $NameKernel,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = $types
        );
    };

    (
        $Name:ident,
        $NameKernel:ident,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = $types:tt
    ) => {
        pub struct $Name<S>(crate::ops::reduction::ReductionOp<$NameKernel, S>);
        impl<S> $Name<S> {
            pub fn new(array: crate::Array<S>, axes: &[usize], keepdims: bool $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                let kernel = $NameKernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(kernel, array, axes, keepdims)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        define_reduction_op_kernel!(
            $NameKernel,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = $types
        );
    };
}

macro_rules! define_reduction_op_kernel {
    (
        $NameKernel:ident,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {
            input = [$($scalar:tt),* $(,)?],
            output = "same"
        }
    ) => {
        define_reduction_op_kernel!(
            $NameKernel,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = {$($scalar => $scalar),*}
        );
    };

    (
        $NameKernel:ident,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {
            input = [$($scalar:tt),* $(,)?],
            output = $output_type:tt
        }
    ) => {
        define_reduction_op_kernel!(
            $NameKernel,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = {$($scalar => $output_type),*}
        );
    };

    (
        $NameKernel:ident,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {$([$($scalar:tt),*] => $output_type:tt),* $(,)?}
    ) => {
        define_reduction_op_kernel!(
            $NameKernel,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = {$($($scalar => $output_type),*),*}
        );
    };

    (
        $NameKernel:ident,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {$($scalar:tt => $reduction_type:tt),* $(,)?}
    ) => {
        struct $NameKernel { $($extra_arg: $extra_ty,)* }
        impl crate::ops::reduction::ReductionOpKernel for $NameKernel {
            fn reduce<'a>(
                &self,
                slice_iter: impl Iterator<Item = (impl Iterator<Item = &'a [u8]> + Clone, &'a mut [u8])>,
                input_dtype: &Dtype,
            ) -> crate::error::Result<()> {
                macro_rules! apply_loop_impl {
                    ($scalar2:ty, $reduction_type2:ty) => {{
                        for (slice, out) in slice_iter {
                            let items = slice.map(|x| unsafe { x.as_ptr().cast::<$scalar2>().read() });
                            let result: $reduction_type2 = {
                                #[allow(unused)]
                                type ReductionType = $reduction_type2;
                                $(let $extra_arg = self.$extra_arg;)*
                                let $arg_items = items;
                                { $body }
                            };
                            unsafe { out.as_mut_ptr().cast::<$reduction_type2>().write(result) };
                        }
                        return Ok(())
                    }};
                }
                macro_rules! apply_loop {
                    (f16, $reduction_type2:ty) => {
                        #[cfg(feature = "half")]
                        apply_loop_impl!(f16, $reduction_type2)
                    };
                    ((Complex<f32>), $reduction_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(Complex<f32>, $reduction_type2)
                    };
                    ((Complex<f64>), $reduction_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(Complex<f64>, $reduction_type2)
                    };
                    ($scalar2:ty, $reduction_type2:ty) => {
                        apply_loop_impl!($scalar2, $reduction_type2)
                    };
                }
                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($scalar)) => {
                        apply_loop!($scalar, $reduction_type)
                    },)*
                    _ => {}
                }
                bail!(UnsupportedDtype, "Reduction op not supported for dtype {input_dtype:#?}");
            }

            fn output_dtype(&self, input_dtype: &crate::dtype::Dtype) -> crate::error::Result<crate::dtype::Dtype> {
                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($scalar)) => {
                        return Ok(<$reduction_type as crate::dtype::Dtyped>::DTYPE);
                    },)*
                    _ => {},

                };
                bail!(UnsupportedDtype, "Reduction op not supported for dtype {input_dtype:#?}");
            }
        }
    };
}
// pub(crate) use {define_reduction_op, define_reduction_op_kernel};

define_reduction_op!(
    Max,
    MaxKernel,
    |items| { items.reduce(|m, x| m.max(x)).unwrap() },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = "same"
    }
);
define_reduction_op!(
    Min,
    MinKernel,
    |items| { items.reduce(|m, x| m.min(x)).unwrap() },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = "same"
    }
);
define_reduction_op!(
    ArgMax,
    ArgMaxKernel,
    |items| {
        items
            .enumerate()
            .reduce(|(m_idx, m), (idx, x)| if x > m { (idx, x) } else { (m_idx, m) })
            .unwrap()
            .0 as u64
    },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = u64
    },
    single_axis = "true"
);
define_reduction_op!(
    ArgMin,
    ArgMinKernel,
    |items| {
        items
            .enumerate()
            .reduce(|(m_idx, m), (idx, x)| if x < m { (idx, x) } else { (m_idx, m) })
            .unwrap()
            .0 as u64
    },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = u64
    },
    single_axis = "true"
);
define_reduction_op!(
    Sum,
    SumKernel,
    |items| { items.fold(crate::ops::astype::cast(0), |m, x| m + crate::ops::astype::cast_as(x, &m)) },
    types = {
        [i8, i16, i32, i64] => i64,
        [u8, u16, u32, u64, bool] => u64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => (Complex::<f64>),
    }
);
define_reduction_op!(
    Product,
    ProductKernel,
    |items| { items.fold(crate::ops::astype::cast(1), |m, x| m * crate::ops::astype::cast_as(x, &m)) },
    types = {
        [i8, i16, i32, i64] => i64,
        [u8, u16, u32, u64] => u64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => (Complex::<f64>),
    }
);
define_reduction_op!(
    Mean,
    MeanKernel,
    |items| {{
        let (size, size_high) = items.size_hint();
        assert_eq!(Some(size), size_high);
        assert!(size > 0);
        let sum = items.fold(crate::ops::astype::cast::<_, ReductionType>(0), |m, x| m + crate::ops::astype::cast_as(x, &m));
        sum / size as f64
     }},
    types = {
        [i8, i16, i32, i64] => f64,
        [u8, u16, u32, u64] => f64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => (Complex::<f64>),
        [bool] => f64,
    }
);
define_reduction_op!(
    Variance,
    VarianceKernel,
    |items, ddof: f64| { variance_impl(items, ddof) },
    types = {
        [i8, i16, i32, i64] => f64,
        [u8, u16, u32, u64] => f64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => f64,
    }
);
define_reduction_op!(
    StandardDeviation,
    StandardDeviationKernel,
    |items, ddof: f64| { variance_impl(items, ddof).sqrt() },
    types = {
        [i8, i16, i32, i64] => f64,
        [u8, u16, u32, u64] => f64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => f64,
    }
);
fn variance_impl<T>(items: impl Iterator<Item = T>, ddof: f64) -> f64
where
    T: VarianceImpl,
    i32: crate::ops::astype::Cast<T::MeanType>,
    T: crate::ops::astype::Cast<T::MeanType>,
    T::MeanType: core::ops::Sub<T::MeanType, Output = T::MeanType>
        + core::ops::Div<f64, Output = T::MeanType>
        + core::ops::AddAssign<T::MeanType>
        + Copy,
{
    let mut mean: T::MeanType = crate::ops::astype::cast(0);
    let mut m2 = 0.0_f64;
    let mut n = 0_u64;

    for x in items {
        let x: T::MeanType = crate::ops::astype::cast(x);
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        let delta2 = x - mean;
        m2 += T::update_m2(delta, delta2);
    }

    let denom = n as f64 - ddof;
    if denom <= 0.0 {
        f64::NAN
    } else {
        m2 / denom
    }
}
trait VarianceImpl {
    type MeanType;
    fn update_m2(delta: Self::MeanType, delta2: Self::MeanType) -> f64;
}
macro_rules! impl_num_variance_impl {
    ($ty:ty) => {
        impl VarianceImpl for $ty {
            type MeanType = f64;
            fn update_m2(delta: f64, delta2: f64) -> f64 {
                delta * delta2
            }
        }
    };
}
macro_rules! impl_complex_variance_impl {
    ($f_ty:ty) => {
        impl VarianceImpl for Complex<$f_ty> {
            type MeanType = Complex<f64>;
            fn update_m2(delta: Complex<f64>, delta2: Complex<f64>) -> f64 {
                delta.re * delta2.re + delta.im * delta2.im
            }
        }
    };
}
impl_num_variance_impl!(i8);
impl_num_variance_impl!(i16);
impl_num_variance_impl!(i32);
impl_num_variance_impl!(i64);
impl_num_variance_impl!(u8);
impl_num_variance_impl!(u16);
impl_num_variance_impl!(u32);
impl_num_variance_impl!(u64);
#[cfg(feature = "half")]
impl_num_variance_impl!(f16);
impl_num_variance_impl!(f32);
impl_num_variance_impl!(f64);
impl_complex_variance_impl!(f32);
impl_complex_variance_impl!(f64);
impl_num_variance_impl!(bool);

define_reduction_op!(
    All,
    AllKernel,
    |items| { items.fold(true, |m, x| m && crate::ops::astype::cast::<_, bool>(x)) },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
        output = bool
    }
);
define_reduction_op!(
    Any,
    AnyKernel,
    |items| { items.fold(false, |m, x| m || crate::ops::astype::cast::<_, bool>(x)) },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
        output = bool
    }
);

macro_rules! define_array_reduction_method {
    ($op:ident : $Name:ident, single_axis = "true" $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op(self, axis: usize, keepdims: bool $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Name<S>> {
            let op = $Name::new(self, axis, keepdims $($(, $extra_arg)*)?).unwrap();
            crate::Array::from_storage(op)
        }
    };

    ($op:ident : $Name:ident $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op(self, axes: &[usize], keepdims: bool $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Name<S>> {
            let op = $Name::new(self, axes, keepdims $($(, $extra_arg)*)?).unwrap();
            crate::Array::from_storage(op)
        }
    };
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(max: Max);
    define_array_reduction_method!(min: Min);
    define_array_reduction_method!(argmax: ArgMax, single_axis = "true");
    define_array_reduction_method!(argmin: ArgMin, single_axis = "true");
    define_array_reduction_method!(sum: Sum);
    define_array_reduction_method!(product: Product);
    define_array_reduction_method!(mean: Mean);
    define_array_reduction_method!(var: Variance, extra_args = (ddof: f64));
    define_array_reduction_method!(std: StandardDeviation, extra_args = (ddof: f64));
    define_array_reduction_method!(all: All);
    define_array_reduction_method!(any: Any);
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::array::Array;
    #[cfg(feature = "half")]
    use crate::dtype::f16;
    #[cfg(feature = "num-complex")]
    use crate::dtype::Complex;
    use crate::util::arr_params;

    fn make<T>(vals: Vec<T>, shape: &[usize]) -> Array<crate::storage::Compact>
    where
        T: Clone + crate::dtype::Dtyped,
    {
        let nd = ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
        Array::from_ndarray(&nd, arr_params(shape)).unwrap()
    }

    fn seq(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    /// Test a reduction along axis 0 for a 2×N array.
    /// `$method` is the reduction method name (max/min/sum/product/all/any).
    /// `$in_ty` is the element type of the input array.
    /// `$out_ty` is the expected element type of the output array.
    /// `rows` gives the two rows of input values; `expected` gives the N expected output values.
    macro_rules! test_reduce_axis0 {
        ($test_name:ident, $method:ident,
         in = $in_ty:ty, out = $out_ty:ty,
         rows: [[$($a:expr),+], [$($b:expr),+]],
         expected: [$($e:expr),+]) => {
            #[test]
            fn $test_name() {
                let row0: Vec<$in_ty> = vec![$($a),+];
                let row1: Vec<$in_ty> = vec![$($b),+];
                let n = row0.len();
                let input: Vec<$in_ty> = row0.into_iter().chain(row1).collect();
                let nd = ArrayD::<$in_ty>::from_shape_vec(vec![2, n], input).unwrap();
                let a = Array::from_ndarray(&nd, crate::util::arr_params(&[2, n])).unwrap();
                let got: ArrayD<$out_ty> = a.$method(&[0], false).data().to_ndarray().unwrap();
                assert_eq!(
                    got,
                    ArrayD::<$out_ty>::from_shape_vec(vec![n], vec![$($e),+]).unwrap()
                );
            }
        };
    }

    // -----------------------------------------------------------------------
    // max
    // -----------------------------------------------------------------------

    mod max {
        use crate::util::arr_params;

        use super::*;

        #[test]
        fn shape_axis0() {
            assert_eq!(make(seq(12), &[3, 4]).max(&[0], false).shape(), &[4]);
        }

        #[test]
        fn shape_axis1() {
            assert_eq!(make(seq(12), &[3, 4]).max(&[1], false).shape(), &[3]);
        }

        #[test]
        fn shape_both_axes() {
            assert_eq!(
                make(seq(12), &[3, 4]).max(&[0, 1], false).shape(),
                &[] as &[u64]
            );
        }

        #[test]
        fn shape_middle_axis_3d() {
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            assert_eq!(a.max(&[1], false).shape(), &[2, 4]);
        }

        // keepdims shape tests
        #[test]
        fn keepdims_shape_axis0() {
            assert_eq!(make(seq(12), &[3, 4]).max(&[0], true).shape(), &[1, 4]);
        }

        #[test]
        fn keepdims_shape_axis1() {
            assert_eq!(make(seq(12), &[3, 4]).max(&[1], true).shape(), &[3, 1]);
        }

        #[test]
        fn keepdims_shape_both_axes() {
            assert_eq!(make(seq(12), &[3, 4]).max(&[0, 1], true).shape(), &[1, 1]);
        }

        #[test]
        fn keepdims_shape_middle_3d() {
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            assert_eq!(a.max(&[1], true).shape(), &[2, 1, 4]);
        }

        // keepdims value tests
        #[test]
        fn keepdims_read_axis0_i32() {
            // [[0..3],[4..7],[8..11]] → max per col = [[8,9,10,11]] shape [1,4]
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .max(&[0], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![1, 4], vec![8, 9, 10, 11]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_axis1_i32() {
            // max per row = [[3],[7],[11]] shape [3,1]
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .max(&[1], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3, 1], vec![3, 7, 11]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_both_axes_i32() {
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .max(&[0, 1], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![1, 1], vec![11]).unwrap());
        }

        #[test]
        fn keepdims_read_3d_middle_axis() {
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            let got: ArrayD<i32> = a.max(&[1], true).data().to_ndarray().unwrap();
            // same values as no-keepdims but shape [2,1,4]
            let expected: Vec<i32> = (0..2)
                .flat_map(|i| (0..4).map(move |k| i * 12 + 8 + k))
                .collect();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2, 1, 4], expected).unwrap()
            );
        }

        #[test]
        fn read_axis0_i32() {
            // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → max per col
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
        fn read_axis1_i32() {
            // max per row
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
        fn read_all_axes_i32() {
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .max(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![11]).unwrap());
        }

        #[test]
        fn read_3d_middle_axis() {
            // a[i,j,k] = i*12 + j*4 + k; max over j → i*12 + 8 + k
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            let got: ArrayD<i32> = a.max(&[1], false).data().to_ndarray().unwrap();
            let expected: Vec<i32> = (0..2)
                .flat_map(|i| (0..4).map(move |k| i * 12 + 8 + k))
                .collect();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2, 4], expected).unwrap());
        }

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

        // dtype coverage: rows [[1,4],[3,2]] → max [3,4]
        test_reduce_axis0!(i8,   max, in = i8,   out = i8,   rows: [[1i8,  4i8 ], [3i8,  2i8 ]], expected: [3i8,  4i8 ]);
        test_reduce_axis0!(u8,   max, in = u8,   out = u8,   rows: [[1u8,  4u8 ], [3u8,  2u8 ]], expected: [3u8,  4u8 ]);
        test_reduce_axis0!(i16,  max, in = i16,  out = i16,  rows: [[1i16, 4i16], [3i16, 2i16]], expected: [3i16, 4i16]);
        test_reduce_axis0!(u16,  max, in = u16,  out = u16,  rows: [[1u16, 4u16], [3u16, 2u16]], expected: [3u16, 4u16]);
        test_reduce_axis0!(i32,  max, in = i32,  out = i32,  rows: [[1i32, 4i32], [3i32, 2i32]], expected: [3i32, 4i32]);
        test_reduce_axis0!(u32,  max, in = u32,  out = u32,  rows: [[1u32, 4u32], [3u32, 2u32]], expected: [3u32, 4u32]);
        test_reduce_axis0!(i64,  max, in = i64,  out = i64,  rows: [[1i64, 4i64], [3i64, 2i64]], expected: [3i64, 4i64]);
        test_reduce_axis0!(u64,  max, in = u64,  out = u64,  rows: [[1u64, 4u64], [3u64, 2u64]], expected: [3u64, 4u64]);
        test_reduce_axis0!(f32,  max, in = f32,  out = f32,  rows: [[1f32, 4f32], [3f32, 2f32]], expected: [3f32, 4f32]);
        test_reduce_axis0!(f64,  max, in = f64,  out = f64,  rows: [[1f64, 4f64], [3f64, 2f64]], expected: [3f64, 4f64]);
        test_reduce_axis0!(bool, max, in = bool, out = bool, rows: [[false, true], [true, false]], expected: [true, true]);
        #[cfg(feature = "half")]
        test_reduce_axis0!(f16, max, in = f16, out = f16,
            rows: [[f16::from_f32(1.0), f16::from_f32(4.0)], [f16::from_f32(3.0), f16::from_f32(2.0)]],
            expected: [f16::from_f32(3.0), f16::from_f32(4.0)]);
    }

    // -----------------------------------------------------------------------
    // min
    // -----------------------------------------------------------------------

    mod min {
        use crate::util::arr_params;

        use super::*;

        #[test]
        fn read_axis0_i32() {
            // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → min per col = [0,1,2,3]
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .min(&[0], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![4], vec![0, 1, 2, 3]).unwrap()
            );
        }

        #[test]
        fn read_axis1_i32() {
            // min per row = [0, 4, 8]
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .min(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![3], vec![0, 4, 8]).unwrap());
        }

        #[test]
        fn read_all_axes_i32() {
            let got: ArrayD<i32> = make(seq(12), &[3, 4])
                .min(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![0]).unwrap());
        }

        #[test]
        fn read_3d_middle_axis() {
            // a[i,j,k] = i*12 + j*4 + k; min over j → i*12 + k  (j=0 is min)
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            let got: ArrayD<i32> = a.min(&[1], false).data().to_ndarray().unwrap();
            let expected: Vec<i32> = (0..2)
                .flat_map(|i| (0..4).map(move |k| i * 12 + k))
                .collect();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2, 4], expected).unwrap());
        }

        #[test]
        fn output_dtype_same_as_input() {
            let a: Array<_> = make(seq(6), &[2, 3]);
            let dtype = a.dtype().clone();
            assert_eq!(a.min(&[0], false).dtype(), &dtype);
        }

        // dtype coverage: rows [[1,4],[3,2]] → min [1,2]
        test_reduce_axis0!(i8,   min, in = i8,   out = i8,   rows: [[1i8,  4i8 ], [3i8,  2i8 ]], expected: [1i8,  2i8 ]);
        test_reduce_axis0!(u8,   min, in = u8,   out = u8,   rows: [[1u8,  4u8 ], [3u8,  2u8 ]], expected: [1u8,  2u8 ]);
        test_reduce_axis0!(i16,  min, in = i16,  out = i16,  rows: [[1i16, 4i16], [3i16, 2i16]], expected: [1i16, 2i16]);
        test_reduce_axis0!(u16,  min, in = u16,  out = u16,  rows: [[1u16, 4u16], [3u16, 2u16]], expected: [1u16, 2u16]);
        test_reduce_axis0!(i32,  min, in = i32,  out = i32,  rows: [[1i32, 4i32], [3i32, 2i32]], expected: [1i32, 2i32]);
        test_reduce_axis0!(u32,  min, in = u32,  out = u32,  rows: [[1u32, 4u32], [3u32, 2u32]], expected: [1u32, 2u32]);
        test_reduce_axis0!(i64,  min, in = i64,  out = i64,  rows: [[1i64, 4i64], [3i64, 2i64]], expected: [1i64, 2i64]);
        test_reduce_axis0!(u64,  min, in = u64,  out = u64,  rows: [[1u64, 4u64], [3u64, 2u64]], expected: [1u64, 2u64]);
        test_reduce_axis0!(f32,  min, in = f32,  out = f32,  rows: [[1f32, 4f32], [3f32, 2f32]], expected: [1f32, 2f32]);
        test_reduce_axis0!(f64,  min, in = f64,  out = f64,  rows: [[1f64, 4f64], [3f64, 2f64]], expected: [1f64, 2f64]);
        test_reduce_axis0!(bool, min, in = bool, out = bool, rows: [[false, true], [true, false]], expected: [false, false]);
        #[cfg(feature = "half")]
        test_reduce_axis0!(f16, min, in = f16, out = f16,
            rows: [[f16::from_f32(1.0), f16::from_f32(4.0)], [f16::from_f32(3.0), f16::from_f32(2.0)]],
            expected: [f16::from_f32(1.0), f16::from_f32(2.0)]);
    }

    // -----------------------------------------------------------------------
    // sum
    // -----------------------------------------------------------------------

    mod sum {
        use super::*;

        #[test]
        fn output_dtype_i32_to_i64() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(6), &[2, 3]);
            assert_eq!(
                a.sum(&[0], false).dtype().try_to_scalar(),
                Some(DtypeScalarKind::I64)
            );
        }

        #[test]
        fn read_axis0_i32() {
            // [[0,1,2],[3,4,5]] sum over rows → [3,5,7] (as i64)
            let got: ArrayD<i64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .sum(&[0], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![3i64, 5, 7]).unwrap()
            );
        }

        #[test]
        fn read_axis1_i32() {
            // [[0,1,2],[3,4,5]] sum over cols → [3, 12] (as i64)
            let got: ArrayD<i64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .sum(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![3i64, 12]).unwrap()
            );
        }

        #[test]
        fn read_all_axes_i32() {
            // sum of 0..6 = 15
            let got: ArrayD<i64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .sum(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![15i64]).unwrap());
        }

        #[test]
        fn read_f64_input() {
            // f64 input → f64 output
            let got: ArrayD<f64> = make(vec![1.0f64, 2.0, 3.0, 4.0], &[2, 2])
                .sum(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![3.0f64, 7.0]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_axis0_i32() {
            // [[0,1,2],[3,4,5]] sum over rows with keepdims → [[3,5,7]] shape [1,3]
            let got: ArrayD<i64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .sum(&[0], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![1, 3], vec![3i64, 5, 7]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_axis1_i32() {
            // [[0,1,2],[3,4,5]] sum over cols with keepdims → [[3],[12]] shape [2,1]
            let got: ArrayD<i64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .sum(&[1], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2, 1], vec![3i64, 12]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_both_axes_i32() {
            // sum of 0..6 = 15, shape [1,1]
            let got: ArrayD<i64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .sum(&[0, 1], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![1, 1], vec![15i64]).unwrap()
            );
        }

        // dtype coverage: rows [[1,2],[3,4]] → sum [4,6]
        // signed integers widen to i64
        test_reduce_axis0!(i8,  sum, in = i8,  out = i64, rows: [[1i8,  2i8 ], [3i8,  4i8 ]], expected: [4i64, 6i64]);
        test_reduce_axis0!(i16, sum, in = i16, out = i64, rows: [[1i16, 2i16], [3i16, 4i16]], expected: [4i64, 6i64]);
        test_reduce_axis0!(i32, sum, in = i32, out = i64, rows: [[1i32, 2i32], [3i32, 4i32]], expected: [4i64, 6i64]);
        test_reduce_axis0!(i64, sum, in = i64, out = i64, rows: [[1i64, 2i64], [3i64, 4i64]], expected: [4i64, 6i64]);
        // unsigned integers widen to u64
        test_reduce_axis0!(u8,  sum, in = u8,  out = u64, rows: [[1u8,  2u8 ], [3u8,  4u8 ]], expected: [4u64, 6u64]);
        test_reduce_axis0!(u16, sum, in = u16, out = u64, rows: [[1u16, 2u16], [3u16, 4u16]], expected: [4u64, 6u64]);
        test_reduce_axis0!(u32, sum, in = u32, out = u64, rows: [[1u32, 2u32], [3u32, 4u32]], expected: [4u64, 6u64]);
        test_reduce_axis0!(u64, sum, in = u64, out = u64, rows: [[1u64, 2u64], [3u64, 4u64]], expected: [4u64, 6u64]);
        // floats widen to f64
        test_reduce_axis0!(f32, sum, in = f32, out = f64, rows: [[1f32, 2f32], [3f32, 4f32]], expected: [4f64, 6f64]);
        test_reduce_axis0!(f64, sum, in = f64, out = f64, rows: [[1f64, 2f64], [3f64, 4f64]], expected: [4f64, 6f64]);
        // bool widens to u64
        test_reduce_axis0!(bool, sum, in = bool, out = u64, rows: [[true, false], [true, true]], expected: [2u64, 1u64]);
        #[cfg(feature = "half")]
        test_reduce_axis0!(f16, sum, in = f16, out = f64,
            rows: [[f16::from_f32(1.0), f16::from_f32(2.0)], [f16::from_f32(3.0), f16::from_f32(4.0)]],
            expected: [4f64, 6f64]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f32, sum, in = Complex<f32>, out = Complex<f64>,
            rows: [[Complex { re: 1f32, im: 0f32 }, Complex { re: 0f32, im: 2f32 }],
                   [Complex { re: 3f32, im: 0f32 }, Complex { re: 0f32, im: 4f32 }]],
            expected: [Complex { re: 4f64, im: 0f64 }, Complex { re: 0f64, im: 6f64 }]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f64, sum, in = Complex<f64>, out = Complex<f64>,
            rows: [[Complex { re: 1f64, im: 0f64 }, Complex { re: 0f64, im: 2f64 }],
                   [Complex { re: 3f64, im: 0f64 }, Complex { re: 0f64, im: 4f64 }]],
            expected: [Complex { re: 4f64, im: 0f64 }, Complex { re: 0f64, im: 6f64 }]);
    }

    // -----------------------------------------------------------------------
    // product
    // -----------------------------------------------------------------------

    mod product {
        use super::*;

        #[test]
        fn output_dtype_i32_to_i64() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(6), &[2, 3]);
            assert_eq!(
                a.product(&[0], false).dtype().try_to_scalar(),
                Some(DtypeScalarKind::I64)
            );
        }

        #[test]
        fn read_axis0_i32() {
            // [[1,2,3],[4,5,6]] product over rows → [4,10,18]
            let got: ArrayD<i64> = make(vec![1i32, 2, 3, 4, 5, 6], &[2, 3])
                .product(&[0], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![4i64, 10, 18]).unwrap()
            );
        }

        #[test]
        fn read_axis1_i32() {
            // [[1,2,3],[4,5,6]] product over cols → [6, 120]
            let got: ArrayD<i64> = make(vec![1i32, 2, 3, 4, 5, 6], &[2, 3])
                .product(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![6i64, 120]).unwrap()
            );
        }

        #[test]
        fn read_all_axes_i32() {
            // product of 1..5 = 24
            let got: ArrayD<i64> = make(vec![1i32, 2, 3, 4], &[2, 2])
                .product(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![24i64]).unwrap());
        }

        // dtype coverage: rows [[2,3],[4,5]] → product [8,15]
        // signed integers widen to i64
        test_reduce_axis0!(i8,  product, in = i8,  out = i64, rows: [[2i8,  3i8 ], [4i8,  5i8 ]], expected: [8i64, 15i64]);
        test_reduce_axis0!(i16, product, in = i16, out = i64, rows: [[2i16, 3i16], [4i16, 5i16]], expected: [8i64, 15i64]);
        test_reduce_axis0!(i32, product, in = i32, out = i64, rows: [[2i32, 3i32], [4i32, 5i32]], expected: [8i64, 15i64]);
        test_reduce_axis0!(i64, product, in = i64, out = i64, rows: [[2i64, 3i64], [4i64, 5i64]], expected: [8i64, 15i64]);
        // unsigned integers widen to u64
        test_reduce_axis0!(u8,  product, in = u8,  out = u64, rows: [[2u8,  3u8 ], [4u8,  5u8 ]], expected: [8u64, 15u64]);
        test_reduce_axis0!(u16, product, in = u16, out = u64, rows: [[2u16, 3u16], [4u16, 5u16]], expected: [8u64, 15u64]);
        test_reduce_axis0!(u32, product, in = u32, out = u64, rows: [[2u32, 3u32], [4u32, 5u32]], expected: [8u64, 15u64]);
        test_reduce_axis0!(u64, product, in = u64, out = u64, rows: [[2u64, 3u64], [4u64, 5u64]], expected: [8u64, 15u64]);
        // floats widen to f64
        test_reduce_axis0!(f32, product, in = f32, out = f64, rows: [[2f32, 3f32], [4f32, 5f32]], expected: [8f64, 15f64]);
        test_reduce_axis0!(f64, product, in = f64, out = f64, rows: [[2f64, 3f64], [4f64, 5f64]], expected: [8f64, 15f64]);
        #[cfg(feature = "half")]
        test_reduce_axis0!(f16, product, in = f16, out = f64,
            rows: [[f16::from_f32(2.0), f16::from_f32(3.0)], [f16::from_f32(4.0), f16::from_f32(5.0)]],
            expected: [8f64, 15f64]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f32, product, in = Complex<f32>, out = Complex<f64>,
            rows: [[Complex { re: 2f32, im: 0f32 }, Complex { re: 3f32, im: 0f32 }],
                   [Complex { re: 4f32, im: 0f32 }, Complex { re: 5f32, im: 0f32 }]],
            expected: [Complex { re: 8f64, im: 0f64 }, Complex { re: 15f64, im: 0f64 }]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f64, product, in = Complex<f64>, out = Complex<f64>,
            rows: [[Complex { re: 2f64, im: 0f64 }, Complex { re: 3f64, im: 0f64 }],
                   [Complex { re: 4f64, im: 0f64 }, Complex { re: 5f64, im: 0f64 }]],
            expected: [Complex { re: 8f64, im: 0f64 }, Complex { re: 15f64, im: 0f64 }]);
    }

    // -----------------------------------------------------------------------
    // all
    // -----------------------------------------------------------------------

    mod all {
        use super::*;

        #[test]
        fn output_dtype_is_bool() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(6), &[2, 3]);
            assert_eq!(
                a.all(&[0], false).dtype().try_to_scalar(),
                Some(DtypeScalarKind::Bool)
            );
        }

        #[test]
        fn read_axis0_with_zero_i32() {
            // [[1,0,1],[1,1,1]] all over rows → [true, false, true]
            let got: ArrayD<bool> = make(vec![1i32, 0, 1, 1, 1, 1], &[2, 3])
                .all(&[0], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![true, false, true]).unwrap()
            );
        }

        #[test]
        fn read_axis1_with_zero_i32() {
            // [[1,1,1],[0,1,1]] all over cols → [true, false]
            let got: ArrayD<bool> = make(vec![1i32, 1, 1, 0, 1, 1], &[2, 3])
                .all(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![true, false]).unwrap()
            );
        }

        #[test]
        fn read_all_axes_true_i32() {
            let got: ArrayD<bool> = make(vec![1i32, 2, 3, 4], &[2, 2])
                .all(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![true]).unwrap());
        }

        #[test]
        fn read_all_axes_false_i32() {
            let got: ArrayD<bool> = make(vec![1i32, 0, 1, 1], &[2, 2])
                .all(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![false]).unwrap());
        }

        #[test]
        fn keepdims_read_axis0_i32() {
            // [[1,0,1],[1,1,1]] all over rows with keepdims → [[true,false,true]] shape [1,3]
            let got: ArrayD<bool> = make(vec![1i32, 0, 1, 1, 1, 1], &[2, 3])
                .all(&[0], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![1, 3], vec![true, false, true]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_axis1_i32() {
            // [[1,1,1],[0,1,1]] all over cols with keepdims → [[true],[false]] shape [2,1]
            let got: ArrayD<bool> = make(vec![1i32, 1, 1, 0, 1, 1], &[2, 3])
                .all(&[1], true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2, 1], vec![true, false]).unwrap()
            );
        }

        // dtype coverage: rows [[1,0],[1,1]] → all [true,false]
        test_reduce_axis0!(i8,   all, in = i8,   out = bool, rows: [[1i8,  0i8 ], [1i8,  1i8 ]], expected: [true, false]);
        test_reduce_axis0!(u8,   all, in = u8,   out = bool, rows: [[1u8,  0u8 ], [1u8,  1u8 ]], expected: [true, false]);
        test_reduce_axis0!(i16,  all, in = i16,  out = bool, rows: [[1i16, 0i16], [1i16, 1i16]], expected: [true, false]);
        test_reduce_axis0!(u16,  all, in = u16,  out = bool, rows: [[1u16, 0u16], [1u16, 1u16]], expected: [true, false]);
        test_reduce_axis0!(i32,  all, in = i32,  out = bool, rows: [[1i32, 0i32], [1i32, 1i32]], expected: [true, false]);
        test_reduce_axis0!(u32,  all, in = u32,  out = bool, rows: [[1u32, 0u32], [1u32, 1u32]], expected: [true, false]);
        test_reduce_axis0!(i64,  all, in = i64,  out = bool, rows: [[1i64, 0i64], [1i64, 1i64]], expected: [true, false]);
        test_reduce_axis0!(u64,  all, in = u64,  out = bool, rows: [[1u64, 0u64], [1u64, 1u64]], expected: [true, false]);
        test_reduce_axis0!(f32,  all, in = f32,  out = bool, rows: [[1f32, 0f32], [1f32, 1f32]], expected: [true, false]);
        test_reduce_axis0!(f64,  all, in = f64,  out = bool, rows: [[1f64, 0f64], [1f64, 1f64]], expected: [true, false]);
        test_reduce_axis0!(bool, all, in = bool, out = bool, rows: [[true, false], [true, true]], expected: [true, false]);
        #[cfg(feature = "half")]
        test_reduce_axis0!(f16, all, in = f16, out = bool,
            rows: [[f16::from_f32(1.0), f16::from_f32(0.0)], [f16::from_f32(1.0), f16::from_f32(1.0)]],
            expected: [true, false]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f32, all, in = Complex<f32>, out = bool,
            rows: [[Complex { re: 1f32, im: 0f32 }, Complex { re: 0f32, im: 0f32 }],
                   [Complex { re: 1f32, im: 0f32 }, Complex { re: 1f32, im: 0f32 }]],
            expected: [true, false]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f64, all, in = Complex<f64>, out = bool,
            rows: [[Complex { re: 1f64, im: 0f64 }, Complex { re: 0f64, im: 0f64 }],
                   [Complex { re: 1f64, im: 0f64 }, Complex { re: 1f64, im: 0f64 }]],
            expected: [true, false]);
    }

    // -----------------------------------------------------------------------
    // any
    // -----------------------------------------------------------------------

    mod any {
        use super::*;

        #[test]
        fn output_dtype_is_bool() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(6), &[2, 3]);
            assert_eq!(
                a.any(&[0], false).dtype().try_to_scalar(),
                Some(DtypeScalarKind::Bool)
            );
        }

        #[test]
        fn read_axis0_i32() {
            // [[0,0,1],[0,0,0]] any over rows → [false, false, true]
            let got: ArrayD<bool> = make(vec![0i32, 0, 1, 0, 0, 0], &[2, 3])
                .any(&[0], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![false, false, true]).unwrap()
            );
        }

        #[test]
        fn read_axis1_i32() {
            // [[0,0,0],[0,0,1]] any over cols → [false, true]
            let got: ArrayD<bool> = make(vec![0i32, 0, 0, 0, 0, 1], &[2, 3])
                .any(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![false, true]).unwrap()
            );
        }

        #[test]
        fn read_all_axes_false_i32() {
            let got: ArrayD<bool> = make(vec![0i32, 0, 0, 0], &[2, 2])
                .any(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![false]).unwrap());
        }

        #[test]
        fn read_all_axes_true_i32() {
            let got: ArrayD<bool> = make(vec![0i32, 0, 0, 1], &[2, 2])
                .any(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![true]).unwrap());
        }

        // dtype coverage: rows [[0,0],[1,0]] → any [true,false]
        test_reduce_axis0!(i8,   any, in = i8,   out = bool, rows: [[0i8,  0i8 ], [1i8,  0i8 ]], expected: [true, false]);
        test_reduce_axis0!(u8,   any, in = u8,   out = bool, rows: [[0u8,  0u8 ], [1u8,  0u8 ]], expected: [true, false]);
        test_reduce_axis0!(i16,  any, in = i16,  out = bool, rows: [[0i16, 0i16], [1i16, 0i16]], expected: [true, false]);
        test_reduce_axis0!(u16,  any, in = u16,  out = bool, rows: [[0u16, 0u16], [1u16, 0u16]], expected: [true, false]);
        test_reduce_axis0!(i32,  any, in = i32,  out = bool, rows: [[0i32, 0i32], [1i32, 0i32]], expected: [true, false]);
        test_reduce_axis0!(u32,  any, in = u32,  out = bool, rows: [[0u32, 0u32], [1u32, 0u32]], expected: [true, false]);
        test_reduce_axis0!(i64,  any, in = i64,  out = bool, rows: [[0i64, 0i64], [1i64, 0i64]], expected: [true, false]);
        test_reduce_axis0!(u64,  any, in = u64,  out = bool, rows: [[0u64, 0u64], [1u64, 0u64]], expected: [true, false]);
        test_reduce_axis0!(f32,  any, in = f32,  out = bool, rows: [[0f32, 0f32], [1f32, 0f32]], expected: [true, false]);
        test_reduce_axis0!(f64,  any, in = f64,  out = bool, rows: [[0f64, 0f64], [1f64, 0f64]], expected: [true, false]);
        test_reduce_axis0!(bool, any, in = bool, out = bool, rows: [[false, false], [true, false]], expected: [true, false]);
        #[cfg(feature = "half")]
        test_reduce_axis0!(f16, any, in = f16, out = bool,
            rows: [[f16::from_f32(0.0), f16::from_f32(0.0)], [f16::from_f32(1.0), f16::from_f32(0.0)]],
            expected: [true, false]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f32, any, in = Complex<f32>, out = bool,
            rows: [[Complex { re: 0f32, im: 0f32 }, Complex { re: 0f32, im: 0f32 }],
                   [Complex { re: 1f32, im: 0f32 }, Complex { re: 0f32, im: 0f32 }]],
            expected: [true, false]);
        #[cfg(feature = "num-complex")]
        test_reduce_axis0!(complex_f64, any, in = Complex<f64>, out = bool,
            rows: [[Complex { re: 0f64, im: 0f64 }, Complex { re: 0f64, im: 0f64 }],
                   [Complex { re: 1f64, im: 0f64 }, Complex { re: 0f64, im: 0f64 }]],
            expected: [true, false]);
    }

    // -----------------------------------------------------------------------
    // argmax
    // -----------------------------------------------------------------------

    mod argmax {
        use super::*;

        // --- shape ---

        #[test]
        fn shape_axis0() {
            // [3,4] reduced on axis 0 → [4]
            assert_eq!(make(seq(12), &[3, 4]).argmax(0, false).shape(), &[4]);
        }

        #[test]
        fn shape_axis1() {
            // [3,4] reduced on axis 1 → [3]
            assert_eq!(make(seq(12), &[3, 4]).argmax(1, false).shape(), &[3]);
        }

        #[test]
        fn shape_1d() {
            assert_eq!(make(seq(5), &[5]).argmax(0, false).shape(), &[] as &[u64]);
        }

        #[test]
        fn keepdims_shape_axis0() {
            assert_eq!(make(seq(12), &[3, 4]).argmax(0, true).shape(), &[1, 4]);
        }

        #[test]
        fn keepdims_shape_axis1() {
            assert_eq!(make(seq(12), &[3, 4]).argmax(1, true).shape(), &[3, 1]);
        }

        #[test]
        fn keepdims_shape_middle_3d() {
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            assert_eq!(a.argmax(1, true).shape(), &[2, 1, 4]);
        }

        // --- output dtype is always u64 ---

        #[test]
        fn output_dtype_is_u64() {
            use crate::dtype::DtypeScalarKind;
            assert_eq!(
                make(seq(6), &[2, 3])
                    .argmax(0, false)
                    .dtype()
                    .try_to_scalar(),
                Some(DtypeScalarKind::U64)
            );
        }

        // --- values ---

        #[test]
        fn read_axis0_i32() {
            // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → argmax per col = [2,2,2,2]
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![4], vec![2u64, 2, 2, 2]).unwrap()
            );
        }

        #[test]
        fn read_axis1_i32() {
            // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → argmax per row = [3,3,3]
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmax(1, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![3u64, 3, 3]).unwrap()
            );
        }

        #[test]
        fn read_axis0_not_last_row() {
            // [[5,1],[2,8],[3,4]] → argmax per col: col0→0(5), col1→1(8)
            let got: ArrayD<u64> = make(vec![5i32, 1, 2, 8, 3, 4], &[3, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn read_1d() {
            // [3,1,4,1,5] → argmax = 4
            let got: ArrayD<u64> = make(vec![3i32, 1, 4, 1, 5], &[5])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![4u64]).unwrap());
        }

        #[test]
        fn read_3d_middle_axis() {
            // a[i,j,k] = i*12+j*4+k; max is always at j=2 → argmax=2
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            let got: ArrayD<u64> = a.argmax(1, false).data().to_ndarray().unwrap();
            let expected = vec![2u64; 8];
            assert_eq!(got, ArrayD::from_shape_vec(vec![2, 4], expected).unwrap());
        }

        #[test]
        fn keepdims_read_axis0_i32() {
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmax(0, true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![1, 4], vec![2u64, 2, 2, 2]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_axis1_i32() {
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmax(1, true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3, 1], vec![3u64, 3, 3]).unwrap()
            );
        }

        // --- dtype coverage ---

        #[test]
        fn dtype_i8() {
            let got: ArrayD<u64> = make(vec![1i8, 4i8, 3i8, 2i8], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            // col0: max(1,3)=3 at idx 1; col1: max(4,2)=4 at idx 0
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_u8() {
            let got: ArrayD<u64> = make(vec![1u8, 4u8, 3u8, 2u8], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_i16() {
            let got: ArrayD<u64> = make(vec![1i16, 4i16, 3i16, 2i16], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_u16() {
            let got: ArrayD<u64> = make(vec![1u16, 4u16, 3u16, 2u16], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_u32() {
            let got: ArrayD<u64> = make(vec![1u32, 4u32, 3u32, 2u32], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_i64() {
            let got: ArrayD<u64> = make(vec![1i64, 4i64, 3i64, 2i64], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_u64() {
            let got: ArrayD<u64> = make(vec![1u64, 4u64, 3u64, 2u64], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_f32() {
            let got: ArrayD<u64> = make(vec![1f32, 4f32, 3f32, 2f32], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_f64() {
            let got: ArrayD<u64> = make(vec![1f64, 4f64, 3f64, 2f64], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[test]
        fn dtype_bool() {
            // [[false, true], [true, false]] → argmax per col: col0→1(true), col1→0(true)
            let got: ArrayD<u64> = make(vec![false, true, true, false], &[2, 2])
                .argmax(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[cfg(feature = "half")]
        #[test]
        fn dtype_f16() {
            let got: ArrayD<u64> = make(
                vec![
                    f16::from_f32(1.0),
                    f16::from_f32(4.0),
                    f16::from_f32(3.0),
                    f16::from_f32(2.0),
                ],
                &[2, 2],
            )
            .argmax(0, false)
            .data()
            .to_ndarray()
            .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        // --- error cases ---

        #[test]
        #[should_panic]
        fn error_axis_out_of_bounds() {
            make(seq(6), &[2, 3]).argmax(2, false);
        }
    }

    // -----------------------------------------------------------------------
    // argmin
    // -----------------------------------------------------------------------

    mod argmin {
        use super::*;

        // --- shape ---

        #[test]
        fn shape_axis0() {
            assert_eq!(make(seq(12), &[3, 4]).argmin(0, false).shape(), &[4]);
        }

        #[test]
        fn shape_axis1() {
            assert_eq!(make(seq(12), &[3, 4]).argmin(1, false).shape(), &[3]);
        }

        #[test]
        fn keepdims_shape_axis0() {
            assert_eq!(make(seq(12), &[3, 4]).argmin(0, true).shape(), &[1, 4]);
        }

        #[test]
        fn keepdims_shape_axis1() {
            assert_eq!(make(seq(12), &[3, 4]).argmin(1, true).shape(), &[3, 1]);
        }

        // --- output dtype is always u64 ---

        #[test]
        fn output_dtype_is_u64() {
            use crate::dtype::DtypeScalarKind;
            assert_eq!(
                make(seq(6), &[2, 3])
                    .argmin(0, false)
                    .dtype()
                    .try_to_scalar(),
                Some(DtypeScalarKind::U64)
            );
        }

        // --- values ---

        #[test]
        fn read_axis0_i32() {
            // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → argmin per col = [0,0,0,0]
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![4], vec![0u64, 0, 0, 0]).unwrap()
            );
        }

        #[test]
        fn read_axis1_i32() {
            // [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → argmin per row = [0,0,0]
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmin(1, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![0u64, 0, 0]).unwrap()
            );
        }

        #[test]
        fn read_axis0_not_first_row() {
            // [[5,8],[2,1],[3,4]] → argmin per col: col0→1(2), col1→1(1)
            let got: ArrayD<u64> = make(vec![5i32, 8, 2, 1, 3, 4], &[3, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 1]).unwrap());
        }

        #[test]
        fn read_1d() {
            // [3,1,4,1,5] → argmin = 1
            let got: ArrayD<u64> = make(vec![3i32, 1, 4, 1, 5], &[5])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![1u64]).unwrap());
        }

        #[test]
        fn read_3d_middle_axis() {
            // a[i,j,k] = i*12+j*4+k; min is always at j=0 → argmin=0
            let nd = ArrayD::from_shape_vec(vec![2, 3, 4], seq(24)).unwrap();
            let a = Array::from_ndarray(&nd, arr_params(&[2, 3, 4])).unwrap();
            let got: ArrayD<u64> = a.argmin(1, false).data().to_ndarray().unwrap();
            let expected = vec![0u64; 8];
            assert_eq!(got, ArrayD::from_shape_vec(vec![2, 4], expected).unwrap());
        }

        #[test]
        fn keepdims_read_axis0_i32() {
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmin(0, true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![1, 4], vec![0u64, 0, 0, 0]).unwrap()
            );
        }

        #[test]
        fn keepdims_read_axis1_i32() {
            let got: ArrayD<u64> = make(seq(12), &[3, 4])
                .argmin(1, true)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3, 1], vec![0u64, 0, 0]).unwrap()
            );
        }

        // --- dtype coverage ---

        #[test]
        fn dtype_i8() {
            let got: ArrayD<u64> = make(vec![1i8, 4i8, 3i8, 2i8], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            // col0: min(1,3)=1 at idx 0; col1: min(4,2)=2 at idx 1
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn dtype_u8() {
            let got: ArrayD<u64> = make(vec![1u8, 4u8, 3u8, 2u8], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn dtype_i16() {
            let got: ArrayD<u64> = make(vec![1i16, 4i16, 3i16, 2i16], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn dtype_u32() {
            let got: ArrayD<u64> = make(vec![1u32, 4u32, 3u32, 2u32], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn dtype_f32() {
            let got: ArrayD<u64> = make(vec![1f32, 4f32, 3f32, 2f32], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn dtype_f64() {
            let got: ArrayD<u64> = make(vec![1f64, 4f64, 3f64, 2f64], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        #[test]
        fn dtype_bool() {
            // [[true, false], [false, true]] → argmin per col: col0→1(false), col1→0(false)
            let got: ArrayD<u64> = make(vec![true, false, false, true], &[2, 2])
                .argmin(0, false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![1u64, 0]).unwrap());
        }

        #[cfg(feature = "half")]
        #[test]
        fn dtype_f16() {
            let got: ArrayD<u64> = make(
                vec![
                    f16::from_f32(1.0),
                    f16::from_f32(4.0),
                    f16::from_f32(3.0),
                    f16::from_f32(2.0),
                ],
                &[2, 2],
            )
            .argmin(0, false)
            .data()
            .to_ndarray()
            .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![2], vec![0u64, 1]).unwrap());
        }

        // --- error cases ---

        #[test]
        #[should_panic]
        fn error_axis_out_of_bounds() {
            make(seq(6), &[2, 3]).argmin(2, false);
        }
    }

    // -----------------------------------------------------------------------
    // mean
    // -----------------------------------------------------------------------

    mod mean {
        use super::*;

        #[test]
        fn output_dtype_i32_to_f64() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(6), &[2, 3]);
            assert_eq!(
                a.mean(&[0], false).dtype().try_to_scalar(),
                Some(DtypeScalarKind::F64)
            );
        }

        #[test]
        fn read_axis0_f64() {
            // [[0,2,4],[2,4,6]] mean over rows → [1,3,5]
            let got: ArrayD<f64> = make(vec![0.0f64, 2.0, 4.0, 2.0, 4.0, 6.0], &[2, 3])
                .mean(&[0], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![3], vec![1.0f64, 3.0, 5.0]).unwrap()
            );
        }

        #[test]
        fn read_axis1_f64() {
            // [[0,2,4],[2,4,6]] mean over cols → [2,4]
            let got: ArrayD<f64> = make(vec![0.0f64, 2.0, 4.0, 2.0, 4.0, 6.0], &[2, 3])
                .mean(&[1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![2.0f64, 4.0]).unwrap()
            );
        }

        #[test]
        fn read_all_axes_i32() {
            // mean of [0..5] = 2.5
            let got: ArrayD<f64> = make(vec![0i32, 1, 2, 3, 4, 5], &[2, 3])
                .mean(&[0, 1], false)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![2.5f64]).unwrap());
        }

        #[test]
        fn keepdims_shape() {
            assert_eq!(make(seq(12), &[3, 4]).mean(&[0], true).shape(), &[1, 4]);
        }
    }

    // -----------------------------------------------------------------------
    // var
    // -----------------------------------------------------------------------

    mod var {
        use super::*;

        #[test]
        fn output_dtype_i32_to_f64() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(4), &[2, 2]);
            assert_eq!(
                a.var(&[0], false, 0.0).dtype().try_to_scalar(),
                Some(DtypeScalarKind::F64)
            );
        }

        #[test]
        fn population_variance_1d() {
            // [1,3]: mean=2, var_0 = 1.0
            let got: ArrayD<f64> = make(vec![1.0f64, 3.0], &[1, 2])
                .var(&[1], false, 0.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![1.0f64]).unwrap());
        }

        #[test]
        fn sample_variance_1d() {
            // [1,3]: mean=2, var_1 = 2.0
            let got: ArrayD<f64> = make(vec![1.0f64, 3.0], &[1, 2])
                .var(&[1], false, 1.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![2.0f64]).unwrap());
        }

        #[test]
        fn read_axis0_f64() {
            // [[1,3],[3,7]]: col means=[2,5], var_0 per col = [1,4]
            let got: ArrayD<f64> = make(vec![1.0f64, 3.0, 3.0, 7.0], &[2, 2])
                .var(&[0], false, 0.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![1.0f64, 4.0]).unwrap()
            );
        }

        #[test]
        fn read_axis1_f64() {
            // [[1,3],[3,7]]: row means=[2,5], var_0 per row = [1,4]
            let got: ArrayD<f64> = make(vec![1.0f64, 3.0, 3.0, 7.0], &[2, 2])
                .var(&[1], false, 0.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![1.0f64, 4.0]).unwrap()
            );
        }

        #[test]
        fn ddof_exceeds_n_gives_nan() {
            // ddof=2 with n=1 → NAN
            let got: ArrayD<f64> = make(vec![1.0f64], &[1, 1])
                .var(&[1], false, 2.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert!(got.iter().all(|v| v.is_nan()));
        }
    }

    // -----------------------------------------------------------------------
    // std
    // -----------------------------------------------------------------------

    mod std {
        use super::*;

        #[test]
        fn output_dtype_i32_to_f64() {
            use crate::dtype::DtypeScalarKind;
            let a: Array<_> = make(seq(4), &[2, 2]);
            assert_eq!(
                a.std(&[0], false, 0.0).dtype().try_to_scalar(),
                Some(DtypeScalarKind::F64)
            );
        }

        #[test]
        fn population_std_1d() {
            // [1,3]: mean=2, std_0 = 1.0
            let got: ArrayD<f64> = make(vec![1.0f64, 3.0], &[1, 2])
                .std(&[1], false, 0.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![1.0f64]).unwrap());
        }

        #[test]
        fn read_axis0_f64() {
            // [[1,3],[3,7]]: var_0 per col = [1,4], std = [1,2]
            let got: ArrayD<f64> = make(vec![1.0f64, 3.0, 3.0, 7.0], &[2, 2])
                .std(&[0], false, 0.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(
                got,
                ArrayD::from_shape_vec(vec![2], vec![1.0f64, 2.0]).unwrap()
            );
        }

        #[test]
        fn classic_8_elements() {
            // [2,4,4,4,5,5,7,9]: mean=5, var=4, std=2 (ddof=0)
            let got: ArrayD<f64> = make(vec![2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[1, 8])
                .std(&[1], false, 0.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![2.0f64]).unwrap());
        }

        #[test]
        fn ddof_exceeds_n_gives_nan() {
            let got: ArrayD<f64> = make(vec![1.0f64], &[1, 1])
                .std(&[1], false, 2.0)
                .data()
                .to_ndarray()
                .unwrap();
            assert!(got.iter().all(|v| v.is_nan()));
        }
    }
}
