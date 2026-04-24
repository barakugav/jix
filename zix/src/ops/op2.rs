use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{f16, Complex, Dtype, Itemsize};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec};
use crate::util::DimArray;

pub(crate) trait Op2Kernel {
    fn apply(&self, data: Op2KernelData, input_dtypes: (&Dtype, &Dtype)) -> Result<()>;

    fn output_dtype(&self, input_dtypes: (&Dtype, &Dtype)) -> Result<Dtype>;
}

pub(crate) struct Op2KernelData<'a> {
    src_a_data: *const u8,
    src_b_data: *const u8,
    dst_data: *mut u8, // potentially an alias to src_a_data or src_b_data
    nitems: usize,
    src_a_itemsize: Itemsize,
    src_b_itemsize: Itemsize,
    dst_itemsize: Itemsize,
    phantom: std::marker::PhantomData<&'a ()>,
}
#[allow(unused)]
impl<'a> Op2KernelData<'a> {
    pub(crate) fn read_as_bytes(&mut self) -> Option<(&[u8], &[u8])> {
        (self.nitems > 0).then(|| unsafe {
            (
                std::slice::from_raw_parts(self.src_a_data, self.src_a_itemsize as usize),
                std::slice::from_raw_parts(self.src_b_data, self.src_b_itemsize as usize),
            )
        })
    }

    pub(crate) unsafe fn read<T1, T2>(&mut self) -> Option<(T1, T2)> {
        debug_assert_eq!(self.src_a_itemsize as usize, size_of::<T1>());
        debug_assert_eq!(self.src_b_itemsize as usize, size_of::<T2>());
        (self.nitems > 0).then(|| unsafe {
            (
                self.src_a_data.cast::<T1>().read(),
                self.src_b_data.cast::<T2>().read(),
            )
        })
    }

    pub(crate) unsafe fn read_bulk<T1, T2, const N: usize>(
        &mut self,
    ) -> Option<([T1; N], [T2; N])> {
        debug_assert_eq!(self.src_a_itemsize as usize, size_of::<T1>());
        debug_assert_eq!(self.src_b_itemsize as usize, size_of::<T2>());
        (self.nitems >= N).then(|| unsafe {
            (
                self.src_a_data.cast::<[T1; N]>().read(),
                self.src_b_data.cast::<[T2; N]>().read(),
            )
        })
    }

    pub(crate) unsafe fn write_bytes(&mut self, data: &[u8]) {
        assert_eq!(self.dst_itemsize as usize, data.len());
        assert!(self.nitems > 0);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.dst_data, data.len());
        }
        self.nitems -= 1;
        self.src_a_data = unsafe { self.src_a_data.add(self.src_a_itemsize as usize) };
        self.src_b_data = unsafe { self.src_b_data.add(self.src_b_itemsize as usize) };
        self.dst_data = unsafe { self.dst_data.add(self.dst_itemsize as usize) };
    }

    pub(crate) unsafe fn write<T>(&mut self, data: T) {
        debug_assert_eq!(self.dst_itemsize as usize, size_of::<T>());
        assert!(self.nitems > 0);
        unsafe {
            self.dst_data.cast::<T>().write(data);
        }
        self.nitems -= 1;
        self.src_a_data = unsafe { self.src_a_data.add(self.src_a_itemsize as usize) };
        self.src_b_data = unsafe { self.src_b_data.add(self.src_b_itemsize as usize) };
        self.dst_data = unsafe { self.dst_data.add(self.dst_itemsize as usize) };
    }

    pub(crate) unsafe fn write_bulk<T, const N: usize>(&mut self, data: [T; N]) {
        debug_assert_eq!(self.dst_itemsize as usize, size_of::<T>());
        assert!(self.nitems >= N);
        unsafe {
            self.dst_data.cast::<[T; N]>().write(data);
        }
        self.nitems -= N;
        self.src_a_data = unsafe { self.src_a_data.add(self.src_a_itemsize as usize * N) };
        self.src_b_data = unsafe { self.src_b_data.add(self.src_b_itemsize as usize * N) };
        self.dst_data = unsafe { self.dst_data.add(self.dst_itemsize as usize * N) };
    }
}

pub(crate) struct Op2<Op, S1, S2> {
    op: Op,

    a: Array<S1>,
    b: Array<S2>,

    output_dtype: Dtype,
    shape: DimArray<u64>,
}
impl<Op, S1, S2> Op2<Op, S1, S2> {
    pub(crate) fn new(op: Op, a: Array<S1>, b: Array<S2>) -> Result<Self>
    where
        Op: Op2Kernel,
        S1: ArrayStorage,
        S2: ArrayStorage,
    {
        let output_dtype = op.output_dtype((a.dtype(), b.dtype()))?;
        ensure!(
            a.shape() == b.shape(),
            InvalidArgument,
            "shape mismatch between a {:?} and b {:?} in Op2",
            a.shape(),
            b.shape()
        );
        Ok(Self {
            op,
            output_dtype,
            shape: a.shape().try_into().unwrap(),
            a,
            b,
        })
    }
}
impl<Op, S1, S2> ArrayStorage for Op2<Op, S1, S2>
where
    Op: Op2Kernel,
    S1: ArrayStorage,
    S2: ArrayStorage,
{
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape, index)?;
        let nitems = check_get_buffer_size(index, &self.output_dtype, buf)?;

        let (a_dtype, b_dtype, dst_dtype) = (self.a.dtype(), self.b.dtype(), &self.output_dtype);
        let (src_a_itemsize, src_b_itemsize, dst_itemsize) = (
            a_dtype.itemsize() as usize,
            b_dtype.itemsize() as usize,
            dst_dtype.itemsize() as usize,
        );

        let a_in_place = src_a_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(a_dtype.alignment().as_usize());
        let b_in_place = src_b_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(b_dtype.alignment().as_usize());
        let mut a_tmp_buf;
        let mut b_tmp_buf;
        let (a_buf, b_buf, dst) = if a_in_place {
            b_tmp_buf = context.tmp_buf(nitems * src_b_itemsize, b_dtype.alignment());
            let b_tmp_buf = b_tmp_buf.as_mut_slice();
            let ptr = buf.as_mut_ptr();
            (buf, b_tmp_buf, ptr)
        } else if b_in_place {
            a_tmp_buf = context.tmp_buf(nitems * src_a_itemsize, a_dtype.alignment());
            let a_tmp_buf = a_tmp_buf.as_mut_slice();
            let dst = buf.as_mut_ptr();
            (a_tmp_buf, buf, dst)
        } else {
            a_tmp_buf = context.tmp_buf(nitems * src_a_itemsize, a_dtype.alignment());
            let a_tmp_buf = a_tmp_buf.as_mut_slice();

            b_tmp_buf = context.tmp_buf(nitems * src_b_itemsize, b_dtype.alignment());
            let b_tmp_buf = b_tmp_buf.as_mut_slice();

            (a_tmp_buf, b_tmp_buf, buf.as_mut_ptr())
        };

        self.a.storage.read_data(index, a_buf, context)?;
        self.b.storage.read_data(index, b_buf, context)?;

        self.op.apply(
            Op2KernelData {
                src_a_data: a_buf.as_ptr(),
                src_b_data: b_buf.as_ptr(),
                dst_data: dst,
                nitems,
                src_a_itemsize: a_dtype.itemsize(),
                src_b_itemsize: b_dtype.itemsize(),
                dst_itemsize: dst_dtype.itemsize(),
                phantom: std::marker::PhantomData,
            },
            (a_dtype, b_dtype),
        )
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.output_dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.a.storage.spec()
    }
}

macro_rules! define_op2 {
    (
        $Name:ident,
        $NameKernel:ident,
        core_op = ($op_trait:ident, $op_fn:ident),
        $($kernel_args:tt)*
    ) => {
        define_op2!($Name, $NameKernel, $($kernel_args)*);

        impl<S1, S2> core::ops::$op_trait<Array<S2>> for Array<S1>
        where
            S1: ArrayStorage,
            S2: ArrayStorage,
        {
            type Output = Array<$Name<S1, S2>>;
            #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $op_fn(self, b: Array<S2>) -> Array<$Name<S1, S2>> {
                let op = $Name::new(self, b).unwrap();
                Array::from_storage(op)
            }
        }

        impl<S, T> core::ops::$op_trait<T> for Array<S>
        where
            S: ArrayStorage,
            T: crate::dtype::Dtyped,
        {
            type Output = Array<$Name<S, crate::storage::Scalar<T>>>;
            #[doc = concat!("Applies the [`", stringify!($Name), "`] operation by broadcasting the scalar, see the op struct docs for details.")]
            #[track_caller]
            fn $op_fn(self, b: T) -> Array<$Name<S, crate::storage::Scalar<T>>> {
                let b = Array::from_scalar(b, self.shape()).unwrap();
                let op = $Name::new(self, b).unwrap();
                Array::from_storage(op)
            }
        }
    };

    (
        $Name:ident,
        $NameKernel:ident,
        $($kernel_args:tt)*
    ) => {
        pub struct $Name<S1, S2>(crate::ops::op2::Op2<$NameKernel, S1, S2>);
        impl<S1, S2> $Name<S1, S2> {
            pub fn new(a: crate::Array<S1>, b: crate::Array<S2>) -> crate::error::Result<Self>
            where
                S1: crate::storage::ArrayStorage,
                S2: crate::storage::ArrayStorage,
            {
                Ok(Self(crate::ops::op2::Op2::new($NameKernel, a, b)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S1, S2> where S1: crate::storage::ArrayStorage, S2: crate::storage::ArrayStorage);

        crate::ops::op2::define_op2_kernel!($NameKernel, $($kernel_args)*);
    };
}
macro_rules! define_op2_kernel {
    (
        $NameKernel:ident,
        |$a:ident, $b:ident| $body:expr,
        [$($scalar:tt),* $(,)?],
        output_type = "same"
    ) => {
        crate::ops::op2::define_op2_kernel!(
            $NameKernel,
            |$a, $b| $body,
            [$($scalar => $scalar),*]
        );
    };

    (
        $NameKernel:ident,
        |$a:ident, $b:ident| $body:expr,
        [$($scalar:tt),* $(,)?],
        output_type = $output_type:tt
    ) => {
        crate::ops::op2::define_op2_kernel!(
            $NameKernel,
            |$a, $b| $body,
            [$($scalar => $output_type),*]
        );
    };

    (
        $NameKernel:ident,
        |$a:ident, $b:ident| $body:expr,
        [$($input_type:tt => $output_type:tt),* $(,)?]
    ) => {
        struct $NameKernel;
        impl crate::ops::op2::Op2Kernel for $NameKernel {
            fn apply(
                &self,
                mut data: crate::ops::op2::Op2KernelData,
                input_dtypes: (&crate::dtype::Dtype, &crate::dtype::Dtype),
            ) -> crate::error::Result<()> {
                macro_rules! apply_loop_impl {
                    ($input_type2:ty, $output_type2:ty) => {{
                        unsafe {
                            while let Some((src_a, src_b)) = data.read_bulk::<$input_type2, $input_type2, { crate::ops::common::BULK }>() {
                                let mut dst: [std::mem::MaybeUninit<$output_type2>; crate::ops::common::BULK]
                                    = std::mem::transmute(std::mem::MaybeUninit::<[$output_type2; crate::ops::common::BULK]>::uninit());
                                for i in 0..crate::ops::common::BULK {
                                    dst[i].write({
                                        let $a = src_a[i];
                                        let $b = src_b[i];
                                        $body
                                    });
                                }
                                data.write_bulk(dst);
                            }
                            while let Some((src_a, src_b)) = data.read::<$input_type2, $input_type2>() {
                                let mut dst = std::mem::MaybeUninit::<$output_type2>::uninit();
                                dst.write({
                                    let $a = src_a;
                                    let $b = src_b;
                                    $body
                                });
                                data.write(dst);
                            }
                        }
                        return Ok(());
                    }};
                }
                macro_rules! apply_loop {
                    (f16, $output_type2:ty) => {
                        #[cfg(feature = "half")]
                        apply_loop_impl!(f16, $output_type2)
                    };
                    ((Complex<f32>), $output_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(crate::dtype::Complex<f32>, $output_type2)
                    };
                    ((Complex<f64>), $output_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(crate::dtype::Complex<f64>, $output_type2)
                    };
                    ($input_type2:ty, $output_type2:ty) => {
                        apply_loop_impl!($input_type2, $output_type2)
                    };
                }

                debug_assert_eq!(input_dtypes.0, input_dtypes.1);
                let input_dtype = input_dtypes.0;

                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($input_type)) => {
                        apply_loop!($input_type, $output_type)
                    },)*
                    _ => {}
                }
                crate::error::bail!(UnsupportedDtype, "op not supported for dtype {input_dtype:#?}");
            }

            fn output_dtype(
                &self,
                input_dtypes: (&crate::dtype::Dtype, &crate::dtype::Dtype),
            ) -> crate::error::Result<crate::dtype::Dtype> {
                let (a_dtype, b_dtype) = input_dtypes;
                crate::error::ensure!(a_dtype == b_dtype, UnsupportedDtype, "dtype mismatch");

                let input_dtype = a_dtype;

                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($input_type)) => {
                        return Ok(<$output_type as crate::dtype::Dtyped>::DTYPE);
                    },)*
                    _ => {},

                };
                crate::error::bail!(UnsupportedDtype, "op not supported for dtype {input_dtype:#?}");
            }
        }
    };
}

pub(crate) use {define_op2, define_op2_kernel};
define_op2!(
    Add,
    AddKernel,
    core_op = (Add, add),
    |a, b| a + b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = "same"
);
define_op2!(
    Sub,
    SubKernel,
    core_op = (Sub, sub),
    |a, b| a - b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = "same"
);
define_op2!(
    Mul,
    MulKernel,
    core_op = (Mul, mul),
    |a, b| a * b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = "same"
);
define_op2!(
    Div,
    DivKernel,
    core_op = (Div, div),
    |a, b| a / b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = "same"
);
define_op2!(
    Power,
    PowerKernel,
    |a, b| a.powf(b),
    [f32, f64],
    output_type = "same"
);

#[cfg(test)]
mod tests {
    // Generates 5 test functions per (op, dtype).
    // op_safe_strategy() controls the sampling range per type to avoid overflow.
    macro_rules! test_op_dtype {
        ($op:tt, $dtype:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<test_ $dtype _1d>](
                        b_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 4usize
                        ),
                        a_extra_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 4usize
                        ),
                    ) {
                        use crate::array::Array;
                        let b = ndarray::ArrayD::from_shape_vec(vec![4], b_vals).unwrap();
                        let a = ndarray::ArrayD::from_shape_vec(vec![4], a_extra_vals).unwrap() + &b;
                        let za = Array::from_ndarray(&a, crate::util::arr_params(&[4])).unwrap();
                        let zb = Array::from_ndarray(&b, crate::util::arr_params(&[4])).unwrap();
                        let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                        proptest::prop_assert_eq!(actual, &a $op &b);
                    }

                    #[test]
                    fn [<test_ $dtype _1d_multi_block>](
                        b_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize
                        ),
                        a_extra_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize
                        ),
                    ) {
                        use crate::array::Array;
                        let b = ndarray::ArrayD::from_shape_vec(vec![6], b_vals).unwrap();
                        let a = ndarray::ArrayD::from_shape_vec(vec![6], a_extra_vals).unwrap() + &b;
                        let za = Array::from_ndarray(&a, crate::util::arr_params(&[2])).unwrap();
                        let zb = Array::from_ndarray(&b, crate::util::arr_params(&[2])).unwrap();
                        let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                        proptest::prop_assert_eq!(actual, &a $op &b);
                    }

                    #[test]
                    fn [<test_ $dtype _2d>](
                        b_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize
                        ),
                        a_extra_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize
                        ),
                    ) {
                        use crate::array::Array;
                        let b = ndarray::ArrayD::from_shape_vec(vec![2, 3], b_vals).unwrap();
                        let a = ndarray::ArrayD::from_shape_vec(vec![2, 3], a_extra_vals).unwrap() + &b;
                        let za = Array::from_ndarray(&a, crate::util::arr_params(&[2, 3])).unwrap();
                        let zb = Array::from_ndarray(&b, crate::util::arr_params(&[2, 3])).unwrap();
                        let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                        proptest::prop_assert_eq!(actual, &a $op &b);
                    }

                    #[test]
                    fn [<test_ $dtype _2d_multi_block>](
                        b_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 16usize
                        ),
                        a_extra_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 16usize
                        ),
                    ) {
                        use crate::array::Array;
                        let b = ndarray::ArrayD::from_shape_vec(vec![4, 4], b_vals).unwrap();
                        let a = ndarray::ArrayD::from_shape_vec(vec![4, 4], a_extra_vals).unwrap() + &b;
                        let za = Array::from_ndarray(&a, crate::util::arr_params(&[2, 2])).unwrap();
                        let zb = Array::from_ndarray(&b, crate::util::arr_params(&[2, 2])).unwrap();
                        let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                        proptest::prop_assert_eq!(actual, &a $op &b);
                    }
                }

                // three_arrays: `a = a_extra + b + c` ensures a >= b+c for sub/div.
                // Skipped for size_of < 2 because chaining e.g. mul on i8 overflows:
                // max of (a*b)*c = (a_extra+b+c)*b*c can exceed i8::MAX even with small ranges.
                #[test]
                fn [<test_ $dtype _three_arrays>]() {
                    if size_of::<$dtype>() < 2 {
                        return;
                    }
                    proptest::proptest!(|(
                        c_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 4usize
                        ),
                        b_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 4usize
                        ),
                        a_extra_vals in proptest::collection::vec(
                            <$dtype as crate::util::ScalarStrategy>::op_safe_strategy(), 4usize
                        ),
                    )| {
                        use crate::array::Array;
                        let c = ndarray::ArrayD::from_shape_vec(vec![4], c_vals).unwrap();
                        let b = ndarray::ArrayD::from_shape_vec(vec![4], b_vals).unwrap();
                        let a = ndarray::ArrayD::from_shape_vec(vec![4], a_extra_vals).unwrap() + &b + &c;
                        let za = Array::from_ndarray(&a, crate::util::arr_params(&[4])).unwrap();
                        let zb = Array::from_ndarray(&b, crate::util::arr_params(&[4])).unwrap();
                        let zc = Array::from_ndarray(&c, crate::util::arr_params(&[4])).unwrap();
                        let zab = za $op zb.as_ref();
                        let actual = (zab $op zc).data().to_ndarray::<$dtype>().unwrap();
                        proptest::prop_assert_eq!(actual, (&(&a $op &b) $op &c));
                    });
                }
            }
        };
    }

    // Creates a module named $mod_name with one test set per dtype, all using $op.
    // Optional trailing groups add feature-gated dtypes: #[cfg(feature = "...")] [dtype, ...]
    macro_rules! test_op {
        ($mod_name:ident, $op:tt, [$($dtype:ident),+] $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+])*) => {
            mod $mod_name {
                // Import feature-gated type aliases defined in the parent tests module.
                $(#[cfg($cfg)] use super::{$($cfg_dtype),+};)*
                $(test_op_dtype!($op, $dtype);)+
                $($(
                    #[cfg($cfg)]
                    test_op_dtype!($op, $cfg_dtype);
                )+)*
            }
        };
    }

    test_op!(add, +,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );
    test_op!(sub, -,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );
    test_op!(mul, *,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );
    test_op!(div, /,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );

    // Bring half::f16 into scope under the name `f16` so the macro ident resolves correctly.
    #[cfg(feature = "half")]
    use crate::dtype::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::dtype::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::dtype::Complex<f64>;

    proptest::proptest! {
        #[test]
        fn test_add_mul_scalar(
            vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 100usize)
        ) {
            use crate::array::Array;
            let a = ndarray::ArrayD::from_shape_vec(vec![10, 10], vals).unwrap();
            let za = Array::from_ndarray(&a, crate::util::arr_params(&[10, 10])).unwrap();
            let zb = za * 2.0f32 + 1.0f32;
            let actual = zb.data().to_ndarray::<f32>().unwrap();
            let expected = &a * 2.0 + 1.0;
            proptest::prop_assert_eq!(actual, expected);
        }
    }
}
