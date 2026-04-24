use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{f16, Complex, Dtype, Itemsize};
use crate::error::{check_get_buffer_size, check_get_range, Result};
use crate::ops::common::define_array_op1_method;
use crate::storage::{ArrayStorage, ArrayStorageSpec};
use crate::util::DimArray;

pub(crate) trait Op1Kernel {
    fn apply(&self, data: Op1KernelData, input_dtype: &Dtype) -> Result<()>;

    fn output_dtype(&self, input_dtype: &Dtype) -> Result<Dtype>;
}
pub(crate) struct Op1KernelData<'a> {
    src_data: *const u8,
    dst_data: *mut u8, // potentially an alias to src_data
    nitems: usize,
    src_itemsize: Itemsize,
    dst_itemsize: Itemsize,
    phantom: std::marker::PhantomData<&'a ()>,
}
#[allow(unused)]
impl<'a> Op1KernelData<'a> {
    pub(crate) fn read_as_bytes(&mut self) -> Option<&[u8]> {
        (self.nitems > 0).then(|| unsafe {
            std::slice::from_raw_parts(self.src_data, self.src_itemsize as usize)
        })
    }

    pub(crate) unsafe fn read<T>(&mut self) -> Option<T> {
        debug_assert_eq!(self.src_itemsize as usize, size_of::<T>());
        (self.nitems > 0).then(|| unsafe { self.src_data.cast::<T>().read() })
    }

    pub(crate) unsafe fn read_bulk<T, const N: usize>(&mut self) -> Option<[T; N]> {
        debug_assert_eq!(self.src_itemsize as usize, size_of::<T>());
        (self.nitems >= N).then(|| unsafe { self.src_data.cast::<[T; N]>().read() })
    }

    pub(crate) unsafe fn write_bytes(&mut self, data: &[u8]) {
        assert_eq!(self.dst_itemsize as usize, data.len());
        assert!(self.nitems > 0);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.dst_data, data.len());
        }
        self.nitems -= 1;
        self.src_data = unsafe { self.src_data.add(self.src_itemsize as usize) };
        self.dst_data = unsafe { self.dst_data.add(self.dst_itemsize as usize) };
    }

    pub(crate) unsafe fn write<T>(&mut self, data: T) {
        debug_assert_eq!(self.dst_itemsize as usize, size_of::<T>());
        assert!(self.nitems > 0);
        unsafe {
            self.dst_data.cast::<T>().write(data);
        }
        self.nitems -= 1;
        self.src_data = unsafe { self.src_data.add(self.src_itemsize as usize) };
        self.dst_data = unsafe { self.dst_data.add(self.dst_itemsize as usize) };
    }

    pub(crate) unsafe fn write_bulk<T, const N: usize>(&mut self, data: [T; N]) {
        debug_assert_eq!(self.dst_itemsize as usize, size_of::<T>());
        assert!(self.nitems >= N);
        unsafe {
            self.dst_data.cast::<[T; N]>().write(data);
        }
        self.nitems -= N;
        self.src_data = unsafe { self.src_data.add(self.src_itemsize as usize * N) };
        self.dst_data = unsafe { self.dst_data.add(self.dst_itemsize as usize * N) };
    }
}

pub(crate) struct Op1<Op, S> {
    op: Op,

    array: Array<S>,

    output_dtype: Dtype,
    shape: DimArray<u64>,
}
impl<Op, S> Op1<Op, S> {
    pub(crate) fn new(op: Op, array: Array<S>) -> Result<Self>
    where
        Op: Op1Kernel,
        S: ArrayStorage,
    {
        let output_dtype = op.output_dtype(array.dtype())?;
        Ok(Self {
            op,
            output_dtype,
            shape: array.shape().try_into().unwrap(),
            array,
        })
    }
}
impl<Op, S> ArrayStorage for Op1<Op, S>
where
    Op: Op1Kernel,
    S: ArrayStorage,
{
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape, index)?;
        let nitems = check_get_buffer_size(index, &self.output_dtype, buf)?;

        let (src_dtype, dst_dtype) = (self.array.dtype(), &self.output_dtype);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);

        let in_place = src_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(src_dtype.alignment().as_usize());
        let mut tmp_buf;
        let (read_buf, dst) = if in_place {
            let ptr = buf.as_mut_ptr();
            (buf, ptr)
        } else {
            tmp_buf = context.tmp_buf(nitems * src_itemsize, src_dtype.alignment());
            let tmp_buf = tmp_buf.as_mut_slice();
            (tmp_buf, buf.as_mut_ptr())
        };
        self.array.storage.read_data(index, read_buf, context)?;
        let src = read_buf.as_ptr();

        self.op.apply(
            Op1KernelData {
                src_data: src,
                dst_data: dst,
                nitems,
                src_itemsize: src_dtype.itemsize(),
                dst_itemsize: dst_dtype.itemsize(),
                phantom: std::marker::PhantomData,
            },
            src_dtype,
        )
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.output_dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.array.storage.spec()
    }
}

macro_rules! define_op1 {
    ($Name:ident, $NameKernel:ident, core_op = ($op_trait:ident, $op_fn:ident), $($kernel_args:tt)*) => {
        define_op1!($Name, $NameKernel, $($kernel_args)*);

        impl<S> core::ops::$op_trait for crate::Array<S>
        where
            S: crate::storage::ArrayStorage,
        {
            type Output = crate::Array<$Name<S>>;
            #[track_caller]
            fn $op_fn(self) -> crate::Array<$Name<S>> {
                let op = $Name::new(self).unwrap();
                crate::Array::from_storage(op)
            }
        }
    };

    ($Name:ident, $NameKernel:ident, $($kernel_args:tt)*) => {
        pub struct $Name<S>(crate::ops::op1::Op1<$NameKernel, S>);
        impl<S> $Name<S> {
            pub fn new(array: crate::Array<S>) -> crate::error::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                Ok(Self(crate::ops::op1::Op1::new($NameKernel, array)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        crate::ops::op1::define_op1_kernel!($NameKernel, $($kernel_args)*);
    };
}
macro_rules! define_op1_kernel {
    (
        $NameKernel:ident,
        |$arg:ident| $body:expr,
        [$($scalar:tt),* $(,)?],
        output_type = "same"
    ) => {
        crate::ops::op1::define_op1_kernel! {
            $NameKernel,
            |$arg| $body,
            [$($scalar => $scalar),*]
        }
    };

    (
        $NameKernel:ident,
        |$arg:ident| $body:expr,
        [$($scalar:tt),* $(,)?],
        output_type = $output_type:tt
    ) => {
        crate::ops::op1::define_op1_kernel! {
            $NameKernel,
            |$arg| $body,
            [$($scalar => $output_type),*]
        }
    };

    (
        $NameKernel:ident,
        |$arg:ident| $body:expr,
        [$($input_type:tt => $output_type:tt),* $(,)?]
    ) => {
        struct $NameKernel;
        impl crate::ops::op1::Op1Kernel for $NameKernel {
            fn apply(
                &self,
                mut data: crate::ops::op1::Op1KernelData,
                input_dtype: &crate::dtype::Dtype,
            ) -> crate::error::Result<()> {
                macro_rules! apply_loop_impl {
                    ($input_type2:ty, $output_type2:ty) => {{
                        unsafe {
                            while let Some(src) = data.read_bulk::<$input_type2, { crate::ops::common::BULK }>() {
                                let mut dst: [std::mem::MaybeUninit<$output_type2>; crate::ops::common::BULK]
                                    = std::mem::transmute(std::mem::MaybeUninit::<[$output_type2; crate::ops::common::BULK]>::uninit());
                                for i in 0..crate::ops::common::BULK {
                                    dst[i].write({
                                        let $arg = src[i];
                                        $body
                                    });
                                }
                                data.write_bulk(dst);
                            }
                            while let Some(src) = data.read::<$input_type2>() {
                                let mut dst = std::mem::MaybeUninit::<$output_type2>::uninit();
                                dst.write({
                                    let $arg = src;
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
                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($input_type)) => {
                        apply_loop!($input_type, $output_type)
                    },)*
                    _ => {}
                }
                crate::error::bail!(UnsupportedDtype, "op not supported for dtype {input_dtype:#?}");
            }

            fn output_dtype(&self, input_dtype: &crate::dtype::Dtype) -> crate::error::Result<crate::dtype::Dtype> {
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
pub(crate) use {define_op1, define_op1_kernel};

define_op1!(
    Neg,
    NegKernel,
    core_op = (Neg, neg),
    |a| -a,
    [i8, i16, i32, i64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = "same"
);
// TODO f16
define_op1!(
    Floor,
    FloorKernel,
    |a| a.floor(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Ceil,
    CeilKernel,
    |a| a.ceil(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Round,
    RoundKernel,
    |a| a.round(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Sqrt,
    SqrtKernel,
    |a| a.sqrt(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Exp,
    ExpKernel,
    |a| a.exp(),
    [f32, f64],
    output_type = "same"
);
define_op1!(Log, LogKernel, |a| a.ln(), [f32, f64], output_type = "same");
define_op1!(
    Sin,
    SinKernel,
    |a| a.sin(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Cos,
    CosKernel,
    |a| a.cos(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Tan,
    TanKernel,
    |a| a.tan(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Asin,
    AsinKernel,
    |a| a.asin(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Acos,
    AcosKernel,
    |a| a.acos(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Atan,
    AtanKernel,
    |a| a.atan(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    Signum,
    SignumKernel,
    |a| a.signum(),
    [f16, f32, f64],
    output_type = "same"
);
define_op1!(
    Abs,
    AbsKernel,
    |a| a.abs(),
    [
        i8 => i8,
        i16 => i16,
        i32 => i32,
        i64 => i64,
        f16 => f16,
        f32 => f32,
        f64 => f64,
        (Complex<f32>) => f32,
        (Complex<f64>) => f64,
    ]
);
#[allow(unused)]
trait AbsImpl {
    type Output;
    fn abs(self) -> Self::Output;
}
#[cfg(feature = "half")]
impl AbsImpl for f16 {
    type Output = f16;
    fn abs(self) -> Self::Output {
        Self::from_f32(self.to_f32().abs())
    }
}
impl AbsImpl for Complex<f32> {
    type Output = f32;
    fn abs(self) -> Self::Output {
        self.re.hypot(self.im)
    }
}
impl AbsImpl for Complex<f64> {
    type Output = f64;
    fn abs(self) -> Self::Output {
        self.re.hypot(self.im)
    }
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor);
    define_array_op1_method!(ceil: Ceil);
    define_array_op1_method!(round: Round);
    define_array_op1_method!(sqrt: Sqrt);
    define_array_op1_method!(exp: Exp);
    define_array_op1_method!(ln: Log);
    define_array_op1_method!(sin: Sin);
    define_array_op1_method!(cos: Cos);
    define_array_op1_method!(tan: Tan);
    define_array_op1_method!(asin: Asin);
    define_array_op1_method!(acos: Acos);
    define_array_op1_method!(atan: Atan);
    define_array_op1_method!(signum: Signum);
    define_array_op1_method!(abs: Abs);
}
