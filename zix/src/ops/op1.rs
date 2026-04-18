use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Complex, Dtype, f16};
use crate::error::{Result, check_get_buffer_size, check_get_range};
use crate::ops::common::define_array_op1_method;
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::DimArray;

pub(crate) trait Op1Kernel {
    fn apply<'a>(
        &self,
        data: impl Iterator<Item = (&'a [u8], &'a mut [u8])>,
        input_dtype: &Dtype,
    ) -> Result<()>;

    fn output_dtype(&self, input_dtype: &Dtype) -> Result<Dtype>;
}

pub(crate) struct Op1<Op, S> {
    op: Op,

    array: Array<S>,

    output_dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
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
            blocks_layout: array.blocks_layout().clone(),
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
        let input_dtype = self.array.dtype();
        let output_dtype = self.dtype();
        // TODO: if the itemsize (and alignment) of the input and output dtype are the same,
        // we can read directly into the output buffer, and perform the op in-place, avoiding the
        // memcopy from temporary buffer. Need to change the op::apply signature.
        let mut tmp_buf = context.tmp_buf(
            nitems * input_dtype.itemsize() as usize,
            input_dtype.alignment(),
        );
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array.storage.read_data(index, tmp_buf, context)?;

        let data_iter = tmp_buf.chunks_exact(input_dtype.itemsize() as usize);
        let out_iter = buf.chunks_exact_mut(output_dtype.itemsize() as usize);
        self.op.apply(data_iter.zip(out_iter), input_dtype)
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.output_dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage.spec()
        }
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
            fn apply<'a>(
                &self,
                data: impl Iterator<Item = (&'a [u8], &'a mut [u8])>,
                input_dtype: &crate::dtype::Dtype,
            ) -> crate::error::Result<()> {
                macro_rules! apply_loop_impl {
                    ($input_type2:ty, $output_type2:ty) => {{
                        let data = data.map(|(src, dst)| {
                            let src = unsafe { src.as_ptr().cast::<$input_type2>().read() };
                            let dst = unsafe { &mut *dst.as_mut_ptr().cast::<$output_type2>() };
                            (src, dst)
                        });
                        for (src, dst) in data {
                            let $arg = src;
                            *dst = $body;
                        }
                        return Ok(())
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
