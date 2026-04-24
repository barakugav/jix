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
    (
        $(#[$meta:meta])*
        $Name:ident,
        $NameKernel:ident,
        core_op = ($op_trait:ident, $op_fn:ident),
        $($kernel_args:tt)*
    ) => {
        define_op1!(
            $(#[$meta])*
            $Name,
            $NameKernel,
            $($kernel_args)*
        );

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

    (
        $(#[$meta:meta])*
        $Name:ident,
        $NameKernel:ident,
        $($kernel_args:tt)*
    ) => {
        $(#[$meta])*
        pub struct $Name<S>(crate::ops::op1::Op1<$NameKernel, S>);
        impl<S> $Name<S> {
            /// Creates a new view storage applying the operation element-wise to `array`.
            ///
            /// See the struct-level documentation for details on supported dtypes, output dtype, and semantics.
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
    /// Arithmetic negation applied element-wise.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | same |
    /// | `f16`, `f32`, `f64` | same |
    /// | `Complex<f32>`, `Complex<f64>` | same |
    ///
    /// The output shape equals the input shape.
    ///
    /// For **integer** types the result is the two's-complement negation.
    /// Negating the minimum representable value (e.g. `i32::MIN`) overflows:
    /// it wraps in release builds and panics in debug builds.
    ///
    /// For **complex** types both components are negated independently:
    /// `-(a + bi) = -a - bi`.
    ///
    /// Available via the unary `-` operator on [`Array`](crate::Array): `-arr`.
    /// Floating-point semantics follow `f32::neg`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.0f32, -2.5, 3.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = (-za).to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-1.0, 2.5, -3.0]);
    ///
    /// // Negating i8::MIN wraps in release builds (two's complement overflow).
    /// let b = ndarray::array![0i8, 1, -1];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = (-zb).to_ndarray::<i8>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0, -1, 1]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Neg,
    NegKernel,
    core_op = (Neg, neg),
    |a| -a,
    [i8, i16, i32, i64, f16, f32, f64, (Complex<f32>), (Complex<f64>)],
    output_type = "same"
);
// TODO f16
define_op1!(
    /// Rounds each element down to the nearest integer (towards −∞).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::floor`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.1f32, 2.9, 3.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.floor().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    ///
    /// // Floor rounds towards −∞, so negative values floor down.
    /// let b = ndarray::array![-1.1f32, -2.9, -3.0];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.floor().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-2.0, -3.0, -3.0]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Floor,
    FloorKernel,
    |a| a.floor(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +∞).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::ceil`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.1f32, 2.0, 3.9];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.ceil().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[2.0, 2.0, 4.0]);
    ///
    /// // Ceil rounds towards +∞, so negative values ceil up.
    /// let b = ndarray::array![-1.7f32, -2.0, -0.1];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.ceil().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-1.0, -2.0, 0.0]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Ceil,
    CeilKernel,
    |a| a.ceil(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    ///
    /// Ties (values exactly halfway between two integers) are broken by rounding
    /// away from zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from
    /// "round-half-to-even" (banker's rounding) used in some other libraries.
    /// Semantics follow [`f32::round`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.4f32, 1.6, 2.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.round().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 2.0]);
    ///
    /// // Ties are broken away from zero: 0.5 → 1.0, -0.5 → -1.0.
    /// let b = ndarray::array![0.5f32, -0.5];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.round().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, -1.0]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Round,
    RoundKernel,
    |a| a.round(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the square root of each element.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    ///
    /// Negative inputs produce `NaN`. Semantics follow [`f32::sqrt`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![4.0f32, 9.0, 16.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.sqrt().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[2.0, 3.0, 4.0]);
    ///
    /// // Negative input produces NaN.
    /// let b = ndarray::array![-1.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.sqrt().to_ndarray::<f32>()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Sqrt,
    SqrtKernel,
    |a| a.sqrt(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::exp`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.0f32, 2.0, 3.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.exp().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - std::f32::consts::E).abs() < 1e-5);
    ///
    /// // exp(0.0) = 1.0 and exp(1.0) ≈ e.
    /// let b = ndarray::array![0.0f32, 1.0];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.exp().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// assert!((result[[1]] - std::f32::consts::E).abs() < 1e-5);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Exp,
    ExpKernel,
    |a| a.exp(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    ///
    /// Negative inputs produce `NaN`; zero produces `-∞`.
    /// Semantics follow [`f32::ln`].
    ///
    /// Available as the `.ln()` method on [`Array`](crate::Array).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.ln().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - 0.0).abs() < 1e-5);
    /// assert!((result[[1]] - 1.0).abs() < 1e-5);
    ///
    /// // Zero produces -inf; negative input produces NaN.
    /// let b = ndarray::array![0.0f32, -1.0];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.ln().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], f32::NEG_INFINITY);
    /// assert!(result[[1]].is_nan());
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Ln,
    LnKernel,
    |a| a.ln(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::sin`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![std::f32::consts::FRAC_PI_2, std::f32::consts::PI];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.sin().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // sin(0.0) = 0.0.
    /// let b = ndarray::array![0.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.sin().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Sin,
    SinKernel,
    |a| a.sin(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::cos`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0.0f32, std::f32::consts::PI];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.cos().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    /// assert!((result[[1]] - (-1.0)).abs() < 1e-5);
    ///
    /// // cos(0.0) = 1.0.
    /// let b = ndarray::array![0.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.cos().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Cos,
    CosKernel,
    |a| a.cos(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::tan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2 * 0.5];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.tan().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // tan(0.0) = 0.0.
    /// let b = ndarray::array![0.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.tan().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Tan,
    TanKernel,
    |a| a.tan(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-π/2, π/2]`.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`. Semantics follow [`f32::asin`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0.0f32, 1.0, -1.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.asin().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // Input outside [-1, 1] produces NaN.
    /// let b = ndarray::array![2.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.asin().to_ndarray::<f32>()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Asin,
    AsinKernel,
    |a| a.asin(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, π]`.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`. Semantics follow [`f32::acos`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.0f32, 0.0, -1.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.acos().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // Input outside [-1, 1] produces NaN.
    /// let b = ndarray::array![2.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.acos().to_ndarray::<f32>()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Acos,
    AcosKernel,
    |a| a.acos(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-π/2, π/2)`.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    /// Semantics follow [`f32::atan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![0.0f32, -1.0, 1.0];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.atan().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 0.0);
    ///
    /// // atan(1.0) = π/4.
    /// let b = ndarray::array![1.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.atan().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Atan,
    AtanKernel,
    |a| a.atan(),
    [f32, f64],
    output_type = "same"
);
define_op1!(
    /// Returns the sign of each element as a floating-point value.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is the same as the input.
    /// The output shape equals the input shape.
    ///
    /// Returns `+1.0` for positive values and `-1.0` for negative values.
    /// Zero is signed: `+0.0` returns `+1.0` and `-0.0` returns `-1.0`.
    /// Semantics follow [`f32::signum`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![3.0f32, -5.0, -0.1];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.signum().to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, -1.0, -1.0]);
    ///
    /// // Positive zero returns +1.0.
    /// let b = ndarray::array![0.0f32];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.signum().to_ndarray::<f32>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Signum,
    SignumKernel,
    |a| a.signum(),
    [f16, f32, f64],
    output_type = "same"
);
define_op1!(
    /// Computes the absolute value of each element.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | same |
    /// | `f16`, `f32`, `f64` | same |
    /// | `Complex<f32>` | `f32` |
    /// | `Complex<f64>` | `f64` |
    ///
    /// The output shape equals the input shape.
    ///
    /// For **complex** types the result is the modulus `sqrt(re² + im²)`, computed
    /// via `hypot` for numerical stability. The output dtype is the real component type
    /// (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`).
    ///
    /// For **signed integer** types, `MIN.abs()` overflows: `(-128i8).abs()` wraps back
    /// to `i8::MIN` in release builds and panics in debug builds.
    ///
    /// Floating-point semantics follow [`f32::abs`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![-3i32, 0, 5, -7];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.abs().to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[3, 0, 5, 7]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    ///
    /// ```
    /// # #[cfg(feature = "num-complex")]
    /// # {
    /// use zix::{Array, ArrayParams};
    /// // For complex input the result is the modulus sqrt(re² + im²).
    /// use zix::dtype::Complex;
    /// let b = ndarray::array![Complex { re: 3.0f32, im: 4.0 }];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.abs().to_ndarray::<f32>()?;
    /// assert!((result[[0]] - 5.0).abs() < 1e-5);
    /// # }
    /// # Ok::<(), zix::error::Error>(())
    /// ```
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
    define_array_op1_method!(ln: Ln);
    define_array_op1_method!(sin: Sin);
    define_array_op1_method!(cos: Cos);
    define_array_op1_method!(tan: Tan);
    define_array_op1_method!(asin: Asin);
    define_array_op1_method!(acos: Acos);
    define_array_op1_method!(atan: Atan);
    define_array_op1_method!(signum: Signum);
    define_array_op1_method!(abs: Abs);
}
