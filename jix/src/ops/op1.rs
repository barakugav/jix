use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, check_dtype_size_nonzero, Result};
use crate::ops::common::define_array_op1_method;
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{
    check_out_buf, ArraySpec, ArrayStorageInfo, ArrayStorageTyped, ElementwisePipeline,
    ElementwisePipelineImpl, Operand, StridedBuf,
};
use crate::{ArrayExt, ArrayStorage, Ty};

pub(crate) struct Op1<S, K> {
    pub(crate) array: S,
    kernel: K,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op1Kernel<T> {
    type Output;
    fn apply(&self, x: T) -> Self::Output;
}
impl<S, K> Op1<S, K> {
    pub(crate) fn new(array: S, kernel: K) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: Op1Kernel<S::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        let mut spec = array.spec().dynamic().clone();
        spec.element_cost += 1.0;
        Ok(Self {
            array,
            kernel,
            spec,
        })
    }
}

impl<S, K> ArrayStorage for Op1<S, K>
where
    S: ArrayStorageTyped,
    K: Op1Kernel<S::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;
    type Dimension = S::Dimension;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(out.as_deref(), self.shape())?;
        self.read_as_elementwise_pipeline::<K::Output>(index, context)?
            .to_buf(index, context, out)
    }

    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S, K>>
    where
        T: Dtyped,
    {
        check_dtype(Dtype::new_ref::<T>(), Dtype::new_ref::<K::Output>())?;
        let inner = self
            .array
            .read_as_elementwise_pipeline::<S::Item>(index, context)?;

        struct Op1Pipeline<'a, P, K, TIn> {
            inner: P,
            kernel: &'a K,
            phantom: std::marker::PhantomData<TIn>,
        }
        impl<TIn, T, P, K> ElementwisePipelineImpl<T> for Op1Pipeline<'_, P, K, TIn>
        where
            P: ElementwisePipelineImpl<TIn>,
            K: Op1Kernel<TIn, Output: Dtyped>,
            T: Dtyped,
        {
            const N_OPERANDS: Option<usize> = P::N_OPERANDS;

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inner.operands()
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(
                &self,
                offset: usize,
            ) -> [T; N] {
                let xs = unsafe { self.inner.read_bulk::<N, CONTIGUOUS>(offset) };
                xs.map_inline(|x| {
                    let x = self.kernel.apply(x);

                    const { assert!(size_of::<K::Output>() == size_of::<T>()) };
                    // SAFETY: we checked `T` and `K::Output` are the same dtype in the outer func
                    unsafe { std::mem::transmute_copy::<K::Output, T>(&x) }
                })
            }
        }

        Ok(Op1Pipeline {
            inner,
            kernel: &self.kernel,
            phantom: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<K::Output>()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Op1", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Op1<S::DimensionChange<NewD>, K>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Op1 {
            array: self.array.dimension_change()?,
            kernel: self.kernel,
            spec: self.spec,
        })
    }

    crate::ops::impl_element_type_change_default!();
}

impl<F, T, O> Op1Kernel<T> for F
where
    F: Fn(T) -> O,
{
    type Output = O;
    #[inline(always)]
    fn apply(&self, x: T) -> Self::Output {
        self(x)
    }
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        $(core_op = $core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: $($trait)::+,
        {
            type Output = <T as $($trait)::+>::Output;

            #[inline(always)]
            fn apply(&self, x: T) -> Self::Output {
                <T as $($trait)::+>::$kernel_fn(x)
            }
        }
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::op1::Op1<S, $Kernel>);
        impl<S> $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::op1::Op1::new(array, $Kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S> ArrayStorage for $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S::Item as $($trait)::+>::Output>;
            type Dimension = S::Dimension;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }

            type DimensionChange<NewD: crate::Dimension> = $Op<S::DimensionChange<NewD>>;
            #[inline]
            fn dimension_change<NewD: crate::Dimension>(
                self,
            ) -> crate::error::Result<Self::DimensionChange<NewD>> {
                Ok($Op(self.0.dimension_change()?))
            }

            crate::ops::impl_element_type_change_default!();
        }

        define_op1!(@define_core
            impl $Op
            $(core_op = $core_op_trait::$core_op_fn,)?
        );
    };

    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output<T> = T,
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = T,
            type Output<S> = S::Item,
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output = $output_type:ty,
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = $output_type,
            type Output<S> = $output_type,
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output<T> = $output_type_t:ty,
        type Output<S> = $output_type_s:ty,
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: $($trait)::+,
        {
            type Output = $output_type_t;

            #[inline(always)]
            fn apply(&self, x: T) -> Self::Output {
                <T as $($trait)::+>::$kernel_fn(x)
            }
        }
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::op1::Op1<S, $Kernel>);
        impl<S> $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::op1::Op1::new(array, $Kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S> ArrayStorage for $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
        {
            type ElementType = crate::Ty<$output_type_s>;
            type Dimension = S::Dimension;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }

            type DimensionChange<NewD: crate::Dimension> = $Op<S::DimensionChange<NewD>>;
            #[inline]
            fn dimension_change<NewD: crate::Dimension>(
                self,
            ) -> crate::error::Result<Self::DimensionChange<NewD>> {
                Ok($Op(self.0.dimension_change()?))
            }

            crate::ops::impl_element_type_change_default!();
        }
    };

    (
        @define_core
        impl $Op:ident
    ) => {};
    (
        @define_core
        impl $Op:ident
        core_op = $core_op_trait:ident::$core_op_fn:ident,
    ) => {
        impl<S> core::ops::$core_op_trait for Array<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: core::ops::$core_op_trait<Output: crate::dtype::Dtyped>,
        {
            type Output = Array<$Op<S>>;
            #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_op_fn(self) -> Self::Output {
                $Op::new_array(self).unwrap()
            }
        }
    };
}

pub(crate) use define_op1;

pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::define_op1_trait;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    define_op1_trait!(
        Abs,
        abs,
        |a| a.abs(),
        [i8, i16, i32, i64, f32, f64] => "same"
    );
    define_op1_trait!(
        Sign,
        sign,
        |a| a.signum(),
        [i8, i16, i32, i64, f32, f64] => "same"
    );
    #[cfg(feature = "half")]
    impl Sign for f16 {
        type Output = f16;

        #[inline(always)]
        fn sign(self) -> Self::Output {
            <Self as num_traits::Float>::signum(self)
        }
    }
    macro_rules! impl_sign_uint {
        ($($t:ty),*) => {
            $(
                impl Sign for $t {
                    type Output = $t;
                    #[inline(always)]
                    fn sign(self) -> Self::Output {
                        if self == 0 { 0 } else { 1 }
                    }
                }
            )*
        };
    }
    impl_sign_uint!(u8, u16, u32, u64);
    #[cfg(feature = "half")]
    impl Abs for f16 {
        type Output = f16;

        #[inline(always)]
        fn abs(self) -> Self::Output {
            <Self as num_traits::Float>::abs(self)
        }
    }
    #[cfg(feature = "num-complex")]
    impl Abs for Complex<f32> {
        type Output = f32;

        #[inline(always)]
        fn abs(self) -> Self::Output {
            self.re.hypot(self.im)
        }
    }
    #[cfg(feature = "num-complex")]
    impl Abs for Complex<f64> {
        type Output = f64;

        #[inline(always)]
        fn abs(self) -> Self::Output {
            self.re.hypot(self.im)
        }
    }
}

define_op1!(
    /// Arithmetic negation applied element-wise.
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
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::neg()`](core::ops::Neg::neg).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, -2.5, 3.0])?;
    /// let result = (-a).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-1.0, 2.5, -3.0]);
    ///
    /// // Negating i8::MIN wraps in release builds (two's complement overflow).
    /// let b = Array::compact_ndarray(&array![0i8, 1, -1])?;
    /// let result = (-b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0, -1, 1]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Neg,
    NegKernel,
    <core::ops::Neg>::neg,
    core_op = Neg::neg,
);
define_op1!(
    /// Rounds each element down to the nearest integer (towards -inf).
    ///
    /// Semantics follow [`f32::floor`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::floor()`](crate::Array::floor).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.1f32, 2.9, 3.0])?;
    /// let result = a.floor().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    ///
    /// // Floor rounds towards -inf, so negative values floor down.
    /// let b = Array::compact_ndarray(&array![-1.1f32, -2.9, -3.0])?;
    /// let result = b.floor().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-2.0, -3.0, -3.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Floor,
    FloorKernel,
    <num_traits::Float>::floor,
    type Output<T> = T,
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +inf).
    ///
    /// Semantics follow [`f32::ceil`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ceil()`](crate::Array::ceil).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.1f32, 2.0, 3.9])?;
    /// let result = a.ceil().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[2.0, 2.0, 4.0]);
    ///
    /// // Ceil rounds towards +inf, so negative values ceil up.
    /// let b = Array::compact_ndarray(&array![-1.7f32, -2.0, -0.1])?;
    /// let result = b.ceil().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-1.0, -2.0, 0.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Ceil,
    CeilKernel,
    <num_traits::Float>::ceil,
    type Output<T> = T,
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// Ties (values exactly halfway between two integers) are broken by rounding
    /// away from zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from
    /// "round-half-to-even" (banker's rounding) used in some other libraries.
    /// Semantics follow [`f32::round`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::round()`](crate::Array::round).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.4f32, 1.6, 2.0])?;
    /// let result = a.round().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 2.0]);
    ///
    /// // Ties are broken away from zero: 0.5 -> 1.0, -0.5 -> -1.0.
    /// let b = Array::compact_ndarray(&array![0.5f32, -0.5])?;
    /// let result = b.round().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, -1.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Round,
    RoundKernel,
    <num_traits::Float>::round,
    type Output<T> = T,
);
define_op1!(
    /// Computes the square root of each element.
    ///
    /// Negative inputs produce `NaN`. Semantics follow [`f32::sqrt`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sqrt()`](crate::Array::sqrt).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![4.0f32, 9.0, 16.0])?;
    /// let result = a.sqrt().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[2.0, 3.0, 4.0]);
    ///
    /// // Negative input produces NaN.
    /// let b = Array::compact_ndarray(&array![-1.0f32])?;
    /// let result = b.sqrt().to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sqrt,
    SqrtKernel,
    <num_traits::Float>::sqrt,
    type Output<T> = T,
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// Semantics follow [`f32::exp`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::exp()`](crate::Array::exp).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
    /// let result = a.exp().to_ndarray()?;
    /// assert!((result[[0]] - std::f32::consts::E).abs() < 1e-5);
    ///
    /// // exp(0.0) = 1.0 and exp(1.0) = e.
    /// let b = Array::compact_ndarray(&array![0.0f32, 1.0])?;
    /// let result = b.exp().to_ndarray()?;
    /// assert_eq!(result[[0]], 1.0);
    /// assert!((result[[1]] - std::f32::consts::E).abs() < 1e-5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Exp,
    ExpKernel,
    <num_traits::Float>::exp,
    type Output<T> = T,
);
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// Negative inputs produce `NaN`; zero produces `-inf`.
    /// Semantics follow [`f32::ln`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ln()`](crate::Array::ln).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E])?;
    /// let result = a.ln().to_ndarray()?;
    /// assert!((result[[0]] - 0.0).abs() < 1e-5);
    /// assert!((result[[1]] - 1.0).abs() < 1e-5);
    ///
    /// // Zero produces -inf; negative input produces NaN.
    /// let b = Array::compact_ndarray(&array![0.0f32, -1.0])?;
    /// let result = b.ln().to_ndarray()?;
    /// assert_eq!(result[[0]], f32::NEG_INFINITY);
    /// assert!(result[[1]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Ln,
    LnKernel,
    <num_traits::Float>::ln,
    type Output<T> = T,
);
define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// Semantics follow [`f32::sin`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sin()`](crate::Array::sin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![std::f32::consts::FRAC_PI_2, std::f32::consts::PI])?;
    /// let result = a.sin().to_ndarray()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // sin(0.0) = 0.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.sin().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sin,
    SinKernel,
    <num_traits::Float>::sin,
    type Output<T> = T,
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// Semantics follow [`f32::cos`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::cos()`](crate::Array::cos).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, std::f32::consts::PI])?;
    /// let result = a.cos().to_ndarray()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    /// assert!((result[[1]] - (-1.0)).abs() < 1e-5);
    ///
    /// // cos(0.0) = 1.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.cos().to_ndarray()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Cos,
    CosKernel,
    <num_traits::Float>::cos,
    type Output<T> = T,
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// Semantics follow [`f32::tan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::tan()`](crate::Array::tan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2 * 0.5])?;
    /// let result = a.tan().to_ndarray()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // tan(0.0) = 0.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.tan().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Tan,
    TanKernel,
    <num_traits::Float>::tan,
    type Output<T> = T,
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`. Semantics follow [`f32::asin`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::asin()`](crate::Array::asin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, 1.0, -1.0])?;
    /// let result = a.asin().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // Input outside [-1, 1] produces NaN.
    /// let b = Array::compact_ndarray(&array![2.0f32])?;
    /// let result = b.asin().to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Asin,
    AsinKernel,
    <num_traits::Float>::asin,
    type Output<T> = T,
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, pi]`.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`. Semantics follow [`f32::acos`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::acos()`](crate::Array::acos).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, 0.0, -1.0])?;
    /// let result = a.acos().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // Input outside [-1, 1] produces NaN.
    /// let b = Array::compact_ndarray(&array![2.0f32])?;
    /// let result = b.acos().to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Acos,
    AcosKernel,
    <num_traits::Float>::acos,
    type Output<T> = T,
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`.
    ///
    /// Semantics follow [`f32::atan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::atan()`](crate::Array::atan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, -1.0, 1.0])?;
    /// let result = a.atan().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    ///
    /// // atan(1.0) = pi/4.
    /// let b = Array::compact_ndarray(&array![1.0f32])?;
    /// let result = b.atan().to_ndarray()?;
    /// assert!((result[[0]] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Atan,
    AtanKernel,
    <num_traits::Float>::atan,
    type Output<T> = T,
);
define_op1!(
    /// Returns the sign of each element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`.
    ///
    /// For **signed integer** types: returns `-1`, `0`, or `+1` of the same type.
    ///
    /// For **unsigned integer** types: returns `0` or `1` of the same type (since
    /// unsigned values cannot be negative).
    ///
    /// For **float** types: returns `+1.0` for positive values and `-1.0` for
    /// negative values. Zero is signed: `+0.0` returns `+1.0` and `-0.0` returns
    /// `-1.0`. Semantics follow [`f32::signum`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sign()`](crate::Array::sign).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![3i32, -5, 0])?;
    /// let result = a.sign().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1, -1, 0]);
    ///
    /// let b = Array::compact_ndarray(&array![3.0f32, -5.0, -0.1])?;
    /// let result = b.sign().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, -1.0, -1.0]);
    ///
    /// // Float: positive zero returns +1.0.
    /// let c = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = c.sign().to_ndarray()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sign,
    SignKernel,
    <crate::scalar::Sign>::sign,
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
    /// For **complex** types the result is the modulus `sqrt(re^2 + im^2)`, computed
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
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::abs()`](crate::Array::abs).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![-3i32, 0, 5, -7])?;
    /// let result = a.abs().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[3, 0, 5, 7]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// ```
    /// # #[cfg(feature = "num-complex")]
    /// # {
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// // For complex input the result is the modulus sqrt(re^2 + im^2).
    /// use jix::scalar::Complex;
    /// let b = Array::compact_ndarray(&array![Complex { re: 3.0f32, im: 4.0 }])?;
    /// let result = b.abs().to_ndarray()?;
    /// assert!((result[[0]] - 5.0).abs() < 1e-5);
    /// # }
    /// # Ok::<(), jix::Error>(())
    /// ```
    Abs,
    AbsKernel,
    <crate::scalar::Abs>::abs,
);

/// Squares each element (`x * x`).
///
/// The output dtype is `<T as Mul>::Output`, which is the same as the input dtype
/// for all built-in scalar and complex types.
///
/// For **integer** types squaring can overflow, following the semantics of the `*`
/// operator: it wraps in release builds and panics in debug builds.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::square()`](crate::Array::square).
///
/// # Examples
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1.0f32, -2.0, 3.0])?;
/// let result = a.square().to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[1.0, 4.0, 9.0]);
///
/// // Works on integer types too.
/// let b = Array::compact_ndarray(&array![2i32, -3, 4])?;
/// let result = b.square().to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[4, 9, 16]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Square<S>(Op1<S, SquareKernel>);
struct SquareKernel;
impl<T> Op1Kernel<T> for SquareKernel
where
    T: core::ops::Mul + Copy,
{
    type Output = <T as core::ops::Mul>::Output;

    #[inline(always)]
    fn apply(&self, x: T) -> Self::Output {
        x * x
    }
}
impl<S> Square<S>
where
    S: ArrayStorageTyped,
    S::Item: core::ops::Mul<Output: Dtyped>,
{
    /// Constructs a [`Square`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> Result<Self> {
        Ok(Self(Op1::new(array, SquareKernel)?))
    }

    /// Constructs an array with [`Square`] storage. See the storage struct docs for semantics
    /// and examples.
    pub fn new_array(array: Array<S>) -> Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S> ArrayStorage for Square<S>
where
    S: ArrayStorageTyped,
    S::Item: core::ops::Mul<Output: Dtyped>,
{
    type ElementType = Ty<<S::Item as core::ops::Mul>::Output>;
    type Dimension = S::Dimension;
    crate::storage::impl_array_storage_forward!(<S>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Square", [&self.0.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Square<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(self) -> Result<Self::DimensionChange<NewD>> {
        Ok(Square(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(ceil: Ceil, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(round: Round, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(sqrt: Sqrt, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(square: Square, core::ops::Mul);
    define_array_op1_method!(exp: Exp, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(ln: Ln, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(sin: Sin, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(cos: Cos, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(tan: Tan, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(asin: Asin, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(acos: Acos, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(atan: Atan, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(sign: Sign, crate::scalar::Sign);
    define_array_op1_method!(abs: Abs, crate::scalar::Abs);
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use proptest::strategy::BoxedStrategy;
    use proptest::test_runner::{Config, TestRunner};

    /// Shared proptest driver for unary-op tests.
    ///
    /// Generic over the dtype only, with the op passed as a fn pointer, to avoid per-op
    /// monomorphization.
    #[inline(never)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn check_op1<T>(
        strategy: BoxedStrategy<T>,
        check: fn(
            &ndarray::ArrayD<T>,
            crate::Array<crate::storage::Compact<crate::Ty<T>, crate::DimDyn>>,
        ),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(
                &crate::util::carray_strategy_from_shape::<T>(
                    crate::util::shape_strategy(),
                    strategy,
                ),
                |(nd, za)| {
                    check(&nd, za);
                    Ok(())
                },
            )
            .unwrap();
    }

    macro_rules! test_op1_dtype {
        ($op_method:ident, |$arg:ident| $body:expr, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                #[test]
                fn [<$op_method _ $dtype>]() {
                    crate::ops::op1::tests::check_op1::<$dtype>(
                        <$dtype as crate::util::ScalarStrategy>::$strategy(),
                        |nd, za| {
                            #[allow(unused_imports)]
                            use std::ops::Neg;
                            let result = za.$op_method();
                            let expected = nd.mapv(|$arg| $body);
                            crate::util::assert_array_matches(&result, &expected);
                        },
                    );
                }
            }
        };
    }

    macro_rules! test_op1 {
        (
            $op_method:ident, |$arg:ident| $body:expr,
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::op1::tests::test_op1_dtype!($op_method, |$arg| $body, $dtype, $strategy);)+
            $($(
                #[cfg($cfg)]
                crate::ops::op1::tests::test_op1_dtype!($op_method, |$arg| $body, $cfg_dtype, $strategy);
            )+)*
        };
    }

    pub(crate) use {test_op1, test_op1_dtype};

    test_op1!(
        neg,
        |a| -a,
        [i8, i16, i32, i64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    #[test]
    fn square_concrete() {
        use crate::Array;
        // i32: negative, zero, positive, and values at the op_safe_strategy bound (+/-100)
        // so the squared result (10000) still fits comfortably in i32.
        let nd = ndarray::array![[-100i32, -1, 0], [1, 5, 100]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a * a);
        crate::util::assert_array_matches(&za.as_ref().square(), &expected);

        // f32/f64: same edge values (negative, zero, positive, near the +/-100.0 bound), plus
        // a non-default block shape on the f64 arm to cross a block boundary.
        let ndf = ndarray::array![[-100.0f32, -0.5, 0.0], [0.5, 5.0, 100.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a * a);
        crate::util::assert_array_matches(&zaf.as_ref().square(), &expectedf);

        let ndd = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 5.0, 100.0]];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[1, 2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a * a);
        crate::util::assert_array_matches(&zad.as_ref().square(), &expectedd);
    }
    #[test]
    fn floor_concrete() {
        use crate::Array;
        // Edge inputs: exact integers, positive/negative fractions, .5 tie, and a
        // multi-block shape so a block-boundary bug still shows up.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.floor());
        crate::util::assert_array_matches(&za.as_ref().floor(), &expected);

        // Second dtype (f64) and a non-default block shape to cross block boundaries.
        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za64 = Array::compact_ndarray_with(&nd64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.floor());
        crate::util::assert_array_matches(&za64.as_ref().floor(), &expected64);
    }
    #[test]
    fn ceil_concrete() {
        use crate::Array;
        // Same edge inputs as floor: exact integers, positive/negative fractions, .5 tie.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.ceil());
        crate::util::assert_array_matches(&za.as_ref().ceil(), &expected);

        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.ceil());
        crate::util::assert_array_matches(&za64.as_ref().ceil(), &expected64);
    }
    #[test]
    fn round_concrete() {
        use crate::Array;
        // .5 ties round away from zero (not banker's rounding): 0.5 -> 1.0, -0.5 -> -1.0,
        // 2.5 -> 3.0, -2.5 -> -3.0.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 1.4]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.round());
        crate::util::assert_array_matches(&za.as_ref().round(), &expected);

        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 1.4]];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.round());
        crate::util::assert_array_matches(&za64.as_ref().round(), &expected64);
    }
    #[test]
    fn sqrt_concrete() {
        use crate::Array;
        // Domain is non-negative (op_safe_non_negative_strategy): 0.0, a perfect square, a
        // non-perfect square, and the strategy's upper bound (100.0).
        let nd = ndarray::array![[0.0f32, 4.0, 2.0], [9.0, 0.25, 100.0]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.sqrt());
        crate::util::assert_array_matches_approx(&za.as_ref().sqrt(), &expected, 1e-6, 1e-6);

        let nd64 = ndarray::array![[0.0f64, 4.0, 2.0], [9.0, 0.25, 100.0]];
        let za64 = Array::compact_ndarray_with(&nd64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.sqrt());
        crate::util::assert_array_matches_approx(&za64.as_ref().sqrt(), &expected64, 1e-12, 1e-12);
    }
    test_op1!(exp, |a| a.exp(), [f32, f64], op_safe_strategy);
    test_op1!(ln, |a| a.ln(), [f32, f64], op_safe_non_negative_strategy);
    #[test]
    fn sin_concrete() {
        use crate::Array;
        // 0, +/-pi/2, pi, and a couple of interior op_safe_strategy values.
        let nd = ndarray::array![
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            -std::f32::consts::FRAC_PI_2,
            1.0,
            -1.0
        ];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.sin());
        crate::util::assert_array_matches_approx(&za.as_ref().sin(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.sin());
        crate::util::assert_array_matches_approx(&za64.as_ref().sin(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn cos_concrete() {
        use crate::Array;
        let nd = ndarray::array![
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            -std::f32::consts::FRAC_PI_2,
            1.0,
            -1.0
        ];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.cos());
        crate::util::assert_array_matches_approx(&za.as_ref().cos(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.cos());
        crate::util::assert_array_matches_approx(&za64.as_ref().cos(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn tan_concrete() {
        use crate::Array;
        // Avoid drawing exactly at the +/-pi/2 asymptote; op_safe_strategy never hits it
        // either (it draws x/100.0 for integer x, never an exact multiple of pi/2).
        let nd = ndarray::array![
            0.0f32,
            std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_4,
            1.0,
            -1.0,
            10.0
        ];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.tan());
        crate::util::assert_array_matches_approx(&za.as_ref().tan(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.tan());
        crate::util::assert_array_matches_approx(&za64.as_ref().tan(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn asin_concrete() {
        use crate::Array;
        // Domain is [-1, 1] (unit_strategy): both endpoints plus interior points.
        let nd = ndarray::array![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.asin());
        crate::util::assert_array_matches_approx(&za.as_ref().asin(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.asin());
        crate::util::assert_array_matches_approx(&za64.as_ref().asin(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn acos_concrete() {
        use crate::Array;
        // Domain is [-1, 1] (unit_strategy): both endpoints plus interior points.
        let nd = ndarray::array![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.acos());
        crate::util::assert_array_matches_approx(&za.as_ref().acos(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.acos());
        crate::util::assert_array_matches_approx(&za64.as_ref().acos(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn atan_concrete() {
        use crate::Array;
        let nd = ndarray::array![0.0f32, 1.0, -1.0, 100.0, -100.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.atan());
        crate::util::assert_array_matches_approx(&za.as_ref().atan(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.atan());
        crate::util::assert_array_matches_approx(&za64.as_ref().atan(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn sign_concrete() {
        use crate::Array;
        let nd = ndarray::array![-5i32, 0, 7];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a.signum());
        crate::util::assert_array_matches(&za.as_ref().sign(), &expected);

        // Unsigned int: zero maps to 0, any positive value maps to 1 (`a - a` is 0,
        // `a - a + 1` is 1).
        let ndu = ndarray::array![0u32, 1, 7];
        let zau = Array::compact_ndarray(&ndu).unwrap();
        let expectedu = ndu.mapv(|a: u32| if a == 0 { a - a } else { a - a + 1 });
        crate::util::assert_array_matches(&zau.as_ref().sign(), &expectedu);

        // f32: zero is signed - +0.0 -> +1.0, -0.0 -> -1.0 - plus a negative and a positive
        // value.
        let ndf = ndarray::array![-3.0f32, -0.0, 0.0, 5.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a.signum());
        crate::util::assert_array_matches(&zaf.as_ref().sign(), &expectedf);

        // f64: same signed-zero behavior, plus a non-default block shape.
        let ndd = ndarray::array![-3.0f64, -0.0, 0.0, 5.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a.signum());
        crate::util::assert_array_matches(&zad.as_ref().sign(), &expectedd);
    }
    // abs: same dtype for scalar types; complex types have a different output dtype (see below).
    #[test]
    fn abs_concrete() {
        use crate::Array;
        // i32: negative, zero, positive, and values at the op_safe_strategy bound (+/-100) -
        // within range so no MIN-overflow wraparound occurs.
        let nd = ndarray::array![-5i32, 0, 7, -100, 100];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a.abs());
        crate::util::assert_array_matches(&za.as_ref().abs(), &expected);

        let ndf = ndarray::array![-5.0f32, 0.0, 7.5, -100.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a.abs());
        crate::util::assert_array_matches(&zaf.as_ref().abs(), &expectedf);

        let ndd = ndarray::array![-5.0f64, 0.0, 7.5, -100.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a.abs());
        crate::util::assert_array_matches(&zad.as_ref().abs(), &expectedd);
    }
    // TODO
    // #[cfg(feature = "half")]
    // [f16]

    #[cfg(feature = "num-complex")]
    mod complex {
        use super::{complex_f32, complex_f64};

        // abs on complex types: output dtype is the real component type, not the input dtype.
        // Reference uses hypot to match the Abs kernel exactly.
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f32, op_safe_strategy);
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f64, op_safe_strategy);
    }
}
