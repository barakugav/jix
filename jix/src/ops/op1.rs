use crate::ops::common::define_array_op1_method;
use crate::ops::prelude::*;

pub(crate) struct Op1<S, K> {
    pub(crate) array: S,
    kernel: K,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op1Kernel<T> {
    type Output;

    fn apply(&self, x: T, status: &mut ArithmeticOpStatus) -> Self::Output;

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        xs: [T; N],
        status: &mut ArithmeticOpStatus,
        simd: S,
    ) -> [Self::Output; N] {
        let _ = simd;
        xs.map_inline(
            #[inline(always)]
            |x| self.apply(x, status),
        )
    }
}
impl<S, K> Op1<S, K> {
    pub(crate) fn new(array: S, kernel: K) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: Op1Kernel<S::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        check_shape_overflow(array.shape(), K::Output::DTYPE.itemsize() as _)?;
        let mut spec = array.spec().dynamic().clone();
        spec.add_elementwise_cost(1.0);
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

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(
            out.as_deref(),
            self.shape(),
            index,
            self.dtype().itemsize() as usize,
        )?;
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
            const MAX_ITEMSIZE: usize = max_itemsize(&[size_of::<T>(), P::MAX_ITEMSIZE]);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inner.operands()
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool, S: fearless_simd::Simd>(
                &self,
                offset: usize,
                status: &mut ArithmeticOpStatus,
                simd: S,
            ) -> [T; N] {
                let xs = unsafe {
                    self.inner
                        .read_bulk::<N, CONTIGUOUS, S>(offset, status, simd)
                };
                let ys = self.kernel.apply_bulk(xs, status, simd);

                const { assert!(size_of::<K::Output>() == size_of::<T>()) };
                // SAFETY: we checked `T` and `K::Output` are the same dtype in the outer func
                unsafe { std::mem::transmute_copy::<[K::Output; N], [T; N]>(&ys) }
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

    crate::ops::impl_element_type_change_default!();
}

impl<F, T, O> Op1Kernel<T> for F
where
    F: Fn(T) -> O,
{
    type Output = O;
    #[inline(always)]
    fn apply(&self, x: T, _status: &mut ArithmeticOpStatus) -> Self::Output {
        self(x)
    }
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        crate::scalar::$trait:ident::$kernel_fn:ident $(($status:ident))?,
        simd: $bulk_fn:ident,
        $(core_op: core::ops::$core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: crate::scalar::$trait + Copy + 'static,
        {
            type Output = <T as crate::scalar::$trait>::Output;

            #[inline(always)]
            fn apply(&self, x: T, status: &mut ArithmeticOpStatus) -> Self::Output {
                { let _ = status; } // suppress unused
                $(let $status = status;)?
                <T as crate::scalar::$trait>::$kernel_fn(x $(, $status)?)
            }

            #[inline(always)]
            fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
                &self,
                xs: [T; N],
                status: &mut ArithmeticOpStatus,
                simd: S,
            ) -> [Self::Output; N] {
                { let _ = status; } // suppress unused
                $(let $status = status;)?
                <T as crate::scalar::$trait>::$bulk_fn(xs, $($status,)? simd)
            }
        }
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::op1::Op1<S, $Kernel>);
        impl<S> $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::$trait<Output: crate::dtype::Dtyped>,
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
            S::Item: crate::scalar::$trait<Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S::Item as crate::scalar::$trait>::Output>;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }

            crate::ops::impl_element_type_change_default!();
        }

        define_op1!(@define_core
            impl $Op,
            $trait,
            $(core_op: $core_op_trait::$core_op_fn,)?
        );
    };

    (
        @define_core
        impl $Op:ident,
        $trait:ident,
    ) => {};
    (
        @define_core
        impl $Op:ident,
        $trait:ident,
        core_op: $core_op_trait:ident::$core_op_fn:ident,
    ) => {
        impl<S> core::ops::$core_op_trait for Array<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::$trait<Output: crate::dtype::Dtyped>,
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
    use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{define_scalar_op1_trait, impl_scalar_op1};
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    define_scalar_op1_trait!(
        /// Sign operation: as [`i32::signum`] for signed integers, `0` or `1` for unsigned
        /// integers, and as `numpy.sign` for floats.
        ///
        /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
        /// `f16`, `f32`, `f64`.
        ///
        /// For **signed integer** types: returns `-1`, `0`, or `+1` of the same type.
        ///
        /// For **unsigned integer** types: returns `0` or `1` of the same type (since
        /// unsigned values cannot be negative).
        ///
        /// For **float** types: returns `+1.0` for positive values, `-1.0` for negative values,
        /// `0.0` for `+0.0` and `-0.0`, and `NaN` for `NaN`, like `numpy.sign`.
        Sign,
        /// Compute the sign of the value.
        sign,
        bulk_fn: sign_bulk,
    );
    impl_scalar_op1!(
        impl Sign for [i8, i16, i32, i64] {
            fn sign = |x| x.signum(),
        }
    );
    impl_scalar_op1!(
        impl Sign for [u8, u16, u32, u64] {
            fn sign = |x| (x != 0) as Self,
        }
    );
    // Unlike `signum`, which maps `+-0.0` to `+-1.0`, zero maps to `0.0` (as in numpy and torch).
    // `NaN` stays `NaN`.
    impl_scalar_op1!(
        impl Sign for [f32, f64] {
            fn sign = |x| if x == 0.0 { 0.0 } else { x.signum() },
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Sign for [f16] {
            fn sign = |x| if x == f16::ZERO { f16::ZERO } else { f16::from_f32(x.to_f32().signum()) },
        }
    );
    define_scalar_op1_trait!(
        /// Absolute value operation: the magnitude for complex types.
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
        /// For **signed integer** types `MIN.abs()` wraps back to `MIN`: `(-128i8).abs()` is
        /// `-128`.
        ///
        /// Floating-point semantics follow [`f32::abs`].
        Abs,
        /// Compute the absolute value.
        abs,
        bulk_fn: abs_bulk,
    );
    impl_scalar_op1!(
        impl Abs for [f32, f64] {
            fn abs = |x| x.abs(),
            fn abs_bulk = |x| x.abs(),
        }
    );
    impl_scalar_op1!(
        impl Abs for [i8, i16, i32, i64] {
            fn abs = |x| x.wrapping_abs(),
            fn abs_bulk = |x| x.abs(),
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Abs for [f16] {
            fn abs = |x| f16::from_f32(x.to_f32().abs()),
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Abs for [Complex<f32>] {
            type Output = f32;
            fn abs = |x| x.re.hypot(x.im),
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Abs for [Complex<f64>] {
            type Output = f64;
            fn abs = |x| x.re.hypot(x.im),
        }
    );

    define_scalar_op1_trait!(
        /// Squaring operation: `x * x`.
        ///
        /// The output dtype is the input dtype.
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        Square,
        /// Compute the square of the value.
        square,
        bulk_fn: square_bulk,
    );
    // SIMD for the types with SIMD mul
    impl_scalar_op1!(
        impl Square for [f32, f64] {
            fn square = |x| x * x,
            fn square_bulk = |x| x * x,
        }
    );
    impl_scalar_op1!(
        impl Square for [i16, i32, i64, u8, u16, u32, u64] {
            fn square = |x| x.wrapping_mul(x),
            fn square_bulk = |x| x * x,
        }
    );
    impl_scalar_op1!(
        impl Square for [i8] {
            fn square = |x| x.wrapping_mul(x),
        }
    );
    // In f32 without FP16
    #[cfg(all(
        feature = "half",
        not(all(target_arch = "aarch64", target_feature = "fp16"))
    ))]
    impl_scalar_op1!(
        impl Square for [f16] {
            fn square = |x| f16::from_f32(x.to_f32() * x.to_f32()),
        }
    );
    #[cfg(all(feature = "half", target_arch = "aarch64", target_feature = "fp16"))]
    impl_scalar_op1!(
        impl Square for [f16] {
            fn square = |x| x * x,
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Square for [Complex<f32>, Complex<f64>] {
            fn square = |x| x * x,
        }
    );
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op1!(
        impl Square for [Complex<f16>] {
            fn square = |x| x * x,
        }
    );

    define_scalar_op1_trait!(
        /// Negation operation, like [`core::ops::Neg`].
        ///
        /// For **integer** types the result is the two's-complement negation, which wraps:
        /// `-i32::MIN` is `i32::MIN`.
        ///
        /// For **complex** types both components are negated independently:
        /// `-(a + bi) = -a - bi`.
        ///
        /// Floating-point semantics follow `f32::neg`.
        Neg,
        /// Negate the value.
        neg,
        bulk_fn: neg_bulk,
    );
    impl_scalar_op1!(
        impl Neg for [f32, f64] {
            fn neg = |x| -x,
            fn neg_bulk = |x| -x,
        }
    );
    impl_scalar_op1!(
        impl Neg for [i8, i16, i32, i64] {
            fn neg = |x| x.wrapping_neg(),
            fn neg_bulk = |x| -x,
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Neg for [f16] {
            fn neg = |x| -x,
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Neg for [Complex<f32>, Complex<f64>] {
            fn neg = |x| -x,
        }
    );
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op1!(
        impl Neg for [Complex<f16>] {
            fn neg = |x| -x,
        }
    );

    macro_rules! float_op1 {
        (
            $(#[$trait_meta:meta])* $Trait:ident,
            $(#[$fn_meta:meta])* $f:ident,
            $f_bulk:ident $(, $simd:ident="enabled")?
        ) => {
            define_scalar_op1_trait!(
                $(#[$trait_meta])* $Trait,
                $(#[$fn_meta])* $f,
                bulk_fn: $f_bulk,
            );
            float_op1!(@impl $Trait, $f, $f_bulk $(, $simd="enabled")?);
            #[cfg(feature = "half")]
            impl_scalar_op1!(
                impl $Trait for [f16] {
                    fn $f = |x| f16::from_f32(x.to_f32().$f()),
                }
            );
        };
        (@impl $Trait:ident, $f:ident, $f_bulk:ident, simd="enabled") => {
            impl_scalar_op1!(
                impl $Trait for [f32, f64] {
                    fn $f = |x| x.$f(),
                    fn $f_bulk = |x| x.$f(),
                }
            );
        };
        (@impl $Trait:ident, $f:ident, $f_bulk:ident) => {
            impl_scalar_op1!(
                impl $Trait for [f32, f64] {
                    fn $f = |x| x.$f(),
                }
            );
        };
        (
            $(#[$trait_meta:meta])* $Trait:ident,
            $(#[$fn_meta:meta])* $f:ident(status),
            $f_bulk:ident,
            errors = |$x:ident, $r:ident| { $($kind:ident: $cond:expr),+ $(,)? }
        ) => {
            define_scalar_op1_trait!(
                $(#[$trait_meta])* $Trait,
                $(#[$fn_meta])* $f(status),
                bulk_fn: $f_bulk,
            );
            impl_scalar_op1!(
                impl $Trait for [f32, f64] {
                    fn $f = |$x, status| {
                        #[allow(unused_variables)]
                        let $r = $x.$f();
                        #[allow(clippy::double_comparisons)]
                        if $($cond)||+ {
                            *status |= float_op1!(@error $x, $r, $($kind: $cond),+);
                        }
                        $r
                    },
                }
            );
            #[cfg(feature = "half")]
            impl_scalar_op1!(
                impl $Trait for [f16] {
                    fn $f = |x, status| {
                        let r = f16::from_f32(x.to_f32().$f());
                        #[allow(unused_variables)]
                        let ($x, $r) = (x.to_f32(), r.to_f32());
                        #[allow(clippy::double_comparisons)]
                        if $($cond)||+ {
                            *status |= float_op1!(@error $x, $r, $($kind: $cond),+);
                        }
                        r
                    },
                }
            );
        };
        // The rare error, inline(never)
        (@error $x:ident, $r:ident, $($kind:ident: $cond:expr),+) => {{
            #[cold]
            #[inline(never)]
            #[allow(unused_variables)]
            fn error($x: f64, $r: f64) -> ArithmeticOpStatus {
                let mut status = ArithmeticOpStatus::default();
                $(status.set_if(ArithmeticErrorKind::$kind, $cond);)+
                status
            }
            error($x.into(), $r.into())
        }};
    }
    float_op1!(
        /// Flooring operation, as [`f32::floor`].
        Floor,
        /// Round down to the nearest integer (towards -inf).
        floor,
        floor_bulk,
        simd = "enabled"
    );
    float_op1!(
        /// Ceiling operation, as [`f32::ceil`].
        Ceil,
        /// Round up to the nearest integer (towards +inf).
        ceil,
        ceil_bulk,
        simd = "enabled"
    );
    define_scalar_op1_trait!(
        /// Square root operation, as [`f32::sqrt`].
        ///
        /// A negative input produces `NaN` and sets [`ArithmeticErrorKind::InvalidOperation`].
        /// `-0.0` gives `-0.0`, and `NaN` propagates, without an error.
        Sqrt,
        /// Compute the square root, reporting an arithmetic error to `status`.
        sqrt(status),
        bulk_fn: sqrt_bulk,
    );
    impl_scalar_op1!(
        impl Sqrt for [f32, f64] {
            fn sqrt = |x, status| {
                status.set_if(ArithmeticErrorKind::InvalidOperation, x < 0.0);
                x.sqrt()
            },
            fn sqrt_bulk = |x, status| {
                status.set_if(ArithmeticErrorKind::InvalidOperation, x.simd_lt(0.0).any_true());
                x.sqrt()
            },
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Sqrt for [f16] {
            fn sqrt = |x, status| f16::from_f32(Sqrt::sqrt(x.to_f32(), status)),
        }
    );
    float_op1!(
        /// Rounding operation, as [`f32::round`].
        ///
        /// Ties (values exactly halfway between two integers) are broken by rounding
        /// away from zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from
        /// "round-half-to-even" (banker's rounding) used in some other libraries.
        Round,
        /// Round to the nearest integer.
        round,
        round_bulk
    );
    float_op1!(
        /// Natural exponential operation (`e^x`), as [`f32::exp`].
        ///
        /// A finite input whose result overflows to `inf` sets [`ArithmeticErrorKind::Overflow`].
        /// Underflow is not reported.
        Exp,
        /// Compute the natural exponential, reporting an arithmetic error to `status`.
        exp(status),
        exp_bulk,
        errors = |x, r| { Overflow: r.is_infinite() && x.is_finite() }
    );
    float_op1!(
        /// Natural logarithm operation, as [`f32::ln`].
        ///
        /// Zero produces `-inf` and sets [`ArithmeticErrorKind::DivideByZero`]; a negative input
        /// produces `NaN` and sets [`ArithmeticErrorKind::InvalidOperation`]. `NaN`
        /// propagates without an error.
        Ln,
        /// Compute the natural logarithm, reporting an arithmetic error to `status`.
        ln(status),
        ln_bulk,
        errors = |x, r| { DivideByZero: x == 0.0, InvalidOperation: x < 0.0 }
    );
    float_op1!(
        /// Sine operation, as [`f32::sin`].
        ///
        /// An infinite input produces `NaN` and sets [`ArithmeticErrorKind::InvalidOperation`].
        Sin,
        /// Compute the sine (input in radians), reporting an arithmetic error to `status`.
        sin(status),
        sin_bulk,
        errors = |x, r| { InvalidOperation: x.is_infinite() }
    );
    float_op1!(
        /// Cosine operation, as [`f32::cos`].
        ///
        /// An infinite input produces `NaN` and sets [`ArithmeticErrorKind::InvalidOperation`].
        Cos,
        /// Compute the cosine (input in radians), reporting an arithmetic error to `status`.
        cos(status),
        cos_bulk,
        errors = |x, r| { InvalidOperation: x.is_infinite() }
    );
    float_op1!(
        /// Tangent operation, as [`f32::tan`].
        ///
        /// An infinite input produces `NaN` and sets [`ArithmeticErrorKind::InvalidOperation`].
        Tan,
        /// Compute the tangent (input in radians), reporting an arithmetic error to `status`.
        tan(status),
        tan_bulk,
        errors = |x, r| { InvalidOperation: x.is_infinite() }
    );
    float_op1!(
        /// Arcsine operation, as [`f32::asin`].
        ///
        /// An input outside `[-1, 1]` produces `NaN` and sets
        /// [`ArithmeticErrorKind::InvalidOperation`].
        Asin,
        /// Compute the arcsine; the output is in radians in `[-pi/2, pi/2]`, reporting an arithmetic error to `status`.
        asin(status),
        asin_bulk,
        errors = |x, r| { InvalidOperation: x.abs() > 1.0 }
    );
    float_op1!(
        /// Arccosine operation, as [`f32::acos`].
        ///
        /// An input outside `[-1, 1]` produces `NaN` and sets
        /// [`ArithmeticErrorKind::InvalidOperation`].
        Acos,
        /// Compute the arccosine; the output is in radians in `[0, pi]`, reporting an arithmetic error to `status`.
        acos(status),
        acos_bulk,
        errors = |x, r| { InvalidOperation: x.abs() > 1.0 }
    );
    float_op1!(
        /// Arctangent operation, as [`f32::atan`].
        Atan,
        /// Compute the arctangent; the output is in radians in `(-pi/2, pi/2)`.
        atan,
        atan_bulk
    );
}

define_op1!(
    /// Arithmetic negation applied element-wise.
    ///
    /// See [`jix::scalar::Neg`](crate::scalar::Neg) scalar trait for the per-element semantics.
    ///
    /// Available via the unary `-` operator on [`Array`](crate::Array): `-arr`.
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
    /// assert_eq!(result, array![-1.0, 2.5, -3.0]);
    ///
    /// // Negating i8::MIN wraps in release builds (two's complement overflow).
    /// let b = Array::compact_ndarray(&array![0i8, 1, -1])?;
    /// let result = (-b).to_ndarray()?;
    /// assert_eq!(result, array![0, -1, 1]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Neg,
    NegKernel,
    crate::scalar::Neg::neg,
    simd: neg_bulk,
    core_op: core::ops::Neg::neg,
);
define_op1!(
    /// Rounds each element down to the nearest integer (towards -inf).
    ///
    /// See [`jix::scalar::Floor`](crate::scalar::Floor) scalar trait for the per-element semantics.
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
    /// assert_eq!(result, array![1.0, 2.0, 3.0]);
    ///
    /// // Floor rounds towards -inf, so negative values floor down.
    /// let b = Array::compact_ndarray(&array![-1.1f32, -2.9, -3.0])?;
    /// let result = b.floor().to_ndarray()?;
    /// assert_eq!(result, array![-2.0, -3.0, -3.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Floor,
    FloorKernel,
    crate::scalar::Floor::floor,
    simd: floor_bulk,
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +inf).
    ///
    /// See [`jix::scalar::Ceil`](crate::scalar::Ceil) scalar trait for the per-element semantics.
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
    /// assert_eq!(result, array![2.0, 2.0, 4.0]);
    ///
    /// // Ceil rounds towards +inf, so negative values ceil up.
    /// let b = Array::compact_ndarray(&array![-1.7f32, -2.0, -0.1])?;
    /// let result = b.ceil().to_ndarray()?;
    /// assert_eq!(result, array![-1.0, -2.0, 0.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Ceil,
    CeilKernel,
    crate::scalar::Ceil::ceil,
    simd: ceil_bulk,
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// See [`jix::scalar::Round`](crate::scalar::Round) scalar trait for the per-element semantics.
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
    /// assert_eq!(result, array![1.0, 2.0, 2.0]);
    ///
    /// // Ties are broken away from zero: 0.5 -> 1.0, -0.5 -> -1.0.
    /// let b = Array::compact_ndarray(&array![0.5f32, -0.5])?;
    /// let result = b.round().to_ndarray()?;
    /// assert_eq!(result, array![1.0, -1.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Round,
    RoundKernel,
    crate::scalar::Round::round,
    simd: round_bulk,
);
define_op1!(
    /// Computes the square root of each element.
    ///
    /// See [`jix::scalar::Sqrt`](crate::scalar::Sqrt) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sqrt()`](crate::Array::sqrt).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode};
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![4.0f32, 9.0, 16.0])?;
    /// let result = a.sqrt().to_ndarray()?;
    /// assert_eq!(result, array![2.0, 3.0, 4.0]);
    ///
    /// // A negative input is an invalid operation, which fails the read by default.
    /// let b = Array::compact_ndarray(&array![-1.0f32])?.sqrt();
    /// assert!(b.to_ndarray::<Ix1>().is_err());
    /// // Ignored, it produces NaN.
    /// let mut ctx = b.read_ctx();
    /// ctx.set_arithmetic_error_mode(
    ///     ArithmeticErrorKind::InvalidOperation,
    ///     ArithmeticErrorMode::Ignore,
    /// );
    /// assert!(b.to_ndarray_sub::<Ix1>(&[0..1], &ctx)?[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sqrt,
    SqrtKernel,
    crate::scalar::Sqrt::sqrt(status),
    simd: sqrt_bulk,
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// See [`jix::scalar::Exp`](crate::scalar::Exp) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::exp()`](crate::Array::exp).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
    /// let result = a.exp().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - std::f32::consts::E).abs() < 1e-5);
    ///
    /// // exp(0.0) = 1.0 and exp(1.0) = e.
    /// let b = Array::compact_ndarray(&array![0.0f32, 1.0])?;
    /// let result = b.exp().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// assert!((result[[1]] - std::f32::consts::E).abs() < 1e-5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Exp,
    ExpKernel,
    crate::scalar::Exp::exp(status),
    simd: exp_bulk,
);
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// See [`jix::scalar::Ln`](crate::scalar::Ln) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ln()`](crate::Array::ln).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode};
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E])?;
    /// let result = a.ln().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 0.0).abs() < 1e-5);
    /// assert!((result[[1]] - 1.0).abs() < 1e-5);
    ///
    /// // Zero divides by zero and a negative input is invalid, which fail the read by default.
    /// let b = Array::compact_ndarray(&array![0.0f32, -1.0])?.ln();
    /// assert!(b.to_ndarray::<Ix1>().is_err());
    /// // Ignored, they produce -inf and NaN.
    /// let mut ctx = b.read_ctx();
    /// for kind in [ArithmeticErrorKind::DivideByZero, ArithmeticErrorKind::InvalidOperation] {
    ///     ctx.set_arithmetic_error_mode(kind, ArithmeticErrorMode::Ignore);
    /// }
    /// let result = b.to_ndarray_sub::<Ix1>(&[0..2], &ctx)?;
    /// assert_eq!(result[[0]], f32::NEG_INFINITY);
    /// assert!(result[[1]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Ln,
    LnKernel,
    crate::scalar::Ln::ln(status),
    simd: ln_bulk,
);
define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// See [`jix::scalar::Sin`](crate::scalar::Sin) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sin()`](crate::Array::sin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![std::f32::consts::FRAC_PI_2, std::f32::consts::PI])?;
    /// let result = a.sin().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // sin(0.0) = 0.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.sin().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sin,
    SinKernel,
    crate::scalar::Sin::sin(status),
    simd: sin_bulk,
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// See [`jix::scalar::Cos`](crate::scalar::Cos) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::cos()`](crate::Array::cos).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, std::f32::consts::PI])?;
    /// let result = a.cos().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    /// assert!((result[[1]] - (-1.0)).abs() < 1e-5);
    ///
    /// // cos(0.0) = 1.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.cos().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Cos,
    CosKernel,
    crate::scalar::Cos::cos(status),
    simd: cos_bulk,
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// See [`jix::scalar::Tan`](crate::scalar::Tan) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::tan()`](crate::Array::tan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2 * 0.5])?;
    /// let result = a.tan().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // tan(0.0) = 0.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.tan().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Tan,
    TanKernel,
    crate::scalar::Tan::tan(status),
    simd: tan_bulk,
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`.
    ///
    /// See [`jix::scalar::Asin`](crate::scalar::Asin) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::asin()`](crate::Array::asin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode};
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, 1.0, -1.0])?;
    /// let result = a.asin().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // An input outside [-1, 1] is invalid, which fails the read by default.
    /// let b = Array::compact_ndarray(&array![2.0f32])?.asin();
    /// assert!(b.to_ndarray::<Ix1>().is_err());
    /// // Ignored, it produces NaN.
    /// let mut ctx = b.read_ctx();
    /// ctx.set_arithmetic_error_mode(
    ///     ArithmeticErrorKind::InvalidOperation,
    ///     ArithmeticErrorMode::Ignore,
    /// );
    /// assert!(b.to_ndarray_sub::<Ix1>(&[0..1], &ctx)?[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Asin,
    AsinKernel,
    crate::scalar::Asin::asin(status),
    simd: asin_bulk,
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, pi]`.
    ///
    /// See [`jix::scalar::Acos`](crate::scalar::Acos) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::acos()`](crate::Array::acos).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode};
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, 0.0, -1.0])?;
    /// let result = a.acos().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // An input outside [-1, 1] is invalid, which fails the read by default.
    /// let b = Array::compact_ndarray(&array![2.0f32])?.acos();
    /// assert!(b.to_ndarray::<Ix1>().is_err());
    /// // Ignored, it produces NaN.
    /// let mut ctx = b.read_ctx();
    /// ctx.set_arithmetic_error_mode(
    ///     ArithmeticErrorKind::InvalidOperation,
    ///     ArithmeticErrorMode::Ignore,
    /// );
    /// assert!(b.to_ndarray_sub::<Ix1>(&[0..1], &ctx)?[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Acos,
    AcosKernel,
    crate::scalar::Acos::acos(status),
    simd: acos_bulk,
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`.
    ///
    /// See [`jix::scalar::Atan`](crate::scalar::Atan) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::atan()`](crate::Array::atan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, -1.0, 1.0])?;
    /// let result = a.atan().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    ///
    /// // atan(1.0) = pi/4.
    /// let b = Array::compact_ndarray(&array![1.0f32])?;
    /// let result = b.atan().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Atan,
    AtanKernel,
    crate::scalar::Atan::atan,
    simd: atan_bulk,
);
define_op1!(
    /// Returns the sign of each element.
    ///
    /// See [`jix::scalar::Sign`](crate::scalar::Sign) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sign()`](crate::Array::sign).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![3i32, -5, 0])?;
    /// let result = a.sign().to_ndarray()?;
    /// assert_eq!(result, array![1, -1, 0]);
    ///
    /// let b = Array::compact_ndarray(&array![3.0f32, -5.0, -0.1])?;
    /// let result = b.sign().to_ndarray()?;
    /// assert_eq!(result, array![1.0, -1.0, -1.0]);
    ///
    /// // Float: both zeros return 0.0.
    /// let c = Array::compact_ndarray(&array![0.0f32, -0.0])?;
    /// let result = c.sign().to_ndarray()?;
    /// assert_eq!(result, array![0.0, 0.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sign,
    SignKernel,
    crate::scalar::Sign::sign,
    simd: sign_bulk,
);
define_op1!(
    /// Computes the absolute value of each element.
    ///
    /// See [`jix::scalar::Abs`](crate::scalar::Abs) scalar trait for the per-element semantics.
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
    /// assert_eq!(result, array![3, 0, 5, 7]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// ```
    /// # #[cfg(feature = "num-complex")]
    /// # {
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// // For complex input the result is the modulus sqrt(re^2 + im^2).
    /// use jix::scalar::Complex;
    /// let b = Array::compact_ndarray(&array![Complex { re: 3.0f32, im: 4.0 }])?;
    /// let result = b.abs().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 5.0).abs() < 1e-5);
    /// # }
    /// # Ok::<(), jix::Error>(())
    /// ```
    Abs,
    AbsKernel,
    crate::scalar::Abs::abs,
    simd: abs_bulk,
);

define_op1!(
    /// Squares each element (`x * x`).
    ///
    /// See [`jix::scalar::Square`](crate::scalar::Square) scalar trait for the
    /// per-element semantics.
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
    /// assert_eq!(result, array![1.0, 4.0, 9.0]);
    ///
    /// // Works on integer types too.
    /// let b = Array::compact_ndarray(&array![2i32, -3, 4])?;
    /// let result = b.square().to_ndarray()?;
    /// assert_eq!(result, array![4, 9, 16]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Square,
    SquareKernel,
    crate::scalar::Square::square,
    simd: square_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor, crate::scalar::Floor);
    define_array_op1_method!(ceil: Ceil, crate::scalar::Ceil);
    define_array_op1_method!(round: Round, crate::scalar::Round);
    define_array_op1_method!(sqrt: Sqrt, crate::scalar::Sqrt);
    define_array_op1_method!(square: Square, crate::scalar::Square);
    define_array_op1_method!(exp: Exp, crate::scalar::Exp);
    define_array_op1_method!(ln: Ln, crate::scalar::Ln);
    define_array_op1_method!(sin: Sin, crate::scalar::Sin);
    define_array_op1_method!(cos: Cos, crate::scalar::Cos);
    define_array_op1_method!(tan: Tan, crate::scalar::Tan);
    define_array_op1_method!(asin: Asin, crate::scalar::Asin);
    define_array_op1_method!(acos: Acos, crate::scalar::Acos);
    define_array_op1_method!(atan: Atan, crate::scalar::Atan);
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

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::{
            AbsKernel, CeilKernel, FloorKernel, NegKernel, Op1Kernel, RoundKernel, SqrtKernel,
            SquareKernel,
        };
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $t:ty, $f:expr) => {
                    let xs = <$t>::simd_test_values(0);
                    let what = concat!(stringify!($kernel), " ", stringify!($t));
                    assert_same_elements(
                        $kernel.apply_bulk(xs, &mut Default::default(), simd),
                        |i| $f(xs[i]),
                        what,
                    );
                };
            }
            simd.vectorize(|| {
                case!(NegKernel, f32, |x: f32| -x);
                case!(NegKernel, f64, |x: f64| -x);
                case!(NegKernel, i8, i8::wrapping_neg);
                case!(NegKernel, i16, i16::wrapping_neg);
                case!(NegKernel, i32, i32::wrapping_neg);
                case!(NegKernel, i64, i64::wrapping_neg);
                case!(FloorKernel, f32, f32::floor);
                case!(FloorKernel, f64, f64::floor);
                case!(CeilKernel, f32, f32::ceil);
                case!(CeilKernel, f64, f64::ceil);
                case!(RoundKernel, f32, f32::round);
                case!(RoundKernel, f64, f64::round);
                case!(SqrtKernel, f32, f32::sqrt);
                case!(SqrtKernel, f64, f64::sqrt);
                case!(AbsKernel, f32, f32::abs);
                case!(AbsKernel, f64, f64::abs);
                case!(AbsKernel, i8, i8::wrapping_abs);
                case!(AbsKernel, i16, i16::wrapping_abs);
                case!(AbsKernel, i32, i32::wrapping_abs);
                case!(AbsKernel, i64, i64::wrapping_abs);
                case!(SquareKernel, f32, |x: f32| x * x);
                case!(SquareKernel, f64, |x: f64| x * x);
                case!(SquareKernel, i16, |x: i16| x.wrapping_mul(x));
                case!(SquareKernel, i32, |x: i32| x.wrapping_mul(x));
                case!(SquareKernel, i64, |x: i64| x.wrapping_mul(x));
                case!(SquareKernel, u8, |x: u8| x.wrapping_mul(x));
                case!(SquareKernel, u16, |x: u16| x.wrapping_mul(x));
                case!(SquareKernel, u32, |x: u32| x.wrapping_mul(x));
                case!(SquareKernel, u64, |x: u64| x.wrapping_mul(x));
                // Not a multiple of the vector length, and a type without a SIMD body: scalar.
                assert_eq!(
                    NegKernel.apply_bulk([1.5f32], &mut Default::default(), simd),
                    [-1.5]
                );
                assert_eq!(
                    FloorKernel.apply_bulk([1.5f32, 2.5], &mut Default::default(), simd),
                    [1.0, 2.0]
                );
            });
        }
        for_each_simd_level!(check);
    }

    /// Shared proptest driver for unary-op tests.
    ///
    /// Generic over the dtype only, with the op passed as a fn pointer, to avoid per-op
    /// monomorphization.
    #[inline(never)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn check_op1<T>(
        strategy: BoxedStrategy<T>,
        check: fn(&ndarray::ArrayD<T>, crate::util::TestArray<T>),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(
                &crate::util::array_strategy_from_shape::<T>(
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

    test_op1!(neg, |a| a.wrapping_neg(), [i8, i16, i32, i64], any_strategy);
    test_op1!(abs, |a| a.wrapping_abs(), [i8, i16, i32, i64], any_strategy);
    test_op1!(
        neg,
        |a| -a,
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_op1!(
        square,
        |a| a.wrapping_mul(a),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    #[test]
    fn square_concrete() {
        use crate::Array;
        // i32: negative, zero, positive, and values at the op_safe_strategy bound (+/-100)
        // so the squared result (10000) still fits comfortably in i32.
        let nd = ndarray::array![[-100i32, -1, 0], [1, 5, 100]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a * a);
        crate::util::assert_array_matches(&za.view().square(), &expected);

        // f32/f64: same edge values (negative, zero, positive, near the +/-100.0 bound), plus
        // a non-default block shape on the f64 arm to cross a block boundary.
        let ndf = ndarray::array![[-100.0f32, -0.5, 0.0], [0.5, 5.0, 100.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a * a);
        crate::util::assert_array_matches(&zaf.view().square(), &expectedf);

        let ndd = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 5.0, 100.0]];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[1, 2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a * a);
        crate::util::assert_array_matches(&zad.view().square(), &expectedd);
    }
    #[test]
    fn floor_concrete() {
        use crate::Array;
        // Edge inputs: exact integers, positive/negative fractions, .5 tie, and a
        // multi-block shape so a block-boundary bug still shows up.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.floor());
        crate::util::assert_array_matches(&za.view().floor(), &expected);

        // Second dtype (f64) and a non-default block shape to cross block boundaries.
        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za64 = Array::compact_ndarray_with(&nd64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.floor());
        crate::util::assert_array_matches(&za64.view().floor(), &expected64);
    }
    #[test]
    fn ceil_concrete() {
        use crate::Array;
        // Same edge inputs as floor: exact integers, positive/negative fractions, .5 tie.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.ceil());
        crate::util::assert_array_matches(&za.view().ceil(), &expected);

        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.ceil());
        crate::util::assert_array_matches(&za64.view().ceil(), &expected64);
    }
    #[test]
    fn round_concrete() {
        use crate::Array;
        // .5 ties round away from zero (not banker's rounding): 0.5 -> 1.0, -0.5 -> -1.0,
        // 2.5 -> 3.0, -2.5 -> -3.0.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 1.4]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.round());
        crate::util::assert_array_matches(&za.view().round(), &expected);

        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 1.4]];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.round());
        crate::util::assert_array_matches(&za64.view().round(), &expected64);
    }
    #[test]
    fn sqrt_concrete() {
        use crate::Array;
        // Domain is non-negative (op_safe_non_negative_strategy): 0.0, a perfect square, a
        // non-perfect square, and the strategy's upper bound (100.0).
        let nd = ndarray::array![[0.0f32, 4.0, 2.0], [9.0, 0.25, 100.0]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.sqrt());
        crate::util::assert_array_matches_approx(&za.view().sqrt(), &expected, 1e-6, 1e-6);

        let nd64 = ndarray::array![[0.0f64, 4.0, 2.0], [9.0, 0.25, 100.0]];
        let za64 = Array::compact_ndarray_with(&nd64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.sqrt());
        crate::util::assert_array_matches_approx(&za64.view().sqrt(), &expected64, 1e-12, 1e-12);
    }
    test_op1!(exp, |a| a.exp(), [f32, f64], op_safe_strategy);
    test_op1!(ln, |a| a.ln(), [f32, f64], op_safe_positive_strategy);
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
        crate::util::assert_array_matches_approx(&za.view().sin(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.sin());
        crate::util::assert_array_matches_approx(&za64.view().sin(), &expected64, 1e-12, 1e-12);
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
        crate::util::assert_array_matches_approx(&za.view().cos(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.cos());
        crate::util::assert_array_matches_approx(&za64.view().cos(), &expected64, 1e-12, 1e-12);
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
        crate::util::assert_array_matches_approx(&za.view().tan(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.tan());
        crate::util::assert_array_matches_approx(&za64.view().tan(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn asin_concrete() {
        use crate::Array;
        // Domain is [-1, 1] (unit_strategy): both endpoints plus interior points.
        let nd = ndarray::array![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.asin());
        crate::util::assert_array_matches_approx(&za.view().asin(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.asin());
        crate::util::assert_array_matches_approx(&za64.view().asin(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn acos_concrete() {
        use crate::Array;
        // Domain is [-1, 1] (unit_strategy): both endpoints plus interior points.
        let nd = ndarray::array![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.acos());
        crate::util::assert_array_matches_approx(&za.view().acos(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.acos());
        crate::util::assert_array_matches_approx(&za64.view().acos(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn atan_concrete() {
        use crate::Array;
        let nd = ndarray::array![0.0f32, 1.0, -1.0, 100.0, -100.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.atan());
        crate::util::assert_array_matches_approx(&za.view().atan(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.atan());
        crate::util::assert_array_matches_approx(&za64.view().atan(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn sign_concrete() {
        use crate::Array;
        let nd = ndarray::array![-5i32, 0, 7];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a.signum());
        crate::util::assert_array_matches(&za.view().sign(), &expected);

        // Unsigned int: zero maps to 0, any positive value maps to 1 (`a - a` is 0,
        // `a - a + 1` is 1).
        let ndu = ndarray::array![0u32, 1, 7];
        let zau = Array::compact_ndarray(&ndu).unwrap();
        let expectedu = ndu.mapv(|a: u32| if a == 0 { a - a } else { a - a + 1 });
        crate::util::assert_array_matches(&zau.view().sign(), &expectedu);

        // f32: both zeros map to +0.0 (not signum's +-1.0), and NaN stays NaN.
        let ndf = ndarray::array![-3.0f32, -0.0, 0.0, 5.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let resf = zaf.view().sign().to_ndarray::<ndarray::Ix1>().unwrap();
        assert_eq!(resf.as_slice().unwrap(), &[-1.0, 0.0, 0.0, 1.0]);
        assert!(resf.iter().all(|x| !x.is_sign_negative() || *x == -1.0));
        let nan = Array::compact_ndarray(&ndarray::array![f32::NAN]).unwrap();
        assert!(nan.sign().to_ndarray::<ndarray::Ix1>().unwrap()[0].is_nan());

        // f64: same, plus a non-default block shape.
        let ndd = ndarray::array![-3.0f64, -0.0, 0.0, 5.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let resd = zad.view().sign().to_ndarray::<ndarray::Ix1>().unwrap();
        assert_eq!(resd.as_slice().unwrap(), &[-1.0, 0.0, 0.0, 1.0]);
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
        crate::util::assert_array_matches(&za.view().abs(), &expected);

        let ndf = ndarray::array![-5.0f32, 0.0, 7.5, -100.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a.abs());
        crate::util::assert_array_matches(&zaf.view().abs(), &expectedf);

        let ndd = ndarray::array![-5.0f64, 0.0, 7.5, -100.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a.abs());
        crate::util::assert_array_matches(&zad.view().abs(), &expectedd);
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

    /// The status of the SIMD bodies against the scalar functions', on every SIMD level: each
    /// edge case in one lane of a bulk of ordinary values.
    #[test]
    fn simd_bodies_status_all_levels() {
        use super::{Op1Kernel, SqrtKernel};
        use crate::error::ArithmeticOpStatus;
        use crate::util::{for_each_simd_level, SimdTestValues, SIMD_TEST_N};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $t:ty) => {
                    for &x in &<$t>::edge_cases() {
                        let mut scalar = ArithmeticOpStatus::default();
                        $kernel.apply(x, &mut scalar);
                        for lane in [0, 9, 40, SIMD_TEST_N - 1] {
                            let mut xs = [1.5 as $t; SIMD_TEST_N];
                            xs[lane] = x;
                            let mut bulk = ArithmeticOpStatus::default();
                            $kernel.apply_bulk(xs, &mut bulk, simd);
                            assert_eq!(
                                bulk,
                                scalar,
                                "{} {}: {x} in lane {lane}",
                                stringify!($kernel),
                                stringify!($t)
                            );
                        }
                    }
                };
            }
            simd.vectorize(|| {
                case!(SqrtKernel, f32);
                case!(SqrtKernel, f64);
            });
        }
        for_each_simd_level!(check);
    }

    /// The errors of the float functions that report them, on edge cases.
    #[test]
    fn float_op1_errors() {
        use crate::error::ArithmeticErrorKind::{self, *};
        use crate::error::ArithmeticOpStatus;
        #[cfg(feature = "half")]
        use crate::scalar::f16;
        use crate::scalar::{Acos, Asin, Cast, Cos, Exp, Ln, Sin, Tan};

        type Cases<'a> = &'a [(f64, &'a [ArithmeticErrorKind])];
        fn check<T>(name: &str, f: fn(T, &mut ArithmeticOpStatus) -> T, cases: Cases<'_>)
        where
            f64: Cast<T>,
        {
            for &(x, expected) in cases {
                let mut status = ArithmeticOpStatus::default();
                f(Cast::cast(x), &mut status);
                for kind in [DivideByZero, Overflow, Underflow, InvalidOperation] {
                    let what = format!("{name}({x}) {}: {kind:?}", std::any::type_name::<T>());
                    assert_eq!(status.contains(kind), expected.contains(&kind), "{what}");
                }
            }
        }
        macro_rules! check {
            ($name:literal, $f:path, $cases:expr) => {
                check::<f32>($name, $f, $cases);
                check::<f64>($name, $f, $cases);
                #[cfg(feature = "half")]
                check::<f16>($name, $f, $cases);
            };
        }
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let ln: Cases = &[
            (0.0, &[DivideByZero]),
            (-0.0, &[DivideByZero]),
            (-1.0, &[InvalidOperation]),
            (-0.001, &[InvalidOperation]),
            (-inf, &[InvalidOperation]),
            (0.001, &[]),
            (1.0, &[]),
            (inf, &[]),
            (nan, &[]),
        ];
        check!("ln", Ln::ln, ln);
        let exp: Cases = &[
            (1000.0, &[Overflow]),
            (-1000.0, &[]),
            (0.0, &[]),
            (inf, &[]),
            (-inf, &[]),
            (nan, &[]),
        ];
        check!("exp", Exp::exp, exp);
        // Past the largest finite result of each type only.
        check::<f32>("exp", Exp::exp, &[(88.8, &[Overflow]), (88.7, &[])]);
        check::<f64>(
            "exp",
            Exp::exp,
            &[(709.8, &[Overflow]), (709.7, &[]), (88.8, &[])],
        );
        #[cfg(feature = "half")]
        check::<f16>("exp", Exp::exp, &[(11.1, &[Overflow]), (11.0, &[])]);
        let trig: Cases = &[
            (inf, &[InvalidOperation]),
            (-inf, &[InvalidOperation]),
            (nan, &[]),
            (0.0, &[]),
            (std::f64::consts::FRAC_PI_2, &[]),
            (60000.0, &[]),
        ];
        check!("sin", Sin::sin, trig);
        check!("cos", Cos::cos, trig);
        check!("tan", Tan::tan, trig);
        let arc: Cases = &[
            (1.5, &[InvalidOperation]),
            (-1.5, &[InvalidOperation]),
            (inf, &[InvalidOperation]),
            (-inf, &[InvalidOperation]),
            (nan, &[]),
            (1.0, &[]),
            (-1.0, &[]),
            (0.0, &[]),
        ];
        check!("asin", Asin::asin, arc);
        check!("acos", Acos::acos, arc);
        // The value after 1 of each type.
        check::<f32>("asin", Asin::asin, &[(1.0000001, &[InvalidOperation])]);
        check::<f64>(
            "acos",
            Acos::acos,
            &[(1.0000000000000002, &[InvalidOperation])],
        );
        #[cfg(feature = "half")]
        check::<f16>("asin", Asin::asin, &[(1.001, &[InvalidOperation])]);
    }

    macro_rules! test_sqrt_invalid {
        ($($t:ident),*) => {$(
            paste::paste! {
                #[test]
                fn [<sqrt_invalid_ $t>]() {
                    use crate::Array;
                    use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};
                    #[cfg(feature = "half")]
                    #[allow(unused_imports)]
                    use crate::scalar::f16;

                    let c = |x: f64| <f64 as crate::scalar::Cast<$t>>::cast(x);
                    let n = 1000;
                    for (x, invalid) in [
                        (-2.0, true),
                        (f64::NEG_INFINITY, true),
                        (-0.25, true),
                        (-0.0, false),
                        (f64::NAN, false),
                        (f64::INFINITY, false),
                    ] {
                        // In the bulk part of the loop, and in its scalar tail.
                        for i in [300, n - 1] {
                            let mut nd = ndarray::Array1::from_shape_fn(n, |i| c(i as f64 * 0.5));
                            nd[i] = c(x);
                            let r = Array::compact_ndarray(&nd).unwrap().sqrt();
                            let what = format!("sqrt({x}) at {i}");
                            let res = r.to_ndarray::<ndarray::Ix1>();
                            if !invalid {
                                res.expect(&what);
                                continue;
                            }
                            let err = res.expect_err(&what);
                            assert_eq!(err.kind(), ErrorKind::Arithmetic);
                            assert_eq!(err.message(), "invalid operation encountered");

                            let mut ctx = r.read_ctx();
                            ctx.set_arithmetic_error_mode(
                                ArithmeticErrorKind::InvalidOperation,
                                ArithmeticErrorMode::Ignore,
                            );
                            let res = r
                                .to_ndarray_sub::<ndarray::Ix1>(&[0..n as u64], &ctx)
                                .unwrap();
                            assert!(res[i].is_nan(), "{what}");
                        }
                    }
                }
            }
        )*};
    }
    test_sqrt_invalid!(f32, f64);
    #[cfg(feature = "half")]
    test_sqrt_invalid!(f16);
}
