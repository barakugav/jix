use crate::ops::common::define_array_op1_method;
use crate::ops::prelude::*;

pub(crate) struct Op1<S, K> {
    pub(crate) array: S,
    kernel: K,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op1Kernel<T> {
    type Output;
    fn apply(&self, x: T) -> Self::Output;

    /// [`apply`](Self::apply) on `N` elements, computing with `simd`'s vectors where the kernel
    /// has a SIMD path for `T` (its scalar trait's `*_bulk`).
    #[inline(always)]
    fn apply_bulk<S: fearless_simd::Simd, const N: usize>(
        &self,
        simd: S,
        xs: [T; N],
    ) -> [Self::Output; N] {
        let _ = simd;
        xs.map_inline(|x| self.apply(x))
    }
}
impl<S, K> Op1<S, K> {
    pub(crate) fn new(array: S, kernel: K) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: Op1Kernel<S::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
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
    type Dimension = S::Dimension;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(out.as_deref(), self.shape())?;
        let out = self
            .read_as_elementwise_pipeline::<K::Output>(index, context)?
            .to_buf(index, context, out);
        Ok(out)
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
            unsafe fn read_bulk<S: fearless_simd::Simd, const N: usize, const CONTIGUOUS: bool>(
                &self,
                simd: S,
                offset: usize,
            ) -> [T; N] {
                let xs = unsafe { self.inner.read_bulk::<S, N, CONTIGUOUS>(simd, offset) };
                let ys = self.kernel.apply_bulk(simd, xs);

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

/// The `apply_bulk` of an [`Op1Kernel`] over `$T` by the scalar trait's bulk function
/// `$bulk_fn(xs, simd)`, with its SIMD bodies typed per impl.
macro_rules! op1_simd_apply_bulk {
    (bulk: $T:ty, $($bulk_fn:tt)+) => {
        #[inline(always)]
        fn apply_bulk<S: fearless_simd::Simd, const N: usize>(
            &self,
            simd: S,
            xs: [$T; N],
        ) -> [Self::Output; N] {
            $($bulk_fn)+(xs, simd)
        }
    };
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        $(core_op = $core_op_trait:ident::$core_op_fn:ident,)?
        $(simd_bulk: $bulk_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: $($trait)::+ + Copy + 'static,
        {
            type Output = <T as $($trait)::+>::Output;

            #[inline(always)]
            fn apply(&self, x: T) -> Self::Output {
                <T as $($trait)::+>::$kernel_fn(x)
            }

            $(
                crate::ops::op1::op1_simd_apply_bulk!(bulk: T, T::$bulk_fn);
            )?
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
            impl $Op, $($trait)::+,
            $(core_op = $core_op_trait::$core_op_fn,)?
        );
    };

    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output<T> = T,
        $(simd_bulk: $bulk_fn:ident,)?
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = T,
            type Output<S> = S::Item,
            $(simd_bulk: $bulk_fn,)?
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output = bool,
        $(simd_bulk: $bulk_fn:ident,)?
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = bool,
            type Output<S> = bool,
            $(simd_bulk: $bulk_fn,)?
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output = $output_type:ty,
        $(simd_bulk: $bulk_fn:ident,)?
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = $output_type,
            type Output<S> = $output_type,
            $(simd_bulk: $bulk_fn,)?
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output<T> = $output_type_t:ty,
        type Output<S> = $output_type_s:ty,
        $(simd_bulk: $bulk_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: $($trait)::+ + Copy + 'static,
        {
            type Output = $output_type_t;

            #[inline(always)]
            fn apply(&self, x: T) -> Self::Output {
                <T as $($trait)::+>::$kernel_fn(x)
            }

            $(
                crate::ops::op1::op1_simd_apply_bulk!(bulk: T, T::$bulk_fn);
            )?
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
        impl $Op:ident, $($trait:ident)::+,
    ) => {};
    (
        @define_core
        impl $Op:ident, $($trait:ident)::+,
        core_op = $core_op_trait:ident::$core_op_fn:ident,
    ) => {
        impl<S> core::ops::$core_op_trait for Array<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
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

pub(crate) use {define_op1, op1_simd_apply_bulk};

pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{define_op1_trait, define_scalar_op1_trait, impl_scalar_op1};
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

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
    define_scalar_op1_trait!(
        /// Scalar kernel of [`Abs`](crate::ops::Abs): the absolute value, the magnitude for
        /// complex types.
        Abs,
        abs,
        abs_bulk
    );
    impl_scalar_op1!(
        Abs::abs / abs_bulk, |x| x.abs(), simd: |x| x.abs(), [i8, i16, i32, i64, f32, f64]
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(Abs::abs, |x| num_traits::Float::abs(x), [f16]);
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        Abs::abs, |x| x.re.hypot(x.im), [Complex<f32> => f32, Complex<f64> => f64]
    );

    define_scalar_op1_trait!(
        /// Scalar kernel of [`Square`](crate::ops::Square): `x * x`.
        Square,
        square,
        square_bulk
    );
    // SIMD for the types with SIMD `Mul` (`crate::scalar::Mul`).
    impl_scalar_op1!(
        Square::square / square_bulk, |x| x * x, simd: |x| x * x,
        [f32, f64, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_scalar_op1!(Square::square, |x| x * x, [i8]);
    #[cfg(feature = "half")]
    impl_scalar_op1!(Square::square, |x| x * x, [f16]);
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(Square::square, |x| x * x, [Complex<f32>, Complex<f64>]);
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op1!(Square::square, |x| x * x, [Complex<f16>]);

    define_scalar_op1_trait!(
        /// Scalar kernel of [`Neg`](crate::ops::Neg), as [`core::ops::Neg`].
        Neg,
        neg,
        neg_bulk
    );
    impl_scalar_op1!(Neg::neg / neg_bulk, |x| -x, simd: |x| -x, [f32, f64, i8, i16, i32, i64]);
    #[cfg(feature = "half")]
    impl_scalar_op1!(Neg::neg, |x| -x, [f16]);
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(Neg::neg, |x| -x, [Complex<f32>, Complex<f64>]);
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op1!(Neg::neg, |x| -x, [Complex<f16>]);

    /// Define the scalar trait `$Trait` of a float rounding / root op, as `num_traits::Float`'s
    /// `$f`, with SIMD for `f32` and `f64`.
    macro_rules! float_op1 {
        ($(#[$meta:meta])* $Trait:ident, $f:ident, $f_bulk:ident) => {
            define_scalar_op1_trait!($(#[$meta])* $Trait, $f, $f_bulk);
            impl_scalar_op1!($Trait::$f / $f_bulk, |x| x.$f(), simd: |x| x.$f(), [f32, f64]);
            #[cfg(feature = "half")]
            impl_scalar_op1!($Trait::$f, |x| num_traits::Float::$f(x), [f16]);
        };
    }
    float_op1!(
        /// Scalar kernel of [`Floor`](crate::ops::Floor), as [`f32::floor`].
        Floor,
        floor,
        floor_bulk
    );
    float_op1!(
        /// Scalar kernel of [`Ceil`](crate::ops::Ceil), as [`f32::ceil`].
        Ceil,
        ceil,
        ceil_bulk
    );
    float_op1!(
        /// Scalar kernel of [`Sqrt`](crate::ops::Sqrt), as [`f32::sqrt`].
        Sqrt,
        sqrt,
        sqrt_bulk
    );
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
    <crate::scalar::Neg>::neg,
    core_op = Neg::neg,
    simd_bulk: neg_bulk,
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
    <crate::scalar::Floor>::floor,
    simd_bulk: floor_bulk,
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
    <crate::scalar::Ceil>::ceil,
    simd_bulk: ceil_bulk,
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
    <crate::scalar::Sqrt>::sqrt,
    simd_bulk: sqrt_bulk,
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
    simd_bulk: abs_bulk,
);

define_op1!(
    /// Squares each element (`x * x`).
    ///
    /// The output dtype is the input dtype.
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
    Square,
    SquareKernel,
    <crate::scalar::Square>::square,
    simd_bulk: square_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor, crate::scalar::Floor);
    define_array_op1_method!(ceil: Ceil, crate::scalar::Ceil);
    define_array_op1_method!(round: Round, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(sqrt: Sqrt, crate::scalar::Sqrt);
    define_array_op1_method!(square: Square, crate::scalar::Square);
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
                    assert_same_elements($kernel.apply_bulk(simd, xs), |i| $f(xs[i]), what);
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
                assert_eq!(NegKernel.apply_bulk(simd, [1.5f32]), [-1.5]);
                assert_eq!(FloorKernel.apply_bulk(simd, [1.5f32, 2.5]), [1.0, 2.0]);
            });
        }
        for_each_simd_level!(check);
    }

    /// A kernel with `simd_bulk:` calls its trait's bulk function.
    #[test]
    fn simd_bulk_calls_trait_fn() {
        use crate::util::for_each_simd_level;

        #[allow(dead_code)]
        mod test_twice {
            use crate::ops::prelude::*;
            use fearless_simd::Simd;
            pub trait TestTwice {
                type Output;
                fn twice(self) -> Self::Output;
                fn twice_bulk<S: Simd, const N: usize>(xs: [Self; N], simd: S) -> [Self::Output; N]
                where
                    Self: Sized;
            }
            impl TestTwice for i32 {
                type Output = i32;
                fn twice(self) -> i32 {
                    self * 2
                }
                fn twice_bulk<S: Simd, const N: usize>(xs: [i32; N], _simd: S) -> [i32; N] {
                    // Not `twice`, so the test sees which is called.
                    xs.map(|x| x * 2 + 1)
                }
            }
            define_op1!(TestTwiceOp, TestTwiceKernel, <TestTwice>::twice, simd_bulk: twice_bulk,);

            pub(super) fn check<S: Simd>(simd: S) {
                use crate::ops::op1::Op1Kernel;
                assert_eq!(TestTwiceKernel.apply(3), 6);
                assert_eq!(TestTwiceKernel.apply_bulk(simd, [3, 4]), [7, 9]);
            }
        }
        use test_twice::check;
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

        // f32: zero is signed - +0.0 -> +1.0, -0.0 -> -1.0 - plus a negative and a positive
        // value.
        let ndf = ndarray::array![-3.0f32, -0.0, 0.0, 5.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a.signum());
        crate::util::assert_array_matches(&zaf.view().sign(), &expectedf);

        // f64: same signed-zero behavior, plus a non-default block shape.
        let ndd = ndarray::array![-3.0f64, -0.0, 0.0, 5.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a.signum());
        crate::util::assert_array_matches(&zad.view().sign(), &expectedd);
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
}
