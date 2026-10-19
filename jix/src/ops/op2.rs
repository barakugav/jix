use crate::ops::common::define_array_op2_method;
use crate::ops::prelude::*;
use crate::util::assert_unchecked_eq;

pub(crate) struct Op2<S1, S2, K> {
    pub(crate) a: S1,
    pub(crate) b: S2,
    kernel: K,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op2Kernel<T1, T2> {
    type Output;

    fn apply(&self, a: T1, b: T2, status: &mut ArithmeticOpStatus) -> Self::Output;

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        a: [T1; N],
        b: [T2; N],
        status: &mut ArithmeticOpStatus,
        simd: S,
    ) -> [Self::Output; N] {
        let _ = simd;
        crate::util::array2_map_inline(
            a,
            b,
            #[inline(always)]
            |_, x, y| self.apply(x, y, status),
        )
    }
}
impl<S1, S2, K> Op2<S1, S2, K> {
    pub(crate) fn new(a: S1, b: S2, kernel: K, swapped_operand_order: bool) -> Result<Self>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped,
        K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        {
            let (mut a_shape, mut b_shape) = (a.shape(), b.shape());
            if swapped_operand_order {
                std::mem::swap(&mut a_shape, &mut b_shape);
            }
            ensure!(
                a_shape == b_shape,
                InvalidArgument,
                "Op2 shape mismatch between `a` {:?} and `b` {:?}",
                a_shape,
                b_shape
            );
        }
        let spec = {
            let (mut a_spec, mut b_spec) = (a.spec(), b.spec());
            if swapped_operand_order {
                std::mem::swap(&mut a_spec, &mut b_spec);
            }
            let (element_cost, read_shape_scale_weight, read_layout_order) =
                combine_elementwise_hints(&[
                    (
                        a_spec.element_cost(),
                        a_spec.read_shape_scale_weight(),
                        a_spec.read_layout_order(),
                    ),
                    (
                        b_spec.element_cost(),
                        b_spec.read_shape_scale_weight(),
                        b_spec.read_layout_order(),
                    ),
                ]);
            let (block_shape, block_shape_fixed_dims) = combine_block_layout(&[
                (a_spec.block_shape(), a_spec.block_shape_fixed_dims()),
                (b_spec.block_shape(), b_spec.block_shape_fixed_dims()),
            ]);
            ArraySpecDynamic::new(
                block_shape,
                block_shape_fixed_dims,
                element_cost,
                read_shape_scale_weight,
                read_layout_order,
            )
        };
        Ok(Self { a, b, kernel, spec })
    }

    #[inline(always)]
    pub(crate) fn dynamic_spec(&self) -> &ArraySpecDynamic {
        &self.spec
    }
}

impl<S1, S2, K> ArrayStorage for Op2<S1, S2, K>
where
    S1: ArrayStorageTyped,
    S2: ArrayStorageTyped,
    K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;

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
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S1, S2, K>>
    where
        T: Dtyped,
    {
        check_dtype(Dtype::new_ref::<T>(), Dtype::new_ref::<K::Output>())?;
        let a = self
            .a
            .read_as_elementwise_pipeline::<S1::Item>(index, context)?;
        let b = self
            .b
            .read_as_elementwise_pipeline::<S2::Item>(index, context)?;

        struct Op2Pipeline<'a, P1, P2, K, T1, T2> {
            a: P1,
            b: P2,
            kernel: &'a K,
            phantom: std::marker::PhantomData<(T1, T2)>,
        }
        impl<T1, T2, T, P1, P2, K> ElementwisePipelineImpl<T> for Op2Pipeline<'_, P1, P2, K, T1, T2>
        where
            P1: ElementwisePipelineImpl<T1>,
            P2: ElementwisePipelineImpl<T2>,
            K: Op2Kernel<T1, T2, Output: Dtyped>,
            T1: Copy,
            T2: Copy,
            T: Dtyped,
        {
            const N_OPERANDS: Option<usize> = n_operands_sum(&[P1::N_OPERANDS, P2::N_OPERANDS]);
            const MAX_ITEMSIZE: usize =
                max_itemsize(&[size_of::<T>(), P1::MAX_ITEMSIZE, P2::MAX_ITEMSIZE]);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.a.operands().chain(self.b.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool, S: fearless_simd::Simd>(
                &self,
                offset: usize,
                status: &mut ArithmeticOpStatus,
                simd: S,
            ) -> [T; N] {
                let a = unsafe { self.a.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let b = unsafe { self.b.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let ys = self.kernel.apply_bulk(a, b, status, simd);

                const { assert!(size_of::<K::Output>() == size_of::<T>()) };
                // SAFETY: we checked `T` and `K::Output` are the same dtype in the outer func
                unsafe { std::mem::transmute_copy::<[K::Output; N], [T; N]>(&ys) }
            }
        }

        Ok(Op2Pipeline {
            a,
            b,
            kernel: &self.kernel,
            phantom: std::marker::PhantomData,
        })
    }
    #[inline(always)]
    fn shape(&self) -> &[u64] {
        let shape = self.a.shape();
        unsafe { assert_unchecked_eq!(shape, self.b.shape()) };
        shape
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<K::Output>()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.a
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Op2", [&self.a, &self.b])
    }

    crate::ops::impl_element_type_change_default!();
}

impl<F, T1, T2, O> Op2Kernel<T1, T2> for F
where
    F: Fn(T1, T2) -> O,
{
    type Output = O;
    #[inline(always)]
    fn apply(&self, a: T1, b: T2, _status: &mut ArithmeticOpStatus) -> Self::Output {
        self(a, b)
    }
}

macro_rules! define_op2 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        crate::scalar::$trait:ident::$kernel_fn:ident $(($status:ident))?,
        simd: $bulk_fn:ident,
        $(core_op: core::ops::$core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T1, T2> crate::ops::op2::Op2Kernel<T1, T2> for $Kernel
        where
            T1: crate::scalar::$trait<T2> + Copy + 'static,
            T2: Copy + 'static,
        {
            type Output = <T1 as crate::scalar::$trait<T2>>::Output;

            #[inline(always)]
            fn apply(&self, a: T1, b: T2, status: &mut ArithmeticOpStatus) -> Self::Output {
                { let _ = status; } // suppress unused
                $(let $status = status;)?
                <T1 as crate::scalar::$trait<T2>>::$kernel_fn(a, b $(, $status)?)
            }

            #[inline(always)]
            fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
                &self,
                a: [T1; N],
                b: [T2; N],
                status: &mut ArithmeticOpStatus,
                simd: S,
            ) -> [Self::Output; N] {
                { let _ = status; } // suppress unused
                $(let $status = status;)?
                <T1 as crate::scalar::$trait<T2>>::$bulk_fn(a, b, $($status,)? simd)
            }
        }
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(a: S1, b: S2) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel, false)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(a: crate::Array<S1>, b: crate::Array<S2>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(a.into_storage(), b.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S1::Item as crate::scalar::$trait<S2::Item>>::Output>;
            crate::storage::impl_array_storage_forward!(<S1, S2>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.a, &self.0.b])
            }

            crate::ops::impl_element_type_change_default!();
        }
        define_op2!(@define_core
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
        impl<S1, S2> core::ops::$core_op_trait<Array<S2>> for Array<S1>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type Output = Array<$Op<S1, S2>>;
            #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_op_fn(self, b: Array<S2>) -> Self::Output {
                $Op::new_array(self, b).unwrap()
            }
        }
    };
}

pub(crate) use define_op2;

pub(crate) mod _traits {
    use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{define_scalar_op2_trait, impl_scalar_op2};
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    use fearless_simd::ExtractToken;

    define_scalar_op2_trait!(
        /// Addition operation, like [`core::ops::Add`].
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        /// For **complex** types each component is added independently:
        /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
        Add,
        /// Add another value.
        add,
        bulk_fn: add_bulk,
    );
    define_scalar_op2_trait!(
        /// Subtraction operation, like [`core::ops::Sub`].
        ///
        /// For **integer** types the result wraps on underflow (two's complement).
        /// For **complex** types each component is subtracted independently:
        /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
        Sub,
        /// Subtract another value.
        sub,
        bulk_fn: sub_bulk,
    );
    define_scalar_op2_trait!(
        /// Multiplication operation, like [`core::ops::Mul`].
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        /// For **complex** types this is full complex multiplication:
        /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
        Mul,
        /// Multiply with another value.
        mul,
        bulk_fn: mul_bulk,
    );
    define_scalar_op2_trait!(
        /// Division operation, like [`core::ops::Div`], reporting arithmetic errors to an
        /// [`ArithmeticOpStatus`] instead of panicking.
        ///
        /// For **integer** types the result is truncating (rounds towards zero). Dividing by zero
        /// gives `0` and sets [`ArithmeticErrorKind::DivideByZero`], and `MIN / -1` gives `MIN`
        /// and sets [`ArithmeticErrorKind::Overflow`].
        /// For **float** types the result is the IEEE 754 one, and the errors are:
        /// a finite non-zero value divided by zero sets [`ArithmeticErrorKind::DivideByZero`],
        /// `0 / 0` and `inf / inf` set [`ArithmeticErrorKind::InvalidOperation`], and an infinite
        /// result of finite operands sets [`ArithmeticErrorKind::Overflow`]. NaN operands
        /// propagate without an error.
        /// For **complex** types this is full complex division; errors are not reported yet.
        Div,
        /// Divide by another value, reporting an arithmetic error to `status`.
        div(status),
        bulk_fn: div_bulk,
    );
    define_scalar_op2_trait!(
        /// Truncated division: the quotient `a / b` rounded toward zero.
        ///
        /// Pairs with [`RemainderTrunc`]: `a == q * b + r`.
        ///
        /// For **integer** types this is Rust's `/`, except that a zero divisor gives `0` and sets
        /// [`ArithmeticErrorKind::DivideByZero`], and `MIN / -1` gives `MIN` and sets
        /// [`ArithmeticErrorKind::Overflow`].
        /// For **float** types the result is the exact quotient rounded toward zero, not the
        /// rounded `a / b` (`1.0 / 0.1` rounds to `10.0`, but `0.1` fits in `1.0` only 9 times).
        /// A zero divisor produces `+-inf` (`NaN` for `0 / 0`) and an infinite dividend `NaN`.
        /// A finite non-zero value divided by zero sets [`ArithmeticErrorKind::DivideByZero`],
        /// `0 / 0` and an infinite dividend set [`ArithmeticErrorKind::InvalidOperation`], and an
        /// infinite result of finite operands sets [`ArithmeticErrorKind::Overflow`]. NaN operands
        /// propagate without an error.
        ///
        /// # Examples
        /// ```
        /// use jix::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        /// use jix::scalar::DivideTrunc;
        ///
        /// let mut status = ArithmeticOpStatus::default();
        /// assert_eq!(7i32.divide_trunc(3, &mut status), 2);
        /// assert_eq!((-7i32).divide_trunc(3, &mut status), -2);
        /// assert_eq!(7i32.divide_trunc(-3, &mut status), -2);
        /// assert_eq!((-7i32).divide_trunc(-3, &mut status), 2);
        /// assert_eq!((-7.5f64).divide_trunc(2.0, &mut status), -3.0);
        /// assert_eq!(1.0f64.divide_trunc(0.1, &mut status), 9.0);
        /// assert!(status.is_ok());
        ///
        /// assert_eq!(7i32.divide_trunc(0, &mut status), 0);
        /// assert!(status.contains(ArithmeticErrorKind::DivideByZero));
        /// assert_eq!(i32::MIN.divide_trunc(-1, &mut status), i32::MIN);
        /// assert!(status.contains(ArithmeticErrorKind::Overflow));
        /// assert!(f64::INFINITY.divide_trunc(2.0, &mut status).is_nan());
        /// assert!(status.contains(ArithmeticErrorKind::InvalidOperation));
        /// ```
        DivideTrunc,
        /// Divide by another value, rounding the quotient toward zero, reporting an arithmetic
        /// error to `status`.
        divide_trunc(status),
        bulk_fn: divide_trunc_bulk,
    );
    define_scalar_op2_trait!(
        /// Floored division: the quotient `a / b` rounded toward negative infinity, like Python's
        /// and numpy's `//`.
        ///
        /// Pairs with [`RemainderFloor`]: `a == q * b + r`.
        ///
        /// For **integer** types this is Rust's `/`, minus one where the signs differ and
        /// `a % b != 0`, with the errors of [`DivideTrunc`].
        /// For **float** types it is the exact quotient rounded down, not the rounded `a / b` (so
        /// `1.0 / 0.1` gives `9.0`), with the special values and errors of [`DivideTrunc`].
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticOpStatus;
        /// use jix::scalar::DivideFloor;
        ///
        /// let mut status = ArithmeticOpStatus::default();
        /// assert_eq!(7i32.divide_floor(3, &mut status), 2);
        /// assert_eq!((-7i32).divide_floor(3, &mut status), -3);
        /// assert_eq!(7i32.divide_floor(-3, &mut status), -3);
        /// assert_eq!((-7i32).divide_floor(-3, &mut status), 2);
        /// assert_eq!((-7.5f64).divide_floor(2.0, &mut status), -4.0);
        /// ```
        DivideFloor,
        /// Divide by another value, rounding the quotient toward negative infinity, reporting an
        /// arithmetic error to `status`.
        divide_floor(status),
        bulk_fn: divide_floor_bulk,
    );
    define_scalar_op2_trait!(
        /// Euclidean division: the quotient for which the [`RemainderEuclid`] is never negative,
        /// `floor(a / b)` for `b > 0` and `ceil(a / b)` for `b < 0`.
        ///
        /// Pairs with [`RemainderEuclid`]: `a == q * b + r`.
        ///
        /// For **integer** types this is [`i32::div_euclid`], with the errors of [`DivideTrunc`].
        /// For **float** types it rounds the exact quotient, with the special values and errors of
        /// [`DivideTrunc`] ([`f32::div_euclid`] rounds `a / b` first, and can be one off).
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticOpStatus;
        /// use jix::scalar::DivideEuclid;
        ///
        /// let mut status = ArithmeticOpStatus::default();
        /// assert_eq!(7i32.divide_euclid(3, &mut status), 2);
        /// assert_eq!((-7i32).divide_euclid(3, &mut status), -3);
        /// assert_eq!(7i32.divide_euclid(-3, &mut status), -2);
        /// assert_eq!((-7i32).divide_euclid(-3, &mut status), 3);
        /// assert_eq!((-7.5f64).divide_euclid(2.0, &mut status), -4.0);
        /// ```
        DivideEuclid,
        /// Divide by another value, with the quotient of the never negative remainder, reporting
        /// an arithmetic error to `status`.
        divide_euclid(status),
        bulk_fn: divide_euclid_bulk,
    );
    define_scalar_op2_trait!(
        /// Truncated remainder operation, like [`core::ops::Rem`]: `a - b * trunc(a / b)`.
        ///
        /// The result has the sign of the dividend `a`, like C's `%` and `fmod`, `numpy.fmod` and
        /// `torch.fmod`. It pairs with [`DivideTrunc`].
        ///
        /// For **integer** types this is Rust's `%`, except that a zero divisor gives `0` and sets
        /// [`ArithmeticErrorKind::DivideByZero`], and `MIN % -1` gives `0`.
        /// For **float** types it is Rust's `%`. A zero divisor or an infinite dividend gives `NaN`
        /// and sets [`ArithmeticErrorKind::InvalidOperation`]. NaN operands propagate
        /// without an error.
        ///
        /// # Examples
        /// ```
        /// use jix::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        /// use jix::scalar::RemainderTrunc;
        ///
        /// let mut status = ArithmeticOpStatus::default();
        /// assert_eq!(7i32.remainder_trunc(3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_trunc(3, &mut status), -1);
        /// assert_eq!(7i32.remainder_trunc(-3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_trunc(-3, &mut status), -1);
        /// assert_eq!((-7.5f64).remainder_trunc(2.0, &mut status), -1.5);
        /// assert_eq!(i32::MIN.remainder_trunc(-1, &mut status), 0);
        /// assert!(status.is_ok());
        ///
        /// assert_eq!(7i32.remainder_trunc(0, &mut status), 0);
        /// assert!(status.contains(ArithmeticErrorKind::DivideByZero));
        /// assert!(7.0f64.remainder_trunc(0.0, &mut status).is_nan());
        /// assert!(status.contains(ArithmeticErrorKind::InvalidOperation));
        /// ```
        RemainderTrunc,
        /// Truncated remainder of the division by another value, reporting an arithmetic error to
        /// `status`.
        remainder_trunc(status),
        bulk_fn: remainder_trunc_bulk,
    );
    define_scalar_op2_trait!(
        /// Floored remainder operation: `a - b * floor(a / b)`.
        ///
        /// The result has the sign of the divisor `b`, like Python's `%`, `numpy.remainder` and
        /// `torch.remainder`. It pairs with [`DivideFloor`].
        ///
        /// For **integer** types this is Rust's `%`, plus `b` where it is not zero and its sign
        /// differs from `b`'s, with the errors of [`RemainderTrunc`].
        /// For **float** types it is Python's float `%`, which gives a zero result the sign of `b`,
        /// with the special values and errors of [`RemainderTrunc`].
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticOpStatus;
        /// use jix::scalar::RemainderFloor;
        ///
        /// let mut status = ArithmeticOpStatus::default();
        /// assert_eq!(7i32.remainder_floor(3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_floor(3, &mut status), 2);
        /// assert_eq!(7i32.remainder_floor(-3, &mut status), -2);
        /// assert_eq!((-7i32).remainder_floor(-3, &mut status), -1);
        /// assert_eq!((-7.5f64).remainder_floor(2.0, &mut status), 0.5);
        /// ```
        RemainderFloor,
        /// Floored remainder of the division by another value, reporting an arithmetic error to
        /// `status`.
        remainder_floor(status),
        bulk_fn: remainder_floor_bulk,
    );
    define_scalar_op2_trait!(
        /// Euclidean remainder operation: `a - |b| * floor(a / |b|)`.
        ///
        /// The result is never negative, and equals the [`RemainderFloor`] when `b > 0`. It pairs
        /// with [`DivideEuclid`].
        ///
        /// For **integer** types this is [`i32::rem_euclid`], and for **float** types
        /// [`f32::rem_euclid`], both with the special values and errors of [`RemainderTrunc`].
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticOpStatus;
        /// use jix::scalar::RemainderEuclid;
        ///
        /// let mut status = ArithmeticOpStatus::default();
        /// assert_eq!(7i32.remainder_euclid(3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_euclid(3, &mut status), 2);
        /// assert_eq!(7i32.remainder_euclid(-3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_euclid(-3, &mut status), 2);
        /// assert_eq!((-7.5f64).remainder_euclid(2.0, &mut status), 0.5);
        /// ```
        RemainderEuclid,
        /// Euclidean remainder of the division by another value, reporting an arithmetic error to
        /// `status`.
        remainder_euclid(status),
        bulk_fn: remainder_euclid_bulk,
    );

    macro_rules! impl_arith {
        (
            $Trait:ident::$f:ident, $op:tt, $wrapping_f:ident, [$($scalar:ty),*],
            simd: $f_bulk:ident, [$($simd:ty),*]
        ) => {
            impl_scalar_op2!(
                impl $Trait for [f32, f64] {
                    fn $f = |a, b| a $op b,
                    fn $f_bulk = |a, b| a $op b,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [$($simd),*] {
                    fn $f = |a, b| a.$wrapping_f(b),
                    fn $f_bulk = |a, b| a $op b,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [$($scalar),*] {
                    fn $f = |a, b| a.$wrapping_f(b),
                }
            );
            // In f32 without FP16, as half's operators do, but inlined: theirs call a software
            // fallback function.
            #[cfg(all(feature = "half", not(all(target_arch = "aarch64", target_feature = "fp16"))))]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b| f16::from_f32(a.to_f32() $op b.to_f32()),
                }
            );
            #[cfg(all(feature = "half", target_arch = "aarch64", target_feature = "fp16"))]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(feature = "num-complex")]
            impl_scalar_op2!(
                impl $Trait for [Complex<f32>, Complex<f64>] {
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(all(feature = "half", feature = "num-complex"))]
            impl_scalar_op2!(
                impl $Trait for [Complex<f16>] {
                    fn $f = |a, b| a $op b,
                }
            );
        };
    }
    impl_arith!(
        Add::add, +, wrapping_add, [],
        simd: add_bulk, [i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Sub::sub, -, wrapping_sub, [],
        simd: sub_bulk, [i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Mul::mul, *, wrapping_mul, [i8], // i8 auto-vectorized scalar is faster
        simd: mul_bulk, [i16, i32, i64, u8, u16, u32, u64]
    );

    impl_scalar_op2!(
        impl Div for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn div = |a, b, status| int_quotient(a.checked_div(b), a, b, status),
        }
    );
    impl_scalar_op2!(
        impl DivideTrunc for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn divide_trunc = |a, b, status| int_quotient(a.checked_div(b), a, b, status),
        }
    );
    impl_scalar_op2!(
        impl DivideFloor for [i8, i16, i32, i64] {
            fn divide_floor = |a, b, status| {
                // https://github.com/rust-lang/rust/blob/1.97.0/library/core/src/num/int_macros.rs#L3308-L3324
                let q = a.checked_div(b).map(|d| {
                    let inexact = a.wrapping_sub(d.wrapping_mul(b)) != 0;
                    d + (((a ^ b) >> (Self::BITS - 1)) & -(inexact as Self))
                });
                int_quotient(q, a, b, status)
            },
        }
    );
    impl_scalar_op2!(
        impl DivideFloor for [u8, u16, u32, u64] {
            fn divide_floor = |a, b, status| int_quotient(a.checked_div(b), a, b, status),
        }
    );
    impl_scalar_op2!(
        impl DivideEuclid for [i8, i16, i32, i64] {
            fn divide_euclid = |a, b, status| {
                // https://github.com/rust-lang/rust/blob/1.97.0/library/core/src/num/int_macros.rs#L3308-L3324
                let q = a.checked_div(b).map(|d| {
                    let r = a.wrapping_sub(d.wrapping_mul(b));
                    d - ((r >> (Self::BITS - 1)) & ((b >> (Self::BITS - 1)) | 1))
                });
                int_quotient(q, a, b, status)
            },
        }
    );
    impl_scalar_op2!(
        impl DivideEuclid for [u8, u16, u32, u64] {
            fn divide_euclid = |a, b, status| int_quotient(a.checked_div(b), a, b, status),
        }
    );
    impl_scalar_op2!(
        impl RemainderTrunc for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn remainder_trunc = |a, b, status| int_remainder(a.checked_rem(b), b, status),
        }
    );
    impl_scalar_op2!(
        impl RemainderFloor for [i8, i16, i32, i64] {
            fn remainder_floor = |a, b, status| {
                // numpy's `remainder`, which compiles without branches, unlike num-integer's:
                // https://github.com/numpy/numpy/blob/v2.3.0/numpy/_core/src/umath/loops_modulo.dispatch.c.src#L595-L602
                let rem = a.checked_rem(b).map(
                    |rem| if (a > 0) == (b > 0) || rem == 0 { rem } else { rem + b }
                );
                int_remainder(rem, b, status)
            },
        }
    );
    impl_scalar_op2!(
        impl RemainderFloor for [u8, u16, u32, u64] {
            fn remainder_floor = |a, b, status| int_remainder(a.checked_rem(b), b, status),
        }
    );
    impl_scalar_op2!(
        impl RemainderEuclid for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn remainder_euclid = |a, b, status| int_remainder(a.checked_rem_euclid(b), b, status),
        }
    );

    #[inline(always)]
    fn int_quotient<T: Copy + Default + PartialEq>(
        q: Option<T>,
        a: T,
        b: T,
        status: &mut ArithmeticOpStatus,
    ) -> T {
        match q {
            Some(q) => q,
            None => {
                let zero = b == T::default();
                let mut error = ArithmeticOpStatus::default();
                error.set_if(ArithmeticErrorKind::DivideByZero, zero);
                error.set_if(ArithmeticErrorKind::Overflow, !zero);
                *status |= int_error_barrier::<T>(error);
                if zero {
                    T::default()
                } else {
                    a
                }
            }
        }
    }

    #[inline(always)]
    fn int_remainder<T: Copy + Default + PartialEq>(
        r: Option<T>,
        b: T,
        status: &mut ArithmeticOpStatus,
    ) -> T {
        match r {
            Some(r) => r,
            None => {
                let mut error = ArithmeticOpStatus::default();
                error.set_if(ArithmeticErrorKind::DivideByZero, b == T::default());
                *status |= int_error_barrier::<T>(error);
                T::default()
            }
        }
    }

    // Identify func.
    //
    // Optimization for 8 and 16 types: Without it, LLVM optimize the loos wrong and cause a
    // slowdown of x2
    #[inline(always)]
    fn int_error_barrier<T>(error: ArithmeticOpStatus) -> ArithmeticOpStatus {
        if size_of::<T>() < 4 {
            core::hint::black_box(error)
        } else {
            error
        }
    }

    #[cfg(feature = "half")]
    impl_scalar_op2!(
        impl Div for [f16] {
            fn div = |a, b, status| {
                #[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
                let q = f16::from_f32(a.to_f32() / b.to_f32());
                #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
                let q = a / b;
                f16_quotient(q, a, b, status)
            },
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op2!(
        impl Div for [Complex<f32>, Complex<f64>] {
            fn div = |a, b, _status| a / b,
        }
    );
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op2!(
        impl Div for [Complex<f16>] {
            fn div = |a, b, _status| a / b,
        }
    );
    macro_rules! float_div_rem {
        (
            mod = $m:ident,
            T = $t:ident,
            V<S> = $v:ident,
            exact_limit = $exact_limit:expr,
        ) => {
            mod $m {
                use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
                use fearless_simd::{Select, Simd, SimdBase, SimdFloat, SimdMask};

                type T = $t;
                type V<S> = <S as Simd>::$v;
                const EXACT_LIMIT: T = $exact_limit;

                /// The exact `trunc(a / b)` and its residual `a - q * b`, which below `EXACT_LIMIT`
                /// is `a % b` (`fmod`), exactly.
                ///
                /// The quotient is `div_trunc` of rust-lang/rust#134062, a proposed fix to
                /// `f64::div_euclid`:
                /// https://github.com/rust-lang/rust/blob/343e091465e81589a164adba621f331201746dc6/library/std/src/f64.rs#L280-L293
                /// Unlike there, it compares the residual with `< 0.0`, not `is_sign_negative()`,
                /// which the NaN residuals of special values may pass, and goes on past
                /// `EXACT_LIMIT`, where #134062 returns NaN, within one ulp. As in numpy's
                /// `floor_divide`, an infinite dividend gives NaN.
                ///
                /// The residual of the exact quotient is the remainder, as in the docs there
                /// (L222-L223), exact as `fmod` is:
                /// https://en.cppreference.com/w/c/numeric/math/fmod
                #[inline(always)]
                fn div_rem_trunc(a: T, b: T) -> (T, T) {
                    let d = a / b; // for the sign of the quotient
                    let (x, y) = (a.abs(), b.abs());
                    let qt = d.abs().trunc();
                    let q = if qt.mul_add(-y, x) < 0.0 { qt - 1.0 } else { qt };
                    let r = if q == 0.0 { x } else { q.mul_add(-y, x) }; // not `0 * inf`
                    let q = if a.is_infinite() && b != 0.0 { T::NAN } else { q };
                    (q.copysign(d), r.copysign(a))
                }

                #[inline(always)]
                pub(super) fn div_trunc(a: T, b: T) -> T {
                    div_rem_trunc(a, b).0
                }

                /// The residual, or `fmod` where the quotient is past `EXACT_LIMIT` or NaN.
                #[inline(always)]
                pub(super) fn rem_trunc(a: T, b: T) -> T {
                    let (q, r) = div_rem_trunc(a, b);
                    if q.abs() < EXACT_LIMIT { r } else { a % b }
                }

                #[inline(always)]
                pub(super) fn div_floor(a: T, b: T) -> T {
                    // CPython's `//`, the floored quotient of `_float_div_mod` from the truncated one:
                    // https://github.com/python/cpython/blob/v3.14.0/Objects/floatobject.c#L635-L671
                    let (div, m) = div_rem_trunc(a, b);
                    if m != 0.0 && (b < 0.0) != (m < 0.0) { div - 1.0 } else { div }
                }

                #[inline(always)]
                pub(super) fn rem_floor(a: T, b: T) -> T {
                    // CPython's float `%`, `float_rem`:
                    // https://github.com/python/cpython/blob/v3.14.0/Objects/floatobject.c#L607-L633
                    let m = rem_trunc(a, b);
                    if m != 0.0 {
                        if (b < 0.0) != (m < 0.0) { m + b } else { m }
                    } else {
                        T::copysign(0.0, b)
                    }
                }

                #[inline(always)]
                pub(super) fn div_euclid(a: T, b: T) -> T {
                    // std's `f64::div_euclid`, with the exact quotient and remainder, as #134062:
                    // https://github.com/rust-lang/rust/blob/1.97.0/library/core/src/num/f64.rs#L1988-L1994
                    let (q, r) = div_rem_trunc(a, b);
                    if r < 0.0 {
                        return if b > 0.0 { q - 1.0 } else { q + 1.0 };
                    }
                    q
                }

                #[inline(always)]
                pub(super) fn rem_euclid(a: T, b: T) -> T {
                    // std's `f64::rem_euclid`:
                    // https://github.com/rust-lang/rust/blob/1.97.0/library/core/src/num/f64.rs#L2023-L2026
                    let r = rem_trunc(a, b);
                    if r < 0.0 { r + b.abs() } else { r }
                }

                // The same, on vectors. Their `fma` must be fused: `mul_add_precise`.

                #[inline(always)]
                fn div_rem_trunc_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> (V<S>, V<S>) {
                    let zero = splat(0.0, simd);
                    let d = a / b;
                    let (x, y) = (a.abs(), b.abs());
                    let qt = d.abs().trunc();
                    let overshot = qt.mul_add_precise(-y, x).simd_lt(zero);
                    let q = overshot.select(qt - splat(1.0, simd), qt);
                    let r = q.simd_eq(zero).select(x, q.mul_add_precise(-y, x));
                    let infinite_a = x.simd_eq(splat(T::INFINITY, simd)) & b.simd_ne(zero);
                    let q = infinite_a.select(splat(T::NAN, simd), q);
                    (q.copysign(d), r.copysign(a))
                }

                #[inline(always)]
                pub(super) fn div_trunc_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> V<S> {
                    div_rem_trunc_v(a, b, simd).0
                }

                #[inline(always)]
                pub(super) fn rem_trunc_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> V<S> {
                    let (q, mut r) = div_rem_trunc_v(a, b, simd);
                    if !q.abs().simd_lt(splat(EXACT_LIMIT, simd)).all_true() {
                        fmod_past_limit(q.as_slice(), a.as_slice(), b.as_slice(), r.as_mut_slice());
                    }
                    r
                }

                #[cold]
                #[inline(never)]
                fn fmod_past_limit(q: &[T], a: &[T], b: &[T], r: &mut [T]) {
                    for i in 0..r.len() {
                        if q[i].abs() < EXACT_LIMIT {
                            continue;
                        }
                        r[i] = a[i] % b[i];
                    }
                }

                #[inline(always)]
                pub(super) fn div_floor_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> V<S> {
                    let zero = splat(0.0, simd);
                    let (div, m) = div_rem_trunc_v(a, b, simd);
                    let adjust = m.simd_ne(zero) & (b.simd_lt(zero) ^ m.simd_lt(zero));
                    adjust.select(div - splat(1.0, simd), div)
                }

                #[inline(always)]
                pub(super) fn rem_floor_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> V<S> {
                    let zero = splat(0.0, simd);
                    let m = rem_trunc_v(a, b, simd);
                    let adjusted = (b.simd_lt(zero) ^ m.simd_lt(zero)).select(m + b, m);
                    m.simd_ne(zero).select(adjusted, zero.copysign(b))
                }

                #[inline(always)]
                pub(super) fn div_euclid_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> V<S> {
                    let (zero, one) = (splat(0.0, simd), splat(1.0, simd));
                    let (q, r) = div_rem_trunc_v(a, b, simd);
                    r.simd_lt(zero).select(b.simd_gt(zero).select(q - one, q + one), q)
                }

                #[inline(always)]
                pub(super) fn rem_euclid_v<S: Simd>(a: V<S>, b: V<S>, simd: S) -> V<S> {
                    let r = rem_trunc_v(a, b, simd);
                    r.simd_lt(splat(0.0, simd)).select(r + b.abs(), r)
                }

                #[inline(always)]
                pub(super) fn quotient(q: T, a: T, b: T, status: &mut ArithmeticOpStatus) -> T {
                    if !q.is_finite() && !a.is_nan() && !b.is_nan() {
                        *status |= super::float_div_error(a, b, q);
                    }
                    q
                }

                #[inline(always)]
                pub(super) fn quotient_v<S: Simd>(
                    q: V<S>,
                    a: V<S>,
                    b: V<S>,
                    status: &mut ArithmeticOpStatus,
                    simd: S,
                ) -> V<S> {
                    use ArithmeticErrorKind::*;
                    let non_finite = !q.is_finite();
                    if non_finite.any_true() {
                        let ordered = a.simd_eq(a) & b.simd_eq(b);
                        if (non_finite & ordered).any_true() {
                            let zero = splat(0.0, simd);
                            let infinite = non_finite & q.simd_eq(q) & a.is_finite();
                            status.set_if(InvalidOperation, (q.simd_ne(q) & ordered).any_true());
                            status.set_if(DivideByZero, (infinite & b.simd_eq(zero)).any_true());
                            status.set_if(Overflow, (infinite & b.simd_ne(zero)).any_true());
                        }
                    }
                    q
                }

                #[inline(always)]
                pub(super) fn remainder(r: T, a: T, b: T, status: &mut ArithmeticOpStatus) -> T {
                    let invalid = r.is_nan() && !a.is_nan() && !b.is_nan();
                    status.set_if(ArithmeticErrorKind::InvalidOperation, invalid);
                    r
                }

                #[inline(always)]
                pub(super) fn remainder_v<S: Simd>(
                    r: V<S>,
                    a: V<S>,
                    b: V<S>,
                    status: &mut ArithmeticOpStatus,
                    _simd: S,
                ) -> V<S> {
                    let nan = r.simd_ne(r);
                    if nan.any_true() {
                        let invalid = nan & a.simd_eq(a) & b.simd_eq(b);
                        status.set_if(ArithmeticErrorKind::InvalidOperation, invalid.any_true());
                    }
                    r
                }

                #[inline(always)]
                fn splat<S: Simd>(x: T, simd: S) -> V<S> {
                    <V<S> as SimdBase<S>>::splat(simd, x)
                }
            }

            impl_scalar_op2!(
                impl Div for [$t] {
                    fn div = |a, b, status| $m::quotient(a / b, a, b, status),
                    fn div_bulk = |a, b, status| $m::quotient_v(a / b, a, b, status, a.token()),
                }
            );
            impl_scalar_op2!(
                impl DivideTrunc for [$t] {
                    fn divide_trunc = |a, b, status| {
                        $m::quotient($m::div_trunc(a, b), a, b, status)
                    },
                    fn divide_trunc_bulk = |a, b, status| {
                        let simd = a.token();
                        $m::quotient_v($m::div_trunc_v(a, b, simd), a, b, status, simd)
                    },
                }
            );
            impl_scalar_op2!(
                impl DivideFloor for [$t] {
                    fn divide_floor = |a, b, status| {
                        $m::quotient($m::div_floor(a, b), a, b, status)
                    },
                    fn divide_floor_bulk = |a, b, status| {
                        let simd = a.token();
                        $m::quotient_v($m::div_floor_v(a, b, simd), a, b, status, simd)
                    },
                }
            );
            impl_scalar_op2!(
                impl DivideEuclid for [$t] {
                    fn divide_euclid = |a, b, status| {
                        $m::quotient($m::div_euclid(a, b), a, b, status)
                    },
                    fn divide_euclid_bulk = |a, b, status| {
                        let simd = a.token();
                        $m::quotient_v($m::div_euclid_v(a, b, simd), a, b, status, simd)
                    },
                }
            );
            impl_scalar_op2!(
                impl RemainderTrunc for [$t] {
                    fn remainder_trunc = |a, b, status| {
                        $m::remainder($m::rem_trunc(a, b), a, b, status)
                    },
                    fn remainder_trunc_bulk = |a, b, status| {
                        let simd = a.token();
                        $m::remainder_v($m::rem_trunc_v(a, b, simd), a, b, status, simd)
                    },
                }
            );
            impl_scalar_op2!(
                impl RemainderFloor for [$t] {
                    fn remainder_floor = |a, b, status| {
                        $m::remainder($m::rem_floor(a, b), a, b, status)
                    },
                    fn remainder_floor_bulk = |a, b, status| {
                        let simd = a.token();
                        $m::remainder_v($m::rem_floor_v(a, b, simd), a, b, status, simd)
                    },
                }
            );
            impl_scalar_op2!(
                impl RemainderEuclid for [$t] {
                    fn remainder_euclid = |a, b, status| {
                        $m::remainder($m::rem_euclid(a, b), a, b, status)
                    },
                    fn remainder_euclid_bulk = |a, b, status| {
                        let simd = a.token();
                        $m::remainder_v($m::rem_euclid_v(a, b, simd), a, b, status, simd)
                    },
                }
            );
        };
    }
    float_div_rem!(
        mod = f32_div_rem,
        T = f32,
        V<S> = f32s,
        exact_limit = (1u32 << 24) as f32,
    );
    float_div_rem!(
        mod = f64_div_rem,
        T = f64,
        V<S> = f64s,
        exact_limit = (1u64 << 53) as f64,
    );

    macro_rules! impl_f16_div_rem {
        ($check:ident: $($Trait:ident::$f:ident = $g:ident),*) => {$(
            #[cfg(feature = "half")]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b, status| {
                        let r = f16::from_f32(f32_div_rem::$g(a.to_f32(), b.to_f32()));
                        $check(r, a, b, status)
                    },
                }
            );
        )*};
    }
    impl_f16_div_rem!(
        f16_quotient:
        DivideTrunc::divide_trunc = div_trunc,
        DivideFloor::divide_floor = div_floor,
        DivideEuclid::divide_euclid = div_euclid
    );
    impl_f16_div_rem!(
        f16_remainder:
        RemainderTrunc::remainder_trunc = rem_trunc,
        RemainderFloor::remainder_floor = rem_floor,
        RemainderEuclid::remainder_euclid = rem_euclid
    );

    #[cfg(feature = "half")]
    #[inline(always)]
    fn f16_quotient(q: f16, a: f16, b: f16, status: &mut ArithmeticOpStatus) -> f16 {
        if !q.is_finite() && !a.is_nan() && !b.is_nan() {
            *status |= float_div_error(a, b, q);
        }
        q
    }
    #[cfg(feature = "half")]
    #[inline(always)]
    fn f16_remainder(r: f16, a: f16, b: f16, status: &mut ArithmeticOpStatus) -> f16 {
        let invalid = r.is_nan() && !a.is_nan() && !b.is_nan();
        status.set_if(ArithmeticErrorKind::InvalidOperation, invalid);
        r
    }
    // q=x/y
    #[cold]
    #[inline(never)]
    fn float_div_error<T: Into<f64>>(x: T, y: T, q: T) -> ArithmeticOpStatus {
        use ArithmeticErrorKind::*;
        let (a, b, q): (f64, f64, f64) = (x.into(), y.into(), q.into());
        let mut status = ArithmeticOpStatus::default();
        status.set_if(InvalidOperation, q.is_nan());
        status.set_if(DivideByZero, q.is_infinite() && a.is_finite() && b == 0.0);
        status.set_if(Overflow, q.is_infinite() && a.is_finite() && b != 0.0);
        status
    }

    define_scalar_op2_trait!(
        /// Exponentiation operation: as [`i32::wrapping_pow`] for integers, [`f32::powi`] and
        /// [`f32::powf`] for floats.
        ///
        /// Integer results wrap on overflow. A negative exponent of an integer base gives `0`
        /// and sets [`ArithmeticErrorKind::InvalidOperation`].
        ///
        /// Float values follow [`f32::powf`]. A negative base with a non-integer exponent
        /// produces `NaN` and sets [`ArithmeticErrorKind::InvalidOperation`], zero to a negative
        /// power gives `inf` and sets [`ArithmeticErrorKind::DivideByZero`], and finite operands
        /// with an infinite result set [`ArithmeticErrorKind::Overflow`].
        Pow,
        /// Raise to the power of another value, reporting an arithmetic error to `status`.
        pow(status),
        bulk_fn: pow_bulk,
    );

    impl_scalar_op2!(
        impl Pow for [i8, i16, i32, i64, u8, u16, u32, u64] x [u8, u16, u32] {
            fn pow = |a, b, _status| a.wrapping_pow(b as u32),
        }
    );
    impl_scalar_op2!(
        impl Pow for [i8, i16, i32, i64, u8, u16, u32, u64] x [i8, i16, i32, i64] {
            fn pow = |a, b, status| {
                status.set_if(ArithmeticErrorKind::InvalidOperation, b < 0);
                if let Ok(b) = u32::try_from(b) {
                    a.wrapping_pow(b)
                } else if b < 0 {
                    0
                } else {
                    let (mut acc, mut base, mut exp): (Self, Self, u64) = (1, a, b as u64);
                    while exp > 0 {
                        if exp & 1 == 1 {
                            acc = acc.wrapping_mul(base);
                        }
                        base = base.wrapping_mul(base);
                        exp >>= 1;
                    }
                    acc
                }
            },
        }
    );
    impl_scalar_op2!(
        impl Pow for [f32, f64] x [i8, u8, i16, u16, i32] {
            fn pow = |a, b, status| float_power(a.powi(b as i32), a, b, status),
        }
    );
    impl_scalar_op2!(
        impl Pow for [f32, f64] x [f32] {
            fn pow = |a, b, status| float_power(a.powf(b as Self), a, b, status),
        }
    );
    impl_scalar_op2!(
        impl Pow for [f64] x [f64] {
            fn pow = |a, b, status| float_power(a.powf(b), a, b, status),
        }
    );

    #[inline(always)]
    fn float_power<T: Into<f64> + Copy, U: Into<f64>>(
        p: T,
        a: T,
        b: U,
        status: &mut ArithmeticOpStatus,
    ) -> T {
        let (x, y, r): (f64, f64, f64) = (a.into(), b.into(), p.into());
        if !r.is_finite() && !x.is_nan() && !y.is_nan() {
            *status |= float_pow_error(x, y, r);
        }
        p
    }
    // r=x^y
    #[cold]
    #[inline(never)]
    fn float_pow_error(x: f64, y: f64, r: f64) -> ArithmeticOpStatus {
        use ArithmeticErrorKind::*;
        let mut status = ArithmeticOpStatus::default();
        status.set_if(InvalidOperation, r.is_nan());
        status.set_if(DivideByZero, x == 0.0 && y < 0.0);
        let finite = x.is_finite() && y.is_finite();
        status.set_if(Overflow, r.is_infinite() && finite && x != 0.0);
        status
    }
}

define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// See [`jix::scalar::Add`](crate::scalar::Add) scalar trait for the per-element semantics.
    ///
    /// Available via the `+` operator on arrays. For adding a constant to every element, use
    /// [`Array::map`]: `a.map(|x| x + 1i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let b = Array::compact_ndarray(&array![10i32, 20, 30])?;
    /// let result = (a + b).to_ndarray()?;
    /// assert_eq!(result, array![11, 22, 33]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Add,
    AddKernel,
    crate::scalar::Add::add,
    simd: add_bulk,
    core_op: core::ops::Add::add,
);
define_op2!(
    /// Element-wise subtraction of two arrays (`a - b`).
    ///
    /// See [`jix::scalar::Sub`](crate::scalar::Sub) scalar trait for the per-element semantics.
    ///
    /// Available via the `-` operator on arrays. For subtracting a constant from every element,
    /// use [`Array::map`]: `a.map(|x| x - 1i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![10i32, 20, 30])?;
    /// let b = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let result = (a - b).to_ndarray()?;
    /// assert_eq!(result, array![9, 18, 27]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sub,
    SubKernel,
    crate::scalar::Sub::sub,
    simd: sub_bulk,
    core_op: core::ops::Sub::sub,
);
define_op2!(
    /// Element-wise multiplication of two arrays.
    ///
    /// See [`jix::scalar::Mul`](crate::scalar::Mul) scalar trait for the per-element semantics.
    ///
    /// Available via the `*` operator on arrays. For scaling every element by a constant, use
    /// [`Array::map`]: `a.map(|x| x * 2i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let b = Array::compact_ndarray(&array![4i32, 5, 6])?;
    /// let result = (a * b).to_ndarray()?;
    /// assert_eq!(result, array![4, 10, 18]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Mul,
    MulKernel,
    crate::scalar::Mul::mul,
    simd: mul_bulk,
    core_op: core::ops::Mul::mul,
);

define_op2!(
    /// Element-wise division of two arrays (`a / b`).
    ///
    /// See [`jix::scalar::Div`](crate::scalar::Div) scalar trait for the per-element semantics.
    ///
    /// Available via the `/` operator on arrays. For dividing every element by a constant, use
    /// [`Array::map`]: `a.map(|x| x / 2i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![10i32, 20, 30])?;
    /// let b = Array::compact_ndarray(&array![2i32, 4, 5])?;
    /// let result = (a / b).to_ndarray()?;
    /// assert_eq!(result, array![5, 5, 6]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Div,
    DivKernel,
    crate::scalar::Div::div(status),
    simd: div_bulk,
    core_op: core::ops::Div::div,
);
define_op2!(
    /// Element-wise truncated division of two arrays.
    ///
    /// See [`jix::scalar::DivideTrunc`](crate::scalar::DivideTrunc) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![7i32, -7, 7, -7])?;
    /// let b = Array::compact_ndarray(&array![2i32, 2, -2, -2])?;
    /// let result = a.divide_trunc(b).to_ndarray()?;
    /// assert_eq!(result, array![3, -3, -3, 3]);
    ///
    /// // Of the exact quotient: `0.1` fits in `1.0` only 9 times, though `1.0 / 0.1` rounds to 10.
    /// let a = Array::compact_ndarray(&array![1.0f64, -7.5])?;
    /// let b = Array::compact_ndarray(&array![0.1f64, 2.0])?;
    /// let result = a.divide_trunc(b).to_ndarray()?;
    /// assert_eq!(result, array![9.0, -3.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    DivideTrunc,
    DivideTruncKernel,
    crate::scalar::DivideTrunc::divide_trunc(status),
    simd: divide_trunc_bulk,
);
define_op2!(
    /// Element-wise floored division of two arrays.
    ///
    /// See [`jix::scalar::DivideFloor`](crate::scalar::DivideFloor) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![7i32, -7, 7, -7])?;
    /// let b = Array::compact_ndarray(&array![2i32, 2, -2, -2])?;
    /// let result = a.divide_floor(b).to_ndarray()?;
    /// assert_eq!(result, array![3, -4, -4, 3]);
    ///
    /// // Of the exact quotient: `0.1` fits in `1.0` only 9 times, though `1.0 / 0.1` rounds to 10.
    /// let a = Array::compact_ndarray(&array![1.0f64, -7.5])?;
    /// let b = Array::compact_ndarray(&array![0.1f64, 2.0])?;
    /// let result = a.divide_floor(b).to_ndarray()?;
    /// assert_eq!(result, array![9.0, -4.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    DivideFloor,
    DivideFloorKernel,
    crate::scalar::DivideFloor::divide_floor(status),
    simd: divide_floor_bulk,
);
define_op2!(
    /// Element-wise Euclidean division of two arrays.
    ///
    /// See [`jix::scalar::DivideEuclid`](crate::scalar::DivideEuclid) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![7i32, -7, 7, -7])?;
    /// let b = Array::compact_ndarray(&array![2i32, 2, -2, -2])?;
    /// let result = a.divide_euclid(b).to_ndarray()?;
    /// assert_eq!(result, array![3, -4, -3, 4]);
    ///
    /// // Of the exact quotient: `0.1` fits in `1.0` only 9 times, though `1.0 / 0.1` rounds to 10.
    /// let a = Array::compact_ndarray(&array![1.0f64, -7.5])?;
    /// let b = Array::compact_ndarray(&array![0.1f64, 2.0])?;
    /// let result = a.divide_euclid(b).to_ndarray()?;
    /// assert_eq!(result, array![9.0, -4.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    DivideEuclid,
    DivideEuclidKernel,
    crate::scalar::DivideEuclid::divide_euclid(status),
    simd: divide_euclid_bulk,
);
define_op2!(
    /// Element-wise truncated remainder of two arrays (`a % b`).
    ///
    /// See [`jix::scalar::RemainderTrunc`](crate::scalar::RemainderTrunc) scalar trait for the
    /// per-element semantics.
    ///
    /// Available via the `%` operator on arrays, with the semantics of Rust's `%`, and as
    /// [`Array::remainder_trunc`]. For the remainder of every element by a constant, use
    /// [`Array::map`]: `a.map(|x| x % 3i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![7i32, -7, 7, -7])?;
    /// let b = Array::compact_ndarray(&array![3i32, 3, -3, -3])?;
    /// let result = (a % b).to_ndarray()?;
    /// assert_eq!(result, array![1, -1, 1, -1]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    RemainderTrunc,
    RemainderTruncKernel,
    crate::scalar::RemainderTrunc::remainder_trunc(status),
    simd: remainder_trunc_bulk,
    core_op: core::ops::Rem::rem,
);
define_op2!(
    /// Element-wise floored remainder of two arrays.
    ///
    /// See [`jix::scalar::RemainderFloor`](crate::scalar::RemainderFloor) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![7i32, -7, 7, -7])?;
    /// let b = Array::compact_ndarray(&array![3i32, 3, -3, -3])?;
    /// let result = a.remainder_floor(b).to_ndarray()?;
    /// assert_eq!(result, array![1, 2, -2, -1]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    RemainderFloor,
    RemainderFloorKernel,
    crate::scalar::RemainderFloor::remainder_floor(status),
    simd: remainder_floor_bulk,
);
define_op2!(
    /// Element-wise Euclidean remainder of two arrays.
    ///
    /// See [`jix::scalar::RemainderEuclid`](crate::scalar::RemainderEuclid) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![7i32, -7, 7, -7])?;
    /// let b = Array::compact_ndarray(&array![3i32, 3, -3, -3])?;
    /// let result = a.remainder_euclid(b).to_ndarray()?;
    /// assert_eq!(result, array![1, 2, 1, 2]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    RemainderEuclid,
    RemainderEuclidKernel,
    crate::scalar::RemainderEuclid::remainder_euclid(status),
    simd: remainder_euclid_bulk,
);
define_op2!(
    /// Element-wise exponentiation (`a` raised to the power `b`).
    ///
    /// See [`jix::scalar::Pow`](crate::scalar::Pow) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::pow()`](crate::Array::pow).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![2.0f32, 3.0, 4.0])?;
    /// let b = Array::compact_ndarray(&array![3.0f32, 2.0, 0.5])?;
    /// let result = a.pow(b).to_ndarray()?;
    /// assert_eq!(result, array![8.0, 9.0, 2.0]);
    ///
    /// // Raise each element to a constant exponent.
    /// let a = Array::compact_ndarray(&array![2.0f32, 3.0, 4.0])?;
    /// let result = a.map(|x| x.powf(2.0)).to_ndarray()?;
    /// assert_eq!(result, array![4.0, 9.0, 16.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Pow,
    PowKernel,
    crate::scalar::Pow::pow(status),
    simd: pow_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(pow: Pow, crate::scalar::Pow);
    define_array_op2_method!(divide_trunc: DivideTrunc, crate::scalar::DivideTrunc);
    define_array_op2_method!(divide_floor: DivideFloor, crate::scalar::DivideFloor);
    define_array_op2_method!(
        divide_euclid: DivideEuclid,
        crate::scalar::DivideEuclid
    );
    define_array_op2_method!(
        remainder_trunc: RemainderTrunc,
        crate::scalar::RemainderTrunc
    );
    define_array_op2_method!(
        remainder_floor: RemainderFloor,
        crate::scalar::RemainderFloor
    );
    define_array_op2_method!(
        remainder_euclid: RemainderEuclid,
        crate::scalar::RemainderEuclid
    );
}

#[cfg(test)]
pub(crate) mod tests {
    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::{AddKernel, DivKernel, MulKernel, Op2Kernel, SubKernel};
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $t:ty, $f:expr) => {
                    let (a, b) = (<$t>::simd_test_values(0), <$t>::simd_test_values(5));
                    let what = concat!(stringify!($kernel), " ", stringify!($t));
                    assert_same_elements(
                        $kernel.apply_bulk(a, b, &mut Default::default(), simd),
                        |i| $f(a[i], b[i]),
                        what,
                    );
                };
            }
            macro_rules! ints {
                ($kernel:ident, $f:ident) => {
                    case!($kernel, i8, i8::$f);
                    case!($kernel, i16, i16::$f);
                    case!($kernel, i32, i32::$f);
                    case!($kernel, i64, i64::$f);
                    case!($kernel, u8, u8::$f);
                    case!($kernel, u16, u16::$f);
                    case!($kernel, u32, u32::$f);
                    case!($kernel, u64, u64::$f);
                };
            }
            simd.vectorize(|| {
                case!(AddKernel, f32, |x, y| x + y);
                case!(AddKernel, f64, |x, y| x + y);
                ints!(AddKernel, wrapping_add);
                case!(SubKernel, f32, |x, y| x - y);
                case!(SubKernel, f64, |x, y| x - y);
                ints!(SubKernel, wrapping_sub);
                case!(MulKernel, f32, |x, y| x * y);
                case!(MulKernel, f64, |x, y| x * y);
                case!(MulKernel, i16, i16::wrapping_mul);
                case!(MulKernel, i32, i32::wrapping_mul);
                case!(MulKernel, i64, i64::wrapping_mul);
                case!(MulKernel, u8, u8::wrapping_mul);
                case!(MulKernel, u16, u16::wrapping_mul);
                case!(MulKernel, u32, u32::wrapping_mul);
                case!(MulKernel, u64, u64::wrapping_mul);
                case!(DivKernel, f32, |x, y| x / y);
                case!(DivKernel, f64, |x, y| x / y);
                // Not a multiple of the vector length: scalar.
                assert_eq!(
                    AddKernel.apply_bulk([1.5f32], [2.0], &mut Default::default(), simd),
                    [3.5]
                );
            });
        }
        for_each_simd_level!(check);
    }

    /// Float `(a, b)` test pairs: every pair of special values, then random bits, and random `b`
    /// with `a` a multiple of it whose quotient is near an integer, near the exactness bound
    /// `$exact`, or far past it.
    macro_rules! float_test_pairs {
        ($t:ty, $from_bits:expr, $from_f64:expr, $exact:expr) => {{
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let specials = [
                0.0,
                -0.0,
                1.0,
                -1.0,
                0.1,
                -2.5,
                3.0,
                1e30,
                f64::MAX,
                f64::MIN_POSITIVE,
            ]
            .into_iter()
            .chain([5e-324, f64::INFINITY, f64::NEG_INFINITY, f64::NAN])
            .map($from_f64)
            .chain([$from_bits(1)]) // the smallest subnormal
            .collect::<Vec<$t>>();
            let mut pairs: Vec<($t, $t)> = specials
                .iter()
                .flat_map(|&a| specials.iter().map(move |&b| (a, b)))
                .collect();
            for i in 0..100_000 {
                let b: $t = $from_bits(next());
                let a: $t = match i % 4 {
                    0 => $from_bits(next()),
                    1 => b * $from_f64((next() % (1 << 30)) as f64 - (1 << 29) as f64),
                    2 => b * $from_f64($exact + (next() % 64) as f64 - 32.0),
                    _ => b * $from_f64((next() % (1 << 40)) as f64 * $exact),
                };
                pairs.push((a, b));
            }
            pairs
        }};
    }

    /// The SIMD bodies of the float divisions and remainders against their scalar ones, bit for
    /// bit, on every SIMD level of the CPU.
    #[test]
    fn div_rem_simd_bodies_all_levels() {
        use super::{
            DivideEuclidKernel, DivideFloorKernel, DivideTruncKernel, Op2Kernel,
            RemainderEuclidKernel, RemainderFloorKernel, RemainderTruncKernel,
        };
        use crate::util::for_each_simd_level;
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($t:ident, $pairs:expr) => {{
                    const N: usize = 64;
                    let mut pairs: Vec<($t, $t)> = $pairs;
                    pairs.resize(pairs.len().next_multiple_of(N), (1.0, 1.0));
                    let same =
                        |x: $t, y: $t| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
                    for chunk in pairs.chunks_exact(N) {
                        let a: [$t; N] = std::array::from_fn(|i| chunk[i].0);
                        let b: [$t; N] = std::array::from_fn(|i| chunk[i].1);
                        macro_rules! kernel {
                            ($kernel:expr) => {
                                let bulk = $kernel.apply_bulk(a, b, &mut Default::default(), simd);
                                for i in 0..N {
                                    let scalar = $kernel.apply(a[i], b[i], &mut Default::default());
                                    assert!(
                                        same(bulk[i], scalar),
                                        "{:e}, {:e}: {:e} != {scalar:e}",
                                        a[i],
                                        b[i],
                                        bulk[i]
                                    );
                                }
                            };
                        }
                        kernel!(DivideTruncKernel);
                        kernel!(DivideFloorKernel);
                        kernel!(DivideEuclidKernel);
                        kernel!(RemainderTruncKernel);
                        kernel!(RemainderFloorKernel);
                        kernel!(RemainderEuclidKernel);
                    }
                }};
            }
            simd.vectorize(|| {
                case!(
                    f32,
                    float_test_pairs!(f32, |x| f32::from_bits(x as u32), |x| x as f32, 16777216.0)
                );
                case!(
                    f64,
                    float_test_pairs!(f64, f64::from_bits, |x| x, 9007199254740992.0)
                );
            });
        }
        for_each_simd_level!(check);
    }

    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use ndarray::{Ix1, Ix2, IxDyn};
    use proptest::strategy::BoxedStrategy;
    use proptest::test_runner::{Config, TestRunner};

    /// Shared proptest driver for binary-op tests.
    ///
    /// Generic over the dtype only, with the op passed as a fn pointer, to avoid per-op
    /// monomorphization.
    #[allow(clippy::type_complexity)]
    #[inline(never)]
    pub(crate) fn check_op2<T>(
        strategy: BoxedStrategy<T>,
        check: fn(
            &ndarray::ArrayD<T>,
            &ndarray::ArrayD<T>,
            crate::util::TestArray<T>,
            crate::util::TestArray<T>,
        ),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(
                &crate::util::arrays2_strategy_generic::<T>(
                    crate::util::shape_strategy(),
                    strategy,
                ),
                |((nd_a, za), (nd_b, zb))| {
                    check(&nd_a, &nd_b, za, zb);
                    Ok(())
                },
            )
            .unwrap();
    }

    macro_rules! test_op2_dtype {
        ($op_method:ident, |$a:ident, $b:ident| $body:expr, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                // `$body` is shared across all dtypes this macro is invoked with. For ops
                // like `greater`/`less` that are also instantiated at `bool`, an ordering
                // comparison such as `a > b` is a plain boolean comparison there rather than
                // a numeric one (clippy::bool_comparison).
                #[test]
                #[allow(clippy::bool_comparison)]
                fn [<$op_method _ $dtype>]() {
                    crate::ops::op2::tests::check_op2::<$dtype>(
                        <$dtype as crate::util::ScalarStrategy>::$strategy(),
                        |nd_a, nd_b, za, zb| {
                            #[allow(unused_imports)]
                            use core::ops::{Add, Div, Mul, Sub};
                            let result = za.$op_method(zb);
                            let expected =
                                ndarray::Zip::from(nd_a).and(nd_b).map_collect(|& $a, & $b| $body);
                            crate::util::assert_array_matches(&result, &expected);
                        },
                    );
                }
            }
        };
    }

    macro_rules! test_op2 {
        (
            $op_method:ident, |$a:ident, $b:ident| $body:expr,
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::op2::tests::test_op2_dtype!($op_method, |$a, $b| $body, $dtype, $strategy);)+
            $($(
                #[cfg($cfg)]
                crate::ops::op2::tests::test_op2_dtype!($op_method, |$a, $b| $body, $cfg_dtype, $strategy);
            )+)*
        };
    }

    pub(crate) use {test_op2, test_op2_dtype};

    test_op2!(
        add,
        |a, b| a.wrapping_add(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op2!(
        sub,
        |a, b| a.wrapping_sub(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op2!(
        mul,
        |a, b| a.wrapping_mul(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op2!(
        add,
        |a, b| a + b,
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );

    #[test]
    fn sub_concrete() {
        use crate::Array;

        // i32: negatives, zero, positive, at the op_safe_strategy bound (+/-100); differences
        // stay well within i32 range so there is no overflow panic in the reference either.
        let nd_a = ndarray::array![[-100i32, -1, 0], [1, 50, 100]];
        let nd_b = ndarray::array![[100i32, 1, 0], [-1, -50, -100]];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a - b);
        crate::util::assert_array_matches(&(za - zb), &expected);

        // f64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default block
        // shape so this arm crosses a block boundary.
        let nd_a64 = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 50.0, 100.0]];
        let nd_b64 = ndarray::array![[100.0f64, 0.5, 0.0], [-0.5, -50.0, -100.0]];
        let za64 = Array::compact_ndarray_with(&nd_a64, crate::util::arr_params(&[1, 2])).unwrap();
        let zb64 = Array::compact_ndarray_with(&nd_b64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = ndarray::Zip::from(&nd_a64)
            .and(&nd_b64)
            .map_collect(|&a, &b| a - b);
        crate::util::assert_array_matches(&(za64 - zb64), &expected64);
    }

    #[test]
    fn mul_concrete() {
        use crate::Array;

        // i32: negatives, zero, positive, at the op_safe_strategy bound (+/-100); products
        // stay well within i32 range so there is no overflow panic in the reference either.
        let nd_a = ndarray::array![[-100i32, -1, 0], [1, 10, 100]];
        let nd_b = ndarray::array![[100i32, -1, 0], [-1, 10, -100]];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za * zb), &expected);

        // f64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default block
        // shape so this arm crosses a block boundary.
        let nd_a64 = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 10.0, 100.0]];
        let nd_b64 = ndarray::array![[100.0f64, -0.5, 0.0], [-0.5, 10.0, -100.0]];
        let za64 = Array::compact_ndarray_with(&nd_a64, crate::util::arr_params(&[1, 2])).unwrap();
        let zb64 = Array::compact_ndarray_with(&nd_b64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = ndarray::Zip::from(&nd_a64)
            .and(&nd_b64)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za64 * zb64), &expected64);
    }

    // mul, complex arm: full complex multiplication `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`,
    // with negative and zero components on both operands.
    #[cfg(feature = "num-complex")]
    #[test]
    fn mul_concrete_complex() {
        use crate::scalar::Complex;
        use crate::Array;

        let nd_a = ndarray::array![
            [
                Complex {
                    re: -3.0f32,
                    im: 4.0
                },
                Complex { re: 0.0, im: 0.0 }
            ],
            [
                Complex { re: 2.5, im: -1.5 },
                Complex {
                    re: 100.0,
                    im: -100.0
                }
            ],
        ];
        let nd_b = ndarray::array![
            [
                Complex {
                    re: 1.0f32,
                    im: -2.0
                },
                Complex { re: 5.0, im: 5.0 }
            ],
            [
                Complex { re: -2.5, im: 1.5 },
                Complex {
                    re: -100.0,
                    im: 100.0
                }
            ],
        ];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za * zb), &expected);
    }

    test_op2!(
        div,
        |a, b| a / b,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_non_zero_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );

    /// The floored remainder, from the truncated one: adjusted to the sign of `b`.
    fn remainder_floor_ref<T>(a: T, b: T) -> T
    where
        T: Copy + PartialOrd + Default + core::ops::Rem<Output = T> + core::ops::Add<Output = T>,
    {
        let zero = T::default();
        let r = a % b;
        if r != zero && (r < zero) != (b < zero) {
            r + b
        } else {
            r
        }
    }

    test_op2!(
        remainder_trunc,
        |a, b| a % b,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_non_zero_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_op2!(
        remainder_floor,
        |a, b| remainder_floor_ref(a, b),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_non_zero_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_op2!(
        remainder_euclid,
        |a, b| a.rem_euclid(b),
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_non_zero_strategy
    );
    #[cfg(feature = "half")]
    test_op2!(
        remainder_euclid,
        |a, b| f16::from_f32(a.to_f32().rem_euclid(b.to_f32())),
        [f16],
        op_safe_non_zero_strategy
    );

    test_op2!(
        divide_trunc,
        |a, b| a / b,
        [i8, i16, i32, i64, u8, u16, u32, u64],
        op_safe_non_zero_strategy
    );
    test_op2!(
        divide_floor,
        |a, b| (a - remainder_floor_ref(a, b)) / b,
        [i8, i16, i32, i64, u8, u16, u32, u64],
        op_safe_non_zero_strategy
    );
    test_op2!(
        divide_euclid,
        |a, b| a.div_euclid(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        op_safe_non_zero_strategy
    );
    // Floats: the quotient of the remainder, which is exact for the truncated one (`fmod`), rounded
    // off. Exact quotients for all inputs are checked in `divide_float_exact`.
    test_op2!(
        divide_trunc,
        |a, b| ((a - a % b) / b).round(),
        [f32, f64],
        op_safe_non_zero_strategy
    );
    test_op2!(
        divide_floor,
        |a, b| ((a - remainder_floor_ref(a, b)) / b).round(),
        [f32, f64],
        op_safe_non_zero_strategy
    );
    test_op2!(
        divide_euclid,
        |a, b| ((a - a.rem_euclid(b)) / b).round(),
        [f32, f64],
        op_safe_non_zero_strategy
    );

    /// `(m, e)` with `x == m * 2^e`, for a finite `x`, from its bits:
    /// https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    fn float_parts(x: f64) -> (i128, i32) {
        let bits = x.to_bits();
        let (exp, frac) = (
            ((bits >> 52) & 0x7ff) as i32,
            (bits & ((1 << 52) - 1)) as i128,
        );
        let (m, e) = if exp == 0 {
            (frac, -1074)
        } else {
            (frac | (1 << 52), exp - 1075)
        };
        (if x.is_sign_negative() { -m } else { m }, e)
    }

    /// The exact truncated, floored and Euclidean quotients of `a / b`, if they fit an `i128`.
    fn exact_quotients(a: f64, b: f64) -> Option<[i128; 3]> {
        let ((ma, ea), (mb, eb)) = (float_parts(a), float_parts(b));
        let e = ea.min(eb);
        if ea - e > 70 || eb - e > 70 {
            return None;
        }
        let (na, nb) = (ma << (ea - e), mb << (eb - e)); // a == na * 2^e, b == nb * 2^e
        let trunc = na / nb;
        let floor = if na % nb != 0 && (na < 0) != (nb < 0) {
            trunc - 1
        } else {
            trunc
        };
        Some([trunc, floor, na.div_euclid(nb)])
    }

    /// The float quotients against the exact ones, rounded: equal below `2^24` (`f32`) and `2^53`
    /// (`f64`) in magnitude, and within one ulp above.
    #[test]
    fn divide_float_exact() {
        use crate::scalar::{DivideEuclid, DivideFloor, DivideTrunc};
        let s = &mut crate::error::ArithmeticOpStatus::default(); // floats report no errors

        macro_rules! check {
            ($t:ty, $pairs:expr, $exact:expr, $to_f64:expr, $round:expr) => {
                for (a, b) in $pairs {
                    let (af, bf): (f64, f64) = ($to_f64(a), $to_f64(b));
                    if !af.is_finite() || !bf.is_finite() || bf == 0.0 {
                        continue;
                    }
                    let Some(exact) = exact_quotients(af, bf) else {
                        continue;
                    };
                    let actual = [
                        a.divide_trunc(b, s),
                        a.divide_floor(b, s),
                        a.divide_euclid(b, s),
                    ];
                    for (q, x) in exact.into_iter().zip(actual) {
                        let expected: $t = $round(q);
                        let what = format!("{a:e} / {b:e}: {x:e}, expected {expected:e}");
                        if q.unsigned_abs() < $exact {
                            assert!(x == expected, "{what}");
                        } else {
                            let ulps = (x.to_bits() as i128 - expected.to_bits() as i128).abs();
                            assert!(ulps <= 1, "{what}");
                        }
                    }
                }
            };
        }
        let f64_pairs = float_test_pairs!(f64, f64::from_bits, |x| x, 9007199254740992.0);
        check!(f64, f64_pairs, 1u128 << 53, |x| x, |q: i128| q as f64);
        let f32_pairs =
            float_test_pairs!(f32, |x| f32::from_bits(x as u32), |x| x as f32, 16777216.0);
        check!(f32, f32_pairs, 1u128 << 24, |x| x as f64, |q: i128| q
            as f32);
        // Exact for every f16: its quotients that are not exact in f32 are too large for f16.
        #[cfg(feature = "half")]
        {
            let pairs = float_test_pairs!(f16, |x| f16::from_bits(x as u16), f16::from_f64, 2048.0);
            check!(f16, pairs, u128::MAX, f16::to_f64, |q: i128| f16::from_f64(
                q as f64
            ));
        }
    }

    /// The float remainders against `fmod` (`%`), CPython's `%` and [`f64::rem_euclid`], bit for
    /// bit.
    #[test]
    fn remainder_float_matches_fmod() {
        use crate::scalar::{RemainderEuclid, RemainderFloor, RemainderTrunc};
        let s = &mut crate::error::ArithmeticOpStatus::default(); // floats report no errors

        macro_rules! check {
            ($t:ty, $pairs:expr, $refs:expr) => {{
                let (trunc_ref, floor_ref, euclid_ref) = $refs;
                let same = |x: $t, y: $t| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
                for (a, b) in $pairs {
                    let what = format!("{a:e} % {b:e}");
                    assert!(
                        same(a.remainder_trunc(b, s), trunc_ref(a, b)),
                        "trunc {what}"
                    );
                    assert!(
                        same(a.remainder_floor(b, s), floor_ref(a, b)),
                        "floor {what}"
                    );
                    assert!(
                        same(a.remainder_euclid(b, s), euclid_ref(a, b)),
                        "euclid {what}"
                    );
                }
            }};
        }
        macro_rules! fmod_refs {
            ($t:ty) => {
                (
                    |a: $t, b: $t| a % b,
                    |a: $t, b: $t| {
                        let r = a % b;
                        if r == 0.0 {
                            <$t>::copysign(0.0, b)
                        } else if (r < 0.0) != (b < 0.0) {
                            r + b
                        } else {
                            r
                        }
                    },
                    |a: $t, b: $t| a.rem_euclid(b),
                )
            };
        }
        let f64_pairs = float_test_pairs!(f64, f64::from_bits, |x| x, 9007199254740992.0);
        check!(f64, f64_pairs, fmod_refs!(f64));
        let f32_pairs =
            float_test_pairs!(f32, |x| f32::from_bits(x as u32), |x| x as f32, 16777216.0);
        check!(f32, f32_pairs, fmod_refs!(f32));
        #[cfg(feature = "half")]
        {
            let (trunc, floor, euclid) = fmod_refs!(f32);
            let via_f32 = |f: fn(f32, f32) -> f32| {
                move |a: f16, b: f16| f16::from_f32(f(a.to_f32(), b.to_f32()))
            };
            let pairs = float_test_pairs!(f16, |x| f16::from_bits(x as u16), f16::from_f64, 2048.0);
            check!(
                f16,
                pairs,
                (via_f32(trunc), via_f32(floor), via_f32(euclid))
            );
        }
    }

    /// CPython's float `//`, as numpy's `floor_divide` (`_float_div_mod` in
    /// https://github.com/python/cpython/blob/main/Objects/floatobject.c).
    fn py_floor_divide(a: f64, b: f64) -> f64 {
        if b == 0.0 {
            return a / b;
        }
        let m = a % b;
        let mut div = (a - m) / b;
        if m != 0.0 && (b < 0.0) != (m < 0.0) {
            div -= 1.0;
        }
        if div == 0.0 {
            return f64::copysign(0.0, a / b);
        }
        let floor = div.floor();
        if div - floor > 0.5 {
            floor + 1.0
        } else {
            floor
        }
    }

    /// Infinities, NaN, zero divisors and signed zeros as numpy's `floor_divide`; the truncated
    /// and Euclidean quotients from it.
    #[test]
    fn divide_float_special_values() {
        use crate::scalar::{DivideEuclid, DivideFloor, DivideTrunc};
        let s = &mut crate::error::ArithmeticOpStatus::default(); // floats report no errors

        let same = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            -2.5,
            1e-300,
            -5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for a in values {
            for b in values {
                let negative = a != 0.0 && (a < 0.0) != (b < 0.0);
                let trunc = if negative {
                    -py_floor_divide(-a, b)
                } else {
                    py_floor_divide(a, b)
                };
                let euclid = if b < 0.0 {
                    -py_floor_divide(a, -b)
                } else {
                    py_floor_divide(a, b)
                };
                let what = format!("{a:e} / {b:e}");
                assert!(
                    same(a.divide_floor(b, s), py_floor_divide(a, b)),
                    "floor {what}"
                );
                assert!(same(a.divide_trunc(b, s), trunc), "trunc {what}");
                assert!(same(a.divide_euclid(b, s), euclid), "euclid {what}");
            }
        }
    }

    // The three divisions with every sign combination, each paired with its remainder.
    #[test]
    fn divide_concrete() {
        use crate::Array;

        let nd_a = ndarray::array![[7i32, -7, 7, -7], [0, 100, -100, 5]];
        let nd_b = ndarray::array![[2i32, 2, -2, -2], [5, 7, 7, -100]];
        let arrays = || {
            let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[1, 2])).unwrap();
            (za, Array::compact_ndarray(&nd_b).unwrap())
        };
        let (za, zb) = arrays();
        let trunc = za.divide_trunc(zb).to_ndarray().unwrap();
        assert_eq!(trunc, ndarray::array![[3, -3, -3, 3], [0, 14, -14, 0]]);
        let (za, zb) = arrays();
        let floor = za.divide_floor(zb).to_ndarray().unwrap();
        assert_eq!(floor, ndarray::array![[3, -4, -4, 3], [0, 14, -15, -1]]);
        let (za, zb) = arrays();
        let euclid = za.divide_euclid(zb).to_ndarray().unwrap();
        assert_eq!(euclid, ndarray::array![[3, -4, -3, 4], [0, 14, -15, 0]]);
        let (za, zb) = arrays();
        let r_trunc = za.remainder_trunc(zb).to_ndarray::<ndarray::Ix2>().unwrap();
        let (za, zb) = arrays();
        let r_floor = za.remainder_floor(zb).to_ndarray::<ndarray::Ix2>().unwrap();
        let (za, zb) = arrays();
        let r_euclid = za
            .remainder_euclid(zb)
            .to_ndarray::<ndarray::Ix2>()
            .unwrap();
        for (q, r) in [(trunc, r_trunc), (floor, r_floor), (euclid, r_euclid)] {
            assert_eq!(&q * &nd_b + &r, nd_a);
        }

        // Of the exact quotient: `0.1` fits in `1.0` 9 times, though `1.0 / 0.1` rounds to `10.0`.
        let nd_f = ndarray::array![1.0f64, -7.5, 7.5, -1e-20, -5.0];
        let nd_g = ndarray::array![0.1f64, 2.0, -2.0, 1.0, f64::INFINITY];
        let floats = || {
            let zf = Array::compact_ndarray(&nd_f).unwrap();
            (zf, Array::compact_ndarray(&nd_g).unwrap())
        };
        let (zf, zg) = floats();
        let result = zf.divide_trunc(zg).to_ndarray().unwrap();
        assert_eq!(result, ndarray::array![9.0, -3.0, -3.0, 0.0, 0.0]);
        let (zf, zg) = floats();
        let result = zf.divide_floor(zg).to_ndarray().unwrap();
        assert_eq!(result, ndarray::array![9.0, -4.0, -4.0, -1.0, -1.0]);
        let (zf, zg) = floats();
        let result = zf.divide_euclid(zg).to_ndarray().unwrap();
        assert_eq!(result, ndarray::array![9.0, -4.0, -3.0, -1.0, -1.0]);
    }

    // The three remainders with every sign combination. The truncated one is `%` and pairs with
    // the truncating div so that `a == (a / b) * b + a % b`.
    #[test]
    fn remainder_concrete() {
        use crate::Array;

        let nd_a = ndarray::array![[7i32, -7, 7, -7], [0, 100, -100, 5]];
        let nd_b = ndarray::array![[3i32, 3, -3, -3], [5, 7, 7, -100]];
        let arrays = || {
            let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[1, 2])).unwrap();
            (za, Array::compact_ndarray(&nd_b).unwrap())
        };
        let trunc = ndarray::array![[1i32, -1, 1, -1], [0, 2, -2, 5]];
        let (za, zb) = arrays();
        crate::util::assert_array_matches(&(za % zb), &trunc);
        let (za, zb) = arrays();
        crate::util::assert_array_matches(&za.remainder_trunc(zb), &trunc);
        let (za, zb) = arrays();
        let floor = ndarray::array![[1i32, 2, -2, -1], [0, 2, 5, -95]];
        crate::util::assert_array_matches(&za.remainder_floor(zb), &floor);
        let (za, zb) = arrays();
        let euclid = ndarray::array![[1i32, 2, 1, 2], [0, 2, 5, 5]];
        crate::util::assert_array_matches(&za.remainder_euclid(zb), &euclid);

        let (za, zb) = arrays();
        let quotient = (za / zb).to_ndarray::<ndarray::Ix2>().unwrap();
        let reconstructed = &quotient * &nd_b + &trunc;
        assert_eq!(reconstructed.into_dyn(), nd_a.clone().into_dyn());

        let nd_f = ndarray::array![-7.5f64, 7.5, -7.5, 7.5, 1.0, -4.0, 4.0, -5.0];
        let nd_g = ndarray::array![2.0f64, 2.0, -2.0, -2.0, 0.0, 2.0, -2.0, f64::INFINITY];
        let floats = || {
            (
                Array::compact_ndarray(&nd_f).unwrap(),
                Array::compact_ndarray(&nd_g).unwrap(),
            )
        };
        // `1.0 % 0.0` is invalid: read with the error ignored.
        fn read<S: crate::storage::ArrayStorageTyped>(r: Array<S>) -> ndarray::Array1<S::Item> {
            use crate::error::{
                ArithmeticErrorKind::InvalidOperation, ArithmeticErrorMode::Ignore,
            };
            let mut ctx = r.read_ctx();
            ctx.set_arithmetic_error_mode(InvalidOperation, Ignore);
            let index = r.shape().iter().map(|&d| 0..d).collect::<Vec<_>>();
            r.to_ndarray_sub(&index, &ctx).unwrap()
        }
        let (zf, zg) = floats();
        let result = read(zf.remainder_trunc(zg));
        assert_eq!(
            result.slice(ndarray::s![..4]),
            ndarray::array![-1.5, 1.5, -1.5, 1.5]
        );
        assert!(result[4].is_nan());
        assert_eq!(result[7], -5.0);
        let (zf, zg) = floats();
        let result = read(zf.remainder_floor(zg));
        assert_eq!(
            result.slice(ndarray::s![..4]),
            ndarray::array![0.5, 1.5, -1.5, -0.5]
        );
        assert!(result[4].is_nan());
        // A zero result takes the sign of the divisor, and an infinite divisor of the opposite
        // sign gives that infinity, as in Python.
        assert!(result[5] == 0.0 && result[5].is_sign_positive());
        assert!(result[6] == 0.0 && result[6].is_sign_negative());
        assert_eq!(result[7], f64::INFINITY);
        let (zf, zg) = floats();
        let result = read(zf.remainder_euclid(zg));
        assert_eq!(
            result.slice(ndarray::s![..4]),
            ndarray::array![0.5, 1.5, 0.5, 1.5]
        );
        assert!(result[4].is_nan());
        assert_eq!(result[7], f64::INFINITY);
    }

    // three_arrays_f32: (za + zb) * zc lazy chain equals (&a + &b) * &c.
    // Fixed 2*3 shape with two different block shapes to exercise block-boundary handling.
    proptest::proptest! {
        #[test]
        fn three_arrays_f32(
            a_vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize),
            b_vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize),
            c_vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize),
        ) {
            use crate::array::Array;
            let nd_a = ndarray::Array::from_shape_vec([2, 3], a_vals).unwrap();
            let nd_b = ndarray::Array::from_shape_vec([2, 3], b_vals).unwrap();
            let nd_c = ndarray::Array::from_shape_vec([2, 3], c_vals).unwrap();
            let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[2, 3])).unwrap();
            let zb = Array::compact_ndarray_with(&nd_b, crate::util::arr_params(&[2, 3])).unwrap();
            let zc = Array::compact_ndarray_with(&nd_c, crate::util::arr_params(&[1, 2])).unwrap();
            let result = (za + zb) * zc;
            let expected = (&nd_a + &nd_b) * &nd_c;
            crate::util::assert_array_matches(&result, &expected);
        }
    }

    // Integer division by zero gives `0` for the element and reports `ArithmeticErrorKind::DivideByZero`
    // to the read context, which raises it by default.
    macro_rules! test_div_by_zero {
        ($($t:ident),*) => {$(
            paste::paste! {
                #[test]
                fn [<div_by_zero_ $t>]() {
                    use crate::Array;
                    use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};

                    // Zero divisors in both the bulk part of the loop and its scalar tail.
                    let n = 1000;
                    let nd_a = ndarray::Array1::from_shape_fn(n, |i| (i % 100) as $t);
                    let mut nd_b = ndarray::Array1::from_shape_fn(n, |i| (i % 7 + 1) as $t);
                    nd_b[300] = 0;
                    nd_b[n - 1] = 0;
                    let expected = ndarray::Zip::from(&nd_a)
                        .and(&nd_b)
                        .map_collect(|&x, &y| if y == 0 { 0 } else { x / y });
                    let q = Array::compact_ndarray(&nd_a).unwrap()
                        / Array::compact_ndarray(&nd_b).unwrap();
                    let full = [0..n as u64];

                    // Raise, the default: the read fails.
                    let err = q.to_ndarray::<Ix1>().unwrap_err();
                    assert_eq!(err.kind(), ErrorKind::Arithmetic);
                    assert_eq!(err.message(), "divide by zero encountered");
                    // Only the elements read count.
                    let ctx = q.read_ctx();
                    let head = q.to_ndarray_sub(&[0..300], &ctx).unwrap();
                    assert_eq!(head, expected.slice(ndarray::s![0..300]));

                    // Warn: the value, and a single warning however many elements and reads hit it.
                    let mut ctx = q.read_ctx();
                    ctx.set_arithmetic_error_mode(ArithmeticErrorKind::DivideByZero, ArithmeticErrorMode::Warn);
                    for _ in 0..2 {
                        assert_eq!(q.to_ndarray_sub(&full, &ctx).unwrap(), expected);
                    }
                    let warnings = ctx.take_warnings();
                    assert_eq!(warnings.len(), 1);
                    assert_eq!(warnings[0].kind(), ErrorKind::Arithmetic);
                    assert_eq!(warnings[0].message(), "divide by zero encountered");
                    assert!(ctx.take_warnings().is_empty());

                    // Ignore: just the value.
                    ctx.set_arithmetic_error_mode(ArithmeticErrorKind::DivideByZero, ArithmeticErrorMode::Ignore);
                    assert_eq!(q.to_ndarray_sub(&full, &ctx).unwrap(), expected);
                    assert!(ctx.take_warnings().is_empty());
                }
            }
        )*};
    }
    test_div_by_zero!(i8, i16, i32, i64, u8, u16, u32, u64);

    #[test]
    fn div_by_zero_strided() {
        use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode};
        use crate::Array;

        // Transposed operands are read with a non-unit inner stride.
        let nd_a = ndarray::Array2::from_shape_fn((40, 30), |(i, j)| (i * 30 + j) as i32 - 600);
        let nd_b = ndarray::Array2::from_shape_fn((40, 30), |(i, j)| ((i + j) % 5) as i32 - 2);
        let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[8, 8])).unwrap();
        let zb = Array::compact_ndarray_with(&nd_b, crate::util::arr_params(&[8, 8])).unwrap();
        let q = za.transpose() / zb.transpose();
        assert!(q.to_ndarray::<Ix2>().is_err());

        let mut ctx = q.read_ctx();
        ctx.set_arithmetic_error_mode(
            ArithmeticErrorKind::DivideByZero,
            ArithmeticErrorMode::Ignore,
        );
        let expected = ndarray::Zip::from(&nd_a.t())
            .and(&nd_b.t())
            .map_collect(|&x, &y| if y == 0 { 0 } else { x / y });
        assert_eq!(q.to_ndarray_sub(&[0..30, 0..40], &ctx).unwrap(), expected);
    }

    #[test]
    fn div_min_by_minus_one_overflows() {
        use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};
        use crate::Array;

        macro_rules! check {
            ($($t:ident),*) => {$(
                let a = Array::compact_ndarray(&ndarray::array![$t::MIN, 7, $t::MAX]).unwrap();
                let b = Array::compact_ndarray(&ndarray::array![-1 as $t, -1, -1]).unwrap();
                let q = a / b;
                // Warn, the default for overflow: `MIN`.
                let ctx = q.read_ctx();
                let res = q.to_ndarray_sub::<ndarray::Ix1>(&[0..3], &ctx).unwrap();
                assert_eq!(res, ndarray::array![$t::MIN, -7, -$t::MAX], stringify!($t));
                let warnings = ctx.take_warnings();
                assert_eq!(warnings.len(), 1);
                assert_eq!(warnings[0].message(), "overflow encountered");
                // Without `MIN`, no error.
                q.to_ndarray_sub::<ndarray::Ix1>(&[1..3], &ctx).unwrap();
                assert!(ctx.take_warnings().is_empty());

                let mut ctx = q.read_ctx();
                ctx.set_arithmetic_error_mode(
                    ArithmeticErrorKind::Overflow,
                    ArithmeticErrorMode::Error,
                );
                let err = q.to_ndarray_sub::<ndarray::Ix1>(&[0..3], &ctx).unwrap_err();
                assert_eq!(err.kind(), ErrorKind::Arithmetic);
            )*};
        }
        check!(i8, i16, i32, i64);
    }

    #[test]
    fn div_by_zero_in_fused_chain() {
        use crate::error::ErrorKind;
        use crate::Array;

        // A fused chain runs as a single pipeline, whichever op runs it reports the error.
        let a = || Array::compact_ndarray(&ndarray::array![6i32, 7, 8]).unwrap();
        let b = || Array::compact_ndarray(&ndarray::array![2i32, 0, 4]).unwrap();
        for err in [
            (-(a() / b())).to_ndarray::<Ix1>().unwrap_err(),
            (a() / b()).cast::<f64>().to_ndarray::<Ix1>().unwrap_err(),
            ((a() / b()) + a()).to_ndarray::<Ix1>().unwrap_err(),
        ] {
            assert_eq!(err.kind(), ErrorKind::Arithmetic);
            assert_eq!(err.message(), "divide by zero encountered");
        }

        // Float division too.
        let fa = Array::compact_ndarray(&ndarray::array![1.0f32, 2.0]).unwrap();
        let fb = Array::compact_ndarray(&ndarray::array![0.0f32, 1.0]).unwrap();
        let err = (fa / fb)
            .map(|x: f32| x * 2.0)
            .to_ndarray::<Ix1>()
            .unwrap_err();
        assert_eq!(err.message(), "divide by zero encountered");
    }

    #[test]
    fn scalar_div_status() {
        use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        use crate::scalar::Div;

        let mut status = ArithmeticOpStatus::default();
        assert_eq!(Div::div(7u8, 2, &mut status), 3);
        assert_eq!(Div::div(-7i32, 2, &mut status), -3);
        assert_eq!(Div::div(1.0f32, 4.0, &mut status), 0.25);
        assert!(status.is_ok());
        assert_eq!(Div::div(-7i64, 0, &mut status), 0);
        assert!(status.contains(ArithmeticErrorKind::DivideByZero));
        assert!(!status.contains(ArithmeticErrorKind::Overflow));
        assert_eq!(Div::div(i32::MIN, -1, &mut status), i32::MIN);
        assert!(status.contains(ArithmeticErrorKind::Overflow));
    }

    /// The integer quotients and remainders of a zero divisor and of `MIN` by `-1`.
    #[test]
    fn int_division_errors() {
        use crate::error::ArithmeticErrorKind::{DivideByZero, Overflow};
        use crate::error::ArithmeticOpStatus;
        use crate::scalar::{
            Div, DivideEuclid, DivideFloor, DivideTrunc, RemainderEuclid, RemainderFloor,
            RemainderTrunc,
        };

        type F<T> = fn(T, T, &mut ArithmeticOpStatus) -> T;
        fn quotients<T>() -> [F<T>; 4]
        where
            T: Div<Output = T> + DivideTrunc<Output = T>,
            T: DivideFloor<Output = T> + DivideEuclid<Output = T>,
        {
            [
                Div::div,
                DivideTrunc::divide_trunc,
                DivideFloor::divide_floor,
                DivideEuclid::divide_euclid,
            ]
        }
        fn remainders<T>() -> [F<T>; 3]
        where
            T: RemainderTrunc<Output = T> + RemainderFloor<Output = T>,
            T: RemainderEuclid<Output = T>,
        {
            [
                RemainderTrunc::remainder_trunc,
                RemainderFloor::remainder_floor,
                RemainderEuclid::remainder_euclid,
            ]
        }
        fn check<T: Copy + PartialEq + core::fmt::Debug>(
            f: F<T>,
            (a, b): (T, T),
            expected: T,
            error: ArithmeticOpStatus,
        ) {
            let mut status = ArithmeticOpStatus::default();
            assert_eq!(f(a, b, &mut status), expected, "{a:?}, {b:?}");
            assert_eq!(status, error, "{a:?}, {b:?}");
        }

        macro_rules! check {
            ($($t:ty),*) => {$(
                for f in quotients::<$t>() {
                    check(f, (7, 3), 2, ArithmeticOpStatus::default());
                    check(f, (7, 0), 0, DivideByZero.into());
                    check(f, (<$t>::MIN, 0), 0, DivideByZero.into());
                }
                for f in remainders::<$t>() {
                    check(f, (7, 3), 1, ArithmeticOpStatus::default());
                    check(f, (7, 0), 0, DivideByZero.into());
                }
            )*};
        }
        check!(i8, i16, i32, i64, u8, u16, u32, u64);
        // `MIN` by `-1`: a quotient of `MIN` and an overflow, and an exact remainder of `0`.
        macro_rules! check_signed {
            ($($t:ty),*) => {$(
                for f in quotients::<$t>() {
                    check(f, (<$t>::MIN, -1), <$t>::MIN, Overflow.into());
                }
                for f in remainders::<$t>() {
                    check(f, (<$t>::MIN, -1), 0, ArithmeticOpStatus::default());
                }
            )*};
        }
        check_signed!(i8, i16, i32, i64);
    }

    // Random shapes, storage backends and layouts, with zero divisors in about a third of the
    // elements: an error iff some divisor is zero, and with errors ignored, the reference values.
    proptest::proptest! {
        #[test]
        fn div_by_zero_random(
            ((nd_a, za), (nd_b, zb)) in crate::util::arrays2_strategy_generic::<i32>(
                crate::util::shape_strategy(),
                <i32 as crate::util::ScalarStrategy>::comparable_strategy()
            )
        ) {
            use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};

            let q = za / zb;
            let index = q.shape().iter().map(|&d| 0..d).collect::<Vec<_>>();
            let has_zero = nd_b.iter().any(|&y| y == 0);
            let res = q.to_ndarray_sub::<IxDyn>(&index, &q.read_ctx());
            proptest::prop_assert_eq!(res.is_err(), has_zero);
            if let Err(err) = res {
                proptest::prop_assert_eq!(err.kind(), ErrorKind::Arithmetic);
            }

            let mut ctx = q.read_ctx();
            ctx.set_arithmetic_error_mode(ArithmeticErrorKind::DivideByZero, ArithmeticErrorMode::Ignore);
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|&x, &y| if y == 0 { 0 } else { x / y });
            proptest::prop_assert_eq!(q.to_ndarray_sub(&index, &ctx).unwrap(), expected);
        }
    }

    /// The errors of a float division of `a` by `b` with the result `q`, from the IEEE values.
    fn expected_div_errors(
        a: f64,
        b: f64,
        q: f64,
    ) -> [(crate::error::ArithmeticErrorKind, bool); 3] {
        use crate::error::ArithmeticErrorKind::*;
        let ordered = !a.is_nan() && !b.is_nan();
        let finite = a.is_finite() && b.is_finite();
        [
            (
                DivideByZero,
                ordered && b == 0.0 && a.is_finite() && a != 0.0,
            ),
            (InvalidOperation, ordered && q.is_nan()),
            (Overflow, finite && b != 0.0 && q.is_infinite()),
        ]
    }

    /// The errors of the float divisions for every pair of edge-case operands.
    #[test]
    fn float_div_errors() {
        use crate::error::ArithmeticOpStatus;
        use crate::scalar::{Div, DivideEuclid, DivideFloor, DivideTrunc};
        use crate::util::SimdTestValues;

        type F<T> = fn(T, T, &mut ArithmeticOpStatus) -> T;
        macro_rules! check {
            ($($t:ty),*) => {$(
                let values = <$t>::edge_cases();
                let ops: [(&str, F<$t>); 4] = [
                    ("div", Div::div),
                    ("divide_trunc", DivideTrunc::divide_trunc),
                    ("divide_floor", DivideFloor::divide_floor),
                    ("divide_euclid", DivideEuclid::divide_euclid),
                ];
                for &a in &values {
                    for &b in &values {
                        for (name, f) in ops {
                            let mut status = ArithmeticOpStatus::default();
                            let r = f(a, b, &mut status);
                            if name == "div" {
                                let q = a / b;
                                assert!(r.to_bits() == q.to_bits() || r.is_nan() && q.is_nan());
                            }
                            let errors = expected_div_errors(a.into(), b.into(), r.into());
                            for (kind, expected) in errors {
                                let what = format!("{name}({a}, {b}): {kind:?}");
                                assert_eq!(status.contains(kind), expected, "{what}");
                            }
                        }
                    }
                }
            )*};
        }
        check!(f32, f64);
    }

    #[cfg(feature = "half")]
    #[test]
    fn float_div_errors_f16() {
        use crate::error::ArithmeticOpStatus;
        use crate::scalar::{f16, Div, DivideEuclid, DivideFloor, DivideTrunc};

        type F<T> = fn(T, T, &mut ArithmeticOpStatus) -> T;
        let ops: [(&str, F<f16>); 4] = [
            ("div", Div::div),
            ("divide_trunc", DivideTrunc::divide_trunc),
            ("divide_floor", DivideFloor::divide_floor),
            ("divide_euclid", DivideEuclid::divide_euclid),
        ];
        // 60000 / 0.5 overflows f16, not the f32 it is computed in.
        let values = [
            0.0,
            -0.0,
            1.0,
            -3.0,
            0.5,
            60000.0,
            f32::INFINITY,
            -f32::INFINITY,
            f32::NAN,
        ];
        for a in values.map(f16::from_f32) {
            for b in values.map(f16::from_f32) {
                for (name, f) in ops {
                    let mut status = ArithmeticOpStatus::default();
                    let r = f(a, b, &mut status);
                    for (kind, expected) in expected_div_errors(a.into(), b.into(), r.into()) {
                        assert_eq!(
                            status.contains(kind),
                            expected,
                            "{name}({a}, {b}): {kind:?}"
                        );
                    }
                }
            }
        }
    }

    /// The errors of the float remainders for every pair of edge-case operands: `NaN`
    /// exactly for `x % 0` and `inf % x`, which are invalid.
    #[test]
    fn float_rem_errors() {
        use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        use crate::scalar::{RemainderEuclid, RemainderFloor, RemainderTrunc};
        use crate::util::SimdTestValues;

        type F<T> = fn(T, T, &mut ArithmeticOpStatus) -> T;
        macro_rules! check {
            ($($t:ty),*) => {$(
                let ops: [(&str, F<$t>); 3] = [
                    ("remainder_trunc", RemainderTrunc::remainder_trunc),
                    ("remainder_floor", RemainderFloor::remainder_floor),
                    ("remainder_euclid", RemainderEuclid::remainder_euclid),
                ];
                for &a in &<$t>::edge_cases() {
                    for &b in &<$t>::edge_cases() {
                        for (name, f) in ops {
                            let mut status = ArithmeticOpStatus::default();
                            let r = f(a, b, &mut status);
                            let ordered = !a.is_nan() && !b.is_nan();
                            let invalid = ordered && (b == 0.0 || a.is_infinite());
                            assert_eq!(r.is_nan(), invalid || !ordered, "{name}({a}, {b})");
                            let mut expected = ArithmeticOpStatus::default();
                            expected.set_if(ArithmeticErrorKind::InvalidOperation, invalid);
                            assert_eq!(status, expected, "{name}({a}, {b})");
                        }
                    }
                }
            )*};
        }
        check!(f32, f64);
    }

    #[cfg(feature = "half")]
    #[test]
    fn float_rem_errors_f16() {
        use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        use crate::scalar::{f16, RemainderEuclid, RemainderFloor, RemainderTrunc};

        type F<T> = fn(T, T, &mut ArithmeticOpStatus) -> T;
        let ops: [(&str, F<f16>); 3] = [
            ("remainder_trunc", RemainderTrunc::remainder_trunc),
            ("remainder_floor", RemainderFloor::remainder_floor),
            ("remainder_euclid", RemainderEuclid::remainder_euclid),
        ];
        let values = [
            0.0,
            -0.0,
            1.0,
            -3.0,
            0.5,
            60000.0,
            f32::INFINITY,
            -f32::INFINITY,
            f32::NAN,
        ];
        for a in values.map(f16::from_f32) {
            for b in values.map(f16::from_f32) {
                for (name, f) in ops {
                    let mut status = ArithmeticOpStatus::default();
                    f(a, b, &mut status);
                    let invalid = !a.is_nan() && !b.is_nan() && (b == f16::ZERO || a.is_infinite());
                    let mut expected = ArithmeticOpStatus::default();
                    expected.set_if(ArithmeticErrorKind::InvalidOperation, invalid);
                    assert_eq!(status, expected, "{name}({a}, {b})");
                }
            }
        }
    }

    /// The status of the SIMD bodies against the scalar functions', on every SIMD level: each
    /// pair of edge-case operands, in one lane of a bulk of ordinary ones.
    #[test]
    fn simd_bodies_status_all_levels() {
        use super::{
            DivKernel, DivideEuclidKernel, DivideFloorKernel, DivideTruncKernel, Op2Kernel,
            RemainderEuclidKernel, RemainderFloorKernel, RemainderTruncKernel,
        };
        use crate::error::ArithmeticOpStatus;
        use crate::util::{for_each_simd_level, SimdTestValues, SIMD_TEST_N};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $t:ty) => {
                    let values = <$t>::edge_cases();
                    for &x in &values {
                        for &y in &values {
                            let mut scalar = ArithmeticOpStatus::default();
                            $kernel.apply(x, y, &mut scalar);
                            for lane in [0, 9, 40, SIMD_TEST_N - 1] {
                                let mut a = [1.5 as $t; SIMD_TEST_N];
                                let mut b = [3.0 as $t; SIMD_TEST_N];
                                (a[lane], b[lane]) = (x, y);
                                let mut bulk = ArithmeticOpStatus::default();
                                $kernel.apply_bulk(a, b, &mut bulk, simd);
                                assert_eq!(
                                    bulk,
                                    scalar,
                                    "{} {}: {x} / {y} in lane {lane}",
                                    stringify!($kernel),
                                    stringify!($t)
                                );
                            }
                        }
                    }
                };
            }
            simd.vectorize(|| {
                case!(DivKernel, f32);
                case!(DivKernel, f64);
                case!(DivideTruncKernel, f32);
                case!(DivideTruncKernel, f64);
                case!(DivideFloorKernel, f32);
                case!(DivideFloorKernel, f64);
                case!(DivideEuclidKernel, f32);
                case!(DivideEuclidKernel, f64);
                case!(RemainderTruncKernel, f32);
                case!(RemainderTruncKernel, f64);
                case!(RemainderFloorKernel, f32);
                case!(RemainderFloorKernel, f64);
                case!(RemainderEuclidKernel, f32);
                case!(RemainderEuclidKernel, f64);
            });
        }
        for_each_simd_level!(check);
    }

    macro_rules! test_float_div_errors {
        ($($t:ident),*) => {$(
            paste::paste! {
                #[test]
                fn [<float_div_errors_read_ $t>]() {
                    use crate::Array;
                    use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};
                    #[cfg(feature = "half")]
                    #[allow(unused_imports)]
                    use crate::scalar::f16;

                    let c = |x: f64| <f64 as crate::scalar::Cast<$t>>::cast(x);
                    let n = 1000;
                    let inf = f64::INFINITY;
                    for (x, y, error) in [
                        (c(5.0), c(0.0), Some(ArithmeticErrorKind::DivideByZero)),
                        (c(0.0), c(0.0), Some(ArithmeticErrorKind::InvalidOperation)),
                        (c(inf), c(inf), Some(ArithmeticErrorKind::InvalidOperation)),
                        ($t::MAX, c(0.25), Some(ArithmeticErrorKind::Overflow)),
                        (c(f64::NAN), c(0.0), None),
                        (c(inf), c(0.0), None),
                        (c(-1.0), c(inf), None),
                    ] {
                        // In the bulk part of the loop, and in its scalar tail.
                        for i in [300, n - 1] {
                            let mut nd_a =
                                ndarray::Array1::from_shape_fn(n, |i| c((i % 100) as f64));
                            let mut nd_b =
                                ndarray::Array1::from_shape_fn(n, |i| c((i % 7 + 1) as f64));
                            (nd_a[i], nd_b[i]) = (x, y);
                            let q = Array::compact_ndarray(&nd_a).unwrap()
                                / Array::compact_ndarray(&nd_b).unwrap();
                            let what = format!("{x} / {y} at {i}");

                            let mut ctx = q.read_ctx();
                            for kind in [
                                ArithmeticErrorKind::DivideByZero,
                                ArithmeticErrorKind::Overflow,
                                ArithmeticErrorKind::InvalidOperation,
                            ] {
                                ctx.set_arithmetic_error_mode(kind, ArithmeticErrorMode::Error);
                            }
                            let res = q.to_ndarray_sub::<ndarray::Ix1>(&[0..n as u64], &ctx);
                            let Some(error) = error else {
                                res.expect(&what);
                                continue;
                            };
                            let err = res.expect_err(&what);
                            assert_eq!(err.kind(), ErrorKind::Arithmetic);
                            let message = match error {
                                ArithmeticErrorKind::DivideByZero => "divide by zero encountered",
                                ArithmeticErrorKind::Overflow => "overflow encountered",
                                _ => "invalid operation encountered",
                            };
                            assert_eq!(err.message(), message, "{what}");

                            // Ignored: the IEEE values.
                            ctx.set_arithmetic_error_mode(error, ArithmeticErrorMode::Ignore);
                            let res = q
                                .to_ndarray_sub::<ndarray::Ix1>(&[0..n as u64], &ctx)
                                .unwrap();
                            let expected =
                                ndarray::Zip::from(&nd_a).and(&nd_b).map_collect(|&x, &y| x / y);
                            let same = |(r, e): (&$t, &$t)| r == e || r.is_nan() && e.is_nan();
                            assert!(res.iter().zip(&expected).all(same), "{what}");
                        }
                    }
                }
            }
        )*};
    }
    test_float_div_errors!(f32, f64);
    #[cfg(feature = "half")]
    test_float_div_errors!(f16);

    #[test]
    fn float_div_errors_strided() {
        use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode};
        use crate::Array;

        // Transposed operands are read with a non-unit inner stride, by the scalar path.
        let nd_a = ndarray::Array2::from_shape_fn((40, 30), |(i, j)| (i * 30 + j) as f32 - 600.0);
        let nd_b = ndarray::Array2::from_shape_fn((40, 30), |(i, j)| ((i + j) % 5) as f32 - 2.0);
        let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[8, 8])).unwrap();
        let zb = Array::compact_ndarray_with(&nd_b, crate::util::arr_params(&[8, 8])).unwrap();
        let q = za.transpose() / zb.transpose();
        let err = q.to_ndarray::<Ix2>().unwrap_err();
        assert_eq!(err.message(), "divide by zero encountered");

        let mut ctx = q.read_ctx();
        ctx.set_arithmetic_error_mode(
            ArithmeticErrorKind::DivideByZero,
            ArithmeticErrorMode::Ignore,
        );
        let expected = ndarray::Zip::from(&nd_a.t())
            .and(&nd_b.t())
            .map_collect(|&x, &y| x / y);
        assert_eq!(q.to_ndarray_sub(&[0..30, 0..40], &ctx).unwrap(), expected);
    }

    #[test]
    fn int_pow() {
        use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        use crate::scalar::Pow;

        let mut status = ArithmeticOpStatus::default();
        // Wraps on overflow.
        assert_eq!(Pow::pow(3i32, 40u32, &mut status), 689956897);
        assert_eq!(Pow::pow(3u8, 9u8, &mut status), 227);
        assert_eq!(Pow::pow(-2i8, 7i8, &mut status), -128);
        assert_eq!(Pow::pow(-3i64, 0i64, &mut status), 1);
        assert_eq!(Pow::pow(0u16, 0i16, &mut status), 1);
        // Exponents beyond `u32`.
        assert_eq!(Pow::pow(1i32, i64::MAX, &mut status), 1);
        assert_eq!(Pow::pow(-1i32, (1i64 << 40) + 1, &mut status), -1);
        assert_eq!(Pow::pow(2u64, 1i64 << 33, &mut status), 0);
        let mut squared = 3u32;
        for _ in 0..33 {
            squared = squared.wrapping_mul(squared);
        }
        assert_eq!(Pow::pow(3u32, 1i64 << 33, &mut status), squared);
        assert!(status.is_ok());

        // A negative exponent: `0`, and an invalid operation.
        assert_eq!(Pow::pow(2i32, -1i32, &mut status), 0);
        assert!(status.contains(ArithmeticErrorKind::InvalidOperation));
        let mut status = ArithmeticOpStatus::default();
        assert_eq!(Pow::pow(1u8, i64::MIN, &mut status), 0);
        assert!(status.contains(ArithmeticErrorKind::InvalidOperation));
    }

    #[test]
    fn int_pow_negative_exponent_read() {
        use crate::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};
        use crate::Array;

        let a = Array::compact_ndarray(&ndarray::array![2i32, 3, 4, -5]).unwrap();
        let b = Array::compact_ndarray(&ndarray::array![3i32, -1, 0, 3]).unwrap();
        let q = a.pow(b);
        let err = q.to_ndarray::<Ix1>().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Arithmetic);
        assert_eq!(err.message(), "invalid operation encountered");
        let mut ctx = q.read_ctx();
        ctx.set_arithmetic_error_mode(
            ArithmeticErrorKind::InvalidOperation,
            ArithmeticErrorMode::Ignore,
        );
        assert_eq!(
            q.to_ndarray_sub(&[0..4], &ctx).unwrap(),
            ndarray::array![8, 0, 1, -125]
        );
    }

    #[test]
    fn float_pow_errors() {
        use crate::error::ArithmeticErrorKind::*;
        use crate::error::ArithmeticOpStatus;
        use crate::scalar::Pow;

        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            -2.5,
            3.0,
            -8.0,
            1.0 / 3.0,
            300.0,
            -300.0,
            1e30,
            -1e-30,
            f64::MAX,
            f64::INFINITY,
            -f64::INFINITY,
            f64::NAN,
        ];
        // `r` is the power of `x` to `y`.
        let check = |what: &str, x: f64, y: f64, r: f64, status: ArithmeticOpStatus| {
            let finite = x.is_finite() && y.is_finite();
            let expected = [
                (InvalidOperation, finite && x < 0.0 && y.fract() != 0.0),
                (DivideByZero, x == 0.0 && y < 0.0),
                (Overflow, finite && x != 0.0 && r.is_infinite()),
                (Underflow, false),
            ];
            for (kind, expected) in expected {
                assert_eq!(
                    status.contains(kind),
                    expected,
                    "{what}({x}, {y}): {kind:?}"
                );
            }
        };
        for &x in &values {
            for &y in &values {
                let mut status = ArithmeticOpStatus::default();
                let r = Pow::pow(x, y, &mut status);
                check("f64", x, y, r, status);
                let (x, y) = (x as f32, y as f32);
                let mut status = ArithmeticOpStatus::default();
                let r = Pow::pow(x, y, &mut status);
                check("f32", x.into(), y.into(), r.into(), status);
                if y.is_finite() && y.fract() == 0.0 {
                    let mut status = ArithmeticOpStatus::default();
                    let r = Pow::pow(x, y as i32, &mut status);
                    check("f32 x i32", x.into(), y.into(), r.into(), status);
                    let mut status = ArithmeticOpStatus::default();
                    let r = Pow::pow(f64::from(x), y as i32, &mut status);
                    check("f64 x i32", x.into(), y.into(), r, status);
                }
            }
        }
    }

    // Integer powers against repeated wrapping multiplication.
    proptest::proptest! {
        #[test]
        fn int_pow_random(a in proptest::num::i32::ANY, b in -3i64..100) {
            use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
            use crate::scalar::Pow;

            let mut status = ArithmeticOpStatus::default();
            let res = Pow::pow(a, b, &mut status);
            if b < 0 {
                proptest::prop_assert_eq!(res, 0);
                proptest::prop_assert!(status.contains(ArithmeticErrorKind::InvalidOperation));
            } else {
                proptest::prop_assert_eq!(res, (0..b).fold(1i32, |acc, _| acc.wrapping_mul(a)));
                proptest::prop_assert!(status.is_ok());
                let mut status = ArithmeticOpStatus::default();
                proptest::prop_assert_eq!(res, Pow::pow(a, b as i32, &mut status));
                proptest::prop_assert_eq!(res, Pow::pow(a, b as u32, &mut status));
            }
        }
    }
}
