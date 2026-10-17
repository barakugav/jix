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

    /// Apply the kernel to `a` and `b`. An [`ArithmeticErrorKind`] is not a panic or an early return: the
    /// kernel produces a fallback value and sets the error in `status`, see
    /// [`ElementwisePipelineImpl::read_bulk`].
    fn apply(&self, a: T1, b: T2, status: &mut ArithmeticOpStatus) -> Self::Output;

    /// [`apply`](Self::apply) on each pair of `a` and `b`.
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
        crate::scalar::$trait:ident::$kernel_fn:ident,
        simd: $bulk_fn:ident,
        $(with_status: $with_status:tt,)?
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
                define_op2!(@apply $trait::$kernel_fn(a, b), status, $($with_status)?)
            }

            #[inline(always)]
            fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
                &self,
                a: [T1; N],
                b: [T2; N],
                status: &mut ArithmeticOpStatus,
                simd: S,
            ) -> [Self::Output; N] {
                define_op2!(@bulk $trait::$bulk_fn(a, b), status, simd, $($with_status)?)
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

    // Calls of the scalar functions, passing `status` if the scalar trait takes one.
    (@apply $trait:ident::$f:ident($a:ident, $b:ident), $status:ident,) => {{
        let _ = $status;
        <T1 as crate::scalar::$trait<T2>>::$f($a, $b)
    }};
    (@apply $trait:ident::$f:ident($a:ident, $b:ident), $status:ident, true) => {
        <T1 as crate::scalar::$trait<T2>>::$f($a, $b, $status)
    };
    (@bulk $trait:ident::$f:ident($a:ident, $b:ident), $status:ident, $simd:ident,) => {{
        let _ = $status;
        <T1 as crate::scalar::$trait<T2>>::$f($a, $b, $simd)
    }};
    (@bulk $trait:ident::$f:ident($a:ident, $b:ident), $status:ident, $simd:ident, true) => {
        <T1 as crate::scalar::$trait<T2>>::$f($a, $b, $status, $simd)
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
    macro_rules! impl_arith {
        (
            $Trait:ident::$f:ident, $op:tt, [$($scalar:ty),*],
            simd: $f_bulk:ident, [$($simd:ty),*]
        ) => {
            impl_scalar_op2!(
                impl $Trait for [$($simd),*] {
                    fn $f = |a, b| a $op b,
                    fn $f_bulk = |a, b| a $op b,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [$($scalar),*] {
                    fn $f = |a, b| a $op b,
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
        Add::add, +, [],
        simd: add_bulk, [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Sub::sub, -, [],
        simd: sub_bulk, [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Mul::mul, *, [i8], // i8 auto-vectorized scalar is faster
        simd: mul_bulk, [f32, f64, i16, i32, i64, u8, u16, u32, u64]
    );
    define_scalar_op2_trait!(
        /// Division operation, like [`core::ops::Div`], reporting arithmetic errors to an
        /// [`ArithmeticOpStatus`] instead of panicking.
        ///
        /// For **integer** types the result is truncating (rounds towards zero). Dividing by zero
        /// gives `0` and sets [`ArithmeticErrorKind::DivideByZero`], and `MIN / -1` gives `MIN`
        /// and sets [`ArithmeticErrorKind::Overflow`].
        /// For **float** types the result is the IEEE 754 one, and the errors are those of numpy:
        /// a finite non-zero value divided by zero sets [`ArithmeticErrorKind::DivideByZero`],
        /// `0 / 0` and `inf / inf` set [`ArithmeticErrorKind::InvalidOperation`], and an infinite
        /// result of finite operands sets [`ArithmeticErrorKind::Overflow`]. NaN operands
        /// propagate without an error.
        /// For **complex** types this is full complex division; errors are not reported yet.
        Div,
        /// Divide by another value, reporting an arithmetic error to `status`.
        div,
        bulk_fn: div_bulk,
        with_status,
    );

    /// The error path of integer division `lhs / rhs`, for a zero `rhs` or `MIN / -1`.
    ///
    /// Out of line on purpose. Besides keeping the rare path out of the loop, the opaque call
    /// keeps LLVM from if-converting the loop to vector code: integer division has no SIMD
    /// instruction, so it would scalarize each division with lane extracts and inserts and a
    /// branch per lane, which measured up to 2x slower (`i16`) than this scalar loop. The status
    /// is returned rather than set through a pointer, which would take the caller's status out of
    /// a register for the whole loop (measured up to 15% slower on strided reads).
    #[cold]
    #[inline(never)]
    fn int_div_error<T: Default + PartialEq>(lhs: T, rhs: T) -> (T, ArithmeticOpStatus) {
        let mut status = ArithmeticOpStatus::default();
        if rhs == T::default() {
            status.set_if(ArithmeticErrorKind::DivideByZero, true);
            (T::default(), status)
        } else {
            // `MIN / -1`: `MIN`, as numpy.
            status.set_if(ArithmeticErrorKind::Overflow, true);
            (lhs, status)
        }
    }

    // Integers have no SIMD division: the default `div_bulk` maps `div`.
    macro_rules! impl_int_div {
        ($($t:ty),*) => {$(
            impl Div for $t {
                type Output = Self;
                #[inline(always)]
                fn div(self, rhs: Self, status: &mut ArithmeticOpStatus) -> Self {
                    match self.checked_div(rhs) {
                        Some(q) => q,
                        None => {
                            let (q, error) = int_div_error(self, rhs);
                            *status |= error;
                            q
                        }
                    }
                }
            }
        )*};
    }
    impl_int_div!(i8, i16, i32, i64, u8, u16, u32, u64);

    /// The error of a float division `a / b`, if any, as numpy: `x / 0` with a finite non-zero
    /// `x` is a division by zero, `0 / 0` and `inf / inf` are invalid, and an infinite result
    /// (`r_is_inf`, in the operands' type) of finite operands is an overflow. NaN operands
    /// propagate silently, and so do infinite operands otherwise.
    #[inline(always)]
    fn float_div_error(a: f64, b: f64, r_is_inf: bool) -> ArithmeticOpStatus {
        use ArithmeticErrorKind::*;
        let mut status = ArithmeticOpStatus::default();
        if a.is_nan() || b.is_nan() {
            return status;
        }
        let error = if b == 0.0 {
            if a == 0.0 {
                Some(InvalidOperation)
            } else {
                a.is_finite().then_some(DivideByZero)
            }
        } else if a.is_infinite() {
            b.is_infinite().then_some(InvalidOperation)
        } else {
            r_is_inf.then_some(Overflow)
        };
        if let Some(error) = error {
            status.set_if(error, true);
        }
        status
    }

    /// [`float_div_error`] out of line, the cold path of a scalar division.
    ///
    /// The status is returned rather than set through a pointer: a pointer would take the
    /// caller's status out of a register for the whole loop.
    #[cold]
    #[inline(never)]
    fn float_div_error_cold(a: f64, b: f64, r_is_inf: bool) -> ArithmeticOpStatus {
        float_div_error(a, b, r_is_inf)
    }

    /// [`float_div_error`] of each pair of `xs` and `ys`, the cold path of a SIMD vector.
    #[cold]
    #[inline(never)]
    fn float_div_errors<T>(xs: &[T], ys: &[T]) -> ArithmeticOpStatus
    where
        T: Copy + Into<f64> + core::ops::Div<Output = T>,
    {
        let mut status = ArithmeticOpStatus::default();
        for (&a, &b) in xs.iter().zip(ys) {
            let r: f64 = (a / b).into();
            status |= float_div_error(a.into(), b.into(), r.is_infinite());
        }
        status
    }

    macro_rules! impl_float_div {
        ($($t:ty),*) => {$(
            impl Div for $t {
                type Output = Self;
                #[inline(always)]
                fn div(self, rhs: Self, status: &mut ArithmeticOpStatus) -> Self {
                    let r = self / rhs;
                    // Errors give non-finite results, and NaN operands are never one. The
                    // finiteness test first, and `&&`: the common no-error case is a single
                    // test. With `&`, LLVM orders the tests itself and puts the operands' first
                    // (measured up to 19% slower for `f16`).
                    if !r.is_finite() && !self.is_nan() && !rhs.is_nan() {
                        *status |= float_div_error_cold(self.into(), rhs.into(), r.is_infinite());
                    }
                    r
                }

                #[inline(always)]
                fn div_bulk<const N: usize, S: fearless_simd::Simd>(
                    xs: [Self; N],
                    ys: [Self; N],
                    status: &mut ArithmeticOpStatus,
                    simd: S,
                ) -> [Self; N] {
                    use fearless_simd::{SimdBase, SimdFloat, SimdMask};
                    let mut errors = ArithmeticOpStatus::default();
                    let zs = crate::scalar::simd::map_vectors2::<Self, Self, Self, N, S>(
                        xs,
                        ys,
                        #[inline(always)]
                        |a, b| Self::div(a, b, status),
                        #[inline(always)]
                        |a, b| {
                            let r = a / b;
                            // Errors give non-finite results: a single test per vector. A
                            // vector that has one, rare, is checked for lanes with non-NaN
                            // operands, which are classified out of line and from a copy, so
                            // the operands stay in registers. NaN operands are not errors.
                            if (!r.is_finite()).any_true() {
                                let maybe_error = !r.is_finite() & a.simd_eq(a) & b.simd_eq(b);
                                if maybe_error.any_true() {
                                    errors |= float_div_errors(
                                        a.to_array().as_ref(),
                                        b.to_array().as_ref(),
                                    );
                                }
                            }
                            r
                        },
                        simd,
                    );
                    *status |= errors;
                    zs
                }
            }
        )*};
    }
    impl_float_div!(f32, f64);

    #[cfg(feature = "half")]
    impl Div for f16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self, status: &mut ArithmeticOpStatus) -> Self {
            // In f32 without FP16, as half's operators do, but inlined: theirs call a software
            // fallback.
            #[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
            let r = f16::from_f32(self.to_f32() / rhs.to_f32());
            #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
            let r = self / rhs;
            // The tests ordered as in the `f32` `div`.
            if !r.is_finite() && !self.is_nan() && !rhs.is_nan() {
                *status |= float_div_error_cold(self.into(), rhs.into(), r.is_infinite());
            }
            r
        }
    }

    #[cfg(feature = "num-complex")]
    macro_rules! impl_complex_div {
        ($($t:ty),*) => {$(
            impl Div for $t {
                type Output = Self;
                #[inline(always)]
                fn div(self, rhs: Self, _status: &mut ArithmeticOpStatus) -> Self {
                    self / rhs
                }
            }
        )*};
    }
    #[cfg(feature = "num-complex")]
    impl_complex_div!(Complex<f32>, Complex<f64>);
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_complex_div!(Complex<f16>);

    define_scalar_op2_trait!(
        /// Exponentiation operation: as [`i32::wrapping_pow`] for integers, [`f32::powi`] and
        /// [`f32::powf`] for floats.
        ///
        /// Integer results wrap on overflow. A negative exponent of an integer base gives `0`
        /// and sets [`ArithmeticErrorKind::InvalidOperation`].
        /// A negative float base with a non-integer exponent produces `NaN`; float errors are not
        /// reported yet. Semantics follow [`f32::powf`].
        Pow,
        /// Raise to the power of another value, reporting an arithmetic error to `status`.
        pow,
        bulk_fn: pow_bulk,
        with_status,
    );

    macro_rules! impl_int_pow {
        ($($t:ty),*) => {$(
            impl_int_pow!(@impl $t, [u8, u16, u32], |a, b, _status| a.wrapping_pow(b as u32));
            // A negative exponent gives `0`, and an invalid operation. Branch-free: a power has no
            // SIMD instruction to be if-converted into, unlike a division.
            impl_int_pow!(@impl $t, [i8, i16, i32], |a, b, status| {
                let negative = b < 0;
                status.set_if(ArithmeticErrorKind::InvalidOperation, negative);
                let r = a.wrapping_pow(if negative { 0 } else { b as u32 });
                if negative { 0 } else { r }
            });
            impl_int_pow!(@impl $t, [i64], |a, b, status| {
                let negative = b < 0;
                status.set_if(ArithmeticErrorKind::InvalidOperation, negative);
                let r = if let Ok(b) = u32::try_from(b) {
                    a.wrapping_pow(b)
                } else if negative {
                    1
                } else {
                    // `wrapping_pow` by squaring, for an exponent beyond `u32`.
                    let (mut acc, mut base, mut exp) = (1 as $t, a, b as u64);
                    while exp > 0 {
                        if exp & 1 == 1 {
                            acc = acc.wrapping_mul(base);
                        }
                        base = base.wrapping_mul(base);
                        exp >>= 1;
                    }
                    acc
                };
                if negative { 0 } else { r }
            });
        )*};
        (@impl $t:ty, [$($rhs:ty),*], |$a:ident, $b:ident, $status:ident| $f:expr) => {$(
            impl Pow<$rhs> for $t {
                type Output = $t;
                #[inline(always)]
                fn pow(self, rhs: $rhs, status: &mut ArithmeticOpStatus) -> $t {
                    let ($a, $b, $status) = (self, rhs, status);
                    $f
                }
            }
        )*};
    }
    impl_int_pow!(i8, i16, i32, i64, u8, u16, u32, u64);

    macro_rules! impl_float_pow {
        ($t:ty, [$($rhs:ty),*], |$a:ident, $b:ident| $f:expr) => {$(
            impl Pow<$rhs> for $t {
                type Output = $t;
                #[inline(always)]
                fn pow(self, rhs: $rhs, _status: &mut ArithmeticOpStatus) -> $t {
                    let ($a, $b) = (self, rhs);
                    $f
                }
            }
        )*};
    }
    impl_float_pow!(f32, [i8, u8, i16, u16, i32], |a, b| a.powi(b as i32));
    impl_float_pow!(f64, [i8, u8, i16, u16, i32], |a, b| a.powi(b as i32));
    impl_float_pow!(f32, [f32], |a, b| a.powf(b));
    impl_float_pow!(f64, [f32], |a, b| a.powf(b as f64));
    impl_float_pow!(f64, [f64], |a, b| a.powf(b));
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
    crate::scalar::Div::div,
    simd: div_bulk,
    with_status: true,
    core_op: core::ops::Div::div,
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
    crate::scalar::Pow::pow,
    simd: pow_bulk,
    with_status: true,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(pow: Pow, crate::scalar::Pow);
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

    // sub excludes unsigned types: independent random arrays don't guarantee a >= b,
    // and unsigned underflow panics in debug mode. See sub_u32 below.
    test_op2!(
        add,
        |a, b| a + b,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
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

    // sub_u32: a = c + b guarantees a - b == c with no unsigned underflow.
    proptest::proptest! {
        #[test]
        fn sub_u32(
            ((nd_c, _zc), (nd_b, zb)) in crate::util::arrays2_strategy_generic::<u32>(
                crate::util::shape_strategy(),
                <u32 as crate::util::ScalarStrategy>::op_safe_strategy()
            )
        ) {
            let nd_a = &nd_c + &nd_b;
            let za = crate::Array::compact_ndarray(&nd_a).unwrap();
            crate::util::assert_array_matches(&(za - zb), &nd_c);
        }
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
                // Warn, the default for overflow: `MIN`, as numpy.
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

    /// The float division errors, as numpy, for every pair of edge-case operands, against a
    /// reference from the IEEE value of the result.
    #[test]
    fn float_div_errors() {
        use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        use crate::scalar::Div;
        use crate::util::SimdTestValues;

        macro_rules! check {
            ($($t:ty),*) => {$(
                let values = <$t>::edge_cases();
                for &a in &values {
                    for &b in &values {
                        let mut status = ArithmeticOpStatus::default();
                        let r = Div::div(a, b, &mut status);
                        assert!(r.to_bits() == (a / b).to_bits() || r.is_nan() && (a / b).is_nan());
                        let ordered = !a.is_nan() && !b.is_nan();
                        let finite = a.is_finite() && b.is_finite();
                        for (kind, expected) in [
                            (
                                ArithmeticErrorKind::DivideByZero,
                                ordered && b == 0.0 && a.is_finite() && a != 0.0,
                            ),
                            (ArithmeticErrorKind::InvalidOperation, ordered && r.is_nan()),
                            (ArithmeticErrorKind::Overflow, finite && b != 0.0 && r.is_infinite()),
                        ] {
                            assert_eq!(status.contains(kind), expected, "{a} / {b}: {kind:?}");
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
        use crate::error::{ArithmeticErrorKind, ArithmeticOpStatus};
        use crate::scalar::{f16, Div};

        let f = f16::from_f32;
        for (a, b, error) in [
            (f(1.0), f(4.0), None),
            (f(-3.0), f(0.0), Some(ArithmeticErrorKind::DivideByZero)),
            (f(0.0), f(-0.0), Some(ArithmeticErrorKind::InvalidOperation)),
            (
                f16::INFINITY,
                f16::NEG_INFINITY,
                Some(ArithmeticErrorKind::InvalidOperation),
            ),
            // Overflows f16, not the f32 it is computed in.
            (f(60000.0), f(0.5), Some(ArithmeticErrorKind::Overflow)),
            (f16::INFINITY, f(0.0), None),
            (f16::NAN, f(0.0), None),
            (f(1.0), f16::INFINITY, None),
        ] {
            let mut status = ArithmeticOpStatus::default();
            Div::div(a, b, &mut status);
            assert_eq!(status.is_ok(), error.is_none(), "{a} / {b}");
            if let Some(error) = error {
                assert!(status.contains(error), "{a} / {b}");
            }
        }
    }

    /// The status of the SIMD bodies against the scalar functions', on every SIMD level: each
    /// pair of edge-case operands, in one lane of a bulk of ordinary ones.
    #[test]
    fn simd_bodies_status_all_levels() {
        use super::{DivKernel, Op2Kernel};
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
        // Wraps on overflow, as numpy.
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
