use crate::ops::elementwise::op2::{define_array_op2_method, define_op2};
use crate::ops::prelude::*;

pub(crate) mod traits {
    use crate::error::{ArithmeticErrorKind, ArithmeticStatus};
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{define_scalar_op2_trait, impl_scalar_op2};
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    use fearless_simd::ExtractToken;

    define_scalar_op2_trait!(
        /// Division operation, like [`core::ops::Div`], reporting an integer division by zero to
        /// an [`ArithmeticStatus`] instead of panicking.
        ///
        /// For **integer** types the result is truncating (rounds towards zero). Dividing by zero
        /// gives `0` and reports a division by zero to `status`, and `MIN / -1` wraps to `MIN`.
        /// For **float** types the result is the IEEE 754 one (`+-inf` and `NaN` included), and
        /// no error is reported.
        /// For **complex** types this is full complex division.
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
        /// For **integer** types this is Rust's `/`, except that a zero divisor gives `0` and
        /// reports a division by zero to `status`, and `MIN / -1` wraps to `MIN`.
        /// For **float** types the result is the exact quotient rounded toward zero, not the
        /// rounded `a / b` (`1.0 / 0.1` rounds to `10.0`, but `0.1` fits in `1.0` only 9 times).
        /// A zero divisor produces `+-inf` (`NaN` for `0 / 0`) and an infinite dividend `NaN`,
        /// without an error.
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticStatus;
        /// use jix::scalar::DivideTrunc;
        ///
        /// let mut status = ArithmeticStatus::default();
        /// assert_eq!(7i32.divide_trunc(3, &mut status), 2);
        /// assert_eq!((-7i32).divide_trunc(3, &mut status), -2);
        /// assert_eq!(7i32.divide_trunc(-3, &mut status), -2);
        /// assert_eq!((-7i32).divide_trunc(-3, &mut status), 2);
        /// assert_eq!((-7.5f64).divide_trunc(2.0, &mut status), -3.0);
        /// assert_eq!(1.0f64.divide_trunc(0.1, &mut status), 9.0);
        /// assert!(status.into_result().is_ok());
        ///
        /// assert_eq!(7i32.divide_trunc(0, &mut status), 0);
        /// assert!(status.into_result().is_err());
        ///
        /// // Not errors: `MIN / -1` wraps, and floats give `inf` and `NaN`.
        /// let mut status = ArithmeticStatus::default();
        /// assert_eq!(i32::MIN.divide_trunc(-1, &mut status), i32::MIN);
        /// assert!(f64::INFINITY.divide_trunc(2.0, &mut status).is_nan());
        /// assert!(status.into_result().is_ok());
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
        /// `1.0 / 0.1` gives `9.0`), with the special values of [`DivideTrunc`].
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticStatus;
        /// use jix::scalar::DivideFloor;
        ///
        /// let mut status = ArithmeticStatus::default();
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
        /// For **float** types it rounds the exact quotient, with the special values of
        /// [`DivideTrunc`] ([`f32::div_euclid`] rounds `a / b` first, and can be one off).
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticStatus;
        /// use jix::scalar::DivideEuclid;
        ///
        /// let mut status = ArithmeticStatus::default();
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
        /// For **integer** types this is Rust's `%`, except that a zero divisor gives `0` and
        /// reports a division by zero to `status`, and `MIN % -1` gives `0`.
        /// For **float** types it is Rust's `%`: a zero divisor or an infinite dividend gives
        /// `NaN`, without an error.
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticStatus;
        /// use jix::scalar::RemainderTrunc;
        ///
        /// let mut status = ArithmeticStatus::default();
        /// assert_eq!(7i32.remainder_trunc(3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_trunc(3, &mut status), -1);
        /// assert_eq!(7i32.remainder_trunc(-3, &mut status), 1);
        /// assert_eq!((-7i32).remainder_trunc(-3, &mut status), -1);
        /// assert_eq!((-7.5f64).remainder_trunc(2.0, &mut status), -1.5);
        /// assert_eq!(i32::MIN.remainder_trunc(-1, &mut status), 0);
        /// assert!(status.into_result().is_ok());
        ///
        /// assert_eq!(7i32.remainder_trunc(0, &mut status), 0);
        /// assert!(status.into_result().is_err());
        ///
        /// // Not an error for floats: `NaN`.
        /// let mut status = ArithmeticStatus::default();
        /// assert!(7.0f64.remainder_trunc(0.0, &mut status).is_nan());
        /// assert!(status.into_result().is_ok());
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
        /// with the special values of [`RemainderTrunc`].
        ///
        /// # Examples
        /// ```
        /// use jix::error::ArithmeticStatus;
        /// use jix::scalar::RemainderFloor;
        ///
        /// let mut status = ArithmeticStatus::default();
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
        /// use jix::error::ArithmeticStatus;
        /// use jix::scalar::RemainderEuclid;
        ///
        /// let mut status = ArithmeticStatus::default();
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
        status: &mut ArithmeticStatus,
    ) -> T {
        match q {
            Some(q) => q,
            None => {
                let zero = b == T::default();
                let mut error = ArithmeticStatus::default();
                error.set_if(ArithmeticErrorKind::DivideByZero, zero);
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
        status: &mut ArithmeticStatus,
    ) -> T {
        match r {
            Some(r) => r,
            None => {
                let mut error = ArithmeticStatus::default();
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
    fn int_error_barrier<T>(error: ArithmeticStatus) -> ArithmeticStatus {
        if size_of::<T>() < 4 {
            core::hint::black_box(error)
        } else {
            error
        }
    }

    #[cfg(feature = "half")]
    impl_scalar_op2!(
        impl Div for [f16] {
            fn div = |a, b, _status| {

                #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
                let r = a / b;

                #[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
                let r = f16::from_f32(a.to_f32() / b.to_f32());

                r
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
                use fearless_simd::{Select, Simd, SimdBase, SimdFloat, SimdMask};

                type T = $t;
                type V<S> = <S as Simd>::$v;
                const EXACT_LIMIT: T = $exact_limit;

                /// The exact `trunc(a / b)` and its residual `a - q * b`, which below `EXACT_LIMIT`
                /// is `a % b` (`fmod`), exactly.
                ///
                /// The quotient is `div_trunc` of rust-lang/rust#134062, a proposed fix to
                /// `f64::div_euclid`:
                /// <https://github.com/rust-lang/rust/blob/343e091465e81589a164adba621f331201746dc6/library/std/src/f64.rs#L280-L293>
                /// Unlike there, it compares the residual with `< 0.0`, not `is_sign_negative()`,
                /// which the NaN residuals of special values may pass, and goes on past
                /// `EXACT_LIMIT`, where #134062 returns NaN, within one ulp. As in numpy's
                /// `floor_divide`, an infinite dividend gives NaN.
                ///
                /// The residual of the exact quotient is the remainder, as in the docs there
                /// (L222-L223), exact as `fmod` is:
                /// <https://en.cppreference.com/w/c/numeric/math/fmod>
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
                fn splat<S: Simd>(x: T, simd: S) -> V<S> {
                    <V<S> as SimdBase<S>>::splat(simd, x)
                }
            }

            impl_scalar_op2!(
                impl Div for [$t] {
                    fn div = |a, b, _status| a / b,
                    fn div_bulk = |a, b, _status| a / b,
                }
            );
            impl_scalar_op2!(
                impl DivideTrunc for [$t] {
                    fn divide_trunc = |a, b, _status| $m::div_trunc(a, b),
                    fn divide_trunc_bulk = |a, b, _status| $m::div_trunc_v(a, b, a.token()),
                }
            );
            impl_scalar_op2!(
                impl DivideFloor for [$t] {
                    fn divide_floor = |a, b, _status| $m::div_floor(a, b),
                    fn divide_floor_bulk = |a, b, _status| $m::div_floor_v(a, b, a.token()),
                }
            );
            impl_scalar_op2!(
                impl DivideEuclid for [$t] {
                    fn divide_euclid = |a, b, _status| $m::div_euclid(a, b),
                    fn divide_euclid_bulk = |a, b, _status| $m::div_euclid_v(a, b, a.token()),
                }
            );
            impl_scalar_op2!(
                impl RemainderTrunc for [$t] {
                    fn remainder_trunc = |a, b, _status| $m::rem_trunc(a, b),
                    fn remainder_trunc_bulk = |a, b, _status| $m::rem_trunc_v(a, b, a.token()),
                }
            );
            impl_scalar_op2!(
                impl RemainderFloor for [$t] {
                    fn remainder_floor = |a, b, _status| $m::rem_floor(a, b),
                    fn remainder_floor_bulk = |a, b, _status| $m::rem_floor_v(a, b, a.token()),
                }
            );
            impl_scalar_op2!(
                impl RemainderEuclid for [$t] {
                    fn remainder_euclid = |a, b, _status| $m::rem_euclid(a, b),
                    fn remainder_euclid_bulk = |a, b, _status| $m::rem_euclid_v(a, b, a.token()),
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
        ($($Trait:ident::$f:ident = $g:ident),*) => {$(
            #[cfg(feature = "half")]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b, _status| f16::from_f32(f32_div_rem::$g(a.to_f32(), b.to_f32())),
                }
            );
        )*};
    }
    impl_f16_div_rem!(
        DivideTrunc::divide_trunc = div_trunc,
        DivideFloor::divide_floor = div_floor,
        DivideEuclid::divide_euclid = div_euclid,
        RemainderTrunc::remainder_trunc = rem_trunc,
        RemainderFloor::remainder_floor = rem_floor,
        RemainderEuclid::remainder_euclid = rem_euclid
    );
}

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

impl<S> Array<S>
where
    S: ArrayStorage,
{
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
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use ndarray::{Ix1, Ix2, IxDyn};

    use crate::ops::elementwise::op2::tests::test_op2;

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::DivKernel;
        use crate::ops::elementwise::op2::Op2Kernel;
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
            simd.vectorize(|| {
                case!(DivKernel, f32, |x, y| x / y);
                case!(DivKernel, f64, |x, y| x / y);
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
            DivideEuclidKernel, DivideFloorKernel, DivideTruncKernel, RemainderEuclidKernel,
            RemainderFloorKernel, RemainderTruncKernel,
        };
        use crate::ops::elementwise::op2::Op2Kernel;
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
        let s = &mut crate::error::ArithmeticStatus::default(); // floats report no errors

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
        let s = &mut crate::error::ArithmeticStatus::default(); // floats report no errors

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
        let s = &mut crate::error::ArithmeticStatus::default(); // floats report no errors

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
        // `1.0 % 0.0` is `NaN`, not an error.
        fn read<S: crate::storage::ArrayStorageTyped>(r: Array<S>) -> ndarray::Array1<S::Item> {
            r.to_ndarray().unwrap()
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

    // Integer division by zero fails the read.
    macro_rules! test_div_by_zero {
        ($($t:ident),*) => {$(
            paste::paste! {
                #[test]
                fn [<div_by_zero_ $t>]() {
                    let n = 1000;
                    let nd_a = ndarray::Array1::from_shape_fn(n, |i| (i % 100) as $t);
                    let mut nd_b = ndarray::Array1::from_shape_fn(n, |i| (i % 7 + 1) as $t);
                    nd_b[300] = 0;
                    nd_b[n - 1] = 0;
                    let expected = ndarray::Zip::from(&nd_a)
                        .and(&nd_b)
                        .map_collect(|&x, &y| if y == 0 { 0 } else { x / y });
                    let q = crate::Array::compact_ndarray(&nd_a).unwrap()
                        / crate::Array::compact_ndarray(&nd_b).unwrap();

                    let err = q.to_ndarray::<ndarray::Ix1>().unwrap_err();
                    assert_eq!(err.kind(), crate::error::ErrorKind::Arithmetic);
                    assert_eq!(err.message(), "integer division or modulo by zero");
                    // Each zero divisor alone: in the bulk part of the loop, and in its scalar
                    // tail.
                    let ctx = q.read_ctx();
                    for range in [0..n as u64 - 1, 301..n as u64] {
                        let err = q.to_ndarray_sub::<ndarray::Ix1>(&[range], &ctx).unwrap_err();
                        assert_eq!(err.kind(), crate::error::ErrorKind::Arithmetic);
                    }
                    // Only the elements read count.
                    let mid = q.to_ndarray_sub(&[301..n as u64 - 1], &ctx).unwrap();
                    assert_eq!(mid, expected.slice(ndarray::s![301..n - 1]));
                }
            }
        )*};
    }
    test_div_by_zero!(i8, i16, i32, i64, u8, u16, u32, u64);

    #[test]
    fn div_by_zero_strided() {
        use crate::Array;

        // Transposed operands are read with a non-unit inner stride.
        let nd_a = ndarray::Array2::from_shape_fn((40, 30), |(i, j)| (i * 30 + j) as i32 - 600);
        let divisors = |zero: i32| {
            ndarray::Array2::from_shape_fn((40, 30), |(i, j)| match ((i + j) % 5) as i32 - 2 {
                0 => zero,
                d => d,
            })
        };
        let q = |nd_b: &ndarray::Array2<i32>| {
            let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[8, 8])).unwrap();
            let zb = Array::compact_ndarray_with(nd_b, crate::util::arr_params(&[8, 8])).unwrap();
            za.transpose() / zb.transpose()
        };
        assert!(q(&divisors(0)).to_ndarray::<Ix2>().is_err());

        let nd_b = divisors(3);
        let expected = ndarray::Zip::from(&nd_a.t())
            .and(&nd_b.t())
            .map_collect(|&x, &y| x / y);
        assert_eq!(q(&nd_b).to_ndarray::<Ix2>().unwrap(), expected);
    }

    #[test]
    fn div_min_by_minus_one_wraps() {
        macro_rules! check {
            ($($t:ident),*) => {$(
                let a = crate::Array::compact_ndarray(&ndarray::array![$t::MIN, 7, $t::MAX]);
                let b = crate::Array::compact_ndarray(&ndarray::array![-1 as $t, -1, -1]);
                let (a, b) = (a.unwrap(), b.unwrap());
                let res = (a / b).to_ndarray::<ndarray::Ix1>().unwrap();
                assert_eq!(res, ndarray::array![$t::MIN, -7, -$t::MAX], stringify!($t));
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
            assert_eq!(err.message(), "integer division or modulo by zero");
        }

        // Float division by zero is not an error: `inf`.
        let fa = Array::compact_ndarray(&ndarray::array![1.0f32, 2.0]).unwrap();
        let fb = Array::compact_ndarray(&ndarray::array![0.0f32, 1.0]).unwrap();
        let res = (fa / fb).map(|x: f32| x * 2.0).to_ndarray::<Ix1>().unwrap();
        assert_eq!(res, ndarray::array![f32::INFINITY, 4.0]);
    }

    #[test]
    fn scalar_div_status() {
        use crate::error::{ArithmeticErrorKind, ArithmeticStatus};
        use crate::scalar::Div;

        let mut status = ArithmeticStatus::default();
        assert_eq!(Div::div(7u8, 2, &mut status), 3);
        assert_eq!(Div::div(-7i32, 2, &mut status), -3);
        assert_eq!(Div::div(1.0f32, 4.0, &mut status), 0.25);
        assert_eq!(Div::div(i32::MIN, -1, &mut status), i32::MIN);
        assert!(Div::div(0.0f64, 0.0, &mut status).is_nan());
        assert!(status.is_ok());
        assert_eq!(Div::div(-7i64, 0, &mut status), 0);
        assert!(status.contains(ArithmeticErrorKind::DivideByZero));
    }

    /// The integer quotients and remainders of a zero divisor and of `MIN` by `-1`.
    #[test]
    fn int_division_errors() {
        use crate::error::ArithmeticErrorKind::DivideByZero;
        use crate::error::ArithmeticStatus;
        use crate::scalar::{
            Div, DivideEuclid, DivideFloor, DivideTrunc, RemainderEuclid, RemainderFloor,
            RemainderTrunc,
        };

        type F<T> = fn(T, T, &mut ArithmeticStatus) -> T;
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
            error: ArithmeticStatus,
        ) {
            let mut status = ArithmeticStatus::default();
            assert_eq!(f(a, b, &mut status), expected, "{a:?}, {b:?}");
            assert_eq!(status, error, "{a:?}, {b:?}");
        }

        macro_rules! check {
            ($($t:ty),*) => {$(
                for f in quotients::<$t>() {
                    check(f, (7, 3), 2, ArithmeticStatus::default());
                    check(f, (7, 0), 0, DivideByZero.into());
                    check(f, (<$t>::MIN, 0), 0, DivideByZero.into());
                }
                for f in remainders::<$t>() {
                    check(f, (7, 3), 1, ArithmeticStatus::default());
                    check(f, (7, 0), 0, DivideByZero.into());
                }
            )*};
        }
        check!(i8, i16, i32, i64, u8, u16, u32, u64);
        // `MIN` by `-1`: a quotient of `MIN`, wrapped, and an exact remainder of `0`, no errors.
        macro_rules! check_signed {
            ($($t:ty),*) => {$(
                for f in quotients::<$t>() {
                    check(f, (<$t>::MIN, -1), <$t>::MIN, ArithmeticStatus::default());
                }
                for f in remainders::<$t>() {
                    check(f, (<$t>::MIN, -1), 0, ArithmeticStatus::default());
                }
            )*};
        }
        check_signed!(i8, i16, i32, i64);
    }

    // Random shapes, storage backends and layouts, with zero divisors in about a third of the
    // elements: an error iff some divisor is zero, and otherwise the reference values.
    proptest::proptest! {
        #[test]
        fn div_by_zero_random(
            ((nd_a, za), (nd_b, zb)) in crate::util::arrays2_strategy_generic::<i32>(
                crate::util::shape_strategy(),
                <i32 as crate::util::ScalarStrategy>::comparable_strategy()
            )
        ) {
            use crate::error::ErrorKind;

            let q = za / zb;
            let index = q.shape().iter().map(|&d| 0..d).collect::<Vec<_>>();
            let res = q.to_ndarray_sub::<IxDyn>(&index, &q.read_ctx());
            if nd_b.iter().any(|&y| y == 0) {
                proptest::prop_assert_eq!(res.unwrap_err().kind(), ErrorKind::Arithmetic);
            } else {
                let expected = ndarray::Zip::from(&nd_a).and(&nd_b).map_collect(|&x, &y| x / y);
                proptest::prop_assert_eq!(res.unwrap(), expected);
            }
        }
    }

    macro_rules! test_float_div_special_values {
        ($($t:ident),*) => {$(
            paste::paste! {
                /// Float division by zero and the other special values give the IEEE values, which
                /// are not errors: in the bulk part of the loop, and in its scalar tail.
                #[test]
                fn [<float_div_special_values_read_ $t>]() {
                    #[cfg(feature = "half")]
                    #[allow(unused_imports)]
                    use crate::scalar::f16;

                    let c = |x: f64| <f64 as crate::scalar::Cast<$t>>::cast(x);
                    let n = 1000;
                    let inf = f64::INFINITY;
                    for (x, y) in [
                        (c(5.0), c(0.0)),
                        (c(0.0), c(0.0)),
                        (c(inf), c(inf)),
                        ($t::MAX, c(0.25)),
                        (c(f64::NAN), c(0.0)),
                        (c(inf), c(0.0)),
                        (c(-1.0), c(inf)),
                    ] {
                        for i in [300, n - 1] {
                            let mut nd_a =
                                ndarray::Array1::from_shape_fn(n, |i| c((i % 100) as f64));
                            let mut nd_b =
                                ndarray::Array1::from_shape_fn(n, |i| c((i % 7 + 1) as f64));
                            (nd_a[i], nd_b[i]) = (x, y);
                            let q = crate::Array::compact_ndarray(&nd_a).unwrap()
                                / crate::Array::compact_ndarray(&nd_b).unwrap();
                            let what = format!("{x} / {y} at {i}");
                            let res = q.to_ndarray::<ndarray::Ix1>().expect(&what);
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
    test_float_div_special_values!(f32, f64);
    #[cfg(feature = "half")]
    test_float_div_special_values!(f16);
}
