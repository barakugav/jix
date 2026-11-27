macro_rules! define_scalar_op1_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident,
        bulk_fn: $f_bulk:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait {
            /// The output element type.
            type Output;

            $(#[$fn_meta])*
            #[allow(clippy::wrong_self_convention)]
            fn $f(self) -> Self::Output;

            #[doc = concat!(
                "Apply [`", stringify!($f), "`](Self::", stringify!($f), ") on each of `xs`, ",
                "vectorized with `simd` where the type has SIMD support."
            )]
            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized,
            {
                let _ = simd;
                crate::util::ArrayExt::map_inline(
                    xs,
                    #[inline(always)] |x| Self::$f(x),
                )
            }
        }
    };
}

macro_rules! impl_scalar_op1 {
    (impl $Trait:ident for [$($t:ty),* $(,)?] $body:tt) => {
        $(crate::scalar::traits_util::impl_scalar_op1!(@impl_one impl $Trait for $t $body);)*
    };

    // special case for bool output
    (
        @impl_one
        impl $Trait:ident for $t:ty {
            type Output = bool8;
            fn $f:ident = |$x:ident| $f_expr:expr,
            fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr,
        }
    ) => {
        impl $Trait for $t {
            type Output = crate::scalar::bool8;
            #[inline(always)]
            fn $f(self) -> crate::scalar::bool8 {
                let $x = self;
                crate::scalar::bool8::from($f_expr)
            }

            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                simd: S,
            ) -> [crate::scalar::bool8; N] {
                #[allow(unused_imports)]
                use fearless_simd::{SimdBase, SimdFloat};
                crate::scalar::simd::map_vectors_to_bools::<Self, N, S>(
                    xs,
                    #[inline(always)] |x| <Self as $Trait>::$f(x),
                    #[inline(always)] |$v| $f_bulk_expr,
                    simd,
                )
            }
        }
    };

    // Without `type Output`, default to `Self`
    (@impl_one impl $Trait:ident for $t:ty { fn $($fns:tt)* }) => {
        crate::scalar::traits_util::impl_scalar_op1!(
            @impl_one impl $Trait for $t { type Output = Self; fn $($fns)* }
        );
    };

    (
        @impl_one
        impl $Trait:ident for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$x:ident| $f_expr:expr,
            $(fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr,)?
        }
    ) => {
        impl $Trait for $t {
            type Output = $Out;
            #[inline(always)]
            fn $f(self) -> Self::Output {
                let $x = self;
                $f_expr
            }

            $(crate::scalar::traits_util::impl_scalar_op1!(
                @bulk $Trait::$f, fn $f_bulk = |$v| $f_bulk_expr
            );)?
        }
    };

    (@bulk $Trait:ident::$f:ident, fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr) => {
        #[inline(always)]
        fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
            xs: [Self; N],
            simd: S,
        ) -> [Self::Output; N] {
            #[allow(unused_imports)]
            use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
            crate::scalar::simd::map_vectors::<Self, Self::Output, N, S>(
                xs,
                #[inline(always)] |x| <Self as $Trait>::$f(x),
                #[inline(always)] |$v| $f_bulk_expr,
                simd,
            )
        }
    };
}

macro_rules! define_scalar_op2_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident $(($fallible:ident))?,
        bulk_fn: $f_bulk:ident,
    ) => {
        crate::scalar::traits_util::define_scalar_op2_trait!(
            $(#[$trait_meta])* $Trait<Rhs = Self>,
            $(#[$fn_meta])* $f $(($fallible))?,
            bulk_fn: $f_bulk,
        );
    };
    (
        $(#[$trait_meta:meta])* $Trait:ident<Rhs = $Rhs:ty>,
        $(#[$fn_meta:meta])* $f:ident,
        bulk_fn: $f_bulk:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait<Rhs = $Rhs> {
            /// The output element type.
            type Output;

            $(#[$fn_meta])*
            fn $f(self, rhs: Rhs) -> Self::Output;

            #[doc = concat!(
                "Apply [`", stringify!($f), "`](Self::", stringify!($f), ") on each pair of `xs` and ",
                "`ys`, vectorized with `simd` where the types have SIMD support."
            )]
            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                ys: [Rhs; N],
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized
            {
                let _ = simd;
                crate::util::array2_map_inline(
                    xs,
                    ys,
                    #[inline(always)] |_, x, y| Self::$f(x, y),
                )
            }
        }
    };
    // A fallible op: returns a `Result`, with the errors of an associated `Error` type.
    (
        $(#[$trait_meta:meta])* $Trait:ident<Rhs = $Rhs:ty>,
        $(#[$fn_meta:meta])* $f:ident(fallible),
        bulk_fn: $f_bulk:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait<Rhs = $Rhs> {
            /// The output element type.
            type Output;

            /// The arithmetic errors of the operation.
            ///
            /// Also an accumulator of the errors of many elements: [`Default`] is no error, `|=`
            /// adds the errors of another, and converting it into a [`Result`] gives `Ok` if it
            /// holds none, else an error of kind
            /// [`ErrorKind::Arithmetic`](crate::error::ErrorKind::Arithmetic).
            ///
            #[doc = concat!(
                "The last bound is into the status that jix's element-wise operations accumulate, ",
                "a type internal to jix: an implementation outside of it uses the `Error` of a jix ",
                "implementation, such as `<i32 as ", stringify!($Trait), ">::Error`."
            )]
            type Error: Default
                + core::ops::BitOrAssign
                + Into<core::result::Result<(), crate::Error>>
                + Into<crate::error::ArithmeticStatus>;

            $(#[$fn_meta])*
            fn $f(self, rhs: Rhs) -> core::result::Result<Self::Output, Self::Error>;

            #[doc = concat!(
                "Apply [`", stringify!($f), "`](Self::", stringify!($f), ") on each pair of `xs` and ",
                "`ys`, vectorized with `simd` where the types have SIMD support, adding the errors ",
                "to `status`. The output of an element with an error is unspecified."
            )]
            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                ys: [Rhs; N],
                status: &mut Self::Error,
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized,
                Self::Output: crate::dtype::Dtyped,
            {
                let _ = simd;
                crate::util::array2_map_inline(
                    xs,
                    ys,
                    #[inline(always)] |_, x, y| {
                        crate::scalar::traits_util::unwrap_or_record(Self::$f(x, y), status)
                    },
                )
            }
        }
    };
}

/// The value of a fallible scalar op's `result`, or a zero placeholder if it is an error, which is
/// added to `status`.
///
/// No branch is needed on the error: an element-wise read checks its status once its inner loop is
/// done, and fails on an error, so the placeholder is never seen.
#[inline(always)]
pub(crate) fn unwrap_or_record<T, E>(result: Result<T, E>, status: &mut E) -> T
where
    T: crate::dtype::Dtyped,
    E: Default + core::ops::BitOrAssign,
{
    let (value, error) = match result {
        Ok(value) => (value, E::default()),
        // SAFETY: every bit pattern is a valid `Dtyped` value
        Err(error) => (
            unsafe { core::mem::MaybeUninit::zeroed().assume_init() },
            error,
        ),
    };
    *status |= error;
    value
}

macro_rules! impl_scalar_op2 {
    // Each left-hand type with each of the right-hand types
    (impl $Trait:ident for [$($t:ty),* $(,)?] x $rhs:tt $body:tt) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(@impl_lhs $body impl $Trait<$rhs> for $t);)*
    };
    (@impl_lhs $body:tt impl $Trait:ident<[$($rhs:ty),* $(,)?]> for $t:ty) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(@impl_one impl $Trait<$rhs> for $t $body);)*
    };

    // Each type with itself as the right-hand side
    (impl $Trait:ident for [$($t:ty),* $(,)?] $body:tt) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(@impl_one impl $Trait<$t> for $t $body);)*
    };

    // special case for bool output
    (
        @impl_one
        impl $Trait:ident<$rhs:ty> for $t:ty {
            type Output = bool8;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,
        }
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = crate::scalar::bool8;
            #[inline(always)]
            fn $f(self, rhs: $rhs) -> crate::scalar::bool8 {
                let $a = self;
                let $b = rhs;
                crate::scalar::bool8::from($f_expr)
            }

            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                ys: [$rhs; N],
                simd: S,
            ) -> [crate::scalar::bool8; N] {
                #[allow(unused_imports)]
                use fearless_simd::SimdBase;
                crate::scalar::simd::map_vectors2_to_bools::<Self, N, S>(
                    xs,
                    ys,
                    #[inline(always)] |a, b| <Self as $Trait>::$f(a, b),
                    #[inline(always)] |$va, $vb| $f_bulk_expr,
                    simd,
                )
            }
        }
    };

    // Without `type Output`, default to `Self`
    (@impl_one impl $Trait:ident<$rhs:ty> for $t:ty { fn $($fns:tt)* }) => {
        crate::scalar::traits_util::impl_scalar_op2!(
            @impl_one impl $Trait<$rhs> for $t { type Output = Self; fn $($fns)* }
        );
    };
    (@impl_one impl $Trait:ident<$rhs:ty> for $t:ty { type Error = $($rest:tt)* }) => {
        crate::scalar::traits_util::impl_scalar_op2!(
            @impl_one impl $Trait<$rhs> for $t { type Output = Self; type Error = $($rest)* }
        );
    };

    (
        @impl_one
        impl $Trait:ident<$rhs:ty> for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            $(fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,)?
        }
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = $Out;
            #[inline(always)]
            fn $f(self, rhs: $rhs) -> Self::Output {
                let $a = self;
                let $b = rhs;
                $f_expr
            }

            $(crate::scalar::traits_util::impl_scalar_op2!(
                @bulk $Trait<$rhs>::$f, fn $f_bulk = |$va, $vb| $f_bulk_expr
            );)?
        }
    };

    // A fallible op: `$f_expr` is a `Result`. The SIMD `$f_bulk_expr` reports no errors.
    (
        @impl_one
        impl $Trait:ident<$rhs:ty> for $t:ty {
            type Output = $Out:ty;
            type Error = $Err:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            $(fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,)?
        }
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = $Out;
            type Error = $Err;
            #[inline(always)]
            fn $f(self, rhs: $rhs) -> core::result::Result<Self::Output, Self::Error> {
                let $a = self;
                let $b = rhs;
                $f_expr
            }

            $(crate::scalar::traits_util::impl_scalar_op2!(
                @bulk_fallible $Trait<$rhs>::$f, fn $f_bulk = |$va, $vb| $f_bulk_expr
            );)?
        }
    };

    (
        @bulk $Trait:ident<$rhs:ty>::$f:ident,
        fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr
    ) => {
        #[inline(always)]
        fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
            xs: [Self; N],
            ys: [$rhs; N],
            simd: S,
        ) -> [Self::Output; N] {
            #[allow(unused_imports)]
            use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
            crate::scalar::simd::map_vectors2::<Self, $rhs, Self::Output, N, S>(
                xs,
                ys,
                #[inline(always)] |a, b| <Self as $Trait<$rhs>>::$f(a, b),
                #[inline(always)] |$va, $vb| $f_bulk_expr,
                simd,
            )
        }
    };
    (
        @bulk_fallible $Trait:ident<$rhs:ty>::$f:ident,
        fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr
    ) => {
        #[inline(always)]
        fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
            xs: [Self; N],
            ys: [$rhs; N],
            status: &mut Self::Error,
            simd: S,
        ) -> [Self::Output; N] {
            #[allow(unused_imports)]
            use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
            crate::scalar::simd::map_vectors2_status::<Self, $rhs, Self::Output, N, S>(
                xs,
                ys,
                status,
                #[inline(always)] |a, b, status| {
                    crate::scalar::traits_util::unwrap_or_record(
                        <Self as $Trait<$rhs>>::$f(a, b),
                        status,
                    )
                },
                #[inline(always)] |$va, $vb, _status| $f_bulk_expr,
                simd,
            )
        }
    };
}

/// Define the float scalar trait `$Trait` as `f32`'s `$f`, with an `f16` impl through `f32`.
macro_rules! float_op1 {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident,
        $f_bulk:ident $(, $simd:ident="enabled")?
    ) => {
        crate::scalar::traits_util::define_scalar_op1_trait!(
            $(#[$trait_meta])* $Trait,
            $(#[$fn_meta])* $f,
            bulk_fn: $f_bulk,
        );
        crate::scalar::traits_util::float_op1!(@impl $Trait, $f, $f_bulk $(, $simd="enabled")?);
        #[cfg(feature = "half")]
        crate::scalar::traits_util::impl_scalar_op1!(
            impl $Trait for [crate::scalar::f16] {
                fn $f = |x| crate::scalar::f16::from_f32(x.to_f32().$f()),
            }
        );
    };
    (@impl $Trait:ident, $f:ident, $f_bulk:ident, simd="enabled") => {
        crate::scalar::traits_util::impl_scalar_op1!(
            impl $Trait for [f32, f64] {
                fn $f = |x| x.$f(),
                fn $f_bulk = |x| x.$f(),
            }
        );
    };
    (@impl $Trait:ident, $f:ident, $f_bulk:ident) => {
        crate::scalar::traits_util::impl_scalar_op1!(
            impl $Trait for [f32, f64] {
                fn $f = |x| x.$f(),
            }
        );
    };
}

pub(crate) use {
    define_scalar_op1_trait, define_scalar_op2_trait, float_op1, impl_scalar_op1, impl_scalar_op2,
};
