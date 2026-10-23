macro_rules! define_scalar_op1_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident $(($status:ident))?,
        bulk_fn: $f_bulk:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait {
            /// The output element type.
            type Output;

            $(#[$fn_meta])*
            #[allow(clippy::wrong_self_convention)]
            fn $f(self $(, $status: &mut crate::error::ArithmeticOpStatus)?) -> Self::Output;

            #[doc = concat!(
                "Apply [`", stringify!($f), "`](Self::", stringify!($f), ") on each of `xs`, ",
                "vectorized with `simd` where the type has SIMD support."
            )]
            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                $($status: &mut crate::error::ArithmeticOpStatus,)?
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized,
            {
                let _ = simd;
                crate::util::ArrayExt::map_inline(
                    xs,
                    #[inline(always)] |x| Self::$f(x $(, $status)?),
                )
            }
        }
    };
}

macro_rules! impl_scalar_op1 {
    (impl $Trait:ident for [$($t:ty),* $(,)?] $body:tt) => {
        $(crate::scalar::traits_util::impl_scalar_op1!(@impl_one impl $Trait for $t $body);)*
    };

    // special case for bool output, `$f_expr` is a `bool` and the bulk SIMD output is a mask
    (
        @impl_one
        impl $Trait:ident for $t:ty {
            type Output = Bool8;
            fn $f:ident = |$x:ident| $f_expr:expr,
            fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr,
        }
    ) => {
        impl $Trait for $t {
            type Output = crate::Bool8;
            #[inline(always)]
            fn $f(self) -> crate::Bool8 {
                let $x = self;
                crate::Bool8::from($f_expr)
            }

            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                simd: S,
            ) -> [crate::Bool8; N] {
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
            fn $f:ident = |$x:ident $(, $status:ident)?| $f_expr:expr,
            $(fn $f_bulk:ident = |$($v:ident),+| $f_bulk_expr:expr,)?
        }
    ) => {
        impl $Trait for $t {
            type Output = $Out;
            #[inline(always)]
            fn $f(self $(, $status: &mut crate::error::ArithmeticOpStatus)?) -> Self::Output {
                let $x = self;
                $f_expr
            }

            $(crate::scalar::traits_util::impl_scalar_op1!(
                @bulk $Trait::$f, fn $f_bulk = |$($v),+| $f_bulk_expr
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
    (
        @bulk $Trait:ident::$f:ident,
        fn $f_bulk:ident = |$v:ident, $status:ident| $f_bulk_expr:expr
    ) => {
        #[inline(always)]
        fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
            xs: [Self; N],
            status: &mut crate::error::ArithmeticOpStatus,
            simd: S,
        ) -> [Self::Output; N] {
            #[allow(unused_imports)]
            use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
            crate::scalar::simd::map_vectors_status::<Self, Self::Output, N, S>(
                xs,
                status,
                #[inline(always)] |x, status| <Self as $Trait>::$f(x, status),
                #[inline(always)] |$v, $status| $f_bulk_expr,
                simd,
            )
        }
    };
}

macro_rules! define_scalar_op2_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident $(($status:ident))?,
        bulk_fn: $f_bulk:ident,
    ) => {
        crate::scalar::traits_util::define_scalar_op2_trait!(
            $(#[$trait_meta])* $Trait<Rhs = Self>,
            $(#[$fn_meta])* $f $(($status))?,
            bulk_fn: $f_bulk,
        );
    };
    (
        $(#[$trait_meta:meta])* $Trait:ident<Rhs = $Rhs:ty>,
        $(#[$fn_meta:meta])* $f:ident $(($status:ident))?,
        bulk_fn: $f_bulk:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait<Rhs = $Rhs> {
            /// The output element type.
            type Output;

            $(#[$fn_meta])*
            fn $f(
                self,
                rhs: Rhs
                $(, $status: &mut crate::error::ArithmeticOpStatus)?
            ) -> Self::Output;

            #[doc = concat!(
                "Apply [`", stringify!($f), "`](Self::", stringify!($f), ") on each pair of `xs` and ",
                "`ys`, vectorized with `simd` where the types have SIMD support."
            )]
            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                ys: [Rhs; N],
                $($status: &mut crate::error::ArithmeticOpStatus,)?
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized
            {
                let _ = simd;
                crate::util::array2_map_inline(
                    xs,
                    ys,
                    #[inline(always)] |_, x, y| Self::$f(x, y $(, $status)?),
                )
            }
        }
    };
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

    // special case for bool output, `$f_expr` is a `bool` and the bulk SIMD output is a mask
    (
        @impl_one
        impl $Trait:ident<$rhs:ty> for $t:ty {
            type Output = Bool8;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,
        }
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = crate::Bool8;
            #[inline(always)]
            fn $f(self, rhs: $rhs) -> crate::Bool8 {
                let $a = self;
                let $b = rhs;
                crate::Bool8::from($f_expr)
            }

            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                ys: [$rhs; N],
                simd: S,
            ) -> [crate::Bool8; N] {
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

    (
        @impl_one
        impl $Trait:ident<$rhs:ty> for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident $(, $status:ident)?| $f_expr:expr,
            $(fn $f_bulk:ident = |$($v:ident),+| $f_bulk_expr:expr,)?
        }
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = $Out;
            #[inline(always)]
            fn $f(
                self,
                rhs: $rhs
                $(, $status: &mut crate::error::ArithmeticOpStatus)?
            ) -> Self::Output {
                let $a = self;
                let $b = rhs;
                $f_expr
            }

            $(crate::scalar::traits_util::impl_scalar_op2!(
                @bulk $Trait<$rhs>::$f, fn $f_bulk = |$($v),+| $f_bulk_expr
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
        @bulk $Trait:ident<$rhs:ty>::$f:ident,
        fn $f_bulk:ident = |$va:ident, $vb:ident, $status:ident| $f_bulk_expr:expr
    ) => {
        #[inline(always)]
        fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
            xs: [Self; N],
            ys: [$rhs; N],
            status: &mut crate::error::ArithmeticOpStatus,
            simd: S,
        ) -> [Self::Output; N] {
            #[allow(unused_imports)]
            use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
            crate::scalar::simd::map_vectors2_status::<Self, $rhs, Self::Output, N, S>(
                xs,
                ys,
                status,
                #[inline(always)] |a, b, status| <Self as $Trait<$rhs>>::$f(a, b, status),
                #[inline(always)] |$va, $vb, $status| $f_bulk_expr,
                simd,
            )
        }
    };
}

pub(crate) use {
    define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
};
