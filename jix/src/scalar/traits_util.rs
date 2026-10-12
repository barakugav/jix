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
                crate::util::ArrayExt::map_inline(xs, #[inline(always)] |x| Self::$f(x))
            }
        }
    };
}

macro_rules! impl_scalar_op1 {
    // special case for bool output
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            type Output = bool;
            fn $f:ident = |$x:ident| $f_expr:expr,
            fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr,
        }
    ) => {
        $(impl $Trait for $t {
            type Output = bool;
            #[inline(always)]
            fn $f(self) -> bool {
                let $x = self;
                $f_expr
            }

            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                simd: S,
            ) -> [bool; N] {
                #[allow(unused_imports)]
                use fearless_simd::{SimdBase, SimdFloat};
                crate::scalar::simd::map_vectors_to_bools::<Self, N, S>(
                    xs,
                    #[inline(always)] |x| <Self as $Trait>::$f(x),
                    #[inline(always)] |$v| $f_bulk_expr,
                    simd,
                )
            }
        })*
    };

    // Without `type Output`, default to `Self`
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            fn $f:ident = |$x:ident| $f_expr:expr,
            $(fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr,)?
        }
    ) => {
        crate::scalar::traits_util::impl_scalar_op1!(
            impl $Trait for [$($t),*] {
                type Output = Self;
                fn $f = |$x| $f_expr,
                $(fn $f_bulk = |$v| $f_bulk_expr,)?
            }
        );
    };

    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            type Output = $Out:ty;
            fn $f:ident = |$x:ident| $f_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op1!(
            @impl_one
            impl $Trait for $t {
                type Output = $Out;
                fn $f = |$x| $f_expr,
            }
        );)*
    };
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            type Output = $Out:ty;
            fn $f:ident = |$x:ident| $f_expr:expr,
            fn $f_bulk:ident = |$v:ident| $f_bulk_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op1!(
            @impl_one
            impl $Trait for $t {
                type Output = $Out;
                fn $f = |$x| $f_expr,
                fn $f_bulk = |$v| $f_bulk_expr,
            }
        );)*
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

            $(
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
            )?
        }
    };
}

macro_rules! define_scalar_op2_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident,
        bulk_fn: $f_bulk:ident,
    ) => {
        crate::scalar::traits_util::define_scalar_op2_trait!(
            $(#[$trait_meta])* $Trait<Rhs = Self>,
            $(#[$fn_meta])* $f,
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
                crate::util::array2_map_inline(xs, ys, #[inline(always)] |_, x, y| Self::$f(x, y))
            }
        }
    };
}

macro_rules! impl_scalar_op2 {
    // special case for bool output
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            type Output = bool;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,
        }
    ) => {
        $(impl $Trait for $t {
            type Output = bool;
            #[inline(always)]
            fn $f(self, rhs: Self) -> bool {
                let $a = self;
                let $b = rhs;
                $f_expr
            }

            #[inline(always)]
            fn $f_bulk<const N: usize, S: fearless_simd::Simd>(
                xs: [Self; N],
                ys: [Self; N],
                simd: S,
            ) -> [bool; N] {
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
        })*
    };

    // Without `type Output`, default to `Self`
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            $(fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,)?
        }
    ) => {
        crate::scalar::traits_util::impl_scalar_op2!(
            impl $Trait for [$($t),*] {
                type Output = Self;
                fn $f = |$a, $b| $f_expr,
                $(fn $f_bulk = |$va, $vb| $f_bulk_expr,)?
            }
        );
    };
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] x $rhs:tt {
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            $(fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,)?
        }
    ) => {
        crate::scalar::traits_util::impl_scalar_op2!(
            impl $Trait for [$($t),*] x $rhs {
                type Output = Self;
                fn $f = |$a, $b| $f_expr,
                $(fn $f_bulk = |$va, $vb| $f_bulk_expr,)?
            }
        );
    };

    // Each type with itself as the right-hand side
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @impl_one
            impl $Trait<$t> for $t {
                type Output = $Out;
                fn $f = |$a, $b| $f_expr,
            }
        );)*
    };
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @impl_one
            impl $Trait<$t> for $t {
                type Output = $Out;
                fn $f = |$a, $b| $f_expr,
                fn $f_bulk = |$va, $vb| $f_bulk_expr,
            }
        );)*
    };

    // Each left-hand type with each of the right-hand types
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] x $rhs:tt {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @impl_cross
            impl $Trait<$rhs> for $t {
                type Output = $Out;
                fn $f = |$a, $b| $f_expr,
            }
        );)*
    };
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] x $rhs:tt {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @impl_cross
            impl $Trait<$rhs> for $t {
                type Output = $Out;
                fn $f = |$a, $b| $f_expr,
                fn $f_bulk = |$va, $vb| $f_bulk_expr,
            }
        );)*
    };

    (
        @impl_cross
        impl $Trait:ident<[$($rhs:ty),* $(,)?]> for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @impl_one
            impl $Trait<$rhs> for $t {
                type Output = $Out;
                fn $f = |$a, $b| $f_expr,
            }
        );)*
    };
    (
        @impl_cross
        impl $Trait:ident<[$($rhs:ty),* $(,)?]> for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
            fn $f_bulk:ident = |$va:ident, $vb:ident| $f_bulk_expr:expr,
        }
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @impl_one
            impl $Trait<$rhs> for $t {
                type Output = $Out;
                fn $f = |$a, $b| $f_expr,
                fn $f_bulk = |$va, $vb| $f_bulk_expr,
            }
        );)*
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

            $(
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
            )?
        }
    };
}

pub(crate) use {
    define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
};
