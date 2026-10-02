/// Define the scalar trait `$Trait` of an element-wise unary op: `$f`, and `$f_bulk` on `N`
/// elements, which the impls with SIMD support override (see `impl_scalar_op1`).
macro_rules! define_scalar_op1_trait {
    ($(#[$meta:meta])* $Trait:ident, $f:ident, $f_bulk:ident) => {
        $(#[$meta])*
        pub trait $Trait {
            /// The output element type.
            type Output;
            #[doc = concat!("Applies `", stringify!($f), "` to `self`.")]
            #[allow(clippy::wrong_self_convention)] // By value, as `f32::is_nan`.
            fn $f(self) -> Self::Output;
            #[doc = concat!(
                "[`", stringify!($f), "`](Self::", stringify!($f), ") on each of `xs`, ",
                "vectorized with `simd` where the type has SIMD support."
            )]
            #[inline(always)]
            #[allow(clippy::redundant_closure)]
            fn $f_bulk<S: fearless_simd::Simd, const N: usize>(
                xs: [Self; N],
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized,
            {
                let _ = simd;
                // A closure, not the fn item: its `FnMut` shim is not always inlined.
                crate::util::ArrayExt::map_inline(xs, #[inline(always)] |x| Self::$f(x))
            }
        }
    };
}

/// Define the scalar trait `$Trait` of an element-wise binary op: `$f`, and `$f_bulk` on `N`
/// pairs, which the impls with SIMD support override (see `impl_scalar_op2`).
macro_rules! define_scalar_op2_trait {
    ($(#[$meta:meta])* $Trait:ident, $f:ident, $f_bulk:ident) => {
        crate::scalar::traits_util::define_scalar_op2_trait!(
            $(#[$meta])* $Trait<Rhs = Self>, $f, $f_bulk
        );
    };
    ($(#[$meta:meta])* $Trait:ident<Rhs = $Rhs:ty>, $f:ident, $f_bulk:ident) => {
        $(#[$meta])*
        pub trait $Trait<Rhs = $Rhs> {
            /// The output element type.
            type Output;
            #[doc = concat!("Applies `", stringify!($f), "` to `self` and `rhs`.")]
            fn $f(self, rhs: Rhs) -> Self::Output;
            #[doc = concat!(
                "[`", stringify!($f), "`](Self::", stringify!($f), ") on each pair of `xs` and ",
                "`ys`, vectorized with `simd` where the types have SIMD support."
            )]
            #[inline(always)]
            fn $f_bulk<S: fearless_simd::Simd, const N: usize>(
                xs: [Self; N],
                ys: [Rhs; N],
                simd: S,
            ) -> [Self::Output; N]
            where
                Self: Sized + Copy,
                Rhs: Copy,
            {
                let _ = simd;
                crate::util::array_from_fn_inline(#[inline(always)] |i| Self::$f(xs[i], ys[i]))
            }
        }
    };
}

/// Implement the unary scalar trait `$Trait` for each `$t`, with output `Self` (or `$Out`), as
/// `$scalar` of `$x`. With `simd`, `$f_bulk` too: `$vector` of the vector `$v`
/// (`<$t as SimdLane>::V<S>`), giving a vector of `Self::Output`.
macro_rules! impl_scalar_op1 {
    (
        $Trait:ident::$f:ident, |$x:ident| $scalar:expr,
        [$($t:ty $(=> $Out:ty)?),* $(,)?]
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op1!(
            @one $Trait::$f, |$x| $scalar, $t, ($($Out)?)
        );)*
    };
    (
        $Trait:ident::$f:ident / $f_bulk:ident, |$x:ident| $scalar:expr,
        simd: |$v:ident| $vector:expr,
        [$($t:ty $(=> $Out:ty)?),* $(,)?]
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op1!(
            @one $Trait::$f, |$x| $scalar, $t, ($($Out)?),
            $f_bulk, |$v| $vector
        );)*
    };
    (
        @one $Trait:ident::$f:ident, |$x:ident| $scalar:expr, $t:ty, ($($Out:ty)?)
        $(, $f_bulk:ident, |$v:ident| $vector:expr)?
    ) => {
        impl $Trait for $t {
            type Output = crate::scalar::traits_util::impl_scalar_op1!(@out $($Out)?);
            #[inline(always)]
            fn $f(self) -> Self::Output {
                let $x = self;
                $scalar
            }
            $(
                #[inline(always)]
                fn $f_bulk<S: fearless_simd::Simd, const N: usize>(
                    xs: [Self; N],
                    simd: S,
                ) -> [Self::Output; N] {
                    #[allow(unused_imports)]
                    use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
                    crate::scalar::simd::map_vectors::<S, Self, Self::Output, 1, 1, N>(
                        simd,
                        xs,
                        #[inline(always)] |x| <Self as $Trait>::$f(x),
                        #[inline(always)] |[$v]| [$vector],
                    )
                }
            )?
        }
    };
    (@out) => { Self };
    (@out $Out:ty) => { $Out };
}

/// Implement the binary scalar trait `$Trait` for each `$t` with itself (or each pair `($t,
/// $rhs)`), with output `$Out` (default `Self`), as `$scalar` of `$a` and `$b`. With `simd`,
/// `$f_bulk` too: `$vector` of the vectors `$va` and `$vb`, giving a vector of `$Out`.
macro_rules! impl_scalar_op2 {
    (
        $Trait:ident::$f:ident, |$a:ident, $b:ident| $scalar:expr,
        [$($t:ty),* $(,)?] => $Out:ty
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $t, $Out
        );)*
    };
    (
        $Trait:ident::$f:ident / $f_bulk:ident, |$a:ident, $b:ident| $scalar:expr,
        simd: |$va:ident, $vb:ident| $vector:expr,
        [$($t:ty),* $(,)?] => $Out:ty
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $t, $Out, $f_bulk, |$va, $vb| $vector
        );)*
    };
    (
        $Trait:ident::$f:ident, |$a:ident, $b:ident| $scalar:expr,
        pairs [$(($t:ty, $rhs:ty)),* $(,)?] => $Out:ty
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $rhs, $Out
        );)*
    };
    (
        $Trait:ident::$f:ident / $f_bulk:ident, |$a:ident, $b:ident| $scalar:expr,
        simd: |$va:ident, $vb:ident| $vector:expr,
        pairs [$(($t:ty, $rhs:ty)),* $(,)?] => $Out:ty
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $rhs, $Out, $f_bulk, |$va, $vb| $vector
        );)*
    };
    (
        @one $Trait:ident::$f:ident, |$a:ident, $b:ident| $scalar:expr, $t:ty, $rhs:ty, $Out:ty
        $(, $f_bulk:ident, |$va:ident, $vb:ident| $vector:expr)?
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = $Out;
            #[inline(always)]
            fn $f(self, rhs: $rhs) -> Self::Output {
                let $a = self;
                let $b = rhs;
                $scalar
            }
            $(
                #[inline(always)]
                fn $f_bulk<S: fearless_simd::Simd, const N: usize>(
                    xs: [Self; N],
                    ys: [$rhs; N],
                    simd: S,
                ) -> [Self::Output; N] {
                    #[allow(unused_imports)]
                    use fearless_simd::{Bytes, Select, SimdBase, SimdFloat, SimdInt, SimdMask};
                    crate::scalar::simd::map_vectors2::<S, Self, $rhs, Self::Output, N>(
                        simd,
                        xs,
                        ys,
                        #[inline(always)] |a, b| <Self as $Trait<$rhs>>::$f(a, b),
                        #[inline(always)] |$va, $vb| $vector,
                    )
                }
            )?
        }
    };
    // Without `=> $Out`: the output is `Self`.
    ($($args:tt)*) => {
        crate::scalar::traits_util::impl_scalar_op2!($($args)* => Self);
    };
}

pub(crate) use {
    define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
};
