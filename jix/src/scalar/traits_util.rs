macro_rules! define_op1_trait {
    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident| $kernel_expr:expr,
        [$($input_type:ty => $output_type:ty),* $(,)?]
    ) => {
        #[doc = concat!("Scalar kernel trait for the `", stringify!($method_name), "` element-wise unary operation.")]
        pub trait $trait_name {
            #[doc = "The output element type produced by this operation."]
            type Output;
            #[doc = concat!("Apply the `", stringify!($method_name), "` operation to `self`, returning a value of type `Self::Output`.")]
            fn $method_name(self) -> Self::Output;
        }
        $(
            impl $trait_name for $input_type {
                type Output = $output_type;

                #[inline(always)]
                fn $method_name(self) -> Self::Output {
                    let $a = self;
                    $kernel_expr
                }
            }
        )*
    };

    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident| $kernel_expr:expr,
        [$($input_type:ty),* $(,)?] => $output_type:ty
    ) => {
        define_op1_trait!(
            $trait_name,
            $method_name,
            |$a| $kernel_expr,
            [$($input_type => $output_type),*]
        );
    };

    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident| $kernel_expr:expr,
        [$($input_type:ty),* $(,)?] => "same"
    ) => {
        define_op1_trait!(
            $trait_name,
            $method_name,
            |$a| $kernel_expr,
            [$($input_type => $input_type),*]
        );
    };
}

pub(crate) use define_op1_trait;

#[allow(unused)]
macro_rules! define_op2_trait {
    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [$(($input_a_type:tt, $input_b_type:tt) => $output_type:tt),* $(,)?]
    ) => {
        pub trait $trait_name<Rhs = Self> {
            type Output;
            fn $method_name(self, rhs: Rhs) -> Self::Output;
        }
        $(
            impl $trait_name<$input_b_type> for $input_a_type {
                type Output = $output_type;

                #[inline(always)]
                fn $method_name(self, rhs: $input_b_type) -> Self::Output {
                    let $a = self;
                    let $b = rhs;
                    $kernel_expr
                }
            }
        )*
    };

    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [[$($input_type:tt),*] => "same"]
    ) => {
        define_op2_trait!(
            $trait_name,
            $method_name,
            |$a, $b| $kernel_expr,
            [$($input_type => $input_type),*]
        );
    };

    // given [multiple] input types for a single output type, expand to multiple input-output type pairs
    // [i8, u32] => i8 means i8 => i8, u32 => i8
    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [$([$($input_type:ty),*] => $output_type:ty),* $(,)?]
    ) => {
        define_op2_trait!(
            $trait_name,
            $method_name,
            |$a, $b| $kernel_expr,
            [$($($input_type => $output_type),*),*]
        );
    };

    // given a single input type, assume both inputs of the same dtype
    // i8 => i8 means (i8, i8) => i8
    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [$($input_type:tt => $output_type:tt),* $(,)?]
    ) => {
        define_op2_trait!(
            $trait_name,
            $method_name,
            |$a, $b| $kernel_expr,
            [$(($input_type, $input_type) => $output_type),*]
        );
    };


    // pairs_of
    (
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [pairs_of[$($input_type:ty),*] => $output_type:ty]
    ) => {
        #[doc = concat!("Scalar kernel trait for the `", stringify!($method_name), "` element-wise binary operation.")]
        pub trait $trait_name<Rhs = Self> {
            #[doc = "The output element type produced by this operation."]
            type Output;
            #[doc = concat!("Apply the `", stringify!($method_name), "` operation to `self` and `rhs`, returning a value of type `Self::Output`.")]
            fn $method_name(self, rhs: Rhs) -> Self::Output;
        }

        define_op2_trait!(
            @pairs_of_impl2
            $trait_name,
            $method_name,
            |$a, $b| $kernel_expr,
            [[$($input_type),*], [$($input_type),*] => $output_type]
        );
    };
    (
        @pairs_of_impl2
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [[$($lhs_ty:ty),*], $rhs_ty:tt => $output_type:ty]
    ) => {
        $(
            define_op2_trait!(
                @pairs_of_impl
                $trait_name,
                $method_name,
                |$a, $b| $kernel_expr,
                [$lhs_ty, $rhs_ty => $output_type]
            );
        )*
    };
    (
        @pairs_of_impl
        $trait_name:ident,
        $method_name:ident,
        |$a:ident, $b:ident| $kernel_expr:expr,
        [$lhs_ty:ty, [$($rhs_ty:ty),*] => $output_type:ty]
    ) => {
        $(
            impl $trait_name<$rhs_ty> for $lhs_ty {
                type Output = $output_type;

                #[inline(always)]
                fn $method_name(self, rhs: $rhs_ty) -> Self::Output {
                    let $a = self;
                    let $b = rhs;
                    $kernel_expr
                }
            }
        )*
    };
}

#[allow(unused)]
pub(crate) use define_op2_trait;

// macro_rules! impl_for_pairs {
//     (
//         $macro:ident,
//         [$($types:ty),*]
//     ) => {
//         impl_for_pairs!(
//             @doit
//             $macro,
//             [$($types),*]
//         );
//     };
//     (
//         @doit
//         $macro:ident,
//         [$lhs_ty:ty, $($rhs:ty),*]
//     ) => {
//         $macro!($lhs_ty, $lhs_ty);
//         $(
//             $macro!($lhs_ty, $rhs);
//         )*
//         impl_for_pairs!(
//             @doit
//             $macro,
//             [$($rhs),*]
//         );
//     };
//     (
//         @doit
//         $macro:ident,
//         [$lhs_ty:ty]
//     ) => {
//         $macro!($lhs_ty, $lhs_ty);
//     };
// }

/// Define the scalar trait `$Trait` of an element-wise unary op: `$f`, and `$f_bulk` on `N`
/// elements, which the impls with SIMD support override (see `impl_scalar_op1`).
macro_rules! define_scalar_op1_trait {
    ($(#[$meta:meta])* $Trait:ident, $f:ident, $f_bulk:ident) => {
        $(#[$meta])*
        pub trait $Trait {
            /// The output element type.
            type Output;
            #[doc = concat!("Applies `", stringify!($f), "` to `self`.")]
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
                crate::util::ArrayExt::map_inline(xs, |x| Self::$f(x))
            }
        }
    };
}

/// Define the scalar trait `$Trait` of an element-wise binary op: `$f`, and `$f_bulk` on `N`
/// pairs, which the impls with SIMD support override (see `impl_scalar_op2`).
macro_rules! define_scalar_op2_trait {
    ($(#[$meta:meta])* $Trait:ident, $f:ident, $f_bulk:ident) => {
        $(#[$meta])*
        pub trait $Trait<Rhs = Self> {
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
                crate::util::array_from_fn_inline(|i| Self::$f(xs[i], ys[i]))
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
                        |x| <Self as $Trait>::$f(x),
                        |[$v]| [$vector],
                    )
                }
            )?
        }
    };
    (@out) => { Self };
    (@out $Out:ty) => { $Out };
}

/// Implement the binary scalar trait `$Trait` for each `$t` with itself (or each pair `($t,
/// $rhs)`), with output `Self`, as `$scalar` of `$a` and `$b`. With `simd`, `$f_bulk` too:
/// `$vector` of the vectors `$va` and `$vb`, giving a vector of `Self`.
macro_rules! impl_scalar_op2 {
    (
        $Trait:ident::$f:ident, |$a:ident, $b:ident| $scalar:expr,
        [$($t:ty),* $(,)?]
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $t
        );)*
    };
    (
        $Trait:ident::$f:ident / $f_bulk:ident, |$a:ident, $b:ident| $scalar:expr,
        simd: |$va:ident, $vb:ident| $vector:expr,
        [$($t:ty),* $(,)?]
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $t, $f_bulk, |$va, $vb| $vector
        );)*
    };
    (
        $Trait:ident::$f:ident, |$a:ident, $b:ident| $scalar:expr,
        pairs [$(($t:ty, $rhs:ty)),* $(,)?]
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $rhs
        );)*
    };
    (
        $Trait:ident::$f:ident / $f_bulk:ident, |$a:ident, $b:ident| $scalar:expr,
        simd: |$va:ident, $vb:ident| $vector:expr,
        pairs [$(($t:ty, $rhs:ty)),* $(,)?]
    ) => {
        $(crate::scalar::traits_util::impl_scalar_op2!(
            @one $Trait::$f, |$a, $b| $scalar, $t, $rhs, $f_bulk, |$va, $vb| $vector
        );)*
    };
    (
        @one $Trait:ident::$f:ident, |$a:ident, $b:ident| $scalar:expr, $t:ty, $rhs:ty
        $(, $f_bulk:ident, |$va:ident, $vb:ident| $vector:expr)?
    ) => {
        impl $Trait<$rhs> for $t {
            type Output = Self;
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
                    crate::scalar::simd::map_vectors2::<S, Self, $rhs, Self, N>(
                        simd,
                        xs,
                        ys,
                        |a, b| <Self as $Trait<$rhs>>::$f(a, b),
                        |$va, $vb| $vector,
                    )
                }
            )?
        }
    };
}

pub(crate) use {
    define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
};
