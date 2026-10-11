macro_rules! define_scalar_op1_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait {
            /// The output element type.
            type Output;

            $(#[$fn_meta])*
            #[allow(clippy::wrong_self_convention)]
            fn $f(self) -> Self::Output;
        }
    };
}

macro_rules! impl_scalar_op1 {
    // Without `type Output`, default to `Self`
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            fn $f:ident = |$x:ident| $f_expr:expr,
        }
    ) => {
        crate::scalar::traits_util::impl_scalar_op1!(
            impl $Trait for [$($t),*] {
                type Output = Self;
                fn $f = |$x| $f_expr,
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
        @impl_one
        impl $Trait:ident for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$x:ident| $f_expr:expr,
        }
    ) => {
        impl $Trait for $t {
            type Output = $Out;
            #[inline(always)]
            fn $f(self) -> Self::Output {
                let $x = self;
                $f_expr
            }
        }
    };
}

macro_rules! define_scalar_op2_trait {
    (
        $(#[$trait_meta:meta])* $Trait:ident,
        $(#[$fn_meta:meta])* $f:ident,
    ) => {
        crate::scalar::traits_util::define_scalar_op2_trait!(
            $(#[$trait_meta])* $Trait<Rhs = Self>,
            $(#[$fn_meta])* $f,
        );
    };
    (
        $(#[$trait_meta:meta])* $Trait:ident<Rhs = $Rhs:ty>,
        $(#[$fn_meta:meta])* $f:ident,
    ) => {
        $(#[$trait_meta])*
        pub trait $Trait<Rhs = $Rhs> {
            /// The output element type.
            type Output;

            $(#[$fn_meta])*
            fn $f(self, rhs: Rhs) -> Self::Output;
        }
    };
}

macro_rules! impl_scalar_op2 {
    // Without `type Output`, default to `Self`
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] {
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
        }
    ) => {
        crate::scalar::traits_util::impl_scalar_op2!(
            impl $Trait for [$($t),*] {
                type Output = Self;
                fn $f = |$a, $b| $f_expr,
            }
        );
    };
    (
        impl $Trait:ident for [$($t:ty),* $(,)?] x $rhs:tt {
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
        }
    ) => {
        crate::scalar::traits_util::impl_scalar_op2!(
            impl $Trait for [$($t),*] x $rhs {
                type Output = Self;
                fn $f = |$a, $b| $f_expr,
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
        @impl_one
        impl $Trait:ident<$rhs:ty> for $t:ty {
            type Output = $Out:ty;
            fn $f:ident = |$a:ident, $b:ident| $f_expr:expr,
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
        }
    };
}

pub(crate) use {
    define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
};
