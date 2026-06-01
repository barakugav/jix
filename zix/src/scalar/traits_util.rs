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
