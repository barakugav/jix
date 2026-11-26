use std::marker::PhantomData;

use crate::ops::elementwise::op1::Op1Kernel;
use crate::ops::elementwise::op2::Op2Kernel;
use crate::ops::prelude::*;

/// A marker type for the right-hand side in a binary operation.
///
/// Used by scalar binary operations, like [`DivScalar<S, T, Rhs>`](crate::ops::DivScalar).
#[non_exhaustive]
pub struct Rhs;

/// A marker type for the left-hand side in a binary operation.
///
/// Used by scalar binary operations, like [`DivScalar<S, T, Lhs>`](crate::ops::DivScalar).
#[non_exhaustive]
pub struct Lhs;

pub(crate) struct ScalarKernel<K, T, Side> {
    kernel: K,
    scalar: T,
    side: PhantomData<Side>,
}
impl<K, T, Side> ScalarKernel<K, T, Side> {
    pub(crate) fn new(kernel: K, scalar: T) -> Self {
        Self {
            kernel,
            scalar,
            side: PhantomData,
        }
    }
}
impl<T1, T2, K> Op1Kernel<T1> for ScalarKernel<K, T2, Rhs>
where
    T2: Copy,
    K: Op2Kernel<T1, T2>,
{
    type Output = K::Output;

    const INNER_LOOPS: InnerLoops = K::INNER_LOOPS;

    #[inline(always)]
    fn apply(&self, x: T1, status: &mut ArithmeticStatus) -> Self::Output {
        self.kernel.apply(x, self.scalar, status)
    }

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        xs: [T1; N],
        status: &mut ArithmeticStatus,
        simd: S,
    ) -> [Self::Output; N] {
        self.kernel.apply_bulk(xs, [self.scalar; N], status, simd)
    }
}
impl<T1, T2, K> Op1Kernel<T1> for ScalarKernel<K, T2, Lhs>
where
    T2: Copy,
    K: Op2Kernel<T2, T1>,
{
    type Output = K::Output;

    const INNER_LOOPS: InnerLoops = K::INNER_LOOPS;

    #[inline(always)]
    fn apply(&self, x: T1, status: &mut ArithmeticStatus) -> Self::Output {
        self.kernel.apply(self.scalar, x, status)
    }

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        xs: [T1; N],
        status: &mut ArithmeticStatus,
        simd: S,
    ) -> [Self::Output; N] {
        self.kernel.apply_bulk([self.scalar; N], xs, status, simd)
    }
}

macro_rules! define_op2_scalar {
    (@lhs true, $ScalarOp:ident, $Kernel:ident, $Trait:ident, $CoreTrait:ident, $core_fn:ident) => {
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(
            @lhs_core_ops Rhs, $ScalarOp, $Trait, $CoreTrait, $core_fn
        );
    };
    (@lhs false, $ScalarOp:ident, $Kernel:ident, $Trait:ident, $CoreTrait:ident, $core_fn:ident) => {
        impl<S, T> $ScalarOp<S, T, crate::ops::Lhs>
        where
            S: crate::Storage,
            T: crate::scalar::$Trait<S::Item, Output: crate::dtype::Dtyped> + Copy + 'static,
        {
            #[doc = concat!("Constructs a [`", stringify!($ScalarOp), "`] storage of `scalar op array`. See the struct docs for semantics and examples.")]
            pub fn new(scalar: T, array: S) -> crate::error::Result<Self> {
                let kernel = crate::ops::elementwise::op2_scalar::ScalarKernel::new($Kernel, scalar);
                Ok(Self(crate::ops::elementwise::op1::Op1::new(array, kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($ScalarOp), "`] storage of `scalar op array`. See the storage struct docs for semantics and examples.")]
            pub fn new_array(scalar: T, array: crate::Array<S>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(scalar, array.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S, T> crate::storage::StorageBase for $ScalarOp<S, T, crate::ops::Lhs>
        where
            S: crate::Storage,
            T: crate::scalar::$Trait<S::Item, Output: crate::dtype::Dtyped> + Copy + 'static,
        {
            type ElementType = <T as crate::scalar::$Trait<S::Item>>::Output;
            crate::storage::impl_array_storage_forward!('a, T2, <S, T>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($ScalarOp), [&self.0.array])
            }

            crate::ops::impl_element_type_change_default!();
        }
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(
            @lhs_core_ops Lhs, $ScalarOp, $Trait, $CoreTrait, $core_fn
        );
    };
    (@lhs_core_ops $($args:tt)*) => {
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [i8]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [i16]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [i32]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [i64]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [u8]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [u16]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [u32]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [u64]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [f32]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [f64]);
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [crate::scalar::bool8]);
        #[cfg(feature = "half")]
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [crate::scalar::f16]);
        #[cfg(feature = "num-complex")]
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [crate::scalar::Complex<f32>]);
        #[cfg(feature = "num-complex")]
        crate::ops::elementwise::op2_scalar::define_op2_scalar!(@lhs_core_op $($args)*, [crate::scalar::Complex<f64>]);
    };
    (@lhs_core_op Rhs, $ScalarOp:ident, $Trait:ident, $CoreTrait:ident, $core_fn:ident, [$T:ty]) => {
        impl<S> core::ops::$CoreTrait<crate::Array<S>> for $T
        where
            S: crate::Storage,
            S::Item: crate::scalar::$Trait<$T, Output: crate::dtype::Dtyped>,
        {
            type Output = crate::Array<$ScalarOp<S, $T, crate::ops::Rhs>>;
            #[doc = concat!("Applies the [`", stringify!($ScalarOp), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_fn(self, array: crate::Array<S>) -> Self::Output {
                $ScalarOp::<S, $T, crate::ops::Rhs>::new_array(array, self).unwrap()
            }
        }
    };
    (@lhs_core_op Lhs, $ScalarOp:ident, $Trait:ident, $CoreTrait:ident, $core_fn:ident, [$T:ty]) => {
        impl<S> core::ops::$CoreTrait<crate::Array<S>> for $T
        where
            S: crate::Storage,
            $T: crate::scalar::$Trait<S::Item, Output: crate::dtype::Dtyped>,
        {
            type Output = crate::Array<$ScalarOp<S, $T, crate::ops::Lhs>>;
            #[doc = concat!("Applies the [`", stringify!($ScalarOp), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_fn(self, array: crate::Array<S>) -> Self::Output {
                $ScalarOp::<S, $T, crate::ops::Lhs>::new_array(self, array).unwrap()
            }
        }
    };
    (
        $(#[$meta:meta])*
        $ScalarOp:ident,
        $Kernel:ident,
        crate::scalar::$Trait:ident,
        $(core_op: core::ops::$CoreTrait:ident::$core_fn:ident, commutative: $commutative:tt,)?
    ) => {
        $(#[$meta])*
        pub struct $ScalarOp<S, T, Side>(
            crate::ops::elementwise::op1::Op1<
                S,
                crate::ops::elementwise::op2_scalar::ScalarKernel<$Kernel, T, Side>,
            >,
        );
        impl<S, T, Side> crate::storage::ArrayStorageImpl for $ScalarOp<S, T, Side> {}
        impl<S, T> $ScalarOp<S, T, crate::ops::Rhs>
        where
            S: crate::Storage,
            T: Copy + 'static,
            S::Item: crate::scalar::$Trait<T, Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($ScalarOp), "`] storage of `array op scalar`. See the struct docs for semantics and examples.")]
            pub fn new(array: S, scalar: T) -> crate::error::Result<Self> {
                let kernel = crate::ops::elementwise::op2_scalar::ScalarKernel::new($Kernel, scalar);
                Ok(Self(crate::ops::elementwise::op1::Op1::new(array, kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($ScalarOp), "`] storage of `array op scalar`. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>, scalar: T) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage(), scalar).map(crate::Array::from_storage)
            }
        }
        impl<S, T> crate::storage::StorageBase for $ScalarOp<S, T, crate::ops::Rhs>
        where
            S: crate::Storage,
            T: Copy + 'static,
            S::Item: crate::scalar::$Trait<T, Output: crate::dtype::Dtyped>,
        {
            type ElementType = <S::Item as crate::scalar::$Trait<T>>::Output;
            crate::storage::impl_array_storage_forward!('a, T2, <S, T>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($ScalarOp), [&self.0.array])
            }

            crate::ops::impl_element_type_change_default!();
        }
        $(
            impl<S, T> core::ops::$CoreTrait<T> for crate::Array<S>
            where
                S: crate::Storage,
                T: crate::dtype::Dtyped,
                S::Item: crate::scalar::$Trait<T, Output: crate::dtype::Dtyped>,
            {
                type Output = crate::Array<$ScalarOp<S, T, crate::ops::Rhs>>;
                #[doc = concat!("Applies the [`", stringify!($ScalarOp), "`] operation, see the op struct docs for details.")]
                #[track_caller]
                fn $core_fn(self, scalar: T) -> Self::Output {
                    $ScalarOp::<S, T, crate::ops::Rhs>::new_array(self, scalar).unwrap()
                }
            }
            crate::ops::elementwise::op2_scalar::define_op2_scalar!(
                @lhs $commutative, $ScalarOp, $Kernel, $Trait, $CoreTrait, $core_fn
            );
        )?
    };
}

pub(crate) use define_op2_scalar;

#[cfg(test)]
pub(crate) mod tests {
    use proptest::strategy::BoxedStrategy;
    use proptest::test_runner::{Config, TestRunner};

    /// Shared proptest driver for the scalar ops: a random array and a random scalar.
    ///
    /// Generic over the dtype only, with the op passed as a fn pointer, to avoid per-op
    /// monomorphization.
    #[inline(never)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn check_op2_scalar<T>(
        strategy: BoxedStrategy<T>,
        check: fn(&ndarray::ArrayD<T>, crate::util::TestArray<T>, T),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let array =
            crate::util::array_strategy_from_shape(crate::util::shape_strategy(), strategy.clone());
        TestRunner::new(Config::default())
            .run(&(array, strategy), |((nd, za), c)| {
                check(&nd, za, c);
                Ok(())
            })
            .unwrap();
    }

    /// Tests `array op scalar` and `scalar op array` against `$body` per element.
    macro_rules! test_op2_scalar {
        (
            $name:ident, $op:tt, |$a:ident, $b:ident| $body:expr,
            [$($dtype:ident),+ $(,)?], $strategy:ident
        ) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$name _ $dtype:lower>]() {
                        crate::ops::elementwise::op2_scalar::tests::check_op2_scalar::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy(),
                            |nd, za, c| {
                                let expected = nd.mapv(|$a| {
                                    let $b = c;
                                    $body
                                });
                                crate::util::assert_array_matches(&(za.clone() $op c), &expected);
                                let expected = nd.mapv(|x| {
                                    let ($a, $b) = (c, x);
                                    $body
                                });
                                crate::util::assert_array_matches(&(c $op za), &expected);
                            },
                        );
                    }
                )+
            }
        };
    }

    pub(crate) use test_op2_scalar;
}
