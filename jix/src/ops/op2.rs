use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, ensure, Result};
use crate::ops::common::define_array_op2_method;
use crate::storage::{
    ArraySpec, ArrayStorageInfo, ArrayStorageTyped, OutBuf, ReadData, ReadDataExt,
};
use crate::util::assert_unchecked_eq;
use crate::{Array, ArrayStorage, Ty};

pub(crate) struct Op2<S1, S2, K> {
    pub(crate) a: S1,
    pub(crate) b: S2,
    kernel: K,
}
pub(crate) trait Op2Kernel<T1, T2> {
    type Output;
    fn apply(&self, a: T1, b: T2) -> Self::Output;

    // TODO: implement ops using explicit SIMD
    // #[inline(always)]
    // fn apply_bulk<const N: usize>(&self, a: [T1; N], b: [T2; N]) -> [Self::Output; N] {
    //     let mut iter = a.into_iter().zip(b);
    //     array_from_fn_inline(|_| {
    //         let (a, b) = iter.next().unwrap();
    //         self.apply(a, b)
    //     })
    // }
}
impl<S1, S2, K> Op2<S1, S2, K> {
    pub(crate) fn new(a: S1, b: S2, kernel: K) -> Result<Self>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped<Dimension = S1::Dimension>,
        K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
    {
        ensure!(
            a.shape() == b.shape(),
            InvalidArgument,
            "Op2 shape mismatch between `a` {:?} and `b` {:?}",
            a.shape(),
            b.shape()
        );
        Ok(Self { a, b, kernel })
    }
}

impl<S1, S2, K> ArrayStorage for Op2<S1, S2, K>
where
    S1: ArrayStorageTyped,
    S2: ArrayStorageTyped<Dimension = S1::Dimension>,
    K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;
    type Dimension = S1::Dimension;

    #[inline]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        self.read_data_typed::<K::Output>(index, context)?
            .to_buf::<Self::Dimension>(buf, index)
    }

    #[inline(always)]
    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, S1, S2, K>>
    where
        T: Dtyped,
    {
        check_dtype(&T::DTYPE, &K::Output::DTYPE)?;
        let a_data = self.a.read_data_typed(index, context)?;
        let b_data = self.b.read_data_typed(index, context)?;
        let data = a_data.zip_items(b_data);
        data.map_items(|(a, b)| self.kernel.apply(a, b))
            .transmute_items::<T>()
    }
    #[inline(always)]
    fn shape(&self) -> &[u64] {
        let shape = self.a.shape();
        unsafe { assert_unchecked_eq!(shape, self.b.shape()) };
        shape
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        const { &K::Output::DTYPE }
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.a.spec().with_cleared_flags()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Op2", [&self.a, &self.b])
    }

    type DimensionChange<NewD: crate::Dimension> =
        Op2<S1::DimensionChange<NewD>, S2::DimensionChange<NewD>, K>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Op2 {
            a: self.a.dimension_change()?,
            b: self.b.dimension_change()?,
            kernel: self.kernel,
        })
    }

    crate::ops::impl_element_type_change_default!();
}

impl<F, T1, T2, O> Op2Kernel<T1, T2> for F
where
    F: Fn(T1, T2) -> O,
{
    type Output = O;
    #[inline(always)]
    fn apply(&self, a: T1, b: T2) -> Self::Output {
        self(a, b)
    }
}

macro_rules! define_op2 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident ($($call_args:tt)*),
        $(core_op = $core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        define_op2!(@kernel_dispatch
            $Kernel,
            $($trait)::+,
            $kernel_fn,
            ($($call_args)*),
            type Output = <T1 as $($trait)::+<T2>>::Output,
        );
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(a: S1, b: S2) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(a: crate::Array<S1>, b: crate::Array<S2>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(a.into_storage(), b.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S1::Item as $($trait)::+<S2::Item>>::Output>;
            type Dimension = S1::Dimension;
            crate::storage::impl_array_storage_forward!(<S1, S2>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.a, &self.0.b])
            }

            type DimensionChange<NewD: crate::Dimension> = $Op<S1::DimensionChange<NewD>, S2::DimensionChange<NewD>>;
            #[inline]
            fn dimension_change<NewD: crate::Dimension>(
                self,
            ) -> crate::error::Result<Self::DimensionChange<NewD>> {
                Ok($Op(self.0.dimension_change()?))
            }

            crate::ops::impl_element_type_change_default!();
        }
        define_op2!(@define_core
            impl $Op
            $(core_op = $core_op_trait::$core_op_fn,)?
        );
    };

    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident ($($call_args:tt)*),
        type Output = $output_type:ty,
    ) => {
        define_op2!(@kernel_dispatch
            $Kernel,
            $($trait)::+,
            $kernel_fn,
            ($($call_args)*),
            type Output = $output_type,
        );
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: $($trait)::+<S2::Item>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(a: S1, b: S2) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(a: crate::Array<S1>, b: crate::Array<S2>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(a.into_storage(), b.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: $($trait)::+<S2::Item>,
        {
            type ElementType = crate::Ty<$output_type>;
            type Dimension = S1::Dimension;
            crate::storage::impl_array_storage_forward!(<S1, S2>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.a, &self.0.b])
            }

            type DimensionChange<NewD: crate::Dimension> = $Op<S1::DimensionChange<NewD>, S2::DimensionChange<NewD>>;
            #[inline]
            fn dimension_change<NewD: crate::Dimension>(
                self,
            ) -> crate::error::Result<Self::DimensionChange<NewD>> {
                Ok($Op(self.0.dimension_change()?))
            }

            crate::ops::impl_element_type_change_default!();
        }
    };

    // @kernel_dispatch: parse the call convention from the args and forward to @kernel.
    // Handles `(a, b)` (value) and `(&a, &b)` (ref-to-value) calling conventions,
    // extracting the ident names so @kernel can use them as both parameter names and
    // call args (ensuring macro hygiene is consistent).
    (
        @kernel_dispatch
        $Kernel:ident,
        $($trait:ident)::+,
        $kernel_fn:ident,
        ($a:ident, $b:ident),
        type Output = $output_type:ty,
    ) => {
        define_op2!(@kernel
            $Kernel, $($trait)::+, $kernel_fn,
            $a, $b, ($a, $b),
            type Output = $output_type,
        );
    };
    (
        @kernel_dispatch
        $Kernel:ident,
        $($trait:ident)::+,
        $kernel_fn:ident,
        (&$a:ident, &$b:ident),
        type Output = $output_type:ty,
    ) => {
        define_op2!(@kernel
            $Kernel, $($trait)::+, $kernel_fn,
            $a, $b, (&$a, &$b),
            type Output = $output_type,
        );
    };

    // @kernel: generates the kernel struct + Op2Kernel impl.
    // $a and $b are the parameter ident names (same hygiene context as $($call_args)*).
    (
        @kernel
        $Kernel:ident, $($trait:ident)::+, $kernel_fn:ident,
        $a:ident, $b:ident, ($($call_args:tt)*),
        type Output = $output_type:ty,
    ) => {
        struct $Kernel;
        impl<T1, T2> crate::ops::op2::Op2Kernel<T1, T2> for $Kernel
        where
            T1: $($trait)::+<T2>,
        {
            type Output = $output_type;
            #[inline(always)]
            fn apply(&self, $a: T1, $b: T2) -> Self::Output {
                <T1 as $($trait)::+<T2>>::$kernel_fn($($call_args)*)
            }
        }
    };

    (
        @define_core
        impl $Op:ident
    ) => {};
    (
        @define_core
        impl $Op:ident
        core_op = $core_op_trait:ident::$core_op_fn:ident,
    ) => {
        impl<S1, S2> core::ops::$core_op_trait<Array<S2>> for Array<S1>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: core::ops::$core_op_trait<S2::Item, Output: crate::dtype::Dtyped>,
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

macro_rules! define_op2_rhs_fixed {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident ($a:ident, $b:ident),
        rhs = $rhs:ty,
        type Output<T1> = $output_type_t:ty,
        type Output<S1> = $output_type_s:ty,
    ) => {
        struct $Kernel;
        impl<T1> crate::ops::op2::Op2Kernel<T1, $rhs> for $Kernel
        where
            T1: $($trait)::+,
        {
            type Output = $output_type_t;
            #[inline(always)]
            fn apply(&self, $a: T1, $b: $rhs) -> Self::Output {
                <T1 as $($trait)::+>::$kernel_fn($a, $b)
            }
        }
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Item = $rhs, Dimension = S1::Dimension>,
            S1::Item: $($trait)::+
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(a: S1, b: S2) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(a: crate::Array<S1>, b: crate::Array<S2>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(a.into_storage(), b.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Item = $rhs, Dimension = S1::Dimension>,
            S1::Item: $($trait)::+
        {
            type ElementType = crate::Ty<$output_type_s>;
            type Dimension = S1::Dimension;
            crate::storage::impl_array_storage_forward!(<S1, S2>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.a, &self.0.b])
            }

            type DimensionChange<NewD: crate::Dimension> = $Op<S1::DimensionChange<NewD>, S2::DimensionChange<NewD>>;
            #[inline]
            fn dimension_change<NewD: crate::Dimension>(
                self,
            ) -> crate::error::Result<Self::DimensionChange<NewD>> {
                Ok($Op(self.0.dimension_change()?))
            }

            crate::ops::impl_element_type_change_default!();
        }
    };
}

pub(crate) use {define_op2, define_op2_rhs_fixed};

define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types each component is added independently:
    /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
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
    /// assert_eq!(result.as_slice().unwrap(), &[11, 22, 33]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Add,
    AddKernel,
    <core::ops::Add>::add(a, b),
    core_op = Add::add,
);
define_op2!(
    /// Element-wise subtraction of two arrays (`a - b`).
    ///
    /// For **integer** types the result wraps on underflow (two's complement).
    /// For **complex** types each component is subtracted independently:
    /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
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
    /// assert_eq!(result.as_slice().unwrap(), &[9, 18, 27]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sub,
    SubKernel,
    <core::ops::Sub>::sub(a, b),
    core_op = Sub::sub,
);
define_op2!(
    /// Element-wise multiplication of two arrays.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types this is full complex multiplication:
    /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
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
    /// assert_eq!(result.as_slice().unwrap(), &[4, 10, 18]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Mul,
    MulKernel,
    <core::ops::Mul>::mul(a, b),
    core_op = Mul::mul,
);

define_op2!(
    /// Element-wise division of two arrays (`a / b`).
    ///
    /// For **integer** types the result is truncating (rounds towards zero); dividing
    /// by zero panics in debug builds and the result is implementation-defined in release.
    /// For **float** types semantics follow `f32::div`.
    /// For **complex** types this is full complex division.
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
    /// assert_eq!(result.as_slice().unwrap(), &[5, 5, 6]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Div,
    DivKernel,
    <core::ops::Div>::div(a, b),
    core_op = Div::div,
);
define_op2!(
    /// Element-wise exponentiation (`a` raised to the power `b`).
    ///
    /// A negative base with a non-integer exponent produces `NaN`.
    /// Semantics follow [`f32::powf`].
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
    /// assert_eq!(result.as_slice().unwrap(), &[8.0, 9.0, 2.0]);
    ///
    /// // Raise each element to a constant exponent.
    /// let a = Array::compact_ndarray(&array![2.0f32, 3.0, 4.0])?;
    /// let result = a.map(|x| x.powf(2.0)).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4.0, 9.0, 16.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Pow,
    PowKernel,
    <num_traits::Pow>::pow(a, b),
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(pow: Pow, num_traits::Pow);
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

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
            crate::Array<crate::storage::Compact<crate::Ty<T>, crate::DimDyn>>,
            crate::Array<crate::storage::Compact<crate::Ty<T>, crate::DimDyn>>,
        ),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(
                &crate::util::carrays2_strategy_generic::<T>(
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
                #[test]
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

    // Pairwise tests: one proptest per (op, dtype) using random shapes and block sizes.
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
    test_op2!(
        sub,
        |a, b| a - b,
        // TODO: unsigned ints
        [i8, i16, i32, i64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_op2!(
        mul,
        |a, b| a * b,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
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
            ((nd_c, _zc), (nd_b, zb)) in crate::util::carrays2_strategy_generic::<u32>(
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
}
