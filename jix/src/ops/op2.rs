use crate::ops::common::define_array_op2_method;
use crate::ops::prelude::*;
use crate::util::assert_unchecked_eq;

pub(crate) struct Op2<S1, S2, K> {
    pub(crate) a: S1,
    pub(crate) b: S2,
    kernel: K,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op2Kernel<T1, T2> {
    type Output;
    fn apply(&self, a: T1, b: T2) -> Self::Output;
}
impl<S1, S2, K> Op2<S1, S2, K> {
    pub(crate) fn new(a: S1, b: S2, kernel: K) -> Result<Self>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped<Dimension = S1::Dimension>,
        K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        ensure!(
            a.shape() == b.shape(),
            InvalidArgument,
            "Op2 shape mismatch between `a` {:?} and `b` {:?}",
            a.shape(),
            b.shape()
        );
        let a_spec = a.spec();
        let b_spec = b.spec();
        let (element_cost, read_shape_scale_weight, read_layout_order) =
            combine_elementwise_hints(&[
                (
                    a_spec.element_cost(),
                    a_spec.read_shape_scale_weight(),
                    a_spec.read_layout_order(),
                ),
                (
                    b_spec.element_cost(),
                    b_spec.read_shape_scale_weight(),
                    b_spec.read_layout_order(),
                ),
            ]);
        let (block_shape, block_shape_fixed_dims) = combine_block_layout(&[
            (a_spec.block_shape(), a_spec.block_shape_fixed_dims()),
            (b_spec.block_shape(), b_spec.block_shape_fixed_dims()),
        ]);
        let spec = ArraySpecDynamic::new(
            block_shape,
            block_shape_fixed_dims,
            element_cost,
            read_shape_scale_weight,
            read_layout_order,
        );
        Ok(Self { a, b, kernel, spec })
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
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(out.as_deref(), self.shape())?;
        let out = self
            .read_as_elementwise_pipeline::<K::Output>(index, context)?
            .to_buf(index, context, out);
        Ok(out)
    }

    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S1, S2, K>>
    where
        T: Dtyped,
    {
        check_dtype(Dtype::new_ref::<T>(), Dtype::new_ref::<K::Output>())?;
        let a = self
            .a
            .read_as_elementwise_pipeline::<S1::Item>(index, context)?;
        let b = self
            .b
            .read_as_elementwise_pipeline::<S2::Item>(index, context)?;

        struct Op2Pipeline<'a, P1, P2, K, T1, T2> {
            a: P1,
            b: P2,
            kernel: &'a K,
            phantom: std::marker::PhantomData<(T1, T2)>,
        }
        impl<T1, T2, T, P1, P2, K> ElementwisePipelineImpl<T> for Op2Pipeline<'_, P1, P2, K, T1, T2>
        where
            P1: ElementwisePipelineImpl<T1>,
            P2: ElementwisePipelineImpl<T2>,
            K: Op2Kernel<T1, T2, Output: Dtyped>,
            T1: Copy,
            T2: Copy,
            T: Dtyped,
        {
            const N_OPERANDS: Option<usize> = n_operands_sum(&[P1::N_OPERANDS, P2::N_OPERANDS]);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.a.operands().chain(self.b.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool>(
                &self,
                offset: usize,
            ) -> [T; N] {
                let a = unsafe { self.a.read_bulk::<N, CONTIGUOUS>(offset) };
                let b = unsafe { self.b.read_bulk::<N, CONTIGUOUS>(offset) };
                array_from_fn_inline(|i| {
                    let x = self.kernel.apply(a[i], b[i]);

                    const { assert!(size_of::<K::Output>() == size_of::<T>()) };
                    // SAFETY: we checked `T` and `K::Output` are the same dtype in the outer func
                    unsafe { std::mem::transmute_copy::<K::Output, T>(&x) }
                })
            }
        }

        Ok(Op2Pipeline {
            a,
            b,
            kernel: &self.kernel,
            phantom: std::marker::PhantomData,
        })
    }
    #[inline(always)]
    fn shape(&self) -> &[u64] {
        let shape = self.a.shape();
        unsafe { assert_unchecked_eq!(shape, self.b.shape()) };
        shape
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<K::Output>()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.a
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
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
            spec: self.spec,
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
        crate::scalar::$trait:ident::$kernel_fn:ident,
        $(core_op: core::ops::$core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T1, T2> crate::ops::op2::Op2Kernel<T1, T2> for $Kernel
        where
            T1: crate::scalar::$trait<T2> + Copy + 'static,
            T2: Copy + 'static,
        {
            type Output = <T1 as crate::scalar::$trait<T2>>::Output;
            #[inline(always)]
            fn apply(&self, a: T1, b: T2) -> Self::Output {
                <T1 as crate::scalar::$trait<T2>>::$kernel_fn(a, b)
            }
        }
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
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
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S1::Item as crate::scalar::$trait<S2::Item>>::Output>;
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
            impl $Op,
            $trait,
            $(core_op: $core_op_trait::$core_op_fn,)?
        );
    };

    (
        @define_core
        impl $Op:ident,
        $trait:ident,
    ) => {};
    (
        @define_core
        impl $Op:ident,
        $trait:ident,
        core_op: $core_op_trait:ident::$core_op_fn:ident,
    ) => {
        impl<S1, S2> core::ops::$core_op_trait<Array<S2>> for Array<S1>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S1::Dimension>,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
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

pub(crate) use define_op2;

pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{define_scalar_op2_trait, impl_scalar_op2};
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    define_scalar_op2_trait!(
        /// Addition operation, like [`core::ops::Add`].
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        /// For **complex** types each component is added independently:
        /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
        Add,
        /// Add another value.
        add,
    );
    define_scalar_op2_trait!(
        /// Subtraction operation, like [`core::ops::Sub`].
        ///
        /// For **integer** types the result wraps on underflow (two's complement).
        /// For **complex** types each component is subtracted independently:
        /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
        Sub,
        /// Subtract another value.
        sub,
    );
    define_scalar_op2_trait!(
        /// Multiplication operation, like [`core::ops::Mul`].
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        /// For **complex** types this is full complex multiplication:
        /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
        Mul,
        /// Multiply with another value.
        mul,
    );
    define_scalar_op2_trait!(
        /// Division operation, like [`core::ops::Div`].
        ///
        /// For **integer** types the result is truncating (rounds towards zero); dividing
        /// by zero panics in both debug and release builds.
        /// For **float** types semantics follow `f32::div`.
        /// For **complex** types this is full complex division.
        Div,
        /// Divide by another value.
        div,
    );

    macro_rules! impl_arith {
        ($Trait:ident::$f:ident, $op:tt, [$($t:ty),*]) => {
            impl_scalar_op2!(
                impl $Trait for [$($t),*] {
                    fn $f = |a, b| a $op b,
                }
            );
            // In f32 without FP16, as half's operators do, but inlined: theirs call a software
            // fallback function.
            #[cfg(all(feature = "half", not(all(target_arch = "aarch64", target_feature = "fp16"))))]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b| f16::from_f32(a.to_f32() $op b.to_f32()),
                }
            );
            #[cfg(all(feature = "half", target_arch = "aarch64", target_feature = "fp16"))]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(feature = "num-complex")]
            impl_scalar_op2!(
                impl $Trait for [Complex<f32>, Complex<f64>] {
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(all(feature = "half", feature = "num-complex"))]
            impl_scalar_op2!(
                impl $Trait for [Complex<f16>] {
                    fn $f = |a, b| a $op b,
                }
            );
        };
    }
    impl_arith!(
        Add::add, +,
        [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Sub::sub, -,
        [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
    );
    // i8: the auto-vectorized scalar kernel is as fast (static analysis).
    impl_arith!(
        Mul::mul, *,
        [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
    );
    // Integers: no SIMD division.
    impl_arith!(
        Div::div, /,
        [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
    );

    define_scalar_op2_trait!(
        /// Exponentiation operation: as [`i32::pow`] for integers (unsigned exponents),
        /// [`f32::powi`] and [`f32::powf`] for floats.
        ///
        /// A negative base with a non-integer exponent produces `NaN`.
        /// Semantics follow [`f32::powf`].
        Pow,
        /// Raise to the power of another value.
        pow,
    );
    impl_scalar_op2!(
        impl Pow for [i8, i16, i32, i64, u8, u16, u32, u64] x [u8, u16, u32] {
            fn pow = |a, b| a.pow(b as u32),
        }
    );
    impl_scalar_op2!(
        impl Pow for [f32, f64] x [i8, u8, i16, u16, i32] {
            fn pow = |a, b| a.powi(b as i32),
        }
    );
    impl_scalar_op2!(
        impl Pow for [f32, f64] {
            fn pow = |a, b| a.powf(b),
        }
    );
    impl_scalar_op2!(
        impl Pow for [f64] x [f32] {
            fn pow = |a, b| a.powf(b as f64),
        }
    );
}

define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// See [`jix::scalar::Add`](crate::scalar::Add) scalar trait for the per-element semantics.
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
    crate::scalar::Add::add,
    core_op: core::ops::Add::add,
);
define_op2!(
    /// Element-wise subtraction of two arrays (`a - b`).
    ///
    /// See [`jix::scalar::Sub`](crate::scalar::Sub) scalar trait for the per-element semantics.
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
    crate::scalar::Sub::sub,
    core_op: core::ops::Sub::sub,
);
define_op2!(
    /// Element-wise multiplication of two arrays.
    ///
    /// See [`jix::scalar::Mul`](crate::scalar::Mul) scalar trait for the per-element semantics.
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
    crate::scalar::Mul::mul,
    core_op: core::ops::Mul::mul,
);

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
    /// assert_eq!(result.as_slice().unwrap(), &[5, 5, 6]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Div,
    DivKernel,
    crate::scalar::Div::div,
    core_op: core::ops::Div::div,
);
define_op2!(
    /// Element-wise exponentiation (`a` raised to the power `b`).
    ///
    /// See [`jix::scalar::Pow`](crate::scalar::Pow) scalar trait for the per-element semantics.
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
    crate::scalar::Pow::pow,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(pow: Pow, crate::scalar::Pow);
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
            crate::util::TestArray<T>,
            crate::util::TestArray<T>,
        ),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(
                &crate::util::arrays2_strategy_generic::<T>(
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
                // `$body` is shared across all dtypes this macro is invoked with. For ops
                // like `greater`/`less` that are also instantiated at `bool`, an ordering
                // comparison such as `a > b` is a plain boolean comparison there rather than
                // a numeric one (clippy::bool_comparison).
                #[test]
                #[allow(clippy::bool_comparison)]
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

    #[test]
    fn sub_concrete() {
        use crate::Array;

        // i32: negatives, zero, positive, at the op_safe_strategy bound (+/-100); differences
        // stay well within i32 range so there is no overflow panic in the reference either.
        let nd_a = ndarray::array![[-100i32, -1, 0], [1, 50, 100]];
        let nd_b = ndarray::array![[100i32, 1, 0], [-1, -50, -100]];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a - b);
        crate::util::assert_array_matches(&(za - zb), &expected);

        // f64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default block
        // shape so this arm crosses a block boundary.
        let nd_a64 = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 50.0, 100.0]];
        let nd_b64 = ndarray::array![[100.0f64, 0.5, 0.0], [-0.5, -50.0, -100.0]];
        let za64 = Array::compact_ndarray_with(&nd_a64, crate::util::arr_params(&[1, 2])).unwrap();
        let zb64 = Array::compact_ndarray_with(&nd_b64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = ndarray::Zip::from(&nd_a64)
            .and(&nd_b64)
            .map_collect(|&a, &b| a - b);
        crate::util::assert_array_matches(&(za64 - zb64), &expected64);
    }

    #[test]
    fn mul_concrete() {
        use crate::Array;

        // i32: negatives, zero, positive, at the op_safe_strategy bound (+/-100); products
        // stay well within i32 range so there is no overflow panic in the reference either.
        let nd_a = ndarray::array![[-100i32, -1, 0], [1, 10, 100]];
        let nd_b = ndarray::array![[100i32, -1, 0], [-1, 10, -100]];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za * zb), &expected);

        // f64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default block
        // shape so this arm crosses a block boundary.
        let nd_a64 = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 10.0, 100.0]];
        let nd_b64 = ndarray::array![[100.0f64, -0.5, 0.0], [-0.5, 10.0, -100.0]];
        let za64 = Array::compact_ndarray_with(&nd_a64, crate::util::arr_params(&[1, 2])).unwrap();
        let zb64 = Array::compact_ndarray_with(&nd_b64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = ndarray::Zip::from(&nd_a64)
            .and(&nd_b64)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za64 * zb64), &expected64);
    }

    // mul, complex arm: full complex multiplication `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`,
    // with negative and zero components on both operands.
    #[cfg(feature = "num-complex")]
    #[test]
    fn mul_concrete_complex() {
        use crate::scalar::Complex;
        use crate::Array;

        let nd_a = ndarray::array![
            [
                Complex {
                    re: -3.0f32,
                    im: 4.0
                },
                Complex { re: 0.0, im: 0.0 }
            ],
            [
                Complex { re: 2.5, im: -1.5 },
                Complex {
                    re: 100.0,
                    im: -100.0
                }
            ],
        ];
        let nd_b = ndarray::array![
            [
                Complex {
                    re: 1.0f32,
                    im: -2.0
                },
                Complex { re: 5.0, im: 5.0 }
            ],
            [
                Complex { re: -2.5, im: 1.5 },
                Complex {
                    re: -100.0,
                    im: 100.0
                }
            ],
        ];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za * zb), &expected);
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

    // sub_u32: a = c + b guarantees a - b == c with no unsigned underflow.
    proptest::proptest! {
        #[test]
        fn sub_u32(
            ((nd_c, _zc), (nd_b, zb)) in crate::util::arrays2_strategy_generic::<u32>(
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
