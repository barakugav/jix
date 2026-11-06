use crate::ops::prelude::*;

pub(crate) struct Op2<S1, S2, K> {
    pub(crate) a: S1,
    pub(crate) b: S2,
    kernel: K,
    cache_id: ArrayCacheId,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op2Kernel<T1, T2> {
    type Output;

    fn apply(&self, a: T1, b: T2, status: &mut ArithmeticStatus) -> Self::Output;

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        a: [T1; N],
        b: [T2; N],
        status: &mut ArithmeticStatus,
        simd: S,
    ) -> [Self::Output; N] {
        let _ = simd;
        crate::util::array2_map_inline(
            a,
            b,
            #[inline(always)]
            |_, x, y| self.apply(x, y, status),
        )
    }
}
impl<S1, S2, K> Op2<S1, S2, K> {
    pub(crate) fn new(a: S1, b: S2, kernel: K, swapped_operand_order: bool) -> Result<Self>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped,
        K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        {
            let (mut a_shape, mut b_shape) = (a.shape(), b.shape());
            if swapped_operand_order {
                std::mem::swap(&mut a_shape, &mut b_shape);
            }
            ensure!(
                a_shape == b_shape,
                InvalidArgument,
                "Op2 shape mismatch between `a` {:?} and `b` {:?}",
                a_shape,
                b_shape
            );
            check_shape_overflow(a_shape, K::Output::DTYPE.itemsize() as _)?;
        }
        let cache_id = ArrayCacheId::new();
        let spec = {
            let (mut a_spec, mut b_spec) = (a.spec(), b.spec());
            if swapped_operand_order {
                std::mem::swap(&mut a_spec, &mut b_spec);
            }
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
            ArraySpecDynamic::new(
                block_shape,
                block_shape_fixed_dims,
                element_cost,
                read_shape_scale_weight,
                read_layout_order,
            )
            .with_cache_ids(Some(cache_id), [a_spec.dynamic(), b_spec.dynamic()])
        };
        Ok(Self {
            a,
            b,
            kernel,
            cache_id,
            spec,
        })
    }

    #[inline(always)]
    pub(crate) fn dynamic_spec(&self) -> &ArraySpecDynamic {
        &self.spec
    }
}

impl<S1, S2, K> crate::storage::ArrayStorageImpl for Op2<S1, S2, K> {}
impl<S1, S2, K> ArrayStorage for Op2<S1, S2, K>
where
    S1: ArrayStorageTyped,
    S2: ArrayStorageTyped,
    K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(
            out.as_deref(),
            self.shape(),
            index,
            self.dtype().itemsize() as usize,
        )?;
        context.read_cached(self.cache_id, index, self.dtype(), out, &|out| {
            self.read_as_elementwise_pipeline::<K::Output>(index, context)?
                .to_buf(index, context, out)
        })
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
            const MAX_ITEMSIZE: usize =
                max_itemsize(&[size_of::<T>(), P1::MAX_ITEMSIZE, P2::MAX_ITEMSIZE]);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.a.operands().chain(self.b.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool, S: fearless_simd::Simd>(
                &self,
                offset: usize,
                status: &mut ArithmeticStatus,
                simd: S,
            ) -> [T; N] {
                let a = unsafe { self.a.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let b = unsafe { self.b.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let ys = self.kernel.apply_bulk(a, b, status, simd);

                const { assert!(size_of::<K::Output>() == size_of::<T>()) };
                // SAFETY: we checked `T` and `K::Output` are the same dtype in the outer func
                unsafe { std::mem::transmute_copy::<[K::Output; N], [T; N]>(&ys) }
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
        debug_assert_eq!(shape, self.b.shape());
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

    crate::ops::impl_element_type_change_default!();
}

impl<F, T1, T2, O> Op2Kernel<T1, T2> for F
where
    F: Fn(T1, T2) -> O,
{
    type Output = O;
    #[inline(always)]
    fn apply(&self, a: T1, b: T2, _status: &mut ArithmeticStatus) -> Self::Output {
        self(a, b)
    }
}

macro_rules! define_op2 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        crate::scalar::$trait:ident::$kernel_fn:ident $(($status:ident))?,
        simd: $bulk_fn:ident,
        $(core_op: core::ops::$core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T1, T2> crate::ops::elementwise::op2::Op2Kernel<T1, T2> for $Kernel
        where
            T1: crate::scalar::$trait<T2> + Copy + 'static,
            T2: Copy + 'static,
        {
            type Output = <T1 as crate::scalar::$trait<T2>>::Output;

            #[inline(always)]
            fn apply(&self, a: T1, b: T2, status: &mut ArithmeticStatus) -> Self::Output {
                { let _ = status; } // suppress unused
                $(let $status = status;)?
                <T1 as crate::scalar::$trait<T2>>::$kernel_fn(a, b $(, $status)?)
            }

            #[inline(always)]
            fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
                &self,
                a: [T1; N],
                b: [T2; N],
                status: &mut ArithmeticStatus,
                simd: S,
            ) -> [Self::Output; N] {
                { let _ = status; } // suppress unused
                $(let $status = status;)?
                <T1 as crate::scalar::$trait<T2>>::$bulk_fn(a, b, $($status,)? simd)
            }
        }
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::elementwise::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(a: S1, b: S2) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::elementwise::op2::Op2::new(a, b, $Kernel, false)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(a: crate::Array<S1>, b: crate::Array<S2>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(a.into_storage(), b.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S1, S2> crate::storage::ArrayStorageImpl for $Op<S1, S2> {}
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: crate::scalar::$trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S1::Item as crate::scalar::$trait<S2::Item>>::Output>;
            crate::storage::impl_array_storage_forward!(<S1, S2>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.a, &self.0.b])
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
            S2: crate::storage::ArrayStorageTyped,
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

macro_rules! define_array_op2_method {
    ($method:ident : $Op:ident, $($trait:ident)::+) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+, fixed_lhs_type = $lhs_type:ty) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Item = $lhs_type>,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
}

pub(crate) use {define_array_op2_method, define_op2};

#[cfg(test)]
pub(crate) mod tests {
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
                // like `greater`/`less` that are also instantiated at `bool8`, an ordering
                // comparison such as `a > b` is a plain boolean comparison there rather than
                // a numeric one (clippy::bool_comparison).
                #[test]
                #[allow(clippy::bool_comparison)]
                fn [<$op_method _ $dtype:lower>]() {
                    crate::ops::elementwise::op2::tests::check_op2::<$dtype>(
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
            $(crate::ops::elementwise::op2::tests::test_op2_dtype!($op_method, |$a, $b| $body, $dtype, $strategy);)+
            $($(
                #[cfg($cfg)]
                crate::ops::elementwise::op2::tests::test_op2_dtype!($op_method, |$a, $b| $body, $cfg_dtype, $strategy);
            )+)*
        };
    }

    pub(crate) use {test_op2, test_op2_dtype};
}
