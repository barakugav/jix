use crate::ops::prelude::*;

pub(crate) struct Op1<S, K> {
    pub(crate) array: S,
    kernel: K,
    cache_id: ArrayCacheId,
    spec: ArraySpecDynamic,
}
pub(crate) trait Op1Kernel<T> {
    type Output;

    fn apply(&self, x: T, status: &mut ArithmeticStatus) -> Self::Output;

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        xs: [T; N],
        status: &mut ArithmeticStatus,
        simd: S,
    ) -> [Self::Output; N] {
        let _ = simd;
        xs.map_inline(
            #[inline(always)]
            |x| self.apply(x, status),
        )
    }
}
impl<S, K> Op1<S, K> {
    pub(crate) fn new(array: S, kernel: K) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: Op1Kernel<S::Item, Output: Dtyped>,
    {
        check_dtype_size_nonzero(&K::Output::DTYPE)?;
        check_shape_overflow(array.shape(), K::Output::DTYPE.itemsize() as _)?;
        let cache_id = ArrayCacheId::new();
        let mut spec = array.spec().dynamic().clone();
        spec.add_elementwise_cost(1.0);
        spec.add_cache_id(cache_id);
        Ok(Self {
            array,
            kernel,
            cache_id,
            spec,
        })
    }
}

impl<S, K> crate::storage::ArrayStorageImpl for Op1<S, K> {}
impl<S, K> ArrayStorage for Op1<S, K>
where
    S: ArrayStorageTyped,
    K: Op1Kernel<S::Item, Output: Dtyped>,
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
        context.read_cached(self.cache_id, index, self.dtype(), out, |out| {
            self.read_as_elementwise_pipeline::<K::Output>(index, context)?
                .to_buf(index, context, out)
        })
    }

    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S, K>>
    where
        T: Dtyped,
    {
        check_dtype(Dtype::new_ref::<T>(), Dtype::new_ref::<K::Output>())?;
        let inner = self
            .array
            .read_as_elementwise_pipeline::<S::Item>(index, context)?;

        struct Op1Pipeline<'a, P, K, TIn> {
            inner: P,
            kernel: &'a K,
            phantom: std::marker::PhantomData<TIn>,
        }
        impl<TIn, T, P, K> ElementwisePipelineImpl<T> for Op1Pipeline<'_, P, K, TIn>
        where
            P: ElementwisePipelineImpl<TIn>,
            K: Op1Kernel<TIn, Output: Dtyped>,
            T: Dtyped,
        {
            const N_OPERANDS: Option<usize> = P::N_OPERANDS;
            const MAX_ITEMSIZE: usize = max_itemsize(&[size_of::<T>(), P::MAX_ITEMSIZE]);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inner.operands()
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool, S: fearless_simd::Simd>(
                &self,
                offset: usize,
                status: &mut ArithmeticStatus,
                simd: S,
            ) -> [T; N] {
                let xs = unsafe {
                    self.inner
                        .read_bulk::<N, CONTIGUOUS, S>(offset, status, simd)
                };
                let ys = self.kernel.apply_bulk(xs, status, simd);

                const { assert!(size_of::<K::Output>() == size_of::<T>()) };
                // SAFETY: we checked `T` and `K::Output` are the same dtype in the outer func
                unsafe { std::mem::transmute_copy::<[K::Output; N], [T; N]>(&ys) }
            }
        }

        Ok(Op1Pipeline {
            inner,
            kernel: &self.kernel,
            phantom: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<K::Output>()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Op1", [&self.array])
    }

    crate::ops::impl_element_type_change_default!();
}

impl<F, T, O> Op1Kernel<T> for F
where
    F: Fn(T) -> O,
{
    type Output = O;
    #[inline(always)]
    fn apply(&self, x: T, _status: &mut ArithmeticStatus) -> Self::Output {
        self(x)
    }
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        crate::scalar::$trait:ident::$kernel_fn:ident,
        simd: $bulk_fn:ident,
        $(core_op: core::ops::$core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T> crate::ops::elementwise::op1::Op1Kernel<T> for $Kernel
        where
            T: crate::scalar::$trait + Copy + 'static,
        {
            type Output = <T as crate::scalar::$trait>::Output;

            #[inline(always)]
            fn apply(&self, x: T, _status: &mut ArithmeticStatus) -> Self::Output {
                <T as crate::scalar::$trait>::$kernel_fn(x)
            }

            #[inline(always)]
            fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
                &self,
                xs: [T; N],
                _status: &mut ArithmeticStatus,
                simd: S,
            ) -> [Self::Output; N] {
                <T as crate::scalar::$trait>::$bulk_fn(xs, simd)
            }
        }
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::elementwise::op1::Op1<S, $Kernel>);
        impl<S> $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::$trait<Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S) -> crate::error::Result<Self> {
                Ok(Self(crate::ops::elementwise::op1::Op1::new(array, $Kernel)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage()).map(crate::Array::from_storage)
            }
        }
        impl<S> crate::storage::ArrayStorageImpl for $Op<S> {}
        impl<S> ArrayStorage for $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::$trait<Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S::Item as crate::scalar::$trait>::Output>;
            crate::storage::impl_array_storage_forward!(<S>);

            fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
                crate::storage::ArrayStorageInfo::new_deps(stringify!($Op), [&self.0.array])
            }

            crate::ops::impl_element_type_change_default!();
        }

        define_op1!(@define_core
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
        impl<S> core::ops::$core_op_trait for Array<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: crate::scalar::$trait<Output: crate::dtype::Dtyped>,
        {
            type Output = Array<$Op<S>>;
            #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_op_fn(self) -> Self::Output {
                $Op::new_array(self).unwrap()
            }
        }
    };
}

macro_rules! define_array_op1_method {
    ($method:ident : $Op:ident, $($trait:ident)::+) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self) -> crate::Array<$Op<S>>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            $Op::new_array(self).unwrap()
        }
    };
}

pub(crate) use {define_array_op1_method, define_op1};

#[cfg(test)]
pub(crate) mod tests {
    use proptest::strategy::BoxedStrategy;
    use proptest::test_runner::{Config, TestRunner};

    /// Shared proptest driver for unary-op tests.
    ///
    /// Generic over the dtype only, with the op passed as a fn pointer, to avoid per-op
    /// monomorphization.
    #[inline(never)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn check_op1<T>(
        strategy: BoxedStrategy<T>,
        check: fn(&ndarray::ArrayD<T>, crate::util::TestArray<T>),
    ) where
        T: crate::util::ScalarStrategy + std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config::default());
        runner
            .run(
                &crate::util::array_strategy_from_shape::<T>(
                    crate::util::shape_strategy(),
                    strategy,
                ),
                |(nd, za)| {
                    check(&nd, za);
                    Ok(())
                },
            )
            .unwrap();
    }

    macro_rules! test_op1_dtype {
        ($op_method:ident, |$arg:ident| $body:expr, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                #[test]
                fn [<$op_method _ $dtype:lower>]() {
                    crate::ops::elementwise::op1::tests::check_op1::<$dtype>(
                        <$dtype as crate::util::ScalarStrategy>::$strategy(),
                        |nd, za| {
                            #[allow(unused_imports)]
                            use std::ops::Neg;
                            let result = za.$op_method();
                            let expected = nd.mapv(|$arg| $body);
                            crate::util::assert_array_matches(&result, &expected);
                        },
                    );
                }
            }
        };
    }

    macro_rules! test_op1 {
        (
            $op_method:ident, |$arg:ident| $body:expr,
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::elementwise::op1::tests::test_op1_dtype!($op_method, |$arg| $body, $dtype, $strategy);)+
            $($(
                #[cfg($cfg)]
                crate::ops::elementwise::op1::tests::test_op1_dtype!($op_method, |$arg| $body, $cfg_dtype, $strategy);
            )+)*
        };
    }

    pub(crate) use {test_op1, test_op1_dtype};
}
