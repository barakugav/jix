use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::ops::common::AxesArg;
#[allow(unused_imports)]
use crate::scalar::{f16, Complex};
use crate::storage::{ArrayStorageSpec, ArrayStorageTyped, BlocksLayout};
use crate::util::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::util::iter::NdIter;
use crate::util::{assert_unchecked_eq, default_strides, dim_arr, DimArray};
use crate::{Array, ArrayStorage, Dimension, Ty};

pub(crate) struct ReductionOp<S, K, D> {
    kernel: K,

    array: S,
    is_reduced: DimArray<bool>,

    out_dtype_: Dtype,
    shape: D,
    blocks_layout: BlocksLayout,
}
pub(crate) trait ReductionOpKernel<T> {
    type Output;
    fn reduce(&self, items: impl Iterator<Item = T>) -> Self::Output;
    fn supports_empty(&self) -> bool;
}
impl<S, K, D> ReductionOp<S, K, D> {
    pub(crate) fn new<Ax>(array: S, kernel: K, axes: Ax) -> Result<Self>
    where
        S: ArrayStorageTyped,
        K: ReductionOpKernel<S::Item, Output: Dtyped>,
        D: Dimension,
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        let input_ndim = array.shape().len();
        let mut is_reduced = dim_arr(input_ndim, |_| false);
        for i in 0..axes.len() {
            let ax = axes.get(i);
            ensure!(
                ax < input_ndim,
                InvalidArgument,
                "axis {ax} out of bounds for array of ndim {input_ndim}"
            );

            ensure!(!is_reduced[ax], InvalidArgument, "duplicate axis {ax}");
            is_reduced[ax] = true;
        }

        if !kernel.supports_empty()
            && array
                .shape()
                .iter()
                .zip(&is_reduced)
                .any(|(&s, &reduced)| reduced && s == 0)
        {
            bail!(
                InvalidArgument,
                "reduction on empty dimension not supported"
            );
        }

        let shape = array
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(dim, &s)| is_reduced[dim].not().then_some(s))
            .collect::<DimArray<_>>();
        let shape = D::from_slice(&shape).unwrap();

        let mut b_layout = array.spec().blocks_layout.clone();
        b_layout.block_shape_hint = (0..input_ndim)
            .filter_map(|d| is_reduced[d].not().then_some(b_layout.block_shape_hint[d]))
            .collect();
        b_layout.block_shape_tag = (0..input_ndim)
            .filter_map(|d| is_reduced[d].not().then_some(b_layout.block_shape_tag[d]))
            .collect();
        b_layout.preferred_read_shape = (0..input_ndim)
            .filter_map(|d| {
                is_reduced[d]
                    .not()
                    .then_some(b_layout.preferred_read_shape[d])
            })
            .collect();

        Ok(Self {
            kernel,
            out_dtype_: K::Output::DTYPE,
            shape,
            blocks_layout: b_layout,
            array,
            is_reduced,
        })
    }
}

impl<S, K, D> ArrayStorage for ReductionOp<S, K, D>
where
    S: ArrayStorageTyped,
    K: ReductionOpKernel<S::Item, Output: Dtyped>,
    D: Dimension,
{
    type ElementType = Ty<K::Output>;
    type Dimension = D;

    #[inline]
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        let src_dtype = self.array.dtype();
        let dst_dtype = self.dtype();
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, dst_dtype, buf)?;

        let orig_shape = self.array.shape();
        let orig_ndim = orig_shape.len();

        // Build inner_index: reduced dims span the full original range,
        // non-reduced dims forward the requested output range.
        let mut out_dim = 0usize;
        let inner_index = (0..orig_ndim)
            .map(|in_d| {
                if self.is_reduced[in_d] {
                    // TODO: we could read it in chunks
                    0..orig_shape[in_d]
                } else {
                    let r = index[out_dim].clone();
                    out_dim += 1;
                    r
                }
            })
            .collect::<DimArray<_>>();

        let inner_read_shape = inner_index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<DimArray<_>>();
        let n_inner: usize = inner_read_shape.iter().product();

        let out_shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<DimArray<_>>();

        // Read the full inner block into a temp buffer.
        let tmp_buf_size = n_inner * src_dtype.itemsize() as usize;
        let mut tmp_buf = context.tmp_buf(tmp_buf_size, src_dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array.read_data(&inner_index, tmp_buf, context)?;

        // C-contiguous byte strides for inner and output layouts.
        let inner_strides = default_strides(&inner_read_shape, src_dtype.itemsize() as usize);
        let out_strides = default_strides(&out_shape, dst_dtype.itemsize() as usize);

        // Strides into tmp_buf for the output iterator: reduced dims are absent.
        let tmp_buf_strides = inner_strides
            .iter()
            .zip(&self.is_reduced)
            .filter_map(|(&s, &reduced)| reduced.not().then_some(s))
            .collect::<DimArray<_>>();

        let mut out_iter = NdIter::new(
            &out_shape,
            (
                NdIterExtStridesPtr::new(&tmp_buf_strides, tmp_buf.as_ptr()),
                NdIterExtStridesPtrMut::new(&out_strides, buf.as_mut_ptr()),
            ),
        );
        let reduction_shape = dim_arr(orig_ndim, |d| {
            if self.is_reduced[d] {
                orig_shape[d]
            } else {
                1
            }
        });

        while let Some((_out_idx, (base_ptr, out_ptr))) = out_iter.next() {
            let reduction_iter = NdIter::new(
                &reduction_shape,
                NdIterExtStridesPtr::new(&inner_strides, base_ptr),
            );
            let reduction_iter =
                reduction_iter.map(|(_idx, in_ptr)| unsafe { in_ptr.cast::<S::Item>().read() });
            let res = self.kernel.reduce(reduction_iter);
            unsafe { out_ptr.cast::<K::Output>().write(res) };
        }

        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        let dtype = &self.out_dtype_;
        unsafe { assert_unchecked_eq!(dtype, &K::Output::DTYPE) };
        dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.spec()
        }
    }
}

macro_rules! define_reduction_op {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        $($trait:ident)::+,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        single_axis = true
    ) => {
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::reduction::ReductionOp<S, $Kernel, <S::Dimension as crate::Dimension>::Smaller>)
        where
            S: crate::ArrayStorage;
        impl<S> $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: S, axis: usize $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self> {
                let kernel = $Kernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(array, kernel, &[axis])?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(array: crate::Array<S>, axis: usize $(, $extra_arg: $extra_ty)*) -> crate::error::Result<crate::Array<Self>> {
                Self::new(array.into_storage(), axis $(, $extra_arg)*).map(crate::Array::from_storage)
            }
        }

        impl<S> crate::ArrayStorage for $Op<S>
        where
            S: crate::storage::ArrayStorageTyped,
             S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S::Item as $($trait)::+>::Output>;
            type Dimension = <S::Dimension as crate::Dimension>::Smaller;

            crate::storage::impl_array_storage_forward!(<S>);
        }

        define_reduction_op!(
            @define_kernel
            $Kernel,
            $($trait)::+,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
        );
    };

    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        $($trait:ident)::+,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
    ) => {
        $(#[$meta])*
        pub struct $Op<S, D>(crate::ops::reduction::ReductionOp<S, $Kernel, D>);
        impl<S, D> $Op<S, D>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
            D: crate::Dimension,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new<Ax>(array: S, axes: Ax $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self>
            where
                Ax: crate::ops::AxesArg<ReducedDimension<S::Dimension> = D>,
            {
                let kernel = $Kernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(array, kernel, axes)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array<Ax>(array: crate::Array<S>, axes: Ax $(, $extra_arg: $extra_ty)*) -> crate::error::Result<crate::Array<Self>>
            where
                Ax: crate::ops::AxesArg<ReducedDimension<S::Dimension> = D>,
            {
                Self::new(array.into_storage(), axes $(, $extra_arg)*).map(crate::Array::from_storage)
            }
        }

        impl<S, D> crate::ArrayStorage for $Op<S, D>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
            D: crate::Dimension,
        {
            type ElementType = crate::Ty<<S::Item as $($trait)::+>::Output>;
            type Dimension = D;

            crate::storage::impl_array_storage_forward!(<S, D>);
        }

        define_reduction_op!(
            @define_kernel
            $Kernel,
            $($trait)::+,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
        );
    };

    (
        @define_kernel
        $Kernel:ident,
        $($trait:ident)::+,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
    ) => {
        struct $Kernel { $($extra_arg: $extra_ty,)* }
        impl<T> crate::ops::reduction::ReductionOpKernel<T> for $Kernel
        where
            T: $($trait)::+,
        {
            type Output = <T as $($trait)::+>::Output;

            #[inline]
            fn reduce(&self, items: impl Iterator<Item = T>) -> Self::Output {
                #[allow(unused)]
                $(let $extra_arg = self.$extra_arg;)*
                let $arg_items = items;
                { $body }.unwrap()
            }

            #[inline(always)]
            fn supports_empty(&self) -> bool {
                $support_empty
            }
        }
    };

    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident: Iterator<Item = $in_type:ty> $(, $extra_arg:ident : $extra_ty:ty)*| -> $out_type:ty { $body:expr },
    ) => {
        $(#[$meta])*
        pub struct $Op<S, D>(crate::ops::reduction::ReductionOp<S, $Kernel, D>);
        impl<S, D> $Op<S, D>
        where
            S: crate::storage::ArrayStorageTyped<Item = $in_type>,
            D: crate::Dimension,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new<Ax>(array: S, axes: Ax $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self>
            where
                Ax: crate::ops::AxesArg<ReducedDimension<S::Dimension> = D>,
            {
                let kernel = $Kernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(array, kernel, axes)?))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array<Ax>(array: crate::Array<S>, axes: Ax $(, $extra_arg: $extra_ty)*) -> crate::error::Result<crate::Array<Self>>
            where
                Ax: crate::ops::AxesArg<ReducedDimension<S::Dimension> = D>,
            {
                Self::new(array.into_storage(), axes $(, $extra_arg)*).map(crate::Array::from_storage)
            }
        }

        impl<S, D> crate::ArrayStorage for $Op<S, D>
        where
            S: crate::storage::ArrayStorageTyped<Item = $in_type>,
            D: crate::Dimension,
        {
            type ElementType = crate::Ty<$out_type>;
            type Dimension = D;

            crate::storage::impl_array_storage_forward!(<S, D>);
        }


        struct $Kernel { $($extra_arg: $extra_ty,)* }
        impl crate::ops::reduction::ReductionOpKernel<$in_type> for $Kernel
        {
            type Output = $out_type;

            #[inline]
            fn reduce(&self, items: impl Iterator<Item = $in_type>) -> Self::Output {
                #[allow(unused)]
                $(let $extra_arg = self.$extra_arg;)*
                let $arg_items = items;
                { $body }.unwrap()
            }

            #[inline(always)]
            fn supports_empty(&self) -> bool {
                $support_empty
            }
        }
    };

}
// pub(crate) use {define_reduction_op};

/// Internal scalar-level reduction kernels used by the reduction op storage wrappers.
pub(crate) mod _traits {
    #[allow(unused_imports)]
    use crate::scalar::{f16, Complex};

    /// Scalar kernel trait for the element-wise `max` reduction.
    pub trait ReduceMax {
        /// The output element type of this reduction (same as the input for scalar max).
        type Output;
        /// Reduce `items` to their maximum value. Returns `None` if the iterator is empty.
        fn reduce_max(items: impl Iterator<Item = Self>) -> Option<Self::Output>;
    }
    impl<T> ReduceMax for T
    where
        T: crate::scalar::Maximum<Output = T>,
    {
        type Output = Self;

        #[inline]
        fn reduce_max(items: impl Iterator<Item = T>) -> Option<Self::Output> {
            items.reduce(|m, x| m.maximum(x))
        }
    }

    /// Scalar kernel trait for the element-wise `min` reduction.
    pub trait ReduceMin {
        /// The output element type of this reduction (same as the input for scalar min).
        type Output;
        /// Reduce `items` to their minimum value. Returns `None` if the iterator is empty.
        fn reduce_min(items: impl Iterator<Item = Self>) -> Option<Self::Output>;
    }
    impl<T> ReduceMin for T
    where
        T: crate::scalar::Minimum<Output = T>,
    {
        type Output = Self;

        #[inline]
        fn reduce_min(items: impl Iterator<Item = Self>) -> Option<Self::Output> {
            items.reduce(|m, x| m.minimum(x))
        }
    }

    /// Scalar kernel trait for `argmax`: finds the index of the maximum element.
    pub trait ArgMax {
        /// The output type for the flat index - always `u64` for concrete impls.
        type Output;
        /// Return the flat index of the maximum element in `items`, or `None` if empty.
        ///
        /// For floating-point types, comparison uses [`PartialOrd`]: `NaN` values behave
        /// as unordered and the result is unspecified when `NaN` is present.
        fn argmax(items: impl Iterator<Item = Self>) -> Option<Self::Output>;
    }
    impl<T> ArgMax for T
    where
        T: PartialOrd,
    {
        type Output = u64;

        #[inline]
        fn argmax(items: impl Iterator<Item = Self>) -> Option<Self::Output> {
            let mut idx = 0u64;
            items
                .map({
                    |x| {
                        let i = idx;
                        idx += 1;
                        (i, x)
                    }
                })
                .reduce(|(m_idx, m), (idx, x)| if x > m { (idx, x) } else { (m_idx, m) })
                .map(|(idx, _)| idx)
        }
    }

    /// Scalar kernel trait for `argmin`: finds the index of the minimum element.
    pub trait ArgMin {
        /// The output type for the flat index - always `u64` for concrete impls.
        type Output;
        /// Return the flat index of the minimum element in `items`, or `None` if empty.
        ///
        /// For floating-point types, comparison uses [`PartialOrd`]: `NaN` values behave
        /// as unordered and the result is unspecified when `NaN` is present.
        fn argmin(items: impl Iterator<Item = Self>) -> Option<Self::Output>;
    }
    impl<T> ArgMin for T
    where
        T: PartialOrd,
    {
        type Output = u64;

        #[inline]
        fn argmin(items: impl Iterator<Item = Self>) -> Option<Self::Output> {
            let mut idx = 0u64;
            items
                .map({
                    |x| {
                        let i = idx;
                        idx += 1;
                        (i, x)
                    }
                })
                .reduce(|(m_idx, m), (idx, x)| if x < m { (idx, x) } else { (m_idx, m) })
                .map(|(idx, _)| idx)
        }
    }

    /// Scalar kernel trait for the element-wise `sum` reduction.
    ///
    /// Accumulates into a wider output type to reduce overflow risk: integer types accumulate
    /// into `i64`/`u64`, floating-point types accumulate into `f64`.
    pub trait ReduceSum {
        /// The output element type (wider than the input for most types).
        type Output;
        /// Sum all elements in `items`, starting from zero. Returns zero for an empty iterator.
        fn reduce_sum(items: impl Iterator<Item = Self>) -> Self::Output;
    }

    macro_rules! impl_sum {
        ($item_ty:ty, $output_ty:ty) => {
            impl ReduceSum for $item_ty {
                type Output = $output_ty;

                #[inline]
                fn reduce_sum(items: impl Iterator<Item = Self>) -> Self::Output {
                    items.fold(<_ as crate::scalar::Cast<Self::Output>>::cast(0), |m, x| {
                        m + <_ as crate::scalar::Cast<Self::Output>>::cast(x)
                    })
                }
            }
        };
    }
    impl_sum!(i8, i64);
    impl_sum!(i16, i64);
    impl_sum!(i32, i64);
    impl_sum!(i64, i64);
    impl_sum!(u8, u64);
    impl_sum!(u16, u64);
    impl_sum!(u32, u64);
    impl_sum!(u64, u64);
    #[cfg(feature = "half")]
    impl_sum!(f16, f64);
    impl_sum!(f32, f64);
    impl_sum!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f64>, Complex<f64>);
    impl_sum!(bool, u64);

    /// Scalar kernel trait for the element-wise `product` reduction.
    ///
    /// Accumulates into a wider output type to reduce overflow risk: integer types accumulate
    /// into `i64`/`u64`, floating-point types accumulate into `f64`.
    pub trait ReduceProduct {
        /// The output element type (wider than the input for most types).
        type Output;
        /// Multiply all elements in `items`, starting from one. Returns one for an empty iterator.
        fn reduce_product(items: impl Iterator<Item = Self>) -> Self::Output;
    }
    macro_rules! impl_product {
        ($item_ty:ty, $output_ty:ty) => {
            impl ReduceProduct for $item_ty {
                type Output = $output_ty;

                #[inline]
                fn reduce_product(items: impl Iterator<Item = Self>) -> Self::Output {
                    items.fold(<_ as crate::scalar::Cast<Self::Output>>::cast(1), |m, x| {
                        m * <_ as crate::scalar::Cast<Self::Output>>::cast(x)
                    })
                }
            }
        };
    }
    impl_product!(i8, i64);
    impl_product!(i16, i64);
    impl_product!(i32, i64);
    impl_product!(i64, i64);
    impl_product!(u8, u64);
    impl_product!(u16, u64);
    impl_product!(u32, u64);
    impl_product!(u64, u64);
    #[cfg(feature = "half")]
    impl_product!(f16, f64);
    impl_product!(f32, f64);
    impl_product!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f64>, Complex<f64>);

    /// Scalar kernel trait for the element-wise `mean` reduction.
    ///
    /// The mean is computed as the sum divided by the count; the output is always `f64`
    /// (or `Complex<f64>` for complex inputs) to preserve precision.
    pub trait ReduceMean {
        /// The output element type - always `f64` or `Complex<f64>`.
        type Output;
        /// Compute the arithmetic mean of `items`. Returns `None` if the iterator is empty.
        fn reduce_mean(items: impl Iterator<Item = Self>) -> Option<Self::Output>;
    }
    macro_rules! impl_mean {
        ($item_ty:ty, $output_ty:ty) => {
            impl ReduceMean for $item_ty {
                type Output = $output_ty;

                #[inline]
                fn reduce_mean(items: impl Iterator<Item = Self>) -> Option<Self::Output> {
                    let (size, size_high) = items.size_hint();
                    assert_eq!(Some(size), size_high);
                    if size == 0 {
                        return None;
                    }
                    let sum = <Self as ReduceSum>::reduce_sum(items);
                    Some(<_ as crate::scalar::Cast<Self::Output>>::cast(sum) / size as f64)
                }
            }
        };
    }
    impl_mean!(i8, f64);
    impl_mean!(i16, f64);
    impl_mean!(i32, f64);
    impl_mean!(i64, f64);
    impl_mean!(u8, f64);
    impl_mean!(u16, f64);
    impl_mean!(u32, f64);
    impl_mean!(u64, f64);
    #[cfg(feature = "half")]
    impl_mean!(f16, f64);
    impl_mean!(f32, f64);
    impl_mean!(f64, f64);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f64>, Complex<f64>);
    impl_mean!(bool, f64);

    /// Scalar kernel trait for the `var` (variance) and `std` (standard deviation) reductions.
    ///
    /// The degree-of-freedom correction is controlled by `ddof`: use `0.0` for population
    /// variance (`N` denominator) and `1.0` for sample variance (`N-1` denominator).
    pub trait ReduceVariance {
        /// The output element type - always a `Float` (i.e. `f64` for most inputs).
        type Output: num_traits::Float;
        /// Compute the variance of `items` with `ddof` degrees-of-freedom correction.
        fn reduce_variance(items: impl Iterator<Item = Self>, ddof: f64) -> Self::Output;
        /// Compute the standard deviation of `items` with `ddof` degrees-of-freedom correction.
        ///
        /// This is `sqrt(variance)` using the same `ddof`.
        #[inline(always)]
        fn reduce_std(items: impl Iterator<Item = Self>, ddof: f64) -> Self::Output {
            let var = Self::reduce_variance(items, ddof);
            <_ as num_traits::Float>::sqrt(var)
        }
    }
    macro_rules! impl_variance {
        ($item_ty:ty) => {
            impl ReduceVariance for $item_ty {
                type Output = f64;

                #[inline(always)]
                fn reduce_variance(items: impl Iterator<Item = Self>, ddof: f64) -> Self::Output {
                    variance_impl(items, ddof)
                }
            }
        };
    }
    impl_variance!(i8);
    impl_variance!(i16);
    impl_variance!(i32);
    impl_variance!(i64);
    impl_variance!(u8);
    impl_variance!(u16);
    impl_variance!(u32);
    impl_variance!(u64);
    #[cfg(feature = "half")]
    impl_variance!(f16);
    impl_variance!(f32);
    impl_variance!(f64);
    #[cfg(feature = "num-complex")]
    impl_variance!(Complex<f32>);
    #[cfg(feature = "num-complex")]
    impl_variance!(Complex<f64>);
    impl_variance!(bool);
    fn variance_impl<T>(items: impl Iterator<Item = T>, ddof: f64) -> f64
    where
        T: VarianceImpl,
        i32: crate::scalar::Cast<T::MeanType>,
        T: crate::scalar::Cast<T::MeanType>,
        T::MeanType: core::ops::Sub<T::MeanType, Output = T::MeanType>
            + core::ops::Div<f64, Output = T::MeanType>
            + core::ops::AddAssign<T::MeanType>
            + Copy,
    {
        let mut mean = <_ as crate::scalar::Cast<T::MeanType>>::cast(0);
        let mut m2 = 0.0_f64;
        let mut n = 0_u64;

        for x in items {
            let x = <_ as crate::scalar::Cast<T::MeanType>>::cast(x);
            n += 1;
            let delta = x - mean;
            mean += delta / n as f64;
            let delta2 = x - mean;
            m2 += T::update_m2(delta, delta2);
        }

        let denom = n as f64 - ddof;
        if denom <= 0.0 {
            f64::NAN
        } else {
            m2 / denom
        }
    }
    trait VarianceImpl {
        type MeanType;
        fn update_m2(delta: Self::MeanType, delta2: Self::MeanType) -> f64;
    }
    macro_rules! impl_num_variance_impl {
        ($ty:ty) => {
            impl VarianceImpl for $ty {
                type MeanType = f64;
                fn update_m2(delta: f64, delta2: f64) -> f64 {
                    delta * delta2
                }
            }
        };
    }
    macro_rules! impl_complex_variance_impl {
        ($f_ty:ty) => {
            impl VarianceImpl for Complex<$f_ty> {
                type MeanType = Complex<f64>;
                fn update_m2(delta: Complex<f64>, delta2: Complex<f64>) -> f64 {
                    delta.re * delta2.re + delta.im * delta2.im
                }
            }
        };
    }
    impl_num_variance_impl!(i8);
    impl_num_variance_impl!(i16);
    impl_num_variance_impl!(i32);
    impl_num_variance_impl!(i64);
    impl_num_variance_impl!(u8);
    impl_num_variance_impl!(u16);
    impl_num_variance_impl!(u32);
    impl_num_variance_impl!(u64);
    #[cfg(feature = "half")]
    impl_num_variance_impl!(f16);
    impl_num_variance_impl!(f32);
    impl_num_variance_impl!(f64);
    impl_complex_variance_impl!(f32);
    impl_complex_variance_impl!(f64);
    impl_num_variance_impl!(bool);
}

define_reduction_op!(
    /// Reduces one or more axes by taking the maximum element.
    ///
    /// For **float** types, `NaN` values are ignored: if at least one non-`NaN` value
    /// is present, the result is the maximum of the non-`NaN` values. If all elements
    /// are `NaN`, the result is `NaN`. This deviates from the element-wise [`Maximum`](crate::ops::Maximum)
    /// op (which propagates `NaN`) but matches `numpy.max`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::max()`](crate::Array::max).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all axes -> scalar
    /// let scalar = Array::compact_array(&nd)?
    ///     .max((0, 1)).to_ndarray()?;
    /// assert_eq!(scalar[[]], 6);
    ///
    /// // Reduce axis 0 -> shape [3]
    /// let col_max = Array::compact_array(&nd)?
    ///     .max(0).to_ndarray()?;
    /// assert_eq!(col_max.as_slice().unwrap(), &[4, 5, 6]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Max,
    MaxKernel,
    crate::scalar::ReduceMax,
    support_empty = false,
    |items| { <T as crate::scalar::ReduceMax>::reduce_max(items) },
);
define_reduction_op!(
    /// Reduces one or more axes by taking the minimum element.
    ///
    /// For **float** types, `NaN` values are ignored: if at least one non-`NaN` value
    /// is present, the result is the minimum of the non-`NaN` values. If all elements
    /// are `NaN`, the result is `NaN`. This deviates from the element-wise [`Minimum`](crate::ops::Minimum)
    /// op (which propagates `NaN`) but matches `numpy.min`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::min()`](crate::Array::min).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all axes -> scalar
    /// let scalar = Array::compact_array(&nd)?
    ///     .min((0, 1)).to_ndarray()?;
    /// assert_eq!(scalar[[]], 1);
    ///
    /// // Reduce axis 0 -> shape [3]
    /// let col_min = Array::compact_array(&nd)?
    ///     .min(0).to_ndarray()?;
    /// assert_eq!(col_min.as_slice().unwrap(), &[1, 2, 3]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Min,
    MinKernel,
    crate::scalar::ReduceMin,
    support_empty = false,
    |items| { <T as crate::scalar::ReduceMin>::reduce_min(items) },
);
define_reduction_op!(
    /// Reduces a single axis by returning the index of the maximum element.
    ///
    /// Output dtype is `u64`.
    ///
    /// Unlike [`Max`], this op accepts only a single axis. If multiple elements share
    /// the maximum value, the index of the first occurrence is returned.
    /// For **float** types, `NaN` values are treated as less than any non-`NaN` value,
    /// so they are never selected unless all elements are `NaN`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::argmax()`](crate::Array::argmax).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of max along axis 1 (per row) -> shape [2]
    /// let idx = Array::compact_array(&nd)?
    ///     .argmax(1).to_ndarray()?;
    /// assert_eq!(idx.as_slice().unwrap(), &[1, 2]); // max of row 0 at col 1, row 1 at col 2
    ///
    /// // Index of max along axis 0 (per column) -> shape [3]
    /// let col_idx = Array::compact_array(&nd)?
    ///     .argmax(0).to_ndarray()?;
    /// assert_eq!(col_idx.as_slice().unwrap(), &[1, 0, 1]); // max of col 0 at row 1, col 1 at row 0, col 2 at row 1
    /// # Ok::<(), jix::Error>(())
    /// ```
    ArgMax,
    ArgMaxKernel,
    crate::scalar::ArgMax,
    support_empty = false,
    |items| {
        <T as crate::scalar::ArgMax>::argmax(items)
    },
    single_axis = true
);
define_reduction_op!(
    /// Reduces a single axis by returning the index of the minimum element.
    ///
    /// Output dtype is `u64`.
    ///
    /// Unlike [`Min`], this op accepts only a single axis. If multiple elements share
    /// the minimum value, the index of the first occurrence is returned.
    /// For **float** types, `NaN` values are treated as greater than any non-`NaN` value,
    /// so they are never selected unless all elements are `NaN`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::argmin()`](crate::Array::argmin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of min along axis 1 (per row) -> shape [2]
    /// let idx = Array::compact_array(&nd)?
    ///     .argmin(1).to_ndarray()?;
    /// assert_eq!(idx.as_slice().unwrap(), &[0, 1]); // min of row 0 at col 0, row 1 at col 1
    ///
    /// // Index of min along axis 0 (per column) -> shape [3]
    /// let col_idx = Array::compact_array(&nd)?
    ///     .argmin(0).to_ndarray()?;
    /// assert_eq!(col_idx.as_slice().unwrap(), &[0, 1, 0]); // min of col 0 at row 0, col 1 at row 1, col 2 at row 0
    /// # Ok::<(), jix::Error>(())
    /// ```
    ArgMin,
    ArgMinKernel,
    crate::scalar::ArgMin,
    support_empty = false,
    |items| {
        <T as crate::scalar::ArgMin>::argmin(items)
    },
    single_axis = true
);
define_reduction_op!(
    /// Reduces one or more axes by summing all elements along those axes.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64`, `bool` | `u64` |
    /// | `f16`, `f32`, `f64` | `f64` |
    /// | `Complex<f32>`, `Complex<f64>` | `Complex<f64>` |
    ///
    /// The output dtype is always widened to avoid overflow on large reductions.
    /// An empty reduction (zero elements along the reduced axes) returns `0`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sum()`](crate::Array::sum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Sum all elements -> i64
    /// let total = Array::compact_array(&nd)?
    ///     .sum((0, 1)).to_ndarray()?;
    /// assert_eq!(total[[]], 21);
    ///
    /// // Sum along axis 0 -> shape [3]
    /// let col_sums = Array::compact_array(&nd)?
    ///     .sum(0).to_ndarray()?;
    /// assert_eq!(col_sums.as_slice().unwrap(), &[5, 7, 9]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sum,
    SumKernel,
    crate::scalar::ReduceSum,
    support_empty = true,
    |items| { Some(<T as crate::scalar::ReduceSum>::reduce_sum(items)) },
);
define_reduction_op!(
    /// Reduces one or more axes by multiplying all elements along those axes.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64` | `u64` |
    /// | `f16`, `f32`, `f64` | `f64` |
    /// | `Complex<f32>`, `Complex<f64>` | `Complex<f64>` |
    ///
    /// Note: `bool` is not supported. The output dtype is always widened.
    /// An empty reduction (zero elements along the reduced axes) returns `1`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::product()`](crate::Array::product).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Product of all elements -> i64
    /// let total = Array::compact_array(&nd)?
    ///     .product((0, 1)).to_ndarray()?;
    /// assert_eq!(total[[]], 720);
    ///
    /// // Product along axis 0 -> shape [3]
    /// let col_products = Array::compact_array(&nd)?
    ///     .product(0).to_ndarray()?;
    /// assert_eq!(col_products.as_slice().unwrap(), &[4, 10, 18]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Product,
    ProductKernel,
    crate::scalar::ReduceProduct,
    support_empty = true,
    |items| { Some(<T as crate::scalar::ReduceProduct>::reduce_product(items)) },
);
define_reduction_op!(
    /// Reduces one or more axes by computing the arithmetic mean.
    ///
    /// Output dtype is `f64` for all scalar inputs; `Complex<f64>` for `Complex<f32>` and
    /// `Complex<f64>` inputs.
    ///
    /// Reducing an empty slice (zero elements) panics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::mean()`](crate::Array::mean).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Mean of all elements -> f64
    /// let total = Array::compact_array(&nd)?
    ///     .mean((0, 1)).to_ndarray()?;
    /// assert_eq!(total[[]], 3.5);
    ///
    /// // Mean along axis 0 -> shape [3]
    /// let col_means = Array::compact_array(&nd)?
    ///     .mean(0).to_ndarray()?;
    /// assert_eq!(col_means.as_slice().unwrap(), &[2.5, 3.5, 4.5]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Mean,
    MeanKernel,
    crate::scalar::ReduceMean,
    support_empty = false,
    |items| { <T as crate::scalar::ReduceMean>::reduce_mean(items) },
);
define_reduction_op!(
    /// Reduces one or more axes by computing the variance.
    ///
    /// Output dtype is `f64`. For complex inputs the result is the real-valued variance
    /// `E[|x - mean|^2]`.
    ///
    /// The `ddof` parameter (delta degrees of freedom) adjusts the divisor: the variance
    /// is computed as `sum((x - mean)^2) / (n - ddof)`. Use `ddof=0` for the population
    /// variance and `ddof=1` for the sample variance (Bessel's correction). If
    /// `n - ddof <= 0`, the result is `NaN`.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::var()`](crate::Array::var).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population variance (ddof=0) of all elements -> f64
    /// let var_all = Array::compact_array(&nd)?
    ///     .var((0, 1), 0.0).to_ndarray()?;
    /// assert!((var_all[[]] - 2.9167).abs() < 0.001);
    ///
    /// // Sample variance (ddof=1) along axis 0 -> shape [3]
    /// let col_vars = Array::compact_array(&nd)?
    ///     .var(0, 1.0).to_ndarray()?;
    /// assert_eq!(col_vars.as_slice().unwrap(), &[4.5, 4.5, 4.5]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Variance,
    VarianceKernel,
    crate::scalar::ReduceVariance,
    support_empty = false,
    |items, ddof: f64| { Some(<T as crate::scalar::ReduceVariance>::reduce_variance(items, ddof)) },
);
define_reduction_op!(
    /// Reduces one or more axes by computing the standard deviation.
    ///
    /// Output dtype is `f64`. For complex inputs the result is the real-valued standard
    /// deviation `sqrt(E[|x - mean|^2])`.
    ///
    /// Equivalent to `sqrt(variance)`. The `ddof` parameter has the same meaning as in
    /// [`Variance`]: use `ddof=0` for population std and `ddof=1` for sample std.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::std()`](crate::Array::std).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population std (ddof=0) of all elements -> f64
    /// let std_all = Array::compact_array(&nd)?
    ///     .std((0, 1), 0.0).to_ndarray()?;
    /// assert!((std_all[[]] - 1.7078).abs() < 0.001);
    ///
    /// // Sample std (ddof=1) along axis 0 -> shape [3]
    /// let col_stds = Array::compact_array(&nd)?
    ///     .std(0, 1.0).to_ndarray()?;
    /// assert!((col_stds[[0]] - 2.1213).abs() < 0.001);
    /// # Ok::<(), jix::Error>(())
    /// ```
    StandardDeviation,
    StandardDeviationKernel,
    crate::scalar::ReduceVariance,
    support_empty = false,
    |items, ddof: f64| {{
        Some(<T as crate::scalar::ReduceVariance>::reduce_std(items, ddof))
    }},
);

define_reduction_op!(
    /// Reduces one or more axes by testing whether all elements are `true`.
    ///
    /// The input array must contain `bool` elements. Output dtype is `bool`.
    /// Returns `true` only when every element is `true`. An empty reduction returns `true`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::all()`](crate::Array::all).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[true, false, true], [true, true, true]];
    ///
    /// // All elements true? -> false (contains false)
    /// let all_true = Array::compact_array(&nd)?
    ///     .all((0, 1)).to_ndarray()?;
    /// assert_eq!(all_true[[]], false);
    ///
    /// // All true along axis 0 (per column) -> shape [3]
    /// let col_all = Array::compact_array(&nd)?
    ///     .all(0).to_ndarray()?;
    /// assert_eq!(col_all.as_slice().unwrap(), &[true, false, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    All,
    AllKernel,
    support_empty = true,
    |items: Iterator<Item = bool>| -> bool {
        #[allow(clippy::unnecessary_fold)]
        Some(items.fold(true, |m, x| m && x))
    },
);
define_reduction_op!(
    /// Reduces one or more axes by testing whether any element is `true`.
    ///
    /// The input array must contain `bool` elements. Output dtype is `bool`.
    /// Returns `true` when at least one element is `true`. An empty reduction returns `false`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::any()`](crate::Array::any).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[false, false, false], [true, true, true]];
    ///
    /// // Any element true? -> true
    /// let any_true = Array::compact_array(&nd)?
    ///     .any((0, 1)).to_ndarray()?;
    /// assert_eq!(any_true[[]], true);
    ///
    /// // Any true along axis 0 (per column) -> shape [3]
    /// let col_any = Array::compact_array(&nd)?
    ///     .any(0).to_ndarray()?;
    /// assert_eq!(col_any.as_slice().unwrap(), &[true, true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Any,
    AnyKernel,
    support_empty = true,
    |items: Iterator<Item = bool>| -> bool {
        #[allow(clippy::unnecessary_fold)]
        Some(items.fold(false, |m, x| m || x))
    },
);

macro_rules! define_array_reduction_method {
    ($method:ident : $Op:ident, $($trait:ident)::+, single_axis = true $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self, axis: usize $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S>>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
            <S::Item as $($trait)::+>::Output: crate::dtype::Dtyped,
        {
            $Op::new_array(self, axis $($(, $extra_arg)*)?).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+ $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<Ax>(self, axis: Ax $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S, Ax::ReducedDimension<S::Dimension>>>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
            <S::Item as $($trait)::+>::Output: crate::dtype::Dtyped,
            Ax: AxesArg,
        {
            $Op::new_array(self, axis $($(, $extra_arg)*)?).unwrap()
        }
    };
    ($method:ident : $Op:ident, $in_type:ty => $out_type:ty $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<Ax>(self, axis: Ax $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Op<S, Ax::ReducedDimension<S::Dimension>>>
        where
            S: crate::storage::ArrayStorageTyped<Item = $in_type>,
            Ax: AxesArg,
        {
            $Op::new_array(self, axis $($(, $extra_arg)*)?).unwrap()
        }
    };
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(max: Max, crate::scalar::ReduceMax);
    define_array_reduction_method!(min: Min, crate::scalar::ReduceMin);
    define_array_reduction_method!(argmax: ArgMax, crate::scalar::ArgMax, single_axis = true);
    define_array_reduction_method!(argmin: ArgMin, crate::scalar::ArgMin, single_axis = true);
    define_array_reduction_method!(sum: Sum, crate::scalar::ReduceSum);
    define_array_reduction_method!(product: Product, crate::scalar::ReduceProduct);
    define_array_reduction_method!(mean: Mean, crate::scalar::ReduceMean);
    define_array_reduction_method!(var: Variance, crate::scalar::ReduceVariance, extra_args = (ddof: f64));
    define_array_reduction_method!(std: StandardDeviation, crate::scalar::ReduceVariance, extra_args = (ddof: f64));
    define_array_reduction_method!(all: All, bool => bool);
    define_array_reduction_method!(any: Any, bool => bool);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::rc::Rc;

    use ndarray::{array, ArrayD};

    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use crate::array::Array;
    use crate::storage::Compact;
    use crate::DimDyn;
    use crate::Ty;

    use proptest::prelude::*;

    pub(crate) fn axis_strategy(ndim: usize) -> impl proptest::strategy::Strategy<Value = usize> {
        0..ndim
    }

    pub(crate) fn axes_strategy(ndim: usize) -> proptest::strategy::BoxedStrategy<Vec<usize>> {
        if ndim == 0 {
            return proptest::strategy::Just(vec![]).boxed();
        }
        let axis_strategy = axis_strategy(ndim).prop_map(|axis| vec![axis]);
        let multi_axes_strategy = prop::collection::vec(0..ndim, 1..=ndim).prop_map(|mut axes| {
            axes.sort_unstable();
            axes.dedup();
            axes
        });
        prop::strategy::Union::new_weighted(vec![
            (3, axis_strategy.boxed()),
            (1, multi_axes_strategy.boxed()),
        ])
        .boxed()
    }

    fn reduction_shape_strategy() -> impl proptest::strategy::Strategy<Value = Vec<usize>> {
        prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=100, 1)),
            (2, proptest::collection::vec(100..=1000, 1)),
            // 2D
            (8, proptest::collection::vec(1..=16, 2)),
            (2, proptest::collection::vec(16..=37, 2)),
            // 3D
            (5, proptest::collection::vec(1..=12, 3)),
            // 4D
            (5, proptest::collection::vec(1..=8, 4)),
            // Many dims
            (3, proptest::collection::vec(1..=4, 1..=8)),
        ])
    }

    pub(crate) fn carray_strategy_for_reduction<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<
        Value = (ArrayD<T>, Rc<Array<Compact<Ty<T>, DimDyn>>>, Vec<usize>),
    > {
        let shape = reduction_shape_strategy();
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axes = axes_strategy(nd.ndim());
                (Just(nd), Just(za), axes)
            })
    }

    // pub(crate) fn carray_strategy_for_reduction_single_axis<T: crate::util::ScalarStrategy>(
    //     elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    // ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<Array<Compact<Ty<T>, DimDyn>>>, usize)>
    // {
    //     let shape = reduction_shape_strategy();
    //     let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
    //     array
    //         .prop_map(|(nd, za)| (nd, Rc::new(za)))
    //         .prop_flat_map(|(nd, za)| {
    //             let axis = axis_strategy(nd.ndim());
    //             (Just(nd), Just(za), axis)
    //         })
    // }

    pub(crate) fn carray_strategy_for_reduction_small<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<
        Value = (ArrayD<T>, Rc<Array<Compact<Ty<T>, DimDyn>>>, Vec<usize>),
    > {
        let shape = prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=4, 1)),
            // 2D
            (8, proptest::collection::vec(1..=2, 2)),
        ]);
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axes = axes_strategy(nd.ndim());
                (Just(nd), Just(za), axes)
            })
    }

    macro_rules! test_reduction_dtype {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axes) in crate::ops::reduction::tests::carray_strategy_for_reduction::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(&axes);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &axes,
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };

        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident,
            small_data = true
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axes) in crate::ops::reduction::tests::carray_strategy_for_reduction_small::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(&axes);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &axes,
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };

        (
            $op_method:ident,
            single_axis = true,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axis) in crate::ops::reduction::tests::carray_strategy_for_reduction_single_axis::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(axis);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &[axis],
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };

    }

    macro_rules! test_reduction {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };

        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*,
            small_data = true
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy,
                small_data = true
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy,
                    small_data = true
                );
            )+)*
        };

        (
            $op_method:ident,
            single_axis = true,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                single_axis = true,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    single_axis = true,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };
    }

    #[allow(unused_imports)]
    pub(crate) use {test_reduction, test_reduction_dtype};

    test_reduction!(
        max,
        |items| { items.reduce(|m, x| if x > m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_reduction!(
        min,
        |items| { items.reduce(|m, x| if x < m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    // test_reduction!( // TODO
    //     argmax,
    //     single_axis = true,
    //     |items| {
    //         items
    //             .enumerate()
    //             .reduce(|(m_i, m), (i, x)| if x > m { (i, x) } else { (m_i, m) })
    //             .unwrap()
    //             .0 as u64
    //     },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     any_strategy,
    //     #[cfg(feature = "half")]
    //     [f16]
    // );
    // test_reduction!(
    //     argmin,
    //     single_axis = true,
    //     |items| {
    //         items
    //             .enumerate()
    //             .reduce(|(m_i, m), (i, x)| if x < m { (i, x) } else { (m_i, m) })
    //             .unwrap()
    //             .0 as u64
    //     },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     any_strategy,
    //     #[cfg(feature = "half")]
    //     [f16]
    // );
    test_reduction!(
        sum,
        |items| { items.fold(0u64, |m, x| m + x as u64) },
        [u8, u16, u32, u64, bool],
        op_safe_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0i64, |m, x| m + x as i64) },
        [i8, i16, i32, i64],
        op_safe_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0.0f64, |m, x| m + <_ as crate::scalar::Cast<f64>>::cast(x)) },
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        sum,
        |items| {
            items.fold(Complex::<f64>::default(), |m, x| {
                m + <_ as crate::scalar::Cast<Complex<f64>>>::cast(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    test_reduction!(
        product,
        |items| { items.fold(1u64, |m, x| m * x as u64) },
        [u8, u16, u32, u64],
        op_safe_strategy,
        small_data = true
    );
    test_reduction!(
        product,
        |items| { items.fold(1i64, |m, x| m * x as i64) },
        [i8, i16, i32, i64],
        op_safe_strategy,
        small_data = true
    );
    test_reduction!(
        product,
        |items| { items.fold(1.0f64, |m, x| m * <_ as crate::scalar::Cast<f64>>::cast(x)) },
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        small_data = true
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        product,
        |items| {
            items.fold(Complex::<f64>::new(1.0, 0.0), |m, x| {
                m * <_ as crate::scalar::Cast<Complex<f64>>>::cast(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy,
        small_data = true
    );
    // mean
    test_reduction!(
        mean,
        |items| {
            {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<f64>>::cast(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        mean,
        |items| {
            {
                let mut sum: Complex<f64> = Complex::default();
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<Complex<f64>>>::cast(x);
                    count += 1;
                }
                sum / Complex::new(count as f64, 0.0)
            }
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    #[test]
    fn variance() {
        let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let var_all = a.as_ref().var((0, 1), 0.0).to_ndarray().unwrap();
        assert!((var_all[[]] - 2.9166).abs() < 0.001);
        let var_col = a.as_ref().var(0, 0.0).to_ndarray().unwrap();
        assert!((var_col[[0]] - 2.25).abs() < 0.001);
        let var_row = a.as_ref().var(1, 0.0).to_ndarray().unwrap();
        assert!((var_row[[0]] - 0.6666).abs() < 0.001);
    }
    #[test]
    fn std() {
        let a = Array::compact_array(&array![[7i32, 8, 9], [4, 5, 6]]).unwrap();
        let std_all = a.as_ref().std((0, 1), 0.0).to_ndarray().unwrap();
        assert!((std_all[[]] - 1.7078).abs() < 0.001);
        let std_col = a.as_ref().std(0, 0.0).to_ndarray().unwrap();
        assert!((std_col[[0]] - 1.5).abs() < 0.001);
        let std_row = a.as_ref().std(1, 0.0).to_ndarray().unwrap();
        assert!((std_row[[0]] - 0.8164).abs() < 0.001);
    }
    // test_reduction!(
    //     all,
    //     |items| { items.fold(true, |m, x| m && <_ as crate::scalar::Cast<bool>>::cast(x)) },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     logical_op_strategy,
    //     #[cfg(feature = "half")]
    //     [f16],
    //     #[cfg(feature = "num-complex")]
    //     [complex_f32, complex_f64]
    // );
    // test_reduction!(
    //     any,
    //     |items| { items.fold(false, |m, x| m || <_ as crate::scalar::Cast<bool>>::cast(x)) },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
    //     logical_op_strategy,
    //     #[cfg(feature = "half")]
    //     [f16],
    //     #[cfg(feature = "num-complex")]
    //     [complex_f32, complex_f64]
    // );

    fn ndarray_reduce<'a, S, D, O>(
        array: &'a ndarray::ArrayBase<S, D>,
        axes: &[usize],
        f: impl Fn(&ndarray::ArrayViewD<'a, S::Elem>) -> O,
    ) -> ndarray::ArrayD<O>
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        // Output shape = original with reduction axes removed
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();

        let out_shape: Vec<usize> = array
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &s)| s)
            .collect();

        let values: Vec<O> = ndarray_reduction_iter(array, &axes)
            .map(|(_, view)| f(&view))
            .collect();

        ndarray::ArrayD::from_shape_vec(out_shape, values).unwrap()
    }

    /// Iterates over all index combinations of the **kept** axes (i.e. axes NOT in `axes`),
    /// yielding for each combination the multi-index into the kept axes and a view spanning
    /// the reduction axes.
    fn ndarray_reduction_iter<'a, S, D>(
        array: &'a ndarray::ArrayBase<S, D>,
        axes: &[usize],
    ) -> impl Iterator<Item = (Vec<usize>, ndarray::ArrayViewD<'a, S::Elem>)> + 'a
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();

        // Kept axes = all axes not being reduced
        let ndim = array.ndim();
        let kept_axes: Vec<usize> = (0..ndim).filter(|i| !axes.contains(i)).collect();

        // Shape of the kept axes - this is what we iterate over
        let kept_shape: Vec<usize> = kept_axes.iter().map(|&ax| array.shape()[ax]).collect();
        let total: usize = kept_shape.iter().product();

        (0..total).map(move |flat_idx| {
            // Convert flat index to multi-index over the kept axes
            let mut remaining = flat_idx;
            let mut kept_indices: Vec<usize> = Vec::with_capacity(kept_axes.len());
            for &dim_size in kept_shape.iter().rev() {
                kept_indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            kept_indices.reverse();

            // Fix each kept axis to its index, remove in descending order.
            // We remove kept axes (which are the non-reduction axes), leaving
            // a view over the reduction axes.
            let mut view = array.view().into_dyn();

            // We must track axis offset: as we remove axes, remaining axis
            // indices shift down. Process kept axes in descending order.
            let mut pairs: Vec<(usize, usize)> = kept_axes
                .iter()
                .copied()
                .zip(kept_indices.iter().copied())
                .collect();
            pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));

            for (ax, idx) in &pairs {
                view = view.index_axis_move(ndarray::Axis(*ax), *idx);
            }

            (kept_indices, view)
        })
    }

    mod ndarray_reduce_tests {
        use super::{ndarray_reduce, ndarray_reduction_iter};

        #[cfg(test)]
        mod tests {
            use super::*;
            use ndarray::{array, Array};

            #[test]
            fn single_axis_0() {
                // Shape [2, 3], reduce axis 0 -> 3 views of shape [2]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0]).collect();

                assert_eq!(views.len(), 3);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[2]);
                }

                // kept axis is 1, so indices are [0], [1], [2]
                // view[0] = a[:, 0] = [1, 4]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(views[0].1, array![1, 4].into_dyn());
                // view[1] = a[:, 1] = [2, 5]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(views[1].1, array![2, 5].into_dyn());
                // view[2] = a[:, 2] = [3, 6]
                assert_eq!(views[2].0, vec![2]);
                assert_eq!(views[2].1, array![3, 6].into_dyn());
            }

            #[test]
            fn single_axis_1() {
                // Shape [2, 3], reduce axis 1 -> 2 views of shape [3]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[1]).collect();

                assert_eq!(views.len(), 2);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[3]);
                }

                // kept axis is 0, so indices are [0], [1]
                // view[0] = a[0, :] = [1, 2, 3]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(views[0].1, array![1, 2, 3].into_dyn());
                // view[1] = a[1, :] = [4, 5, 6]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(views[1].1, array![4, 5, 6].into_dyn());
            }

            #[test]
            fn multi_axis_3d() {
                // Shape [2, 3, 4], reduce axes [0, 2] -> 3 views of shape [2, 4]
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0, 2]).collect();

                assert_eq!(views.len(), 3);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[2, 4]);
                }

                // kept axis is 1, indices are [0], [1], [2]
                // view[0] = a[:, 0, :] = [[0,1,2,3],[12,13,14,15]]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(
                    views[0].1,
                    array![[0, 1, 2, 3], [12, 13, 14, 15]].into_dyn()
                );
                // view[1] = a[:, 1, :] = [[4,5,6,7],[16,17,18,19]]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(
                    views[1].1,
                    array![[4, 5, 6, 7], [16, 17, 18, 19]].into_dyn()
                );
                // view[2] = a[:, 2, :] = [[8,9,10,11],[20,21,22,23]]
                assert_eq!(views[2].0, vec![2]);
                assert_eq!(
                    views[2].1,
                    array![[8, 9, 10, 11], [20, 21, 22, 23]].into_dyn()
                );
            }

            #[test]
            fn reduce_all_axes() {
                // Shape [2, 3], reduce both -> 1 view of shape [2, 3] (no kept axes)
                let a = Array::from_shape_vec(vec![2, 3], vec![10, 20, 30, 40, 50, 60]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0, 1]).collect();

                assert_eq!(views.len(), 1);
                assert_eq!(views[0].0, Vec::<usize>::new());
                assert_eq!(views[0].1, array![[10, 20, 30], [40, 50, 60]].into_dyn());
            }

            #[test]
            fn no_axes_returns_scalar_views() {
                // Reduce no axes -> 6 scalar views (iterate over everything)
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[]).collect();

                assert_eq!(views.len(), 6);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[] as &[usize]);
                }

                assert_eq!(views[0].0, vec![0, 0]);
                assert_eq!(*views[0].1.first().unwrap(), 1);
                assert_eq!(views[1].0, vec![0, 1]);
                assert_eq!(*views[1].1.first().unwrap(), 2);
                assert_eq!(views[5].0, vec![1, 2]);
                assert_eq!(*views[5].1.first().unwrap(), 6);
            }

            #[test]
            fn axes_order_independent() {
                // [0, 2] and [2, 0] should yield identical results
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();

                let v1: Vec<_> = ndarray_reduction_iter(&a, &[0, 2]).collect();
                let v2: Vec<_> = ndarray_reduction_iter(&a, &[2, 0]).collect();

                assert_eq!(v1.len(), v2.len());
                for ((idx1, view1), (idx2, view2)) in v1.iter().zip(v2.iter()) {
                    assert_eq!(idx1, idx2);
                    assert_eq!(view1, view2);
                }
            }

            #[test]
            fn dim_1_reduce_axis_0() {
                // Shape [5], reduce axis 0 -> 1 view of shape [5] (no kept axes)
                let a = Array::from_shape_vec(vec![5], vec![10, 20, 30, 40, 50]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0]).collect();

                assert_eq!(views.len(), 1);
                assert_eq!(views[0].0, Vec::<usize>::new());
                assert_eq!(views[0].1, array![10, 20, 30, 40, 50].into_dyn());
            }

            #[test]
            fn reduce_middle_axis() {
                // Shape [2, 3, 4], reduce axis 1 -> 2*4=8 views of shape [3]
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[1]).collect();

                assert_eq!(views.len(), 8);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[3]);
                }

                // kept axes are [0, 2]
                // view[0]: kept=[0,0] -> a[0, :, 0] = [0, 4, 8]
                assert_eq!(views[0].0, vec![0, 0]);
                assert_eq!(views[0].1, array![0, 4, 8].into_dyn());
                // view[3]: kept=[0,3] -> a[0, :, 3] = [3, 7, 11]
                assert_eq!(views[3].0, vec![0, 3]);
                assert_eq!(views[3].1, array![3, 7, 11].into_dyn());
                // view[4]: kept=[1,0] -> a[1, :, 0] = [12, 16, 20]
                assert_eq!(views[4].0, vec![1, 0]);
                assert_eq!(views[4].1, array![12, 16, 20].into_dyn());
                // view[7]: kept=[1,3] -> a[1, :, 3] = [15, 19, 23]
                assert_eq!(views[7].0, vec![1, 3]);
                assert_eq!(views[7].1, array![15, 19, 23].into_dyn());
            }

            // --- ndarray_reduce tests ---

            #[test]
            fn reduce_sum_axis_0() {
                // np.sum(a, axis=0) for shape [2, 3]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[0], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[3]);
                assert_eq!(result, array![5, 7, 9].into_dyn());
            }

            #[test]
            fn reduce_sum_axis_1() {
                // np.sum(a, axis=1) for shape [2, 3]
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[1], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[2]);
                assert_eq!(result, array![6, 15].into_dyn());
            }

            #[test]
            fn reduce_sum_multi_axis() {
                // np.sum(a, axis=(0, 2)) for shape [2, 3, 4]
                let a = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
                let result = ndarray_reduce(&a, &[0, 2], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[3]);
                // axis 1 index 0: sum of a[:, 0, :] = sum(0..4) + sum(12..16) = 6 + 54 = 60
                // axis 1 index 1: sum of a[:, 1, :] = sum(4..8) + sum(16..20) = 22 + 70 = 92
                // axis 1 index 2: sum of a[:, 2, :] = sum(8..12) + sum(20..24) = 38 + 86 = 124
                assert_eq!(result, array![60, 92, 124].into_dyn());
            }

            #[test]
            fn reduce_all_axes_to_scalar() {
                // np.sum(a) - reduce everything
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[0, 1], |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[] as &[usize]);
                assert_eq!(*result.first().unwrap(), 21);
            }

            #[test]
            fn reduce_no_axes_identity() {
                // Reducing no axes -> same shape, each element passed through f
                let a = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[], |v| *v.first().unwrap());

                assert_eq!(result.shape(), &[2, 3]);
                assert_eq!(result, array![[1, 2, 3], [4, 5, 6]].into_dyn());
            }

            #[test]
            fn reduce_max_axis() {
                // np.max(a, axis=0)
                let a = Array::from_shape_vec(vec![3, 2], vec![5, 1, 3, 8, 7, 2]).unwrap();
                let result = ndarray_reduce(&a, &[0], |v| *v.iter().max().unwrap());

                assert_eq!(result.shape(), &[2]);
                assert_eq!(result, array![7, 8].into_dyn());
            }
        }
    }
}
