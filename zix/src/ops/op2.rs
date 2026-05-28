use std::mem::MaybeUninit;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::ops::common::{bulk_size2, define_array_op2_method};
use crate::storage::{ArrayStorageSpec, ArrayStorageTyped};
use crate::util::assert_unchecked_eq;
use crate::{Array, ArrayStorage, Ty};

pub(crate) struct Op2<S1, S2, K> {
    a: Array<S1>,
    b: Array<S2>,
    out_dtype_: Dtype,
    kernel: K,
}
pub(crate) trait Op2Kernel<T1, T2> {
    type Output;
    fn apply(&self, a: T1, b: T2) -> Self::Output;
}
impl<S1, S2, K> Op2<S1, S2, K> {
    pub(crate) fn new(a: Array<S1>, b: Array<S2>, kernel: K) -> Result<Self>
    where
        S1: ArrayStorageTyped,
        S2: ArrayStorageTyped,
        K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
    {
        ensure!(
            a.shape() == b.shape(),
            InvalidArgument,
            "Op2 shape mismatch between `a` {:?} and `b` {:?}",
            a.shape(),
            b.shape()
        );
        Ok(Self {
            a,
            b,
            out_dtype_: K::Output::DTYPE,
            kernel,
        })
    }
}

impl<S1, S2, K> ArrayStorage for Op2<S1, S2, K>
where
    S1: ArrayStorageTyped,
    S2: ArrayStorageTyped,
    K: Op2Kernel<S1::Item, S2::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;
    type Dimension = S1::Dimension;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        unsafe {
            op2_read_data_unchecked::<S1::Item, S2::Item, K::Output>(
                self.a.shape(),
                &self.a.storage,
                &self.b.storage,
                &self.kernel,
                index,
                buf,
                context,
            )
        }
    }

    fn shape(&self) -> &[u64] {
        let shape = self.a.shape();
        unsafe { assert_unchecked_eq!(shape, self.b.shape()) };
        shape
    }

    fn dtype(&self) -> &Dtype {
        let dtype = &self.out_dtype_;
        unsafe { assert_unchecked_eq!(*dtype, K::Output::DTYPE) };
        dtype
    }

    fn _spec(&self) -> ArrayStorageSpec<'_> {
        self.a.storage._spec()
    }
}

pub(crate) unsafe fn op2_read_data_unchecked<T1, T2, Out>(
    shape: &[u64],
    a: &impl ArrayStorage,
    b: &impl ArrayStorage,
    kernel: &impl Op2Kernel<T1, T2, Output = Out>,
    index: &[Range<u64>],
    buf: &mut [u8],
    context: &ReadContext,
) -> Result<()>
where
    T1: Dtyped,
    T2: Dtyped,
    Out: Dtyped,
{
    let bulk_size = bulk_size2::<T1, T2>();
    assert!(bulk_size.is_power_of_two());

    // this is a compile time check, the compiler knows the value of `bulk_size2::<T1, T2>()`
    let read_fn = match bulk_size {
        1 => op2_read_data_unchecked_impl::<T1, T2, Out, 1>,
        2 => op2_read_data_unchecked_impl::<T1, T2, Out, 2>,
        4 => op2_read_data_unchecked_impl::<T1, T2, Out, 4>,
        8 => op2_read_data_unchecked_impl::<T1, T2, Out, 8>,
        16 => op2_read_data_unchecked_impl::<T1, T2, Out, 16>,
        32 => op2_read_data_unchecked_impl::<T1, T2, Out, 32>,
        64 => op2_read_data_unchecked_impl::<T1, T2, Out, 64>,
        128 => op2_read_data_unchecked_impl::<T1, T2, Out, 128>,
        256 => op2_read_data_unchecked_impl::<T1, T2, Out, 256>,
        512 => op2_read_data_unchecked_impl::<T1, T2, Out, 512>,
        _ => op2_read_data_unchecked_impl::<T1, T2, Out, 1024>,
    };
    unsafe { read_fn(shape, a, b, kernel, index, buf, context) }
}
unsafe fn op2_read_data_unchecked_impl<T1, T2, Out, const BULK: usize>(
    shape: &[u64],
    a: &impl ArrayStorage,
    b: &impl ArrayStorage,
    kernel: &impl Op2Kernel<T1, T2, Output = Out>,
    index: &[Range<u64>],
    buf: &mut [u8],
    context: &ReadContext,
) -> Result<()>
where
    T1: Dtyped,
    T2: Dtyped,
    Out: Dtyped,
{
    check_get_range(shape, index)?;
    let (a_dtype, b_dtype, out_dtype) = (T1::DTYPE, T2::DTYPE, Out::DTYPE);
    let nitems = check_get_buffer_size(index, &out_dtype, buf)?;

    let (src_a_itemsize, src_b_itemsize, dst_itemsize) = (
        a_dtype.itemsize() as usize,
        b_dtype.itemsize() as usize,
        out_dtype.itemsize() as usize,
    );

    let a_in_place = src_a_itemsize == dst_itemsize
        && (buf.as_ptr() as usize).is_multiple_of(a_dtype.alignment().as_usize());
    let b_in_place = src_b_itemsize == dst_itemsize
        && (buf.as_ptr() as usize).is_multiple_of(b_dtype.alignment().as_usize());
    let mut a_tmp_buf;
    let mut b_tmp_buf;
    let (a_buf, b_buf, dst) = if a_in_place {
        b_tmp_buf = context.tmp_buf(nitems * src_b_itemsize, b_dtype.alignment());
        let b_tmp_buf = b_tmp_buf.as_mut_slice();
        let ptr = buf.as_mut_ptr();
        (buf, b_tmp_buf, ptr)
    } else if b_in_place {
        a_tmp_buf = context.tmp_buf(nitems * src_a_itemsize, a_dtype.alignment());
        let a_tmp_buf = a_tmp_buf.as_mut_slice();
        let dst = buf.as_mut_ptr();
        (a_tmp_buf, buf, dst)
    } else {
        a_tmp_buf = context.tmp_buf(nitems * src_a_itemsize, a_dtype.alignment());
        let a_tmp_buf = a_tmp_buf.as_mut_slice();

        b_tmp_buf = context.tmp_buf(nitems * src_b_itemsize, b_dtype.alignment());
        let b_tmp_buf = b_tmp_buf.as_mut_slice();

        (a_tmp_buf, b_tmp_buf, buf.as_mut_ptr())
    };

    unsafe { assert_unchecked_eq!(a_dtype, *a.dtype()) };
    unsafe { assert_unchecked_eq!(shape, a.shape()) };
    a.read_data(index, a_buf, context)?;
    unsafe { assert_unchecked_eq!(b_dtype, *b.dtype()) };
    unsafe { assert_unchecked_eq!(shape, b.shape()) };
    b.read_data(index, b_buf, context)?;

    let mut src_a_data = a_buf.as_ptr().cast::<T1>();
    let mut src_b_data = b_buf.as_ptr().cast::<T2>();
    let mut dst_data = dst.cast::<Out>();
    let mut nitems = nitems;
    unsafe {
        while nitems >= BULK {
            let src_a = src_a_data.cast::<[T1; BULK]>().read();
            let src_b = src_b_data.cast::<[T2; BULK]>().read();
            let dst = &mut *dst_data.cast::<[MaybeUninit<Out>; BULK]>();
            for i in 0..BULK {
                dst[i].write(kernel.apply(src_a[i], src_b[i]));
            }
            nitems -= BULK;
            src_a_data = src_a_data.add(BULK);
            src_b_data = src_b_data.add(BULK);
            dst_data = dst_data.add(BULK);
        }
        while nitems > 0 {
            let src_a = src_a_data.read();
            let src_b = src_b_data.read();
            dst_data.write(kernel.apply(src_a, src_b));
            nitems -= 1;
            src_a_data = src_a_data.add(1);
            src_b_data = src_b_data.add(1);
            dst_data = dst_data.add(1);
        }
    }
    Ok(())
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
        impl<S1, S2> $Op<S1, S2> {
            pub fn new(a: Array<S1>, b: Array<S2>) -> crate::error::Result<Self>
            where
                S1: crate::storage::ArrayStorageTyped,
                S2: crate::storage::ArrayStorageTyped,
                S1::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
            {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel)?))
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S1::Item as $($trait)::+<S2::Item>>::Output>;
            type Dimension = S1::Dimension;
            crate::storage::impl_array_storage_forward!();
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
        impl<S1, S2> $Op<S1, S2> {
            pub fn new(a: Array<S1>, b: Array<S2>) -> crate::error::Result<Self>
            where
                S1: crate::storage::ArrayStorageTyped,
                S2: crate::storage::ArrayStorageTyped,
                S1::Item: $($trait)::+<S2::Item>,
            {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel)?))
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: $($trait)::+<S2::Item>,
        {
            type ElementType = crate::Ty<$output_type>;
            type Dimension = S1::Dimension;
            crate::storage::impl_array_storage_forward!();
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
            S2: crate::storage::ArrayStorageTyped,
            S1::Item: core::ops::$core_op_trait<S2::Item, Output: crate::dtype::Dtyped>,
        {
            type Output = Array<$Op<S1, S2>>;
            #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_op_fn(self, b: Array<S2>) -> Self::Output {
                let op = $Op::new(self, b).unwrap();
                Array::from_storage(op)
            }
        }

        impl<S1, T2> core::ops::$core_op_trait<T2> for Array<S1>
        where
            S1: crate::storage::ArrayStorageTyped,
            T2: crate::dtype::Dtyped,
            S1::Item: core::ops::$core_op_trait<T2, Output: crate::dtype::Dtyped>,
        {
            type Output = Array<$Op<S1, crate::storage::Scalar<T2, S1::Dimension>>>;
            #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_op_fn(self, b: T2) -> Self::Output {
                let shape = <S1::Dimension as crate::Dimension>::from_slice(self.shape()).unwrap();
                let b = Array::plain_scalar(b, shape).unwrap();
                let op = $Op::new(self, b).unwrap();
                Array::from_storage(op)
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
            fn apply(&self, $a: T1, $b: $rhs) -> Self::Output {
                <T1 as $($trait)::+>::$kernel_fn($a, $b)
            }
        }
        $(#[$meta])*
        pub struct $Op<S1, S2>(crate::ops::op2::Op2<S1, S2, $Kernel>);
        impl<S1, S2> $Op<S1, S2> {
            pub fn new(a: Array<S1>, b: Array<S2>) -> crate::error::Result<Self>
            where
                S1: crate::storage::ArrayStorageTyped,
                S2: crate::storage::ArrayStorageTyped<Item = $rhs>,
                S1::Item: $($trait)::+
            {
                Ok(Self(crate::ops::op2::Op2::new(a, b, $Kernel)?))
            }
        }
        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Item = $rhs>,
            S1::Item: $($trait)::+
        {
            type ElementType = crate::Ty<$output_type_s>;
            type Dimension = S1::Dimension;
            crate::storage::impl_array_storage_forward!();
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
    /// Available via the `+` operator on arrays. A raw scalar can be used as one of the operands
    /// and it will be broadcasted to the array's shape: `arr + 1i32`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1i32, 2, 3])?;
    /// let b = Array::compact_array(&array![10i32, 20, 30])?;
    /// let result = (a + b).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[11, 22, 33]);
    ///
    /// let a = Array::compact_array(&array![1i32, 2, 3])?;
    /// let result = (a + 10i32).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[11, 12, 13]);
    /// # Ok::<(), zix::Error>(())
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
    /// Available via the `-` operator on arrays. A raw scalar can be used as the right-hand
    /// side and is broadcast to the array's shape: `arr - 1i32`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![10i32, 20, 30])?;
    /// let b = Array::compact_array(&array![1i32, 2, 3])?;
    /// let result = (a - b).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[9, 18, 27]);
    ///
    /// let a = Array::compact_array(&array![10i32, 20, 30])?;
    /// let result = (a - 5i32).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[5, 15, 25]);
    /// # Ok::<(), zix::Error>(())
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
    /// Available via the `*` operator on arrays. A raw scalar can be used as the right-hand
    /// side and is broadcast to the array's shape: `arr * 2i32`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1i32, 2, 3])?;
    /// let b = Array::compact_array(&array![4i32, 5, 6])?;
    /// let result = (a * b).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 10, 18]);
    ///
    /// let a = Array::compact_array(&array![1i32, 2, 3])?;
    /// let result = (a * 3i32).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[3, 6, 9]);
    /// # Ok::<(), zix::Error>(())
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
    /// Available via the `/` operator on arrays. A raw scalar can be used as the right-hand
    /// side and is broadcast to the array's shape: `arr / 2i32`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![10i32, 20, 30])?;
    /// let b = Array::compact_array(&array![2i32, 4, 5])?;
    /// let result = (a / b).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[5, 5, 6]);
    ///
    /// let a = Array::compact_array(&array![10i32, 20, 30])?;
    /// let result = (a / 10i32).to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1, 2, 3]);
    /// # Ok::<(), zix::Error>(())
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
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![2.0f32, 3.0, 4.0])?;
    /// let b = Array::compact_array(&array![3.0f32, 2.0, 0.5])?;
    /// let result = a.pow(b).to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[8.0, 9.0, 2.0]);
    ///
    /// // Raise each element to a scalar exponent.
    /// let a = Array::compact_array(&array![2.0f32, 3.0, 4.0])?;
    /// let exp = Array::plain_scalar(2.0f32, &[3])?;
    /// let result = a.pow(exp).to_ndarray::<f32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4.0, 9.0, 16.0]);
    /// # Ok::<(), zix::Error>(())
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

    macro_rules! test_op2_dtype {
        ($op_method:ident, |$a:ident, $b:ident| $body:expr, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        ((nd_a, za), (nd_b, zb)) in crate::util::carrays2_strategy_generic::<$dtype>(
                            crate::util::shape_strategy(),
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        #[allow(unused_imports)] use core::ops::{Add, Sub, Mul, Div};

                        let result = za.$op_method(zb);
                        let expected = ndarray::Zip::from(&nd_a).and(&nd_b).map_collect(|& $a, & $b| $body);
                        crate::util::assert_array_matches(&result, &expected);
                    }
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
            let za = crate::Array::compact_array(&nd_a).unwrap();
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
            let nd_a = ndarray::ArrayD::from_shape_vec(vec![2, 3], a_vals).unwrap();
            let nd_b = ndarray::ArrayD::from_shape_vec(vec![2, 3], b_vals).unwrap();
            let nd_c = ndarray::ArrayD::from_shape_vec(vec![2, 3], c_vals).unwrap();
            let za = Array::compact_array_with(&nd_a, crate::util::arr_params(&[2, 3])).unwrap();
            let zb = Array::compact_array_with(&nd_b, crate::util::arr_params(&[2, 3])).unwrap();
            let zc = Array::compact_array_with(&nd_c, crate::util::arr_params(&[1, 2])).unwrap();
            let result = (za + zb) * zc;
            let expected = (&nd_a + &nd_b) * &nd_c;
            crate::util::assert_array_matches(&result, &expected);
        }
    }

    proptest::proptest! {
        #[test]
        fn add_mul_scalar(
            vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 100usize)
        ) {
            use crate::array::Array;
            let a = ndarray::ArrayD::from_shape_vec(vec![10, 10], vals).unwrap();
            let za = Array::compact_array_with(&a, crate::util::arr_params(&[10, 10])).unwrap();
            let zb = za * 2.0f32 + 1.0f32;
            let actual = zb.to_ndarray::<f32>().unwrap();
            let expected = &a * 2.0 + 1.0;
            proptest::prop_assert_eq!(actual, expected);
        }
    }
}
