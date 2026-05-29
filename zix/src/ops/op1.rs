use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};
use crate::ops::common::define_array_op1_method;
use crate::storage::{ArrayStorageSpec, ArrayStorageTyped, ReadData, ReadDataExt};
use crate::util::assert_unchecked_eq;
use crate::{ArrayStorage, Ty};

pub(crate) struct Op1<S, K> {
    array: Array<S>,
    out_dtype_: Dtype,
    kernel: K,
}
pub(crate) trait Op1Kernel<T> {
    type Output;
    fn apply(&self, x: T) -> Self::Output;
}
impl<S, K> Op1<S, K> {
    pub(crate) fn new(array: Array<S>, kernel: K) -> Result<Self>
    where
        S: ArrayStorage + ArrayStorageTyped,
        K: Op1Kernel<S::Item, Output: Dtyped>,
    {
        Ok(Self {
            array,
            out_dtype_: K::Output::DTYPE,
            kernel,
        })
    }
}

impl<S, K> ArrayStorage for Op1<S, K>
where
    S: ArrayStorage + ArrayStorageTyped,
    K: Op1Kernel<S::Item, Output: Dtyped>,
{
    type ElementType = Ty<K::Output>;
    type Dimension = S::Dimension;

    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        self.read_data_typed::<K::Output>(index, context)?
            .to_buf(buf)
    }

    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, S, K>>
    where
        T: Dtyped,
    {
        check_dtype(&T::DTYPE, &K::Output::DTYPE)?;

        let data = self
            .array
            .storage
            .read_data_typed(index, context)?
            .map_items(|x| self.kernel.apply(x));

        // SAFETY: We checked that `T` has the same dtype as `K::Output`
        Ok(unsafe { data.transmute_items::<T>() })
    }

    fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    fn dtype(&self) -> &Dtype {
        let dtype = &self.out_dtype_;
        unsafe { assert_unchecked_eq!(*dtype, K::Output::DTYPE) };
        dtype
    }

    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.array.storage.spec()
    }
}

impl<F, T, O> Op1Kernel<T> for F
where
    F: Fn(T) -> O,
{
    type Output = O;
    fn apply(&self, x: T) -> Self::Output {
        self(x)
    }
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        $(core_op = $core_op_trait:ident::$core_op_fn:ident,)?
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: $($trait)::+,
        {
            type Output = <T as $($trait)::+>::Output;
            fn apply(&self, x: T) -> Self::Output {
                <T as $($trait)::+>::$kernel_fn(x)
            }
        }
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::op1::Op1<S, $Kernel>);
        impl<S> $Op<S> {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: Array<S>) -> crate::error::Result<Self>
            where
                S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
                S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
            {
                Ok(Self(crate::ops::op1::Op1::new(array, $Kernel)?))
            }
        }
        impl<S> ArrayStorage for $Op<S>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            type ElementType = crate::Ty<<S::Item as $($trait)::+>::Output>;
            type Dimension = S::Dimension;
            crate::storage::impl_array_storage_forward!(<S>);
        }

        define_op1!(@define_core
            impl $Op
            $(core_op = $core_op_trait::$core_op_fn,)?
        );
    };

    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output<T> = T,
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = T,
            type Output<S> = S::Item,
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output = $output_type:ty,
    ) => {
        define_op1!(
            $(#[$meta])*
            $Op,
            $Kernel,
            <$($trait)::+> :: $kernel_fn,
            type Output<T> = $output_type,
            type Output<S> = $output_type,
        );
    };
    (
        $(#[$meta:meta])*
        $Op:ident,
        $Kernel:ident,
        <$($trait:ident)::+> :: $kernel_fn:ident,
        type Output<T> = $output_type_t:ty,
        type Output<S> = $output_type_s:ty,
    ) => {
        struct $Kernel;
        impl<T> crate::ops::op1::Op1Kernel<T> for $Kernel
        where
            T: $($trait)::+,
        {
            type Output = $output_type_t;
            fn apply(&self, x: T) -> Self::Output {
                <T as $($trait)::+>::$kernel_fn(x)
            }
        }
        $(#[$meta])*
        pub struct $Op<S>(crate::ops::op1::Op1<S, $Kernel>);
        impl<S> $Op<S> {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(array: Array<S>) -> crate::error::Result<Self>
            where
                S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
                S::Item: $($trait)::+,
            {
                Ok(Self(crate::ops::op1::Op1::new(array, $Kernel)?))
            }
        }
        impl<S> ArrayStorage for $Op<S>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
        {
            type ElementType = crate::Ty<$output_type_s>;
            type Dimension = S::Dimension;
            crate::storage::impl_array_storage_forward!(<S>);
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
        impl<S> core::ops::$core_op_trait for Array<S>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: core::ops::$core_op_trait<Output: crate::dtype::Dtyped>,
        {
            type Output = Array<$Op<S>>;
            #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
            #[track_caller]
            fn $core_op_fn(self) -> Self::Output {
                let op = $Op::new(self).unwrap();
                Array::from_storage(op)
            }
        }
    };
}

pub(crate) use define_op1;

pub(crate) mod _traits {
    #[allow(unused_imports)]
    use crate::scalar::{f16, Complex};

    use crate::scalar::traits_util::define_op1_trait;

    define_op1_trait!(
        Abs,
        abs,
        |a| a.abs(),
        [i8, i16, i32, i64, f32, f64] => "same"
    );
    #[cfg(feature = "half")]
    impl Abs for f16 {
        type Output = f16;
        fn abs(self) -> Self::Output {
            // Self::from_f32(self.to_f32().abs())
            <Self as num_traits::Float>::abs(self)
        }
    }
    impl Abs for Complex<f32> {
        type Output = f32;
        fn abs(self) -> Self::Output {
            self.re.hypot(self.im)
        }
    }
    impl Abs for Complex<f64> {
        type Output = f64;
        fn abs(self) -> Self::Output {
            self.re.hypot(self.im)
        }
    }
}

define_op1!(
    /// Arithmetic negation applied element-wise.
    ///
    /// For **integer** types the result is the two's-complement negation.
    /// Negating the minimum representable value (e.g. `i32::MIN`) overflows:
    /// it wraps in release builds and panics in debug builds.
    ///
    /// For **complex** types both components are negated independently:
    /// `-(a + bi) = -a - bi`.
    ///
    /// Available via the unary `-` operator on [`Array`](crate::Array): `-arr`.
    /// Floating-point semantics follow `f32::neg`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::neg()`](core::ops::Neg::neg).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.0f32, -2.5, 3.0])?;
    /// let result = (-a).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-1.0, 2.5, -3.0]);
    ///
    /// // Negating i8::MIN wraps in release builds (two's complement overflow).
    /// let b = Array::compact_array(&array![0i8, 1, -1])?;
    /// let result = (-b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[0, -1, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Neg,
    NegKernel,
    <core::ops::Neg>::neg,
    core_op = Neg::neg,
);
define_op1!(
    /// Rounds each element down to the nearest integer (towards -inf).
    ///
    /// Semantics follow [`f32::floor`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::floor()`](crate::Array::floor).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.1f32, 2.9, 3.0])?;
    /// let result = a.floor().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    ///
    /// // Floor rounds towards -inf, so negative values floor down.
    /// let b = Array::compact_array(&array![-1.1f32, -2.9, -3.0])?;
    /// let result = b.floor().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-2.0, -3.0, -3.0]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Floor,
    FloorKernel,
    <num_traits::Float>::floor,
    type Output<T> = T,
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +inf).
    ///
    /// Semantics follow [`f32::ceil`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ceil()`](crate::Array::ceil).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.1f32, 2.0, 3.9])?;
    /// let result = a.ceil().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[2.0, 2.0, 4.0]);
    ///
    /// // Ceil rounds towards +inf, so negative values ceil up.
    /// let b = Array::compact_array(&array![-1.7f32, -2.0, -0.1])?;
    /// let result = b.ceil().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[-1.0, -2.0, 0.0]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Ceil,
    CeilKernel,
    <num_traits::Float>::ceil,
    type Output<T> = T,
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// Ties (values exactly halfway between two integers) are broken by rounding
    /// away from zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from
    /// "round-half-to-even" (banker's rounding) used in some other libraries.
    /// Semantics follow [`f32::round`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::round()`](crate::Array::round).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.4f32, 1.6, 2.0])?;
    /// let result = a.round().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 2.0]);
    ///
    /// // Ties are broken away from zero: 0.5 -> 1.0, -0.5 -> -1.0.
    /// let b = Array::compact_array(&array![0.5f32, -0.5])?;
    /// let result = b.round().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, -1.0]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Round,
    RoundKernel,
    <num_traits::Float>::round,
    type Output<T> = T,
);
define_op1!(
    /// Computes the square root of each element.
    ///
    /// Negative inputs produce `NaN`. Semantics follow [`f32::sqrt`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sqrt()`](crate::Array::sqrt).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![4.0f32, 9.0, 16.0])?;
    /// let result = a.sqrt().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[2.0, 3.0, 4.0]);
    ///
    /// // Negative input produces NaN.
    /// let b = Array::compact_array(&array![-1.0f32])?;
    /// let result = b.sqrt().to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), zix::Error>(())
    /// ```
    Sqrt,
    SqrtKernel,
    <num_traits::Float>::sqrt,
    type Output<T> = T,
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// Semantics follow [`f32::exp`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::exp()`](crate::Array::exp).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.0f32, 2.0, 3.0])?;
    /// let result = a.exp().to_ndarray()?;
    /// assert!((result[[0]] - std::f32::consts::E).abs() < 1e-5);
    ///
    /// // exp(0.0) = 1.0 and exp(1.0) = e.
    /// let b = Array::compact_array(&array![0.0f32, 1.0])?;
    /// let result = b.exp().to_ndarray()?;
    /// assert_eq!(result[[0]], 1.0);
    /// assert!((result[[1]] - std::f32::consts::E).abs() < 1e-5);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Exp,
    ExpKernel,
    <num_traits::Float>::exp,
    type Output<T> = T,
);
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// Negative inputs produce `NaN`; zero produces `-inf`.
    /// Semantics follow [`f32::ln`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ln()`](crate::Array::ln).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E])?;
    /// let result = a.ln().to_ndarray()?;
    /// assert!((result[[0]] - 0.0).abs() < 1e-5);
    /// assert!((result[[1]] - 1.0).abs() < 1e-5);
    ///
    /// // Zero produces -inf; negative input produces NaN.
    /// let b = Array::compact_array(&array![0.0f32, -1.0])?;
    /// let result = b.ln().to_ndarray()?;
    /// assert_eq!(result[[0]], f32::NEG_INFINITY);
    /// assert!(result[[1]].is_nan());
    /// # Ok::<(), zix::Error>(())
    /// ```
    Ln,
    LnKernel,
    <num_traits::Float>::ln,
    type Output<T> = T,
);
define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// Semantics follow [`f32::sin`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sin()`](crate::Array::sin).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![std::f32::consts::FRAC_PI_2, std::f32::consts::PI])?;
    /// let result = a.sin().to_ndarray()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // sin(0.0) = 0.0.
    /// let b = Array::compact_array(&array![0.0f32])?;
    /// let result = b.sin().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Sin,
    SinKernel,
    <num_traits::Float>::sin,
    type Output<T> = T,
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// Semantics follow [`f32::cos`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::cos()`](crate::Array::cos).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0.0f32, std::f32::consts::PI])?;
    /// let result = a.cos().to_ndarray()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    /// assert!((result[[1]] - (-1.0)).abs() < 1e-5);
    ///
    /// // cos(0.0) = 1.0.
    /// let b = Array::compact_array(&array![0.0f32])?;
    /// let result = b.cos().to_ndarray()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Cos,
    CosKernel,
    <num_traits::Float>::cos,
    type Output<T> = T,
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// Semantics follow [`f32::tan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::tan()`](crate::Array::tan).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2 * 0.5])?;
    /// let result = a.tan().to_ndarray()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // tan(0.0) = 0.0.
    /// let b = Array::compact_array(&array![0.0f32])?;
    /// let result = b.tan().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Tan,
    TanKernel,
    <num_traits::Float>::tan,
    type Output<T> = T,
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`. Semantics follow [`f32::asin`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::asin()`](crate::Array::asin).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0.0f32, 1.0, -1.0])?;
    /// let result = a.asin().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // Input outside [-1, 1] produces NaN.
    /// let b = Array::compact_array(&array![2.0f32])?;
    /// let result = b.asin().to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), zix::Error>(())
    /// ```
    Asin,
    AsinKernel,
    <num_traits::Float>::asin,
    type Output<T> = T,
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, pi]`.
    ///
    /// Inputs outside `[-1, 1]` produce `NaN`. Semantics follow [`f32::acos`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::acos()`](crate::Array::acos).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.0f32, 0.0, -1.0])?;
    /// let result = a.acos().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // Input outside [-1, 1] produces NaN.
    /// let b = Array::compact_array(&array![2.0f32])?;
    /// let result = b.acos().to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// # Ok::<(), zix::Error>(())
    /// ```
    Acos,
    AcosKernel,
    <num_traits::Float>::acos,
    type Output<T> = T,
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`.
    ///
    /// Semantics follow [`f32::atan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::atan()`](crate::Array::atan).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![0.0f32, -1.0, 1.0])?;
    /// let result = a.atan().to_ndarray()?;
    /// assert_eq!(result[[0]], 0.0);
    ///
    /// // atan(1.0) = pi/4.
    /// let b = Array::compact_array(&array![1.0f32])?;
    /// let result = b.atan().to_ndarray()?;
    /// assert!((result[[0]] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Atan,
    AtanKernel,
    <num_traits::Float>::atan,
    type Output<T> = T,
);
define_op1!(
    /// Returns the sign of each element as a floating-point value.
    ///
    /// Returns `+1.0` for positive values and `-1.0` for negative values.
    /// Zero is signed: `+0.0` returns `+1.0` and `-0.0` returns `-1.0`.
    /// Semantics follow [`f32::signum`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::signum()`](crate::Array::signum).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![3.0f32, -5.0, -0.1])?;
    /// let result = a.signum().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1.0, -1.0, -1.0]);
    ///
    /// // Positive zero returns +1.0.
    /// let b = Array::compact_array(&array![0.0f32])?;
    /// let result = b.signum().to_ndarray()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Signum,
    SignumKernel,
    <num_traits::Float>::signum,
    type Output<T> = T,
);
define_op1!(
    /// Computes the absolute value of each element.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | same |
    /// | `f16`, `f32`, `f64` | same |
    /// | `Complex<f32>` | `f32` |
    /// | `Complex<f64>` | `f64` |
    ///
    /// For **complex** types the result is the modulus `sqrt(re^2 + im^2)`, computed
    /// via `hypot` for numerical stability. The output dtype is the real component type
    /// (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`).
    ///
    /// For **signed integer** types, `MIN.abs()` overflows: `(-128i8).abs()` wraps back
    /// to `i8::MIN` in release builds and panics in debug builds.
    ///
    /// Floating-point semantics follow [`f32::abs`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::abs()`](crate::Array::abs).
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![-3i32, 0, 5, -7])?;
    /// let result = a.abs().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[3, 0, 5, 7]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    ///
    /// ```
    /// # #[cfg(feature = "num-complex")]
    /// # {
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// // For complex input the result is the modulus sqrt(re^2 + im^2).
    /// use zix::scalar::Complex;
    /// let b = Array::compact_array(&array![Complex { re: 3.0f32, im: 4.0 }])?;
    /// let result = b.abs().to_ndarray()?;
    /// assert!((result[[0]] - 5.0).abs() < 1e-5);
    /// # }
    /// # Ok::<(), zix::Error>(())
    /// ```
    Abs,
    AbsKernel,
    <crate::scalar::Abs>::abs,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(ceil: Ceil, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(round: Round, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(sqrt: Sqrt, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(exp: Exp, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(ln: Ln, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(sin: Sin, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(cos: Cos, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(tan: Tan, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(asin: Asin, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(acos: Acos, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(atan: Atan, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(signum: Signum, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(abs: Abs, crate::scalar::Abs);
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

    macro_rules! test_op1_dtype {
        ($op_method:ident, |$arg:ident| $body:expr, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za) in crate::util::carray_strategy_from_shape::<$dtype>(
                            crate::util::shape_strategy(),
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        #[allow(unused_imports)] use std::ops::Neg;
                        let result = za.$op_method();
                        let expected = nd.mapv(|$arg| $body);
                        crate::util::assert_array_matches(&result, &expected);
                    }
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
            $(crate::ops::op1::tests::test_op1_dtype!($op_method, |$arg| $body, $dtype, $strategy);)+
            $($(
                #[cfg($cfg)]
                crate::ops::op1::tests::test_op1_dtype!($op_method, |$arg| $body, $cfg_dtype, $strategy);
            )+)*
        };
    }

    pub(crate) use {test_op1, test_op1_dtype};

    test_op1!(
        neg,
        |a| -a,
        [i8, i16, i32, i64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_op1!(floor, |a| a.floor(), [f32, f64], op_safe_strategy);
    test_op1!(ceil, |a| a.ceil(), [f32, f64], op_safe_strategy);
    test_op1!(round, |a| a.round(), [f32, f64], op_safe_strategy);
    test_op1!(
        sqrt,
        |a| a.sqrt(),
        [f32, f64],
        op_safe_non_negative_strategy
    );
    test_op1!(exp, |a| a.exp(), [f32, f64], op_safe_strategy);
    test_op1!(ln, |a| a.ln(), [f32, f64], op_safe_non_negative_strategy);
    test_op1!(sin, |a| a.sin(), [f32, f64], op_safe_strategy);
    test_op1!(cos, |a| a.cos(), [f32, f64], op_safe_strategy);
    test_op1!(tan, |a| a.tan(), [f32, f64], op_safe_strategy);
    // asin/acos domain is [-1, 1]: use unit_strategy to avoid NaN comparison failures.
    test_op1!(asin, |a| a.asin(), [f32, f64], unit_strategy);
    test_op1!(acos, |a| a.acos(), [f32, f64], unit_strategy);
    test_op1!(atan, |a| a.atan(), [f32, f64], op_safe_strategy);
    test_op1!(
        signum,
        |a| a.signum(),
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    // abs: same dtype for scalar types; complex types have a different output dtype (see below).
    test_op1!(
        abs,
        |a| a.abs(),
        [i8, i16, i32, i64, f32, f64],
        op_safe_strategy
    );
    // TODO
    // #[cfg(feature = "half")]
    // [f16]

    #[cfg(feature = "num-complex")]
    mod complex {
        use super::{complex_f32, complex_f64};

        // abs on complex types: output dtype is the real component type, not the input dtype.
        // Reference uses hypot to match the Abs kernel exactly.
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f32, op_safe_strategy);
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f64, op_safe_strategy);
    }
}
