use crate::dtype::f16;
use crate::ops::common::define_array_op2_method;
use crate::ops::define_op2;
use crate::storage::ArrayStorage;
use crate::Array;

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, `NaN != NaN` per IEEE 754: comparing two `NaN` values
    /// returns `false`.
    /// For **complex** types, both the real and imaginary components must be equal.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1i32, 2, 3];
    /// let b = ndarray::array![1i32, 0, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.equal(zb).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, true]);
    ///
    /// // NaN != NaN per IEEE 754.
    /// let c = ndarray::array![f32::NAN, 1.0f32];
    /// let d = ndarray::array![f32::NAN, 1.0f32];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.equal(zd).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Equal,
    EqualKernel,
    |a, b| a == b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
    output_type = bool
);
define_op2!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, `NaN != NaN` returns `true` per IEEE 754.
    /// For **complex** types, returns `true` if either the real or imaginary component differs.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1i32, 2, 3];
    /// let b = ndarray::array![1i32, 0, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.not_equal(zb).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
    ///
    /// // NaN != NaN is true per IEEE 754.
    /// let c = ndarray::array![f32::NAN, 1.0f32];
    /// let d = ndarray::array![f32::NAN, 2.0f32];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.not_equal(zd).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    NotEqual,
    NotEqualKernel,
    |a, b| a != b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
    output_type = bool
);
define_op2!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported as they have no
    /// total ordering. Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `true > false`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![3i32, 1, 2];
    /// let b = ndarray::array![1i32, 1, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.greater(zb).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    ///
    /// // true > false for bool dtype.
    /// let c = ndarray::array![true, false, true];
    /// let d = ndarray::array![false, false, true];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.greater(zd).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Greater,
    GreaterKernel,
    |a, b| a > b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
    output_type = bool
);
define_op2!(
    /// Element-wise greater-than-or-equal test (`a >= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported as they have no
    /// total ordering. Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `true >= false`, and both `true >= true` and `false >= false` hold.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![3i32, 1, 2];
    /// let b = ndarray::array![1i32, 1, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.greater_equal(zb).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    ///
    /// // true >= false and false >= false hold.
    /// let c = ndarray::array![true, false, true];
    /// let d = ndarray::array![false, false, true];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.greater_equal(zd).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, true]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    GreaterEqual,
    GreaterEqualKernel,
    |a, b| a >= b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
    output_type = bool
);
define_op2!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported as they have no
    /// total ordering. Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `false < true`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1i32, 1, 3];
    /// let b = ndarray::array![3i32, 1, 2];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.less(zb).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    ///
    /// // false < true for bool dtype.
    /// let c = ndarray::array![false, false, true];
    /// let d = ndarray::array![true, false, true];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.less(zd).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Less,
    LessKernel,
    |a, b| a < b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
    output_type = bool
);
define_op2!(
    /// Element-wise less-than-or-equal test (`a <= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported as they have no
    /// total ordering. Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `false <= true`, and both `false <= false` and `true <= true` hold.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1i32, 1, 3];
    /// let b = ndarray::array![3i32, 1, 2];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.less_equal(zb).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    ///
    /// // false <= true and true <= true hold.
    /// let c = ndarray::array![false, false, true];
    /// let d = ndarray::array![true, false, true];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.less_equal(zd).data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, true]);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    LessEqual,
    LessEqualKernel,
    |a, b| a <= b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
    output_type = bool
);

define_op2!(
    /// Element-wise maximum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype and shape equal the input.
    ///
    /// For **integer** and **bool** types the result is `std::cmp::max(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This deviates from [`f32::max`], which returns the non-`NaN`
    /// operand when exactly one is `NaN`, but matches the behaviour of `numpy.maximum`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1i32, 5, 3];
    /// let b = ndarray::array![4i32, 2, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.maximum(zb).data().to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 5, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = ndarray::array![f32::NAN, 1.0f32];
    /// let d = ndarray::array![2.0f32, 3.0f32];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.maximum(zd).data().to_ndarray::<f32>()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 3.0);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Maximum,
    MaximumKernel,
    |a, b| MaximumTrait::maximum(a, b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
    output_type = "same"
);
define_op2!(
    /// Element-wise minimum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype and shape equal the input.
    ///
    /// For **integer** and **bool** types the result is `std::cmp::min(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This deviates from [`f32::min`], which returns the non-`NaN`
    /// operand when exactly one is `NaN`, but matches the behaviour of `numpy.minimum`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1i32, 5, 3];
    /// let b = ndarray::array![4i32, 2, 3];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = za.minimum(zb).data().to_ndarray::<i32>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1, 2, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = ndarray::array![f32::NAN, 1.0f32];
    /// let d = ndarray::array![2.0f32, 3.0f32];
    /// let zc = Array::from_ndarray(&c, ArrayParams::new())?;
    /// let zd = Array::from_ndarray(&d, ArrayParams::new())?;
    /// let result = zc.minimum(zd).data().to_ndarray::<f32>()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 1.0);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    Minimum,
    MinimumKernel,
    |a, b| MinimumTrait::minimum(a, b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
    output_type = "same"
);

trait MaximumTrait {
    fn maximum(self, other: Self) -> Self;
}
macro_rules! impl_integer_maximum {
    ($($t:ty),* $(,)?) => {
        $(impl MaximumTrait for $t {
            fn maximum(self, other: Self) -> Self {
                std::cmp::max(self, other)
            }
        })*
    };
}
macro_rules! impl_float_maximum {
    ($($t:ty),* $(,)?) => {
        $(impl MaximumTrait for $t {
            fn maximum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.max(other)
                }
            }
        })*
    };
}
impl_integer_maximum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
impl_float_maximum!(f32, f64);
#[cfg(feature = "half")]
impl_float_maximum!(f16);

trait MinimumTrait {
    fn minimum(self, other: Self) -> Self;
}
macro_rules! impl_integer_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl MinimumTrait for $t {
            fn minimum(self, other: Self) -> Self {
                std::cmp::min(self, other)
            }
        })*
    };
}
macro_rules! impl_float_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl MinimumTrait for $t {
            fn minimum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.min(other)
                }
            }
        })*
    };
}
impl_integer_minimum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
impl_float_minimum!(f32, f64);
#[cfg(feature = "half")]
impl_float_minimum!(f16);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(equal: Equal);
    define_array_op2_method!(not_equal: NotEqual);
    define_array_op2_method!(greater: Greater);
    define_array_op2_method!(greater_equal: GreaterEqual);
    define_array_op2_method!(less: Less);
    define_array_op2_method!(less_equal: LessEqual);
    define_array_op2_method!(maximum: Maximum);
    define_array_op2_method!(minimum: Minimum);
}
