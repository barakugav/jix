use std::sync::LazyLock;

use jix_core::scalar::{f16, Complex};
use pyo3::types::PyAnyMethods;

use crate::ops::common::{
    broadcast_operands, define_op2, define_op2_desc, OpDescriptor, Operand, Rank, Scalar,
};

macro_rules! unary_shortcut_desc {
    (|$x:ident| $f:expr, [$(($a_ty:ty, $b_ty:ty)),* $(,)?]) => {
        vec![
            $(
                crate::ops::common::OpFnDescriptor::new2::<$a_ty, $b_ty>(
                    crate::ops::common::CastKind::Safe,
                    |$x, _b| {
                        let res = { $f }.map(|res| res.into_any());
                        <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
                    },
                ),
            )*
        ]
    };
    (|$x:ident| $f:expr, [$($ty:ty),* $(,)?]) => {
        unary_shortcut_desc!(|$x| $f, [$(($ty, $ty)),*])
    };
}

define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result wraps on overflow (two's complement).
    /// For **complex** types each component is added independently:
    /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
    ///
    /// Available via the `+` operator on arrays.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 + i32 -> i32`, `i32 + f32 -> f64`. This is similar to
    /// numpy's type promotion but may pick a different common type.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly (prepend missing leading dimensions as 1, then expand size-1 dims).
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([10, 20, 30], dtype=np.int32)
    ///     result = jix.add(a, b)  # same as `a + b`
    ///     assert np.array_equal(result.numpy(), [11, 22, 33])
    ///
    ///     # Broadcasting: (3, 1) + (1, 4) -> (3, 4)
    ///     x = jix.compact(np.arange(3, dtype=np.int32).reshape(3, 1))
    ///     y = jix.compact(np.arange(4, dtype=np.int32).reshape(1, 4))
    ///     assert jix.add(x, y).shape == (3, 4)
    ///
    ///     # Mixed types: u8 + i32 -> i32
    ///     a8 = jix.compact([1, 2, 3], dtype=np.uint8)
    ///     b32 = jix.compact([100, 200, 300], dtype=np.int32)
    ///     assert jix.add(a8, b32).dtype == np.int32
    ///     ```
    add,
    Add,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
);

define_op2!(
    /// Element-wise subtraction of two arrays (`a - b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types the result wraps on underflow (two's complement).
    /// For **complex** types each component is subtracted independently:
    /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
    /// Two `bool` operands give `int8`: `True - False` is `1`, `False - True` is `-1` (numpy
    /// raises).
    ///
    /// Available via the `-` operator on arrays.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8 - i32 -> i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([10, 20, 30], dtype=np.int32)
    ///     b = jix.compact([1, 2, 3], dtype=np.int32)
    ///     result = jix.subtract(a, b)  # same as `a - b`
    ///     assert np.array_equal(result.numpy(), [9, 18, 27])
    ///     ```
    subtract,
    Sub,
    dispatch = {
        // i8 first, for two bools
        [i8, u8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
);
/// Element-wise multiplication of two arrays.
///
/// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
/// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
///
/// For **integer** types the result wraps on overflow (two's complement).
/// For **complex** types this is full complex multiplication:
/// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
///
/// Available via the `*` operator on arrays.
///
/// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
/// smallest type that can represent both without information loss (Safe casting
/// rules).
///
/// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
///
/// Args:
///     a: First operand.
///     b: Second operand.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
///         No computation occurs until the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1, 2, 3], dtype=np.int32)
///     b = jix.compact([4, 5, 6], dtype=np.int32)
///     result = jix.multiply(a, b)  # same as `a * b`
///     assert np.array_equal(result.numpy(), [4, 10, 18])
///     ```
#[pyo3::pyfunction]
pub fn multiply<'py>(
    a: &pyo3::Bound<'py, pyo3::PyAny>,
    b: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<crate::Array> {
    let a_op = Operand::from_any(a)?;

    // fast path for x * x
    if a.is(b) {
        let is_supported_dtype = a_op
            .rank_precision()
            .is_some_and(|(rank, _)| !matches!(rank, Rank::Bool));
        if is_supported_dtype {
            return crate::square(a);
        }
    }

    static DISPATCH_TABLE: LazyLock<OpDescriptor<2, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "multiply",
            define_op2_desc!(
                Mul,
                [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
                Safe
            ),
        )
    });

    let a = a_op;
    let b = Operand::from_any(b)?;
    let [a, b] = broadcast_operands([a, b])?;
    let res = DISPATCH_TABLE.dispatch2(a, b)?;
    Ok(crate::Array::from_core(res))
}
define_op2!(
    /// Element-wise (true) division of two arrays (`a / b`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    /// For integer quotients use [`jix.divide_floor()`][jix.divide_floor] (or the `//`
    /// operator).
    ///
    /// For **float** types the results are the IEEE 754 ones (`+/-inf`, `NaN`), and the errors
    /// those of numpy, reported according to [`jix.errstate()`][jix.errstate]: `x / 0` as
    /// `divide`, `0 / 0` and `inf / inf` as `invalid`, and an infinite quotient of finite operands
    /// as `over`.
    /// For **complex** types this is full complex division.
    ///
    /// Available via the `/` operator on arrays.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([10.0, 20.0, 30.0], dtype=np.float32)
    ///     b = jix.compact([4.0, 5.0, 8.0], dtype=np.float32)
    ///     result = jix.divide(a, b)  # same as `a / b`
    ///     assert np.array_equal(result.numpy(), [2.5, 4.0, 3.75])
    ///     ```
    divide,
    Div,
    dispatch = {
        [f16, f32, f64, Complex<f32>, Complex<f64>],
        Safe
    }
);
define_op2!(
    /// Element-wise truncated division of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
    /// `f64`.
    ///
    /// The quotient `a / b` rounded toward zero, like C's and Rust's integer `/`. It pairs with
    /// [`jix.remainder_trunc()`][jix.remainder_trunc].
    ///
    /// See also [`jix.divide_floor()`][jix.divide_floor] (the `//` operator) and
    /// [`jix.divide_euclid()`][jix.divide_euclid].
    ///
    /// For **integer** dtypes a zero divisor gives `0`, and the minimum signed value by `-1` gives
    /// itself, as in numpy. They are reported as `divide` and `over`, according to
    /// [`jix.errstate()`][jix.errstate].
    /// For **float** dtypes the quotient is of the exact `a / b`, as an integer-valued float:
    /// `1.0 // 0.1` is `9.0`, though `1.0 / 0.1` rounds to `10.0`. It is exact below `2^53` (`2^24`
    /// for `f32`) in magnitude, and within one ulp above. A zero divisor produces `+-inf` (`NaN`
    /// for `0 / 0`) and an infinite dividend `NaN`, as in numpy.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8` and `i32` -> `i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the dividend).
    ///     b: Second operand (the divisor).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
    ///     b = jix.compact([2, 2, -2, -2], dtype=np.int32)
    ///     assert np.array_equal(jix.divide_trunc(a, b).numpy(), [3, -3, -3, 3])
    ///     ```
    divide_trunc,
    DivideTrunc,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);
define_op2!(
    /// Element-wise floored division of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
    /// `f64`.
    ///
    /// The quotient `a / b` rounded toward negative infinity, like Python's and numpy's `//`. It
    /// pairs with [`jix.remainder_floor()`][jix.remainder_floor], the `%` operator:
    /// `a == (a // b) * b + a % b`. For floats it matches numpy, except for some quotients beyond
    /// `2^51` (`2^22` for `f32`) in magnitude, where numpy's own result is off by one.
    ///
    /// Available via the `//` operator on arrays. See also
    /// [`jix.divide_trunc()`][jix.divide_trunc] and
    /// [`jix.divide_euclid()`][jix.divide_euclid].
    ///
    /// For **integer** dtypes a zero divisor gives `0`, and the minimum signed value by `-1` gives
    /// itself, as in numpy. They are reported as `divide` and `over`, according to
    /// [`jix.errstate()`][jix.errstate].
    /// For **float** dtypes the quotient is of the exact `a / b`, as an integer-valued float:
    /// `1.0 // 0.1` is `9.0`, though `1.0 / 0.1` rounds to `10.0`. It is exact below `2^53` (`2^24`
    /// for `f32`) in magnitude, and within one ulp above. A zero divisor produces `+-inf` (`NaN`
    /// for `0 / 0`) and an infinite dividend `NaN`, as in numpy.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8` and `i32` -> `i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the dividend).
    ///     b: Second operand (the divisor).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
    ///     b = jix.compact([2, 2, -2, -2], dtype=np.int32)
    ///     result = jix.divide_floor(a, b)  # same as `a // b`
    ///     assert np.array_equal(result.numpy(), [3, -4, -4, 3])
    ///     ```
    divide_floor,
    DivideFloor,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);
define_op2!(
    /// Element-wise Euclidean division of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
    /// `f64`.
    ///
    /// The quotient for which [`jix.remainder_euclid()`][jix.remainder_euclid] is never
    /// negative: `floor(a / b)` for `b > 0` and `ceil(a / b)` for `b < 0`, like Rust's
    /// `div_euclid`. numpy and torch have no equivalent.
    ///
    /// See also [`jix.divide_trunc()`][jix.divide_trunc] and
    /// [`jix.divide_floor()`][jix.divide_floor] (the `//` operator).
    ///
    /// For **integer** dtypes a zero divisor gives `0`, and the minimum signed value by `-1` gives
    /// itself, as in numpy. They are reported as `divide` and `over`, according to
    /// [`jix.errstate()`][jix.errstate].
    /// For **float** dtypes the quotient is of the exact `a / b`, as an integer-valued float:
    /// `1.0 // 0.1` is `9.0`, though `1.0 / 0.1` rounds to `10.0`. It is exact below `2^53` (`2^24`
    /// for `f32`) in magnitude, and within one ulp above. A zero divisor produces `+-inf` (`NaN`
    /// for `0 / 0`) and an infinite dividend `NaN`, as in numpy.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8` and `i32` -> `i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the dividend).
    ///     b: Second operand (the divisor).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
    ///     b = jix.compact([2, 2, -2, -2], dtype=np.int32)
    ///     assert np.array_equal(jix.divide_euclid(a, b).numpy(), [3, -4, -3, 4])
    ///     ```
    divide_euclid,
    DivideEuclid,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);
define_op2!(
    /// Element-wise truncated remainder of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
    /// `f64`.
    ///
    /// Computes `a - b * trunc(a / b)`: the result has the sign of the dividend `a`, like Rust's
    /// and C's `%`, `numpy.fmod` and `torch.fmod`. It pairs with
    /// [`jix.divide_trunc()`][jix.divide_trunc].
    ///
    /// See also [`jix.remainder_floor()`][jix.remainder_floor] (the `%` operator) and
    /// [`jix.remainder_euclid()`][jix.remainder_euclid].
    ///
    /// For **integer** dtypes a zero divisor gives `0`, as in numpy, reported as `divide`
    /// according to [`jix.errstate()`][jix.errstate]. For **float** dtypes it produces `NaN`.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8` and `i32` -> `i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the dividend).
    ///     b: Second operand (the divisor).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
    ///     b = jix.compact([3, 3, -3, -3], dtype=np.int32)
    ///     assert np.array_equal(jix.remainder_trunc(a, b).numpy(), [1, -1, 1, -1])
    ///     ```
    remainder_trunc,
    RemainderTrunc,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);
define_op2!(
    /// Element-wise floored remainder of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
    /// `f64`.
    ///
    /// Computes `a - b * floor(a / b)`: the result has the sign of the divisor `b`, like Python's
    /// `%`, `numpy.remainder` and `torch.remainder`. For **float** dtypes it is exactly Python's
    /// float `%`, which gives a zero result the sign of `b`.
    ///
    /// Available via the `%` operator on arrays. See also
    /// [`jix.remainder_trunc()`][jix.remainder_trunc] and
    /// [`jix.remainder_euclid()`][jix.remainder_euclid].
    ///
    /// For **integer** dtypes a zero divisor gives `0`, as in numpy, reported as `divide`
    /// according to [`jix.errstate()`][jix.errstate]. For **float** dtypes it produces `NaN`.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8` and `i32` -> `i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the dividend).
    ///     b: Second operand (the divisor).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
    ///     b = jix.compact([3, 3, -3, -3], dtype=np.int32)
    ///     result = jix.remainder_floor(a, b)  # same as `a % b`
    ///     assert np.array_equal(result.numpy(), [1, 2, -2, -1])
    ///     ```
    remainder_floor,
    RemainderFloor,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);
define_op2!(
    /// Element-wise Euclidean remainder of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
    /// `f64`.
    ///
    /// Computes `a - |b| * floor(a / |b|)`: the result is never negative, like Rust's
    /// `rem_euclid`, and equals [`jix.remainder_floor()`][jix.remainder_floor] when `b > 0`. numpy
    /// and torch have no equivalent.
    ///
    /// See also [`jix.remainder_trunc()`][jix.remainder_trunc] and
    /// [`jix.remainder_floor()`][jix.remainder_floor] (the `%` operator).
    ///
    /// For **integer** dtypes a zero divisor gives `0`, as in numpy, reported as `divide`
    /// according to [`jix.errstate()`][jix.errstate]. For **float** dtypes it produces `NaN`.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
    /// smallest type that can represent both without information loss (Safe casting
    /// rules). For example `u8` and `i32` -> `i32`.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand (the dividend).
    ///     b: Second operand (the divisor).
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast shape.
    ///         No computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
    ///     b = jix.compact([3, 3, -3, -3], dtype=np.int32)
    ///     assert np.array_equal(jix.remainder_euclid(a, b).numpy(), [1, 2, 1, 2])
    ///     ```
    remainder_euclid,
    RemainderEuclid,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);

/// Element-wise exponentiation (`a` raised to the power `b`).
///
/// Supported base dtypes: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
/// `f16`, `f32`, `f64`. For an **integer** base the exponent must be an integer; for a
/// **float** base it may be an integer or a float. Complex dtypes are not supported yet.
///
/// For a directly supported `(base, exponent)` pair the output dtype equals the
/// base dtype - e.g. `u8 ** u8 -> u8`, `i32 ** i32 -> i32`, `f32 ** i32 -> f32`,
/// `f64 ** f64 -> f64`. Integer results wrap on overflow (two's complement), as in
/// numpy.
///
/// A negative exponent of an integer base gives `0` and is reported as `invalid`
/// according to [`jix.errstate()`][jix.errstate] (numpy raises a `ValueError`
/// instead). For a float base, as numpy: a negative base with a non-integer exponent
/// produces `NaN`, `invalid`, zero to a negative power `inf`, `divide`, and finite operands
/// with an infinite result `over`.
///
/// **Type promotion**: when `a` and `b` do not directly match a supported pair,
/// both operands are cast under Safe casting rules and the first matching pair
/// is used, e.g. `u8 ** i8 -> i16`. A `u64` base or exponent with a signed one
/// promotes to float: `u64 ** i64 -> f64`.
///
/// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
///
/// Args:
///     a: First operand (the base).
///     b: Second operand (the exponent).
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the result dtype (after type promotion) and broadcast
///         shape. No computation occurs until the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([2.0, 3.0, 4.0], dtype=np.float32)
///     b = jix.compact([3.0, 2.0, 0.5], dtype=np.float32)
///     result = jix.power(a, b)
///     assert np.array_equal(result.numpy(), [8.0, 9.0, 2.0])
///
///     # Raise each element to a fixed exponent using a typed scalar.
///     result2 = jix.power(a, np.float32(2.0))
///     assert np.array_equal(result2.numpy(), [4.0, 9.0, 16.0])
///
///     # An integer base and exponent keep the base dtype.
///     ai = jix.compact([2, 3, 4], dtype=np.int32)
///     result3 = jix.power(ai, np.int32(3))
///     assert result3.dtype == np.dtype("int32")
///     assert np.array_equal(result3.numpy(), [8, 27, 64])
///     assert jix.power(ai, 3).dtype == np.dtype("int32")
///     ```
#[pyo3::pyfunction]
pub fn power<'py>(
    a: &pyo3::Bound<'py, pyo3::PyAny>,
    b: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<crate::Array> {
    static DISPATCH_TABLE: LazyLock<OpDescriptor<2, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "power",
            // TODO complex bases: (Complex<f32>, i32), (Complex<f32>, f32),
            // (Complex<f64>, i64), (Complex<f64>, f64).
            define_op2_desc!(
                Pow,
                [
                    // ints, unsigned exponents
                    (u8, u8),
                    (i8, u8),
                    (u16, u16),
                    (i16, u16),
                    (u32, u32),
                    (i32, u32),
                    (u64, u32),
                    (i64, u32),
                    // ints, signed exponents
                    (i8, i8),
                    (i16, i16),
                    (i32, i32),
                    (i64, i64),
                    // floats
                    (f16, i32),
                    (f16, f16),
                    (f32, i32),
                    (f32, f32),
                    (f64, i32),
                    (f64, f64)
                ],
                Safe
            ),
        )
    });

    // `x ** 2`. Same pairs as `DISPATCH_TABLE`, so the promotion is identical.
    static SQUARE_TABLE: LazyLock<OpDescriptor<2, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "power",
            unary_shortcut_desc!(
                |a| jix_core::ops::Square::new_array(a),
                [
                    (u8, u8),
                    (i8, u8),
                    (u16, u16),
                    (i16, u16),
                    (u32, u32),
                    (i32, u32),
                    (u64, u32),
                    (i64, u32),
                    (i8, i8),
                    (i16, i16),
                    (i32, i32),
                    (i64, i64),
                    (f16, i32),
                    (f16, f16),
                    (f32, i32),
                    (f32, f32),
                    (f64, i32),
                    (f64, f64),
                ]
            ),
        )
    });
    // `x ** 0.5`.
    static SQRT_TABLE: LazyLock<OpDescriptor<2, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "power",
            unary_shortcut_desc!(
                |a| jix_core::ops::Sqrt::new_array(a),
                [
                    (f16, i32),
                    (f16, f16),
                    (f32, i32),
                    (f32, f32),
                    (f64, i32),
                    (f64, f64)
                ]
            ),
        )
    });

    let a = Operand::from_any(a)?;
    let b = Operand::from_any(b)?;

    let mut table = &*DISPATCH_TABLE;
    if let Operand::Scalar { value, .. } = &b {
        match value {
            Scalar::UInt(2) | Scalar::Int(2) | Scalar::Float(2.0) => table = &*SQUARE_TABLE,
            Scalar::Float(0.5) => table = &*SQRT_TABLE,
            _ => {}
        };
    }

    let [a, b] = broadcast_operands([a, b])?;
    let res = table.dispatch2(a, b)?;
    Ok(crate::Array::from_core(res))
}
