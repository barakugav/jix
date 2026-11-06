use std::sync::LazyLock;

use jix_core::scalar::{f16, Complex};
use pyo3::exceptions::PyValueError;
use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::{prelude::*, type_hint_identifier, type_hint_subscript};

use crate::ops::common::{
    broadcast_operands, define_op2, define_op2_table, OpDescriptor, Operand, Rank, Scalar,
};
use crate::Array;

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
pub fn multiply<'py>(a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Array> {
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

    let a = a_op;
    let b = Operand::from_any(b)?;
    let [a, b] = broadcast_operands([a, b])?;

    let dispatch_table = define_op2_table!(
        multiply,
        Mul,
        dispatch = {
            [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
            Safe
        }
    );
    let res = dispatch_table.dispatch2(a, b)?;

    Ok(Array::from_core(res))
}

/// Element-wise division of two arrays (`a / b`), with the quotient optionally rounded to an
/// integer.
///
/// `rounding_mode` picks the rounding:
///
/// - `None` (the default): true division, for the float and complex dtypes. Available via the `/`
///   operator.
/// - `"trunc"`: toward zero, like C's and Rust's integer `/`.
/// - `"floor"`: toward negative infinity, like Python's and numpy's `//`. Available via the `//`
///   operator.
/// - `"euclid"`: `floor(a / b)` for `b > 0` and `ceil(a / b)` for `b < 0`, the quotient for which
///   the `"euclid"` remainder is never negative, like Rust's `div_euclid`. numpy and torch have no
///   equivalent.
///
/// A rounded quotient pairs with the [`jix.remainder()`][jix.remainder] of the same
/// `rounding_mode`: `a == divide(a, b) * b + remainder(a, b)`.
///
/// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>` for true division, and
/// `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64` with a
/// `rounding_mode`.
///
/// For true division of **float** types the results are the IEEE 754 ones, and division by zero
/// produces `+/-inf` or `NaN`. For **complex** types this is full complex division.
///
/// With a `rounding_mode`, for **integer** dtypes a zero divisor raises an `ArithmeticError` when
/// the result is read, and the minimum signed value by `-1` wraps to itself.
/// For **float** dtypes the quotient is of the exact `a / b`, as an integer-valued float:
/// `1.0 // 0.1` is `9.0`, though `1.0 / 0.1` rounds to `10.0`. It is exact below `2^53` (`2^24`
/// for `f32`) in magnitude, and within one ulp above. A zero divisor produces `+-inf` (`NaN` for
/// `0 / 0`) and an infinite dividend `NaN`, as in numpy. `"floor"` matches numpy, except for some
/// quotients beyond `2^51` (`2^22` for `f32`) in magnitude, where numpy's own result is off by one.
/// `"trunc"` differs from `torch.div`, which truncates the rounded `a / b`, and so gives `10.0`
/// for `1.0 / 0.1`.
///
/// **Type promotion**: if `a` and `b` have different dtypes, both are cast to the
/// smallest type that can represent both without information loss (Safe casting
/// rules). For example `u8` and `i32` -> `i32` with a `rounding_mode`, and `i32` and `i32` ->
/// `f64` for true division.
///
/// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
///
/// Args:
///     a: First operand (the dividend).
///     b: Second operand (the divisor).
///     rounding_mode: How to round the quotient: `None` for true division, or `"trunc"`,
///         `"floor"` or `"euclid"`.
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
///
///     a = jix.compact([7, -7, 7, -7], dtype=np.int32)
///     b = jix.compact([2, 2, -2, -2], dtype=np.int32)
///     assert np.array_equal(jix.divide(a, b, rounding_mode="trunc").numpy(), [3, -3, -3, 3])
///     result = jix.divide(a, b, rounding_mode="floor")  # same as `a // b`
///     assert np.array_equal(result.numpy(), [3, -4, -4, 3])
///     assert np.array_equal(jix.divide(a, b, rounding_mode="euclid").numpy(), [3, -4, -3, 4])
///     ```
#[pyo3::pyfunction(signature = (a, b, *, rounding_mode=None))]
pub fn divide<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rounding_mode: Option<RoundingMode>,
) -> PyResult<Array> {
    let table = match rounding_mode {
        None => define_op2_table!(
            divide,
            Div,
            dispatch = { [f16, f32, f64, Complex<f32>, Complex<f64>], Safe }
        ),
        Some(RoundingMode::Trunc) => define_op2_table!(
            divide,
            DivideTrunc,
            dispatch = { [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64], Safe }
        ),
        Some(RoundingMode::Floor) => define_op2_table!(
            divide,
            DivideFloor,
            dispatch = { [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64], Safe }
        ),
        Some(RoundingMode::Euclid) => define_op2_table!(
            divide,
            DivideEuclid,
            dispatch = { [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64], Safe }
        ),
    };
    let a = Operand::from_any(a)?;
    let b = Operand::from_any(b)?;
    let [a, b] = broadcast_operands([a, b])?;
    Ok(Array::from_core(table.dispatch2(a, b)?))
}

/// Element-wise remainder of two arrays: `a - b * q`, where `q` is the quotient `a / b` rounded to
/// an integer.
///
/// `rounding_mode` picks the rounding of `q`:
///
/// - `"floor"` (the default): toward negative infinity, so the result has the sign of the divisor
///   `b`, like Python's `%`, `numpy.remainder` and `torch.remainder`. Available via the `%`
///   operator. For **float** dtypes it is exactly Python's float `%`, which gives a zero result
///   the sign of `b`.
/// - `"trunc"`: toward zero, so the result has the sign of the dividend `a`, like Rust's and C's
///   `%`, `numpy.fmod` and `torch.fmod`.
/// - `"euclid"`: so that the result is never negative, `a - |b| * floor(a / |b|)`, like Rust's
///   `rem_euclid`. It equals the `"floor"` remainder when `b > 0`. numpy and torch have no
///   equivalent.
///
/// It pairs with the [`jix.divide()`][jix.divide] of the same `rounding_mode`:
/// `a == divide(a, b) * b + remainder(a, b)`.
///
/// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`,
/// `f64`.
///
/// For **integer** dtypes a zero divisor raises an `ArithmeticError` when the result is read.
/// For **float** dtypes it produces `NaN`.
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
///     rounding_mode: How to round the quotient: `"floor"`, `"trunc"` or `"euclid"`.
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
///     result = jix.remainder(a, b)  # same as `a % b`
///     assert np.array_equal(result.numpy(), [1, 2, -2, -1])
///     assert np.array_equal(jix.remainder(a, b, rounding_mode="trunc").numpy(), [1, -1, 1, -1])
///     assert np.array_equal(jix.remainder(a, b, rounding_mode="euclid").numpy(), [1, 2, 1, 2])
///     ```
#[pyo3::pyfunction(
    signature = (a, b, *, rounding_mode: "Literal['trunc', 'floor', 'euclid']" = "floor")
)]
pub fn remainder<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rounding_mode: &str,
) -> PyResult<Array> {
    let table = match rounding_mode.parse()? {
        RoundingMode::Trunc => define_op2_table!(
            remainder,
            RemainderTrunc,
            dispatch = { [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64], Safe }
        ),
        RoundingMode::Floor => define_op2_table!(
            remainder,
            RemainderFloor,
            dispatch = { [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64], Safe }
        ),
        RoundingMode::Euclid => define_op2_table!(
            remainder,
            RemainderEuclid,
            dispatch = { [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64], Safe }
        ),
    };
    let a = Operand::from_any(a)?;
    let b = Operand::from_any(b)?;
    let [a, b] = broadcast_operands([a, b])?;
    Ok(Array::from_core(table.dispatch2(a, b)?))
}

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
/// A negative exponent of an integer base raises an `ArithmeticError` when the result is read.
/// For a float base the results are those of numpy, `+/-inf` and `NaN` included.
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
pub fn power<'py>(a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Array> {
    // `x ** 2`. Same pairs as the `Pow` table, so the promotion is identical.
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

    // TODO complex bases: (Complex<f32>, i32), (Complex<f32>, f32),
    // (Complex<f64>, i64), (Complex<f64>, f64).
    let mut table = define_op2_table!(
        power,
        Pow,
        dispatch = {
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
        }
    );
    if let Operand::Scalar { value, .. } = &b {
        match value {
            Scalar::UInt(2) | Scalar::Int(2) | Scalar::Float(2.0) => table = &*SQUARE_TABLE,
            Scalar::Float(0.5) => table = &*SQRT_TABLE,
            _ => {}
        };
    }

    let [a, b] = broadcast_operands([a, b])?;
    let res = table.dispatch2(a, b)?;
    Ok(Array::from_core(res))
}

/// The `rounding_mode` of [`crate::ops::divide`] and [`crate::ops::remainder`]: how the quotient
/// `a / b` is rounded to an integer.
#[derive(Clone, Copy)]
pub enum RoundingMode {
    /// Toward zero.
    Trunc,
    /// Toward negative infinity.
    Floor,
    /// Toward the quotient whose remainder is never negative.
    Euclid,
}

impl<'a, 'py> FromPyObject<'a, 'py> for RoundingMode {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("typing", "Literal"),
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("trunc")
        },
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("floor")
        },
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("euclid")
        }
    );

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        obj.extract::<&str>()?.parse()
    }
}

impl std::str::FromStr for RoundingMode {
    type Err = PyErr;

    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "trunc" => Ok(Self::Trunc),
            "floor" => Ok(Self::Floor),
            "euclid" => Ok(Self::Euclid),
            other => Err(PyValueError::new_err(format!(
                "invalid rounding_mode value: '{other}', expected 'trunc', 'floor' or 'euclid'"
            ))),
        }
    }
}
