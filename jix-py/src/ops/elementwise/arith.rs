use std::sync::LazyLock;

use jix_core::scalar::{f16, Complex};
use pyo3::prelude::*;

use crate::ops::common::{
    broadcast_operands, define_op1, define_op1_desc, define_op1_table, define_op2,
    define_op2_table, CastKind, OpDescriptor, OpFnDescriptor, Operand, Rank, Scalar,
};
use crate::Array;

/// Arithmetic negation applied element-wise (`-array`).
///
/// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `f16`, `f32`, `f64`,
/// `Complex<f32>`, `Complex<f64>`.
///
/// For **integer** types the result is the two's-complement negation. Negating the
/// minimum representable value (e.g. `int8` `-128`) overflows and wraps.
/// For **complex** types both components are negated independently.
///
/// Available via the unary `-` operator on arrays: `-arr`.
///
/// Args:
///     array: Input array. An unsigned integer input is negated as the signed integer of the
///         same width, `u8 -> i8`, ..., `u64 -> i64`, so it wraps to the same bits as numpy's
///         unsigned negation: `-uint8(200)` is `int8(56)` (numpy `uint8(56)`).
///         A `bool` input is cast to `i8` (False -> 0, True -> -1).
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
///         the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.0, -2.5, 3.0], dtype=np.float32)
///     result = jix.negative(a)
///     assert np.array_equal(result.numpy(), [-1.0, 2.5, -3.0])
///
///     # Unsigned integers are negated as the signed type of the same width.
///     b = jix.compact([1, 2, 200], dtype=np.uint8)
///     result = jix.negative(b)
///     assert result.dtype == np.int8
///     assert np.array_equal(result.numpy(), [-1, -2, 56])
///     ```
#[pyo3::pyfunction]
pub fn negative<'py>(array: &Bound<'py, PyAny>) -> PyResult<Array> {
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, ()>> = LazyLock::new(|| {
        let mut fns = vec![
            OpFnDescriptor::new1::<u8>(CastKind::None, |a| Ok((-a.cast::<i8>()).into_any())),
            OpFnDescriptor::new1::<u16>(CastKind::None, |a| Ok((-a.cast::<i16>()).into_any())),
            OpFnDescriptor::new1::<u32>(CastKind::None, |a| Ok((-a.cast::<i32>()).into_any())),
            OpFnDescriptor::new1::<u64>(CastKind::None, |a| Ok((-a.cast::<i64>()).into_any())),
        ];
        fns.extend(define_op1_desc!(
            Neg,
            [i8, i16, i32, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
            Safe
        ));
        OpDescriptor::new("negative", fns)
    });
    let array = Operand::from_any(array)?;
    Ok(Array::from_core(DISPATCH_TABLE.dispatch1(array)?))
}

/// Computes the absolute value of each element.
///
/// Supported dtypes and output dtype:
///
/// | Input dtype | Output dtype |
/// |-------------|--------------|
/// | `bool` | same |
/// | `u8`, `u16`, `u32`, `u64` | same |
/// | `i8`, `i16`, `i32`, `i64` | same |
/// | `f16`, `f32`, `f64` | same |
/// | `Complex<f32>` | `f32` |
/// | `Complex<f64>` | `f64` |
///
/// For **unsigned integer** and **bool** types this is a no-op: a [`jix.Array`][jix.Array]
/// input is returned as is (the same object), not wrapped in a new view.
///
/// For **complex** types the result is the modulus `sqrt(re^2 + im^2)` computed via
/// `hypot` for numerical stability. The output dtype is the real component type.
///
/// For **signed integer** types, the minimum value overflows: `abs(-128)` on `int8`
/// wraps back to `-128`.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
///         the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([-3, 0, 5, -7], dtype=np.int32)
///     result = jix.absolute(a)
///     assert np.array_equal(result.numpy(), [3, 0, 5, 7])
///     ```
#[pyo3::pyfunction]
pub fn absolute<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let mut operand = Operand::from_any(array)?;

    // Unsigned and bool values are their own absolute value, no-op
    if let Some((rank, _)) = operand.rank_precision()
        && matches!(rank, Rank::UInt | Rank::Bool)
    {
        return operand.into_py_array(py);
    }

    // Array is a scalar, compute abs directly
    if let Operand::Scalar {
        value,
        shape: _,
        precision: _,
        params: _,
    } = &mut operand
    {
        *value = match *value {
            Scalar::Bool(value) => Scalar::Bool(value),
            Scalar::UInt(value) => Scalar::UInt(value),
            Scalar::Int(value) => Scalar::Int(value.wrapping_abs()),
            Scalar::Float(value) => Scalar::Float(value.abs()),
            Scalar::Complex(complex) => Scalar::Float(complex.norm()),
        };
        return operand.into_py_array(py);
    }

    let dispatch_table = define_op1_table!(
        absolute,
        Abs,
        dispatch = { [i8, i16, i32, i64, f16, f32, f64, Complex<f32>, Complex<f64>], None }
    );
    let res = dispatch_table.dispatch1(operand)?;
    Bound::new(py, Array::from_core(res))
}

define_op1!(
    /// Returns the sign of each element.
    ///
    /// Supported dtypes: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`,
    /// `uint64`, `float16`, `float32`, `float64`.
    ///
    /// For **signed integer** types: returns `-1`, `0`, or `+1` of the same dtype.
    ///
    /// For **unsigned integer** types: returns `0` or `1` of the same dtype (unsigned values
    /// cannot be negative).
    ///
    /// For **float** types: returns `+1.0` for positive values, `-1.0` for negative values,
    /// `0.0` for `+0.0` and `-0.0`, and `NaN` for `NaN`.
    ///
    /// The input is never cast, so the output dtype always equals the input dtype; `bool` is not
    /// supported (as in numpy).
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape and dtype as `array`. No
    ///         computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([3, -5, 0], dtype=np.int32)
    ///     result = jix.sign(a)
    ///     assert np.array_equal(result.numpy(), [1, -1, 0])
    ///
    ///     b = jix.compact([3.0, -5.0, -0.1], dtype=np.float32)
    ///     result = jix.sign(b)
    ///     assert np.array_equal(result.numpy(), [1.0, -1.0, -1.0])
    ///
    ///     c = jix.compact([1, 0, 5], dtype=np.uint8)
    ///     result = jix.sign(c)
    ///     assert result.dtype == np.uint8
    ///     assert np.array_equal(result.numpy(), [1, 0, 1])
    ///     ```
    sign,
    Sign,
    dispatch = {
        [i8, u8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        None
    }
);

define_op1!(
    /// Squares each element (`x * x`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`,
    /// `Complex<f32>`, `Complex<f64>`. The output dtype always matches the input dtype.
    ///
    /// For **integer** types squaring can overflow, following two's-complement `*` semantics
    /// (the result wraps).
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape and dtype as `array`. No
    ///         computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.0, -2.0, 3.0], dtype=np.float32)
    ///     result = jix.square(a)
    ///     assert np.array_equal(result.numpy(), [1.0, 4.0, 9.0])
    ///
    ///     # Integer dtypes are preserved.
    ///     b = jix.compact([2, -3, 4], dtype=np.int32)
    ///     result = jix.square(b)
    ///     assert result.dtype == np.int32
    ///     assert np.array_equal(result.numpy(), [4, 9, 16])
    ///     ```
    square,
    Square,
    dispatch = {
        [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    }
);

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
