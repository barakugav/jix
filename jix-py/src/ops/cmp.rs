use std::sync::LazyLock;

use jix_core::dtype::Dtyped;
use jix_core::scalar::{f16, Complex};
use pyo3::prelude::*;

use crate::asarray;
use crate::ops::common::{
    broadcast_operands, define_op2, CastKind, OpDescriptor, OpFnDescriptor, Operand, Scalar,
};
use crate::util::IntoPyResult;

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    ///
    /// For **float** types, `NaN != NaN` per IEEE 754: comparing two `NaN` values returns
    /// `False`. For **complex** types, both the real and imaginary components must be equal.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation occurs
    ///         until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([1, 0, 3], dtype=np.int32)
    ///     result = jix.equal(a, b)
    ///     assert np.array_equal(result.numpy(), [True, False, True])
    ///
    ///     # NaN != NaN per IEEE 754.
    ///     c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    ///     d = jix.compact([float('nan'), 1.0], dtype=np.float32)
    ///     result = jix.equal(c, d)
    ///     assert np.array_equal(result.numpy(), [False, True])
    ///     ```
    equal,
    Equal,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    ///
    /// For **float** types, `NaN != NaN` returns `True` per IEEE 754.
    /// For **complex** types, returns `True` if either the real or imaginary component differs.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation occurs
    ///         until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 2, 3], dtype=np.int32)
    ///     b = jix.compact([1, 0, 3], dtype=np.int32)
    ///     result = jix.not_equal(a, b)
    ///     assert np.array_equal(result.numpy(), [False, True, False])
    ///     ```
    not_equal,
    NotEqual,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True > False`.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation occurs
    ///         until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([3, 1, 2], dtype=np.int32)
    ///     b = jix.compact([1, 1, 3], dtype=np.int32)
    ///     result = jix.greater(a, b)
    ///     assert np.array_equal(result.numpy(), [True, False, False])
    ///     ```
    greater,
    Greater,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise greater-than-or-equal test (`a >= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True >= False`, and both `True >= True` and `False >= False` hold.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation occurs
    ///         until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([3, 1, 2], dtype=np.int32)
    ///     b = jix.compact([1, 1, 3], dtype=np.int32)
    ///     result = jix.greater_equal(a, b)
    ///     assert np.array_equal(result.numpy(), [True, True, False])
    ///     ```
    greater_equal,
    GreaterEqual,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False < True`.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation occurs
    ///         until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 1, 3], dtype=np.int32)
    ///     b = jix.compact([3, 1, 2], dtype=np.int32)
    ///     result = jix.less(a, b)
    ///     assert np.array_equal(result.numpy(), [True, False, False])
    ///     ```
    less,
    Less,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise less-than-or-equal test (`a <= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False <= True`, and both `False <= False` and `True <= True` hold.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation occurs
    ///         until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 1, 3], dtype=np.int32)
    ///     b = jix.compact([3, 1, 2], dtype=np.int32)
    ///     result = jix.less_equal(a, b)
    ///     assert np.array_equal(result.numpy(), [True, True, False])
    ///     ```
    less_equal,
    LessEqual,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise maximum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the promoted input dtype.
    ///
    /// For **integer** and **bool** types the result is `max(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This matches `numpy.maximum` (not `numpy.fmax`, which ignores NaN).
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] with the element-wise maximum and the broadcast shape. No
    ///         computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 5, 3], dtype=np.int32)
    ///     b = jix.compact([4, 2, 3], dtype=np.int32)
    ///     result = jix.maximum(a, b)
    ///     assert np.array_equal(result.numpy(), [4, 5, 3])
    ///
    ///     # NaN is propagated: if either operand is NaN the result is NaN.
    ///     c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    ///     d = jix.compact([2.0, 3.0], dtype=np.float32)
    ///     result = jix.maximum(c, d)
    ///     assert np.isnan(result.numpy()[0])
    ///     assert result.numpy()[1] == 3.0
    ///     ```
    maximum,
    Maximum,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);

define_op2!(
    /// Element-wise minimum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the promoted input dtype.
    ///
    /// For **integer** and **bool** types the result is `min(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This matches `numpy.minimum` (not `numpy.fmin`, which ignores NaN).
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] with the element-wise minimum and the broadcast shape. No
    ///         computation occurs until the result is read.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1, 5, 3], dtype=np.int32)
    ///     b = jix.compact([4, 2, 3], dtype=np.int32)
    ///     result = jix.minimum(a, b)
    ///     assert np.array_equal(result.numpy(), [1, 2, 3])
    ///
    ///     # NaN is propagated: if either operand is NaN the result is NaN.
    ///     c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    ///     d = jix.compact([2.0, 3.0], dtype=np.float32)
    ///     result = jix.minimum(c, d)
    ///     assert np.isnan(result.numpy()[0])
    ///     assert result.numpy()[1] == 1.0
    ///     ```
    minimum,
    Minimum,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);

/// Clamps each element to the range `[min, max]`.
///
/// Elements below `min` are replaced by `min`; elements above `max` are replaced by `max`.
/// Both `min` and `max` are optional: omitting one removes that bound.
/// Passing neither `min` nor `max` returns a lazy view of the array unchanged.
///
/// Supported dtypes: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
/// `f16`, `f32`, `f64`.
///
/// Args:
///     array: Input array.
///     min: Lower bound (inclusive). A Python scalar or NumPy scalar. When omitted or `None`,
///         no lower bound is applied.
///     max: Upper bound (inclusive). A Python scalar or NumPy scalar. When omitted or `None`,
///         no upper bound is applied.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape and dtype as `array`. No
///         computation occurs until the result is read.
///
/// Raises:
///     ValueError: If both `min` and `max` are provided and `min > max` (or either is `NaN`).
///     TypeError: If a complex limit is provided.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([-3, 0, 5, 10], dtype=np.int32)
///
///     # Clamp to [0, 7].
///     result = jix.clamp(a, min=0, max=7)
///     assert np.array_equal(result.numpy(), [0, 0, 5, 7])
///
///     # Only a lower bound.
///     result_min = jix.clamp(a, min=2)
///     assert np.array_equal(result_min.numpy(), [2, 2, 5, 10])
///
///     # Only an upper bound.
///     result_max = jix.clamp(a, max=4)
///     assert np.array_equal(result_max.numpy(), [-3, 0, 4, 4])
///
///     # Float array: clamp to [0.0, 1.0].
///     b = jix.compact([-0.5, 0.3, 1.2], dtype=np.float32)
///     result_f = jix.clamp(b, min=0.0, max=1.0)
///     assert np.allclose(result_f.numpy(), [0.0, 0.3, 1.0])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyo3::pyfunction(signature = (array, min=None, max=None))]
pub fn clamp<'py>(
    array: &Bound<'py, PyAny>,
    min: Option<&Bound<'py, PyAny>>,
    max: Option<&Bound<'py, PyAny>>,
) -> pyo3::PyResult<Bound<'py, crate::Array>> {
    if min.is_none() && max.is_none() {
        return asarray(array);
    }
    let py = array.py();

    struct ClampArgs {
        min: Option<Scalar>,
        max: Option<Scalar>,
    }
    fn clamp_op_descriptor<T>() -> OpFnDescriptor<1, ClampArgs>
    where
        T: Dtyped + PartialOrd + std::fmt::Debug,
        bool: jix_core::scalar::Cast<T>,
        u64: jix_core::scalar::Cast<T>,
        i64: jix_core::scalar::Cast<T>,
        f64: jix_core::scalar::Cast<T>,
    {
        OpFnDescriptor::new1_args::<T>(CastKind::None, |a, args: ClampArgs| {
            let [min, max] = [args.min, args.max]
                .map(|limit| {
                    limit
                        .map(|limit| {
                            Ok(match limit {
                        Scalar::Bool(v) => <bool as jix_core::scalar::Cast<T>>::cast(v),
                        Scalar::UInt(v) => <u64 as jix_core::scalar::Cast<T>>::cast(v),
                        Scalar::Int(v) => <i64 as jix_core::scalar::Cast<T>>::cast(v),
                        Scalar::Float(v) => <f64 as jix_core::scalar::Cast<T>>::cast(v),
                        Scalar::Complex(_) => return Err(pyo3::exceptions::PyTypeError::new_err(
                            "clamp does not support complex limits; pass a real-valued min and max",
                        )),
                    })
                        })
                        .transpose()
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .unwrap();
            Ok(match (min, max) {
                (None, None) => a.into_type_dyn().into_any(),
                (Some(min), None) => {
                    let res =
                        jix_core::ops::Map::new_array(a, move |x| if x < min { min } else { x })
                            .into_py_result()?;
                    res.into_type_dyn().into_any()
                }
                (None, Some(max)) => {
                    let res =
                        jix_core::ops::Map::new_array(a, move |x| if x > max { max } else { x })
                            .into_py_result()?;
                    res.into_type_dyn().into_any()
                }
                (Some(min), Some(max)) => {
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if !(min <= max) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "min must be less than or equal to max",
                        ));
                    }
                    let res = jix_core::ops::Map::new_array(a, move |x| {
                        if x < min {
                            min
                        } else if x > max {
                            max
                        } else {
                            x
                        }
                    })
                    .into_py_result()?;
                    res.into_type_dyn().into_any()
                }
            })
        })
    }
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, ClampArgs>> = LazyLock::new(|| {
        OpDescriptor::new(
            "clamp",
            vec![
                clamp_op_descriptor::<bool>(),
                clamp_op_descriptor::<u8>(),
                clamp_op_descriptor::<i8>(),
                clamp_op_descriptor::<u16>(),
                clamp_op_descriptor::<i16>(),
                clamp_op_descriptor::<u32>(),
                clamp_op_descriptor::<i32>(),
                clamp_op_descriptor::<u64>(),
                clamp_op_descriptor::<i64>(),
                clamp_op_descriptor::<f16>(),
                clamp_op_descriptor::<f32>(),
                clamp_op_descriptor::<f64>(),
            ],
        )
    });
    let array = Operand::from_any(array)?;
    let args = ClampArgs {
        min: min.map(|m| Scalar::from_any(m)).transpose()?,
        max: max.map(|m| Scalar::from_any(m)).transpose()?,
    };
    let res = DISPATCH_TABLE.dispatch_args([array], args)?;
    Bound::new(py, crate::Array::from_core(res))
}

/// Element-wise approximate equality test, analogous to `numpy.isclose`.
///
/// Returns a `bool` array that is `True` at each position where the corresponding elements
/// of `a` and `b` are within tolerance:
/// `|a - b| <= max(atol, rtol * max(|a|, |b|))`
///
/// `NaN` inputs always produce `False` because `NaN != NaN`.
///
/// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
///
/// For complex arrays `rtol` must be a real scalar (applied per-component).
/// `atol` may be either a real scalar (the same absolute tolerance is applied to both the
/// real and imaginary components) or a complex scalar whose real and imaginary parts set
/// independent per-component absolute tolerances.
///
/// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a common type
/// (Safe casting rules) before comparison.
///
/// **Broadcasting**: shapes are broadcast to a common shape following numpy rules.
///
/// Args:
///     a: First operand.
///     b: Second operand.
///     rtol: Relative tolerance. A real Python or NumPy scalar.
///     atol: Absolute tolerance. A real Python or NumPy scalar, or - for complex arrays only -
///         a complex scalar whose real/imaginary parts set the per-component tolerances.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the broadcast shape. No computation
///         occurs until the result is read.
///
/// Raises:
///     TypeError: If `rtol` is complex, or if `atol` is complex for a non-complex array dtype.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.0, 2.0, 3.0], dtype=np.float32)
///     b = jix.compact([1.0, 2.01, 4.0], dtype=np.float32)
///     result = jix.isclose(a, b, rtol=0.0, atol=0.02)
///     assert np.array_equal(result.numpy(), [True, True, False])
///
///     # NaN inputs always produce False.
///     c = jix.compact([float('nan'), 1.0], dtype=np.float32)
///     d = jix.compact([float('nan'), 1.0], dtype=np.float32)
///     result = jix.isclose(c, d, rtol=0.0, atol=0.0)
///     assert np.array_equal(result.numpy(), [False, True])
///
///     # Same infinities are equal; opposite signs are not.
///     e = jix.compact([float('inf'), float('-inf')], dtype=np.float32)
///     f = jix.compact([float('inf'), float('inf')], dtype=np.float32)
///     result = jix.isclose(e, f, rtol=0.0, atol=0.0)
///     assert np.array_equal(result.numpy(), [True, False])
///
///     # Complex array: scalar atol applies to both components; complex atol splits them.
///     g = jix.compact([1.0 + 2.0j], dtype=np.complex64)
///     h = jix.compact([1.01 + 2.1j], dtype=np.complex64)
///     result = jix.isclose(g, h, rtol=0.0, atol=complex(0.02, 0.2))
///     assert np.array_equal(result.numpy(), [True])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyo3::pyfunction(signature = (a, b, rtol, atol))]
pub fn isclose<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rtol: &Bound<'py, PyAny>,
    atol: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, crate::Array>> {
    let py = a.py();

    struct ApproxEqualArgs {
        rtol: Scalar,
        atol: Scalar,
    }

    fn cast_tol<T>(scalar: Scalar) -> PyResult<T>
    where
        bool: jix_core::scalar::Cast<T>,
        u64: jix_core::scalar::Cast<T>,
        i64: jix_core::scalar::Cast<T>,
        f64: jix_core::scalar::Cast<T>,
    {
        Ok(match scalar {
            Scalar::Bool(v) => <bool as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::UInt(v) => <u64 as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::Int(v) => <i64 as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::Float(v) => <f64 as jix_core::scalar::Cast<T>>::cast(v),
            Scalar::Complex(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "a complex can't be cast to a real type for rtol or atol",
                ))
            }
        })
    }

    fn float_descriptor<T>() -> OpFnDescriptor<2, ApproxEqualArgs>
    where
        T: Dtyped + jix_core::scalar::ApproxEq<AbsoluteTolerance = T, RelativeTolerance = T>,
        bool: jix_core::scalar::Cast<T>,
        u64: jix_core::scalar::Cast<T>,
        i64: jix_core::scalar::Cast<T>,
        f64: jix_core::scalar::Cast<T>,
    {
        OpFnDescriptor::new2_args::<T>(CastKind::Safe, |a, b, args: ApproxEqualArgs| {
            let rtol = cast_tol::<T>(args.rtol)?;
            let atol = cast_tol::<T>(args.atol)?;
            let res = jix_core::ops::ApproxEq::new_array(a, b, rtol, atol).into_py_result()?;
            Ok(res.into_type_dyn().into_any())
        })
    }

    fn complex_descriptor<F>() -> OpFnDescriptor<2, ApproxEqualArgs>
    where
        F: Copy + Send + Sync + 'static,
        Complex<F>: Dtyped
            + jix_core::scalar::ApproxEq<AbsoluteTolerance = Complex<F>, RelativeTolerance = F>,
        bool: jix_core::scalar::Cast<F>,
        u64: jix_core::scalar::Cast<F>,
        i64: jix_core::scalar::Cast<F>,
        f64: jix_core::scalar::Cast<F>,
    {
        OpFnDescriptor::new2_args::<Complex<F>>(CastKind::Safe, |a, b, args: ApproxEqualArgs| {
            let rtol = cast_tol::<F>(args.rtol)?;
            let atol = match args.atol {
                Scalar::Bool(_) | Scalar::UInt(_) | Scalar::Int(_) | Scalar::Float(_) => {
                    let atol = cast_tol::<F>(args.atol)?;
                    Complex { re: atol, im: atol }
                }
                Scalar::Complex(complex) => Complex {
                    re: <f64 as jix_core::scalar::Cast<F>>::cast(complex.re),
                    im: <f64 as jix_core::scalar::Cast<F>>::cast(complex.im),
                },
            };
            let res = jix_core::ops::ApproxEq::new_array(a, b, rtol, atol).into_py_result()?;
            Ok(res.into_type_dyn().into_any())
        })
    }

    static DISPATCH_TABLE: LazyLock<OpDescriptor<2, ApproxEqualArgs>> = LazyLock::new(|| {
        OpDescriptor::new(
            "isclose",
            vec![
                float_descriptor::<f16>(),
                float_descriptor::<f32>(),
                float_descriptor::<f64>(),
                complex_descriptor::<f32>(),
                complex_descriptor::<f64>(),
            ],
        )
    });

    let rtol_scalar = Scalar::from_any(rtol)?;
    let atol_scalar = Scalar::from_any(atol)?;
    let a = Operand::from_any(a)?;
    let b = Operand::from_any(b)?;
    let [a, b] = broadcast_operands([a, b])?;
    let res = DISPATCH_TABLE.dispatch_args(
        [a, b],
        ApproxEqualArgs {
            rtol: rtol_scalar,
            atol: atol_scalar,
        },
    )?;
    Bound::new(py, crate::Array::from_core(res))
}
