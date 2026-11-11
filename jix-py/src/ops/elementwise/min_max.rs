use std::sync::LazyLock;

use jix_core::dtype::Dtyped;
use jix_core::scalar::bool8;
use jix_core::scalar::f16;
use pyo3::prelude::*;

use crate::ops::asarray_simple;
use crate::ops::common::{
    broadcast_operands, define_op2, CastKind, OpDescriptor, OpFnDescriptor, Operand,
};
use crate::util::IntoPyResult;
use crate::Array;

define_op2!(
    /// Element-wise maximum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`.
    ///
    /// For floats, a `NaN` in either operand gives `NaN`, as in `numpy.maximum` (not
    /// `numpy.fmax`, which ignores NaN).
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common dtype (Safe casting
    /// rules), which is the output dtype. **Broadcasting**: shapes broadcast following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] with the broadcast shape.
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
        [bool8, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);

define_op2!(
    /// Element-wise minimum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`.
    ///
    /// For floats, a `NaN` in either operand gives `NaN`, as in `numpy.minimum` (not
    /// `numpy.fmin`, which ignores NaN).
    ///
    /// **Type promotion**: operands of different dtypes are cast to a common dtype (Safe casting
    /// rules), which is the output dtype. **Broadcasting**: shapes broadcast following numpy rules.
    ///
    /// Args:
    ///     a: First operand.
    ///     b: Second operand.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] with the broadcast shape.
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
        [bool8, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        Safe
    }
);

/// Clamps each element to the range `[min, max]`.
///
/// Computed as `minimum(maximum(array, min), max)`, like `numpy.clip` and `torch.clamp`: where
/// `min > max` the result is `max`. The bounds broadcast with `array`. With only one bound this
/// is [`jix.maximum()`][jix.maximum] or [`jix.minimum()`][jix.minimum], and with neither
/// `array` is returned unchanged. For float dtypes a `NaN` in any operand produces `NaN`.
///
/// Supported dtypes: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
/// `f16`, `f32`, `f64`.
///
/// **Type promotion**: the operands are cast to a common dtype under Safe casting rules, as in
/// [`jix.maximum()`][jix.maximum]. For example, clamping an `int32` array to float bounds gives
/// `float64`.
///
/// Args:
///     array: Input array.
///     min: Lower bound (inclusive), a scalar or an array. `None` (default) applies no lower
///         bound.
///     max: Upper bound (inclusive), a scalar or an array. `None` (default) applies no upper
///         bound.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the broadcast shape and the promoted dtype.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([-3, 0, 5, 10], dtype=np.int32)
///
///     # Clamp to [0, 7].
///     assert np.array_equal(jix.clamp(a, min=0, max=7).numpy(), [0, 0, 5, 7])
///
///     # Only a lower bound.
///     assert np.array_equal(jix.clamp(a, min=2).numpy(), [2, 2, 5, 10])
///
///     # Per-element bounds.
///     lo = jix.compact([0, 1, 6, 0], dtype=np.int32)
///     assert np.array_equal(jix.clamp(a, min=lo, max=7).numpy(), [0, 1, 6, 7])
///
///     # min > max gives max.
///     assert np.array_equal(jix.clamp(a, min=5, max=2).numpy(), [2, 2, 2, 2])
///     ```
#[pyo3::pyfunction(signature = (array, min=None, max=None))]
pub fn clamp<'py>(
    array: &Bound<'py, PyAny>,
    min: Option<&Bound<'py, PyAny>>,
    max: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let (min, max) = match (min, max) {
        (None, None) => return asarray_simple(array),
        (Some(min), None) => return Bound::new(py, maximum(array, min)?),
        (None, Some(max)) => return Bound::new(py, minimum(array, max)?),
        (Some(min), Some(max)) => (min, max),
    };

    fn clamp_op_descriptor<T>() -> OpFnDescriptor<3, ()>
    where
        T: Dtyped + jix_core::scalar::Maximum<Output = T> + jix_core::scalar::Minimum<Output = T>,
    {
        OpFnDescriptor::new3::<T>(CastKind::Safe, |a, min, max| {
            let res = jix_core::ops::Clamp::new_array(a, min, max).into_py_result()?;
            Ok(res.into_any())
        })
    }
    static DISPATCH_TABLE: LazyLock<OpDescriptor<3, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "clamp",
            vec![
                clamp_op_descriptor::<bool8>(),
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
    let [array, min, max] = broadcast_operands([
        Operand::from_any(array)?,
        Operand::from_any(min)?,
        Operand::from_any(max)?,
    ])?;
    let res = DISPATCH_TABLE.dispatch3(array, min, max)?;
    Bound::new(py, Array::from_core(res))
}
