use jix_core::scalar::f16;

use crate::ops::common::define_op1;

define_op1!(
    /// Tests whether each element is `NaN` (not a number).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([float('nan'), 1.0, float('inf'), -1.0], dtype=np.float32)
    ///     result = jix.isnan(a)
    ///     assert np.array_equal(result.numpy(), [True, False, False, False])
    ///     ```
    isnan,
    IsNan,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
define_op1!(
    /// Tests whether each element is finite (not `+/-inf` and not `NaN`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float32)
    ///     result = jix.isfinite(a)
    ///     assert np.array_equal(result.numpy(), [True, False, False, False])
    ///     ```
    isfinite,
    IsFinite,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
define_op1!(
    /// Tests whether each element is infinite (`+inf` or `-inf`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([float('inf'), float('-inf'), float('nan'), 1.0], dtype=np.float32)
    ///     result = jix.isinf(a)
    ///     assert np.array_equal(result.numpy(), [True, True, False, False])
    ///     ```
    isinf,
    IsInfinite,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
