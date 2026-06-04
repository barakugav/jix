use jix_core::scalar::f16;

use crate::ops::common::define_op1;

define_op1!(
    /// Tests whether each element is `NaN` (not a number).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`. `True` where the
    ///     element is `NaN`, `False` otherwise.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([float('nan'), 1.0, float('inf'), -1.0], dtype=np.float32)
    ///     result = jix.is_nan(a)
    ///     assert np.array_equal(result.numpy(), [True, False, False, False])
    ///     ```
    is_nan,
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
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`. `True` for finite
    ///     values, `False` for `+/-inf` and `NaN`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float32)
    ///     result = jix.is_finite(a)
    ///     assert np.array_equal(result.numpy(), [True, False, False, False])
    ///     ```
    is_finite,
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
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`. `True` only for
    ///     `+inf` and `-inf`, `False` for finite values and `NaN`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([float('inf'), float('-inf'), float('nan'), 1.0], dtype=np.float32)
    ///     result = jix.is_infinite(a)
    ///     assert np.array_equal(result.numpy(), [True, True, False, False])
    ///     ```
    is_infinite,
    IsInfinite,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
