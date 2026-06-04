use jix_core::scalar::f16;

use crate::ops::common::define_op1;

define_op1!(
    /// Tests whether each element is `NaN` (not a number).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `True` if the element is `NaN`, `False` otherwise.
    ///
    /// The `array` argument may be anything that `jix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([float('nan'), 1.0, float('inf'), -1.0], dtype=np.float32)
    /// result = jix.is_nan(a)
    /// assert np.array_equal(result.numpy(), [True, False, False, False])
    /// ```
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
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `True` if the element is a finite number, `False` for `+/-inf` and `NaN`.
    ///
    /// The `array` argument may be anything that `jix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float32)
    /// result = jix.is_finite(a)
    /// assert np.array_equal(result.numpy(), [True, False, False, False])
    /// ```
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
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `True` only for `+inf` and `-inf`; returns `False` for finite values and `NaN`.
    ///
    /// The `array` argument may be anything that `jix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([float('inf'), float('-inf'), float('nan'), 1.0], dtype=np.float32)
    /// result = jix.is_infinite(a)
    /// assert np.array_equal(result.numpy(), [True, True, False, False])
    /// ```
    is_infinite,
    IsInfinite,
    dispatch = {
        [f16, f32, f64],
        None
    }
);
