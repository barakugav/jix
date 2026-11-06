use jix_core::scalar::f16;

use crate::ops::common::define_op1;

define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// An infinite input produces `NaN`.
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
    ///     a = jix.compact([0.0, np.pi / 2], dtype=np.float32)
    ///     result = jix.sin(a)
    ///     assert abs(result.numpy()[0]) < 1e-5   # sin(0) = 0
    ///     assert abs(result.numpy()[1] - 1.0) < 1e-5  # sin(pi/2) = 1
    ///     ```
    sin,
    Sin,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// An infinite input produces `NaN`.
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
    ///     a = jix.compact([0.0, np.pi], dtype=np.float32)
    ///     result = jix.cos(a)
    ///     assert abs(result.numpy()[0] - 1.0) < 1e-5   # cos(0) = 1
    ///     assert abs(result.numpy()[1] - (-1.0)) < 1e-5  # cos(pi) = -1
    ///     ```
    cos,
    Cos,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// An infinite input produces `NaN`.
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
    ///     a = jix.compact([0.0, np.pi / 4], dtype=np.float32)
    ///     result = jix.tan(a)
    ///     assert abs(result.numpy()[0]) < 1e-5        # tan(0) = 0
    ///     assert abs(result.numpy()[1] - 1.0) < 1e-5  # tan(pi/4) = 1
    ///     ```
    tan,
    Tan,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// An input outside `[-1, 1]` produces `NaN`.
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
    ///     a = jix.compact([0.0, 1.0, -1.0], dtype=np.float32)
    ///     result = jix.asin(a)
    ///     assert abs(result.numpy()[0]) < 1e-5
    ///     assert abs(result.numpy()[1] - np.pi / 2) < 1e-5
    ///     ```
    asin,
    Asin,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, pi]`.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
    ///
    /// An input outside `[-1, 1]` produces `NaN`.
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
    ///     a = jix.compact([1.0, 0.0, -1.0], dtype=np.float32)
    ///     result = jix.acos(a)
    ///     assert abs(result.numpy()[0]) < 1e-5
    ///     assert abs(result.numpy()[1] - np.pi / 2) < 1e-5
    ///     assert abs(result.numpy()[2] - np.pi) < 1e-5
    ///     ```
    acos,
    Acos,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`.
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
    ///     a = jix.compact([0.0, 1.0], dtype=np.float32)
    ///     result = jix.atan(a)
    ///     assert abs(result.numpy()[0]) < 1e-5
    ///     assert abs(result.numpy()[1] - np.pi / 4) < 1e-5
    ///     ```
    atan,
    Atan,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);
