use jix_core::scalar::{f16, Complex};

use crate::ops::common::define_op2;

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    /// Output dtype is `bool`.
    ///
    /// For **float** types, `NaN != NaN` per IEEE 754: comparing two `NaN` values returns
    /// `False`. For **complex** types, both the real and imaginary components must be equal.
    ///
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 2, 3], dtype=np.int32)
    /// b = jix.compact([1, 0, 3], dtype=np.int32)
    /// result = jix.equal(a, b)
    /// assert np.array_equal(result.numpy(), [True, False, True])
    ///
    /// # NaN != NaN per IEEE 754.
    /// c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    /// d = jix.compact([float('nan'), 1.0], dtype=np.float32)
    /// result = jix.equal(c, d)
    /// assert np.array_equal(result.numpy(), [False, True])
    /// ```
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
    /// Output dtype is `bool`.
    ///
    /// For **float** types, `NaN != NaN` returns `True` per IEEE 754.
    /// For **complex** types, returns `True` if either the real or imaginary component differs.
    ///
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 2, 3], dtype=np.int32)
    /// b = jix.compact([1, 0, 3], dtype=np.int32)
    /// result = jix.not_equal(a, b)
    /// assert np.array_equal(result.numpy(), [False, True, False])
    /// ```
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
    /// Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True > False`.
    ///
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([3, 1, 2], dtype=np.int32)
    /// b = jix.compact([1, 1, 3], dtype=np.int32)
    /// result = jix.greater(a, b)
    /// assert np.array_equal(result.numpy(), [True, False, False])
    /// ```
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
    /// Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True >= False`, and both `True >= True` and `False >= False` hold.
    ///
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([3, 1, 2], dtype=np.int32)
    /// b = jix.compact([1, 1, 3], dtype=np.int32)
    /// result = jix.greater_equal(a, b)
    /// assert np.array_equal(result.numpy(), [True, True, False])
    /// ```
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
    /// Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False < True`.
    ///
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 1, 3], dtype=np.int32)
    /// b = jix.compact([3, 1, 2], dtype=np.int32)
    /// result = jix.less(a, b)
    /// assert np.array_equal(result.numpy(), [True, False, False])
    /// ```
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
    /// Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False <= True`, and both `False <= False` and `True <= True` hold.
    ///
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules) before comparison.
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 1, 3], dtype=np.int32)
    /// b = jix.compact([3, 1, 2], dtype=np.int32)
    /// result = jix.less_equal(a, b)
    /// assert np.array_equal(result.numpy(), [True, True, False])
    /// ```
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
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 5, 3], dtype=np.int32)
    /// b = jix.compact([4, 2, 3], dtype=np.int32)
    /// result = jix.maximum(a, b)
    /// assert np.array_equal(result.numpy(), [4, 5, 3])
    ///
    /// # NaN is propagated: if either operand is NaN the result is NaN.
    /// c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    /// d = jix.compact([2.0, 3.0], dtype=np.float32)
    /// result = jix.maximum(c, d)
    /// assert np.isnan(result.numpy()[0])
    /// assert result.numpy()[1] == 3.0
    /// ```
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
    /// Both `a` and `b` may be anything that `jix.asarray()` accepts.
    ///
    /// **Type promotion**: if `a` and `b` have different dtypes, both are cast to a
    /// common type (Safe casting rules).
    ///
    /// **Broadcasting**: shapes are broadcast to a common shape following numpy rules
    /// exactly.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 5, 3], dtype=np.int32)
    /// b = jix.compact([4, 2, 3], dtype=np.int32)
    /// result = jix.minimum(a, b)
    /// assert np.array_equal(result.numpy(), [1, 2, 3])
    ///
    /// # NaN is propagated: if either operand is NaN the result is NaN.
    /// c = jix.compact([float('nan'), 1.0], dtype=np.float32)
    /// d = jix.compact([2.0, 3.0], dtype=np.float32)
    /// result = jix.minimum(c, d)
    /// assert np.isnan(result.numpy()[0])
    /// assert result.numpy()[1] == 1.0
    /// ```
    minimum,
    Minimum,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, bool],
        Safe
    }
);
