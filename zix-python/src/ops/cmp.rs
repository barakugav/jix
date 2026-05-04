use crate::ops::common::define_op2;

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, `NaN != NaN` per IEEE 754: comparing two `NaN` values returns
    /// `False`. For **complex** types, both the real and imaginary components must be equal.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 2, 3], dtype=np.int32)
    /// b = zix.compact([1, 0, 3], dtype=np.int32)
    /// result = zix.equal(a, b)
    /// assert np.array_equal(result.numpy(), [True, False, True])
    ///
    /// # NaN != NaN per IEEE 754.
    /// c = zix.compact([float('nan'), 1.0], dtype=np.float32)
    /// d = zix.compact([float('nan'), 1.0], dtype=np.float32)
    /// result = zix.equal(c, d)
    /// assert np.array_equal(result.numpy(), [False, True])
    /// ```
    equal,
    Equal
);

define_op2!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `bool`.
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, `NaN != NaN` returns `True` per IEEE 754.
    /// For **complex** types, returns `True` if either the real or imaginary component differs.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 2, 3], dtype=np.int32)
    /// b = zix.compact([1, 0, 3], dtype=np.int32)
    /// result = zix.not_equal(a, b)
    /// assert np.array_equal(result.numpy(), [False, True, False])
    /// ```
    not_equal,
    NotEqual
);

define_op2!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True > False`.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([3, 1, 2], dtype=np.int32)
    /// b = zix.compact([1, 1, 3], dtype=np.int32)
    /// result = zix.greater(a, b)
    /// assert np.array_equal(result.numpy(), [True, False, False])
    /// ```
    greater,
    Greater
);

define_op2!(
    /// Element-wise greater-than-or-equal test (`a >= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `True >= False`, and both `True >= True` and `False >= False` hold.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([3, 1, 2], dtype=np.int32)
    /// b = zix.compact([1, 1, 3], dtype=np.int32)
    /// result = zix.greater_equal(a, b)
    /// assert np.array_equal(result.numpy(), [True, True, False])
    /// ```
    greater_equal,
    GreaterEqual
);

define_op2!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False < True`.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 1, 3], dtype=np.int32)
    /// b = zix.compact([3, 1, 2], dtype=np.int32)
    /// result = zix.less(a, b)
    /// assert np.array_equal(result.numpy(), [True, False, False])
    /// ```
    less,
    Less
);

define_op2!(
    /// Element-wise less-than-or-equal test (`a <= b`).
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Complex types are not supported (no total ordering).
    /// Output dtype is `bool`. The output shape equals the input shape.
    ///
    /// For **float** types, any comparison involving `NaN` returns `False` (IEEE 754).
    /// For **bool**: `False <= True`, and both `False <= False` and `True <= True` hold.
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 1, 3], dtype=np.int32)
    /// b = zix.compact([3, 1, 2], dtype=np.int32)
    /// result = zix.less_equal(a, b)
    /// assert np.array_equal(result.numpy(), [True, True, False])
    /// ```
    less_equal,
    LessEqual
);

define_op2!(
    /// Element-wise maximum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype and shape equal the input.
    ///
    /// For **integer** and **bool** types the result is `max(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This matches `numpy.maximum` (not `numpy.fmax`, which ignores NaN).
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 5, 3], dtype=np.int32)
    /// b = zix.compact([4, 2, 3], dtype=np.int32)
    /// result = zix.maximum(a, b)
    /// assert np.array_equal(result.numpy(), [4, 5, 3])
    ///
    /// # NaN is propagated: if either operand is NaN the result is NaN.
    /// c = zix.compact([float('nan'), 1.0], dtype=np.float32)
    /// d = zix.compact([2.0, 3.0], dtype=np.float32)
    /// result = zix.maximum(c, d)
    /// assert np.isnan(result.numpy()[0])
    /// assert result.numpy()[1] == 3.0
    /// ```
    maximum,
    Maximum
);

define_op2!(
    /// Element-wise minimum of two arrays.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype and shape equal the input.
    ///
    /// For **integer** and **bool** types the result is `min(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This matches `numpy.minimum` (not `numpy.fmin`, which ignores NaN).
    ///
    /// Both `a` and `b` may be anything that `zix.asarray()` accepts; a Python scalar
    /// is broadcast to the other operand's shape.
    ///
    /// This function deviates from numpy in a few ways:
    /// - both inputs must have the same dtype (numpy will upcast if they differ)
    /// - if both inputs are arrays they must have the same shape; a scalar operand is
    ///   broadcast to match (numpy broadcasts any pair of shapes)
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 5, 3], dtype=np.int32)
    /// b = zix.compact([4, 2, 3], dtype=np.int32)
    /// result = zix.minimum(a, b)
    /// assert np.array_equal(result.numpy(), [1, 2, 3])
    ///
    /// # NaN is propagated: if either operand is NaN the result is NaN.
    /// c = zix.compact([float('nan'), 1.0], dtype=np.float32)
    /// d = zix.compact([2.0, 3.0], dtype=np.float32)
    /// result = zix.minimum(c, d)
    /// assert np.isnan(result.numpy()[0])
    /// assert result.numpy()[1] == 1.0
    /// ```
    minimum,
    Minimum
);
