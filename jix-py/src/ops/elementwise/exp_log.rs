use std::sync::LazyLock;

use jix_core::dtype::Dtyped;
use jix_core::scalar::f16;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ops::common::{define_op1, CastKind, OpDescriptor, OpFnDescriptor, Operand};
use crate::util::IntoPyResult;
use crate::Array;

define_op1!(
    /// Computes the square root of each element.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`; integers and `bool` are cast to a float first.
    ///
    /// A negative input produces `NaN`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([4.0, 9.0, 16.0], dtype=np.float32)
    ///     result = jix.sqrt(a)
    ///     assert np.array_equal(result.numpy(), [2.0, 3.0, 4.0])
    ///
    ///     # A negative input produces NaN.
    ///     b = jix.compact([-1.0], dtype=np.float32)
    ///     assert np.isnan(jix.sqrt(b).numpy()[0])
    ///     ```
    sqrt,
    Sqrt,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`; integers and `bool` are cast to a float first.
    ///
    /// A result too large for the dtype is `inf`.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([0.0, 1.0], dtype=np.float32)
    ///     result = jix.exp(a)
    ///     assert result.numpy()[0] == 1.0
    ///     assert abs(result.numpy()[1] - np.e) < 1e-5
    ///     ```
    exp,
    Exp,
    dispatch = {
        [f16, f32, f64],
        Safe
    }
);

/// Computes the logarithm of each element: `ln x` by default, `ln(x) / ln(base)` with `base`.
///
/// Supported dtypes: `f16`, `f32`, `f64`; integers and `bool` are cast to a float first.
///
/// Zero produces `-inf` and a negative input `NaN`.
///
/// Args:
///     array: Input array.
///     base: Logarithm base, a positive number other than 1. `None` (the default) means `e`.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # Natural logarithm (default).
///     a = jix.compact([1.0, np.e], dtype=np.float32)
///     result = jix.log(a)
///     assert abs(result.numpy()[0]) < 1e-5         # ln(1) = 0
///     assert abs(result.numpy()[1] - 1.0) < 1e-5   # ln(e) = 1
///
///     # Base-2 logarithm.
///     c = jix.compact([1.0, 2.0, 8.0], dtype=np.float32)
///     result_c = jix.log(c, base=2)
///     assert abs(result_c.numpy()[0]) < 1e-5        # log2(1) = 0
///     assert abs(result_c.numpy()[1] - 1.0) < 1e-5  # log2(2) = 1
///     assert abs(result_c.numpy()[2] - 3.0) < 1e-5  # log2(8) = 3
///
///     # Base-10 logarithm.
///     d = jix.compact([1.0, 10.0, 100.0], dtype=np.float64)
///     result_d = jix.log(d, base=10)
///     assert abs(result_d.numpy()[0]) < 1e-10       # log10(1) = 0
///     assert abs(result_d.numpy()[1] - 1.0) < 1e-10 # log10(10) = 1
///     assert abs(result_d.numpy()[2] - 2.0) < 1e-10 # log10(100) = 2
///
///     # Zero produces -inf and a negative input NaN.
///     b = jix.compact([0.0, -1.0], dtype=np.float32)
///     result_b = jix.log(b).numpy()
///     assert np.isneginf(result_b[0])
///     assert np.isnan(result_b[1])
///     ```
#[pyo3::pyfunction(signature = (array, base=None))]
pub fn log<'py>(array: &Bound<'py, PyAny>, base: Option<f64>) -> PyResult<Array> {
    struct LogArgs {
        base: f64,
    }
    fn log_op_descriptor<T>() -> OpFnDescriptor<1, LogArgs>
    where
        T: Dtyped + jix_core::scalar::Cast<f64>,
        T: jix_core::scalar::Ln<Output = T>
            + jix_core::scalar::Log2<Output = T>
            + jix_core::scalar::Log10<Output = T>,
        f64: jix_core::scalar::Cast<T>,
    {
        use jix_core::ops::{Cast, Ln, Log10, Log2};
        OpFnDescriptor::new1_args::<T>(CastKind::Safe, |a, args: LogArgs| {
            Ok(match args.base {
                core::f64::consts::E => Ln::new_array(a).into_py_result()?.into_any(),
                2.0 => Log2::new_array(a).into_py_result()?.into_any(),
                10.0 => Log10::new_array(a).into_py_result()?.into_any(),
                base => {
                    let ln_base = base.ln();
                    let a = Cast::<_, f64>::new_array(a).into_py_result()?;
                    let res = Ln::new_array(a).into_py_result()?;
                    let res = res
                        .map(move |x: f64| <f64 as jix_core::scalar::Cast<T>>::cast(x / ln_base));
                    res.into_any()
                }
            })
        })
    }
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, LogArgs>> = LazyLock::new(|| {
        OpDescriptor::new(
            "log",
            vec![
                log_op_descriptor::<f16>(),
                log_op_descriptor::<f32>(),
                log_op_descriptor::<f64>(),
            ],
        )
    });
    let base = base.unwrap_or(core::f64::consts::E);
    if base <= 0.0 || base == 1.0 {
        return Err(PyValueError::new_err(format!(
            "log base must be a positive number other than 1, got {base}"
        )));
    }
    let array = Operand::from_any(array)?;
    let args = LogArgs { base };
    let res = DISPATCH_TABLE.dispatch_args([array], args)?;
    Ok(Array::from_core(res))
}
