use std::sync::LazyLock;

use jix_core::scalar::Complex;
use pyo3::prelude::*;

use crate::ops::common::{define_op1, define_op1_desc, OpDescriptor, Operand, Rank};

/// Extracts the real part of each complex element.
///
/// Supported dtypes: `Complex<f32>`, `Complex<f64>`. Output dtype is the
/// corresponding real component type (`f32` for `Complex<f32>`, `f64` for
/// `Complex<f64>`). An array of any other dtype is returned as is.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`, holding the
///         real components.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64)
///     result = jix.real(a)
///     assert result.dtype == np.float32
///     assert np.array_equal(result.numpy(), [1.0, 3.0, -5.0])
///     ```
#[pyo3::pyfunction]
pub fn real<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, crate::Array>> {
    let py = array.py();
    let array = Operand::from_any(array)?;
    #[allow(clippy::collapsible_if)]
    if let Some((rank, _precision)) = array.rank_precision() {
        if rank != Rank::Complex {
            return array.into_py_array(py);
        }
    }

    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "real",
            define_op1_desc!(Real, [Complex<f32>, Complex<f64>], None),
        )
    });
    Bound::new(
        py,
        crate::Array::from_core(DISPATCH_TABLE.dispatch1(array)?),
    )
}

define_op1!(
    /// Extracts the imaginary part of each complex element.
    ///
    /// Supported dtypes: `Complex<f32>`, `Complex<f64>`. Output dtype is the
    /// corresponding real component type (`f32` for `Complex<f32>`, `f64` for
    /// `Complex<f64>`).
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// Args:
    ///     array: Input array.
    ///
    /// Returns:
    ///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`, holding the
    ///         imaginary components.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64)
    ///     result = jix.imag(a)
    ///     assert result.dtype == np.float32
    ///     assert np.array_equal(result.numpy(), [2.0, -4.0, 6.0])
    ///     ```
    imag,
    Imaginary,
    dispatch = {
        [Complex<f32>, Complex<f64>],
        None
    }
);
