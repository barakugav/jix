//! NumPy-style handling of the arithmetic errors element-wise operations hit while computing:
//! [`geterr`] and the [`errstate`] context manager.
//!
//! The modes are per thread, and each read applies the calling thread's modes to its read
//! context, see [`configure_context_error_modes`].

use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::CString;

use jix_core::ReadContext;

use jix_core::error::{ArithmeticErrorKind, ArithmeticErrorMode};
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;

use crate::type_hint::ErrMode;

thread_local! {
    static MODES: Cell<ArithmeticErrorModes> = const { Cell::new(ArithmeticErrorModes::DEFAULT) };
}

#[derive(Clone, Copy)]
pub(crate) struct ArithmeticErrorModes {
    divide_by_zero: ArithmeticErrorMode,
    overflow: ArithmeticErrorMode,
    underflow: ArithmeticErrorMode,
    invalid_operation: ArithmeticErrorMode,
}
impl ArithmeticErrorModes {
    const DEFAULT: Self = Self {
        divide_by_zero: ArithmeticErrorMode::Error,
        overflow: ArithmeticErrorMode::Warn,
        underflow: ArithmeticErrorMode::Warn,
        invalid_operation: ArithmeticErrorMode::Error,
    };
}
/// Sets how element-wise operations treat arithmetic errors, like `numpy.errstate`.
///
/// An operation that hits an error still produces a value for the element - `0` for an integer
/// division by zero - and the mode decides what else happens when the array is read: `'ignore'`
/// does nothing, `'warn'` emits a `RuntimeWarning` and `'raise'` raises a `FloatingPointError`.
///
/// The new modes take effect as soon as `errstate` is called, so it can be used as a plain
/// function call that changes the modes for good, or as a context manager with the `with`
/// statement, which restores the previous modes on exit.
///
/// The modes are per thread, and since operations are lazy, the modes in effect when the array
/// is *read* apply, not those in effect when the operation was built.
///
/// The defaults are `'raise'` for `divide` and `invalid`, and `'warn'` for `over` and `under`.
///
/// Args:
///     all: Mode for every error kind. A kind given explicitly takes precedence.
///     divide: Mode for division by zero.
///     over: Mode for overflow.
///     under: Mode for underflow.
///     invalid: Mode for invalid operations.
///
/// Returns:
///     A context manager that can be used with the `with` statement and restores the previous modes
///     on exit.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact(np.array([6, 7], dtype=np.int32))
///     b = jix.compact(np.array([2, 0], dtype=np.int32))
///
///     with jix.errstate(divide="ignore"):
///         assert np.array_equal((a // b).numpy(), [3, 0])
///
///     jix.errstate(divide="warn")  # changes the modes until changed again
///     ```
#[pyfunction]
#[pyo3(signature = (
    *, all=None, divide=None, over=None, under=None, invalid=None
))]
pub fn errstate<'py>(
    py: Python<'py>,
    all: Option<ErrMode>,
    divide: Option<ErrMode>,
    over: Option<ErrMode>,
    under: Option<ErrMode>,
    invalid: Option<ErrMode>,
) -> PyResult<Bound<'py, PyAny>> {
    let state = ErrState::new(all, divide, over, under, invalid);
    Bound::new(py, state).map(Bound::into_any)
}

/// Context manager that sets how element-wise operations treat arithmetic errors, like
/// `numpy.errstate`.
#[pyclass(module = "jix")]
struct ErrState {
    prev: Option<ArithmeticErrorModes>,
}
impl ErrState {
    fn new(
        all: Option<ErrMode>,
        divide: Option<ErrMode>,
        over: Option<ErrMode>,
        under: Option<ErrMode>,
        invalid: Option<ErrMode>,
    ) -> Self {
        let all = all.map(|m| m.0);
        let divide = divide.map(|m| m.0);
        let over = over.map(|m| m.0);
        let under = under.map(|m| m.0);
        let invalid = invalid.map(|m| m.0);

        let prev = MODES.with(|modes| {
            let prev = modes.get();

            modes.set(ArithmeticErrorModes {
                divide_by_zero: divide.or(all).unwrap_or(prev.divide_by_zero),
                overflow: over.or(all).unwrap_or(prev.overflow),
                underflow: under.or(all).unwrap_or(prev.underflow),
                invalid_operation: invalid.or(all).unwrap_or(prev.invalid_operation),
            });
            prev
        });
        Self { prev: Some(prev) }
    }
}
#[pymethods]
impl ErrState {
    fn __enter__(&mut self) {}

    fn __exit__(
        &mut self,
        exc_type: &Bound<'_, PyAny>,
        exc_value: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) {
        let _ = (exc_type, exc_value, traceback);
        if let Some(prev) = self.prev.take() {
            MODES.with(|modes| modes.set(prev));
        }
    }
}

/// Get how element-wise operations treat arithmetic errors, like `numpy.geterr`.
///
/// See [`jix.errstate()`][jix.errstate].
///
/// Returns:
///     A dict from error kind to mode.
#[pyfunction]
pub fn geterr() -> BTreeMap<&'static str, &'static str> {
    let modes = MODES.with(Cell::get);
    let options = [
        ("divide", modes.divide_by_zero),
        ("over", modes.overflow),
        ("under", modes.underflow),
        ("invalid", modes.invalid_operation),
    ];

    options
        .into_iter()
        .map(|(key, mode)| {
            let mode_str = match mode {
                ArithmeticErrorMode::Ignore => "ignore",
                ArithmeticErrorMode::Warn => "warn",
                ArithmeticErrorMode::Error => "raise",
            };
            (key, mode_str)
        })
        .collect()
}

pub(crate) fn configure_context_error_modes(context: &mut ReadContext) {
    let modes = MODES.with(Cell::get);
    let kind_modes = [
        (ArithmeticErrorKind::DivideByZero, modes.divide_by_zero),
        (ArithmeticErrorKind::Overflow, modes.overflow),
        (ArithmeticErrorKind::Underflow, modes.underflow),
        (
            ArithmeticErrorKind::InvalidOperation,
            modes.invalid_operation,
        ),
    ];
    for (kind, mode) in kind_modes {
        context.set_arithmetic_error_mode(kind, mode);
    }
    drop(context.take_warnings());
}

pub(crate) fn emit_warnings(py: Python<'_>, warnings: Vec<jix_core::Error>) -> PyResult<()> {
    for warning in warnings {
        let message = CString::new(warning.message())?;
        PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), &message, 1)?;
    }
    Ok(())
}
