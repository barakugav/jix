//! NumPy-style handling of the arithmetic errors element-wise operations hit while computing:
//! [`seterr`], [`geterr`] and the [`ErrState`] (`errstate`) context manager.
//!
//! The modes are per thread, and each read applies the calling thread's modes to its read
//! context, see [`configure_context`].

use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::CString;

use jix_core::ReadContext;

use jix_core::error::{ArithError, ArithErrorMode};
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;

use crate::type_hint::ErrMode;

#[derive(Clone, Copy)]
struct Modes {
    divide: ArithErrorMode,
}
impl Modes {
    /// NumPy's defaults.
    const DEFAULT: Self = Self {
        divide: ArithErrorMode::Warn,
    };

    fn to_dict(self) -> BTreeMap<&'static str, &'static str> {
        BTreeMap::from([("divide", mode_name(self.divide))])
    }
}

thread_local! {
    static MODES: Cell<Modes> = const { Cell::new(Modes::DEFAULT) };
}

/// Applies the calling thread's modes to `context`, and drops any warnings a failed read left in
/// it.
pub(crate) fn configure_context(context: &mut ReadContext) {
    let modes = MODES.with(Cell::get);
    context.set_arith_error_mode(ArithError::DivideByZero, modes.divide);
    drop(context.take_warnings());
}

/// Emits the warnings of a read as `RuntimeWarning`s.
pub(crate) fn emit_warnings(py: Python<'_>, warnings: Vec<jix_core::Error>) -> PyResult<()> {
    for warning in warnings {
        let message = CString::new(warning.message())?;
        PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), &message, 1)?;
    }
    Ok(())
}

fn mode_name(mode: ArithErrorMode) -> &'static str {
    match mode {
        ArithErrorMode::Ignore => "ignore",
        ArithErrorMode::Warn => "warn",
        ArithErrorMode::Raise => "raise",
    }
}

fn apply(modes: Modes, all: Option<ArithErrorMode>, divide: Option<ArithErrorMode>) -> Modes {
    Modes {
        divide: divide.or(all).unwrap_or(modes.divide),
    }
}

/// Set how element-wise operations treat arithmetic errors, like `numpy.seterr`.
///
/// An operation that hits an error still produces a value for the element - `0` for an integer
/// division by zero - and the mode decides what else happens when the array is read:
/// `'ignore'` does nothing, `'warn'` emits a `RuntimeWarning` and `'raise'` raises a
/// `FloatingPointError`. The modes are per thread, and apply to the reads made on the thread,
/// not to where the (lazy) operations were built.
///
/// Supported error kinds: `divide` (integer division by zero, defaults to `'warn'`).
///
/// Args:
///     all: Mode for every error kind. A kind given explicitly takes precedence.
///     divide: Mode for integer division by zero.
///
/// Returns:
///     The previous modes, as a dict that can be passed back to `seterr(**old)`.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact(np.array([6, 7], dtype=np.int32))
///     b = jix.compact(np.array([2, 0], dtype=np.int32))
///     old = jix.seterr(divide="ignore")
///     assert np.array_equal((a // b).numpy(), [3, 0])
///     jix.seterr(**old)
///     ```
#[pyfunction]
#[pyo3(signature = (*, all=None, divide=None))]
pub fn seterr(
    all: Option<ErrMode>,
    divide: Option<ErrMode>,
) -> BTreeMap<&'static str, &'static str> {
    let old = MODES.with(Cell::get);
    MODES.with(|modes| modes.set(apply(old, all.map(|m| m.0), divide.map(|m| m.0))));
    old.to_dict()
}

/// Get how element-wise operations treat arithmetic errors, like `numpy.geterr`.
///
/// See [`jix.seterr()`][jix.seterr].
///
/// Returns:
///     A dict from error kind to mode.
#[pyfunction]
pub fn geterr() -> BTreeMap<&'static str, &'static str> {
    MODES.with(Cell::get).to_dict()
}

/// Context manager that sets how element-wise operations treat arithmetic errors, like
/// `numpy.errstate`.
///
/// Sets the modes on entry and restores the previous ones on exit. See
/// [`jix.seterr()`][jix.seterr] for the modes and error kinds.
///
/// Note that operations are lazy: the modes in effect when an array is *read* (e.g. by
/// `.numpy()`) are the ones that apply.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///     import pytest
///
///     a = jix.compact(np.array([6, 7], dtype=np.int32))
///     b = jix.compact(np.array([2, 0], dtype=np.int32))
///     with jix.errstate(divide="raise"):
///         with pytest.raises(FloatingPointError):
///             (a // b).numpy()
///     ```
#[pyclass(module = "jix", name = "errstate")]
pub struct ErrState {
    all: Option<ArithErrorMode>,
    divide: Option<ArithErrorMode>,
    old: Vec<Modes>,
}

#[pymethods]
impl ErrState {
    #[new]
    #[pyo3(signature = (*, all=None, divide=None))]
    fn new(all: Option<ErrMode>, divide: Option<ErrMode>) -> Self {
        Self {
            all: all.map(|m| m.0),
            divide: divide.map(|m| m.0),
            old: Vec::new(),
        }
    }

    fn __enter__(&mut self) {
        let old = MODES.with(Cell::get);
        MODES.with(|modes| modes.set(apply(old, self.all, self.divide)));
        self.old.push(old);
    }

    fn __exit__(
        &mut self,
        exc_type: &Bound<'_, PyAny>,
        exc_value: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) {
        let _ = (exc_type, exc_value, traceback);
        if let Some(old) = self.old.pop() {
            MODES.with(|modes| modes.set(old));
        }
    }
}
