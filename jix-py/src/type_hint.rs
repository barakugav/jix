//! Thin wrappers whose only job is to give arguments and return values a precise type hint in the
//! generated `.pyi` stubs, for types that do not carry one themselves (e.g. the `numpy` crate's
//! types are all `Incomplete`).

use std::convert::Infallible;
use std::ops::Deref;

use numpy::{PyArrayDescr, PyUntypedArray};
use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::prelude::*;
use pyo3::{type_hint_identifier, type_hint_subscript};

/// `numpy.typing.NDArray[typing.Any]`
const NDARRAY_TYPE_HINT: PyStaticExpr = type_hint_subscript!(
    type_hint_identifier!("numpy.typing", "NDArray"),
    type_hint_identifier!("typing", "Any")
);

/// A NumPy array of any dtype and shape.
#[derive(Debug)]
pub struct NdArray<'py>(pub Bound<'py, PyUntypedArray>);

impl<'py> Deref for NdArray<'py> {
    type Target = Bound<'py, PyUntypedArray>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for NdArray<'py> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = NDARRAY_TYPE_HINT;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract()?))
    }
}

impl<'py> IntoPyObject<'py> for NdArray<'py> {
    type Target = PyUntypedArray;
    type Output = Bound<'py, PyUntypedArray>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = NDARRAY_TYPE_HINT;

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// A NumPy dtype.
pub struct NpDtype<'py>(pub Bound<'py, PyArrayDescr>);

impl<'py> IntoPyObject<'py> for NpDtype<'py> {
    type Target = PyArrayDescr;
    type Output = Bound<'py, PyArrayDescr>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_identifier!("numpy", "dtype");

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// The `validation` mode of [`crate::archive::read_array`].
pub struct ArchiveValidation(pub jix_core::ArchiveValidation);

impl<'a, 'py> FromPyObject<'a, 'py> for ArchiveValidation {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("typing", "Literal"),
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("minimal")
        },
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("strict")
        }
    );

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match obj.extract::<&str>()? {
            "minimal" => Ok(Self(jix_core::ArchiveValidation::Minimal)),
            "strict" => Ok(Self(jix_core::ArchiveValidation::Strict)),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid validation value: '{other}'"
            ))),
        }
    }
}

/// An arithmetic error mode of [`crate::errstate::seterr`].
pub struct ErrMode(pub jix_core::error::ArithmeticErrorMode);

impl<'a, 'py> FromPyObject<'a, 'py> for ErrMode {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("typing", "Literal"),
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("ignore")
        },
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("warn")
        },
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("raise")
        }
    );

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match obj.extract::<&str>()? {
            "ignore" => Ok(Self(jix_core::error::ArithmeticErrorMode::Ignore)),
            "warn" => Ok(Self(jix_core::error::ArithmeticErrorMode::Warn)),
            "raise" => Ok(Self(jix_core::error::ArithmeticErrorMode::Raise)),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid error mode '{other}', expected one of 'ignore', 'warn', 'raise'"
            ))),
        }
    }
}
