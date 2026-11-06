//! Thin wrappers whose only job is to give arguments and return values a precise type hint in the
//! generated `.pyi` stubs, for types that do not carry one themselves (e.g. the `numpy` crate's
//! types are all `Incomplete`).

use std::convert::Infallible;
use std::ops::Deref;

use numpy::{PyArrayDescr, PyUntypedArray};
use pyo3::exceptions::PyTypeError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt};
use pyo3::{type_hint_identifier, type_hint_subscript, PyClass, PyTypeInfo};

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

/// A dim argument. Unlike a plain `i32` it rejects a `bool`.
#[derive(Clone, Copy)]
pub struct Dim(pub i32);

impl<'a, 'py> FromPyObject<'a, 'py> for Dim {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = <PyInt as PyTypeInfo>::TYPE_HINT;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("a dim must be an int, not a bool"));
        }
        Ok(Self(obj.extract()?))
    }
}

/// The result of a binary dunder. `NotImplemented` makes Python try the other operand's reflected
/// op. Typed as an `Array`, like numpy's dunders.
pub enum ValueOrUnimplemented<T>
where
    T: PyClass + Into<PyClassInitializer<T>> + PyTypeInfo,
{
    Value(T),
    NotImplemented,
}

impl<'py, T> IntoPyObject<'py> for ValueOrUnimplemented<T>
where
    T: PyClass + Into<PyClassInitializer<T>> + PyTypeInfo,
{
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    const OUTPUT_TYPE: PyStaticExpr = <T as PyTypeInfo>::TYPE_HINT;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Self::Value(value) => Ok(Bound::new(py, value)?.into_any()),
            Self::NotImplemented => Ok(py.NotImplemented().into_bound(py)),
        }
    }
}
