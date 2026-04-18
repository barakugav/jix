mod as_array;
pub use as_array::*;

use pyo3::prelude::*;

use crate::array::Array;
use crate::util::IntoPyResult;

#[pyfunction]
pub fn add<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Array> {
    let (a, b) = (as_core_array(py, a)?, as_core_array(py, b)?);
    let ret = zix_core::ops::Add::new(a, b).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyfunction]
pub fn sub<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Array> {
    let (a, b) = (as_core_array(py, a)?, as_core_array(py, b)?);
    let ret = zix_core::ops::Sub::new(a, b).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyfunction]
pub fn mul<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Array> {
    let (a, b) = (as_core_array(py, a)?, as_core_array(py, b)?);
    let ret = zix_core::ops::Mul::new(a, b).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyfunction]
pub fn div<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Array> {
    let (a, b) = (as_core_array(py, a)?, as_core_array(py, b)?);
    let ret = zix_core::ops::Div::new(a, b).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
