use pyo3::prelude::*;
use zix_core::Array as ZixArray;

use crate::ops::as_core_array;
use crate::util::IntoPyResult;
use crate::Array;

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    new_shape,
    copy=true,
))]
pub fn broadcast<'py>(
    array: &Bound<'py, Array>,
    new_shape: Vec<u64>,
    copy: bool,
) -> PyResult<Array> {
    let py = array.py();
    let array = array.borrow().to_core_array();
    let ret = zix_core::ops::Broadcast::new(array, &new_shape).into_py_result()?;
    if !copy {
        return Ok(Array::from_core_storage(ret));
    }
    // release the GIL while copying the data
    py.detach(|| {
        let ret = ZixArray::from_storage(ret).data().copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })
}

// TODO slice

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn insert_axes<'py>(array: &Bound<'py, Array>, axes: Vec<usize>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let ret = zix_core::ops::InsertAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn remove_axes<'py>(array: &Bound<'py, Array>, axes: Vec<usize>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let ret = zix_core::ops::RemoveAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn permute_axes<'py>(array: &Bound<'py, Array>, axes: Vec<usize>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let ret = zix_core::ops::PermuteAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    new_shape,
    copy=true,
))]
pub fn reshape<'py>(array: &Bound<'py, Array>, new_shape: Vec<u64>, copy: bool) -> PyResult<Array> {
    let py = array.py();
    let array = array.borrow().to_core_array();
    let ret = zix_core::ops::Reshape::new(array, &new_shape).into_py_result()?;
    if !copy {
        return Ok(Array::from_core_storage(ret));
    }
    // release the GIL while copying the data
    py.detach(|| {
        let ret = ZixArray::from_storage(ret).data().copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn concatenate<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: usize) -> PyResult<Array> {
    let arrays = arrays
        .into_iter()
        .map(|arr| as_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    let ret = zix_core::ops::Concatenate::new(arrays, axis).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn stack<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: usize) -> PyResult<Array> {
    let arrays = arrays
        .into_iter()
        .map(|arr| as_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    let ret = zix_core::ops::Stack::new(arrays, axis).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
