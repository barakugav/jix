use pyo3::prelude::*;
use zix_core::Array as ZixArray;

use crate::ops::as_core_array;
use crate::util::{normalize_axes, normalize_axis, IntoPyResult, ItemOrSequence};
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
    new_shape: ItemOrSequence<u64>,
    copy: bool,
) -> PyResult<Array> {
    let py = array.py();
    let array = array.borrow().to_core_array();
    let new_shape = new_shape.into_vec();
    let ret = zix_core::ops::Broadcast::new(array, &new_shape).into_py_result()?;
    if !copy {
        return Ok(Array::from_core_storage(ret));
    }
    // release the GIL while copying the data
    py.detach(|| {
        let ret = ZixArray::from_storage(ret).copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })
}

// TODO slice

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn insert_axes<'py>(array: &Bound<'py, Array>, axes: ItemOrSequence<i32>) -> PyResult<Array> {
    // NOTE: API different than numpy: axes are specified with respect to the original ndim, not the new ndim. Same
    // axis can be specified multiple times to insert multiple axes in the same place.
    let array = array.borrow().to_core_array();
    let axes = normalize_axes(axes.into_vec(), array.ndim() + 1)?;
    let ret = zix_core::ops::InsertAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn remove_axes<'py>(array: &Bound<'py, Array>, axes: ItemOrSequence<i32>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let axes = normalize_axes(axes.into_vec(), array.ndim())?;
    let ret = zix_core::ops::RemoveAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    axes=None,
))]
pub fn permute_axes<'py>(array: &Bound<'py, Array>, axes: Option<Vec<usize>>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let axes = axes.unwrap_or_else(|| (0..array.ndim()).rev().collect());
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
pub fn reshape<'py>(
    array: &Bound<'py, Array>,
    new_shape: ItemOrSequence<i64>,
    copy: bool,
) -> PyResult<Array> {
    let py = array.py();
    let array = array.borrow().to_core_array();

    // handle -1 in new_shape
    let new_shape = {
        let mut new_shape = new_shape.into_vec();
        let mut inferred_dim = None;
        let mut known_size = 1;
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim >= 0 {
                known_size *= dim as u64;
            } else if dim == -1 {
                if inferred_dim.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Only one dimension can be inferred (-1)",
                    ));
                }
                inferred_dim = Some(i);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "new_shape must be non negative or -1",
                ));
            }
        }
        if let Some(inferred_dim) = inferred_dim {
            let array_size = array.shape().iter().product::<u64>();
            if array_size == 0 || known_size == 0 || !array_size.is_multiple_of(known_size) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot infer dimension size",
                ));
            }
            new_shape[inferred_dim] = (array_size / known_size) as i64;
        }
        new_shape.iter().map(|&dim| dim as u64).collect::<Vec<_>>()
    };

    let ret = zix_core::ops::Reshape::new(array, &new_shape).into_py_result()?;
    if !copy {
        return Ok(Array::from_core_storage(ret));
    }
    // release the GIL while copying the data
    py.detach(|| {
        let ret = ZixArray::from_storage(ret).copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    arrays,
    axis=0,
))]
pub fn concatenate<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: i32) -> PyResult<Array> {
    let arrays = arrays
        .into_iter()
        .map(|arr| as_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "arrays must contain at least one array",
        ));
    }
    let ndim = arrays[0].ndim();
    for arr in &arrays {
        if arr.ndim() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All arrays must have the same number of dimensions",
            ));
        }
    }
    let axis = normalize_axis(axis, ndim)?;
    let ret = zix_core::ops::Concatenate::new(arrays, axis).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    arrays,
    axis=0,
))]
pub fn stack<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: i32) -> PyResult<Array> {
    let arrays = arrays
        .into_iter()
        .map(|arr| as_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "arrays must contain at least one array",
        ));
    }
    let ndim = arrays[0].ndim();
    for arr in &arrays {
        if arr.ndim() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All arrays must have the same number of dimensions",
            ));
        }
    }
    let axis = normalize_axis(axis, ndim + 1)?;
    let ret = zix_core::ops::Stack::new(arrays, axis).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
