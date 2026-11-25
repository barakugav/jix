use std::io::Cursor;

use jix_core::storage::{ArchiveValidation, CompactArrayMethods, CompactParams, ReadParams};
use jix_core::{Array, StorageBase};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::PyArray;
use crate::archive::apply_read_params;
use crate::array::resolve_array_params;
use crate::util::{IntoPyResult, StorageFormatter};

const PICKLE_VERSION: u32 = 1;

pub(crate) fn reduce_array<'py>(
    array: &Bound<'py, PyArray>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
    let py = array.py();
    let array = array.get();
    let arr = &array.arr;
    let Some(compact) = arr.storage().as_compact() else {
        return Err(PyTypeError::new_err(format!(
            "Can't pickle jix array with storage `{}`, consider calling .compact() first",
            StorageFormatter(arr.storage()),
        )));
    };
    let compact = Array::from_storage(compact);

    let params = PyDict::new(py);
    let fixed_dims = (0..arr.ndim())
        .map(|dim| compact.block_shape_fixed_dim(dim))
        .collect::<Result<Vec<_>, _>>()
        .into_py_result()?;
    params.set_item("block_shape_fixed_dims", fixed_dims)?;
    let read_size = arr.read_params().read_size();
    if read_size != ReadParams::default().read_size() {
        params.set_item("read_size", read_size)?;
    }

    let mut bytes = Cursor::new(Vec::new());
    py.detach(|| arr.write_to(&mut bytes, CompactParams::default()))
        .into_py_result()?;
    let bytes = PyBytes::new(py, bytes.get_ref());

    let unpickle = py.import("jix")?.getattr("_unpickle_array")?;
    let args = (PICKLE_VERSION, "Compact", (bytes, params)).into_pyobject(py)?;
    Ok((unpickle, args))
}

/// Unpickle an array.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn _unpickle_array(args: &Bound<'_, PyTuple>) -> PyResult<PyArray> {
    let py = args.py();
    let version = args.get_item(0)?.extract::<u32>()?;
    if version != PICKLE_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported jix pickle version {version}, it may come from a newer jix than {}",
            env!("CARGO_PKG_VERSION")
        )));
    }
    let (_, kind, kind_args) = args.extract::<(u32, String, Bound<'_, PyAny>)>()?;
    match kind.as_str() {
        "Compact" => {
            let (bytes, params) = kind_args.extract::<(Bound<'_, PyBytes>, Bound<'_, PyDict>)>()?;

            // Unlike user kwargs, ignore unknown keys: a newer jix may add some.
            let keys = ["read_size", "block_shape_fixed_dims"];
            let known = PyDict::new(py);
            for key in keys {
                if let Some(value) = params.get_item(key)? {
                    known.set_item(key, value)?;
                }
            }
            let params = resolve_array_params(py, Some(known), &keys)?;

            let bytes = bytes.as_bytes();
            let mut array = py
                .detach(|| Array::read_from(Cursor::new(bytes), None, ArchiveValidation::default()))
                .into_py_result()?;
            apply_read_params(&mut array, &params)?;
            Ok(PyArray::from_core(array.into_any()))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported jix pickle array kind {kind:?}"
        ))),
    }
}
