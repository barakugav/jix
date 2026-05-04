use pyo3::prelude::*;
use zix_core::storage::ArrayStorage;
use zix_core::Array as ZixArray;

use crate::util::IntoPyResult;
use crate::Array;

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
))]
pub fn copy<'py>(array: &Bound<'py, Array>) -> PyResult<Bound<'py, Array>> {
    copy_impl(array.py(), &array.get().arr)
}
pub(crate) fn copy_impl<'py, S>(py: Python<'py>, array: &ZixArray<S>) -> PyResult<Bound<'py, Array>>
where
    S: ArrayStorage + Sync,
{
    let array = py.detach::<PyResult<_>, _>(|| {
        let ret = array.copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })?;
    Bound::new(py, array)
}
