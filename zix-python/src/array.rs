use std::sync::Arc;

use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zix_core::array::BlocksLayout;
use zix_core::codec::ReadContext;
use zix_core::dtype::{Dtype, DtypeScalarKind};
use zix_core::storage::ArrayStorage;

use zix_core::array::Array as ZixArray;

use crate::dtype::dtype_to_numpy;

#[gen_stub_pyclass]
#[pyclass]
pub struct Array(ZixArray<DynStorage>);

#[gen_stub_pymethods]
#[pymethods]
impl Array {
    // #[new]
    // pub fn new() -> Self {
    //     Self { dummy: 0 }
    // }

    fn numpy(&self, py: Python) -> PyResult<()> {
        let dtype = self.0.dtype();
        let dtype_np = dtype_to_numpy(py, dtype)?;
        // numpy::PY_ARRAY_API.PyArray_Empty
        // let array = self.0.data().to_ndarray();
        Ok(())
    }

    fn __add__(&self, _other: &Self) -> PyResult<Self> {
        let a = ZixArray::from_storage(self.0.storage().clone());
        let b = ZixArray::from_storage(self.0.storage().clone());
        let storage = DynStorage(Arc::new(zix_core::ops::Add::new(a, b)?));
        Ok(Self(ZixArray::from_storage(storage)))
    }
}

#[derive(Clone)]
struct DynStorage(Arc<dyn ArrayStorage + Send + Sync>);
impl ArrayStorage for DynStorage {
    fn dtype(&self) -> &Dtype {
        self.0.dtype()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        self.0.blocks_layout()
    }

    fn read_data(
        &self,
        index: &[std::ops::Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> std::io::Result<()> {
        self.0.read_data(index, buf, context)
    }
}
