use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zix_core::array::BlocksLayout;
use zix_core::codec::ReadContext;
use zix_core::dtype::Dtype;
use zix_core::storage::ArrayStorage;

use zix_core::array::Array as ZixArray;

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

    fn __add__(&self, _other: &Self) -> pyo3::PyResult<Self> {
        let arr = &self.0 + &self.0;
        let a = ZixArray::from_storage(self.0.storage().clone());
        let b = ZixArray::from_storage(self.0.storage().clone());
        let storage = DynStorage(Arc::new(zix_core::ops::Add::new(a, b)?));
        // ZixArray::from_storage(
        unimplemented!()
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
