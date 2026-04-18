use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod array;
use array::Array;

mod dtype;
mod ops;
mod storage;
mod util;

#[pymodule]
fn zix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<Array>()?;

    m.add_function(wrap_pyfunction!(ops::asarray, m)?)?;

    // pyo3_log::Logger::new(m.py(), pyo3_log::Caching::Nothing)
    //     .unwrap()
    //     .filter(log::LevelFilter::Trace)
    //     .install()
    //     .unwrap();

    Ok(())
}

define_stub_info_gatherer!(gen_pyi);
