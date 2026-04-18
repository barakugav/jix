use pyo3::prelude::*;
use zix_core::NDIM_MAX;

pub(crate) type DimArray<T> = arrayvec::ArrayVec<T, NDIM_MAX>;

pub(crate) fn dim_arr<T>(ndim: usize, f: impl FnMut(usize) -> T) -> DimArray<T> {
    (0..ndim).map(f).collect()
}
pub(crate) fn check_ndim(ndim: usize) -> PyResult<()> {
    if ndim > NDIM_MAX {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )))
    } else {
        Ok(())
    }
}

pub(crate) trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}
impl<T> IntoPyResult<T> for zix_core::error::Result<T> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    #[ctor::ctor]
    fn init_python() {
        // Usually when writing tests for Python bindings, we need add the modules using `append_to_inittab`, but
        // because we are using the venv, `zix` is already installed and we can import it directly.
        //
        // Note that due to the above, when modifying the bindings, re-install the package to make the changes
        // effective in the tests.
        //
        // pyo3::append_to_inittab!(zix);

        Python::initialize();

        Python::attach(|py| {
            // Pyo3 doesn't detect venv on MacOS.
            //
            // In its documentation it says it does, but it seems to behave weird.
            // Setting PYO3_PYTHON or any other env var (that I tried) doesn't work, so we have to do it manually.
            //
            // (1)
            // Python sys.path contains the path to the site-packages of the global python, so when we add the venv
            // site-packages, we have to add it at the beginning of the list, so it's the first one to be checked.
            // This is considered a bad practice, but it's the only way to make it work.
            // This solves most use cases, but does not work if there are editable packages installed in the venv.
            //
            // (2)
            // To make editable packages work, we have to call site.addsitedir. It processes the .pth files under
            // site-packages, properly adding the editable packages.
            //
            // Existing open issues in Pyo3:
            // - https://github.com/PyO3/pyo3/issues/3284
            // - https://github.com/PyO3/pyo3/issues/1741
            if std::env::var("VIRTUAL_ENV").is_ok() {
                py.run(
                    cr#"
import os, sys
venv_path = os.environ['VIRTUAL_ENV']
if os.name == 'nt':
    packages_path = f"{venv_path}/Lib/site-packages"
else:
    packages_path = f"{venv_path}/lib/python3.13/site-packages"
# DO NOT CHANGE to `.append(..)`
sys.path.insert(0, packages_path)

# DO NOT REMOVE
import site
site.addsitedir(packages_path)
        "#,
                    None,
                    None,
                )
                .unwrap();
            }
        });
    }
}
