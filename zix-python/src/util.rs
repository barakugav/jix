use numpy::npyffi::npy_intp;
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray};
use pyo3::prelude::*;
use pyo3_stub_gen::impl_stub_type;
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
impl<T> IntoPyResult<T> for Result<T, zix_core::Error> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))
    }
}

pub(crate) fn numpy_empty<'py>(
    dtype: Bound<'py, PyArrayDescr>,
    shape: &[u64],
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = dtype.py();
    let ndim = shape.len();
    let shape = dim_arr(ndim, |dim| shape[dim] as npy_intp);
    let is_fortran = false;
    let np_arr = unsafe {
        numpy::PY_ARRAY_API.PyArray_Empty(
            py,
            shape.len() as _,
            shape.as_ptr().cast_mut(),
            dtype.into_dtype_ptr(),
            if is_fortran { -1 } else { 0 },
        )
    };
    Ok(unsafe { Bound::from_owned_ptr(py, np_arr).cast_into_unchecked() })
}

pub(crate) fn normalize_axis(axis: i32, ndim: usize) -> pyo3::PyResult<usize> {
    let ndim = ndim as i32;
    if axis < -ndim || axis >= ndim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(if axis < 0 {
        (ndim + axis) as usize
    } else {
        axis as usize
    })
}

pub(crate) fn normalize_axes(axes: Vec<i32>, ndim: usize) -> pyo3::PyResult<Vec<usize>> {
    axes.into_iter().map(|a| normalize_axis(a, ndim)).collect()
}

#[derive(FromPyObject)]
pub enum ItemOrSequence<T> {
    Item(T),
    Sequence(Vec<T>),
}
impl<T> ItemOrSequence<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            ItemOrSequence::Item(item) => vec![item],
            ItemOrSequence::Sequence(seq) => seq,
        }
    }
}
impl_stub_type!(ItemOrSequence<i32> = i32 | Vec<i32>);
impl_stub_type!(ItemOrSequence<i64> = i64 | Vec<i64>);
impl_stub_type!(ItemOrSequence<u64> = u64 | Vec<u64>);

#[allow(unused)]
pub(crate) struct UnsafeSend<T>(T);
unsafe impl<T> Send for UnsafeSend<T> {}
impl<T> UnsafeSend<T> {
    pub(crate) unsafe fn new(value: T) -> Self {
        Self(value)
    }

    pub(crate) unsafe fn into_inner(self) -> T {
        self.0
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
