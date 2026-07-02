use std::mem::MaybeUninit;

use jix_core::NDIM_MAX;
use numpy::npyffi::npy_intp;
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PySliceIndices};
use pyo3_stub_gen::impl_stub_type;

pub(crate) type DimArray<T> = arrayvec::ArrayVec<T, NDIM_MAX>;

/// Resolves a Python `slice` against a sequence of length `length`, WITHOUT the
/// numpy-style clamping performed by [`PySlice::indices`][pyo3::types::PySliceMethods::indices].
#[inline]
pub(crate) fn slice_unpack(slice: &Bound<'_, PySlice>, length: i64) -> PyResult<PySliceIndices> {
    let mut start: pyo3::ffi::Py_ssize_t = 0;
    let mut stop: pyo3::ffi::Py_ssize_t = 0;
    let mut step: pyo3::ffi::Py_ssize_t = 0;
    // SAFETY: `slice` is a live `PySlice` and the out-pointers are valid and non-null.
    // `PySlice_Unpack` writes through them on success and sets a Python error on failure.
    if unsafe { pyo3::ffi::PySlice_Unpack(slice.as_ptr(), &mut start, &mut stop, &mut step) } < 0 {
        return Err(PyErr::fetch(slice.py()));
    }
    // An omitted forward `stop` unpacks to the max sentinel; substitute the axis length
    // (its Python default) so it is not later mistaken for an out-of-range bound.
    if stop == pyo3::ffi::Py_ssize_t::MAX {
        stop = length as pyo3::ffi::Py_ssize_t;
    }
    Ok(PySliceIndices::new(start, stop, step))
}

#[inline(always)]
pub(crate) fn dim_arr<T>(ndim: usize, f: impl FnMut(usize) -> T) -> DimArray<T> {
    (0..ndim).map(f).collect()
}
#[inline(always)]
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
impl<T> IntoPyResult<T> for Result<T, jix_core::Error> {
    #[inline(always)]
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))
    }
}

#[inline]
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
    unsafe { Bound::from_owned_ptr_or_err(py, np_arr).map(|ob| ob.cast_into_unchecked()) }
}

#[inline]
pub(crate) fn numpy_reshape<'py>(
    arr: Bound<'py, PyUntypedArray>,
    shape: &[u64],
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = arr.py();
    let ndim = shape.len();
    let shape = dim_arr(ndim, |dim| shape[dim] as npy_intp);
    let mut shape = numpy::npyffi::PyArray_Dims {
        ptr: shape.as_ptr().cast_mut(),
        len: ndim as _,
    };
    let np_arr = unsafe {
        numpy::PY_ARRAY_API.PyArray_Newshape(
            py,
            arr.as_array_ptr(),
            &mut shape as *mut _,
            numpy::npyffi::NPY_ORDER::NPY_ANYORDER,
        )
    };

    unsafe { Bound::from_owned_ptr_or_err(py, np_arr).map(|ob| ob.cast_into_unchecked()) }
}

#[inline]
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
#[inline]
pub(crate) fn normalize_axis_optional(axis: Option<i32>, ndim: usize) -> pyo3::PyResult<usize> {
    match axis {
        Some(axis) => normalize_axis(axis, ndim),
        None => {
            if ndim != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "axis must be specified for arrays with ndim != 1",
                ));
            }
            Ok(0)
        }
    }
}

#[inline]
pub(crate) fn normalize_axes(axes: &[i32], ndim: usize) -> pyo3::PyResult<DimArray<usize>> {
    if ndim > NDIM_MAX {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )));
    }
    axes.iter().map(|a| normalize_axis(*a, ndim)).collect()
}
#[inline]
pub(crate) fn normalize_axes_optional(
    axes: Option<&[i32]>,
    ndim: usize,
) -> pyo3::PyResult<DimArray<usize>> {
    if ndim > NDIM_MAX {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )));
    }
    match axes {
        Some(axes) => normalize_axes(axes, ndim),
        None => Ok((0..ndim).collect()),
    }
}

#[derive(FromPyObject)]
pub enum ItemOrSequence<T> {
    Item(T),
    Sequence(Vec<T>),
}
impl<T> ItemOrSequence<T> {
    #[inline]
    pub(crate) fn into_dim_array(self) -> PyResult<DimArray<T>> {
        match self {
            ItemOrSequence::Item(item) => {
                let mut arr = DimArray::new();
                arr.push(item);
                Ok(arr)
            }
            ItemOrSequence::Sequence(seq) => {
                if seq.len() > NDIM_MAX {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Number of dimensions {} exceeds the maximum supported {}",
                        seq.len(),
                        NDIM_MAX
                    )));
                }
                Ok(seq.into_iter().collect())
            }
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            ItemOrSequence::Item(_) => 1,
            ItemOrSequence::Sequence(seq) => seq.len(),
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl<T> From<T> for ItemOrSequence<T> {
    #[inline]
    fn from(item: T) -> Self {
        ItemOrSequence::Item(item)
    }
}
impl<T> From<Vec<T>> for ItemOrSequence<T> {
    #[inline]
    fn from(seq: Vec<T>) -> Self {
        ItemOrSequence::Sequence(seq)
    }
}
impl<T, const N: usize> From<[T; N]> for ItemOrSequence<T> {
    #[inline]
    fn from(arr: [T; N]) -> Self {
        ItemOrSequence::Sequence(arr.into())
    }
}
impl_stub_type!(ItemOrSequence<i32> = i32 | Vec<i32>);
impl_stub_type!(ItemOrSequence<i64> = i64 | Vec<i64>);
impl_stub_type!(ItemOrSequence<u64> = u64 | Vec<u64>);

#[allow(unused)]
pub(crate) struct UnsafeSend<T>(T);
unsafe impl<T> Send for UnsafeSend<T> {}
#[allow(unused)]
impl<T> UnsafeSend<T> {
    #[inline]
    pub(crate) unsafe fn new(value: T) -> Self {
        Self(value)
    }

    #[inline]
    pub(crate) unsafe fn into_inner(self) -> T {
        self.0
    }
}

pub(crate) trait IterExt: Iterator {
    #[inline]
    fn try_collect_array<T, E, const N: usize>(self) -> Result<Option<[T; N]>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
        T: Sized,
    {
        let mut iter = self;
        let mut res = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        let mut res_iter = res.iter_mut();
        loop {
            match (iter.next(), res_iter.next()) {
                (Some(item), Some(res)) => {
                    res.write(item?);
                }
                (None, None) => break,
                (_, _) => return Ok(None), // length mismatch
            }
        }
        let res = unsafe { std::mem::transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&res) };
        Ok(Some(res))
    }
}
impl<I> IterExt for I where I: Iterator {}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    #[ctor::ctor(unsafe)]
    fn init_python() {
        // Usually when writing tests for Python bindings, we need add the modules using `append_to_inittab`, but
        // because we are using the venv, `jix` is already installed and we can import it directly.
        //
        // Note that due to the above, when modifying the bindings, re-install the package to make the changes
        // effective in the tests.
        //
        // pyo3::append_to_inittab!(jix);

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
