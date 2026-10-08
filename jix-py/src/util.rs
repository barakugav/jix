use jix_core::NDIM_MAX;
use numpy::npyffi::{npy_intp, PyArray_Dims};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PySliceIndices, PyTuple};

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
    // An omitted forward `stop` unpacks to the max sentinel; substitute the dim length
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
pub(crate) fn maybe_detach<T, F>(py: Python<'_>, detach: bool, f: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    if detach {
        py.detach(f)
    } else {
        f()
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

/// Allocate an uninitialized NumPy array of `shape` whose dims are nested in `order` - outermost
/// dim first - instead of always C-contiguous.
pub(crate) fn numpy_empty_ordered<'py>(
    dtype: Bound<'py, PyArrayDescr>,
    shape: &[u64],
    order: &[usize],
) -> PyResult<Bound<'py, PyUntypedArray>> {
    debug_assert_eq!(order.len(), shape.len());
    if order.iter().enumerate().all(|(dim, &d)| dim == d) {
        return numpy_empty(dtype, shape);
    }
    let py = dtype.py();
    let ndim = shape.len();
    let arr = numpy_empty(dtype, dim_arr(ndim, |i| shape[order[i]]).as_slice())?;

    // Dim `d` of the result is allocation dim `i`, where `order[i] == d`.
    let mut permute = dim_arr(ndim, |_| 0 as npy_intp);
    for (i, &d) in order.iter().enumerate() {
        permute[d] = i as npy_intp;
    }
    let mut permute = PyArray_Dims {
        ptr: permute.as_mut_ptr(),
        len: ndim as std::ffi::c_int,
    };
    // SAFETY: `arr` is a live NumPy array of `ndim` dims and `permute` is a permutation of them.
    let np_arr =
        unsafe { numpy::PY_ARRAY_API.PyArray_Transpose(py, arr.as_array_ptr(), &mut permute) };
    unsafe { Bound::from_owned_ptr_or_err(py, np_arr).map(|ob| ob.cast_into_unchecked()) }
}

#[allow(unused)]
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
pub(crate) fn normalize_dim(dim: i32, ndim: usize) -> pyo3::PyResult<usize> {
    let ndim = ndim as i32;
    if dim < -ndim || dim >= ndim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dim {dim} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(if dim < 0 {
        (ndim + dim) as usize
    } else {
        dim as usize
    })
}
#[inline]
pub(crate) fn normalize_dim_optional(dim: Option<i32>, ndim: usize) -> pyo3::PyResult<usize> {
    match dim {
        Some(dim) => normalize_dim(dim, ndim),
        None => {
            if ndim != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "dim must be specified for arrays with ndim != 1",
                ));
            }
            Ok(0)
        }
    }
}

#[inline]
pub(crate) fn normalize_dims(dims: &[i32], ndim: usize) -> pyo3::PyResult<DimArray<usize>> {
    if ndim > NDIM_MAX {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )));
    }
    dims.iter().map(|a| normalize_dim(*a, ndim)).collect()
}
#[inline]
pub(crate) fn normalize_dims_optional(
    dims: Option<&[i32]>,
    ndim: usize,
) -> pyo3::PyResult<DimArray<usize>> {
    if ndim > NDIM_MAX {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )));
    }
    match dims {
        Some(dims) => normalize_dims(dims, ndim),
        None => Ok((0..ndim).collect()),
    }
}

/// A single `T`, or a `tuple` of `T`. With `LIST` (the default), a `list` of `T` is accepted too.
pub enum ItemOrSequence<T, const LIST: bool = true> {
    Item(T),
    Sequence(Vec<T>),
}
/// A single `T` or a `tuple` of `T`, for dim arguments. Like numpy's reductions and `squeeze`, a
/// `list` is rejected.
pub type ItemOrTuple<T> = ItemOrSequence<T, false>;
impl<'py, T, const LIST: bool> FromPyObject<'_, 'py> for ItemOrSequence<T, LIST>
where
    T: FromPyObjectOwned<'py>,
{
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        // Only tuples (and lists, with `LIST`) count as sequences, unlike pyo3's `Vec<T>`, which
        // accepts any object with the sequence protocol: an array passed where dims are expected
        // is rejected instead of having its elements read as dims.
        if obj.is_instance_of::<PyTuple>() || (LIST && obj.is_instance_of::<PyList>()) {
            let items = obj
                .try_iter()?
                .map(|item| item?.extract::<T>().map_err(Into::into))
                .collect::<PyResult<Vec<T>>>()?;
            return Ok(Self::Sequence(items));
        }
        obj.extract::<T>().map(Self::Item).map_err(|err| {
            let err: PyErr = err.into();
            if !err.is_instance_of::<PyTypeError>(obj.py()) {
                return err; // e.g. OverflowError for an out-of-range int
            }
            let type_name = obj.get_type().name().map(|n| n.to_string());
            PyTypeError::new_err(format!(
                "expected an int, or a tuple{} of ints, got {}",
                if LIST { " or list" } else { "" },
                type_name.as_deref().unwrap_or("?")
            ))
        })
    }
}
impl<T, const LIST: bool> ItemOrSequence<T, LIST> {
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
macro_rules! impl_item_or_sequence_stub_type {
    ($($t:ty),*) => {
        $(
            impl<const LIST: bool> pyo3_stub_gen::PyStubType for ItemOrSequence<$t, LIST> {
                fn type_output() -> pyo3_stub_gen::TypeInfo {
                    let mut info = <$t as pyo3_stub_gen::PyStubType>::type_input();
                    let item = std::mem::take(&mut info.name);
                    info.name = format!("{item} | builtins.tuple[{item}, ...]");
                    if LIST {
                        info.name += &format!(" | builtins.list[{item}]");
                    }
                    info
                }
            }
        )*
    };
}
impl_item_or_sequence_stub_type!(i32, i64, u64);

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
        let mut res = arrayvec::ArrayVec::<T, N>::new();
        let mut res_iter = 0..N;
        loop {
            match (iter.next(), res_iter.next()) {
                (Some(item), Some(_)) => res.push(item?),
                (None, None) => break,
                (_, _) => return Ok(None), // length mismatch
            }
        }
        Ok(Some(res.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}
impl<I> IterExt for I where I: Iterator {}

/// Compute the byte span of a region accessed by `shape` and `strides`.
#[inline]
pub(crate) fn strided_span_bytes(shape: &[usize], strides: &[usize], itemsize: usize) -> usize {
    let mut biggest_offset = 0;
    for (&len, &stride) in shape.iter().zip(strides) {
        if len == 0 {
            return 0;
        }
        biggest_offset += stride * (len - 1);
    }
    biggest_offset + itemsize
}

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
