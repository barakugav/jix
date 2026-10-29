use std::mem::{ManuallyDrop, MaybeUninit};

use jix_core::NDIM_MAX;
use numpy::npyffi::{npy_intp, PyArray_Dims};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{
    PyFloatingPointError, PyIndexError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::slice::PySliceIndices;
use pyo3::types::{PySlice, PyTuple};
use pyo3::{type_hint_subscript, type_hint_union};

pub(crate) type DimArray<T> = arrayvec::ArrayVec<T, NDIM_MAX>;
pub(crate) type NumpyDimArray<T> = arrayvec::ArrayVec<T, 32>;

/// Resolves a Python `slice` against a sequence of length `length`, WITHOUT the
/// numpy-style clamping performed by [`PySlice::indices`][pyo3::types::PySliceMethods::indices].
#[inline]
pub(crate) fn slice_unpack(slice: &Bound<'_, PySlice>, length: i64) -> PyResult<PySliceIndices> {
    use pyo3::ffi::Py_ssize_t;
    let Ok(length) = Py_ssize_t::try_from(length) else {
        return Err(PyIndexError::new_err(format!(
            "cannot slice a dim of length {length} on this platform"
        )));
    };
    let mut start: Py_ssize_t = 0;
    let mut stop: Py_ssize_t = 0;
    let mut step: Py_ssize_t = 0;
    // SAFETY: `slice` is a live `PySlice` and the out-pointers are valid and non-null.
    // `PySlice_Unpack` writes through them on success and sets a Python error on failure.
    if unsafe { pyo3::ffi::PySlice_Unpack(slice.as_ptr(), &mut start, &mut stop, &mut step) } < 0 {
        return Err(PyErr::fetch(slice.py()));
    }
    if stop == Py_ssize_t::MAX
        && unsafe {
            (*slice.as_ptr().cast::<pyo3::ffi::PySliceObject>()).stop == pyo3::ffi::Py_None()
        }
    {
        stop = length;
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
        Err(PyValueError::new_err(format!(
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
        self.map_err(|e| {
            let err_str = format!("{e}");
            match e.kind() {
                jix_core::error::ErrorKind::Arithmetic => PyFloatingPointError::new_err(err_str),
                jix_core::error::ErrorKind::UnsupportedDtype => PyTypeError::new_err(err_str),
                _ => PyRuntimeError::new_err(err_str),
            }
        })
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
    if shape.iter().any(|&d| d > npy_intp::MAX as u64) {
        return Err(PyValueError::new_err(format!(
            "array of shape {shape:?} is too big"
        )));
    }
    let shape = shape
        .iter()
        .map(|&d| d as npy_intp)
        .collect::<NumpyDimArray<_>>();
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
    let arr = numpy_empty(
        dtype,
        &order
            .iter()
            .map(|&d| shape[d])
            .collect::<NumpyDimArray<_>>(),
    )?;

    // Dim `d` of the result is allocation dim `i`, where `order[i] == d`.
    let mut permute = (0..ndim).map(|_| 0).collect::<NumpyDimArray<npy_intp>>();
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

pub(crate) struct NumpyItems {
    pub(crate) generic: Py<PyAny>,
    pub(crate) asarray: Py<PyAny>,
}
pub(crate) fn numpy_items(py: Python) -> PyResult<&'static NumpyItems> {
    static NUMPY_ITEMS: PyOnceLock<NumpyItems> = PyOnceLock::new();
    NUMPY_ITEMS.get_or_try_init(py, || {
        let np = numpy::get_array_module(py)?;
        Ok(NumpyItems {
            generic: np.getattr("generic")?.unbind(),
            asarray: np.getattr("asarray")?.unbind(),
        })
    })
}

#[inline]
pub(crate) fn normalize_dim(dim: i32, ndim: usize) -> pyo3::PyResult<usize> {
    let ndim = ndim as i32;
    if dim < -ndim || dim >= ndim {
        return Err(PyValueError::new_err(format!(
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
                return Err(PyValueError::new_err(
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
        return Err(PyValueError::new_err(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )));
    }
    if dims.len() > NDIM_MAX {
        return Err(PyValueError::new_err(format!(
            "Got {} dims, more than the maximum supported {NDIM_MAX}",
            dims.len()
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
        return Err(PyValueError::new_err(format!(
            "Number of dimensions {ndim} exceeds the maximum supported {NDIM_MAX}"
        )));
    }
    match dims {
        Some(dims) => normalize_dims(dims, ndim),
        None => Ok((0..ndim).collect()),
    }
}

pub enum ItemOrTuple<T> {
    Item(T),
    Sequence(Vec<T>),
}
impl<'py, T> FromPyObject<'_, 'py> for ItemOrTuple<T>
where
    T: FromPyObjectOwned<'py>,
{
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_union!(
        T::INPUT_TYPE,
        type_hint_subscript!(
            <PyTuple as pyo3::PyTypeInfo>::TYPE_HINT,
            T::INPUT_TYPE,
            PyStaticExpr::Constant {
                value: PyStaticConstant::Ellipsis
            }
        )
    );

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        // Only tuples count as sequences, unlike pyo3's `Vec<T>`, which
        // accepts any object with the sequence protocol: an array passed where dims are expected
        // is rejected instead of having its elements read as dims.
        if obj.is_instance_of::<PyTuple>() {
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
                "expected an int, or a tuple of ints, got {}",
                type_name.as_deref().unwrap_or("?")
            ))
        })
    }
}
impl<T> ItemOrTuple<T> {
    #[inline]
    pub(crate) fn into_dim_array(self) -> PyResult<DimArray<T>> {
        match self {
            ItemOrTuple::Item(item) => {
                let mut arr = DimArray::new();
                arr.push(item);
                Ok(arr)
            }
            ItemOrTuple::Sequence(seq) => {
                if seq.len() > NDIM_MAX {
                    return Err(PyValueError::new_err(format!(
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
            ItemOrTuple::Item(_) => 1,
            ItemOrTuple::Sequence(seq) => seq.len(),
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl<T> From<T> for ItemOrTuple<T> {
    #[inline]
    fn from(item: T) -> Self {
        ItemOrTuple::Item(item)
    }
}
impl<T> From<Vec<T>> for ItemOrTuple<T> {
    #[inline]
    fn from(seq: Vec<T>) -> Self {
        ItemOrTuple::Sequence(seq)
    }
}
impl<T, const N: usize> From<[T; N]> for ItemOrTuple<T> {
    #[inline]
    fn from(arr: [T; N]) -> Self {
        ItemOrTuple::Sequence(arr.into())
    }
}
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

#[inline]
pub(crate) fn array_try_from_fn<T, E, const N: usize>(
    mut f: impl FnMut(usize) -> Result<T, E>,
) -> Result<[T; N], E> {
    let mut data = MaybeUninit::<[T; N]>::uninit();
    let ptr = data.as_mut_ptr().cast::<T>();
    if const { !std::mem::needs_drop::<T>() } {
        for i in 0..N {
            #[allow(clippy::question_mark)]
            let value = match f(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            unsafe { ptr.add(i).write(value) };
        }
    } else {
        struct InitGuard<T> {
            ptr: *mut T,
            initialized: usize,
        }
        impl<T> Drop for InitGuard<T> {
            #[inline(always)]
            fn drop(&mut self) {
                unsafe {
                    std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                        self.ptr,
                        self.initialized,
                    ))
                };
            }
        }
        let mut guard = InitGuard {
            ptr,
            initialized: 0,
        };
        while guard.initialized < N {
            #[allow(clippy::question_mark)]
            let value = match f(guard.initialized) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            unsafe { ptr.add(guard.initialized).write(value) };
            guard.initialized += 1;
        }
        std::mem::forget(guard);
    }
    Ok(unsafe { data.assume_init() })
}

#[inline]
pub(crate) fn array_try_map<T, U, E, const N: usize>(
    src: [T; N],
    mut f: impl FnMut(usize, T) -> Result<U, E>,
) -> Result<[U; N], E> {
    let mut src = ManuallyDrop::new(src);
    if const { !std::mem::needs_drop::<T>() } {
        // SAFETY: every index is read at most once, and `src` is never dropped (`T` has no drop)
        return array_try_from_fn(
            #[inline(always)]
            |i| f(i, unsafe { std::ptr::read(&src[i]) }),
        );
    }

    /// Drops the items `src[next..]` that were not mapped yet
    struct DropGuard<'a, T, const N: usize> {
        src: &'a mut ManuallyDrop<[T; N]>,
        next: usize,
    }
    impl<T, const N: usize> Drop for DropGuard<'_, T, N> {
        #[inline(always)]
        fn drop(&mut self) {
            // SAFETY: the items `next..` were never read out, and are dropped exactly once here.
            unsafe { std::ptr::drop_in_place(&mut self.src[self.next..]) };
        }
    }
    let mut unconsumed = DropGuard {
        src: &mut src,
        next: 0,
    };
    array_try_from_fn(
        #[inline(always)]
        |i| {
            let item = unsafe { std::ptr::read(&unconsumed.src[i]) };
            unconsumed.next = i + 1;
            f(i, item)
        },
    )
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
