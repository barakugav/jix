//! Thin wrappers whose only job is to give arguments and return values a precise type hint in the
//! generated `.pyi` stubs, for types that do not carry one themselves (e.g. the `numpy` crate's
//! types are all `Incomplete`).

use std::convert::Infallible;
use std::ops::Deref;

use jix_core::NDIM_MAX;
use numpy::{PyArrayDescr, PyUntypedArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyTuple};
use pyo3::{type_hint_identifier, type_hint_subscript, type_hint_union, PyClass, PyTypeInfo};

use crate::util::DimArray;

/// `numpy.typing.NDArray[typing.Any]`
const NDARRAY_TYPE_HINT: PyStaticExpr = type_hint_subscript!(
    type_hint_identifier!("numpy.typing", "NDArray"),
    type_hint_identifier!("typing", "Any")
);

/// A NumPy array of any dtype and shape.
#[derive(Debug)]
pub struct NdArray<'py>(pub Bound<'py, PyUntypedArray>);

impl<'py> Deref for NdArray<'py> {
    type Target = Bound<'py, PyUntypedArray>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for NdArray<'py> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = NDARRAY_TYPE_HINT;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract()?))
    }
}

impl<'py> IntoPyObject<'py> for NdArray<'py> {
    type Target = PyUntypedArray;
    type Output = Bound<'py, PyUntypedArray>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = NDARRAY_TYPE_HINT;

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// A NumPy dtype.
pub struct NpDtype<'py>(pub Bound<'py, PyArrayDescr>);

impl<'py> IntoPyObject<'py> for NpDtype<'py> {
    type Target = PyArrayDescr;
    type Output = Bound<'py, PyArrayDescr>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_identifier!("numpy", "dtype");

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// An array shape, typed as `tuple[int, ...]` instead of a PyTuple's bare `tuple`.
pub struct Shape<'py>(pub Bound<'py, PyTuple>);

impl<'py> IntoPyObject<'py> for Shape<'py> {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        <PyTuple as PyTypeInfo>::TYPE_HINT,
        <PyInt as PyTypeInfo>::TYPE_HINT,
        PyStaticExpr::Constant {
            value: PyStaticConstant::Ellipsis
        }
    );

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// A dim argument. Unlike a plain `i32` it rejects a `bool`.
#[derive(Clone, Copy)]
pub struct Dim(pub i32);

impl<'a, 'py> FromPyObject<'a, 'py> for Dim {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = <PyInt as PyTypeInfo>::TYPE_HINT;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("a dim must be an int, not a bool"));
        }
        Ok(Self(obj.extract()?))
    }
}

/// The result of a binary dunder. `NotImplemented` makes Python try the other operand's reflected
/// op. Typed as an `Array`, like numpy's dunders.
pub enum ValueOrUnimplemented<T>
where
    T: PyClass + Into<PyClassInitializer<T>> + PyTypeInfo,
{
    /// A value
    Value(T),
    /// Function is not implemented
    NotImplemented,
}

impl<'py, T> IntoPyObject<'py> for ValueOrUnimplemented<T>
where
    T: PyClass + Into<PyClassInitializer<T>> + PyTypeInfo,
{
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    const OUTPUT_TYPE: PyStaticExpr = <T as PyTypeInfo>::TYPE_HINT;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Self::Value(value) => Ok(Bound::new(py, value)?.into_any()),
            Self::NotImplemented => Ok(py.NotImplemented().into_bound(py)),
        }
    }
}

/// A single item or a tuple of items, e.g. a dim argument that takes `int | tuple[int, ...]`.
/// Unlike pyo3's `Vec<T>`, only a `tuple` is read as a sequence.
pub enum ItemOrTuple<T> {
    /// A single item
    Item(T),
    /// A tuple of items
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
            <PyTuple as PyTypeInfo>::TYPE_HINT,
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
