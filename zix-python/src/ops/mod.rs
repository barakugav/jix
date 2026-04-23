mod common;
use common::{define_op1, define_op2};

mod as_array;
pub use as_array::*;

mod bitwise;
pub use bitwise::*;

mod cmp;
pub use cmp::*;

mod shape_ops;
pub use shape_ops::*;

use pyo3::prelude::*;

use crate::array::Array;
use crate::dtype::dtype_from_any;
use crate::util::IntoPyResult;

// op1
define_op1!(negative, Abs);
define_op1!(floor, Floor);
define_op1!(ceil, Ceil);
define_op1!(round, Round);
define_op1!(sqrt, Sqrt);
define_op1!(exp, Exp);
define_op1!(log, Log);
define_op1!(sin, Sin);
define_op1!(cos, Cos);
define_op1!(tan, Tan);
define_op1!(asin, Asin);
define_op1!(acos, Acos);
define_op1!(atan, Atan);
define_op1!(signum, Signum);
define_op1!(absolute, Abs);

// op2
define_op2!(
    /// TODO
    add, Add
);
define_op2!(subtract, Sub);
define_op2!(multiply, Mul);
define_op2!(divide, Div);
define_op2!(power, Power);

// logical1
define_op1!(is_nan, IsNan);
define_op1!(is_finite, IsFinite);
define_op1!(is_infinite, IsInfinite);

// reduction
macro_rules! define_reduction_op {
    ($(#[$meta:meta])* $name:ident, $core_op:ident) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axes=None,
            keepdims=false,
        ))]
        pub fn $name<'py>(
            py: pyo3::Python<'py>,
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axes: Option<Vec<usize>>,
            keepdims: bool,
        ) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::as_core_array(py, array)?;
            let axes = axes.unwrap_or_else(|| (0..array.ndim()).collect());
            let res = zix_core::ops::$core_op::new(array, &axes, keepdims);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
    ($(#[$meta:meta])* $name:ident, $core_op:ident, single_axis = "true") => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axis=None,
            keepdims=false,
        ))]
        pub fn $name<'py>(
            py: pyo3::Python<'py>,
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axis: Option<usize>,
            keepdims: bool,
        ) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::as_core_array(py, array)?;
            let axis = match axis {
                Some(axis) => axis,
                None => {
                    if array.ndim() != 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "axis must be specified for arrays with ndim != 1",
                        ));
                    }
                    0
                },
            };
            let res = zix_core::ops::$core_op::new(array, axis, keepdims);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
}
define_reduction_op!(max, Max);
define_reduction_op!(min, Min);
define_reduction_op!(argmax, ArgMax, single_axis = "true");
define_reduction_op!(argmin, ArgMin, single_axis = "true");
define_reduction_op!(product, Product);
define_reduction_op!(all, All);
define_reduction_op!(any, Any);

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn astype<'py>(
    py: Python<'py>,
    array: &Bound<'py, Array>,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let dtype = dtype_from_any(py, dtype)?;
    let ret = zix_core::ops::AsType::new(array, dtype).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
