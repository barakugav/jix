macro_rules! define_op1 {
    ($(#[$meta:meta])* $name:ident, $core_op:ident) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        pub fn $name<'py>(array: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::as_core_array(array)?;
            let res = zix_core::ops::$core_op::new(array);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
}

macro_rules! define_op2 {
    ($(#[$meta:meta])* $name:ident, $core_op:ident) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        pub fn $name<'py>(
            a: &pyo3::Bound<'py, pyo3::PyAny>,
            b: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::PyResult<crate::Array> {
            let a = crate::ops::as_array::as_core_array(a)?;
            let b = crate::ops::as_array::as_core_array(b)?;
            let res = zix_core::ops::$core_op::new(a, b);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
}

pub(crate) use {define_op1, define_op2};
