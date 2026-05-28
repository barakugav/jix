mod operand;
pub(crate) use operand::*;

mod dtype_promote;
pub(crate) use dtype_promote::*;

macro_rules! scalar_kind {
    (i8) => {
        zix_core::dtype::DtypeScalarKind::I8
    };
    (i16) => {
        zix_core::dtype::DtypeScalarKind::I16
    };
    (i32) => {
        zix_core::dtype::DtypeScalarKind::I32
    };
    (i64) => {
        zix_core::dtype::DtypeScalarKind::I64
    };
    (u8) => {
        zix_core::dtype::DtypeScalarKind::U8
    };
    (u16) => {
        zix_core::dtype::DtypeScalarKind::U16
    };
    (u32) => {
        zix_core::dtype::DtypeScalarKind::U32
    };
    (u64) => {
        zix_core::dtype::DtypeScalarKind::U64
    };
    (f16) => {
        zix_core::dtype::DtypeScalarKind::F16
    };
    (f32) => {
        zix_core::dtype::DtypeScalarKind::F32
    };
    (f64) => {
        zix_core::dtype::DtypeScalarKind::F64
    };
    (Complex<f32>) => {
        zix_core::dtype::DtypeScalarKind::ComplexF32
    };
    ((Complex<f32>)) => {
        zix_core::dtype::DtypeScalarKind::ComplexF32
    };
    (Complex<f64>) => {
        zix_core::dtype::DtypeScalarKind::ComplexF64
    };
    ((Complex<f64>)) => {
        zix_core::dtype::DtypeScalarKind::ComplexF64
    };
    (bool) => {
        zix_core::dtype::DtypeScalarKind::Bool
    };
    ($ty:ty) => {
        compile_error!(concat!("Unsupported scalar type: ", stringify!($ty)));
    };
}

macro_rules! define_op1 {
    ($(#[$meta:meta])* $name:ident, $core_op:ident, [$($type:tt),*]) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        pub fn $name<'py>(array: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::any_to_core_array(array)?;
            let res = match array.dtype().try_to_scalar() {
                $(
                    Some(crate::ops::common::scalar_kind!($type)) => {
                        #[allow(unused_parens)]
                        let array = array.to_typed::<$type>().unwrap();
                        zix_core::ops::$core_op::new(array).map(crate::Array::from_core_storage)
                    }
                )*
                _ => Err(zix_core::Error::new(
                    zix_core::ErrorKind::UnsupportedDtype,
                    format!(
                        "Op {} does not support dtype {:#?}",
                        stringify!($name),
                        array.dtype()
                    ),
                )),
            };
            <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
        }
    };
}

macro_rules! define_op2 {
    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        [$(($input_a_type:tt, $input_b_type:tt)),* $(,)?]
    ) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        pub fn $name<'py>(
            a: &pyo3::Bound<'py, pyo3::PyAny>,
            b: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::PyResult<crate::Array> {
            let (a, b) = crate::ops::op2::asarray22(a, b)?;
            let a = a.get().to_core_array();
            let b = b.get().to_core_array();
            let res = match a.dtype().try_to_scalar().zip(b.dtype().try_to_scalar()) {
                $(
                    Some((
                        crate::ops::common::scalar_kind!($input_a_type),
                        crate::ops::common::scalar_kind!($input_b_type)
                    )) => {
                        #[allow(unused_parens)]
                        let a = a.to_typed::<$input_a_type>().unwrap();
                        #[allow(unused_parens)]
                        let b = b.to_typed::<$input_b_type>().unwrap();
                        zix_core::ops::$core_op::new(a, b).map(crate::Array::from_core_storage)
                    }
                )*
                _ => Err(zix_core::Error::new(
                    zix_core::ErrorKind::UnsupportedDtype,
                    format!(
                        "Op {} does not support dtypes {:#?} and {:#?}",
                        stringify!($name),
                        a.dtype(),
                        b.dtype()
                    ),
                )),
            };
            <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
        }
    };

    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        [$($input_type:tt),* $(,)?]
    ) => {
        define_op2!(
            $(#[$meta])*
            $name,
            $core_op,
            [$(($input_type, $input_type)),*]
        );
    };
}

pub(crate) use {define_op1, define_op2, scalar_kind};
