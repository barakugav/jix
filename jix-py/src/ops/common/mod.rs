mod operand;
pub(crate) use operand::*;

mod dtype_promote;
pub(crate) use dtype_promote::*;

mod broadcast;
pub(crate) use broadcast::*;

mod dispatch;
pub(crate) use dispatch::*;

macro_rules! define_op1_desc {
    (
        $core_op:ident,
        [$($ty:ty),*],
        $cast_kind:ident
    ) => {
        vec![
            $(
                crate::ops::common::OpFnDescriptor::new1::<$ty>(crate::ops::common::CastKind::$cast_kind, |a| {
                    let res = jix_core::ops::$core_op::new_array(a)
                        .map(|res| res.into_type_dyn().into_any());
                    <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
                }),
            )*
        ]
    };

    (
        $core_op:ident,
        extra_args = $extra_args_struct:ident $extra_args_group:tt,
        [$($ty:ty),*],
        $cast_kind:ident
    ) => {
        vec![
            $(
                crate::ops::common::define_op1_desc!(
                    @inner_with_extra_args
                    $core_op,
                    extra_args = $extra_args_struct $extra_args_group,
                    $ty,
                    $cast_kind
                ),
            )*
        ]
    };
    (
        @inner_with_extra_args
        $core_op:ident,
        extra_args = $extra_args_struct:ident { $($extra_arg:ident),* },
        $ty:ty,
        $cast_kind:ident
    ) => {
        crate::ops::common::OpFnDescriptor::<1, $extra_args_struct>::new1_args::<$ty>(crate::ops::common::CastKind::$cast_kind, |a, extra_args_struct| {
            let res = jix_core::ops::$core_op::new_array(a, $(extra_args_struct.$extra_arg),*)
                .map(|res| res.into_type_dyn().into_any());
            <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
        })
    };
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
    ) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::PyResult<crate::Array> {
            static DISPATCH_TABLE: std::sync::LazyLock<crate::ops::common::OpDescriptor<1, ()>> = std::sync::LazyLock::new(|| {
                crate::ops::common::OpDescriptor::new(
                    stringify!($name),
                    crate::ops::common::define_op1_desc!(
                        $core_op,
                        $($dispatch)*
                    ),
                )
            });

            let array = crate::ops::common::Operand::from_any(array)?;
            let res = DISPATCH_TABLE.dispatch1(array)?;
            Ok(crate::Array::from_core(res))
        }
    };
}

macro_rules! define_op2_desc {
    (
        $core_op:ident,
        [$(($a_ty:ty, $b_ty:ty)),*],
        $cast_kind:ident
    ) => {
        vec![
            $(
                crate::ops::common::define_op2_desc!(
                    @inner
                    $core_op,
                    ($a_ty, $b_ty),
                    $cast_kind
                ),
            )*
        ]
    };

    (
        $core_op:ident,
        [$($ty:ty),*],
        $cast_kind:ident
    ) => {
        vec![
            $(
                crate::ops::common::define_op2_desc!(
                    @inner
                    $core_op,
                    $ty,
                    $cast_kind
                ),
            )*
        ]
    };

    (
        @inner
        $core_op:ident,
        ($a_ty:ty, $b_ty:ty),
        [$($cast_kind:ident),*]
    ) => {
        crate::ops::common::OpFnDescriptor::new2::<$a_ty, $b_ty>([$(crate::ops::common::CastKind::$cast_kind),*], |a, b| {
            let res = jix_core::ops::$core_op::new_array(a, b)
                .map(|res| res.into_type_dyn().into_any());
            <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
        })
    };

    (
        @inner
        $core_op:ident,
        ($a_ty:ty, $b_ty:ty),
        $cast_kind:ident
    ) => {
        crate::ops::common::define_op2_desc!(
            @inner
            $core_op,
            ($a_ty, $b_ty),
            [$cast_kind, $cast_kind]
        )
    };

    (
        @inner
        $core_op:ident,
        $ty:ty,
        $cast_kind:ident
    ) => {
        crate::ops::common::define_op2_desc!(
            @inner
            $core_op,
            ($ty, $ty),
            $cast_kind
        )
    };
}

macro_rules! define_op2 {
    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
    ) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        pub fn $name<'py>(
            a: &pyo3::Bound<'py, pyo3::PyAny>,
            b: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::PyResult<crate::Array> {
            static DISPATCH_TABLE: std::sync::LazyLock<crate::ops::common::OpDescriptor<2, ()>> = std::sync::LazyLock::new(|| {
                crate::ops::common::OpDescriptor::new(
                    stringify!($name),
                    crate::ops::common::define_op2_desc!(
                        $core_op,
                        $($dispatch)*
                    ),
                )
            });

            let a = crate::ops::common::Operand::from_any(a)?;
            let b = crate::ops::common::Operand::from_any(b)?;
            let [a, b] = crate::ops::common::broadcast_operands([a, b])?;
            let res = DISPATCH_TABLE.dispatch2(a, b)?;
            Ok(crate::Array::from_core(res))
        }
    };
}

pub(crate) use {define_op1, define_op1_desc, define_op2, define_op2_desc};
