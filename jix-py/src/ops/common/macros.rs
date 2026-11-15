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
                        .map(|res| res.into_any());
                    <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
                }),
            )*
        ]
    };

    (
        $core_op:ident,
        extra_args = $extra_args_struct:ident as $args:ident $extra_args_group:tt,
        [$($ty:ty),*],
        $cast_kind:ident
    ) => {
        vec![
            $(
                crate::ops::common::define_op1_desc!(
                    @inner_with_extra_args
                    $core_op,
                    extra_args = $extra_args_struct as $args $extra_args_group,
                    $ty,
                    $cast_kind
                ),
            )*
        ]
    };
    (
        @inner_with_extra_args
        $core_op:ident,
        extra_args = $extra_args_struct:ident as $args:ident { $($accessor:expr),* $(,)? },
        $ty:ty,
        $cast_kind:ident
    ) => {
        crate::ops::common::OpFnDescriptor::<1, $extra_args_struct>::new1_args::<$ty>(crate::ops::common::CastKind::$cast_kind, |a, $args| {
            let res = jix_core::ops::$core_op::new_array(a, $($accessor),*)
                .map(|res| res.into_any());
            <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
        })
    };
}

macro_rules! define_op1_table {
    (
        $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
    ) => {{
        static DISPATCH_TABLE: ::std::sync::LazyLock<crate::ops::common::OpDescriptor<1, ()>> = ::std::sync::LazyLock::new(|| {
            crate::ops::common::OpDescriptor::new(
                stringify!($name),
                crate::ops::common::define_op1_desc!(
                    $core_op,
                    $($dispatch)*
                ),
            )
        });
        &*DISPATCH_TABLE
    }};
}

macro_rules! define_op1 {
    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
    ) => {
        $(#[$meta])*
        #[pyo3::pyfunction]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::PyResult<crate::PyArray> {
            let array = crate::ops::common::Operand::from_any(array)?;

            let dispatch_table = crate::ops::common::define_op1_table!(
                $name,
                $core_op,
                dispatch = { $($dispatch)* }
            );
            let res = dispatch_table.dispatch1(array)?;

            Ok(crate::PyArray::from_core(res))
        }
    };
}

macro_rules! define_op2_desc {
    (
        $core_op:ident,
        [$($items:tt)*],
        ($a_cast:ident, $b_cast:ident)
    ) => {{
        let mut descs = Vec::new();
        crate::ops::common::define_op2_desc!(@push descs, $core_op, $a_cast, $b_cast, $($items)*);
        descs
    }};
    (
        $core_op:ident,
        [$($items:tt)*],
        $cast_kind:ident
    ) => {
        crate::ops::common::define_op2_desc!($core_op, [$($items)*], ($cast_kind, $cast_kind))
    };

    (
        @push $descs:ident, $core_op:ident, $a_cast:ident, $b_cast:ident,
        ($a_ty:ty, $b_ty:ty) $(, $($rest:tt)*)?
    ) => {
        $descs.push(crate::ops::common::define_op2_desc!(
            @desc $core_op, $a_ty, $b_ty, $a_cast, $b_cast
        ));
        crate::ops::common::define_op2_desc!(
            @push $descs, $core_op, $a_cast, $b_cast, $($($rest)*)?
        );
    };
    (
        @push $descs:ident, $core_op:ident, $a_cast:ident, $b_cast:ident,
        $ty:ty $(, $($rest:tt)*)?
    ) => {
        $descs.push(crate::ops::common::define_op2_desc!(
            @desc $core_op, $ty, $ty, $a_cast, $b_cast
        ));
        crate::ops::common::define_op2_desc!(
            @push $descs, $core_op, $a_cast, $b_cast, $($($rest)*)?
        );
    };
    (@push $descs:ident, $core_op:ident, $a_cast:ident, $b_cast:ident,) => {};

    (@desc $core_op:ident, $a_ty:ty, $b_ty:ty, $a_cast:ident, $b_cast:ident) => {
        crate::ops::common::OpFnDescriptor::new2::<$a_ty, $b_ty>(
            [crate::ops::common::CastKind::$a_cast, crate::ops::common::CastKind::$b_cast],
            |a, b| {
                let res = jix_core::ops::$core_op::new_array(a, b).map(|res| res.into_any());
                <_ as crate::util::IntoPyResult<_>>::into_py_result(res)
            },
        )
    };
}

macro_rules! define_op2_table {
    (
        $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
        $(, $option:ident = $option_val:expr)*
    ) => {{
        static DISPATCH_TABLE: ::std::sync::LazyLock<crate::ops::common::OpDescriptor<2, ()>> = ::std::sync::LazyLock::new(|| {
            crate::ops::common::OpDescriptor::new(
                stringify!($name),
                crate::ops::common::define_op2_desc!(
                    $core_op,
                    $($dispatch)*
                ),
            )
            $(.$option($option_val))*
        });
        &*DISPATCH_TABLE
    }};
}

macro_rules! define_op2 {
    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
        $(, $option:ident = $option_val:expr)*
    ) => {
        $(#[$meta])*
        #[pyo3::pyfunction]
        pub fn $name<'py>(
            a: &pyo3::Bound<'py, pyo3::PyAny>,
            b: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::PyResult<crate::PyArray> {
            let a = crate::ops::common::Operand::from_any(a)?;
            let b = crate::ops::common::Operand::from_any(b)?;
            let [a, b] = crate::ops::common::broadcast_operands([a, b])?;

            let dispatch_table = crate::ops::common::define_op2_table!(
                $name,
                $core_op,
                dispatch = { $($dispatch)* }
                $(, $option = $option_val)*
            );
            let res = dispatch_table.dispatch2(a, b)?;

            Ok(crate::PyArray::from_core(res))
        }
    };
}

pub(crate) use {
    define_op1, define_op1_desc, define_op1_table, define_op2, define_op2_desc, define_op2_table,
};
