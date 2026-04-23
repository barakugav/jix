macro_rules! define_array_op1_method {
    ($op:ident : $Name:ident) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op(self) -> crate::Array<$Name<S>> {
            let op = $Name::new(self).unwrap();
            crate::Array::from_storage(op)
        }
    };
}
macro_rules! define_array_op2_method {
    ($op:ident : $Name:ident) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op<S2>(self, other: crate::Array<S2>) -> crate::Array<$Name<S, S2>>
        where
            S2: crate::storage::ArrayStorage,
        {
            let op = $Name::new(self, other).unwrap();
            crate::Array::from_storage(op)
        }
    };
}

macro_rules! scalar_kind {
    (i8) => {
        crate::dtype::DtypeScalarKind::I8
    };
    (i16) => {
        crate::dtype::DtypeScalarKind::I16
    };
    (i32) => {
        crate::dtype::DtypeScalarKind::I32
    };
    (i64) => {
        crate::dtype::DtypeScalarKind::I64
    };
    (u8) => {
        crate::dtype::DtypeScalarKind::U8
    };
    (u16) => {
        crate::dtype::DtypeScalarKind::U16
    };
    (u32) => {
        crate::dtype::DtypeScalarKind::U32
    };
    (u64) => {
        crate::dtype::DtypeScalarKind::U64
    };
    (f16) => {
        crate::dtype::DtypeScalarKind::F16
    };
    (f32) => {
        crate::dtype::DtypeScalarKind::F32
    };
    (f64) => {
        crate::dtype::DtypeScalarKind::F64
    };
    (Complex<f32>) => {
        crate::dtype::DtypeScalarKind::ComplexF32
    };
    ((Complex<f32>)) => {
        crate::dtype::DtypeScalarKind::ComplexF32
    };
    (Complex<f64>) => {
        crate::dtype::DtypeScalarKind::ComplexF64
    };
    ((Complex<f64>)) => {
        crate::dtype::DtypeScalarKind::ComplexF64
    };
    (bool) => {
        crate::dtype::DtypeScalarKind::Bool
    };
    ($ty:ty) => {
        compile_error!(concat!("Unsupported scalar type: ", stringify!($ty)));
    };
}

pub(crate) use {define_array_op1_method, define_array_op2_method, scalar_kind};

// TODO: optimize per scalar type
pub(crate) const BULK: usize = 8;
