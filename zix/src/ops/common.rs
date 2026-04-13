macro_rules! define_array_op1_method {
    ($op:ident : $Name:ident) => {
        #[track_caller]
        pub fn $op(self) -> crate::Array<$Name<S>> {
            let op = $Name::new(self).unwrap();
            crate::Array::from_storage(op)
        }
    };
}
macro_rules! define_array_op2_method {
    ($op:ident : $Name:ident) => {
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
pub(crate) use {define_array_op1_method, define_array_op2_method};
