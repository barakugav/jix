mod proto_gen {
    include!(concat!(env!("OUT_DIR"), "/proto_gen/_includes.rs"));
}
pub(crate) use proto_gen::zix::v1::*;

impl crate::dtype::Dtype {
    pub(crate) fn from_proto(dtype: &Dtype) -> Self {
        todo!()
    }
    pub(crate) fn to_proto(&self) -> Dtype {
        todo!()
    }
}
