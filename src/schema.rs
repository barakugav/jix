mod proto_gen {
    include!(concat!(env!("OUT_DIR"), "/proto_gen/_includes.rs"));
}
pub(crate) use proto_gen::zix::v1::*;
