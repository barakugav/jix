#[cfg(feature = "build-schema")]
mod proto_gen {
    include!(concat!(env!("OUT_DIR"), "/proto_gen/_includes.rs"));
}
#[cfg(not(feature = "build-schema"))]
mod proto_gen {
    mod _includes;
    pub use _includes::*;
}

pub(crate) use proto_gen::zix::v1::*;

mod conversions;
