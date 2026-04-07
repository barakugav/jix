#![cfg_attr(docsrs, feature(doc_cfg))]

mod archive;
pub mod array;
mod block;
mod codec;
pub mod dtype;
mod iter;
mod ops;
mod schema;
mod storage;
mod util;

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
