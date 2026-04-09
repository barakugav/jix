#![cfg_attr(docsrs, feature(doc_cfg))]

mod archive;
pub mod array;
mod block;
pub mod codec;
pub mod dtype;
mod iter;
pub mod ops;
mod schema;
pub mod storage;
mod util;

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
