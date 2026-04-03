#![cfg_attr(docsrs, feature(doc_cfg))]

mod array;
pub mod dtype;
mod error;
mod iter;
mod ops;
mod schema;
mod storage;
mod util;

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
