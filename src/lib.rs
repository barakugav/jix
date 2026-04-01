#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod dtype;
mod iter;
mod util;

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
