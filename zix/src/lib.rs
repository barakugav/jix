#![cfg_attr(docsrs, feature(doc_cfg))]

mod archive;

mod array;
pub use array::{Array, ArrayData, ArrayParams};

pub mod codec;
pub mod dtype;
pub mod ops;
pub mod storage;
mod util;
pub use util::{ArraySequence, ArraySequenceItem};

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
