#![cfg_attr(docsrs, feature(doc_cfg))]

// This crate is a Rust port and independent evolution of ideas and
//
// See the `NOTICE` file in the repository root for full attribution and license text.

mod array;
pub use array::{Array, ArrayParams};

pub mod codec;
pub mod dtype;
pub mod ops;
pub mod storage;
mod util;
pub use util::{ArraySequence, ArraySequenceItem};
pub mod error;

mod archive;

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
