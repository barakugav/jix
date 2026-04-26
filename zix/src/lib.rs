#![cfg_attr(docsrs, feature(doc_cfg))]

// This crate is a Rust port and independent evolution of ideas and
//
// See the `NOTICE` file in the repository root for full attribution and license text.

mod array;
pub use array::Array;

pub mod codec;
pub mod dtype;
mod params;
pub use params::ArrayParams;

pub mod storage;

mod archive;

pub mod ops;

mod util;
pub use util::ArraySequence;

pub mod error;

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;
