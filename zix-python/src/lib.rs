#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Multi-dimensional array library with block-compressed, lazy-evaluated storage.
//!
//! The crate provide Python bindings for the core `zix` library, which is implemented in Rust.
//! `zix` is a multi-dimensional array library that stores data in **block-compressed format**
//! and evaluates operations **lazily**. It is designed around two ideas:
//!
//! - **Block-based compression** — the array is split into an n-dimensional grid of fixed-size
//!   blocks, each compressed independently with Zstd. Only the blocks that overlap a read
//!   request are decompressed, so random access into large arrays avoids loading the whole
//!   dataset into memory.
//!
//! - **Lazy operation chains** — every operation (arithmetic, shape manipulation, type cast,
//!   reduction, ...) builds a new `Array` that records the transformation without executing
//!   it. The full pipeline runs in a single decompression pass the moment you ask for output.
//!   While the pass runs the GIL is released, so Python threads can make progress
//!   concurrently.
//!
//! The library is NumPy-compatible: arrays expose a NumPy `dtype`, accept NumPy index syntax,
//! and materialize to NumPy arrays on demand.
//!
//!
//! # Quick start
//!
//! ```python,ignore
//! import zix
//! import numpy as np
//!
//! # Compress a NumPy array into a zix array.
//! a = zix.compact(np.arange(100, dtype=np.float32).reshape(10, 10))
//!
//! # Build a lazy pipeline — no data is read yet.
//! result = (a - a.mean(axis=0)).abs()
//!
//! # Materialize the pipeline into a NumPy array.
//! out = result.numpy()
//!
//! # Save and reload.
//! a.write_to("data.zix")
//! b = zix.read_array("data.zix")
//! assert b.shape == (10, 10)
//! ```
//!
//!
//! # [`Array`]
//!
//! The central type. Wraps compressed array data together with any pending lazy operations.
//! Every operation returns a new `Array`; no data is copied or computed until you ask for
//! output.
//!
//! **Creating an `Array`:**
//!
//! | Function | Description |
//! |---|---|
//! | `zix.compact(...)` | Compress any array-like (NumPy array, list, scalar) into a new zix array. This is the primary constructor. |
//! | `zix.asarray(...)` | Wrap any array-like as a zero-copy zix view without compressing. Useful for mixing plain NumPy data with zix arrays in operations. |
//! | `zix.read_array(...)` | Load a `.zix` file from disk. |
//!
//! **Reading data from an `Array`:**
//!
//! The primary output method is `Array.numpy()`, or equivalently `array[...]`. Both accept
//! the same indexing syntax as NumPy: integers (drop that axis), slices (keep that axis),
//! `...` (fill remaining axes). Note: slices must have step 1; bounds are checked strictly.
//!
//! ```python,ignore
//! a.numpy()            # full array
//! a.numpy(0)           # row 0 (integer drops axis 0)
//! a.numpy(slice(1, 4)) # rows 1–3 (slice keeps axis 0)
//! a[0, 1:3]            # row 0, columns 1–2 (shorthand)
//! a[..., -1]           # last column of any-rank array
//! ```
//!
//! ## Block shape
//!
//! Every zix array stores its data in a grid of fixed-size nd-blocks, each compressed
//! independently. The block shape has a large impact on both read performance and compression
//! ratio: only the blocks that overlap a read request are decompressed, so a block shape that
//! matches your access pattern avoids wasteful work. For example, a `[1, ncols]` block shape
//! means reading a single row decompresses exactly one block; a `[nrows, 1]` shape is
//! similarly efficient for column reads.
//!
//! When no block shape is specified, zix picks one automatically — it greedily expands each
//! dimension (innermost first) until the block byte-size reaches the L1 data cache.
//!
//! You can supply an explicit block shape through [`ArrayParams`]:
//!
//! ```python,ignore
//! a = zix.compact(data, params={"block_shape": [64, 64]})
//! ```
//!
//! After shape-changing operations (`reshape`, `permute_axes`, etc.) the original block
//! layout may no longer match the new access pattern. Call `zix.copy(arr, params=...)` to
//! re-encode with a layout suited to the new shape.
//!
//!
//! # Operations
//!
//! Every operation — arithmetic, comparisons, reductions, shape changes, type casts — returns
//! a new `Array` **view** that wraps the input(s) and records the transformation. No data is read or
//! computed at call time. The deferred work only runs when you ask for output (`.numpy()`, `[...]`,
//! `.write_to()`, `zix.copy()`, etc.).
//!
//! Chains compose without intermediate allocations: the full pipeline is executed in a single
//! pass over the compressed source data, block by block.
//!
//! ```python,ignore
//! # Nothing is read or computed during these calls.
//! a = zix.read_array("data.zix")
//! result = (
//!     a
//!      .astype("float64")
//!      .exp()
//!      .sum(axis=0)
//! )
//!
//! # This single call decompresses, transforms, and materializes the pipeline.
//! out = result.numpy()
//! ```
//!
//!
//! # Persistence
//!
//! Arrays are saved to and loaded from `.zix` files. The format stores metadata (shape, dtype,
//! block layout, codec settings) in a protobuf header followed by the raw compressed block
//! data.
//!
//! ```python,ignore
//! # Write to a file path.
//! zix.write_array(a, "data.zix")
//!
//! # Load back.
//! b = zix.read_array("data.zix")
//!
//! # Memory-mapped read: blocks are paged in from disk on demand.
//! # Fast startup, zero copy, but the file must not be modified while the array is live.
//! c = zix.read_array("data.zix", mmap=True)
//! ```
//!
//! `write_array` accept a file path or any writable binary
//! file-like object. `read_array` accepts a file path or any seekable binary file-like
//! object.
//!
//! A key property: **a lazy array can be written directly without fully materializing it in
//! memory**. The write path compresses block by block, pulling data from the lazy chain on
//! demand. For example, the result of a large matrix operation can be streamed straight to
//! disk.
//!
//!
//! # Limits
//!
//! - Maximum array dimensions: 8.
//! - Maximum inner-shape dimensions for struct dtypes: 4.
//! - Little-endian platforms only.
//!
//!
//! # Disclaimer
//!
//! This project would not exist without the work of several upstream authors and communities.
//! Specifically, this project was greatly inspired by the [C-Blosc2](https://github.com/Blosc/c-blosc2) library.
//! This crate can almost be seen as a port of ideas and natural Rust evolution of C-Blosc2.
//! See the `THANKS.md` at the repository root for a more complete list of contributors and inspirations,
//! and the `NOTICE` file for full attribution and license text.

use pyo3::prelude::*;

mod archive;
mod array;
mod codec;
mod dtype;
mod ops;
mod params;
mod util;

#[pymodule]
mod zix {
    /// The version of the zix library.
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    pub const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    pub use crate::{array::Array, codec::ReadContext, params::ArrayParams};

    #[pymodule_export]
    pub use crate::array::compact;

    #[pymodule_export]
    pub use crate::archive::{read_array, write_array};

    #[pymodule_export]
    pub use crate::ops::{asarray, astype};

    #[pymodule_export]
    pub use crate::ops::copy;

    #[pymodule_export]
    pub use crate::ops::{add, divide, multiply, power, subtract};

    #[pymodule_export]
    pub use crate::ops::{
        bitwise_and, bitwise_left_shift, bitwise_not, bitwise_or, bitwise_right_shift,
        bitwise_rotate_left, bitwise_rotate_right, bitwise_xor, count_ones, count_zeros,
        leading_zeros, logical_and, logical_not, logical_or, logical_xor, reverse_bits, swap_bytes,
        trailing_zeros,
    };

    #[pymodule_export]
    pub use crate::ops::r#where;

    #[pymodule_export]
    pub use crate::ops::{
        equal, greater, greater_equal, less, less_equal, maximum, minimum, not_equal,
    };

    #[pymodule_export]
    pub use crate::ops::{
        broadcast, concatenate, flatten, insert_axis, permute_axes, remove_axis, reshape, squeeze,
        stack, unsqueeze,
    };

    #[pymodule_export]
    pub use crate::ops::{
        absolute, acos, asin, atan, ceil, cos, exp, floor, log, negative, round, signum, sin, sqrt,
        tan,
    };

    #[pymodule_export]
    pub use crate::ops::{is_finite, is_infinite, is_nan};

    #[pymodule_export]
    pub use crate::ops::{all, any, argmax, argmin, max, mean, min, product, std, sum, var};

    #[pymodule_export]
    pub use crate::ops::dtype_sub_field;
}
pub use zix::*;

#[doc(hidden)]
pub mod __private {
    use pyo3_stub_gen::define_stub_info_gatherer;

    define_stub_info_gatherer!(gen_pyi);
}
