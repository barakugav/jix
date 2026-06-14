#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Multi-dimensional array library with block-compressed, lazy-evaluated storage.
//!
//! Jix arrays behave like regular n-dimensional arrays, but store their data in compressed blocks
//! and decode them on demand. The library is designed around two ideas:
//!
//! - **Block-based compression** - the array is divided into an n-dimensional grid of fixed-size
//!   blocks, each compressed independently. Only the blocks touched by a read request are
//!   decompressed, enabling efficient random-access into large arrays without loading everything
//!   into memory.
//!
//! - **Lazy operation chains** - every operation (arithmetic, shape manipulation, type cast,
//!   reduction, ...) returns a new [`Array<OpStorage<...>>`](Array), rather than a materialized result.
//!   Computation only runs when data is explicitly requested (e.g. via
//!   [`.to_ndarray()`](Array::to_ndarray) or [`.compact()`](Array::compact)). Because the full
//!   operation chain is encoded in the static type, the compiler can inline the entire pipeline
//!   into a single read loop with no virtual dispatch very efficiently.
//!
//! # Quick start
//!
//! ```
//! use jix::dtype::Dtyped;
//! use jix::Array;
//! use ndarray::array;
//!
//! // Compress a 2-D f32 ndarray into a block-compressed Array<Compact>.
//! let a = Array::compact_ndarray(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
//! assert_eq!(a.shape(), &[2, 3]);
//! assert_eq!(a.dtype(), &f32::DTYPE);
//!
//! // Decompress into a regular `ndarray::Array<f32>` array.
//! let decompressed = a.to_ndarray()?;
//! assert_eq!(decompressed[[0, 0]], 1.5);
//!
//! // Build a lazy operation pipeline - no data is read yet.
//! let ones = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
//! let result = a                  // Array<Compact>
//!     .exp()                      // Array<Exp<Compact>>
//!     .floor()                    // Array<Floor<Exp<Compact>>>
//!     .map(|x| x * 2.0f32)        // Array<Map<Floor<...>>>
//!     + ones;                     // Array<Add<Map<...>, Compact>>
//!
//! // materialize the pipeline into a new compressed Array<Compact>
//! let result_compressed = result.compact()?;
//! // or alternatively, materialize into an uncompressed ndarray::Array
//! let result_decompressed = result.to_ndarray()?;
//! // or alternatively, materialize directly to disk without ever holding the
//! // full result in memory - blocks are decompressed, transformed, re-compressed
//! // and written one at a time.
//! let tmp_dir = tempfile::tempdir()?;
//! result.write_to_file(&tmp_dir.path().join("result.jix"))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Core type: `Array<S>`
//!
//! [`Array<S>`](Array) is generic over its storage backend [`S: ArrayStorage`]. The storage
//! trait has three methods: `shape()`, `dtype()`, and `read_data()`. Everything else - slicing,
//! arithmetic, reductions, serialization - is implemented on top of those three.
//!
//! The type parameter `S` carries the full operation chain at compile time:
//!
//! ```text
//! Array<Compact>
//!   .neg()                 -> Array<Neg<Compact>>
//!   .reshape_view(...)     -> Array<Reshape<Neg<Compact>>>
//!   .permute_axes(&[1, 0]) -> Array<PermuteAxes<Reshape<...>>>
//!   .add(other)            -> Array<Add<PermuteAxes<...>, Compact>>
//!   .sum(0).               -> Array<Sum<Add<...>>>
//!   .compact()?            -> Array<Compact>  - materialize
//! ```
//!
//! There is no runtime evaluation graph or scheduler. The type system *is* the execution plan.
//!
//! # Storage backends
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Array<Compact<...>>`](storage::Compact) | Heap-allocated block-compressed array. The main backend of the library. |
//! | [`Array<Op<...>>`](ops) | Lazy operation views defined in [`ops`]. Wrap one or more arrays; apply their transformation on each read. |
//! | [`Array<Plain<...>>`](storage::Plain) | Zero-copy view of a contiguous or strided in-memory buffer. Created by [`Array::plain_ndarray_ref`]. |
//!
//! # Operations
//!
//! All operations live in [`ops`] and are also available as methods on [`Array<S>`](Array).
//! The support list of operations is still growing, but includes:
//!
//! **Element-wise unary** - `neg`, `abs`, `exp`, `ln`, `sqrt`, `floor`, `ceil`,
//! `round`, `sign`, `sin`, `cos`, `tan`, ...
//!
//! **Element-wise binary** (array op array, via `+`, `-`, `*`, `/`,
//! operator overloads and named methods) - `add`, `sub`, `mul`, `div`, `pow`, `minimum`,
//! `maximum`, ...
//!
//! For scaling, shifting, or any element-wise transform involving a constant value, use
//! [`map`](Array::map) - e.g. `a.map(|x| x * 2.0f32)` rather than `a * 2.0f32`.
//!
//! **Comparisons** - `equal`, `not_equal`, `greater`, `greater_equal`, `less`, ...
//!
//! **Logical** - `not`, `logical_and`, `logical_or`, `logical_xor`
//!
//! **Bitwise** - `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`
//!
//! **Reductions** - `sum`, `mean`, `min`, `max`, `argmin`, `argmax`, `any`, `all`, ...
//!
//! **Shape operations** - `reshape`, `slice`, `permute_axes`, `broadcast`,
//! `insert_axis`, `remove_axis`, `concatenate`, `stack`
//!
//! **Type cast** - `cast::<T>()` converts each element to T.
//!
//!
//! # Element types
//!
//! Jix tracks element types at two levels:
//!
//! **Runtime - [`Dtype`](dtype::Dtype)**
//!
//! Every array carries a runtime [`Dtype`](dtype::Dtype) that records the kind, size, and
//! alignment of each element. Dtypes come in two flavors:
//!
//! - *Scalar dtypes* cover all primitive numeric and boolean types:
//!   `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`,
//!   `Complex<f32>`, `Complex<f64>`, `bool`.
//! - *Struct dtypes* group named fields with explicit byte offsets (C aligned or packed layout),
//!   enabling NumPy-style structured dtypes.
//!
//! Both flavors support an *inner shape*: a small fixed-size sub-array baked into each logical
//! element (e.g. `[f32; 3]` has dtype shape `[3]` and itemsize `12`).
//!
//! The [`Dtyped`](dtype::Dtyped) trait maps a Rust type to its `Dtype` at compile time.
//! Implement it for your own `#[repr(C)]` structs:
//!
//! ```rust,ignore
//! use jix::dtype::{Dtype, Dtyped};
//!
//! #[derive(Copy, Clone, Dtyped)]
//! #[repr(C)]
//! struct Pixel { r: u8, g: u8, b: u8 }
//!
//! assert_eq!(Pixel::DTYPE.itemsize(), 3);
//! let fields = Pixel::DTYPE.fields().unwrap();
//! assert_eq!(fields[0].0, "r");
//! ```
//!
//! **Compile-time - [`ElementType`], [`Ty<T>`](Ty), [`TypeDyn`]**
//!
//! In addition to the runtime [`Dtype`](dtype::Dtype), the storage type parameter `S` carries the
//! element type at the *type level* via `S::ElementType`:
//!
//! - [`Ty<T>`](Ty) - the scalar element type `T` is known at compile time.
//!   Arrays constructed from typed sources carry this automatically (e.g.
//!   `Array::compact_ndarray(&array![1.0f32, 2.0])` yields `Array<Compact<Ty<f32>, Dim<1>>>`).
//!   Most of the element-wise operations require `Ty<T>`, as they are bounded by scalar trait of `T`.
//!
//! - [`TypeDyn`] - the element type is only known at runtime. Arrays loaded
//!   from disk start with this (`Array<Compact<TypeDyn, DimDyn>>`). Call
//!   [`Array::into_typed::<T>()`](Array::into_typed) to assert the expected element type (checked
//!   against the file header at runtime) and unlock element-wise operations:
//!
//! ```no_run
//! use std::path::Path;
//!
//! use jix::{Array, ArrayParams};
//!
//! let src = Array::read_from_file(Path::new("data.jix"), ArrayParams::default())?;
//! // src: Array<Compact<TypeDyn, DimDyn>> - element type unknown at compile time
//! // src: Array<S::ElementType = TypeDyn>
//!
//! let typed = src.into_typed::<f32>()?; // runtime check: dtype must be f32
//! // typed: Array<S::ElementType = Ty<f32>>
//! let result = typed.exp().sum(0).compact()?;
//! # Ok::<(), jix::Error>(())
//! ```
//!
//!
//! # Dimension types
//!
//! Every [`ArrayStorage`] carries an associated `type Dimension:
//! Dimension` that records the number of axes at the type level. When the ndim is known
//! statically, it is [`Dim<N>`]: the const generic `N` is the axis count and is visible to the
//! compiler. When the ndim is only known at runtime (e.g. arrays loaded from files), it is
//! [`DimDyn`]: a stack-allocated array of sizes with capacity [`NDIM_MAX`].
//! The dimension type propagates through every shape-changing operation automatically.
//! See [`Dimension`] for details.
//!
//!
//! # Codec pipeline
//!
//! Each compressed block passes through the following pipeline on write:
//!
//! ```text
//! raw element bytes  ->  filters  ->  codec compress  ->  stored bytes
//! ```
//!
//! On read, the pipeline is reversed. Filters include the byte-shuffle filter (enabled by default),
//! and bit shuffle, improving the codec's ratio on numerical data.
//!
//! Codec settings are controlled via [`ArrayParams`]:
//!
//! - [`encoder_params`](ArrayParams::encoder_params) - codec choice, compression level, filter.
//! - [`decoder_params`](ArrayParams::decoder_params) - decoder configuration.
//!
//! The codec and filter configuration is serialized into the array archive, so readers never need
//! to know ahead of time which settings were used.
//!
//!
//! # Block layout and performance
//!
//! The n-dimensional block shape has a large impact on both compression ratio and read
//! performance. If the access pattern is known in advance, providing a matching block shape can
//! improve performance significantly.
//!
//! When no block shape is specified, the library automatically selects one that fits within the
//! L1 data cache: starting from a block shape of all-ones, it greedily increases each dimension
//! (from last to first) as long as the block byte-size does not exceed the target size.
//!
//! [`ArrayParams`] groups all layout and codec settings. Unset fields are inherited from the
//! source array when copying.
//!
//! For tile-at-a-time access patterns:
//!
//! ```
//! use jix::{Array, ArrayParams};
//!
//! let data = ndarray::Array2::<f32>::zeros((512, 512));
//!
//! // Store with 64*64 blocks - one decompression per tile.
//! let mut params = ArrayParams::new();
//! params.block_shape(&[64, 64]);
//! let array = Array::compact_ndarray_with(&data, params)?;
//!
//! let context = array.read_ctx();
//! for tile_row in 0..7 {
//!     for tile_col in 0..7 {
//!         let row_range = (tile_row * 64)..((tile_row + 2) * 64);
//!         let col_range = (tile_col * 64)..((tile_col + 2) * 64);
//!         let tile = array.to_ndarray_sub(&[row_range, col_range], &context)?;
//!         println!("tile ({tile_row},{tile_col}) sum: {}", tile.sum());
//!     }
//! }
//! # Ok::<(), jix::Error>(())
//! ```
//!
//! Shape-changing operations (`reshape_view`, `permute_axes`, `broadcast`) remap how output indices
//! translate to positions in the underlying blocks. When the new layout crosses block boundaries
//! that the original layout respected, a single read may decompress many more blocks than
//! needed.
//!
//! To avoid this, call [`.compact()`](Array::compact) (or the eager variant `reshape`)
//! after a shape change to re-encode with a freshly derived block shape:
//!
//! ```
//! use jix::{Array, ArrayParams};
//!
//! // Compress with column-friendly blocks.
//! let mut params = ArrayParams::new();
//! params.block_shape(&[64, 64]);
//! let a = Array::compact_ndarray_with(&ndarray::Array2::<f32>::zeros((1024, 1024)), params)?;
//!
//! // Transpose and re-encode with row-friendly blocks.
//! let mut out_params = ArrayParams::new();
//! out_params.block_shape(&[128, 128]);
//! let ctx = a.read_ctx();
//! let transposed = a.permute_axes(&[1, 0]).compact_with(out_params, &ctx)?;
//! # Ok::<(), jix::Error>(())
//! ```
//!
//!
//! # Serialization (`.jix` files)
//!
//! Arrays are serialized to a binary archive format (`.jix`). The format is defined with a mix of
//! protobuf for metadata such as the array shape, block shape, codec configuration, and a raw binary
//! format for the compressed block data.
//! Multiple arrays can be packed into a single file back-to-back; each is read back
//! independently using a byte offset and length.
//!
//! The primary I/O methods are on [`Array`]:
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`write_to_file`](Array::write_to_file) | Write to a new file. |
//! | [`read_from_file`](Array::read_from_file) | Load from file into heap-allocated storage. |
//! | [`read_from_file_mmap`](Array::read_from_file_mmap) | Memory-map a file for zero-copy block access. |
//!
//! A key property: **a lazy view array can be written directly to a file without ever
//! materializing the full result in memory**. The write path compresses block by block, reading
//! from the source lazily:
//!
//! ```
//! use std::fs::File;
//! use std::io::BufWriter;
//!
//! use jix::{ArchiveValidation, Array, ArrayParams};
//! use ndarray::array;
//!
//! let tmp_dir = tempfile::tempdir()?;
//! let path = tmp_dir.path().join("large.jix");
//! Array::compact_ndarray(&array![[2.3_f32, 6.99], [-99.1, 0.0]])?.write_to_file(&path)?;
//! let len = std::fs::metadata(&path)?.len();
//!
//! // Memory-map the source - blocks are paged in on demand.
//! // Safety: the file is not modified while `src` is live.
//! let src = unsafe {
//!     Array::read_from_file_mmap(
//!         &path,
//!         0,
//!         len,
//!         ArrayParams::default(),
//!         ArchiveValidation::default(),
//!     )?
//! };
//!
//! // Build a lazy pipeline over the mmap'd data.
//! let processed = src.into_typed::<f32>()?.exp().map(|x| x + 1.0f32);
//!
//! // Streaming write: blocks are decompressed, transformed, and re-compressed one at a time.
//! processed.write_to(BufWriter::new(File::create(
//!     tmp_dir.path().join("modified.jix"),
//! )?))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//!
//! # Limits
//!
//! - Maximum array dimensions: [`NDIM_MAX`] (8).
//! - Maximum dtype inner-shape dimensions: [`dtype::DTYPE_MAX_NDIM`] (4).
//! - Little-endian targets only - enforced by a compile-time assertion.
//! - Element types must implement [`Dtyped`](dtype::Dtyped); they must be `Copy + Send + Sync +
//!   'static` and must not implement `Drop`.
//!
//!
//! # Disclaimer
//!
//! This project would not exist without the work of several upstream authors and communities.
//! Specifically, this project was greatly inspired by the [C-Blosc2](https://github.com/Blosc/c-blosc2) library.
//! This crate can almost be seen as a port of ideas and natural Rust evolution of C-Blosc2.
//! See the `THANKS.md` at the repository root for a more complete list of contributors and inspirations,
//! and the `NOTICE` file for full attribution and license text.

mod array;
pub use array::Array;

pub mod codec;
pub mod dtype;
mod params;
pub use params::ArrayParams;

pub mod scalar;

pub mod storage;
pub use storage::core::ArrayStorage;

mod archive;
pub use archive::ArchiveValidation;

pub mod ops;

mod dimension;
mod element_type;
pub use dimension::*;
pub use element_type::*;

mod util;
pub use util::ArraySequence;

mod error;
pub use error::{Error, ErrorKind};

/// A fully type-erased array whose storage backend is hidden behind an `Arc<dyn ArrayStorage>`.
///
/// Use `ArrayAny` when you need to hold arrays of different concrete storage types in the same
/// place - for example a `Vec<ArrayAny>` mixing on-disk, in-memory, and lazy views. All runtime
/// metadata (shape, dtype) is available; element-wise operations that require a known scalar type
/// at compile time are not.
///
/// Create one with [`Array::into_any`] or [`ArrayStorageAny::new`](storage::ArrayStorageAny::new).
pub type ArrayAny = Array<storage::ArrayStorageAny>;

#[doc(hidden)]
pub mod __private {
    pub use crate::storage::scalar::Scalar;
}
