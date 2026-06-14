# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jix is a high-performance multi-dimensional array library written in Rust with Python bindings. It features lazy evaluation, block-based compressed storage (Zstd), and NumPy-compatible Python bindings via PyO3.

## Crate Structure

- **`jix/`** - Core Rust library (`Array<S>`, dtype system, ops, storage, archive)
- **`jix-macros/`** - Procedural macros used by the core library
- **`jix-py/`** - PyO3 Python bindings (`jix-python` crate, publishes as `jix` Python package)

There is no workspace-level `Cargo.toml`; each crate is built independently.

## Common Commands

```bash
# Build core library
cargo build -p jix

# Build Python extension
cargo build -p jix-python

# Run Rust tests (core)
cargo test -p jix

# Run a single Rust test
cargo test -p jix <test_name>

# Format code (100-char line width, see .rustfmt.toml)
cargo fmt

# Build and install Python package (development mode)
cd jix-py && maturin develop

# Run Python tests
cd jix-py && pytest python/tests/

# Generate Python type stubs (.pyi)
cargo run -p jix-python --bin gen_pyi
```

## Architecture

### `Array<S: ArrayStorage>` - Generic Core

The central type is `Array<S>` where `S` implements `ArrayStorage`:
- `read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()>`
- `shape() -> &[u64]`, `dtype() -> &Dtype`

`ArrayStorage` has two associated types:
- `type ElementType: ElementType` - compile-time element type, either `Ty<T>` (concrete scalar known at compile time) or `TypeDyn` (runtime only, for arrays loaded from disk).
- `type Dimension: Dimension` - compile-time dimension, either `Dim<N>` (known statically) or `DimDyn`.

`ArrayStorageTyped` is a supertrait shorthand for `ArrayStorage<ElementType = Ty<T>>`. All element-wise operations (arithmetic, comparisons, cast, reductions) require it.

Storage backends: `Compact<T, D>` (heap-allocated block-compressed), `CompactMmap<T, D>` (memory-mapped), `Plain<A, T, D>` (uncompressed buffer).

### Lazy Evaluation

Shape operations (`Reshape`, `Slice`, `Broadcast`, `PermuteAxes`, `InsertAxis`, `RemoveAxis`) implement `ArrayStorage` as thin wrappers that transform index ranges without copying data. Data is only read when explicitly requested (e.g., `.to_ndarray()`). This means operation chains compose generically at the type level.

### Block-Based Storage

Arrays are stored in fixed-size blocks, each independently Zstd-compressed. `BlockStorage` (in `jix/src/storage/block.rs`) tracks per-block byte offsets for efficient random access. A `ReadContext` carries an optional block cache.

### Codec Pipeline

`Input -> [ByteShuffle filter] -> [Zstd compress] -> Block bytes`

Defined in `jix/src/codec.rs`; codec/filter parameters serialized in protobuf headers.

### Type System

**Runtime element type - `Dtype`** (in `jix/src/dtype.rs`):
- Scalar types: `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16` (optional), `f32/f64`, `Complex<f32>/Complex<f64>` (optional), `bool`
- Struct types: named fields with offsets
- Inner shapes: dtypes can have up to 4 inner dimensions
- Alignment and itemsize tracked for safe raw memory access

**Compile-time element type - `ElementType`** (in `jix/src/storage/mod.rs`):
- `Ty<T>` - concrete element type `T` known at compile time; enables all element-wise ops
- `TypeDyn` - runtime-only; arrays from disk start here; call `Array::to_typed::<T>()` to recover `Ty<T>`
- `f16` and `Complex<T>` live in `jix::scalar` (previously they were in `jix::dtype`)

### Serialization

Protocol Buffers (via `prost`) define the archive format under `jix/proto/jix/v1/`. `build.rs` compiles these to Rust at build time. Archive structs live in `jix/src/archive/`.

### Python Bindings (`jix-py`)

PyO3 + `numpy` crate. The Python `Array` class wraps a type-erased `AnyArray` enum. Operations return new `Array` objects. `pyo3-stub-gen` generates `.pyi` stubs via the `gen_pyi` binary. The Python source lives in `jix-py/python/`.

## Developer Guides

- [Testing guide](docs/dev/tests.md) - test crates, shared utilities, when to use proptest, macros, and how to test a new op

## Key Constraints

- **Little-endian only** - enforced by a compile-time assertion
- **Max 8 array dimensions** (`NDIM_MAX`)
- **Max 4 inner dtype dimensions** (`DTYPE_MAX_NDIM`)
- **Rust edition 2024**

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep - these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
