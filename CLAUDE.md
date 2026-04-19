# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zix is a high-performance multi-dimensional array library written in Rust with Python bindings. It features lazy evaluation, block-based compressed storage (Zstd), and NumPy-compatible Python bindings via PyO3.

## Crate Structure

- **`zix/`** — Core Rust library (`Array<S>`, dtype system, ops, storage, archive)
- **`zix-macros/`** — Procedural macros used by the core library
- **`zix-python/`** — PyO3 Python bindings (`zix-pyo3` crate, publishes as `zix` Python package)

There is no workspace-level `Cargo.toml`; each crate is built independently.

## Common Commands

```bash
# Build core library
cargo build -p zix

# Build Python extension
cargo build -p zix-pyo3

# Run Rust tests (core)
cargo test -p zix

# Run a single Rust test
cargo test -p zix <test_name>

# Format code (100-char line width, see .rustfmt.toml)
cargo fmt

# Build and install Python package (development mode)
cd zix-python && maturin develop

# Run Python tests
cd zix-python && pytest python/tests/

# Generate Python type stubs (.pyi)
cargo run -p zix-pyo3 --bin gen_pyi
```

## Architecture

### `Array<S: ArrayStorage>` — Generic Core

The central type is `Array<S>` where `S` implements `ArrayStorage`:
- `read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()>`
- `shape() -> &[u64]`, `dtype() -> &Dtype`, `spec() -> ArrayStorageSpec`

Storage backends: `PlainStorage`, `ScalarStorage`, `CompressedStorage`, `BlockStorage`.

### Lazy Evaluation

Shape operations (`Reshape`, `Slice`, `Broadcast`, `PermuteAxes`, `InsertAxes`, `RemoveAxes`) implement `ArrayStorage` as thin wrappers that transform index ranges without copying data. Data is only read when explicitly requested (e.g., `.copy()`). This means operation chains compose generically at the type level.

### Block-Based Storage

Arrays are stored in fixed-size blocks, each independently Zstd-compressed. `BlockStorage` (in `zix/src/storage/block.rs`) tracks per-block byte offsets for efficient random access. A `ReadContext` carries an optional block cache.

### Codec Pipeline

`Input → [ByteShuffle filter] → [Zstd compress] → Block bytes`

Defined in `zix/src/codec.rs`; codec/filter parameters serialized in protobuf headers.

### Type System (`Dtype`)

Defined in `zix/src/dtype.rs`. Supports:
- Scalar types: `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16` (optional), `f32/f64`, `ComplexF32/ComplexF64` (optional), `bool`
- Struct types: named fields with offsets
- Inner shapes: dtypes can have up to 4 inner dimensions

Alignment and itemsize are tracked explicitly for safe raw memory access.

### Serialization

Protocol Buffers (via `prost`) define the archive format under `zix/proto/zix/v1/`. `build.rs` compiles these to Rust at build time. Archive structs live in `zix/src/archive/`.

### Python Bindings (`zix-python`)

PyO3 + `numpy` crate. The Python `Array` class wraps a type-erased `AnyArray` enum. Operations return new `Array` objects. `pyo3-stub-gen` generates `.pyi` stubs via the `gen_pyi` binary. The Python source lives in `zix-python/python/`.

## Key Constraints

- **Little-endian only** — enforced by a compile-time assertion
- **Max 8 array dimensions** (`NDIM_MAX`)
- **Max 4 inner dtype dimensions** (`DTYPE_MAX_NDIM`)
- **Rust edition 2024**
