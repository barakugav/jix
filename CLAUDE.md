# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jix is a multi-dimensional array library with **block-compressed storage** and **lazy-evaluated operations**, written
in Rust with Python bindings. Two independent features: (1) arrays split into a grid of
independently Zstd-compressed nd-blocks, so reads decompress only the touched blocks; and (2) lazy
operation chains where every op returns a new view and the whole pipeline is encoded in the static
type, then runs in a single decompression pass when output is requested.

## Crate Layout - there is NO Cargo workspace

Each crate is built and tested **independently from its own directory**. The root has no
`Cargo.toml`, and `Cargo.lock` files are per-crate (gitignored). `cargo <cmd> -p jix` from the repo
root will NOT work - you must pass `--manifest-path` or `cd` into the crate directory first.
Prefer `--manifest-path`.

| Directory | Cargo package | Purpose |
|-----------|---------------|---------|
| `jix/` | `jix` | Core Rust library - `Array<S>`, dtype system, ops, storage, archive |
| `jix-macros/` | `jix-macros` | Proc-macro crate. Provides `#[derive(Dtyped)]` |
| `jix-py/` | `jix-python` (lib name `jix`) | PyO3 bindings; publishes the `jix` Python package. Depends on `jix` as `jix_core` with `half` + `num-complex` enabled |
| `jix/schema/` | `jix-schema-gen` (`publish = false`) | Standalone protobuf codegen (see Serialization below) |

## Common Commands

Run from within the relevant crate directory.

```bash
# --- Core library (cd jix) ---
cargo build
cargo test --all-features                 # run all Rust tests
cargo test --all-features <test_name>     # run a single test by name substring
cargo hack check --feature-powerset --depth 2   # CI checks the feature powerset

# --- Python bindings (cd jix-py) ---
maturin develop                           # build + install the extension into the active venv
cargo test --all-features --all-targets   # Rust-side tests of the bindings
cargo run --bin generate_pyi              # regenerate the .pyi type stubs
pytest python/tests --numprocesses auto   # Python tests - ALWAYS use --numprocesses auto (pytest-xdist)

# --- Formatting & linting (per crate, plus ruff/ascii from repo root) ---
cargo fmt --all -- --check                 # in each of jix/schema, jix-macros, jix, jix-py
cargo clippy --all-features
ruff --config .ruff.toml format            # Python formatting
ruff --config .ruff.toml check             # Python linting
python scripts/check_only_ascii.py         # see "ASCII-only" constraint below

# --- Regenerate protobuf Rust (requires protoc installed; cd jix/schema) ---
cargo run                                  # rewrites jix/src/archive/schema/_proto_gen/
```

The `.venv` at `{repo-root}/.venv` is normally already activated in the shell - do NOT
prefix commands with `source .venv/bin/activate` by default. Only activate (or create) it
if a command actually fails because the venv is missing or inactive.

Python dev dependencies: before installing them, make sure a venv is activated at
`{repo-root}/.venv`. If it does not exist, create it with uv using Python 3.13:

```bash
uv venv --python 3.13 .venv      # only if .venv does not already exist
source .venv/bin/activate
uv pip install -r scripts/dev_requirements.txt   # maturin, pytest, pytest-xdist, hypothesis, ruff, mkdocs...
```

## Architecture

### `Array<S: ArrayStorage>` - the generic core

`ArrayStorage` (`jix/src/storage/core.rs`) exposes exactly three things: `shape()`, `dtype()`, and
`read_data(index_ranges, buf, ctx)` which reads a rectangular sub-region into a caller buffer.
**Everything** - arithmetic, slicing, reductions, serialization - is built on top of these three.

Storage carries two pieces of compile-time info as associated types:
- **`ElementType`** - either `Ty<T>` (scalar type `T` known at compile time) or `TypeDyn`
  (runtime-only; arrays loaded from disk start here). `ArrayStorageTyped` is the supertrait
  shorthand for `ArrayStorage<ElementType = Ty<T>>`, and **all element-wise ops require it**. Recover
  a typed array from a `TypeDyn` one with `Array::into_typed::<T>()` (runtime-checked against the
  header).
- **`Dimension`** - either `Dim<N>` (ndim known statically) or `DimDyn` (runtime only).

### Lazy evaluation via the type system

Every operation returns `Array<Op<...>>` wrapping its input(s); the type parameter accumulates the
whole pipeline (e.g. `Array<Sum<Add<PermuteAxes<Compact>, Compact>>>`). There is **no runtime
evaluation graph or scheduler - the type IS the execution plan.** Shape ops (`Reshape`, `Slice`,
`Broadcast`, `PermuteAxes`, `InsertAxis`, `RemoveAxis`, ...) just remap index ranges without copying.
Nothing runs until you materialize via `.to_ndarray()`, `.compact()`, or `.write_to_file()`/`.write_to()`
- at which point the compiler-inlined pipeline executes in a single block-by-block read loop.

### Storage backends (`jix/src/storage/`)

- `Compact<T, D>` / `CompactMmap<T, D>` - heap-allocated / memory-mapped block-compressed storage (the
  main backends).
- `Plain<...>` - zero-copy view over a contiguous/strided in-memory buffer, so plain ndarrays can
  participate in the same op chains.
- `ArrayAny` (= `Array<ArrayStorageAny>`) - type-erased (`Arc<dyn ArrayStorage>`) for holding mixed
  storage types in one collection; loses compile-time element-wise ops.

### Block storage & codec pipeline

Arrays are an n-d grid of fixed-size blocks, each compressed independently; `BlockStorage`
(`jix/src/storage/block.rs`) tracks per-block offsets for random access, and a `ReadContext` carries
an optional block cache. When no block shape is given, one is auto-selected according to the CPU cache sizes.
Codec pipeline (`jix/src/codec/`): `raw bytes -> filters (byte-shuffle default, bit-shuffle) -> codec
(zstd) -> stored bytes`, reversed on read. All settings live in `ArrayParams` and are serialized into
the archive, so readers never need to know them in advance.

### Type system (`jix/src/dtype.rs`)

Runtime `Dtype` records kind/size/alignment per element. Supports **scalar dtypes** (`i8..i64`,
`u8..u64`, `f16`, `f32/f64`, `Complex<f32>/Complex<f64>`, `bool`), **struct dtypes** (named fields
with byte offsets, NumPy-style), and an **inner shape** (a small fixed sub-array per element). The
`Dtyped` trait maps a Rust type to its `Dtype`; `#[derive(Dtyped)]` (from `jix-macros`) implements it
for `#[repr(C)]` structs. `f16`/`Complex` live in `jix::scalar` and are gated behind the `half` /
`num-complex` features.

### Serialization (`.jix` files, `jix/src/archive/`)

Archive = protobuf metadata (shape, block shape, codec config) + raw compressed block bytes. Multiple
arrays can be packed back-to-back in one file, each read back by byte offset + length. A lazy view can
be streamed straight to disk without ever materializing the full result.

The `.proto` sources live in `jix/schema/proto/jix/v1/`. They are **not** compiled at build time -
the `jix-schema-gen` crate (`cd jix/schema && cargo run`, needs `protoc`) regenerates committed Rust
into `jix/src/archive/schema/_proto_gen/`. Edit a `.proto` -> rerun the generator -> commit the output.

### Python bindings (`jix-py/`)

PyO3 + the `numpy` crate. The Python `Array` wraps a type-erased enum; operations dispatch over the
runtime dtype (`jix-py/src/ops/common/dispatch.rs`, `dtype_promote.rs`, `broadcast.rs`). `pyo3-stub-gen`
produces the `.pyi` stubs via the `generate_pyi` binary. The generated `jix-py/python/jix/__init__.pyi`
is gitignored (regenerate locally; it won't show up in `git status`). Python source (the thin
`__init__.py` re-export + tests) is under `jix-py/python/`.

## Adding an operation

An op typically exists in two places that must stay in sync: the Rust implementation under
`jix/src/ops/` (e.g. `op1.rs` unary, `op2.rs` binary, `cmp.rs`, `reduction.rs`, `shape_ops/`) and its
Python dispatch wrapper under `jix-py/src/ops/`. After changing the Python surface, regenerate stubs
with `cargo run --bin generate_pyi`.

## Testing Conventions

- **Rust:** tests are inline `#[cfg(test)] mod tests` blocks inside the source files (no top-level
  `tests/` directory). Property tests use `proptest`; shared strategies/assertions live in
  `jix/src/util/test_util.rs`.
- **Python:** `hypothesis` property tests in `jix-py/python/tests/`, with strategies in
  `tests_util.py` that intentionally mirror the Rust `test_util.rs`. Always run under pytest-xdist
  (`--numprocesses auto`). Requires `maturin develop` first.
- Write elementwise reference implementations as plain Python loops, not `np.vectorize`: it is simpler
  and has no measurable effect on test speed.

## Key Constraints

- **ASCII-only source** - `scripts/check_only_ascii.py` fails CI on any non-ASCII byte in a tracked
  file. Use `-` (hyphen), not em-dashes/unicode, in code and docs.
- **Warnings-as-errors is CI-only.** `.github/workflows/ci.yaml` sets `RUSTFLAGS=-D warnings` on each
  build/check/clippy/test step, so warnings - and, via `#![warn(missing_docs)]` in each crate root,
  missing doc comments on public items - are hard errors in CI but only warnings locally. Reproduce
  locally with `RUSTFLAGS="-D warnings" cargo <cmd>`.
- **Little-endian targets only** - enforced by a compile-time assertion.
- **Max 8 array dimensions** (`NDIM_MAX`); **max 4 inner dtype dimensions** (`DTYPE_MAX_NDIM`).
- **Rust edition 2024, MSRV 1.89.0.** Element types must be `Copy + Send + Sync + 'static` and not
  `Drop`.
- Formatting: rustfmt `max_width = 100` (`.rustfmt.toml`); ruff `line-length = 120` (`.ruff.toml`).
- Cargo features (core `jix`): `half` (enables `f16`), `num-complex` (enables `Complex`); default is
  neither. CI checks the full feature powerset, so guard feature-gated code carefully.
