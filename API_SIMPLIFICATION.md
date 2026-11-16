# Core `jix` Rust API - simplification review

Scope: the public Rust API of the `jix` crate (not internals, not the Python package).
Method: read the public surface (rustdoc JSON: 114 public structs, 69 public traits, ~200
public items, plus ~180 inherent methods on `Array`), wrote a small "new user" crate doing
everyday tasks, and prototyped the most promising ideas on branch `experimental/api-simplify`.

The two core ideas (block-compressed nd-storage, type-encoded lazy pipelines) are strong. Most
of the friction is not in those ideas but in what a user must learn *around* them: several
parallel ways to do each thing, an explicit context object, a type-level element-type tag, and
missing NumPy basics that force users into `map` closures and hand-built shapes.

Findings are ordered by how much they block or slow down adoption.

## TL;DR

Hard blockers (some use cases are not possible today):
- **No user-defined data sources** (sec. 5): `ArrayStorage` is sealed, so jix pipelines cannot
  run on top of HDF5/Zarr/object stores/generators.
- **No dtype-dynamic math in Rust** (sec. "Bigger picture"): arrays loaded from files are
  `TypeDyn` and support only shape ops; a tool that processes arbitrary `.jix` files must write
  its own dtype dispatch (jix-py has ~6.5k lines of exactly that).
- **Read-only arrays, no stated format compatibility** (`todo.md`: "support writes",
  "document archive compatibility"). A storage format without a compatibility promise, and
  without a way to fill a large array incrementally, is hard to adopt for datasets. (Not API
  shape, but it gates the advertised data-loader use case.)

Biggest everyday friction (things work, but users fight the API):
- **Binary ops need same-shape arrays**: no scalars, no broadcasting, no `full` (sec. 1) -
  prototyped scalar support.
- **`ReadContext` plumbing, and the easy path is ~29x slower** on small reads (sec. 2) -
  prototyped a per-thread context: 17.5 us -> 0.43 us with no signature change.
- **Several ways to do each thing**: 12 constructors, 12 materializers, ~180 public op-struct
  constructors that duplicate the methods with opposite error behavior (sec. 3, 4).
- **Type-level ceremony**: `Compact<Ty<f32>>`, erased arrays lose `T`, `Compact` is not
  `Clone` (sec. 6, 7) - prototyped a typed `ArrayAny<ET>`.

---

## 1. Binary ops only take same-shape arrays: no scalars, no broadcasting

**Problem.** Every binary op (`+ - * / %`, `maximum`, `greater`, `pow`, `clamp`, `select`, ...)
requires an `Array` of exactly the same shape. Everyday NumPy code becomes awkward:

```rust
// x * 2 + 1                     -> must go through `map`
let y = x.map(|v| v * 2.0 + 1.0);
// relu(x), clip(x, 0, 1), x > 0.5 -> need constant arrays, but there is no `full`/`zeros`
//                                   in the public API (the `Scalar` storage that does this is
//                                   `#[doc(hidden)] __private::Scalar`, used only by jix-py)
// (a - a.mean(1)) / a.std(1)     -> 4 lines, explicit shape plumbing
let shape = a.shape().to_vec();
let mean = a.view().mean(1).insert_dim(1);
let std = a.view().std(1, 0.0).insert_dim(1);
let z = (a.view() - mean.broadcast(&shape)) / std.broadcast(&shape);
```

`map` is fine for arithmetic, but it does not cover comparisons that should produce `bool8`,
`clamp` (which today requires *two full arrays* as bounds), `select` with a constant branch, or
mixing arrays of different ranks.

**Proposal.**
- Accept scalars wherever an array operand is accepted, via one trait:
  `trait IntoOperand<T> { type Storage; fn into_operand(self, shape) -> Result<Array<..>> }`,
  implemented for `T: Dtyped` (becomes a zero-copy broadcast `Scalar` storage) and for
  `Array<S>`. Operators get a second, non-overlapping impl `impl<S, T: Dtyped> Add<T> for
  Array<S>` plus `impl<S> Add<Array<S>> for f32` (one per scalar type) for `2.0 * a`.
- Expose `Array::full(value, shape)` (and `zeros`/`ones` sugar).
- NumPy broadcasting for binary ops (rank alignment + size-1 stretching), so the normalization
  above becomes `(a.view() - a.view().mean(1).insert_dim(1)) / a.std(1, 0.0).insert_dim(1)`, or
  with a `keepdims` reduction variant, `(a - a.mean_keepdims(1)) / a.std_keepdims(1, 0.0)`.
  The Python bindings already implement all of this on top of core
  (`jix-py/src/ops/common/broadcast.rs`); Rust users get none of it. Best done inside the
  `Op2` engine: a broadcast operand is read at the reduced index and its pulled buffer gets
  0-strides, which the elementwise pipeline already supports.

**Prototype (done, compiles, tests pass).** Scalar operands for every binary op, scalar on the
left for operators, `clamp(min, max)` with scalar bounds, `greater(_eq)` with scalars, and
`Array::full`. Bare literals infer correctly (`a + 2.0` -> `f32` for an `f32` array, `i + 3` ->
`i32`, `ints.bitwise_shift_left(2)` -> `u32`), because the operator impls key on `T: Dtyped`
directly rather than on an associated type. Changes:
`jix/src/ops/elementwise/operand.rs` (new), `op2.rs` (macros), `min_max.rs` (`clamp`), `cmp.rs`
(`greater`, `greater_equal`), `storage/mod.rs` (re-export `Scalar`).

```rust
let lin  = a.view() * 2.0 + 1.0;
let inv  = 1.0 / a.view();
let relu = a.view().maximum(0.0);
let unit = a.view().clamp(0.0, 1.0);
let mask = a.view().greater(0.5);
let ones = Array::full(1.0f32, &[2, 3])?;
```

---

## 2. `ReadContext` is required plumbing, and the convenient path is ~29x slower

**Problem.** `to_ndarray_sub`, `to_buf_slice`, `to_buf_strided(_into)`, `compact_with`,
`maybe_compact_with` and `write_to` all take `&ReadContext`. The docs explain it is a perf
optimization ("reuse one across repeated reads"), and that `ReadContext::default()` is always
correct. In fact `Array::read_ctx()` and `ReadContext::default()` are identical today -
`DecoderParams` is an empty struct (`codec/mod.rs:213`) - so the per-array "decoder
configuration" concept has no content, and `read_ctx()` panics on a failure that cannot happen.

Meanwhile the context-free methods (`to_ndarray`, `compact`, `write_to_file`, `maybe_compact`)
build a fresh context per call. Measured on a 16x16 `f32` compact array (release build):

| call | time / call |
|------|-------------|
| `a.to_ndarray()` (fresh context inside) | 17.5 us |
| `a.to_ndarray_sub(full, &ctx)` (reused context) | 0.6 us |
| `ReadContext::default()` alone | 16.6 us (zstd decompressor setup) |

So the easy API is ~29x slower for small reads - exactly the "ML data loader sampling small
chunks" use case the docs advertise. Users must discover the context to get acceptable speed.

**Prototype (done).** A `thread_local!` cached context used by `to_ndarray()` and `compact()`
(`context.rs`, `with_default_context`). No signature changes, all tests pass, and:

| call | before | after |
|------|--------|-------|
| `a.to_ndarray()` on a 16x16 `f32` array | 17.5 us | 0.43 us |

**Proposal.** Use the per-thread context (`ReadContext` is `Send` but not `Sync`, so a
per-thread instance is the natural fit) everywhere, and drop the `context` parameter from every
method that does not return a buffer borrowing from it. Remove `Array::read_ctx()`. For the
one pull-mode API that lends a buffer out of the context (`to_buf_strided`), make the context an
owned reader object instead of a parameter: `let r = a.reader(); let buf = r.read(&idx)?;`.
Expert users who want an isolated pool can still construct a `ReadContext` and use the reader.
Cost: pooled buffers live as long as the thread (cap the pool), and nested reads on one thread
must share the context - which the current design already supports, since every op passes the
same context down. A side benefit: parallel readers (`rayon` over tiles of one shared `&Array`)
get a context per worker for free, instead of managing a `!Sync` context per task.

---

## 3. Too many constructors and materializers - collapse to "source -> array -> compact"

**Problem.** Construction is a matrix of source x destination x params:

- `compact_ndarray`, `compact_ndarray_with`, `compact_ndarray_ptr`,
  `compact_fn`, `compact_fn_with`,
- `plain_ndarray`, `plain_ndarray_with`, `plain_ndarray_ref`, `plain_ndarray_ref_with`,
  `plain_ndarray_view`, `plain_ndarray_view_with`, `plain_ndarray_ptr`,

and materialization has a similar spread: `compact`, `compact_with`, `maybe_compact`,
`maybe_compact_with`, `to_plain`, `to_ndarray`, `to_ndarray_sub`, `to_buf_slice`,
`to_buf_strided`, `to_buf_strided_into`, `write_to_file`, `write_to`.

Observations:
- `compact_ndarray_with(arr, p)` is literally `plain_ndarray_ref(arr).compact_with(p, ctx)`
  (`array/construct.rs:100`). The compact constructors are sugar for "wrap, then compact".
- `plain_ndarray_ref` and `plain_ndarray_view` differ only in how the lifetime is carried;
  ndarray's `AsArray<'a, T, D>` trait accepts both `&ArrayBase` and `ArrayView` in one function.
- The `params` of every `plain_*_with` "never change how the data is stored" and the block shape
  is ignored; they only seed later `compact()` calls. Same for the `params` of all `read_from_*`
  functions. A parameter that does nothing at the call site is confusing.
- `maybe_compact` exists only because `compact()` cannot return the source unchanged: `Compact`
  is not `Clone` (`Array` derives `Clone`, but `Array<Compact<_>>: Clone` does not hold). If
  `Compact`'s block bytes were `Arc`-backed (like `CompactMmap` already is), `compact()` on an
  already-compact array would be a cheap clone and `MaybeCompact` + 2 methods disappear.
- `to_ndarray_sub(&[Range<u64>], &ctx)` and `slice((tuple of ranges))` are two different
  region languages for the same thing (`u64` ranges vs `i64` tuples with negative indexing).
- `to_buf_slice` takes `&mut [u8]`, so its own doc example needs `unsafe` to read into a `Vec`
  of the element type.

**Proposal.**
```rust
// Sources produce cheap views (no compression, no copy):
Array::from_ndarray(arr)            // owned ndarray -> Plain   (TryFrom<ndarray::Array>)
Array::from_view(&arr | arr.view()) // borrowed       -> Plain   (one fn via ndarray::AsArray)
Array::from_fn(shape, f)            // lazy generator (today's FnStorage, made lazy)
unsafe Array::from_raw_parts(ptr, shape, strides, dtype)
// Compression is always the same explicit step:
x.compact()?  /  x.compact_with(&params)?
// Reads:
x.to_ndarray()?                      // whole array
x.slice(s![..]).to_ndarray()?        // a region: one region language (slice), no ctx
x.read_into(&mut dst)?               // dst: &mut [T] (packed) or ndarray::ArrayViewMut
```
That is ~4 constructors instead of 12, and the `_with` doubling only remains where params
matter (compaction and writing). `to_ndarray_sub` can stay as a fast path internally, but the
user-facing region API should be `slice`. Consider accepting `ndarray::s![]` slices directly.

---

## 4. Two parallel op APIs, with opposite error behavior

**Problem.** Every op exists three ways: an `Array` method, `ops::X::new(storage, ..)` and
`ops::X::new_array(array, ..)` - roughly 90 op structs x 2 public constructors = ~180 public
functions with one-line docs ("Constructs a [`X`] storage. See the struct docs..."). The method
form panics on invalid arguments (`.unwrap()` + `#[track_caller]`), the struct form returns
`Result`. So the fallible API exists, but it is hidden behind a different calling style
(`ops::Reshape::new_array(a, &[2, 3])?` instead of `a.reshape(&[2, 3])`), and users who need
graceful errors (shapes from files, user input) must switch styles mid-pipeline. jix-py is the
main user of `new_array`, precisely because it needs `Result`.

**Proposal.** One documented rule, one calling style:
- Programmer errors (dim index out of range, wrong number of dims in a literal) panic, like
  ndarray's `sum_axis(Axis(9))`.
- Data-dependent shape errors return `Result` from the *method* (`reshape`, `broadcast`,
  `concatenate`, `stack`), like ndarray's `into_shape_with_order` / `broadcast`. With
  broadcasting binary ops (section 1), operator shape errors become rare enough to keep
  operators panicking.
- Make the op struct constructors `pub(crate)` (or `#[doc(hidden)]` for jix-py). Keep the
  structs public only as type names.

---

## 5. Users cannot bring their own storage

**Problem.** `ArrayStorage` is sealed, and `OutBuf` (the argument of `read_data`) has no public
constructor. So `read_data`, `Index`, `OutBuf`, `ArrayStorageInfo` are public but unusable:
nobody outside the crate can implement or call them. At the same time there is no way to put
jix's lazy pipeline on top of data that lives elsewhere - HDF5/Zarr/NetCDF files, an object
store, a database, a procedural generator - which is one of the main reasons people reach for a
blosc2/zarr-like library. `compact_fn` is the only escape hatch, and it materializes eagerly.

**Proposal.** Keep the internal `ArrayStorage` sealed (its pull/push/pipeline protocol is an
implementation detail), and add a small, safe, push-only trait that users implement, adapted
internally:

```rust
pub trait Source: Send + Sync {
    type Item: Dtyped;
    fn shape(&self) -> &[u64];
    /// Preferred read granularity (e.g. the source's own chunk shape).
    fn chunk_shape(&self) -> Option<&[u64]> { None }
    /// Fill `out` with the region starting at `start` (`out.shape()` gives the size).
    fn read(&self, start: &[u64], out: ndarray::ArrayViewMut<'_, Self::Item, IxDyn>) -> Result<()>;
}
let a = Array::from_source(my_hdf5_dataset); // lazy, composes with every op
```

Then hide `ArrayStorage::read_data`, `Index`, `OutBuf`, `StridedBuf` from the default docs (or
move them under an `advanced`/`raw` module), and `compact_fn` becomes `from_fn(..).compact()`.

---

## 6. The element-type tag: `Compact<Ty<f32>>`, `Ty`, `TypeDyn`, `ElementType`, `ArrayStorageTyped`

**Problem.** To know if `T` is static, users meet five concepts (`ElementType`, `Ty<T>`,
`TypeDyn`, `ArrayStorageTyped`, `S::Item`) and three conversions (`into_type`, `into_typed`,
`into_type_dyn`), and types read `Array<Compact<Ty<f32>>>`. The `Ty<_>` wrapper carries no
information beyond `T`.

**Proposal.** Implement `ElementType` directly for every `T: Dtyped`, plus one runtime marker
(`TypeDyn`, maybe renamed `Dyn`). Then types read `Array<Compact<f32>>` and
`Array<Compact<Dyn>>`, `ArrayStorageTyped` is a blanket impl over `ElementType: Dtyped`, and
`into_type::<NewET>` covers both `into_typed::<f32>()` and `into_type_dyn()` (keep those two as
sugar, drop `into_type` from the docs). Validated in a stand-alone crate that mirrors jix's
traits: `impl<T: Dtyped> ElementType for T` + `impl ElementType for TypeDyn` is accepted by
coherence (both the trait and the marker are local, and downstream crates cannot implement
`Dtyped` for `TypeDyn`). The change is mechanical but touches most files (`Ty<` appears
everywhere), so it was not applied to the real crate.

---

## 7. Erased arrays lose their element type; most storages are not `Clone`

**Problem.**
- `into_any()` always returns `Array<ArrayStorageAny>` with `ElementType = TypeDyn`, even
  from an `f32` pipeline. The common uses of erasure - storing a pipeline in a struct field,
  returning one of two pipelines from an `if`, a `Vec` of heterogeneous `f32` sources - then
  need a runtime `into_typed::<f32>()?` to get the ops back.
- `Compact` and `Plain` are not `Clone`, so reuse goes through `view()` (`Array<&S>`), which
  cannot be returned alongside its source, stored, or erased (`into_any` needs `'static`).
  A function taking `CompactArray<f32>` cannot be called twice with the same array.

- Naming a pipeline is not realistic. The normalization from section 1 has this type (396
  characters, module paths trimmed), which is why erasure is the only way to store one:
  `Array<Div<Sub<&Compact<Ty<f32>>, Broadcast<InsertDim<Mean<&Compact<Ty<f32>>>>>>,
  Broadcast<InsertDim<StandardDeviation<&Compact<Ty<f32>>>>>>>`

Both in-repo consumers already hand-roll a typed erased array: the core test suite
(`TestArray<T> = Array<IntoType<ArrayStorageAny, Ty<T>>>`, `util/test_util.rs`) and jix-py
(`TypedOperand<T>`, `jix-py/src/ops/common/dispatch.rs`).

**Prototype (done).** `ArrayStorageAny<ET = TypeDyn>`, `into_any()` keeps `S::ElementType`,
and `into_typed` on an erased array re-tags it in place instead of wrapping it in `IntoType`.
The test suite's `TestArray<T>` becomes simply `ArrayAny<Ty<T>>`. User code:

```rust
struct Model { features: ArrayAny<Ty<f32>> }
let features = if log_scale { a.ln().into_any() } else { a.into_any() };
let m = Model { features };
let total = m.features.clone().sum([0, 1]).item()?;   // no into_typed::<f32>()? needed
```

**Proposal.** `ArrayStorageAny<ET = TypeDyn>` that carries the static element type
(`type ArrayAny<ET = TypeDyn> = Array<ArrayStorageAny<ET>>`; with section 6 that reads
`ArrayAny<f32>`), and `into_any()` keeps `S::ElementType`. Make the owned leaf storages cheap
to clone (`Arc`-backed block table and plain allocation), so `Array<Compact<f32>>: Clone`.

---

## 8. File I/O signatures

**Problem.**
- Paths are `&Path`, so `a.write_to_file("out.jix")` does not compile; the README itself
  writes `Path::new("result.jix")`. Use `impl AsRef<Path>` (prototyped; source-compatible for
  existing `&Path` / `&PathBuf` callers).
- `write_to_file` fails if the file exists (`File::create_new`). That is surprising next to
  `std::fs::write` / `numpy.save`; at least make it explicit (`write_to_file` overwrites,
  `create_new` is an option).
- `read_from_file_mmap(path, offset, len, params, validation)` has no short form even though the
  whole-file case is the common one, and three of its five arguments are usually defaults
  (`params` does not affect the loaded data at all).
- `write_to(writer, params, ctx)` silently ignores `params` when the source is compact.
- Loading always yields `TypeDyn`, so typed loads read
  `Array::read_from_file(p)?.into_typed::<f32>()?`.

**Proposal.** An options struct, with typed and untyped entry points:
```rust
let a = jix::open::<f32>("data.jix")?;                    // heap, typed, whole file
let a = jix::open_dyn("data.jix")?;                       // TypeDyn
let a = unsafe { ReadOptions::new().mmap().range(off, len).validation(..).open::<f32>(p)? };
a.save("out.jix")?;                                       // overwrite; save_with(&params)
```

---

## 9. Missing everyday basics

- **Constructors:** `full`, `zeros`, `ones`, `arange`, `linspace` (only `compact_fn` exists).
- **Scalars out:** a full reduction to a value is
  `a.sum([0, 1]).to_ndarray::<Ix0>()?.into_scalar()`. Add `item()` (already in `todo.md`;
  prototyped: `a.sum([0, 1]).item()?`) and a reduce-all form (`sum(..)` with a `RangeFull` /
  `All` `DimsArg`).
- **Single element:** `get(&[i, j])` instead of
  `to_ndarray_sub::<Ix2>(&[i..i+1, j..j+1], &ctx)?[[0, 0]]`.
- **Integer indexing in `slice`** that drops the dim (today `slice((1..2, ..)).remove_dim(0)`).
- **keepdims** on reductions (see section 1).
- `to_ndarray()` cannot infer `D` when the result is just printed or passed to a generic, so
  users write `to_ndarray::<IxDyn>()`; a `to_ndarray_dyn()` (or making the dyn version the
  default name) removes the turbofish from most call sites.

---

## 10. Smaller cleanups of the public surface

- **`jix::scalar::*` (60 traits) leak `fearless_simd`.** Their `*_bulk` methods take
  `S: fearless_simd::Simd`, making that crate's major version part of jix's semver. Seal them
  or hide the bulk methods (`#[doc(hidden)]`), and offer a few umbrella traits for generic user
  code (`Float`, `Integer`, `Numeric`) so a generic function does not need
  `S::Item: jix::scalar::Exp<Output: Dtyped> + jix::scalar::Sum<..> + ...`.
- **Three `Add`s:** `std::ops::Add`, `jix::ops::Add` (op struct) and `jix::scalar::Add`
  (kernel trait) - same for `Sub`, `Mul`, `Div`, `Not`, `Sum`, ... Import collisions and
  ambiguous docs. Rename the kernel traits (`ScalarAdd`, or keep them in `scalar::kernels`).
- **`map2` vs `map_multiple`:** `map_multiple((a, b), |(x, y)| ..)` subsumes `map2`; one
  function (`ops::zip_map`?) is enough.
- **`ArrayParams` fallible setters** (`block_shape`, `block_shape_fixed_dims`, `level`,
  `filters` return `Result<Self>`) force `?` mid-builder:
  `ArrayParams::new().block_shape(&[64, 64])?`. Validate when the params are used instead.
  Expert knobs (`block_shape_fixed_dims`, `read_size`, `block_size`) could move to an
  `advanced` sub-builder; `read_size` is a read-time knob and belongs with the reader/context.
- **`jix::error::Result` is `pub(crate)`**, so users cannot write `jix::Result<T>`
  (prototyped: now `pub` and re-exported at the root).
- **Unusable public items:** `OutBuf`, `Index::new`, `ArrayStorageInfo`,
  `ArrayStorage::read_data/info`, `ArraySequenceElementType`/`ArraySequenceTyped` at the crate
  root - see section 5.
- **`_dyn` twins:** `dtype_sub_field` / `dtype_sub_field_dyn`, `transmute_elements` /
  `transmute_elements_dyn`; with section 6 these can be one generic method each
  (`field::<f32>("x")`, `field::<Dyn>("x")`).
- **`is_compact`, `storage`, `from_storage`, `into_storage`** are documented "rarely needed";
  `from_storage` is useless outside the crate since users cannot construct storages.

---

## Bigger-picture suggestion: a first-class dynamic API

Many Rust consumers do not know the dtype at compile time: CLI tools, servers, anything that
opens arbitrary `.jix` files. Today they get `TypeDyn` arrays with only shape ops, and must write
their own dtype dispatch to do math. jix-py already contains a complete, well-tested dynamic
layer (dtype dispatch, promotion, broadcasting, scalar operands: ~6.5k lines in
`jix-py/src/ops`) - all of it PyO3-independent in spirit. Moving it into core (or a `jix-dyn`
crate) as `DynArray` with `a.add(&b)?`-style runtime-dispatched methods would give Rust users
the NumPy-like experience, shrink jix-py to a thin wrapper, and leave the static `Array<S>`
pipeline as the zero-overhead expert API.

---

## What is on the `experimental/api-simplify` branch

Prototypes only, to check feasibility and feel the resulting API; doc comments are minimal and
marked `EXPERIMENT`. The full core test suite (1402 unit tests + 156 doctests) passes with the
experiments applied.

| Commit | Content | Sections |
|--------|---------|----------|
| scalar operands | `IntoOperand<T>`, `array op scalar` / `scalar op array` operators, scalar `maximum`/`minimum`/`greater`/`clamp`/..., public `storage::Scalar`, `Array::full` | 1 |
| per-thread context etc. | thread-local `ReadContext` for `to_ndarray`/`compact`, `impl AsRef<Path>`, public `jix::Result`, `Array::item()` | 2, 8, 9, 10 |
| typed erased arrays | `ArrayStorageAny<ET>`, `ArrayAny<ET = TypeDyn>`, `into_any()` keeps the element type; jix-py migrated (`.into_any()` -> `.into_any().into_type_dyn()` at 43 sites, `TypedOperand<T>` = `Array<ArrayStorageAny<Ty<T>>>`) | 7 |

jix-py compiles unchanged against the first two commits (they are source-compatible); only the
typed `into_any()` needed the mechanical migration above.

Not applied (validated separately or design only): dropping `Ty<T>` (coherence checked in a
stand-alone crate), NumPy broadcasting in the `Op2` engine, constructor consolidation, the
`Source` trait, the fallibility rule, the dynamic API.

## Suggested order

1. Scalars + broadcasting + `full`/`item` (largest everyday win; prototype exists).
2. Implicit per-thread context (simplifies ~6 signatures and fixes a 29x perf trap).
3. Constructor/materializer consolidation and `AsRef<Path>`/I/O options (mostly deletions).
4. One fallibility rule; hide op-struct constructors.
5. Drop `Ty<T>`; typed `ArrayAny<T>`; cloneable leaf storages.
6. `Source` trait for user storages.
7. Dynamic API in core.
