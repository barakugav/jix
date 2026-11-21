# Core `jix` Rust API - simplification review

Scope: the public Rust API of the `jix` crate on `main` at `d6005d4` (version 0.1.0); not
internals, not the Python package. Method: the public surface from rustdoc JSON, a small "new
user" crate doing everyday tasks against `main`, compile checks of common snippets, and
benchmarks of the read paths. The previous version of this review was against `468201e`; the
numbers below compare the two, measured on the same VM.

Section 2 (`ReadContext`) has a deep dive at the end, with measurements and a survey of how other
libraries handle per-thread resources.

## Stats: `468201e` -> `d6005d4`

Public surface (from rustdoc JSON, hidden items excluded):

| | `468201e` | `d6005d4` |
|---|---|---|
| public structs / traits | 115 / 73 | 126 / 73 |
| `Array` inherent methods | 119 | 118 |
| structs in `jix::ops` | 92 | 102 (8 `*Scalar` ops + `Lhs` / `Rhs`) |
| public op constructors `X::new` + `X::new_array` | 92 + 89 | 100 + 97 |
| `Array` constructors / read-materialize methods / file readers | 9 / 12 / 4 | 9 / 12 / 4 |
| methods taking a `ReadContext` | 7, required | 3, optional (`to_buf_*`) |
| binary ops taking a scalar operand | none | the 8 operators `+ - * / % & \| ^`; none of the 22 named binary methods |
| visible `*_bulk` methods with `fearless_simd` in the signature | 373 | 373 |
| jix-py calls to `X::new_array` | 34 | 34 |
| `jix/src` lines (excluding generated protobuf) | 59.3k | 59.9k |

Read paths (16x16 `f32` compact array unless noted, release, median of 15 interleaved rounds,
both commits on the same VM):

| call | `468201e` | `d6005d4` |
|---|---|---|
| `to_ndarray()` | 16.1 us | 0.40 us |
| `get(&[3, 3])` (`get(.., &a.read_ctx())` before) | 16.1 us | 0.40 us |
| `to_buf_slice(..)`, context per call (`&a.read_ctx()` before, `None` now) | 16.0 us | 0.38 us |
| `to_buf_slice(..)`, one reused context | 0.33 us | 0.33 us |
| `to_buf_slice(.., Some(&ReadContext::new_for_array(&a)?))` per call | - | 15.9 us |
| 64x64 tile of a 4096x4096 array, context per call | 147 us | 129 us |
| 64x64 tile of a 4096x4096 array, reused context | 128 us | 129 us |

The convenience reads are 40x faster and within 15% of a reused context. The one slow path left
is creating a context with `ReadContext::new` / `new_for_array` per read (sec. 2).

## Done on `main` since the previous review

- **Per-thread zstd decompressor** (#113): reads without a context borrow the thread's
  decompressor; see the table above.
- **Context parameter dropped** from `to_ndarray_sub`, `get`, `compact_with`, `write_to`; optional
  (`Option<&ReadContext>`) on `to_buf_slice`, `to_buf_strided`, `to_buf_strided_into`.
  `Array::read_ctx()` removed; `ReadContext::new`, `new_for_array`, `thread_ctx` are public.
- **Typed `ArrayAny<ET = TypeDyn>`**: `into_any()` keeps the element type, so a pipeline can be
  stored in a struct field or chosen in an `if` without `into_typed` (previous sec. 6).
- **Scalar operands for the operators**: `a * 2.0 + 1.0`, `1.0 - a`, `a % 3`, `mask & 1`, on
  either side, passed to the kernel without building a constant array (`AddScalar<S, T, Side>`
  and friends). The crate docs now show it in the quick start.
- `And` / `Or` / `Xor` renamed to `BitAnd` / `BitOr` / `BitXor`.
- `.jix` format compatibility is stated in the crate docs (newer versions read older files).
- `ArraySpec` hidden; `DtypeField` added; the `CompactParams` "codec / level / filters are
  inherited as a group" rule and the "`write_to` ignores `params` for a compact source" rule are
  now documented.

## TL;DR

Hard blockers (some use cases are not possible):
- No user-defined data sources: `StorageBase` is sealed (sec. 5).
- No dtype-dynamic math in core: arrays loaded from files are `TypeDyn` and only get shape ops
  (sec. 10).
- Read-only arrays (`todo.md`).

Biggest everyday friction, now:
1. Scalars only work through operators. `a.maximum(0.0)`, `a.clamp(0.0, 1.0)`,
   `a.greater(0.0)`, `a.pow(2.0)` do not compile; there is no `full` / `zeros` to build the
   operand, and no broadcasting (sec. 1).
2. Sharing and erasure: leaf arrays are not `Clone`, and `into_any()` needs `'static` storage, so
   `a.view().into_any()` does not compile; the workaround is wrapping the storage in an `Arc`
   (sec. 6).
3. Several ways to do each thing: 9 constructors, 12 read methods, and 197 public op constructors
   that duplicate the methods with opposite error behavior (sec. 3, 4).
4. `ReadContext` is fast by default now, but has two kinds of contexts, a 40x trap, constructors
   that return `Result` but cannot fail, and the read size in two places (sec. 2).

---

## 1. Binary ops: scalars beyond the operators, broadcasting, constants

The operators take scalars now. The named binary methods do not: `pow`, `divide_*`,
`remainder_*`, `equal` ... `approx_equal`, `maximum`, `minimum`, `clamp`, `bitwise_shift_*`,
`bitwise_rotate_*`, `select` (22 methods) take `Array<S2>` only, and there is no public way to
build a constant array (`Scalar` storage is in `__private`). Checked against `main`:

```rust
let lin = a.view() * 2.0 + 1.0;                 // compiles now
let relu = a.view().maximum(0.0);               // E0308: expected `Array<_>`, found float
let unit = a.view().clamp(0.0, 1.0);            // E0308
let pos = a.view().greater(0.0);                // E0308
let sq = a.view().pow(2.0);                     // E0308
// row normalization still needs insert_dim + broadcast:
let shape = a.shape().to_vec();
let mean = a.view().mean(1).insert_dim(1).broadcast(&shape);
let std = a.view().std(1, 0.0).insert_dim(1).broadcast(&shape);
let z = (a.view() - mean) / std;
// without the broadcast it panics at runtime (shape mismatch), it does not broadcast
```

Proposal:
- **Scalars in the named methods**: one `IntoOperand<T>` trait implemented for `T: Dtyped` and
  for `Array<S>`; the methods take `impl IntoOperand<S::Item>`. A scalar becomes the `Scalar`
  storage at `self`'s shape, which the element-wise pipeline already reads as a 0-stride
  operand, so this needs one trait for all 22 methods and no new op structs. The 8 operators keep
  their specialized `*Scalar` kernels (or move to the same mechanism, if the speed difference is
  small: measure first).
- **Constants**: `Array::full(value, shape)`, `zeros`, `ones` - the `Scalar` storage, made
  public.
- **Broadcasting** in the `Op2` engine: an operand of a smaller shape is read at the reduced
  index and its pulled buffer gets 0-strides. jix-py already broadcasts on top of core.
- **`keepdims`** on reductions, so normalization is
  `(a.view() - a.view().mean_keepdims(1)) / a.std_keepdims(1, 0.0)`.

## 2. `ReadContext`: what is left

Main implemented most of the previous recommendation (see the deep dive). Remaining issues:

- **Two kinds of contexts, and a 40x trap.** `ReadContext::thread_ctx()` and the context-free
  reads borrow the thread's decompressor; `ReadContext::new`, `new_for_array` and `default`
  create their own (~16 us). The docs call `new_for_array` "the preferred way", and creating one
  per read is the natural pattern - it costs 15.9 us per 16x16 read vs 0.38 us with `None`.
  `thread_ctx()` returns `impl AsRef<ReadContext>`, so it cannot be stored in a struct field
  without a generic, and its read params are always the defaults.
- **Infallible constructors return `Result`**: `ReadContext::new(params)` and `new_for_array`
  always return `Ok`; `Default` and `thread_ctx` `unwrap` them.
- **The read size lives in two places**: on the leaf array (`read_params_mut` via the extension
  traits, inherited by views) and on the context (`read_params(_mut)`, overriding the array's).
  `new_for_array` exists only to copy one into the other.

Proposal:
1. Every context borrows the thread's decompressor lazily (take it on the first compressed
   block, give it back in `Drop`; a nested or concurrent context finds the slot empty and creates
   its own). This was the prototype on this branch before #113 (~25 lines). Then
   `thread_ctx()` is the same as `ReadContext::default()` and can go, and the trap disappears.
2. Drop the read params from the context: reads always use the array's. Then
   `ReadContext::new()` takes no arguments and cannot fail, and `new_for_array` goes.
3. Result: `ReadContext` is just "scratch buffers you can reuse across reads", passed as
   `Option<&ReadContext>` to the three low-level `to_buf_*` methods. Reusing one is worth ~15% on
   a 16x16 compact read and nothing on a 64x64 tile, which matches that scope.

## 3. Too many constructors and read methods

Unchanged since the previous review. Constructors (9): `compact_ndarray`, `compact_ndarray_with`,
`compact_ndarray_ptr`, `compact_fn`, `compact_fn_with`, `plain_ndarray`, `plain_ndarray_ref`,
`plain_ndarray_view`, `plain_ndarray_ptr` (plus `PlainBase::new`, `from_storage`).
Reads/materializers (12): `to_ndarray`, `to_ndarray_sub`, `to_buf_slice`, `to_buf_strided`,
`to_buf_strided_into`, `get`, `item`, `to_plain`, `compact`, `compact_with`, `write_to_file`,
`write_to`. File readers (4): `read_from_file`, `read_from_file_with`, `read_from`,
`read_from_file_mmap`.

- The `compact_*` constructors are "wrap as plain, then compact": `compact_ndarray_with(&x, p)` is
  `plain_ndarray_ref(&x)?.compact_with(p)`.
- `plain_ndarray_ref` / `plain_ndarray_view` can be one function via `ndarray::AsArray`.
- `compact_fn` materializes eagerly; a lazy `from_fn` would compose with ops.
- Two region languages: `to_ndarray_sub(&[Range<u64>])` vs `slice((..))` (i64, negative indices).
- `to_buf_slice` takes `&mut [u8]`, so reading into a `Vec<f32>` needs `unsafe`
  (`slice::from_raw_parts_mut`), as its own doc example shows.
- `to_ndarray()` needs `::<Ix2>` / `::<IxDyn>` whenever nothing else pins `D` (E0283).

Proposed:
```rust
Array::from_ndarray(arr)                // owned -> Plain
Array::from_view(&arr | arr.view())     // borrowed -> PlainView (ndarray::AsArray)
Array::from_fn(shape, f)                // lazy
unsafe Array::from_raw_parts(ptr, shape, strides, dtype)
Array::compact_ndarray(&arr)?           // keep as the one-liner
x.compact()?  /  x.compact_with(params)?
x.to_ndarray()?  x.to_ndarray_dyn()?  x.slice(..).to_ndarray()?
x.read_into(&mut [T] | ArrayViewMut)?   // typed, no unsafe
x.get(&[i, j])?  x.item()?
```
`to_buf_strided(_into)` and `to_buf_slice` move to an "advanced / raw" section.

## 4. Two op APIs with opposite error behavior

Every op is an `Array` method (panics on bad arguments, `#[track_caller]`) and also
`ops::X::new(storage, ..)` / `ops::X::new_array(array, ..)` (returns `Result`). The scalar ops
added 8 + 8 more: 100 `new` + 97 `new_array`, with one-line docs. jix-py calls `new_array` 34
times, just to get a `Result`.

Proposal: one rule. Programmer errors panic (dim index out of range); data-dependent shape errors
return `Result` from the method itself (`reshape`, `broadcast`, `concatenate`, `stack`). Make
the struct constructors `pub(crate)` (or hidden for jix-py) and keep the structs only as type
names.

## 5. Users cannot bring their own storage

Unchanged. `StorageBase` is sealed and `OutBuf` has no public constructor, so `read_data`,
`Index`, `ArrayStorageInfo` are public but unusable outside the crate. There is no way to run a
jix pipeline over HDF5/Zarr, an object store or a generator.

Proposal: a small, safe, push-only trait, adapted internally:
```rust
pub trait Source: Send + Sync {
    type Item: Dtyped;
    fn shape(&self) -> &[u64];
    fn chunk_shape(&self) -> Option<&[u64]> { None }
    fn read(&self, start: &[u64], out: ndarray::ArrayViewMut<'_, Self::Item, IxDyn>) -> Result<()>;
}
let a = Array::from_source(src); // lazy, composes with every op
```
Then `read_data`, `OutBuf`, `Index`, `StridedBuf`, `ArrayStorageInfo` can leave the default docs.

## 6. Sharing and erasure

Typed erasure is done. What is left:
- **Leaf arrays are not `Clone`**: `a.clone()` on `Array<Compact<f32>>` does not compile. Using
  one array in two pipelines needs `a.view()` (borrowed) or
  `Array::from_storage(Arc::new(a.into_storage()))`, which works but is undocumented, and
  `from_storage` is documented as "rarely needed".
- **`into_any()` needs `'static` storage**: `a.view().into_any()` fails (E0597), so a pipeline
  over a borrowed array cannot be erased. Combined with the above, choosing between two
  pipelines over the same array needs the `Arc` route:
  ```rust
  let shared = Array::from_storage(Arc::new(a.into_storage()));
  let p: ArrayAny<f32> =
      if flip { (shared.clone() * -1.0).into_any() } else { shared.clone().exp().into_any() };
  ```

Proposal: put the data of `Compact` / `Plain` (owned) behind an `Arc`, so leaf arrays are cheaply
`Clone` (the compressed blocks are immutable already). Then `a.clone().exp().into_any()` works,
and most `view()` calls become unnecessary. A lifetime on `ArrayAny<'a, ET>` would also allow
erasing views, but is more churn for less benefit.

## 7. File I/O

- Paths are `&Path`: `a.write_to_file("out.jix")` does not compile (E0308); docs write
  `Path::new(..)`. Use `impl AsRef<Path>`.
- `write_to_file` fails if the file exists (an `Io` error; `create_new`). Surprising next to
  `std::fs::write` / `np.save`.
- mmap has no whole-file form: docs do `std::fs::metadata(&path)?.len()` and then
  `read_from_file_mmap(&path, 0, len, ArchiveValidation::default())`.
- Typed load is always `read_from_file(p)?.into_typed::<f32>()?`.
- `jix::error::Result` is `pub(crate)`: users cannot write `jix::Result<T>` (E0425).

Proposed: `jix::open::<f32>(path)` / `jix::open_dyn(path)`, a `ReadOptions` (mmap, byte range,
validation), `save` (overwrite) / `save_with(params)`, `pub type jix::Result<T>`.

## 8. Missing basics

- `full`, `zeros`, `ones`, `arange`, `linspace` (sec. 1).
- Integer indexing in `slice` that drops the dim (today `slice((1..2, ..)).remove_dim(0)`).
- `keepdims` on reductions; a reduce-all form without listing dims (`sum((0, 1))` today).
- `to_ndarray_dyn()` (sec. 3).

## 9. Smaller cleanups

- `jix::scalar::*` leaks `fearless_simd`: 373 visible `*_bulk` methods take
  `S: fearless_simd::Simd`, making that crate's major version part of jix's semver - which now
  matters, with 0.1.0 released. Hide or seal them; offer a few umbrella traits (`Float`,
  `Integer`) for generic user code.
- Three of each name: `std::ops::Add`, `jix::ops::Add`, `jix::scalar::Add` (same for `Sub`,
  `Mul`, `Div`, `Neg`, `Not`, `Sum`, and since the rename `BitAnd`, `BitOr`, `BitXor`). Glob
  imports of `jix::ops::*` and `std::ops::*` collide.
- `map2` vs `map_multiple`: the second covers the first.
- `CompactParams`: `block_shape`, `block_shape_fixed_dims`, `level`, `filters` return `Result`,
  forcing `?` mid-builder; validate on use instead. Setting any of codec / level / filters resets
  the other two (now documented), so `compact_with(CompactParams::new().level(9)?)` on a
  bit-shuffled source silently switches to byte shuffle; inherit per field instead.
- Extension traits need an import (`use jix::storage::CompactArrayMethods` to call
  `a.block_shape()`): add a `jix::prelude`.
- Internals in the public API: `CompactView` ("used internally"), `ElementwisePipeline` ("users
  rarely need it"), `ArrayStorageInfo` (old prefix), `ArithmeticStatus` (only meaningful with the
  `*_bulk` / scalar trait methods), `ArraySequence*` traits.
- `_dyn` twins: `dtype_sub_field(_dyn)`, `transmute_elements(_dyn)`, `into_typed(_dyn)`.

## 10. Bigger picture: a dynamic API in core

CLI tools, servers and anything opening arbitrary `.jix` files only get `TypeDyn` arrays with
shape ops. jix-py already contains a complete dynamic layer (dtype dispatch, promotion,
broadcasting, scalar operands - ~6.4k lines in `jix-py/src/ops`), mostly PyO3-independent, and
`todo.md` lists "provide ops on dyn arrays in core". Moving it into core (or a `jix-dyn` crate)
as `DynArray` would give Rust users the NumPy-like experience and shrink jix-py to a thin wrapper.

## Suggested order

Breaking changes are best batched for 0.2.0.

1. Quick wins: contexts borrow the thread decompressor (removes the trap, `thread_ctx` goes),
   infallible `ReadContext::new()`, public `jix::Result`, `impl AsRef<Path>`, prelude, hide
   `*_bulk`.
2. Cheap `Clone` for leaf arrays.
3. Scalars in the named methods (`IntoOperand`) + `full` / `zeros` / `ones`.
4. Broadcasting + `keepdims`.
5. Read params with one home (drop them from the context).
6. Constructor / read consolidation, hide op struct constructors, the error rule.
7. `Source` trait.
8. Dynamic API.

---

# Deep dive: `ReadContext`

Status: `main` implemented the core of the recommendation below in #113 (per-thread
decompressor, no context parameter on the convenience methods); "Status on `main`" at the end
lists what is left. The measurements in this section were taken before #113 (on `468201e`, or
with the prototype that preceded it) and explain why it was done this way.

## What a context holds

| resource | size | grows? | cost to create |
|---|---|---|---|
| zstd decompression context (`ZSTD_DCtx`), created lazily on the first compressed block | 93 KB (`ZSTD_sizeof_DCtx`, zstd 1.5.7) | no - the one-shot API needs no window buffer (see "zstd context sizes") | ~20 us (create + free) |
| buffer pool: one free list per alignment class | high-water mark, see below | yes, no cap on size or count | ~0 |
| read cache for arrays read twice in one expression | scoped to one read | no (freed at scope end) | ~0 |
| read params (read size) | bytes | no | ~0 |

The pool keeps every returned buffer; a popped buffer that is too small is freed and replaced by a
larger one. So it settles at (max buffers in flight during one read) x (largest size requested).
Measured retention after one read, with the context still alive (4096x4096 `f32`, a counting
global allocator):

| read | retained by the context |
|---|---|
| default block shape: full read, 64x64 tiles, `exp().map()`, `transpose()`, `sum(0)`, `compact_with` | 0.1 - 0.2 MB |
| default block shape: `(a + a.T).exp()` full | 2.2 MB |
| blocks of 1024x1024 (4 MB), one 64x64 tile | 8.4 MB |
| one 4096x4096 block (64 MB), one 64x64 tile | 134 MB |
| default blocks, read size set to 64-128 MB, `(a + a.T).exp()` | 134 MB |
| `to_buf_strided(full)` pull, buffer dropped | 0.1 MB (the dropped buffer is freed, not pooled) |

So with defaults retention is small, but it scales with the largest block or read size ever used:
about 2x the block size (decompressed block + filter scratch).

Where the fresh-context cost goes:

| call (16x16 `f32`) | per call |
|---|---|
| plain array, `map().to_ndarray()`, fresh context | 307 ns |
| plain array, same, reused context | 293 ns |
| compact array, `to_ndarray()`, fresh context | 15.3 - 17.5 us |
| compact array, `to_ndarray_sub()`, reused context | 0.46 - 0.6 us |
| `zstd::bulk::Decompressor::new()` + drop alone | 20 us |
| 64x64 tile from a 4096x4096 compact array, fresh vs reused context | 183 vs 167 us |

Conclusion: practically the whole cost of a fresh context is creating the zstd context. Reusing
pooled buffers across calls is worth little (measured below); within one call, buffers are
already reused across the blocks it reads.

## Buffer reuse across calls, with the zstd context cached

Expression `-((a + b).into_any().into_typed::<i32>())`, `a` and `b` 512x512 `i32` (plain or
compact), many reads of an s x s window at random offsets with `to_buf_slice`, with the zstd
context cached per thread (the prototype that preceded #113; `main` behaves the same).
`into_any` stops fusion, so every read materializes `a + b` into a pooled scratch buffer and
then negates it into the output. Instruction counts from callgrind
(deterministic; wall time on this VM is +-15-25%, too noisy for single-digit differences):

| array | window | fresh context, instr/read | reused context, instr/read | extra for fresh |
|---|---|---|---|---|
| plain | 1x1 | 3,935 | 3,383 | 552 (16.3%) |
| plain | 4x4 | 5,388 | 4,832 | 556 (11.5%) |
| plain | 16x16 | 8,325 | 7,769 | 556 (7.2%) |
| plain | 64x64 | 20,889 | 19,647 | 1,242 (6.3%) |
| plain | 256x256 | 115,271 | 113,138 | 2,133 (1.9%) |
| compact | 1x1 | 138,424 | 135,039 | 3,385 (2.5%) |
| compact | 16x16 | 228,172 | 224,783 | 3,389 (1.5%) |
| compact | 256x256 | 1,692,228 | 1,683,322 | 8,907 (0.5%) |

Wall time (median of 25 interleaved rounds) agrees: plain 1x1 609 vs 552 ns, 4x4 938 vs 870
ns, 256x256 66 vs 67 us; compact within 0-4%. An earlier run that always measured "fresh" first
showed 1.3-1.6x on large windows; that was ordering bias, gone once rounds were interleaved.

A fresh context costs 2 allocations per plain read (the scratch buffer and the pool's free-list
`Vec`, which is allocated on the first return) and 5 per compact read (block-sized decompression
and filter scratch, ~144 KB here). The fixed ~550 instructions are ~60 ns: noticeable only for
reads of a few elements of a plain (no decompression) pipeline. Callgrind does not count kernel
time, so page faults on fresh large buffers are not in these numbers, but the wall times show no
cost from them up to 256x256 (glibc raises its mmap threshold after the first free, so freed
buffers are reused from the heap).

## zstd context sizes: decompression vs compression

A `ReadContext` holds only a decompression context. Sizes after one-shot calls (the APIs jix
uses: `compress2` for writing, `decompress` into a full buffer for reading), zstd 1.5.7:

| block size | level | compression context | decompression context |
|---|---|---|---|
| 64 KB | 1 / 3 / 19 | 297 KB / 649 KB / 1.7 MB | 93 KB |
| 1 MB | 1 / 3 / 9 / 19 | 569 KB / 1.2 MB / 10.5 MB / 17.3 MB | 93 KB |
| 16 MB | 3 / 19 / 22 | 1.2 MB / 81 MB / 257 MB | 93 KB |
| 64 MB | 3 / 19 / 22 | 1.2 MB / 81 MB / 641 MB | 93 KB |

The decompression context stays at 93 KB for every block size and level: one-shot decompression
writes straight into the destination, so it needs no window buffer. (The streaming decompression
API would allocate a window buffer of up to the frame's window size, 128 MB by default; jix does
not use it.) The compression context grows with level and block size; jix creates one per
compaction / write (`zstd::bulk::Compressor` in the encoder) and frees it at the end, so it is
not part of the read-side caching question.

## The concern with a thread-local context

A thread-local *buffer pool* would pin its high-water mark for the life of the thread. With a
rayon pool of 32 threads, one huge-block read per thread pins 32 x 134 MB = 4.3 GB, with no owner
the user can drop. Thread-local destructors only run at thread exit, and pool threads rarely
exit. That concern is real, and the usual answer elsewhere (below) is either "don't cache large
things" or "trim on a timer or memory pressure" - jix has no timer, no GC and no memory-pressure
signal to hook into.

A second concern is reentrancy: user closures (`map`, `compact_fn`) may read other jix arrays, so
a thread-local must never be borrowed across a read. A take/put-back slot (`Cell<Option<T>>`)
avoids this: a nested read finds the slot empty and creates its own.

## Why zstd has no thread-local context

zstd leaves the policy to the caller and says so in `zstd.h`:

> When decompressing many times, it is recommended to allocate a context only once, and reuse it
> for each successive compression operation. This will make workload friendlier for system's
> memory. Use one context per thread for parallel execution.

Reasons a library like zstd cannot hide a thread-local context:
- It runs where thread-locals don't exist or allocation is not allowed: the Linux kernel (btrfs,
  squashfs, zram, kernel image decompression), bootloaders, and static builds that place the
  context in caller memory (`ZSTD_initStaticDCtx(workspace, size)`), or use a caller allocator
  (`ZSTD_customMem`).
- Context sizes span orders of magnitude: a one-shot decompression context is 93 KB, but a
  compression context at level 19-22 is 17 MB - 641 MB (table above), and a streaming
  decompression context holds a window buffer of up to 128 MB. Pinning one per thread is a
  decision only the application can make.
- A C library has no portable, reliable way to free thread-local state at thread exit.

Where zstd does own pools or reusable memory, it bounds them:
- The multi-threaded compressor's buffer pool stores at most `totalBuffers` buffers (sized by the
  worker count), and reuses a pooled buffer only if it is between 1x and 8x the needed size;
  otherwise it frees it (`ZSTDMT_getBuffer` / `ZSTDMT_releaseBuffer`, `zstdmt_compress.c`).
- A reused context shrinks itself: if its workspace has been >= 3x larger than needed for 128
  consecutive uses, it is freed and reallocated (`ZSTD_WORKSPACETOOLARGE_FACTOR = 3`,
  `ZSTD_WORKSPACETOOLARGE_MAXDURATION = 128`).

The Rust `zstd` crate follows the same model: `zstd::bulk::decompress` creates a context per call,
`zstd::bulk::Decompressor` is the reusable one; no thread-locals.

## What other libraries do

| library | implicit resource | scope | how growth is bounded / released | explicit escape hatch |
|---|---|---|---|---|
| c-blosc2 (jix's inspiration) | global context for `blosc2_compress` / `blosc2_decompress`, behind a global mutex; super-chunks own their own contexts | process / object | n/a | `blosc2_create_dctx` + `blosc2_decompress_ctx`, one per thread |
| Rust `regex` | pool of search caches inside each `Regex`; first thread gets a lock-free owner slot, others a mutex-guarded stack | object | dropped with the `Regex` | `regex-automata` `search_with(&mut Cache, ..)` |
| Apache Arrow C++ | `ExecContext* ctx = default_exec_context()` default argument; memory from a global pool (jemalloc/mimalloc) | process | the allocator's own decay | pass your own `ExecContext` / `MemoryPool` |
| Go `sync.Pool` | per-processor caches | process | emptied by the GC; `fmt` refuses to pool buffers > 64 KiB (Go issue 23199) | none |
| .NET `ArrayPool<T>.Shared` | thread-local + per-core stacks | process | trimmed on Gen2 GCs, faster under memory pressure | `ArrayPool<T>.Create()` |
| jemalloc / glibc malloc | per-thread caches | thread | bounded per size class; incremental tcache GC (jemalloc); decay-based purge to the OS | allocator options |
| Intel oneMKL | internal memory manager with per-thread buffers | process / thread | `mkl_free_buffers()`, `mkl_thread_free_buffers()`; a docs page exists because users report the cache as a leak | `MKL_DISABLE_FAST_MM` |
| Linux btrfs (zstd) | pool of zstd workspaces | kernel | idle workspaces reclaimed by a timer (~307 s); one kept for forward progress | none |
| Lucene | `CloseableThreadLocal` | object | Java `ThreadLocal` values can live arbitrarily long, so thread-locals are tied to an owner that can be `close()`d | `close()` |
| PyTorch CUDA allocator | cache of freed device blocks | process | `garbage_collection_threshold` (off by default) | `torch.cuda.empty_cache()` |

Patterns:
1. Explicit context; the convenience call makes a fresh one (zstd, zstd-rs). Fast and
   predictable, slow default - jix today.
2. Object-owned pool (regex, blosc2 super-chunks, Lucene). Resources die with a natural owner. jix
   has no natural owner: pipelines mix many arrays, and arrays are shared across threads.
3. Global or thread-local cache with a bound plus a release policy (allocators, Go, .NET, MKL,
   btrfs). Always comes with a cap on what is kept (especially: don't keep huge items), a
   time / pressure based trim, and often an explicit release call or an opt-out.
4. Optional explicit context with a global default (Arrow).

What they agree on: cache what is fixed-size and expensive to build; don't let a cache keep
memory proportional to the largest request it ever saw without a way to shrink.

## Recommendation for jix

Split the context by how each resource behaves:

1. **zstd decompression context: one per thread** (done in #113). Fixed size (93 KB), expensive
   (~20 us), cannot grow. A fresh `ReadContext` takes it from a thread-local slot on first use
   and puts it back on drop (`Cell<Option<..>>`, so reentrant and nested reads just create
   another). Worst case pinned memory: 93 KB per thread that ever decompressed. On `main` only
   `thread_ctx()` and the context-free reads borrow it; `ReadContext::new` still creates its own.
2. **Buffer pool: stays scoped to one `ReadContext`** (as on `main`), so for the convenience
   methods it lives for one call and is freed when the call ends. Cross-call reuse is worth ~550
   instructions per read (16% on a 1-element plain read, under 2% from 256x256, under 2.5% on
   compact reads). Half of that is the free list's own `Vec` allocation, which an inline free
   list (e.g. `ArrayVec`) removes without any cache. If the rest matters, a capped thread-local
   cache of small buffers only (e.g. at most 4 buffers of at most 64 KB: <= 256 KB per thread,
   the Go `fmt` rule) would catch it with bounded memory. For long-lived explicit contexts, add a
   cap in the same spirit: don't return buffers above some size (e.g. a few MB, or 2x the read
   size) to the free list, so one huge-block read does not pin 134 MB.
3. **Read size: one home, the array** (it is already stored there and inherited by views). Drop
   `ReadContext::read_params(_mut)`. (`Array::read_ctx()` is gone on `main`;
   `ReadContext::new_for_array` took its place and would go too.)
4. **API: no context parameter on the convenience methods** (done in #113 for `to_ndarray_sub`,
   `get`, `compact_with`, `write_to`); each creates a cheap per-call context. The three
   low-level `to_buf_*` methods take `Option<&ReadContext>`, which keeps buffer reuse available
   where it can matter (small reads) without burdening the common path.

Why not a thread-local pool instead: cross-call buffer reuse is worth ~60 ns per read (measured
above), while an uncapped pool risks unbounded per-thread pinning with no trim signal. The system
allocator already caches freed memory per thread and trims it with its own policy; a second
cache layer on top mostly duplicates that.

## Status on `main` (`d6005d4`)

PR #113 implemented recommendation 1 (for context-free reads and `thread_ctx()`) and most of 4.
Same VM, 16x16 `f32` compact array, release, median of interleaved rounds:

| call | `468201e` | `d6005d4` |
|---|---|---|
| `to_ndarray()` | 16.1 us | 0.40 us |
| `get(&[3, 3])` | 16.1 us | 0.40 us |
| `to_buf_slice(.., None)` (a fresh context per call before) | 16.0 us | 0.38 us |
| `to_buf_slice(.., Some(&reused))` | 0.33 us | 0.33 us |
| `to_buf_slice(.., Some(thread_ctx().as_ref()))`, reused | - | 0.33 us |
| `to_buf_slice(.., Some(&ReadContext::new_for_array(&a)?))` per call | - | 15.9 us |
| 64x64 tile of 4096x4096, fresh / reused context | 147 / 128 us | 129 / 129 us |

What is left (sec. 2 of the review):
1. `ReadContext::new` / `new_for_array` / `default` still create their own decompressor, so
   creating one per read is 40x slower than passing `None`. Making every context borrow the
   thread's (take on first use, return on drop) removes the trap and makes `thread_ctx()` the
   same as `ReadContext::default()`.
2. `ReadContext::new` and `new_for_array` return `Result` but cannot fail.
3. The read size still has two homes (recommendation 3).

Sources:
- zstd 1.5.7 sources (`zstd.h`, `zstdmt_compress.c`, `zstd_internal.h`, `zstd_decompress.c`),
  vendored in `zstd-sys`.
- c-blosc2 contexts: https://blosc.org/c-blosc2/reference/context.html
- Go issue 23199 and the `fmt` fix: https://github.com/golang/go/issues/23199,
  https://go.googlesource.com/go/+/e82d152e66e81812a5c6ebf075cc99efd2602b19
- .NET shared array pool:
  https://source.dot.net/System.Private.CoreLib/src/libraries/System.Private.CoreLib/src/System/Buffers/SharedArrayPool.cs.html
- oneMKL memory manager:
  https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-1/avoiding-memory-leaks-in-onemkl.html
- btrfs zstd workspace reclaim:
  https://gbmc.googlesource.com/linux/+/refs/heads/linux-6.9.y/fs/btrfs/zstd.c
- regex pool: https://docs.rs/regex-automata/0.4.6/src/regex_automata/util/pool.rs.html
- Arrow `ExecContext`:
  https://apache.googlesource.com/arrow/+/3fe2df7b00291752e49beb3ee671a39df9a69cf4/cpp/src/arrow/compute/exec.h
- jemalloc tuning: https://sources.debian.org/src/jemalloc/5.3.0-4/TUNING.md
- Lucene `CloseableThreadLocal`:
  https://lucene.apache.org/core/9_9_1/core/org/apache/lucene/util/CloseableThreadLocal.html
- PyTorch CUDA allocator settings: https://docs.pytorch.org/docs/main/cuda_environment_variables.html
