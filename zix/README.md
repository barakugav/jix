# zix

A multi-dimensional array library with block-compressed, lazy-evaluated storage.

**Block-based compression.** An array is split into a grid of fixed-size nd-blocks, each compressed independently.
Only the blocks that overlap a read request are decompressed, so random access into large arrays is cheap.

**Lazy operation chains.** Every operation returns a new `Array<OpStorage<...>>` that wraps the
input and records the transformation — nothing is computed until data is explicitly requested.
The full operation chain is encoded in the static type, so the compiler can inline the entire
pipeline into a single read loop with no virtual dispatch.

```rust
use zix::{Array, ArrayParams};
use ndarray::array;

// Compress a 2-D f32 ndarray into block-compressed storage.
let a = Array::compact_array(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;

// Build a lazy pipeline — no data is read yet.
// The full chain is a single static type: Array<Sum<Exp<Compact<...>>>>
let result = a.exp().sum(0) - 1.0;

// Materialize and persist. Blocks are decompressed, transformed,
// and re-compressed one at a time — no full copy in memory,
// not even the compressed form of the full result.
result.write_to_file("result.zix")?;
```

`Array<S>` is generic over its storage backend `S: ArrayStorage`, which can be `Compact` (block-compressed data),
`Mmap` (memory-mapped file), `Plain` (uncompressed in-memory), a lazy operation view like `Neg<Compact>`,
`Reshape<Neg<Compact>>`, etc.

The storage type `S` carries the full operation chain at the type level:

```text
Array<Compact>
  .neg()                 → Array<Neg<Compact>>
  .reshape_view(...)     → Array<Reshape<Neg<Compact>>>
  .permute_axes(&[1, 0]) → Array<PermuteAxes<Reshape<...>>>
  .sum(0)                → Array<Sum<PermuteAxes<...>>>
  .copy()?               → Array<Compact>   ← materialize
```
