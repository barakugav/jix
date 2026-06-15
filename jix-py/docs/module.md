
Multi-dimensional array library with block-compressed, lazy-evaluated storage.

The crate provide Python bindings for the core `jix` library, which is implemented in Rust.
`jix` is a multi-dimensional array library that stores data in **block-compressed format**
and evaluates operations **lazily**. It is designed around two ideas:

- **Block-based compression** - the array is split into an n-dimensional grid of fixed-size
  blocks, each compressed independently with Zstd. Only the blocks that overlap a read
  request are decompressed, so random access into large arrays avoids loading the whole
  dataset into memory.

- **Lazy operation chains** - every operation (arithmetic, shape manipulation, type cast,
  reduction, ...) builds a new `Array` that records the transformation without executing
  it. The full pipeline runs in a single decompression pass the moment you ask for output.
  While the pass runs the GIL is released, so Python threads can make progress
  concurrently.

The library is NumPy-compatible: arrays expose a NumPy `dtype`, accept NumPy index syntax,
and materialize to NumPy arrays on demand.


# Quick start

```python
import jix
import numpy as np

# Compress a NumPy array into block-compressed storage.
a = jix.compact(np.arange(1_000_000, dtype=np.float32).reshape(1000, 1000))

# Accessing the data triggers decompression of the relevant blocks
assert a[0, 0] == 0
assert a[999, 999] == 999_999
assert a[0, 0:10].tolist() == list(range(10))

# Build a lazy pipeline - no decompression happens yet.
result = (a - a.mean(axis=0)) / a.std(axis=0)

# Materialize the pipeline into a NumPy array.
out = result.numpy()

# Or write straight to disk - blocks are decompressed, transformed,
# and re-compressed one at a time without materializing the full result,
# not even in its compressed form.
result.write_to("normalized.jix")

# Load back; use mmap=True for zero-copy access to large files.
b = jix.read_array("normalized.jix", mmap=True)
print(b.shape, b.dtype)   # (1000, 1000) float32
```

# When should I use this library?

Jix's two main features — block-compressed ndarrays and lazy operation chains —
can be used independently, and each fits a different scenario.

- **Random access to a compressed array.**
    When you want to minimize the size of an array — on disk or in memory — but still need
    to read small regions of it at a time, jix's compact arrays let you decompress just the
    blocks that overlap each read. The same applies when you have many small arrays and want
    to keep their combined footprint low.
    A classic example is a machine-learning data loader that randomly samples chunks from
    a large dataset. This use case needs only the compact array — no lazy pipeline required.
    Note that if you want to compress an array but always read it in full, you don't need jix
    at all — just zip and unzip the whole array with a general-purpose compressor.
- **Computation on arrays that don't fit in memory.**
    For arrays too large to hold in memory, jix's lazy operation chains let you mmap an array
    from disk, apply a pipeline of operations on top of it, and stream the result back to
    disk — without ever holding the full array in memory, not even in compressed form.
    This use case needs only the lazy pipeline; you can build it on a plain array backed
    by an mmap'd file, without using the compact format.
- **Long and/or complex pipelines of operations.**
    NumPy evaluates eagerly: every step in a chain — every arithmetic op, cast,
    reduction — allocates a fresh intermediate buffer. For long pipelines on large
    arrays, the intermediates dominate both memory use and runtime. The same pipeline
    expressed in jix builds a single lazy view and materializes only the final result, with
    no intermediates, which is often faster than NumPy due to less memory overhead and cache
    locality.


# Array

The central type. Wraps compressed array data together with any pending lazy operations.
Every operation returns a new [`Array`][jix.Array]; no data is copied or computed until you ask for
output.

**Creating an `Array`:**

| Function | Description |
|---|---|
| `jix.compact(...)` | Compress any array-like (NumPy array, list, scalar) into a new jix array. This is the primary constructor. |
| `jix.asarray(...)` | Wrap any array-like as a zero-copy jix view without compressing. Useful for mixing plain NumPy data with jix arrays in ations. |
| `jix.read_array(...)` | Load a `.jix` file from disk. |

**Reading data from an `Array`:**

The primary output method is `Array.numpy()`, or equivalently `array[...]`. Both accept
the same indexing syntax as NumPy: integers (drop that axis), slices (keep that axis),
`...` (fill remaining axes). Note: slices must have step 1; bounds are checked strictly.

```python
a.numpy()            # full array
a.numpy(0)           # row 0 (integer drops axis 0)
a.numpy(slice(1, 4)) # rows 1-3 (slice keeps axis 0)
a[0, 1:3]            # row 0, columns 1-2 (shorthand)
a[..., -1]           # last column of any-rank array
```

## Block shape

Every jix array stores its data in a grid of fixed-size nd-blocks, each compressed
independently. The block shape has a large impact on both read performance and compression
ratio: only the blocks that overlap a read request are decompressed, so a block shape that
matches your access pattern avoids wasteful work. For example, a `[1, ncols]` block shape
means reading a single row decompresses exactly one block; a `[nrows, 1]` shape is
similarly efficient for column reads.

When no block shape is specified, jix picks one automatically - it greedily expands each
dimension (innermost first) until the block byte-size reaches the L1 data cache.

You can supply an explicit block shape through `params` when constructing an array:

```python
a = jix.compact(data, params={"block_shape": [64, 64]})
```

After shape-changing operations (`reshape`, `permute_axes`, etc.) the original block
layout may no longer match the new access pattern. Call `jix.compact(arr, params=...)` to
re-encode with a layout suited to the new shape.


# Operations

Every operation - arithmetic, comparisons, reductions, shape changes, type casts - returns
a new `Array` **view** that wraps the input(s) and records the transformation. No data is read or
computed at call time. The deferred work only runs when you ask for output (`.numpy()`, `[...]`,
`.write_to()`, `jix.compact()`, etc.).

Chains compose without intermediate allocations: the full pipeline is executed in a single
pass over the compressed source data, block by block.

```python
# Nothing is read or computed during these calls.
a = jix.read_array("data.jix")
result = (
    a
     .astype("float64")
     .exp()
     .sum(axis=0)
)

# This single call decompresses, transforms, and materializes the pipeline.
out = result.numpy()
```


# Persistence

Arrays are saved to and loaded from `.jix` files. The format stores metadata (shape, dtype,
block layout, codec settings) in a protobuf header followed by the raw compressed block
data.

```python
# Write to a file path.
jix.write_array(a, "data.jix")

# Load back.
b = jix.read_array("data.jix")

# Memory-mapped read: blocks are paged in from disk on demand.
# Fast startup, zero copy, but the file must not be modified while the array is live.
c = jix.read_array("data.jix", mmap=True)
```

`write_array` accept a file path or any writable binary
file-like object. `read_array` accepts a file path or any seekable binary file-like
object.

A key property: **a lazy array can be written directly without fully materializing it in
memory**. The write path compresses block by block, pulling data from the lazy chain on
demand. For example, the result of a large matrix operation can be streamed straight to
disk.


# Limits

- Maximum array dimensions: 8.
- Maximum inner-shape dimensions for struct dtypes: 4.
- Little-endian platforms only.
