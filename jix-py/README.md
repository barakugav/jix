# jix

Python bindings for the [`jix`](https://github.com/barakugav/jix) Rust library.

**Block-based compression.** An array is split into a grid of fixed-size nd-blocks, each compressed independently.
Only the blocks that overlap a read request are decompressed, so random access into large arrays is cheap.

**Lazy operation chains.** Every operation - arithmetic, shape change, reduction, type cast - returns a new view that wraps the
input(s) and records the transformation, nothing is computed until data is explicitly requested.
The full pipeline runs in a single decompression pass the moment you ask for output.

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

## When should I use this library?

Jix's two main features - block-compressed ndarrays and lazy operation chains -
can be used independently, and each fits a different scenario.

- **Random access to a compressed array.**
    When you want to minimize the size of an array - on disk or in memory - but still need
    to read small regions of it at a time, jix's compact arrays let you decompress just the
    blocks that overlap each read. The same applies when you have many small arrays and want
    to keep their combined footprint low.
    A classic example is a machine-learning data loader that randomly samples chunks from
    a large dataset. This use case needs only the compact array - no lazy pipeline required.
    Note that if you want to compress an array but always read it in full, you don't need jix
    at all - just zip and unzip the whole array with a general-purpose compressor.
- **Computation on arrays that don't fit in memory.**
    For arrays too large to hold in memory, jix's lazy operation chains let you mmap an array
    from disk, apply a pipeline of operations on top of it, and stream the result back to
    disk - without ever holding the full array in memory, not even in its compressed form.
    This use case needs only the lazy pipeline; you can build it on a plain array backed
    by an mmap'd file, without using the compact format.
- **Long and/or complex pipelines of operations.**
    NumPy evaluates eagerly: every step in a chain - every arithmetic op, cast,
    reduction - allocates a fresh intermediate buffer. For long pipelines on large
    arrays, the intermediates dominate both memory use and runtime. The same pipeline
    expressed in jix builds a single lazy view and materializes only the final result, with
    no intermediates, which is often faster than NumPy due to less memory overhead and cache
    locality.