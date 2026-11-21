# jix

Python bindings for the [`jix`](https://github.com/barakugav/jix) Rust library.

**Block-based compression.** An array is split into a grid of fixed-size nd-blocks, each
compressed independently. On a read request for a sub-region of the array, only the blocks that
overlap the requested region are decompressed, so random access into large arrays is relatively
cheap.

**Lazy operation chains.** Every operation - arithmetic, shape change, reduction, type cast -
returns a new view that wraps the input(s) and records the transformation; nothing is computed
until data is explicitly requested. A chain of such operations builds a pipeline that runs in a
single decompression pass the moment you ask for output (read an element or a sub-region).

```python
import jix
import numpy as np

# Compress a NumPy array into block-compressed storage.
a = jix.compact(np.arange(1_000_000, dtype=np.float32).reshape(1000, 1000))

# Read single elements and a sub-region. Only the blocks they touch are
# decompressed, not the whole array, so random access reads are relatively fast.
assert a[0, 0] == 0
assert a[999, 999] == 999_999
assert a[0, 0:10].tolist() == list(range(10))

# Build a lazy pipeline - no decompression happens yet.
result = (a - a.mean(dim=0)) / a.std(dim=1)

# Materialize the pipeline into a NumPy array.
out = result.numpy()

# Or write straight to disk - blocks are decompressed, transformed,
# and re-compressed one at a time without materializing the full result,
# not even in its compressed form.
result.write_to("normalized.jix")

# Load back; use mmap=True for zero-copy access to large files.
b = jix.read_array("normalized.jix", mmap=True)
print(b.shape, b.dtype)  # (1000, 1000) float32
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
- **Reducing peak memory usage of array ops**
    Usually, in crates like `ndarray` or the `NumPy` library, a full intermediate array is
    allocated for every step in a chain of operations, which can lead to high peak memory usage.
    Jix's lazy operation chains avoid this by building a pipeline that compute each output block
    one by one, so the peak memory is bounded by O(block_size). This also has the benefit of
    better cache locality, but most of the times this alone doesn't achieve better performance
    than the common eager libraries.
- **Computation on arrays that don't fit in memory.**
    For arrays too large to hold in memory, jix's lazy operation chains let you mmap an array
    from disk, apply a pipeline of operations on top of it, and stream the result back to
    disk - without ever holding the full array in memory, not even in its compressed form.
    This use case needs only the lazy pipeline; you can build it on a plain array backed
    by an mmap'd file, without using the compact format.

### When NOT to use jix

- if you want to compress an array but always read it in full, you don't need jix at all - just
    zip and unzip the whole array with a general-purpose compressor. Although some of jix filters
    are dtype aware, so you may want to test the compression ratio and speed for your specific
    data.
