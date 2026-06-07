# jix (Python)

Python bindings for the [`jix`](https://github.com/barakugav/jix) Rust library. Built with [PyO3](https://pyo3.rs).

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
