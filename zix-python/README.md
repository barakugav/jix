# zix (Python)

Python bindings for the [`zix`](../zix/) Rust library. Built with [PyO3](https://pyo3.rs).

Arrays are stored as a grid of independently compressed blocks. Only the blocks that
overlap a read are decompressed, so slicing into a large array is cheap. Every operation
returns a new `Array` view — nothing is computed until you ask for output (`.numpy()`, `[...]`,
`.write_to()`, `zix.copy()`). The full pipeline then runs in a single pass with the GIL
released.

```python
import zix
import numpy as np

# Compress a NumPy array into block-compressed storage.
a = zix.compact(np.arange(1_000_000, dtype=np.float32).reshape(1000, 1000))

# Build a lazy pipeline — no decompression happens yet.
result = (a - a.mean(axis=0)) / a.std(axis=0)

# Materialize the pipeline into a NumPy array.
out = result.numpy()

# Or write straight to disk — blocks are decompressed, transformed,
# and re-compressed one at a time without materializing the full result.
result.write_to("normalized.zix")

# Load back; use mmap=True for zero-copy access to large files.
b = zix.read_array("normalized.zix", mmap=True)
print(b.shape, b.dtype)   # (1000, 1000) float32
```
