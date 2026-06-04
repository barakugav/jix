# jix

A multi-dimensional array library with block-compressed, lazy-evaluated storage — written in Rust, with Python bindings.

**Block-based compression.** An array is split into a grid of fixed-size nd-blocks, each compressed independently.
Only the blocks that overlap a read request are decompressed, so random access into large arrays is cheap.

**Lazy operation chains.** Every operation - arithmetic, shape change, reduction, type cast - returns a new view that wraps the
input(s) and records the transformation, nothing is computed until data is explicitly requested.
The full pipeline runs in a single decompression pass the moment you ask for output.

```rust
use jix::{Array, ArrayParams};
use ndarray::array;

// Compress a 2-D f32 ndarray into block-compressed storage.
let a = Array::compact_array(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;

// Build a lazy pipeline — no data is read yet.
// The full chain is a single static type: Array<Sum<Exp<Compact<...>>>>
let result = a.exp().sum(0) - 1.0;

// Materialize and persist. Blocks are decompressed, transformed,
// and re-compressed one at a time — no full copy in memory.
result.copy()?.write_to_file("result.jix")?;
```

Python bindings are also available:

```python
import jix
import numpy as np

# Compress a NumPy array into block-compressed storage.
a = jix.compact(np.random.rand(1024, 1024).astype(np.float32))

# Build a lazy pipeline — nothing is read yet.
result = (a - a.mean(axis=0)) / a.std(axis=0)

# Materialize: decompress, transform, and write to disk in one pass.
result.write_to("normalized.jix")

# Load and read a sub-region; only the touched blocks are decompressed.
b = jix.read_array("normalized.jix")
row = b[42]   # only the blocks covering row 42 are decompressed
```


## Thanks

This project would not exist without the work of several upstream
authors and communities.
Specifically, this project was greatly inspired by the [C-Blosc2](https://github.com/Blosc/c-blosc2) library.
See the [`THANKS.md`](./THANKS.md) file for more details and attribution.

## License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the
License. A copy of the License is available in the LICENSE file at the root of this repository, or at http://www.apache.org/licenses/LICENSE-2.0.

Legal attribution and full license text for third-party components are in the [`NOTICE`](./NOTICE) file at the root of this repository.
