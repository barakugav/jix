# jix

[![crates.io](https://img.shields.io/crates/v/jix.svg)](https://crates.io/crates/jix/)
[![PyPI](https://img.shields.io/pypi/v/jix)](https://pypi.org/project/jix/)
[![docs.rs](https://img.shields.io/docsrs/jix?label=docs.rs)](https://docs.rs/jix/latest/jix/)
[![Read the Docs](https://img.shields.io/badge/readthedocs-passing-brightgreen)](https://jix.readthedocs.io/en/stable/)


A multi-dimensional array library with block-compressed storage and lazy-evaluated operations - written in Rust, with Python bindings.

**Block-based compression.** An array is split into a grid of fixed-size nd-blocks, each compressed independently.
Only the blocks that overlap a read request are decompressed, so random access into large arrays is relatively cheap.

**Lazy operation chains.** Every operation - arithmetic, shape change, reduction, type cast - returns a new view that wraps the
input(s) and records the transformation, nothing is computed until data is explicitly requested.
A chain of such operations build a pipeline that runs in a single decompression pass the moment you ask for output.

```rust
use jix::Array;
use ndarray::array;

// Compress a 2-D f32 ndarray into block-compressed storage.
let a = Array::compact_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;

// Build a lazy pipeline - no data is read yet.
// The full chain is a single static type:
//     Array<Map<Sum<Exp<Compact>>>>
let result = a.exp().sum(0).map(|x| x - 1.0);

// Materialize and persist. Blocks are decompressed, transformed,
// and re-compressed one at a time - no full copy in memory.
result.write_to_file("result.jix")?;
```

Python bindings are also available:

```python
import jix
import numpy as np

# Compress a NumPy array into block-compressed storage.
a = jix.compact(np.random.rand(1024, 1024).astype(np.float32))

# Build a lazy pipeline - nothing is read yet.
result = (a - a.mean(axis=0)) / a.std(axis=1)

# Materialize: decompress, transform, and write to disk in one pass.
result.write_to("normalized.jix")

# Load and read a sub-region; only the touched blocks are decompressed.
b = jix.read_array("normalized.jix")
row = b[42]  # only the blocks covering row 42 are decompressed
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
    In Python, NumPy evaluates eagerly: every step in a chain - every arithmetic op, cast,
    reduction - allocates a fresh intermediate buffer. For long pipelines on large
    arrays, the intermediates dominate both memory use and runtime. The same pipeline
    expressed in jix builds a single lazy view and materializes only the final result, with
    no intermediates, which is often faster than NumPy due to less memory overhead and cache
    locality.

    In Rust, plain iterators over regular ndarrays already
    give you lazy element-wise evaluation for free (although naive use of the `ndarray` crate may still
    produces NumPy-style intermediates), so for simple `map`/`zip`-style pipelines
    jix offers little over hand-written iterator code.
    The advantage shows up once the pipeline includes
    operations that change the shape or the access pattern - reductions, broadcasts,
    axis permutations, reshapes, tiling, slicing, rolling, `concatenate`/`stack`, and so on.
    These are awkward or impractical to express as plain iterator chains,
    especially when combined with element-wise operations.
    Jix composes element-wise and shape-changing operations uniformly in the same operation chain,
    all as lazy views, and because the full chain is encoded in the static type the compiler is able to
    inline the whole pipeline into a single read loop - no virtual dispatch, no runtime scheduler,
    no (large) intermediate allocations beyond the final output.

## Thanks

This project would not exist without the work of several upstream
authors and communities.
Specifically, this project was greatly inspired by the [C-Blosc2](https://github.com/Blosc/c-blosc2) library.
See the [`THANKS.md`](./THANKS.md) file for more details and attribution.

## License

This project is licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([`LICENSE-MIT`](./LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Legal attribution and full license text for third-party components are in the [`NOTICE`](./NOTICE) file at the root of this repository.
