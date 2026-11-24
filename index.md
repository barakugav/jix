---
layout: default
title: "jix: compressed nd-arrays with fast random access"
subtitle: "Block-compressed, lazy nd-arrays for Rust and Python - read only the blocks you touch, and run a whole op chain in one pass."
description: "Block-compressed, lazy nd-arrays for Rust and Python - read only the blocks you touch, and run a whole op chain in one pass."
image: /assets/og-image.png
kicker: "Rust + Python"
date: 2026-11-23
# author: defaults to `author` in _config.yml
---

Suppose you want to minimize the memory/disk usage of your large nd-array, and still be able to
perform random access to it efficiently.
Without any libraries, you have to choose between a regular Python-NumPy/Rust-ndarray array,
which is fast but memory-hungry, or a compressed format (like zip/zstd of the whole array), which
saves space but makes random access expensive as you have to decompress the entire array to read a
small region. [jix](https://github.com/barakugav/jix) is a library that compresses an ndarray
in a format that supports reads and operations efficiently.

{% include bench-figure.html plot="read.svg"
  alt="Reading a random region: jix, jix-plain, Blosc2 and Zarr against NumPy, for different block and read shapes"
  caption="Reading a random region, speed relative to NumPy (higher is faster)." %}

{% include bench-figure.html plot="compress_ratio.svg"
  alt="Compressed size of jix, Blosc2 and Zarr, with and without byte shuffle, relative to the raw array, for random, smooth and few-unique-value data"
  caption="How many times smaller a 104 MB i32 array gets when stored (higher is smaller). Random data doesn't compress; smooth data gets about 34x smaller with jix's default byte-shuffle filter." %}

## 1D Blocks

Starting simple with a 1D array (vector): how can we compress it and still allow efficient random
access to its elements?
You can divide it into blocks, of some fixed length (say 100 elements), compress each block
independently and store them in a table. When you want to read an element, you only need to
decompress the block that contains it, rather than the entire array. Pretty straightforward, right?

{% include diagrams/blocks-1d.html %}

jix uses zstd to compress each block, with additional dtype-aware filters to improve compression
efficiency.


## ND Blocks

The same idea works for N dimensions: you choose a fixed block shape (e.g. 100x100 for 2D), divide
the array into blocks of that shape, compress each block independently, and store them in a table.
When you want to read a rectangular region, you compute which blocks overlap the region and only
decompress those blocks.

{% include diagrams/blocks-nd.html %}

<div class="code-tabs" markdown="1">

```rust
use jix::Array;
use jix::storage::CompactParams;
use ndarray::{Array2, Ix2};

// A 600x800 array, cut into 100x100 blocks, each compressed on its own
let data = Array2::<f32>::from_shape_fn((600, 800), |(i, j)| (i + j) as f32);
let params = CompactParams::new().block_shape(&[100, 100])?;
let a = Array::compact_ndarray_with(&data, params)?;

// Decompresses only the 6 (out of 48) blocks that overlap the region
let region = a.to_ndarray_sub::<Ix2>(&[150..250, 380..520])?;
let x = a.get(&[599, 0])?; // a single element, a single block
```

```python
import jix
import numpy as np

# A 600x800 array, cut into 100x100 blocks, each compressed on its own
data = np.random.rand(600, 800).astype(np.float32)
a = jix.compact(data, params={"block_shape": [100, 100]})

# Decompresses only the 6 (out of 48) blocks that overlap the region
region = a[150:250, 380:520]  # a np.ndarray
x = a[599, 0]  # a single element, a single block
```

</div>

Congrats! You now have a block-compressed array that allows efficient random access to any region
without decompressing the entire array. This is jix's `Compact` storage, and it supports any
indexed read.

A classical use case is a machine learning dataloader that picks a random sample out of a large
dataset, for example an audio or weather measurement segment out of a long recording.
The next sections are focused on operations, which are built on top of the `Compact` storage,
but if all you need is reads this storage is sufficient on its own.


{% include bench-figure.html plot="compress.svg"
  alt="Compression throughput of jix, Blosc2 and Zarr, with and without byte shuffle, for random, smooth and few-unique-value data, on x86_64 and aarch64"
  caption="Compressing a 104 MB i32 array, in MB/s (higher is faster). On data that compresses, jix's byte shuffle makes compression faster, not slower: zstd handles long runs of similar bytes better and faster." %}


## Elementwise Operations

Suppose that you have two block-compressed arrays of the same shape and dtype, `a` and `b`, and you
want to perform elementwise addition `a+b` between them. A naive approach will decompress `a` in
full, decompress `b`, then call NumPy or Rust's ndarray to perform the addition on the fully
decompressed arrays, and compress the output again. This works, but we can do better than that!
Instead, we can iterate over matching blocks in `a`, `b`, and the output,
decompressing + processing + compressing one
block at a time. This way, we never hold the fully decompressed arrays in memory, and we only touch
the blocks that are actually needed for the computation.

{% include diagrams/elementwise.html %}

<div class="code-tabs" markdown="1">

```rust
let a = Array::compact_ndarray(&data_a)?;
let b = Array::compact_ndarray(&data_b)?;

// Block by block: decompress a block of `a` and of `b`, add, compress the sum
let c = (a + b).compact()?;
```

```python
a = jix.compact(data_a)
b = jix.compact(data_b)

# Block by block: decompress a block of `a` and of `b`, add, compress the sum
c = (a + b).compact()
```

</div>


## Elementwise Operation Chains

Now instead of two, what if you have three block-compressed arrays, all of the same shape and
dtype, `a`, `b`, and `c`, and you want to perform `(a+b)*c`? Using the approach above naively, we
iterate block by block of `a+b` and decompress, process and compress, and then again iterate block
by block of the intermediate result `(a+b)` and `c` to decompress and compute `(a+b)*c` and compress
again. That is wasteful. A better approach is to "fuse" the chain, iterate over blocks
of `a`, `b`, and `c` simultaneously, compute `(a+b)*c` block by block, and compress once. For this,
our execution model must become lazy, because `a+b` doesn't "know" `c` is going to multiply the
result. In jix, every operation is lazy, so `a+b` is of type `Array<Add<Compact, Compact>>` and
`(a+b)*c` is of type `Array<Mul<Add<...>, Compact>>`, kind of like Rust iterators.
An expression of arrays can be materialized into a new block-compressed array
(`.compact() -> Array<Compact>`) or into a standard in-memory array (Python `.numpy() -> np.ndarray`
or Rust `.to_ndarray() -> ndarray::ArrayD<T>`).

{% include diagrams/chain.html %}

<div class="code-tabs" markdown="1">

```rust
// Nothing is computed yet, `expr` is a lazy view of type
// Array<Mul<Add<Compact<f32>, Compact<f32>>, Compact<f32>>>
let expr = (a + b) * c;

let out = expr.compact()?; // one fused pass, compress once
let arr = expr.to_ndarray::<Ix2>()?; // or into an in-memory ndarray
let part = expr.to_ndarray_sub::<Ix2>(&[0..10, 0..10])?; // or compute only a region
expr.write_to_file(Path::new("out.jix"))?; // or write to a compressed .jix file
```

```python
expr = (a + b) * c  # lazy, nothing is computed yet

out = expr.compact()  # one fused pass, compress once
arr = expr.numpy()  # or into a NumPy array
part = expr[0:10, 0:10]  # or compute only a region
expr.write_to("out.jix")  # or write to a compressed .jix file
```

</div>


## Generic Storage and Non-Elementwise Operations

jix arrays are built on a generic abstract `Storage` which has a main function
`read_data(sub_region)`. Lazy elementwise operations implement this function by reading the
corresponding sub-regions from their input arrays and applying the operation block by block, but you
can imagine how `broadcast`, `insert_dim`, `permute_dims`, and even a generic `reshape`, can be
implemented lazily as well - every such op plays around with the sub-region coordinates before
calling the inner (original) storage one or more times.

Reduction ops are also lazy. For example, for an array `a` of shape `(100, 40)`, the lazy array
`a.sum(dim=0)` is a lazy array of shape `(40,)` that implements `read_data` by reading the larger
corresponding sub-region from `a` and summing along the first axis. A reduction op may split the
inner read into smaller reads to keep the peak memory usage low.

jix supports elementwise operations, reductions, shape ops, concat/stack, and more, all as lazy
views. In Python we build a graph of these lazy ops dynamically, and in Rust we do the same
using the type system so that the compiler can inline and fuse operations at compile time.

{% include diagrams/read-data.html %}

<div class="code-tabs" markdown="1">

```rust
let a = Array::compact_ndarray(&data)?; // shape [100, 40]

let s = a.view().sum(0); // shape [40]
let part = s.to_ndarray_sub::<Ix1>(&[10..20])?; // reads only a[0..100, 10..20]

let t = a.view().permute_dims(&[1, 0]); // shape [40, 100]
let r = a.view().reshape(&[20, 200]); // shape [20, 200]
let both = jix::ops::concatenate([a.view(), a.view()], 0); // shape [200, 40]

// Still lazy: mean, insert_dim, broadcast and sub run block by block on read
let mean = a.view().mean(0).insert_dim(0).broadcast(&[100, 40]);
// The compiler knows the type and can optimize the fused ops
// Array<Sub<Compact, Broadcast<InsertDim<Mean<Compact>>>>>
let centered = (a.view() - mean).compact()?;
```

```python
a = jix.compact(data)  # shape (100, 40)

s = a.sum(dim=0)  # shape (40,)
part = s[10:20]  # reads only a[0:100, 10:20]

t = a.T  # shape (40, 100)
r = a.reshape((20, 200))  # shape (20, 200)
both = jix.concat([a, a], dim=0)  # shape (200, 40)

# Still lazy: mean, broadcasting and sub run block by block on read
centered = (a - a.mean(dim=0)).compact()
```

</div>


## Peak memory usage and processing arrays larger than RAM

A cool unlock of the lazy evaluation model, and evaluating in blocks, is
a [very low memory usage](https://github.com/barakugav/jix/blob/{{ site.source_commit }}/scripts/bench/report/report.md#peak-memory),
as only small, block-sized intermediate buffers are allocated.
It also allows us to process very large arrays, even if they are bigger than the machine's RAM
(even in their compressed form). The way you would go about it: if you have a large array saved
to disk, whether a raw binary file or in `.jix` compressed format, you can load the array using memory
map, build a pipeline of operations, and write the result back to disk directly. The pipeline will
run block by block, never materializing the entire array into RAM. The result can either streamed
to another file or reduced to a smaller in-memory value.

<div class="code-tabs" markdown="1">

```rust
use jix::Array;
use jix::dtype::Dtyped;
use jix::storage::ArchiveValidation;

// A .jix file, memory mapped: blocks are paged in only when read
let path = Path::new("huge.jix");
let len = std::fs::metadata(path)?.len();
// Safety: the file is not modified while `src` is alive
let src = unsafe { Array::read_from_file_mmap(path, 0, len, ArchiveValidation::default())? };

// Build a pipeline and stream it to disk, one block at a time
let out = src.into_typed::<f32>()?.exp().map(|x| x * 0.5);
out.write_to_file(Path::new("out.jix"))?;

// A raw f32 file of shape [rows, 200], memory mapped, no copy
let file = std::fs::File::open("huge.f32")?;
let mmap = unsafe { memmap2::Mmap::map(&file)? };
// Safety: the file holds rows*200 f32 values and outlives `src`
let src = unsafe {
    jix::PlainArrayView::<f32>::plain_ndarray_ptr(
        mmap.as_ptr(),
        &[rows, 200],
        &[200 * 4, 4], // strides in bytes
        f32::DTYPE,
    )?
};
src.sum(1).write_to_file(Path::new("row_sums.jix"))?;
```

```python
# A .jix file, memory mapped: blocks are paged in only when read
src = jix.read_array("huge.jix", mmap=True)
out = src.exp() * 0.5
out.write_to("out.jix")  # streamed to disk, one block at a time

# A raw f32 file, memory mapped by NumPy; jix wraps it without a copy
raw = np.memmap("huge.f32", dtype=np.float32, mode="r", shape=(rows, 200))
src = jix.asarray(raw)
src.sum(dim=1).write_to("row_sums.jix")
```

</div>


## Performance

I could write a full blog post just on the performance optimizations that went into jix (I know, a
cliche). This was at least 75% of the time I invested in the library. The fact that the mainstream
library for this kind of data structure, Blosc2, is using (mostly) NumPy under the hood for its
expression engine, forced me to optimize the library again and again just to achieve a close to
comparable runtime. Honestly, this was quite a journey and it made me value NumPy much more, seeing
the level of optimization and thought that went into it, even though its basic concept, the ndarray, is
pretty straightforward (it's just a shape, data pointer and strides, how hard could it be, right??).

The [benchmark
report](https://github.com/barakugav/jix/blob/{{ site.source_commit }}/scripts/bench/report/report.md)
compares jix against a few libraries in Python and Rust, on reads, compression, elementwise ops,
reductions and lazy chains, etc. For example, elementwise add and reductions:

{% include bench-figure.html plot="add.svg"
  alt="Adding two arrays: jix, jix-plain, Blosc2 and Zarr against NumPy, for f32 and i32, on x86_64 and aarch64"
  caption="Adding two 104 MB arrays, speed relative to NumPy (higher is faster). jix-plain runs the same ops on uncompressed in-memory arrays." %}

{% include bench-figure.html plot="reduction.svg"
  alt="Sums and standard deviations: jix, jix-plain, Blosc2 and Zarr against NumPy, on x86_64 and aarch64"
  caption="Sums and standard deviations over a 104 MB array, speed relative to NumPy (higher is faster)." %}


### Optimizing Elementwise Negation

I'll now take you through my journey of optimizing one of these operations as an example of the
optimizations in jix. We will try to optimize the simplest elementwise
op: negating an array. This function is the bare operation, operating on uncompressed data
without dividing into blocks, and can be used by an outer loop that iterates over blocks,
decompresses each one, calls this function and compresses it again.
The function accepts a preallocated destination array and a source array,
along with their shapes and strides.

(I didn't invent most of these ideas, but this is my journey discovering them.)

#### 1. Baseline

We will start with the most straightforward implementation: a loop over an n-dimensional index,
calculating the source and destination offsets for each element using the strides, and then
performing the negation.

```rust
unsafe fn negative_v1<T>(
    dst: *mut T,
    src: *const T,
    shape: &[usize],
    dst_strides: &[usize],
    src_strides: &[usize],
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    assert!(!shape.contains(&0), "unsupported for simplicity");
    let ndim = shape.len();
    let mut nd_index = vec![0; ndim];
    loop {
        // calc element offsets
        let (mut dst_offset, mut src_offset) = (0, 0);
        for i in 0..ndim {
            dst_offset += nd_index[i] * dst_strides[i];
            src_offset += nd_index[i] * src_strides[i];
        }

        // actual computation
        let elm = src.byte_add(src_offset).read();
        let neg = -elm;
        dst.byte_add(dst_offset).write(neg);

        // advance the multi-dim index
        let mut i = ndim;
        loop {
            if i == 0 {
                return; // done
            }
            i -= 1;
            nd_index[i] += 1;
            if nd_index[i] < shape[i] {
                break;
            }
            nd_index[i] = 0;
        }
    }
}
```

We are going to use llvm-mca, a tool to estimate the number of CPU cycles for a given piece of code,
to measure the performance of each version. The v1 loop in its steady state (long running inner
loop) takes **63506** cycles for 8192 elements of f32 on Ice Lake (no `target-cpu=native`).


#### 2. Inner loop

The advancing of the multi-dim index is hard for the compiler to optimize, as it doesn't recognize
the pattern of the loop. We can help the compiler by using a multi-dim index only for the outer
dims, and have an `inline(never)` inner loop for the innermost dimension, which the compiler can
optimize independently.

```rust
unsafe fn negative_v2<T>(
    // ... like v1
)
{
    assert!(!shape.contains(&0), "unsupported for simplicity");
    assert!(!shape.is_empty(), "unsupported for simplicity");

    // a loop over all dims except the innermost one
    let outer_ndim = shape.len() - 1;
    let mut nd_index = vec![0; outer_ndim];
    loop {
        // calc element offsets for outer dimensions
        let (mut dst_offset, mut src_offset) = (0, 0);
        for i in 0..outer_ndim {
            dst_offset += nd_index[i] * dst_strides[i];
            src_offset += nd_index[i] * src_strides[i];
        }

        negative_v2_inner_loop(
            dst.byte_add(dst_offset),
            src.byte_add(src_offset),
            shape[outer_ndim],
            dst_strides[outer_ndim],
            src_strides[outer_ndim],
        );

        // advance the multi-dim index for outer dimensions
        // ... like v1
    }
}
#[inline(never)]
unsafe fn negative_v2_inner_loop<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
)
{
    // simple 1D loop
    for i in 0..len {
        let elm = src.byte_add(i * src_stride).read();
        let neg = -elm;
        dst.byte_add(i * dst_stride).write(neg);
    }
}
```


llvm-mca shows improvement: **63506 -> 7518**.

From now on we will focus only on the inner loop.

#### 3. Contiguity flags as const generics

The v2 inner loop is straightforward and handles both contiguous and strided memory layouts, but the
compiler cannot easily vectorize reads and writes due to the potential striding. By adding const
generic parameters to indicate whether the input and output are contiguous, and adding a small
runtime dispatch at the outer loop, we can get an auto-vectorized inner loop.

```rust
unsafe fn negative_v3_inner_loop<
    T,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
>(
    dst: *mut T,
    src: *const T,
    // ... like v2
)
{
    for i in 0..len {
        let elm = if IN_CONTIGUOUS {
            src.add(i).read()
        } else {
            src.byte_add(i * src_stride).read()
        };
        let neg = -elm;
        if OUT_CONTIGUOUS {
            dst.add(i).write(neg);
        } else {
            dst.byte_add(i * dst_stride).write(neg);
        }
    }
}
```

llvm-mca shows improvement: **7518 -> 2057**.


#### 4. Noalias hack

There is still something that can block auto-vectorization, and that's aliasing. The compiler cannot
prove that `dst` and `src` do not alias, so writes may change the contents of `src`, which prevents
certain optimizations. In Rust there isn't a built-in way to express that two raw pointers do not
alias, like C++ `__restrict`, so we use a hack - a mut reference is noalias. We use two wrapper
types `PtrNoalias` and `PtrMutNoalias`, create them around the raw pointers outside the inner loop,
and extract the pointers right away inside it.

```rust
struct PtrNoalias<'a, T>(&'a [u8], PhantomData<T>);
struct PtrMutNoalias<'a, T>(&'a mut [u8], PhantomData<T>);
impl<'a, T> PtrNoalias<'a, T> {
    unsafe fn new(ptr: *const T, span: usize) -> Self {
        unsafe { Self(slice::from_raw_parts(ptr.cast::<u8>(), span), PhantomData) }
    }
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr().cast::<T>()
    }
}

unsafe fn negative_v4_inner_loop<
    T,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
>(
    dst: PtrMutNoalias<'_, T>,
    src: PtrNoalias<'_, T>,
    // ... like v3
)
{
    let dst = dst.as_mut_ptr(); // extract raw pointer, LLVM remembers noalias
    let src = src.as_ptr();
    for i in 0..len {
        // ... like v3
    }
}
```

llvm-mca shows no improvement, because the compiler already generated a specialized loop for v3 with
noalias guarded by a runtime check. v4 guarantees the noalias loop is generated, and improves the
binary size.


#### 5. Manual unrolling

From the compiler's perspective, `len` could be any value, and it doesn't know we want it to
optimize the loop aggressively, so we introduce the `LANES` const generic for manual unrolling of
the inner loop. In case the input/output is contiguous, we read/write `[T; LANES]` chunks directly.
The value of `LANES` is chosen based on the size of T, the target SIMD width, etc.

```rust
unsafe fn negative_v5_inner_loop<
    T,
    const LANES: usize,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
>(
    // ... like v4
)
{
    let body_limit = len - len % LANES;
    let mut i = 0;
    while i < body_limit {
        let elms = if IN_CONTIGUOUS {
            src.add(i).cast::<[T; LANES]>().read()
        } else {
            std::array::from_fn(|j| src.byte_add((i + j) * src_stride).read())
        };

        let neg = std::array::from_fn(|j| -elms[j]);

        if OUT_CONTIGUOUS {
            dst.add(i).cast::<[T; LANES]>().write(neg);
        } else {
            for j in 0..LANES {
                dst.byte_add((i + j) * dst_stride).write(neg[j]);
            }
        }
        i += LANES;
    }
    while i < len {
        // ... tail like v4
    }
}
```

llvm-mca shows improvement: **2057 -> 1716**.


#### 6. Explicit SIMD

To further force the compiler to generate the asm we want, there is no better alternative than
explicit SIMD usage, so we use the `fearless_simd` crate. In addition to a more direct intent for
the compiler, we also gain from `fearless_simd`'s multiversioning capabilities, and the inner loop
is generated multiple times for different CPU targets and is selected at runtime.

```rust
unsafe fn negative_v6_inner_loop<
    T,
    TS: fearless_simd::SimdBase<S, Element = T> + std::ops::Neg<Output = TS>,
    const LANES: usize,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
    S: fearless_simd::Simd,
>(
    // ... like v5
    simd: S,
)
{
    while i < body_limit {
        unsafe {
            let elms = // ... like v5
            let neg = negative_v6_kernel::<T, TS, LANES, S>(elms, simd);
            // ... write to dst like v5
        }
        i += LANES;
    }
    // ... tail like v5
}
#[inline(always)]
fn negative_v6_kernel<T, TS, const N: usize, S: fearless_simd::Simd>(a: [T; N], simd: S) -> [T; N]
where
    TS: fearless_simd::SimdBase<S, Element = T> + std::ops::Neg<Output = TS>,
{
    let lanes = TS::LEN;
    if !N.is_multiple_of(lanes) {
        return std::array::from_fn(|i| -a[i]);
    }
    let mut result = [T::default(); N];
    for i in 0..(N / lanes) {
        let vec = TS::from_slice(simd, &a[i * lanes..][..lanes]);
        let neg = -vec; // SIMD negation
        neg.store_slice(&mut result[i * lanes..][..lanes]);
    }
    result
}
```

llvm-mca shows improvement: **1716 -> 437** (863 with AVX2).

From v1's 63506 cycles to v6's 437 cycles, that's a **~145x speedup**.
Sometimes the compiler generates good loops without us being explicit, and sometimes it doesn't.
It depends on the kernel, platform, code layout, and other voodoo factors.
From working on these kernels I learned that it's best to be explicit, and v6 (or similar
approaches) are the most consistent. Relying on the compiler alone is fragile and unreliable.

I hope this example gave you a sense of the detail and effort involved in optimizing jix's
low-level array operations. If you are curious, the real code can be found at
[jix/src/ops/elementwise/pipeline.rs](https://github.com/barakugav/jix/blob/{{ site.source_commit }}/jix/src/ops/elementwise/pipeline.rs#L602).
It is the common engine for all unary, binary, N-ary or fused elementwise operations.


## Thanks and miscellaneous

I didn't invent any of these ideas from scratch, and specifically I want to say a huge thanks to
[c-blosc2](https://github.com/Blosc/c-blosc2) and [NumPy](https://github.com/numpy/numpy) for
developing these approaches and mastering them.
jix can be seen as a port of ideas and natural evolution of these concepts, both to Rust
(over C) and for specific design and performance choices.

I use AI freely, in a way I judge as reasonable and helpful for my work. This project is far, far,
from slop, and I'm afraid to admit how many hours I put into it. I used AI mostly for scripts,
benches, quick experiments, docs and large refactors, and since Opus 5.5 I use it more freely. If
you want to judge for yourself, take a look at the commits.

If you want to use the library, you can find it on [crates.io](https://crates.io/crates/jix) and
[PyPI](https://pypi.org/project/jix/). If you have any feedback, questions or requests, feel free to
reach out to me at [barakugav@gmail.com](mailto:barakugav@gmail.com) or open an issue on the [GitHub
repository](https://github.com/barakugav/jix).


## Bonus optimization

When optimizing code, it's important to remember memory access is (almost) always the bottleneck,
especially when dealing with large arrays, and especially when the computation per element is
trivial from the perspective of the CPU. The single most effective optimization in jix is to
prioritize small strides for inner loops, and to match existing arrays' memory layout when allocating
new arrays.

Assume an input i8 array of size [100, 40], stored in Fortran (column-major) order, meaning its
strides are [1, 100]. When computing `negative`, if we allocate a C-order array (row-major), its
strides will be [40, 1], and no matter how we choose to iterate over the data, the inner loop
will have a large stride (40 or 100) for one of the arrays,
causing cache misses and poor performance. That's why when jix allocates output arrays, it always
tries to match the inputs' layouts. In addition, when iterating over a set of arrays, whether they
have matching layout or not, we always prefer to iterate over the dimension with the smallest stride
as the inner loop, to maximize cache efficiency. A general-purpose `NdIter` is used in jix, which
sorts dimensions by their strides, drops dimensions of size 1, and merges consecutive dimensions
when the merged dimension keeps a contiguous memory layout.

{% include diagrams/strides.html %}

This optimization alone can improve performance by up to 40x in certain scenarios. For some reason, Rust's
ndarray crate doesn't perform this optimization (as far as I know) and only supports special cases
(C-order, F-order, contiguous, etc.).

I really like this optimization, I hope you do too! All this auto-vectorization, SIMD, loop
unrolling, arithmetic float ops (out of order), etc., and memory access is still the real bottleneck.
Just thought it was cool.

Thanks for reading!
