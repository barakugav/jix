> **These numbers are low-fidelity** - a --fast validation run: reduced sampling, enough to check the pipeline, not to quote.

*Measured on linux-x86_64 (AMD EPYC 7763 64-Core Processor), linux-aarch64 (aarch64). Benched commit `94f72a73`.*

# jix benchmarks

jix is a multi-dimensional array library with block-compressed storage and lazy operation chains.
This page measures both against the libraries you would otherwise reach for: `ndarray` in Rust,
NumPy, Blosc2 and Zarr in Python.

**Machines.** GitHub-hosted runners: `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (aarch64).
rustc 1.89.0, Python 3.13, numpy 2.3.1, blosc2 4.9.1, zarr 3.0.6.

**Everything is single-threaded.** jix has no threading. Blosc2, Zarr and numexpr are each pinned to
one thread, and NumPy's elementwise and reduction kernels are single-threaded regardless. A
multi-threaded Blosc2 will beat jix on throughput-bound work; that comparison is not on this page.

**Codec settings are matched** across jix, Blosc2 and Zarr: zstd level 3, byte-shuffle. The array
is `[130000, 200]` throughout, in Rust and Python alike - 104 MB as `f32`.

**Reading the plots.** Each tick on the x axis is one configuration; each bar is one library. Bars
are scaled to the baseline - NumPy in Python, `ndarray` in Rust - which is the gray bar and which by
definition tops out on the black rule at 1x. **Shorter is better**: a bar at 4x took four times as
long, or used four times the memory, or stored four times the bytes. The one exception is
compression throughput, which is plotted in absolute MB/s with no baseline, where taller is faster.

The scale is logarithmic, so a bar twice the height of another is not twice the number. The value
above each bar is exact, and the collapsed table under each plot carries the absolute measurements.

`jix` and `blosc2` always mean the byte-shuffled build - the configuration anyone would actually
use. The unfiltered variants appear only in the compression section, where the filter is the thing
being measured. `jix-plain` is jix over an ordinary uncompressed in-memory buffer: it isolates the
cost of jix's operation machinery from the cost of decompression.

---

## Reading a region

The case block-compressed storage exists for: pull a small region out of a large array without
decompressing the whole thing. Every timed call reads a different one of 1024 randomly placed
regions, so nothing stays warm in cache.

![Reading a random region](plots/read.png)

NumPy is the floor, not a competitor: it holds the array uncompressed in RAM and a read is a slice
plus a memcpy. The question this plot answers is what you pay for the four-to-fifty-fold drop in
memory that the next section shows, and how that compares to the other libraries that offer the
same trade.

The first three cases are dominated by per-call overhead. The last three are where each read does
real work, and they are the honest measure of decode throughput - Zarr in particular spends most of
a small read inside its Python indexing layer rather than its codec.

The `block 64x200 / read 16x16` case is the one to learn from: a read smaller than a block still decompresses a
whole block and throws most of it away. **Match the block shape to how you read.** jix picks a
block shape from the CPU cache sizes when you do not pass one, which is a fair default and a poor
choice if your access pattern is lopsided.

`blosc2-chunked` is Blosc2 with `chunks` forced equal to `blocks`. Left to choose for itself Blosc2
picks ~8.7 MiB chunks against a 4 KiB block; the block is still the decompression unit, but walking
a large chunk's block table costs real time - about 2.3x here. It also silently rewrites a
requested `(64, 200)` block to `(52, 200)` unless `chunks` is pinned too. Both arms are on the plot
so Blosc2 is shown at its best as well as at its default.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | blosc2-chunked | zarr |
|---|---|---|---|---|---|---|
| block 16x200 read 1x200 800 B | 720 ns | 842 ns | 2.63 us | 87.49 us | 24.53 us | 464.17 us |
| block 16x200 read 16x200 13 KB | 1.34 us | 1.49 us | 4.44 us | 89.71 us | 26.48 us | 646.75 us |
| block 64x200 read 16x16 1 KB | 909 ns | 977 ns | 7.99 us | 46.40 us | 28.14 us | 518.90 us |
| block 64x200 read 256x200 205 KB | 10.36 us | 10.49 us | 30.28 us | 71.83 us | 52.16 us | 1.55 ms |
| block 64x200 read 4096x200 3 MB | 141.63 us | 142.87 us | 361.85 us | 432.85 us | 383.34 us | 15.29 ms |
| block 64x200 read whole array 104 MB | 13.07 ms | 13.53 ms | 19.12 ms | 24.96 ms | 24.27 ms | 472.53 ms |

</details>

### Rust: against a raw `ndarray` slice

![Rust: reading a random region](plots/rust_read.png)

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| block 32x32 read 32x32 4 KB | 4.87 us | 387 ns | 8.87 us |
| block 32x32 read 1x200 800 B | 135 ns | 173 ns | 15.90 us |
| block 512x32 read 128x200 102 KB | 4.65 us | 5.16 us | 262.83 us |

</details>

---

## Compression

![Compressed size](plots/compress_ratio.png)

Each bar is what the library stores as a fraction of the raw array, `prod(shape) * itemsize`, so
NumPy is 1x by definition and shorter means smaller on disk. This is the one section where the
unfiltered builds appear, because the byte-shuffle filter is exactly what separates them.

Sizes track Blosc2 closely, which is what should happen - same codec, same filter, so what is
really being measured is the cost of the block layout. Zarr uses Blosc internally and lands on the
same numbers.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-noshuffle | jix | blosc2-noshuffle | blosc2 | zarr |
|---|---|---|---|---|---|---|
| random | 1.0x smaller | 1.0x smaller | 1.0x smaller | 1.0x smaller | 1.0x smaller | 1.0x smaller |
| smooth | 1.0x smaller | 753.7x smaller | 769.3x smaller | 720.5x smaller | 668.5x smaller | 649.2x smaller |
| 16 unique | 1.0x smaller | 4.8x smaller | 7.9x smaller | 4.5x smaller | 7.6x smaller | 7.6x smaller |
| 4 unique | 1.0x smaller | 7.3x smaller | 12.2x smaller | 7.7x smaller | 12.7x smaller | 12.7x smaller |

</details>

![Compression throughput](plots/compress.png)

Absolute throughput, in original array bytes per second - no baseline, and taller is faster. This
is the one plot on the page that is not a ratio: there is no meaningful NumPy arm to normalize
against, since not compressing is not a compression speed.

Note that the more compressible the data, the *faster* compression gets. That is the mechanism
behind the distribution cases in the next section.

<details>
<summary>Absolute numbers</summary>

| case | jix-noshuffle | jix | blosc2-noshuffle | blosc2 | zarr |
|---|---|---|---|---|---|
| random | 62 ms | 82 ms | 106 ms | 194 ms | 901 ms |
| smooth | 23 ms | 49 ms | 20 ms | 47 ms | 896 ms |
| 16 unique | 376 ms | 92 ms | 1208 ms | 413 ms | 1029 ms |
| 4 unique | 328 ms | 205 ms | 1084 ms | 446 ms | 1024 ms |

</details>

---

## Negate

![Negate](plots/negate.png)

On an uncompressed in-memory buffer jix matches NumPy. That parity is what makes the chain results
further down meaningful: entering a jix pipeline costs nothing that has to be earned back later.

Against the other compressed-array libraries jix is several times faster. Both write their result
into a plain uncompressed NumPy buffer - Blosc2's `expr[:]` does not re-compress on the way out, so
it is not being billed for a pass jix skips.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | zarr |
|---|---|---|---|---|---|
| f32 | 11.37 ms | 13.03 ms | 69.13 ms | 104.46 ms | 79.11 ms |
| i32 | 12.79 ms | 16.19 ms | 22.68 ms | 69.23 ms | 56.80 ms |

</details>

### The same operation, on data that compresses differently

![Negate, by how well the data compresses](plots/negate_dist.png)

Nothing changes here but the number of distinct values in the data. Better compression means less
memory to move and longer zstd matches, so the operation gets faster. The same effect shows up in
the filter choice: byte-shuffled data negates measurably faster than unshuffled, with identical op
code on both sides.

It does not catch NumPy, and it should not be expected to. An elementwise operation writes a
full-size uncompressed output whatever the input was, so compression only ever helps the read half
of the work.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | zarr |
|---|---|---|---|---|---|
| random | 12.22 ms | 13.88 ms | 67.49 ms | 102.04 ms | 71.31 ms |
| smooth | 11.55 ms | 12.66 ms | 69.83 ms | 104.45 ms | 79.59 ms |
| 16 unique | 12.63 ms | 13.27 ms | 90.71 ms | 139.43 ms | 88.62 ms |
| 4 unique | 12.28 ms | 12.90 ms | 96.30 ms | 141.46 ms | 87.23 ms |

</details>

---

## Add

![Add](plots/add.png)

The same shape as negate with two operands instead of one.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | zarr |
|---|---|---|---|---|---|
| f32 | 15.13 ms | 16.79 ms | 130.72 ms | 167.22 ms | 152.62 ms |
| i32 | 15.90 ms | 19.53 ms | 35.32 ms | 78.05 ms | 155.76 ms |

</details>

---

## Reductions

![Reductions](plots/reduction.png)

`sum` and `std` are on one plot because they land on opposite sides of the rule, and splitting them
would be choosing which result to show.

jix's reduction kernels read in whatever layout the source already has rather than forcing a
canonical order, and they vectorize integer accumulation that NumPy leaves scalar. **Compare the
two platform rows before reading anything into the integer numbers**: jix's kernels are tuned on
arm64, and NumPy is a general-purpose library with far more x86 attention behind its hot paths.
That is a statement about which machine you are on, not about which library is faster.

`std` goes the other way on every platform - a real deficiency in the kernel rather than a
measurement artifact, and the clearest thing to fix next. On compressed input Blosc2 also edges jix
out on reductions while losing badly on elementwise; both facts are in the same chart.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | zarr |
|---|---|---|---|---|---|
| sum f32 axis 0 | 6.83 ms | 6.38 ms | 61.27 ms | 64.12 ms | 81.58 ms |
| sum f32 axis 1 | 8.01 ms | 6.57 ms | 61.63 ms | 66.58 ms | 82.54 ms |
| sum f32 all | 6.71 ms | 5.50 ms | 60.31 ms | 67.62 ms | 77.55 ms |
| sum i32 axis 0 | 12.30 ms | 8.26 ms | 18.77 ms | 22.75 ms | 65.14 ms |
| sum i32 all | 9.24 ms | 7.21 ms | 16.91 ms | 25.05 ms | 59.63 ms |
| std f32 axis 0 | 35.48 ms | 41.42 ms | 98.45 ms | 176.38 ms | 109.11 ms |
| std f32 all | 33.26 ms | 121.73 ms | 182.49 ms | 156.46 ms | 104.73 ms |

</details>

### Rust

![Rust: elementwise and reductions](plots/rust_op.png)

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| negate f32 | 13.69 ms | 13.81 ms | 53.13 ms |
| negate i32 | 12.94 ms | 13.93 ms | 42.49 ms |
| add f32 | 16.07 ms | 17.15 ms | 97.65 ms |
| add i32 | 14.84 ms | 16.36 ms | 76.16 ms |
| sum f32 axis 0 | 13.07 ms | 6.39 ms | 44.17 ms |
| sum f32 axis 1 | 5.99 ms | 6.68 ms | 44.44 ms |
| sum i32 all | 5.12 ms | 7.09 ms | 36.35 ms |

</details>

---

## Operation chains

Every jix operation returns a lazy view; the whole chain is encoded in the type and runs in a
single pass when output is requested. NumPy and `ndarray` make a full pass per step, so the gap
should widen as the chain gets longer.

Every step is a binary operation over two source arrays, starting from `a*a` - no scalar constants
anywhere. Two reasons: a chain of scalar operations would measure jix's handling of a 0-stride
operand rather than whether fusing pays off, and a chain of scalar constants can be constant-folded
by a compiler into a single operation, which measures nothing at all.

![Operation chains](plots/chain.png)

A chain of N steps is N x 26M element-operations whether or not it is fused, so the total time
grows either way. What fusing removes is the memory traffic: NumPy re-reads an operand and writes a
full intermediate per step, while jix materializes the chain one read region at a time, so every
intermediate for that region stays in cache. The bars show whether that saved traffic is enough to
win, which is why the trend across chain lengths is the result and a single length would be a
number with no mechanism behind it.

Everything here runs at jix's default read region. That default is derived from the CPU's cache
sizes, and on a machine with a large L2 it comes out far larger than a chain wants - shrinking it
to a region that fits L1 is worth a further 1.4x on the longest chains. That is a tuning result
rather than what a caller gets by default, so it is not what these bars show; the measurement is in
the repository's benchmark notes.

`f32` only here; the integer chain behaves the same way and adds nothing but width.

The `exp/log` case is the counter-example, and it is on the plot on purpose. `log(exp(a) + exp(b))`
is dominated by transcendental math rather than memory traffic, so the libm implementation decides
the outcome and the intermediates jix saves are noise beside it. Chains win when the operations are
cheap and the array is large; they do not when one expensive kernel dominates.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix |
|---|---|---|---|
| 1 op | 12.36 ms | 13.94 ms | 125.82 ms |
| 2 ops | 27.97 ms | 22.31 ms | 187.35 ms |
| 4 ops | 57.96 ms | 32.36 ms | 305.53 ms |
| 8 ops | 116.56 ms | 49.61 ms | 527.52 ms |
| 16 ops | 236.31 ms | 83.17 ms | 969.28 ms |
| 32 ops | 488.79 ms | 152.44 ms | 1.87 s |
| exp/log | 140.82 ms | 224.97 ms | 339.82 ms |

</details>

### Peak memory

![Operation chains: peak memory](plots/chain_memory.png)

Peak RSS of a fresh subprocess running the same chain, so NumPy's C-level allocations are included.

All three are flat across chain length, and that is worth being precise about: NumPy allocates an
intermediate per step but frees each one as the next is produced, so its peak is the two inputs
plus one live intermediate, not a growing pile. Avoiding intermediates entirely is therefore worth
a fifth, not a multiple.

The real difference is what the arrays cost to hold. The compact arm keeps its inputs compressed
for the whole computation, and that is where the factor comes from. Memory has no noise floor,
which makes this the cleanest measurement on the page.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix |
|---|---|---|---|
| 1 op | 385 MB | 382 MB | 193 MB |
| 2 ops | 480 MB | 382 MB | 194 MB |
| 4 ops | 484 MB | 382 MB | 194 MB |
| 8 ops | 484 MB | 382 MB | 194 MB |
| 16 ops | 484 MB | 382 MB | 193 MB |
| 32 ops | 484 MB | 382 MB | 194 MB |

</details>

### Rust

The Rust half runs the same chain against `ndarray` in its efficient form - an owned left operand
reuses its buffer, so it does not allocate per step, but it still makes one full read-write pass
per step.

![Rust: operation chains](plots/rust_chain.png)

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| 1 op | 13.94 ms | 14.69 ms | 90.93 ms |
| 2 ops | 22.32 ms | 17.21 ms | 134.37 ms |
| 4 ops | 39.09 ms | 18.66 ms | 216.06 ms |
| 8 ops | 72.89 ms | 22.71 ms | 377.45 ms |
| 16 ops | 136.14 ms | 36.68 ms | 702.41 ms |

</details>

---

## A reduction inside a broadcast

`a / a.std(axis)`, with the reduction broadcast back over the array, is a different shape from an
elementwise chain, and it is the case where laziness costs rather than pays.

![Rust: normalize, a reduction inside a broadcast](plots/rust_normalize.png)

The reduction sits inside the lazy pipeline, so producing an output element re-runs it over that
element's whole column: O(N*M) instead of O(N+M). Over axis 0 the reduced axis is 130000 elements
long and the result is what the plot shows. Over axis 1 it is 200 elements and jix comes out ahead.

This is not a kernel deficiency, it is what fusing a broadcast reduction means, and the remedy is
the one the `ops` module already gives for reshape: materialize the reduction before broadcasting
it when the reduced axis is long. It is on the page because a caller who writes this expression
without knowing that will pay for it.

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| normalize axis 0 | 103.22 ms | 8.32 s | 16.76 s |
| normalize axis 1 | 139.82 ms | 131.06 ms | 211.82 ms |

</details>

---

## Axis order (Rust)

`ndarray` classifies an array's layout four ways - C, F, first-axis-contiguous,
last-axis-contiguous - and falls back to logical-order iteration when none of them fit. jix sorts
every axis by descending stride, so it always walks memory in order.

The difference needs three dimensions to appear at all. A 2-D transpose is exactly F-layout and
`ndarray` handles it perfectly. A *rotation* of a 3-D array is neither C nor F and has a stride-1
axis at neither end, so `ndarray` ends up iterating with the largest stride in the innermost loop.

![Rust: axis order](plots/rust_axis_order.png)

The last two cases are controls, and they are on the plot deliberately: they show the effect is
specific to layouts `ndarray` cannot classify, not a general claim about strided data.

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain |
|---|---|---|
| 3-D rotate [1,2,0] f32 | 28.76 ms | 36.14 ms |
| 3-D rotate [1,2,0] i32 | 29.31 ms | 38.27 ms |
| 3-D reverse [2,1,0] f32 | 29.23 ms | 37.69 ms |
| 2-D transpose f32 | 199.91 us | 251.82 us |

</details>

There is no Python counterpart - NumPy sorts axes too, so there is nothing to compare.

---

## Reproducing

    python jix/benches/run.py --report          # rust, only the comparison benches
    python jix-py/python/benches/run_all.py     # python

The CI matrix across the three runners is documented in
[`docs/benchmarks.md`](../docs/benchmarks.md), along with `scripts/bench/compare.py` for
same-machine A/B against a base ref. Every plot and every table on this page is generated from the
benchmark JSON by `benches/report_bars.py`; nothing is typed by hand.
