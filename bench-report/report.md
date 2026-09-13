> **DRAFT.** Rendered from fake benchmark output by `build_fake_report.py` so the plot style can be
> iterated on before any real run. Some values are real (measured locally, single run); the rest are
> invented. Do not quote anything here.

# jix benchmarks

jix is a multi-dimensional array library with block-compressed storage and lazy operation chains.
This page measures both against the libraries you would otherwise reach for: `ndarray` in Rust,
NumPy, Blosc2 and Zarr in Python.

**Machine.** Apple M2 Pro, 10 cores, 32 GB, macOS 15.6; plus linux-x86_64 and linux-aarch64 CI
runners. rustc 1.89.0, Python 3.13, numpy 2.3.1, blosc2 4.9.1, zarr 3.0.6.

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
| block 16x200 read 1x200 800 B | 400 ns | 500 ns | 1.60 us | 76.00 us | 34.00 us | 330.00 us |
| block 16x200 read 16x200 13 KB | 600 ns | 700 ns | 2.60 us | 80.00 us | 36.00 us | 340.00 us |
| block 64x200 read 16x16 1 KB | 350 ns | 450 ns | 3.60 us | 52.00 us | 42.00 us | 336.00 us |
| block 64x200 read 256x200 205 KB | 12.00 us | 13.00 us | 26.00 us | 128.00 us | 95.00 us | 420.00 us |
| block 64x200 read 4096x200 3 MB | 190.00 us | 200.00 us | 380.00 us | 880.00 us | 780.00 us | 1.20 ms |
| block 64x200 read whole array 104 MB | 5.20 ms | 5.50 ms | 12.00 ms | 23.00 ms | 22.00 ms | 40.00 ms |

</details>

### Rust: against a raw `ndarray` slice

![Rust: reading a random region](plots/rust_read.png)

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| block 32x32 read 32x32 4 KB | 280 ns | 300 ns | 1.60 us |
| block 32x32 read 1x200 800 B | 350 ns | 380 ns | 10.50 us |
| block 512x32 read 128x200 102 KB | 11.00 us | 11.50 us | 42.00 us |

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
| smooth | 1.0x smaller | 3.5x smaller | 3.9x smaller | 3.4x smaller | 3.7x smaller | 3.7x smaller |
| 16 unique | 1.0x smaller | 24.0x smaller | 31.0x smaller | 22.5x smaller | 28.4x smaller | 28.4x smaller |
| 4 unique | 1.0x smaller | 46.0x smaller | 55.2x smaller | 41.0x smaller | 48.6x smaller | 48.6x smaller |

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
| random | 133 ms | 190 ms | 128 ms | 196 ms | 540 ms |
| smooth | 163 ms | 133 ms | 176 ms | 147 ms | 610 ms |
| 16 unique | 88 ms | 74 ms | 102 ms | 88 ms | 420 ms |
| 4 unique | 71 ms | 62 ms | 84 ms | 72 ms | 390 ms |

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
| f32 | 2.16 ms | 2.19 ms | 49.60 ms | 443.00 ms | 30.80 ms |
| i32 | 2.21 ms | 2.25 ms | 45.00 ms | 420.00 ms | 30.00 ms |

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
| random | 2.16 ms | 2.19 ms | 96.40 ms | 510.00 ms | 44.00 ms |
| smooth | 2.16 ms | 2.19 ms | 49.60 ms | 443.00 ms | 30.80 ms |
| 16 unique | 2.16 ms | 2.19 ms | 34.50 ms | 210.00 ms | 21.00 ms |
| 4 unique | 2.16 ms | 2.19 ms | 21.30 ms | 155.00 ms | 16.00 ms |

</details>

---

## Add

![Add](plots/add.png)

The same shape as negate with two operands instead of one.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | zarr |
|---|---|---|---|---|---|
| f32 | 2.89 ms | 2.88 ms | 99.10 ms | 561.00 ms | 59.20 ms |
| i32 | 2.95 ms | 2.94 ms | 94.00 ms | 540.00 ms | 58.00 ms |

</details>

---

## Reductions

![Reductions](plots/reduction.png)

`sum` and `std` are on one plot because they land on opposite sides of the rule, and splitting them
would be choosing which result to show.

jix's reduction kernels read in whatever layout the source already has rather than forcing a
canonical order, and they vectorize integer accumulation that NumPy leaves scalar. **Compare the
three platform rows before reading anything into the integer numbers**: jix's kernels are tuned on
arm64, and NumPy is a general-purpose library with far more x86 attention behind its hot paths.
That is a statement about which machine you are on, not about which library is faster.

`std` goes the other way on every platform - a real deficiency in the kernel rather than a
measurement artifact, and the clearest thing to fix next. On compressed input Blosc2 also edges jix
out on reductions while losing badly on elementwise; both facts are in the same chart.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix | blosc2 | zarr |
|---|---|---|---|---|---|
| sum f32 axis 0 | 2.33 ms | 2.03 ms | 49.60 ms | 34.70 ms | 30.90 ms |
| sum f32 axis 1 | 3.97 ms | 1.55 ms | 49.00 ms | 36.30 ms | 32.90 ms |
| sum f32 all | 3.21 ms | 1.31 ms | 48.60 ms | 39.60 ms | 31.50 ms |
| sum i32 axis 0 | 11.70 ms | 3.12 ms | 22.00 ms | 20.40 ms | 37.10 ms |
| sum i32 all | 4.39 ms | 1.36 ms | 18.60 ms | 14.20 ms | 29.60 ms |
| std f32 axis 0 | 11.06 ms | 14.44 ms | 61.70 ms | 72.90 ms | 41.90 ms |
| std f32 all | 10.74 ms | 34.81 ms | 82.40 ms | 74.00 ms | 39.10 ms |

</details>

### Rust

![Rust: elementwise and reductions](plots/rust_op.png)

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| negate f32 | 11.00 ms | 11.20 ms | 49.50 ms |
| negate i32 | 10.90 ms | 11.10 ms | 45.00 ms |
| add f32 | 16.10 ms | 16.40 ms | 99.00 ms |
| add i32 | 16.00 ms | 16.30 ms | 95.00 ms |
| sum f32 axis 0 | 7.10 ms | 5.40 ms | 49.00 ms |
| sum f32 axis 1 | 5.20 ms | 4.90 ms | 48.00 ms |
| sum i32 all | 6.80 ms | 2.80 ms | 19.00 ms |

</details>

---

## Operation chains

Every jix operation returns a lazy view; the whole chain is encoded in the type and runs in a
single pass when output is requested. NumPy evaluates eagerly and allocates a full intermediate per
step, so the gap should grow with the length of the chain.

![Operation chains](plots/chain.png)

NumPy's cost is linear in chain length because it makes one full pass per operation. jix's is
nearly flat: read once, apply the fused chain in registers, write once. Since bars are relative to
NumPy and NumPy is the one growing, jix's bars *fall* as the chain gets longer - that descent is
the result. A single chain length would be a number with no mechanism behind it.

`f32` only here; the integer chain behaves the same way and adds nothing but width.

The `exp/log` case is the counter-example, and it is on the plot on purpose.
`(exp(a) * 0.5 + 1).log()` is dominated by transcendental math rather than memory traffic, so
NumPy's vectorized libm decides the outcome and the intermediates jix saves are noise beside it.
Chains win when the operations are cheap and the array is large; they do not when one expensive
kernel dominates.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix |
|---|---|---|---|
| 1 op | 2.90 ms | 2.90 ms | 50.00 ms |
| 2 ops | 6.10 ms | 3.40 ms | 51.00 ms |
| 4 ops | 12.80 ms | 4.60 ms | 53.00 ms |
| 8 ops | 26.00 ms | 7.10 ms | 57.00 ms |
| exp/log | 80.90 ms | 90.90 ms | 138.00 ms |

</details>

### Peak memory

![Operation chains: peak memory](plots/chain_memory.png)

Peak RSS of a fresh subprocess running the same chain, so NumPy's C-level allocations are included.
jix is flat at input plus output whatever the chain does in between, and the compact arm is flat
and lower still because the input is never held uncompressed. Memory has no noise floor, which
makes this the cleanest evidence on the page.

<details>
<summary>Absolute numbers</summary>

| case | numpy | jix-plain | jix |
|---|---|---|---|
| 1 op | 218 MB | 215 MB | 135 MB |
| 2 ops | 322 MB | 215 MB | 135 MB |
| 4 ops | 428 MB | 215 MB | 135 MB |
| 8 ops | 430 MB | 215 MB | 135 MB |

</details>

### Rust

For pure elementwise chains in Rust, plain iterators already fuse for free and hand-written
iterator code will match jix. The comparison worth making is the other one - chains that mix
elementwise work with reductions, broadcasts and axis permutations, which are awkward to write as
iterators and which `ndarray` materializes at every step.

![Rust: operation chains](plots/rust_chain.png)

<details>
<summary>Absolute numbers</summary>

| case | ndarray | jix-plain | jix |
|---|---|---|---|
| 1 op | 11.00 ms | 11.20 ms | 50.00 ms |
| 2 ops | 22.40 ms | 12.10 ms | 52.00 ms |
| 4 ops | 45.10 ms | 13.80 ms | 55.00 ms |
| 8 ops | 90.30 ms | 17.20 ms | 60.00 ms |
| normalize axis 0 | 96.00 ms | 38.00 ms | 88.00 ms |
| normalize axis 1 | 104.00 ms | 41.00 ms | 92.00 ms |

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
| 3-D rotate [1,2,0] f32 | 88.00 ms | 12.00 ms |
| 3-D rotate [1,2,0] i32 | 86.00 ms | 11.80 ms |
| 3-D reverse [2,1,0] f32 | 12.00 ms | 12.20 ms |
| 2-D transpose f32 | 2.00 ms | 1.90 ms |

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
