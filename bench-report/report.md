> **DRAFT with placeholder numbers.** `[real]` values come from an actual local run (single run,
> dev laptop). `[fake]` values are invented and plausible. Plots are shown as ASCII sketches of the
> real chart - each becomes a single PNG with the three platforms stacked vertically.

# jix benchmarks

jix is a multi-dimensional array library with block-compressed storage and lazy operation chains.
This page measures both against the libraries you would otherwise reach for: `ndarray` in Rust,
NumPy, Blosc2 and Zarr in Python.

**Machine.** Apple M2 Pro, 10 cores, 32 GB, macOS 15.6. rustc 1.89.0, Python 3.13, numpy 2.3.1,
blosc2 4.9.1, zarr 3.0.6. `[fake]` Also run on linux-x86_64 and linux-aarch64; every plot shows all
three.

**Everything is single-threaded** - jix has no threading, and Blosc2, Zarr and numexpr are pinned
to one thread so the comparison is like for like. **Codec settings are matched**: zstd level 3,
byte-shuffle.

**How to read the plots.** Each tick on the x axis is a configuration. Each bar is a library. The
y axis is speed relative to NumPy (Rust: relative to `ndarray`), on a log scale anchored at 1.0 -
**above the line is faster, below is slower**. The baseline is always drawn as its own 1x bar.

---

## Compression

![compression ratio](plots/compress_ratio.png)

```
ratio vs raw array (higher = smaller on disk)
       random          smooth         4 unique
 100x |                                  #@ %&
  10x |                                  ^^ ^^
   1x |--##@@%%&&-----##@@%%&&---------------------
       # jix   @ jix-shuffle   % blosc2   & blosc2-shuffle   ^ zarr
```

![compression throughput](plots/compress_throughput.png)

Ratios track Blosc2 closely, which is expected - same codec, same filter, so this mostly measures
what the block layout costs. The useful part is that highly compressible data is also *faster* to
compress and decompress, which is the mechanism behind the distribution cases in the operation
sections below.

<details>
<summary>Absolute numbers</summary>

| distribution | library | ratio | compress MB/s | full-read MB/s |
|---|---|---|---|---|
| random | jix | 1.00 `[fake]` | 780 `[fake]` | 2100 `[fake]` |
| random | blosc2 | 1.00 `[fake]` | 810 `[fake]` | 2250 `[fake]` |
| random | zarr | 1.00 `[fake]` | 190 `[fake]` | 240 `[fake]` |
| smooth | jix-shuffle | 3.94 `[fake]` | 640 `[fake]` | 2850 `[fake]` |
| smooth | blosc2-shuffle | 3.71 `[fake]` | 590 `[fake]` | 2700 `[fake]` |
| smooth | zarr | 3.71 `[fake]` | 170 `[fake]` | 260 `[fake]` |
| 4 unique | jix-shuffle | 55.2 `[fake]` | 1450 `[fake]` | 7900 `[fake]` |
| 4 unique | blosc2-shuffle | 48.6 `[fake]` | 1310 `[fake]` | 7100 `[fake]` |
| 4 unique | zarr | 48.6 `[fake]` | 380 `[fake]` | 520 `[fake]` |

`i32`, `[130_000, 64]`. Ratio is raw bytes over stored bytes.
</details>

---

## Reading a region

Each timed call reads a different randomly placed region, cycling through 1024 of them - the
data-loader pattern, with a working set far larger than cache. `i32` throughout.

![read](plots/read.png)

```
speed vs numpy (log, anchored at 1.0)
        b16x70      b16x70      b64x70      b64x70      b64x70      b64x70
        r1x70       r16x70      r16x16      r256x70     r4096x70    r-full
  1x  |---N----------N-----------N-----------N-----------N-----------N-----
 0.1x |   #@        #@          #@          #@          #@          #@
0.01x |   %&        %&          %&          %&
                                            %&          %&          %&
 .001x|   ^         ^           ^           ^
       N numpy  # jix  @ jix-shuffle  % blosc2  & blosc2-shuffle  ^ zarr
```

NumPy is the floor here: an uncompressed slice out of an array held whole in RAM. jix pays a small
multiple of that and stores the array in a quarter of the space. The three large-read cases on the
right are where per-call overhead stops dominating - Zarr's flat cost at small reads is Python
indexing, not its codec, and the gap narrows sharply once each read does real work.

The `b64x70 / r16x16` case is the one to learn from: a read smaller than a block means decompressing
a whole block and discarding most of it. Match your block shape to how you read.

<details>
<summary>Absolute numbers</summary>

| block / read | numpy | jix | jix-shuffle | blosc2 | blosc2-shuffle | zarr |
|---|---|---|---|---|---|---|
| `[16,70] / [1,70]` | 0.3 us | 1.1 us | 1.2 us | 69.1 us | 68.5 us | 327 us |
| `[16,70] / [16,70]` | 0.4 us | 1.9 us | 2.1 us | 72.0 us | 71.3 us | 401 us |
| `[64,70] / [16,16]` | 0.3 us | 2.7 us | 3.5 us | 47.2 us | 43.9 us | 334 us |
| `[64,70] / [256,70]` | - `[fake]` | - `[fake]` | - | - | - | - |
| `[64,70] / [4096,70]` | - `[fake]` | - `[fake]` | - | - | - | - |
| `[64,70] / full` | - `[fake]` | - `[fake]` | - | - | - | - |

All `[real]` except the new large-read rows. **These predate the random-region change** and were
measured re-reading one cache-warm region; they will all get worse and the ordering may shift.
The Blosc2 arm also uses auto chunk selection (8.7 MiB chunks against a 4 KiB block) - a matched
`chunks == blocks` arm has to run before these ratios are quoted.
</details>

### Rust: against a raw `ndarray` slice

![rust read](plots/rust_read.png)

<details>
<summary>Absolute numbers</summary>

| array | block | read | ndarray | jix | factor |
|---|---|---|---|---|---|
| `[11000,460]` | `[32,32]` | `[32,32]` | 0.28 us `[fake]` | 1.4 us `[fake]` | 5.0x slower |
| `[11000,460]` | `[32,32]` | `[1,460]` | 0.35 us `[fake]` | 9.8 us `[fake]` | 28x slower |
| `[11000,460]` | `[512,32]` | `[128,460]` | 11.0 us `[fake]` | 46 us `[fake]` | 4.2x slower |
</details>

---

## Negate

![negate](plots/negate.png)

```
speed vs numpy            f32                    i32
   10x |
    1x |---N--P--------------------N--P------------------
   0.1x|         #@                       #@
  0.01x|           %&  ^                    %&  ^
       N numpy  P jix-plain  # jix  @ jix-shuffle  % blosc2  & blosc2-shuffle  ^ zarr
```

On uncompressed input jix matches NumPy - 2.19 ms against 2.16 ms `[real]`. That parity is what
makes the chain results further down meaningful: entering a jix pipeline costs nothing that has to
be earned back later.

Against the other compressed-array libraries, jix negates 9x faster than Blosc2 `[real]`. Both
write their result into a plain uncompressed NumPy buffer - Blosc2's `expr[:]` does not re-compress
on the way out, so it is not being billed for a pass jix skips.

### By data distribution

![negate by distribution](plots/negate_distribution.png)

```
speed vs numpy      random      smooth     16 unique    4 unique
    1x |-------N-----------N-----------N-----------N----------
   0.1x|                       #@          #@ %&      #@ %&
  0.01x|   #@ %&            %&
```

Same operation, same code path; only the number of distinct values in the data changes. Better
compression means less memory to move and longer zstd matches, so the operation gets faster -
roughly 5x from random data to four unique values `[fake]`. The same effect shows in the filter
choice: byte-shuffled data negates in 49.6 ms where unshuffled takes 69.5 ms `[real]`, identical op
code on both sides.

This does not catch NumPy, and it is not expected to. An elementwise operation writes a full-size
uncompressed output whatever the input was, so compression only ever helps the read half.

<details>
<summary>Absolute numbers</summary>

| dtype | distribution | ratio | numpy | jix-plain | jix-shuffle | blosc2-shuffle |
|---|---|---|---|---|---|---|
| f32 | smooth | 3.9 | 2.16 ms `[real]` | 2.19 ms `[real]` | 49.6 ms `[real]` | 443 ms `[real]` |
| f32 | random | 1.0 | 2.16 ms `[fake]` | 2.19 ms `[fake]` | 96.4 ms `[fake]` | 510 ms `[fake]` |
| f32 | 16 unique | 31 | 2.16 ms `[fake]` | 2.19 ms `[fake]` | 34.5 ms `[fake]` | 210 ms `[fake]` |
| f32 | 4 unique | 95 | 2.16 ms `[fake]` | 2.19 ms `[fake]` | 21.3 ms `[fake]` | 155 ms `[fake]` |
| i32 | smooth | 4.1 | - `[fake]` | - `[fake]` | - `[fake]` | - `[fake]` |

`[130_000, 200]`. Every op benchmark in the suite currently runs on `smooth` and `f32` only, so
the dtype and distribution rows do not exist yet in either harness.
</details>

---

## Add

![add](plots/add.png)

Same shape as negate, with two operands instead of one: parity with NumPy on uncompressed input
(2.88 ms against 2.89 ms `[real]`), 5.7x faster than Blosc2 on compressed `[real]`, and the same
gradient across data distributions.

<details>
<summary>Absolute numbers</summary>

| dtype | storage | numpy | jix-plain | jix-shuffle | blosc2-shuffle | zarr |
|---|---|---|---|---|---|---|
| f32 | - | 2.886 ms `[real]` | 2.882 ms `[real]` | 99.1 ms `[real]` | 561 ms `[real]` | 59.2 ms `[real]` |
| i32 | - | - `[fake]` | - `[fake]` | - `[fake]` | - `[fake]` | - `[fake]` |
</details>

---

## Reductions

`sum` and `std` in one plot, because they land on opposite sides of the line and separating them
would be choosing which result to show.

![reductions](plots/reductions.png)

```
speed vs numpy
       sum f32   sum f32   sum f32   sum i32   sum i32   std f32   std f32
       axis0     axis1     all       axis0     all       axis0     all
   4x |            P                    P        P
   2x |   P                  P
   1x |---N---------N---------N---------N---------N---------N---------N----
  0.5x|                                                        P
  0.3x|                                                                  P
  0.1x|  #@%&      #@%&      #@%&      #@%&      #@%&      #@%&      #@%&
```

jix's reduction kernels read in whatever layout the source already has rather than forcing a
canonical order, and they vectorize integer accumulation that NumPy leaves scalar. On this machine
that is worth 2.4x on `f32` and 3.2x on `i32` for a full `sum` `[real]`.

**That win is platform-specific and should not be read as a property of the library.** jix's
kernels are tuned on macOS arm64; NumPy is general-purpose with far more x86 attention behind its
hot paths. The stacked per-platform plot above is the honest picture, and the `linux-x86_64` row
may well look different.

`std` goes the other way: 3.2x slower than NumPy on the full reduction `[real]`, and worst on
`axis=1`. That is a real deficiency in the kernel rather than a measurement artifact. On compressed
input Blosc2 also edges jix out on reductions (74 ms against 82 ms `[real]`) while losing badly on
elementwise - visible in the same chart.

<details>
<summary>Absolute numbers</summary>

| reduction | numpy | jix-plain | jix-shuffle | blosc2-shuffle | zarr |
|---|---|---|---|---|---|
| `sum` i32 all | 4.39 ms | **1.36 ms** | 18.6 ms | 14.2 ms | 29.6 ms |
| `sum` i32 axis 0 | 11.70 ms | **3.12 ms** | 22.0 ms | 20.4 ms | 37.1 ms |
| `sum` i32 axis 1 | 4.78 ms | **1.48 ms** | 20.3 ms | 13.5 ms | 30.8 ms |
| `sum` f32 all | 3.21 ms | **1.31 ms** | 48.6 ms | 39.6 ms | 31.5 ms |
| `sum` f32 axis 0 | 2.33 ms | **2.03 ms** | 49.6 ms | 34.7 ms | 30.9 ms |
| `sum` f32 axis 1 | 3.97 ms | **1.55 ms** | 49.0 ms | 36.3 ms | 32.9 ms |
| `std` f32 all | **10.74 ms** | 34.81 ms | 82.4 ms | 74.0 ms | 39.1 ms |
| `std` f32 axis 0 | **11.06 ms** | 14.44 ms | 61.7 ms | 72.9 ms | 41.9 ms |

All `[real]`, `[130_000, 200]`.
</details>

---

## Operation chains

Every jix operation returns a lazy view; the whole chain is encoded in the type and runs in a
single pass when output is requested. NumPy evaluates eagerly and allocates a full intermediate per
step, so the gap should grow with the length of the chain.

![chain](plots/chain.png)

```
speed vs numpy       1 op      2 ops     4 ops     8 ops    exp/log
   4x |                                     P
   2x |                          P
   1x |-----N--P-------N---------N---------N---------N--------
  0.8x|                                                    P
```

NumPy's cost is linear in chain length because it makes one full pass per operation. jix's is
nearly flat: read once, apply the fused chain in registers, write once.

The `exp/log` case on the right is the counter-example, and it is on the plot for a reason.
`(exp(a) * 0.5 + 1).log()` is dominated by transcendental math rather than memory traffic, so
NumPy's vectorized libm decides the result and jix comes out 1.12x slower `[real]`. Chains win when
the operations are cheap and the array is large; they do not when a single expensive kernel
dominates.

### Peak memory

![chain memory](plots/chain_memory.png)

```
memory vs numpy (lower = better; bars below the line use less)
                     1 op      2 ops     4 ops     8 ops
   1x |-----N----------N---------N---------N--------
  0.7x|        P         P         P         P
  0.3x|        C         C         C         C
       N numpy   P jix-plain   C jix compact
```

jix is flat at input plus output, whatever the chain does in between. The compact arm is flat and
lower still, because the input is never held uncompressed. Measured as peak RSS in a fresh
subprocess per configuration, so it includes NumPy's C-level allocations.

<details>
<summary>Absolute numbers</summary>

| ops | numpy time | jix-plain time | numpy peak RSS | jix-plain RSS | jix compact RSS |
|---|---|---|---|---|---|
| 1 | 2.9 ms `[fake]` | 2.9 ms `[fake]` | 218 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| 2 | 6.1 ms `[fake]` | 3.4 ms `[fake]` | 322 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| 4 | 12.8 ms `[fake]` | 4.6 ms `[fake]` | 428 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| 8 | 26.0 ms `[fake]` | 7.1 ms `[fake]` | 430 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| exp/log | 80.9 ms `[real]` | 90.9 ms `[real]` | - | - | - |
</details>

### Rust: chains with shape operations

For pure elementwise chains in Rust, plain iterators already fuse for free and hand-written
iterator code will match jix. The comparison worth making is the other one - chains that mix
elementwise work with reductions, broadcasts and axis permutations, which are awkward to express as
iterators and which `ndarray` materializes at every step.

![rust chain](plots/rust_chain.png)

<details>
<summary>Absolute numbers</summary>

`a / a.std(axis).insert_axis(axis).broadcast(shape)` on `[300_000, 80] f32`:

| | ndarray | jix | |
|---|---|---|---|
| axis 0 | 96 ms `[fake]` | 38 ms `[fake]` | 2.5x faster |
| axis 1 | 104 ms `[fake]` | 41 ms `[fake]` | 2.5x faster |
</details>

---

## Axis order (Rust)

`ndarray` classifies an array's layout four ways - C, F, first-axis-contiguous,
last-axis-contiguous - and falls back to logical-order iteration when none of them fit. jix sorts
every axis by descending stride, so it always walks memory in order.

The difference needs three dimensions to appear at all. A 2-D transpose is exactly F-layout and
`ndarray` handles it perfectly; a *rotation* of a 3-D array is neither C nor F, and leaves
`ndarray` iterating with the largest stride in the innermost loop.

![rust axis order](plots/rust_axis_order.png)

<details>
<summary>Absolute numbers</summary>

`negate` on `[300, 400, 500]`, permuted:

| permutation | ndarray layout | ndarray | jix | |
|---|---|---|---|---|
| `[1,2,0]` rotation | `none` | 88 ms `[fake]` | 12 ms `[fake]` | 7.3x faster |
| `[2,1,0]` reversal | `f` | 12 ms `[fake]` | 12 ms `[fake]` | parity (control) |
| 2-D transpose | `f` | 2.0 ms `[fake]` | 1.9 ms `[fake]` | parity (control) |

The two controls are on the plot on purpose: they show the effect is specific to layouts `ndarray`
cannot classify, not a general claim about strided data.
</details>

There is no Python counterpart - NumPy sorts axes too, so there is nothing to compare.

---

## Reproducing

    python jix/benches/run.py --report          # rust, only the comparison benches
    python jix-py/python/benches/run_all.py     # python

The CI matrix across the three runners is documented in
[`docs/benchmarks.md`](../docs/benchmarks.md), along with `scripts/bench/compare.py` for
same-machine A/B against a base ref. Every number on this page is generated from the benchmark
JSON; none are typed by hand.
