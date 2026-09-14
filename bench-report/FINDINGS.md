# Findings

Investigation results that shape the benchmark design. Facts about other libraries, established
by reading their source or by direct introspection - not by benchmarking.

## ndarray has no axis sorting - but reaching that fact is harder than it looks

`ndarray-0.17.2/src/zip/mod.rs::array_layout` is a four-way classifier - C, F,
first-axis-contiguous, last-axis-contiguous, or nothing - with no general axis sort anywhere. jix
sorts every axis by descending stride (`jix/src/storage/plain.rs:160`). So jix should win whenever
the stride-sorted order differs from logical order.

**Measured, and it did not.** The first version of this benchmark negated permuted views of a
contiguous array and reported 1.08x to 1.39x - jix slightly *slower* - identically across the
treatment and both controls. Treatment and controls agreeing is the tell that the benchmark is
measuring nothing.

Two reasons, and the second is the one that matters:

1. `array_layout` governs `Zip`, which is what binary operations go through. `ArrayBase::map`, which
   a unary operation uses, has its own short-circuit: `as_slice_memory_order()`, falling back to
   `self.iter()` in logical order.
2. `as_slice_memory_order` requires `is_contiguous`, and **that check sorts the strides first**
   (`dimension_trait.rs:295` calls `_fastest_varying_stride_order`). A pure permutation of a
   contiguous array is therefore still contiguous *in memory order*, so the flat fast path is taken
   for every permutation, rotation and reversal alike. There is nothing to measure.

To reach the logical-order iterator the view has to be genuinely non-contiguous, so the benchmark
now slices the last axis before permuting. With `[1,2,0]` on a sliced view the largest stride ends
up on the innermost logical axis and `ndarray` jumps a full plane per element, while jix sorts and
walks memory in order. Controls: a permutation that keeps logical order near memory order, and a
contiguous rotation that still reaches the fast path.

**This is not yet confirmed**, and the section is withheld from the published report. The first
full run (94f72a73) predates the fix, so its axis-order numbers come from the version that measured
nothing. `build_report.py` now warns when a section's cases are only partly present, which is what
that looks like: surviving benchmark ids get drawn under the new labels, which is worse than an
empty plot. Restore the section once a run includes the fixed benchmark.

## Compression ratio does not predict decode speed; the kind of redundancy does

Two report results looked wrong and turned out to be the same thing: `smooth` compressed better
than either unique-value set, and `negate` on compact storage got *slower* as the data compressed
better. From `debug_whole_array_read.py`, whole-array decode of `[130000, 200] i32`:

| distribution | ratio | jix MB/s | jix-noshuffle MB/s |
|---|---|---|---|
| random | 1.0x | **20216** | **45246** |
| smooth (fixed generator) | 34.5x | 5700 | 6591 |
| 16 unique | 7.9x | 3254 | 1034 |
| 4 unique | 12.2x | **1864** | 777 |

Decode throughput runs almost opposite to the ratio. The reason is *how* zstd achieves each ratio:

- **random** is incompressible, so zstd stores raw blocks and decoding is a memcpy - the fastest
  case on the page, at 1.0x compression.
- **smooth** compresses through long matches, which decode cheaply.
- **the unique-value sets** compress through entropy coding over a small alphabet, and entropy
  decoding costs real work per output byte. 4 unique is slower than 16 unique *despite* a better
  ratio.

blosc2 shows the same ordering, so this is a property of zstd rather than of jix. The practical
consequence for anyone choosing settings: a compression ratio tells you about storage, not about
read speed, and the two can point in opposite directions.

Byte-shuffle is worth noting separately. On random data it costs 2.2x on decode (45 GB/s to
20 GB/s) for no ratio benefit at all; on 16-unique data it *gains* 3.1x (1034 to 3254 MB/s) by
making three of four byte planes constant.

### The `smooth` generator was size-dependent, and has been fixed

It built its sine with `linspace(0, 8*pi, n)` - four periods across the array *however large the
array was*. At 26M elements consecutive values were almost always equal after rounding, giving a
769x ratio: runs, not the moderate redundancy the profile is supposed to represent, and a
compressibility that changed with every array size. It now uses a fixed period of 1000 elements,
which gives 34.5x and 2249 distinct values at any size.

## blosc2's decompression unit is the block, but chunk size still costs

Measured directly (one-off, 130000x70 i32, 16x70 block, reading 16x70 regions, single-threaded):

| chunk | chunk size | us per read |
|---|---|---|
| `(16, 70)` = the block | 4 KiB | 33.6 |
| `(1024, 70)` | 280 KiB | 34.1 |
| `(16384, 70)` | 4.4 MiB | 53.2 |
| `(32512, 70)` = auto | 8.7 MiB | 75.9 |

If the chunk were the decompression unit, the 8.7 MiB case would cost roughly two thousand times
the 4 KiB case rather than 2.3x. So the block is the unit, as expected. But the cost is flat only
to ~280 KiB and then climbs, which looks like the price of walking a large chunk's block-offset
table on every read. blosc2's own auto-chunking is the worst case in the table.

Consequence: the `blosc2-chunked` arm (`chunks == blocks`) is worth carrying, but as "blosc2 given
its best configuration" rather than as a correction to an unfair comparison. Expect it to move the
read gap by roughly 2x.

## blosc2 auto-selects chunks two thousand times larger than the block

Measured by introspection, not benchmarking:

| requested shape | requested blocks | actual chunks | actual blocks |
|---|---|---|---|
| `(130000, 70)` i32 | `(16, 70)` | `(32512, 70)` = **8.7 MiB** | `(16, 70)` = 4 KiB |
| `(130000, 70)` i32 | `(64, 70)` | `(32500, 70)` = **8.7 MiB** | `(52, 70)` = 14 KiB |
| `(130000, 200)` i32 | auto | `(16250, 200)` = 12.4 MiB | `(130, 200)` = 102 KiB |

Two separate problems:

1. **The decompression unit is unclear.** blosc2 can decompress individual blocks inside a chunk,
   so a small read *should* cost 4 KiB, not 8.7 MiB. Whether the `__getitem__` path actually takes
   that route is exactly what the current read numbers cannot distinguish. A `chunks == blocks`
   arm settles it: forcing both is accepted and honored exactly (8125 chunks for `(16,70)`).
2. **Requested block shapes are silently adjusted.** Asking for `(64, 70)` yields `(52, 70)`. The
   benchmark case labelled `b64x70` has never compared equal block shapes between jix and blosc2.
   Forcing `chunks` as well makes blosc2 honor `(64, 70)` exactly.

## blosc2 mis-evaluates a negative literal added to a nested expression

Found by the cross-check test that runs every library's operations against NumPy on the same data.

```python
a = np.arange(12, dtype=np.float32).reshape(3, 4) / 4.0
b = blosc2.asarray(a)

np.asarray((b + -3.0)[:])  # correct
np.asarray((((b * 2.0) + 1.0) * 0.5 + -3.0)[:])  # WRONG
np.asarray((((b * 2.0) + 1.0) * 0.5 - 3.0)[:])  # correct
```

For input `[0, 0.25, 0.5, 0.75]` the third line gives `[-3, -1.25, -3, -1.1875]` where the right
answer is `[-2.5, -2.25, -2, -1.75]`. blosc2 builds the expression as a string - the failing one is
`'((((o0 * 2.0) + 1.0) * 0.5) + -3.0)'` - and something in that path mishandles the negative
literal once the left operand is itself an expression. A bare array plus a negative literal is
fine, so it is specific to the nested form.

Worth reporting upstream. Until then the benchmark chain ends with `- 3.0` rather than `+ -3.0`,
which is the same arithmetic and which every library agrees on.

This is the whole argument for the cross-check test: a benchmark where one arm silently computes
something else is worse than no benchmark, and nothing about the timings would have revealed it.

## blosc2's `expr[:]` does not re-compress

`blosc2/lazyexpr.py`: `LazyExpr.__getitem__` passes `_getitem=True`. The `blosc2.asarray(result)`
re-compression branch in `compute()` is guarded by `"_getitem" not in kwargs`, and `fast_eval`
allocates `np.empty(shape, dtype)` when `getitem` is set versus `blosc2.empty(...)` when it is not.
So `(-a)[:]` decompresses straight into a plain numpy buffer, exactly as jix does. The elementwise
arm is a fair comparison as written.

`expr.compute()` is the re-compressing path. A "compressed in, compressed out" comparison would
use that, and would be a different benchmark.

## blosc2 evaluates lazy expressions through numexpr, which has its own threads

`blosc2.set_nthreads(1)` does not touch numexpr's pool, which defaults to one thread per core
(12 here). Every elementwise blosc2 number measured before this was fixed was multi-threaded
against single-threaded jix and numpy. Fixed in `array_impls.py` with
`numexpr.set_num_threads(NTHREADS)`.

## Chains: measured on CI at full fidelity

Run 34788352028, both runners, `[130000, 200] f32` x2, default read region. Ratios are against the
language's baseline, lower is better.

| steps | Python numpy=1 (x86) | (aarch64) | Rust ndarray=1 (x86) | (aarch64) |
|---|---|---|---|---|
| 1 | 0.99x | 1.23x | 1.04x | 1.01x |
| 4 | 0.40x | 0.66x | 0.52x | 0.72x |
| 8 | 0.25x | 0.50x | 0.29x | 0.49x |
| 16 | **0.19x** | 0.40x | **0.17x** | 0.39x |
| 32 | **0.15x** | 0.36x | - | - |
| exp/log | 2.45x | 1.01x | - | - |

**6.7x faster than numpy at 32 steps on x86_64**, with no tuning - the default read region. The
Intel Xeon 8573C has far less memory bandwidth per core than the M3 Pro the earlier local numbers
came from, so numpy's per-step DRAM traffic costs much more there and jix's cache-resident regions
win outright. The mechanism predicted this before the run; see the section below on read regions.

`exp/log` is the counter-example and it is much sharper on x86 (2.45x slower) than on aarch64
(1.01x): when one transcendental kernel decides the result, saving memory traffic buys nothing.

## Earlier local measurements: fusion pays off in Rust, and is flat in Python

Measured on an Apple M3 Pro, `[130000, 200] f32` x2, chain of array-operand steps starting from
`a*a` (see `array_impls.CHAIN_STEPS`). Criterion at `--fast` for Rust; best of five warm rounds for
Python. `ns/op` is per element per step - 26M elements x N steps - which is the number that matters,
since a chain of N steps is N x 26M element-ops whether or not it is fused.

| steps | ndarray | Rust jix | ratio | ns/op | numpy | Python jix | ratio | ns/op |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.4 ms | 2.3 ms | 0.99x | 0.090 | 2.2 ms | 2.3 ms | 1.02x | 0.088 |
| 2 | 5.0 ms | 3.2 ms | 0.64x | 0.061 | 5.2 ms | 5.4 ms | 1.04x | 0.104 |
| 4 | 10.6 ms | 4.5 ms | 0.42x | 0.043 | 11.1 ms | 10.6 ms | 0.96x | 0.102 |
| 8 | 21.0 ms | 6.9 ms | 0.33x | 0.033 | 22.0 ms | 21.3 ms | 0.97x | 0.102 |
| 16 | 43.5 ms | **11.5 ms** | **0.26x** | 0.028 | 46.3 ms | 42.5 ms | 0.92x | 0.102 |

**Rust is the claim working.** The per-element-op cost *falls* with chain length, 0.090 down to
0.028 ns - at one step the pass is memory-bound, and every extra step amortizes that fixed traffic
over more arithmetic. At sixteen steps jix is 3.8x faster than ndarray and runs about 9 element-ops
per cycle.

**Python is flat at 0.102 ns per element-op**, so extra steps buy nothing and the result sits at
parity with numpy throughout. Flatness is the signature: it is what per-step passes cost, not what
a fused loop costs. Parity with numpy is a coincidence of this machine - numpy is memory-bound here
at roughly the same rate that Python jix is bound by whatever it is bound by.

The likely cause is type erasure at the binding boundary. `jix-py`'s `Array` is
`pub struct Array { arr: ArrayAny }`, and `ArrayAny` is `Arc<dyn ArrayStorage>` - so each Python-level
operation wraps a trait object and the compiler cannot inline one step into the next. In Rust the
whole chain is one static type and collapses into a single loop. The 3.7x gap at sixteen steps
(11.5 ms against 42.5 ms, same arithmetic, same arrays) is consistent with that, though it is a
hypothesis from the type signatures rather than something confirmed from the generated code.

Prediction worth checking on the x86 runner: Python jix should *win* on a machine whose memory
bandwidth is lower relative to its compute, since numpy's cost is traffic and jix's is not.

### Where jix does beat numpy: a cache-sized read region

The flat 0.102 ns per element-op above is not the whole story. jix materializes a lazy chain one
read region at a time, so every intermediate for that region stays in cache while the inputs and
the output stream past it - no compiler fusion required, the block-wise iteration alone gives it.
What decides whether that pays is the size of the read region, and the default is chosen for cache
sizes generally rather than for long elementwise chains.

Sweeping `read_size` on `[130000, 200] f32` x2, arms alternated round by round to cancel drift:

| steps | numpy | jix default | jix `read_size=(12K, 24K)` | speedup |
|---|---|---|---|---|
| 4 | 10.9 ms | 10.8 ms (1.02x) | 12.2 ms | 0.89x |
| 8 | 22.8 ms | 22.3 ms (1.03x) | 20.5 ms | **1.11x** |
| 16 | 51.6 ms | 45.6 ms (1.13x) | 38.5 ms | **1.34x** |
| 32 | 105.1 ms | 92.1 ms (1.22x) | 73.9 ms | **1.42x** |
| 64 | 202.7 ms | 184.7 ms (1.23x) | 146.6 ms | **1.38x** |

Tuning drops jix's cost from 0.111 to 0.088 ns per element-op, and the win grows with chain length
because numpy's per-step cost is DRAM traffic while jix's is an L1 round-trip. 8K-16K and 12K-24K
were the best of a sweep from 1K to 16M; below 4K the per-region overhead takes over (1K-2K is
0.35x) and above 32K the intermediates stop fitting.

The win is modest here because an M3 Pro has a lot of memory bandwidth, so numpy's DRAM penalty is
small. The gap should be wider on a machine with less bandwidth per core - the x86 runner will say.

**The optimal read region is opposite for compact storage.** A region smaller than a block still
decompresses the whole block and discards most of it, and the next region decompresses it again.
The same `(12K, 24K)` that wins on `Plain` takes a compact unary chain from 45 ms to 244 ms.

### No array shape wins with the default read region, because the default is an L2-derived byte budget

Sweeping aspect ratio at a fixed 26M elements - `(26000000,)`, `(2600000, 10)`, `(260000, 100)`,
`(130000, 200)`, `(26000, 1000)`, `(2600, 10000)`, `(260, 100000)`, `(26, 1000000)` - gives
1.07x to 1.09x at 8, 16 and 32 steps. Every shape. Sweeping total size (1M to 64M elements) and
dtype (f32, f64) stays in the same 1.00x to 1.09x band.

That is what should happen. The read region is a *byte* budget, so its size does not depend on the
array's shape at all, and `params.rs` derives it as:

```rust
let read_size_min = max(l1_data / 2, l2 / 16).max(block_size);
let read_size_max = l2 / 2;
```

On an Apple M3 Pro (L1d 128 KB, L2 16 MB) that is `max(64 KB, 1024 KB) .. 8 MB` - the L2 term wins
and the floor lands at 1 MB, **85x larger than the 12-24 KB optimum measured above**. No shape can
move it.

The reason the optimum is well below L1 rather than at it: a chain has roughly four or five live
buffers at any moment - two inputs, the running value, the output. At 12-24 KB each that is
60-120 KB and fits a 128 KB L1; at 64 KB each it is over 300 KB and spills. The budget wants to be
L1 divided by the number of live buffers, not L1 itself.

Worth considering, though it is a library change rather than a benchmark one: the floor is a poor
fit for any CPU with a large L2, and Apple Silicon is the extreme case. **It cannot simply be made
smaller, though** - compact storage wants the opposite. A region smaller than a block decompresses
the whole block and discards most of it, which is why `(12K, 24K)` takes a compact unary chain from
45 ms to 244 ms. A budget that serves both would have to depend on whether the pipeline's leaves
are compressed.

### Compact beats numpy too, but only on very long chains

Unary chain (one leaf reference, so the array is decompressed once), default read region:

| steps | numpy | jix compact (14.3x ratio) | speedup |
|---|---|---|---|
| 8 | 17.3 ms | 58.4 ms | 0.30x |
| 32 | 76.7 ms | 104.1 ms | 0.74x |
| 64 | 162.5 ms | 165.5 ms | 0.98x |
| 128 | 340.0 ms | 287.3 ms | **1.18x** |

Decompression is a fixed ~45 ms; after that jix grows at ~1.9 ms per step against numpy's ~2.6 ms,
so the crossover is near 64 steps. The arithmetic predicts 64 and the measurement agrees.

**Reusing an operand multiplies decompression.** A binary chain that references `a` and `b` at every
step decompresses their blocks once per reference - a 32-step binary chain on compact cost 1430 ms,
about 29 times a single full decompression. There is no reuse of a decompressed block across
references within one pipeline. That is worth knowing independently of the report.

### Correction: an earlier version of this measurement was wrong

The first Rust chain benchmark chained `.map(|x| x * 2.0)` closures with literal constants and
reported jix as flat at 2.6-4.0 ms from one to eight steps, 5.2x faster than ndarray. That result
is withdrawn. An affine chain of scalar closures constant-folds: LLVM inlines all eight and
collapses them to a single multiply-add, so the benchmark measured one operation no matter how many
were written. The numbers above use two array operands, which cannot be folded away.

The same mistake is why the chain now avoids scalar constants entirely - and separately, why it
should: jix has no specialized loop for a 0-stride operand, so a scalar is read through the same
strided loop as a real array. A scalar chain measures that gap rather than whether fusing pays off.
An earlier round of this also found a `Cast<Scalar>` in the tree for Python float constants, since
fixed - `a * 2.0` now builds `Mul<Plain, Scalar>` directly.

## A Compact chain costs one full decompression per operand *reference*

Why the Rust chain gets slower with every step on compact storage, from run 34788352028 (x86_64,
`[130000, 200] f32` x2):

| steps | leaf references | jix compact | delta | per reference |
|---|---|---|---|---|
| 1 | 2 | 117.8 ms | - | - |
| 2 | 3 | 171.7 ms | 53.9 ms | 53.9 ms |
| 4 | 5 | 269.8 ms | 98.1 ms | 49.1 ms |
| 8 | 9 | 467.2 ms | 197.4 ms | 49.4 ms |
| 16 | 17 | 855.9 ms | 388.7 ms | 48.6 ms |

A least-squares fit gives 49.2 ms per reference with a 19.4 ms intercept - and that intercept is
jix-plain's own cost for the same chain (20.7 ms), while 49 ms is what decompressing this array
once costs. So the model is exact:

    compact chain time = (number of operand references) x (one full decompression) + plain chain time

Each time the chain names `a` or `b`, that operand's blocks are decompressed again. A decompressed
block is not reused across references within a single pipeline, so a 16-step chain that mentions an
operand every step decompresses 17 times what it needs twice.

The contrast with `jix-plain` is the confirmation: 20.7 ms to 34.1 ms across the same range, nearly
flat, because re-reading an *uncompressed* operand is just a memory read and costs almost nothing.

Two consequences:

- The benchmark's chain is close to the worst case for compact storage by construction, since every
  step references a source array. A chain of unary steps references its leaf once and does not
  degrade this way - which is why the Compact crossover measurement above uses one.
- A per-pipeline cache of decompressed blocks, keyed by (storage, block index), would collapse the
  17 decompressions to 2 and make the compact chain track the plain one. That is the single largest
  lever on these numbers.

## `normalize` over a long axis re-runs the reduction per output element

`a / a.std(axis).insert_axis(axis).broadcast(shape)` on `[130000, 200] f32`:

| axis | ndarray | jix-plain | jix (compact) |
|---|---|---|---|
| 0 | 6.2 ms | 109.0 ms (17.6x) | 397.0 ms (64x) |
| 1 | 84.1 ms | **48.4 ms** (0.58x) | 121.2 ms (1.44x) |

Not a kernel deficiency - it is what fusing a broadcast reduction costs. The `std` is inside the
lazy pipeline, so reading an output element re-runs the reduction over its whole column. Over axis
0 each column is 130000 long, so the work is O(N*M) instead of O(N+M). Over axis 1 the reduced axis
is 200 long and jix comes out ahead.

The lesson is the one `ops/mod.rs` already gives for reshape: materialize the reduction before
broadcasting it when the reduced axis is long. Worth saying in the report next to this plot, since
a reader will otherwise take the axis-0 bar as a general result.
