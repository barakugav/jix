# Findings

Investigation results that shape the benchmark design. Facts about other libraries, established
by reading their source or by direct introspection - not by benchmarking.

## ndarray has no axis sorting, only a 4-way layout classifier

`ndarray-0.17.2/src/zip/mod.rs::array_layout`:

```rust
if is_layout_c(dim, strides) { Layout::c() }
else if n > 1 && is_layout_f(dim, strides) { Layout::f() }
else if n > 1 {
    if dim[0] > 1 && strides[0] == 1 { Layout::fpref() }
    else if dim[n-1] > 1 && strides[n-1] == 1 { Layout::cpref() }
    else { Layout::none() }
} else { Layout::none() }
```

That is the whole of it. C, F, "first axis is contiguous", "last axis is contiguous", or nothing.
On `Layout::none()` the `Zip` machinery iterates in logical index order regardless of what the
strides actually are.

jix sorts *every* axis by descending stride (`jix/src/storage/plain.rs:160`), which is a full
permutation rather than a four-way choice.

**Consequence for the benchmark.** jix can only win where the stride-sorted permutation differs
from logical order *and* ndarray fails to classify the array. That rules out the case currently
benchmarked:

- **2-D transpose does not qualify.** A transposed contiguous array is exactly F-layout, so
  `array_layout` returns `Layout::f()` and ndarray handles it properly. On top of that,
  `ArrayBase::map` short-circuits through `as_slice_memory_order()` whenever the array is
  contiguous in memory order, which a transpose is. `op1.rs::bench_op1_plain_transposed` is
  therefore measuring nothing interesting on the ndarray side.
- **Strided 2-D slices do not qualify either.** `s![..;2, ..;2]` gives strides `(2B, 2)`, which is
  `Layout::none()`, but logical order already walks it in descending-stride order, so jix's sort
  produces the same permutation. No difference to measure.
- **3-D with a rotation does qualify.** Take `[A,B,C]` C-contiguous, strides `(B*C, C, 1)`, and
  permute axes to `[1,2,0]`. Shape becomes `[B,C,A]` with strides `(C, 1, B*C)`:
  - not C, not F
  - `strides[0] == C != 1`, so not `fpref`
  - `strides[n-1] == B*C != 1`, so not `cpref`
  - therefore `Layout::none()`

  ndarray then iterates logically, making the **innermost** loop the one with stride `B*C` - a
  full row jump per element. jix sorts to `A, B, C` and walks memory in order.

A reversal permutation (`[2,1,0]`) does *not* qualify - it is exactly F-layout. The distinction is
rotation versus reversal, and it needs three or more dimensions to exist at all.

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

## Chains: fusion pays off in Rust, and is flat in Python

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
