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

np.asarray((b + -3.0)[:])                          # correct
np.asarray((((b * 2.0) + 1.0) * 0.5 + -3.0)[:])    # WRONG
np.asarray((((b * 2.0) + 1.0) * 0.5 - 3.0)[:])     # correct
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

## jix fuses chains; the Python scalar-operand path costs 2.6x per element-op

Rust fuses exactly as claimed. `[130000, 200] f32`, alternating `* 2.0` / `+ 1.0`, Criterion at
`--fast`:

| steps | ndarray | jix-plain |
|---|---|---|
| 1 | 2.6 ms | 2.6 ms |
| 8 | 20.9 ms | **4.0 ms** |

ndarray is linear because it allocates per step; jix is flat, 5.2x faster at eight steps.

Python fuses too - eight steps build one nested view materialized once - but a chain of eight steps
is 208M element-ops whether or not it is fused, so the number that matters is the cost per
element-op, not whether the total is flat. Normalized, on an Apple M3 Pro:

| 8-step chain | numpy | jix-plain | ratio | jix ns per element-op |
|---|---|---|---|---|
| unary `-a` | 17.8 ms | **15.2 ms** | **0.85x** | 0.073 |
| `a * 2.0` / `a + 1.0` | 16.7 ms | 39.2 ms | 2.34x | 0.189 |

**The unary chain already wins**, so fusion pays off in Python as well. The scalar-operand path
costs 2.6x more per element-op than the unary path, and that difference is the whole gap. At
0.073 ns per element-op the unary kernel runs about 3.3 element-ops per cycle - vectorized. At
0.189 it runs about 1.3, which is what a stride-0 broadcast operand fetched inside the loop would
cost rather than one hoisted out of it.

Bringing `Mul<X, Scalar>` to the throughput `Neg<X>` already reaches would put the eight-step chain
near 15 ms against numpy's 16.7 ms, widening with chain length.

An earlier round of this measurement also showed a `Cast<Scalar>` in the tree for Python float
constants. That has since been fixed - `a * 2.0` now builds `Mul<Plain, Scalar>` directly and
Python floats cost the same as `np.float32` constants - which took about 19% off the chain
(48.6 ms to 39.4 ms at eight steps). The per-element-op gap above is what remains underneath it.

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
