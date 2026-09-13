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
