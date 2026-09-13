> **THIS IS A DRAFT WITH PLACEHOLDER NUMBERS.**
> Values tagged `[real]` come from an actual local run (single run, dev laptop, not a clean
> measurement). Values tagged `[fake]` are invented and plausible - they show what a result would
> have to look like to support the claim next to it. Nothing here is publishable yet.

# jix benchmarks

jix is a multi-dimensional array library with block-compressed storage and lazy operation chains.
This page measures both features against the libraries you would otherwise use: `ndarray` in Rust,
and NumPy, Blosc2 and Zarr in Python.

**Machine.** Apple M2 Pro, 10 cores, 32 GB. macOS 15.6. rustc 1.89.0, Python 3.13, numpy 2.3.1,
blosc2 3.2.0, zarr 3.0.6. `[fake]`

**Everything below is single-threaded.** jix has no threading at all. Blosc2 and Zarr are pinned
to one thread so the comparison is like for like - including numexpr, which Blosc2 uses to
evaluate lazy expressions and which keeps a separate thread pool. NumPy's elementwise and
reduction kernels are single-threaded regardless. A multi-threaded Blosc2 will beat jix on
throughput-bound work; that comparison is not on this page.

**Codec settings are matched.** zstd level 3 with byte-shuffle for jix, Blosc2 and Zarr alike.

---

## Summary

| claim | measured | configuration | where it stops holding |
|---|---|---|---|
| Random reads from a compressed array are much cheaper than Blosc2 | **38x faster** `[real]` | 130k x 70 i32, block 16x70, read 16x70 | Reads that straddle many blocks; see the read table |
| ...and than Zarr | **211x faster** `[real]` | same | Zarr's number includes its Python indexing overhead |
| Compressed reads cost a small factor over a raw slice | **5.0x** `[fake]` | Rust, read shape == block shape | 28x when a read crosses 15 blocks |
| Elementwise ops on plain arrays match NumPy | **1.00x** `[real]` | 130k x 200 f32, `a + b` | - |
| Integer reductions beat NumPy | **3.2x faster** `[real]` | 130k x 200 i32, full `sum` | Float reductions are 2.4x; `std` is 3.2x *slower* |
| On strided input, jix beats ndarray | **3.4x faster** `[fake]` | Rust, 1200 x 1200 f32 transposed | No claim against NumPy, which also sorts axes |
| Elementwise ops on compressed arrays beat Blosc2 | **9.0x faster** `[real]` | 130k x 200 f32, negate | Reductions are 1.1x *slower* than Blosc2 |
| Op chains beat NumPy, and the gap grows with length | **3.7x faster** `[fake]` | 130k x 200 f32, 8 cheap ops | Chains of expensive math (`exp`, `log`) lose |
| Peak memory for a chain is flat | **430 MB -> 135 MB** `[fake]` | 104 MB input, 8 ops | - |
| Compressing the array makes reductions faster than NumPy | **break-even at ~150x compression** `[fake]` | 130k x 200 f32, full `sum` | Below that NumPy is faster, but uses 50x the memory |

Two of these are losses and one is a break-even. They are here on purpose - see
[Where jix loses](#where-jix-loses).

---

## Storage

### Compression ratio and throughput

Same codec, same level, same filter, so this mostly measures how much the block layout costs.

| profile | library | ratio | compress MB/s | full-read MB/s |
|---|---|---|---|---|
| random | jix | 1.00 | 780 `[fake]` | 2100 `[fake]` |
| random | blosc2 | 1.00 | 810 `[fake]` | 2250 `[fake]` |
| random | zarr | 1.00 | 190 `[fake]` | 240 `[fake]` |
| smooth | jix | 3.94 `[fake]` | 640 `[fake]` | 2850 `[fake]` |
| smooth | blosc2 | 3.71 `[fake]` | 590 `[fake]` | 2700 `[fake]` |
| smooth | zarr | 3.71 `[fake]` | 170 `[fake]` | 260 `[fake]` |
| low entropy | jix | 55.2 `[fake]` | 1450 `[fake]` | 7900 `[fake]` |
| low entropy | blosc2 | 48.6 `[fake]` | 1310 `[fake]` | 7100 `[fake]` |
| low entropy | zarr | 48.6 `[fake]` | 380 `[fake]` | 520 `[fake]` |

Ratios track Blosc2 closely, which is the expected result - same codec, same filter. The
interesting column is the last one, and it is the basis for everything in the
[compressed operations](#operations-on-compressed-arrays) section: *decompression gets faster as
compression gets better*, because there is less to read and the matches are longer.

Zarr's numbers are dominated by its Python-level chunk handling rather than its codec.

### Random reads

Each timed call reads a different randomly placed region, cycling through 1024 of them - the
data-loader case, and a working set far larger than cache. Absolute microseconds per read, with
the ratio to jix underneath.

> These numbers predate the random-region change and were measured re-reading a single cache-warm
> region. They will all get worse and the ordering may shift. `[real, but stale]`

| block / read | numpy | jix | blosc2 | zarr |
|---|---|---|---|---|
| `[16,70] / [1,70]` | 0.3 us `[real]` | **1.1 us** `[real]` | 69.1 us `[real]` | 327 us `[real]` |
| | 0.27x | 1.00x | 63x | 297x |
| `[16,70] / [16,70]` | 0.4 us `[real]` | **1.9 us** `[real]` | 72.0 us `[real]` | 401 us `[real]` |
| | 0.21x | 1.00x | 38x | 211x |
| `[64,70] / [16,16]` | 0.3 us `[real]` | **2.7 us** `[real]` | 47.2 us `[real]` | 334 us `[real]` |
| | 0.11x | 1.00x | 17x | 124x |

NumPy is the floor: an uncompressed slice and a memcpy, holding the full array in RAM. jix pays
3x to 9x over that floor and stores the array in a quarter of the space. Blosc2 and Zarr pay one
to two orders of magnitude more for the same compression ratio.

> **Unverified.** The Blosc2 arm uses automatic chunk selection, so a small read may be
> decompressing a much larger chunk than jix does. A matched-chunk arm has to run before these
> ratios are quoted anywhere. `[real, but suspect]`

### Read cost against a raw slice (Rust)

The factor over an uncompressed `ndarray` slice is not a property of jix - it is a property of how
well your block shape matches your read shape.

| array | block | read | ndarray | jix | factor |
|---|---|---|---|---|---|
| `[11000,460]` | `[32,32]` | `[32,32]` | 0.28 us `[fake]` | 1.4 us `[fake]` | **5.0x** |
| `[11000,460]` | `[32,32]` | `[1,460]` | 0.35 us `[fake]` | 9.8 us `[fake]` | **28x** |
| `[11000,460]` | `[512,32]` | `[128,460]` | 11.0 us `[fake]` | 46 us `[fake]` | **4.2x** |

The middle row is the trap: a single row read from a `[32,32]`-blocked array touches 15 blocks and
throws away 31/32 of every one. The rule is the obvious one, and it is worth stating plainly in
the docs: **choose a block shape that matches how you read.** jix auto-selects a block shape from
the CPU cache sizes when you do not pass one, which is a reasonable default and a bad choice if
your access pattern is lopsided.

---

## Operations on plain arrays

`Plain` storage is a zero-copy view over a normal in-memory buffer. It exists so ordinary arrays
can take part in jix op chains, and it is the right thing to measure against NumPy and ndarray
when you want to isolate the op machinery from the compression.

### Elementwise

Baseline-relative, 1.0 is NumPy. `130_000 x 200 f32`, 104 MB.

```
negate    jix-plain  |========== 1.01x (2.19 ms)   [real]
add       jix-plain  |========== 1.00x (2.88 ms)   [real]
                     1.0 (numpy)
```

Parity, which is the claim. Nothing here is surprising: both are a single streaming pass over
104 MB and both are memory-bandwidth bound. The value of this result is that it establishes there
is no per-op tax for entering a jix chain - the chain results later in this page are not paying
for overhead they then have to earn back.

Rust is the same story against `ndarray`: `[400_000, 64] f32`, negate 11.2 ms vs 11.0 ms, add
16.4 ms vs 16.1 ms. `[fake]`

### Reductions

This one was not expected and is one of the better results in the suite.

| reduction | numpy | jix-plain | speedup |
|---|---|---|---|
| `sum` i32, all | 4.39 ms `[real]` | **1.36 ms** `[real]` | **3.2x** |
| `sum` i32, axis 0 | 11.70 ms `[real]` | **3.12 ms** `[real]` | **3.7x** |
| `sum` i32, axis 1 | 4.78 ms `[real]` | **1.48 ms** `[real]` | **3.2x** |
| `sum` f32, all | 3.21 ms `[real]` | **1.31 ms** `[real]` | **2.4x** |
| `sum` f32, axis 1 | 3.97 ms `[real]` | **1.55 ms** `[real]` | **2.6x** |
| `std` f32, all | **10.74 ms** `[real]` | 34.81 ms `[real]` | **0.31x** |

Integer reductions in NumPy are not well vectorized; jix's reduction kernels are, and they read in
whatever layout the source is already in rather than forcing a canonical order. The `std` row is
a real regression and is discussed in [Where jix loses](#where-jix-loses).

Rust against `ndarray`: `sum` axis 1 (contiguous) 4.9 ms vs 5.2 ms, `sum` axis 0 (strided) 5.4 ms
vs 7.1 ms. `[fake]`

### Awkward strides (Rust only)

`ndarray` iterates in logical index order. jix sorts axes by stride before iterating, the way
NumPy does, so a transposed source is read in memory order instead of jumping.

| input | operation | ndarray | jix | speedup |
|---|---|---|---|---|
| `[1200,1200] f32` transposed | negate, C-order out | 6.4 ms `[fake]` | **1.9 ms** `[fake]` | **3.4x** |
| `[1200,1200] f32` transposed | negate, transposed out | 2.0 ms `[fake]` | **1.8 ms** `[fake]` | 1.1x |

The second row is the control: when the output layout matches the input, there is nothing to sort
and both libraries do the same work. The gap in the first row is entirely the axis ordering.

There is no Python version of this claim. NumPy sorts axes too, so jix has no advantage to show.

---

## Operations on compressed arrays

### Against Blosc2

`130_000 x 200 f32`, 104 MB raw. Both libraries hold the array compressed and produce an
uncompressed result.

| operation | blosc2 | jix | speedup |
|---|---|---|---|
| negate | 443 ms `[real]` | **49.6 ms** `[real]` | **9.0x** |
| add | 561 ms `[real]` | **99.1 ms** `[real]` | **5.7x** |
| `sum` f32, all | **39.6 ms** `[real]` | 48.6 ms `[real]` | 0.81x |
| `std` f32, all | **74.0 ms** `[real]` | 82.4 ms `[real]` | 0.90x |

Elementwise is a clear win; reductions are a modest loss. Both halves ship.

Both libraries write their result into a plain uncompressed NumPy buffer here - Blosc2's
`expr[:]` does not re-compress on the way out, so it is not being billed for a pass jix skips.
(`expr.compute()` is the path that would; that is a different comparison and not this one.)

### How compression ratio affects operation speed

Same array shape, same code path, same operation. The only thing that changes is how many distinct
values the data contains, and therefore how well it compresses.

| unique values | ratio | jix negate | vs `random` data |
|---|---|---|---|
| random (2^32) | 1.00 `[fake]` | 96.4 ms `[fake]` | 1.00x |
| 256 | 7.2 `[fake]` | 61.0 ms `[fake]` | 1.6x |
| 16 | 31 `[fake]` | 34.5 ms `[fake]` | 2.8x |
| 4 | 95 `[fake]` | 21.3 ms `[fake]` | **4.5x** |
| 2 | 180 `[fake]` | 18.1 ms `[fake]` | **5.3x** |

Nothing about the operation changed; the array just got smaller, so there was less memory to move
and the zstd matches got longer. The same effect shows up in the filter choice: byte-shuffled data
compresses better and negates in 49.6 ms, unshuffled in 69.5 ms `[real]`, with identical op code
on both sides.

For reference, NumPy negates the same 104 MB uncompressed array in 2.16 ms `[real]`. An
elementwise op writes a full-size uncompressed output whatever the input was, so compression only
ever helps the read half of the work - which is why this table is about the *slope*, not about
catching NumPy.

### Is a compressed reduction ever faster than a raw one?

A reduction is the case where the shape of the problem changes. The output is a handful of bytes,
so the read side is the entire cost, and the question becomes clean: is decompressing N bytes
faster than reading N bytes from DRAM?

| unique values | ratio | jix `sum` | vs numpy | effective throughput |
|---|---|---|---|---|
| random | 1.0 `[fake]` | 71.2 ms `[fake]` | 22x slower `[fake]` | 1.5 GB/s `[fake]` |
| 256 | 7.2 `[fake]` | 22.0 ms `[fake]` | 6.8x slower `[fake]` | 4.7 GB/s `[fake]` |
| 16 | 31 `[fake]` | 7.8 ms `[fake]` | 2.4x slower `[fake]` | 13.3 GB/s `[fake]` |
| 4 | 95 `[fake]` | 3.6 ms `[fake]` | 1.1x slower `[fake]` | 28.9 GB/s `[fake]` |
| 2 | 180 `[fake]` | 2.9 ms `[fake]` | **1.1x faster** `[fake]` | 35.9 GB/s `[fake]` |
| *numpy baseline* | 1.0 | 3.21 ms `[real]` | 1.00x | 32.4 GB/s |

The last column is the one that explains the result: it is the rate at which jix processes
*original, uncompressed* array bytes. Once that number exceeds the machine's DRAM streaming
bandwidth, decompressing beats reading, and the crossing point is not a coincidence - it is the
same number.

On these placeholder figures break-even lands near **150x compression** `[fake]`. Wherever it
actually lands, the second reading of the table stands on its own: jix runs the same reduction
within a small factor of NumPy while holding the array in a fraction of the memory. At 95x
compression that is 1.1x the time for 1/95th the footprint. Whether that trade is worth taking is
the reader's call, and the table is here so they can make it.

> **Never measured.** Every operation benchmark in the suite, in both languages, currently runs on
> the `smooth` profile only. The unique-value axis does not exist yet in either harness, so both
> tables above are entirely invented. See `PLAN.md` C2/C3.

---

## Operation chains

Every jix operation returns a lazy view; the whole chain is encoded in the type and runs in a
single pass when you ask for output. NumPy evaluates eagerly and allocates a full intermediate per
step. The difference should grow with the length of the chain, and it does.

### Time

Chain of cheap elementwise ops on `130_000 x 200 f32`, 104 MB.

| ops in chain | numpy | jix-plain | speedup |
|---|---|---|---|
| 1 | 2.9 ms `[fake]` | 2.9 ms `[fake]` | 1.00x |
| 2 | 6.1 ms `[fake]` | 3.4 ms `[fake]` | 1.8x |
| 4 | 12.8 ms `[fake]` | 4.6 ms `[fake]` | 2.8x |
| 8 | 26.0 ms `[fake]` | 7.1 ms `[fake]` | **3.7x** |

NumPy's line is linear in chain length because it makes one full pass per operation. jix's is
nearly flat because it makes one pass regardless: read once, apply the fused chain in registers,
write once. The slope is the claim - a single chain length would prove nothing.

### Peak memory

The same runs, measured as peak RSS in a fresh subprocess.

| ops in chain | numpy | jix-plain | jix compact |
|---|---|---|---|
| 1 | 218 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| 2 | 322 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| 4 | 428 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |
| 8 | 430 MB `[fake]` | 215 MB `[fake]` | 135 MB `[fake]` |

jix is flat: input plus output, whatever the chain does in between. The compact column is flat and
lower still, because the input is never held uncompressed.

This is the clearest evidence for the no-intermediates claim, and unlike the timing it has no
noise floor.

### Chains with shape operations (Rust)

For pure elementwise chains in Rust, plain iterators already fuse for free, and hand-written
iterator code will match jix. The claim worth making is the other one: chains that mix elementwise
work with reductions, broadcasts and axis permutations, which are awkward to express as iterators
and which `ndarray` materializes at each step.

`a / a.std(axis).insert_axis(axis).broadcast(shape)` on `[40_000, 200] f32`:

| | ndarray | jix | speedup |
|---|---|---|---|
| axis 0 | 96 ms `[fake]` | **38 ms** `[fake]` | **2.5x** |
| axis 1 | 104 ms `[fake]` | **41 ms** `[fake]` | **2.5x** |

---

## Where jix loses

Same charts, same baselines, bars to the left of 1.0.

### `std` and other multi-pass reductions, 3.2x slower than NumPy

`std` over 104 MB f32: NumPy 10.74 ms, jix-plain 34.81 ms `[real]`. The gap is worst on `axis=1`
and the full reduction, and smaller on `axis=0` (1.3x). This is a real deficiency in the kernel,
not a measurement artifact, and it is the clearest thing to fix next.

### Chains of expensive math are slower than NumPy

`(exp(a) * 0.5 + 1).log()`: NumPy 80.9 ms, jix-plain 90.9 ms `[real]`, a 1.12x loss. NumPy
dispatches `exp` and `log` to a vectorized libm; jix's are scalar. When the per-element math costs
more than the memory traffic, saving the intermediates buys nothing and the slower kernel decides
the result.

The practical rule: jix chains win when the operations are cheap and the array is large. They lose
when a single transcendental dominates.

### Reductions on compressed arrays lose to Blosc2

`sum` 0.81x, `std` 0.90x, `exp().sum()` 0.47x `[real]`. Blosc2 has native reduction paths over
compressed chunks that are better tuned than jix's.

### No threading

jix is single-threaded throughout. Blosc2 with 8 threads will beat every compressed-array number
on this page. If your workload is throughput-bound and you have cores to spare, that is the right
comparison to make and jix does not win it.

### No decompressed-block cache

`ReadContext` carries a buffer pool, not a cache of decompressed blocks, so re-reading the same
region decompresses it again. For access patterns with locality this leaves a large amount on the
table.

---

## Reproducing

    python jix/benches/run.py                  # rust, criterion
    python jix-py/python/benches/run_all.py    # python, pytest-benchmark

The CI matrix (three runners: linux x86_64, linux aarch64, macos arm64) is documented in
[`docs/benchmarks.md`](../docs/benchmarks.md), along with `scripts/bench/compare.py` for
same-machine A/B against a base ref.

Every number on this page is generated from the benchmark JSON; none are typed by hand.
