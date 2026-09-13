# Benchmark plan

One claim, one benchmark, one configuration. We are not sweeping a cartesian product; every
measurement in the report exists because it supports a specific sentence we want to write.

**The claims below are the report's outline, not hypotheses to be proved.** They say what we want
the report to *talk about* and therefore what has to be measured. Whatever the numbers turn out to
be is what gets published - a claim that lands at 0.8x instead of 3x is still a section, just a
differently worded one. Nothing here is designed to reach a predetermined answer.

## Ground rules

**Everything single-threaded.** jix has no threading. blosc2 and zarr do, and are pinned to one
thread (`NTHREADS = 1`, already done in `array_impls.py`). NumPy's elementwise and reduction
kernels are single-threaded anyway. This is stated at the top of the report, not in a footnote,
because it is the first thing a reader will challenge.

**One baseline per language.** Rust normalizes to `ndarray`. Python normalizes to `numpy`.
Every relative number in the report is against that baseline and nothing else.

**Pinned defaults.** Unless a claim explicitly varies one of these:

| knob | value | why |
|---|---|---|
| element dtype | `f32` for math, `i32` for storage/read | f32 is the ML default; i32 shuffles well and matches the existing suite |
| ndim | 2 | every competitor handles 2-D well; higher ndim adds noise, not insight |
| codec | zstd level 3 | blosc2 and zarr are configured identically; level 3 is everyone's default |
| filters | byte-shuffle | the fair default; a no-shuffle arm appears only where the filter is the subject |
| data profile | `smooth` | the existing generator; realistic and moderately compressible |
| python array | `[130_000, 200] f32` (104 MB) | large enough to leave cache, small enough to run everywhere |
| rust array | `[400_000, 64] f32` (102 MB) | same working set as the Python one, so the two halves are comparable |

**Thread pinning is not just `blosc2.set_nthreads`.** blosc2 evaluates lazy expressions through
numexpr, which keeps its own pool (12 threads by default on this machine). `array_impls.py` now
calls `numexpr.set_num_threads(1)` as well. Without it the blosc2 elementwise arms were running
multi-threaded against single-threaded jix and numpy - and still losing, but the comparison was
not the one we claimed to be making.

---

## Claims and their benchmarks

Status key: `EXISTS` runs today as-is, `ADAPT` exists but needs an arm or a trim, `NEW` must be
written from scratch.

### Storage: compression and reads

| id | claim | benchmark | status |
|---|---|---|---|
| S1 | jix compresses as well as blosc2 and zarr at matched codec settings | `test_compress` (py) | EXISTS |
| S2 | Random reads from a compressed array are far cheaper in jix than blosc2 or zarr | `test_read` (py) | ADAPT |
| S3 | A compressed read costs a small constant factor over a raw ndarray slice, and that factor is set by block/read alignment | `bench_read_vs_ndarray` (rust) | NEW |

**S1** - already produces the ratio table. Change: join it with throughput so one table carries
ratio, compress MB/s, and full-read MB/s per profile. Three profiles (`random`, `smooth`,
`low_entropy`), four libraries, one array shape. No sweep.

**S2** - the existing three (block, read) pairs are well chosen and stay:

- `block=[16,70] read=[1,70]` - single row, read spans one block row
- `block=[16,70] read=[16,70]` - read is exactly one block, the best case
- `block=[64,70] read=[16,16]` - read is a fraction of a block, the wasteful case

**Done:** the benchmark now cycles through 1024 pre-generated random regions instead of re-reading
one fixed region. This is the data-loader pattern from the README, and it is the only read pattern
that matters. The old version re-read a single region that stayed warm in cache, which flattered
every library and hid jix's lack of a decompressed-block cache. Expect all the read numbers to get
worse and the relative ordering to change.

**Still to do:** a **blosc2 arm with `chunks == blocks`**. blosc2 currently gets `chunks=None`
(auto), so a small read may decompress a much larger auto-selected chunk than jix does. Worth an
arm to find out; it may explain a good part of the gap.

**S3** - new. Arms: jix `Compact` vs `ndarray` (`.slice(...).to_owned()`). Array
`[11_000, 460] i32`. Three deliberate points, chosen because they bracket the alignment story
rather than sweep it:

| block | read | what it shows |
|---|---|---|
| `[32,32]` | `[32,32]` | read == block: the floor, expect ~5x |
| `[32,32]` | `[1,460]` | one row crossing 15 blocks: the pathological case, expect ~25x |
| `[512,32]` | `[128,460]` | large slab, well aligned: expect ~4x |

The point of S3 is not "jix is 5x slower". It is "the factor is yours to choose, and here is the
rule". That doubles as user documentation.

### Operations on plain (uncompressed) arrays

| id | claim | benchmark | status |
|---|---|---|---|
| O1 | Elementwise ops on `Plain` match ndarray / numpy | `bench_op_plain_vs_ndarray` (rust), `test_negate`/`test_add` (py) | NEW / EXISTS |
| O2 | Integer and axis reductions are faster in jix than numpy | `test_sum` (py), `bench_op_plain_vs_ndarray` (rust) | ADAPT / NEW |
| O3 | On awkwardly strided input, jix beats ndarray because it sorts axes | `bench_op_strided_vs_ndarray` (rust) | ADAPT |

**O1** - Rust side is entirely new; there is currently no ndarray arm anywhere in
`jix/benches/`. Ops: `neg`, `a + b`. Python side already measures this and already shows parity
(2.88 ms vs 2.89 ms) - no change needed beyond trimming sizes to the single canonical one.

**O2** - the Python suite already shows jix winning here (i32 full sum 1.36 ms vs numpy 4.39 ms).
This was not on your original list and it is one of the strongest honest results in the data.
Keep `sum` over `axis=0`, `axis=1`, and all; both `i32` and `f32`. Drop the `std` variants from
this section and move them to "Where jix loses" - `std` is 3.2x slower and hiding it would be
worse than showing it.

**O3** - `op1.rs::bench_op1_plain_transposed` already has the jix half. Add an `ndarray` arm on
the same transposed view. `[1200, 1200] f32`, source transposed, op `neg`, output in both C
order and transposed. Python has no equivalent because numpy sorts axes too, so there is no
claim to make there.

### Operations on compressed arrays

| id | claim | benchmark | status |
|---|---|---|---|
| C1 | Elementwise ops on Compact beat blosc2 by a wide margin | `test_negate`/`test_add` (py) | ADAPT |
| C2 | Better compression makes ops on Compact faster | `test_ops_by_entropy` (py), `bench_op_compact_by_entropy` (rust) | NEW |
| C3 | Above some compression ratio, reducing the compressed array beats reducing the raw one | `test_reduce_breakeven` (py) | NEW |

**C1** - holds in the existing data (negate: jix-shuffle 49.6 ms vs blosc2-shuffle 443 ms, 9x).

**Fairness checked, and the arm is sound.** The worry was that `(-a)[:]` materializes into a
*compressed* blosc2 array before handing back numpy, billing blosc2 for a compression pass jix
never does. It does not. In `blosc2/lazyexpr.py`, `LazyExpr.__getitem__` passes `_getitem=True`,
and the `blosc2.asarray(result)` re-compression branch inside `compute()` is guarded by
`"_getitem" not in kwargs`. Further down in `fast_eval`, the destination is
`np.empty(shape, dtype)` when `getitem` is set and `blosc2.empty(...)` only when it is not. So the
existing arm decompresses straight into a plain numpy buffer, exactly as jix does.

(`.compute()` *is* the recompressing path. If we ever add a "compressed in, compressed out" arm,
that is the method to call - and it would be a different, also interesting, comparison.)

The numexpr thread pin above is the fairness issue that was actually there.

Note the flip side, which the report will state: reductions on Compact are *slower* in jix than
blosc2 (`std` all-axis: 82 ms vs 74 ms; `exp().sum()`: 83 ms vs 39 ms). "jix beats blosc2" is
true for elementwise and false for reductions.

**C2 - the data these benchmarks run on today is the problem.** Every op benchmark, in both
languages, is hard-wired to the `smooth` profile (`test_ops.py::_build` and `op1.rs` both pass
`Smooth`). Nothing in the op suites has ever run on highly compressible data, so there is no
measurement of this claim at all - not a weak one, none.

The generators already have what we need: Python's `low_entropy` is `rng.integers(0, 4)`, four
unique values; Rust's `Profile::LowEntropy` is `rng.i32(0..8)`, eight. They are only wired into
`test_compress` and `compact.rs`.

Two changes:

1. Make the data profile a parameter of the op benchmarks, not a constant.
2. Replace the fixed `low_entropy` profile with an explicit **unique-value count**
   (`2, 4, 16, 256, random`), so the x-axis is a number we control rather than a profile name.
   Report the achieved compression ratio next to each point.

Then the op time and the compression ratio sit in one table and the reader can see whether they
move together. Some evidence they do: jix-shuffle negates in 49.6 ms where no-shuffle jix takes
69.5 ms, same op code, only a smaller compressed footprint between them.

**C3** - the same setup, pointed at a **reduction** instead of an elementwise op, because the two
have very different shapes. An elementwise op writes a full-size uncompressed output no matter
what, so compression only ever helps the read side. A reduction's output is a handful of bytes, so
the read side is the entire cost and the question reduces to something clean: is decompressing N
bytes faster than reading N bytes from DRAM?

Plot jix time relative to numpy against compression ratio, with the 1.0 line drawn, and report
where it crosses - or that it does not. Report effective throughput in *raw array bytes per
second* alongside, so it can be read against the machine's measured DRAM bandwidth. That makes the
result a mechanism rather than a number, whichever way it lands.

Rust mirror (`bench_op_compact_by_entropy`) runs the same unique-value counts against `ndarray`.

### Operation chains

| id | claim | benchmark | status |
|---|---|---|---|
| P1 | A chain of cheap elementwise ops beats numpy / ndarray, and the gap grows with chain length | `test_chain` (py), `bench_chain_vs_ndarray` (rust) | NEW |
| P2 | Chains mixing elementwise with shape ops are where jix wins in Rust | `bench_chain_shape_ops_vs_ndarray` (rust) | ADAPT |
| P3 | Peak memory for a chain is flat in jix and grows with chain length in numpy | `bench_peak_rss.py` (py) | NEW |

**P1 - the existing chain benchmark measures the wrong thing.** `(exp(a) * 0.5 + 1).log()` is
dominated by transcendental math, not memory traffic, so numpy's vectorized libm wins and jix
comes out 1.12x *slower* (90.9 ms vs 80.9 ms). The intermediates being saved are noise next to the
cost of `exp`.

Replace it with a chain of cheap ops - `((a * 2 + b) - 3) * 0.5 ...` - where memory traffic is the
entire cost. Chain length is an explicit axis (1, 2, 4, 8). This is the one place a multi-point
axis earns its keep: the *slope* is the claim. numpy does one full pass per op; jix does one read
and one write regardless of length. A single point proves nothing; the divergence proves the
mechanism.

Keep the `exp`/`log` chain, but move it to "Where jix loses" with an honest explanation.

**P2** - `normalize.rs` already benchmarks
`a / a.std(axis).insert_axis(axis).broadcast(shape)`, which is exactly the README's stated Rust
pitch: elementwise fused with reductions and broadcasts, the thing that is awkward to hand-write
as an iterator chain. It needs an `ndarray` arm. This is the most important Rust result, because
for pure elementwise chains the README itself concedes that Rust iterators already fuse for free.

**P3** - new, Python only. A subprocess per (library, chain length), peak RSS from
`resource.getrusage(RUSAGE_SELF).ru_maxrss`. `tracemalloc` will not work: it does not see NumPy's
C-level allocations. This is the most direct evidence for the no-intermediates claim and it is
almost noise-free, unlike timing.

---

## What has to be written

**New Rust benchmarks** (all of them need an `ndarray` arm, which does not exist anywhere today):

1. `benches/read_vs_ndarray.rs` - S3
2. `benches/op_vs_ndarray.rs` - O1, O2, and the ndarray arm for O3
3. `benches/chain_vs_ndarray.rs` - P1 Rust half
4. `benches/op_compact_by_entropy.rs` - C2 Rust half
5. ndarray arm added to `benches/normalize.rs` - P2

**New Python benchmarks:**

1. `test_ops_by_entropy.py` - C2, unique-value count as a parameter
2. `test_reduce_breakeven.py` - C3
3. `test_chain.py` - P1, replacing `test_elementwise_pipeline`
4. `bench_peak_rss.py` - P3, a standalone runner, not a pytest-benchmark test
5. `Blosc2ChunkedArray` impl with `chunks == blocks` - S2 fairness
6. A `unique_values` generator in `data.py`, replacing the fixed `low_entropy` profile for op work

**Already done in this branch:**

- `test_read.py` now cycles through 1024 random regions instead of re-reading one - S2
- `numexpr.set_num_threads(1)` in `array_impls.py`, plus numexpr in `requirements.txt` - fairness
- blosc2 elementwise arm verified as non-recompressing - C1, no change needed

**Reporting changes** (`benches/report.py`):

1. Baseline-relative bar renderer, log2 axis, 1.0 line, shaded noise band
2. Claims table generator
3. Chain-length line chart (time and peak RSS)
4. Join the ratio table with throughput
5. Retire `plot_throughput` - see `DISPLAY.md` for why

---

## Open risks

1. **Every existing read number is now stale.** The random-region change means reads no longer hit
   a cache-warm region. All of S2's numbers have to be re-measured and the ordering may shift.
2. **blosc2 chunk selection.** `chunks=None` may hand blosc2 a much larger decompression unit than
   jix's block. Unmeasured; the matched-chunk arm settles it.
3. **zarr's flat ~330 us read time** looks like per-call Python overhead, not decompression. If so,
   the report should say "including zarr's indexing overhead" rather than implying a codec
   difference.
4. **No block cache.** `ReadContext` carries a buffer pool, not a decompressed-block cache, so
   repeated reads of one region re-decompress. The random-region arm will make this visible one way
   or the other. Decide whether to report it as a gap or leave it alone.
5. **Single-run laptop numbers** are what the fake report is anchored to. The real report needs
   the CI matrix across the three runners, and the `compare.py` statistics.

## Questions for iteration 2

1. Chain length (1/2/4/8) and unique-value count (5 points) are the only multi-point axes left in
   the plan. Both are there because the *shape* of the curve is the result. Keep both?
2. Does the report cover Rust and Python in one page, or split into two?
3. Do we publish per-platform numbers (linux x86_64, linux aarch64, macos arm64) or pick one and
   mention the others?
4. `test_ops.py` currently runs 5 sizes x 7 libraries x every op. With the report driven by one
   canonical size, most of that is dead weight. Trim it to the canonical size, or keep the sizes
   for regression tracking and just not plot them?
