# Benchmark plan

The report is organized **by operation**, not by claim. Each section shows one or two plots of a
single operation across deliberately chosen cases, with every library side by side. The claims
below are notes between you and me about what each section is likely to show - they do not appear
in the report, and the measurement decides the wording.

We are not sweeping a cartesian product. Every case on every x axis is there because we chose it.

## Ground rules

**Single-threaded everywhere.** jix has no threading. blosc2, zarr and numexpr are all pinned to
one thread (numexpr matters: blosc2 evaluates lazy expressions through it and it keeps a separate
pool - see `FINDINGS.md`). numpy's elementwise and reduction kernels are single-threaded anyway.

**One baseline per language.** Rust normalizes to `ndarray`, Python to `numpy`. The baseline is
always drawn as a 1x bar.

**Pinned defaults**, unless a section's cases vary one of them:

| knob | value |
|---|---|
| math dtypes | **`f32` and `i32`, both, almost always** |
| read dtype | **`i32` only** |
| ndim | 2, except the axis-order section which needs 3 |
| codec | zstd level 3 |
| filters | byte-shuffle. `jix` and `blosc2` always mean the shuffled build; `-noshuffle` arms appear only in the compression section |
| data distribution | `smooth`, except where distribution is a case |
| array shape | **`[130_000, 200]` in both Rust and Python** (104 MB as f32), stated on every plot |

Two dtypes is not a sweep - it is the minimum honest sample. The suite already shows the two
diverging sharply (`sum` all-axis: jix beats numpy 3.2x on `i32` and 2.4x on `f32`), and showing
only the flattering one would be a choice.

**Do not get attached to the integer-reduction win.** jix's kernels are tuned on macOS arm64;
numpy is a general-purpose library whose hot paths have had far more x86 attention. The 3.2x may
shrink or invert on `linux-x86_64`. It is reported as a measurement in the reduction plot, not
framed as a property of the library, and the per-platform stacking will show it either way.

---

## Sections, and what each one measures

### 2. Compression

Ratio and compression throughput. Cases on x: the three data distributions.

| | |
|---|---|
| cases | `random`, `smooth`, `4 unique values` |
| libraries | jix, jix-shuffle, blosc2, blosc2-shuffle, zarr |
| dtype | `i32` |
| status | EXISTS (`test_compress.py`); join ratio and throughput into one section |

Two plots: compression ratio, and compress throughput.

Ratio is normalized to the uncompressed array, `prod(shape) * itemsize`, so the raw array is 1x by
definition. For throughput the numpy bar is a plain `memcpy` of the same array - the price of not
compressing at all - which keeps the baseline meaningful on a plot where numpy has no codec.

### 1. Read

Random regions - each timed call reads a different one of 1024 pre-generated regions. This is the
data-loader pattern, and the only read pattern worth measuring.

| | |
|---|---|
| cases | see below |
| libraries | numpy, jix, jix-shuffle, blosc2, blosc2-shuffle, zarr |
| dtype | `i32` only |
| status | ADAPT (`test_read.py`) |

Cases, block shape and read shape together:

| block | read | what it isolates |
|---|---|---|
| `[16,70]` | `[1,70]` | one row, spans a block row |
| `[16,70]` | `[16,70]` | read == block, the aligned best case |
| `[64,70]` | `[16,16]` | read is a fraction of a block, the wasteful case |
| `[64,70]` | `[256,70]` | **new** - several blocks per read |
| `[64,70]` | `[4096,70]` | **new** - large slab, per-call overhead amortized |
| `[64,70]` | full array | **new** - the pure-throughput limit |

The three new rows are the ones that answer "how do these libraries behave once you stop measuring
call overhead". zarr's flat ~330 us at every small read size looks like pure Python indexing cost;
the large-read cases are where its codec performance actually shows.

**`blosc2-chunked` arm, measured and justified.** The block *is* blosc2's decompression unit, as
expected - an 8.7 MiB chunk costs 2.3x a 4 KiB one, not the ~2000x that chunk-granular decompression
would imply. But 2.3x is not nothing, and blosc2 also silently rewrites a requested `(64, 200)`
block to `(52, 200)` unless `chunks` is pinned too. So the arm ships as "blosc2 at its best
configuration", not as a correction to an unfair one. Numbers in `FINDINGS.md`.

**Rust half:** jix `Compact` against `ndarray` slicing on the same `[130_000, 200] i32` array, the
three alignment cases.

### 3. Negate, and 4. Add

The two elementwise operations. Same structure for both.

**Plot A - dtype and storage.** Cases: `f32` and `i32`. Bars: numpy, jix-plain, jix, jix-shuffle,
blosc2, blosc2-shuffle, zarr. This puts the uncompressed comparison (jix-plain against numpy, where
we expect parity) and the compressed comparison (against blosc2, where jix currently wins ~9x) in
one picture.

**Plot B - data distribution.** Cases: `random`, `smooth`, `16 unique`, `4 unique`. Bars: the
compressed libraries plus the numpy 1x reference. This is where the "better compression makes ops
faster" story lives - **not as its own section and not as its own axis**, just as extra cases in
the same plot. The compression ratio achieved goes in the collapsed table underneath.

An elementwise op writes a full-size uncompressed output whatever the input was, so compression
only helps the read half. The distribution cases show how much that is worth; they are not
expected to catch numpy.

Status: ADAPT. `test_ops.py::_build` is hard-wired to the `smooth` profile and `f32`, so neither
the dtype nor the distribution case exists today. Rust: NEW, there is no `ndarray` arm anywhere in
`jix/benches/`.

### 5. Reductions

One plot, `sum` and `std` together, because they behave differently and splitting them would be
choosing which result to show.

| | |
|---|---|
| cases | `sum axis=0`, `sum axis=1`, `sum all`, `std axis=0`, `std all` - each for `f32` and `i32` |
| libraries | numpy, jix-plain, jix-shuffle, blosc2-shuffle, zarr |
| status | ADAPT |

Expect `sum` above the line and `std` well below it (currently 3.2x slower than numpy). Both in
the same chart, same axis. The `std` result is a real deficiency in the kernel and the plot says so
without a paragraph apologizing for it.

Note that blosc2 beats jix on compressed reductions (`std` all-axis 74 ms against 82 ms) while
losing badly on compressed elementwise. Same plot, both facts visible.

### 6. Operation chains

The no-intermediates story. Cases on x: **chain length 1, 2, 4, 8**, for `f32` and `i32`.

| | |
|---|---|
| chain | cheap elementwise only - `((a * 2 + b) - 3) * 0.5 ...` |
| libraries | numpy, jix-plain, jix-shuffle |
| status | NEW |

Chain length stays a multi-point axis because the *slope* is the result: numpy makes one full pass
per operation, jix makes one pass regardless. A single length would show a number with no mechanism
behind it.

**The existing chain benchmark measures the wrong thing.** `(exp(a) * 0.5 + 1).log()` is dominated
by transcendental math rather than memory traffic, so numpy's vectorized libm decides the result
and jix comes out 1.12x slower (90.9 ms against 80.9 ms). It stays in the report as one extra case
labelled `exp/log`, next to the cheap-op cases, where the contrast is the point.

**Peak memory** is a second plot in this section, same cases. It works the way `test_compress`
works: a normal pytest-benchmark test that records the extra metric into `extra_info`, so it lands
in `python.json` next to everything else and needs no separate file or merge step. The measurement
itself runs the chain in a fresh subprocess and reads its `ru_maxrss`, because `tracemalloc` cannot
see numpy's C-level allocations and an in-process high-water mark is contaminated by every test
that ran before it. Python only. Memory has no noise floor, which makes it the cleanest evidence on
the page.

**Rust half:** the same cheap chain against idiomatic `ndarray` (one allocation per step), plus
`normalize.rs` given an `ndarray` arm. That second one matters more - for pure elementwise chains
Rust iterators already fuse for free, so the honest Rust claim is about chains that mix elementwise
work with reductions and broadcasts.

### 7. Axis order (Rust only)

jix sorts all axes by descending stride; ndarray classifies into C, F, first-axis-contiguous,
last-axis-contiguous, or nothing (`FINDINGS.md`). The benchmark has to hit the "nothing" case.

| | |
|---|---|
| cases | 3-D `[A,B,C]` permuted `[1,2,0]` (rotation, hits `Layout::none()`), `[2,1,0]` (reversal, is F-layout - the control), 2-D transpose (the control that shows no difference) |
| libraries | ndarray, jix-plain |
| dtypes | `f32`, `i32` |
| status | NEW - the existing `bench_op1_plain_transposed` uses a 2-D transpose, which ndarray handles perfectly, so it measures nothing |

No Python counterpart: numpy sorts axes too, so there is no comparison to make.

---

## Work to do

### Python - trim to only what the report shows

The Python suite exists solely to compare against other libraries. Anything not on the page comes
out.

- `test_ops.py`: drop the 5-size sweep to the single canonical size. Add `i32` alongside `f32`
  throughout. Add the data-distribution cases. Replace `test_elementwise_pipeline` with the
  chain-length benchmark, keeping `exp/log` as one case.
- `test_read.py`: drop the 5-size sweep to the single canonical size, add the three large read
  shapes.
- `test_compress.py`: keep, one size.
- `data.py`: add a unique-value-count generator to replace the fixed `low_entropy` profile.
- `array_impls.py`: add `Blosc2ChunkedArray` with `chunks == blocks`.
- New: `test_peak_rss.py`, shaped like `test_compress.py` - a pytest-benchmark test that records
  `peak_rss_bytes` into `extra_info`, measured in a subprocess.

### Rust - add only, never trim

The Rust benches are development tools for optimizing the library, so they stay as they are. The
report benchmarks go in new files alongside them:

- `benches/vs_ndarray.rs` - reads, negate, add, sum, std, chains, axis order. Everything the report
  needs, in one bench target with an `ndarray` arm throughout.

`jix/benches/run.py` grows a `--report` flag that runs only that target
(`cargo bench --bench vs_ndarray`) instead of the whole suite, so generating the report does not
pay for the optimization benches.

### Reporting - `benches/report.py`

- The grouped normalized bar renderer described in `DISPLAY.md`, replacing `plot_throughput`
- Per-platform vertical stacking into a single PNG per operation
- Collapsed absolute-number tables under each plot
- A fixed library-to-color map shared by the Rust and Python halves

### Already done on this branch

- `test_read.py` cycles 1024 random regions instead of re-reading one cache-warm region
- `numexpr.set_num_threads(1)` in `array_impls.py`; numexpr added to `requirements.txt`

---

## Status

Everything in this plan is implemented. The Python suite and `jix/benches/vs_ndarray.rs` both
parametrize themselves from `report_spec.py`, so the benchmarks and the plots cannot disagree
about what a case is. `bench-report/README.md` is the runbook.

What has not happened yet: a real run on CI. The numbers in `report.md` are still fake.
