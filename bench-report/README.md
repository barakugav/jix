# bench-report

Design workspace for the jix benchmark report. Nothing here is published; this is where we
decide *what* to measure and *how* to show it before writing a single new benchmark.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | The spine. One section per operation, the exact cases on each plot's x axis, and whether the benchmark exists today or must be written. |
| `FINDINGS.md` | Facts about ndarray, blosc2 and zarr established by reading their source or by introspection. These decide what the benchmarks have to look like. |
| `DISPLAY.md` | The chart and table vocabulary. How each claim gets rendered, and what we refuse to render. |
| `report.template.md` | The report's prose, with `<!-- plot:KEY -->` and `<!-- table:KEY -->` markers. **Edit this, never `report.md`.** |
| `report.md` | Generated. The full docs page with real plots rendered from fake numbers. |
| `build_fake_report.py` | Makes up benchmark output, drives the real renderer over it, and expands the template. Lets the plot style be iterated on with no benchmark run. |
| `readme-snippet.md` | The fake 4-number block destined for the top-level `README.md`. |

## How to use this

`report.md` is written as if the numbers were already in. Read it end to end and ask: is this
the document I want to publish? Every number in it is traceable to exactly one benchmark in
`PLAN.md`, so anything you want to say that is not backed by a row there is a missing benchmark.

Numbers marked `[real]` came from an actual local run (`jix-py/python/benches/results/python.json`,
dev laptop, single run, not a clean-room measurement). Numbers marked `[fake]` are invented and
plausible. The mix is deliberate: the real ones anchor the report to reality, the fake ones show
what a result would have to look like to support the claim.

## Iteration log

### Iteration 1

First pass. Established the claim-driven structure, wrote the fake docs page, identified the
new benchmarks. Open questions raised for iteration 2 are at the bottom of `PLAN.md`.

### Iteration 2

Acted on three things rather than writing them down:

- **Blosc2 lazy-op fairness: investigated, and the existing arm is fair.** `expr[:]` does not
  re-compress. Details in `PLAN.md` under C1.
- **numexpr was not pinned.** Blosc2 evaluates lazy expressions through numexpr, which has its own
  12-thread pool, so the "single-threaded" elementwise comparison was not. Pinned in
  `array_impls.py`; numexpr added to `requirements.txt`.
- **`test_read.py` now reads random regions.** 1024 pre-generated regions cycled per call, instead
  of re-reading one cache-warm region. Every read number in `report.md` is stale as a result.

Reframed the claims as the report's outline rather than hypotheses, per feedback. The measurement
decides the wording; the plan only decides what gets measured.

Also replaced the fixed `low_entropy` profile in the C2/C3 plan with an explicit unique-value
count, since no op benchmark has ever run on anything but `smooth`.

### Iteration 3

Restructured from claim-driven to **operation-driven**: one section per operation, one or two plots
each, no claims table in the report itself. The claims live in `PLAN.md` as notes between us.

Chart type settled (`DISPLAY.md`): discrete cases on x, one bar per library, speed relative to the
baseline on a log axis anchored at 1.0, baseline always drawn as its own 1x bar. Absolute numbers
in a collapsed `<details>` block under every plot. Per-platform results stack vertically into one
PNG per operation.

Two investigations landed in `FINDINGS.md` and changed the plan:

- **ndarray has no axis sorting**, just a four-way layout classifier. The existing 2-D transpose
  benchmark measures nothing, because a transpose is exactly F-layout. The case that actually
  separates the two libraries is a 3-D *rotation*, which lands on `Layout::none()`.
- **blosc2 auto-selects 8.7 MiB chunks** against a 4 KiB block, and silently rewrites a requested
  `(64,70)` block to `(52,70)`. Forcing `chunks == blocks` is accepted and honored exactly.

Config changes: both `f32` and `i32` for all math, `i32` only for reads, Rust array shape to
`[300_000, 80]`, three large read shapes added so the comparison is not all per-call overhead.

Merged the two compression-quality claims into extra *cases* on the negate and add plots rather
than a section and an axis of their own. Dropped the "where jix loses" section - `std` now sits in
the reduction plot next to `sum`, and the `exp`/`log` chain sits in the chain plot next to the
cheap chains.

Stopped headlining the integer-reduction win: it is probably jix-tuned-for-arm64 against
numpy-tuned-for-x86, and may invert on linux-x86_64.

### Iteration 4

Built the thing that renders the report. `jix-py/python/benches/report_bars.py` is the real
renderer; `report_spec.py` declares every plot's cases and library order in one place, so the
benchmarks and the renderer cannot disagree about what an x tick means.
`bench-report/build_fake_report.py` invents benchmark output, runs it through the real loader and
renderer, and expands `report.template.md` into `report.md`. So `report.md` is now generated, plots
and tables alike - nothing in it is typed by hand.

    python bench-report/build_fake_report.py

Chart shape settled by rendering it and looking: bars rise from a common floor on a log axis with a
rule at 1x, rather than growing out of the 1x line. Growing from the line left the baseline as a
zero-height invisible bar, which is the opposite of showing it.

Measured blosc2's decompression unit rather than guessing: the block, as expected, but an 8.7 MiB
auto-chunk still costs 2.3x a 4 KiB one. `FINDINGS.md` has the table.

Config: one array shape `[130000, 200]` everywhere; read section moved ahead of compression; numpy
kept as a memcpy baseline on plots where it has no codec; compression ratio normalized to
`prod(shape) * itemsize`; peak RSS restructured to record into `extra_info` the way `test_compress`
does, so it lands in `python.json` with everything else.

### Iteration 5

Direction flipped: every relative plot now shows **cost** against the baseline, so shorter is
better and "blosc2 is 190x slower" reads straight off the bar instead of having to be inverted from
a 0.005x speedup. Compression throughput is the single exception - absolute MB/s, no baseline, no
rule, taller is faster.

Library set cut to what a reader would actually use: `jix` and `blosc2` now always mean the
byte-shuffled build, with `-noshuffle` arms only in the compression section where the filter is the
subject. `jix-plain` added to every non-compression plot, since without it a compressed-storage
number cannot be split between op machinery and decompression.

Also: chains are f32 only; array shape and per-case read sizes are on every plot; all thirteen
per-section palette subsets re-validated after the library changes (all pass); figure width now
accounts for the header text, which was being clipped on the narrow plots.
