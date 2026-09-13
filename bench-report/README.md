# bench-report

Design workspace for the jix benchmark report. Nothing here is published; this is where we
decide *what* to measure and *how* to show it before writing a single new benchmark.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | The spine. One section per operation, the exact cases on each plot's x axis, and whether the benchmark exists today or must be written. |
| `FINDINGS.md` | Facts about ndarray, blosc2 and zarr established by reading their source or by introspection. These decide what the benchmarks have to look like. |
| `DISPLAY.md` | The chart and table vocabulary. How each claim gets rendered, and what we refuse to render. |
| `report.md` | A fake, fully-written docs page with placeholder numbers. The point is to find out what we need to run. |
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
