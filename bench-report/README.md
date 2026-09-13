# bench-report

Design workspace for the jix benchmark report. Nothing here is published; this is where we
decide *what* to measure and *how* to show it before writing a single new benchmark.

## Files

| file | what it is |
|---|---|
| `PLAN.md` | The spine. One row per claim, one benchmark per claim, exact configuration, and whether the benchmark exists today or must be written. |
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
