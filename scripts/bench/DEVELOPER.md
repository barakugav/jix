# Working on the benchmark suite

`README.md` says how to *produce* a report. This says how the thing is built, and how to change it.

## The shape of it

```
report_spec.py     one entry per plot: its cases, its libraries, its baseline, its metric
   |
   +-- benchmarks parametrize themselves from it and tag results with its labels
   |     python:  jix-py/python/benches/test_*.py
   |     rust:    jix/benches/vs_ndarray.rs
   |
   +-- report_bars.py reads the same file to draw the plots and tables
         |
         +-- build_report.py         real artifacts  -> report/plots + report/report.md
```

The point of the middle file is that a plot's x axis and the configuration that produced it are
declared once. If you add a case to `report_spec.py` and not to the benchmark, the build warns.

`report/report.template.md` holds the prose, with `<!-- plot:KEY -->` and `<!-- table:KEY -->`
markers. `report/report.md` is generated; editing it is always wrong. `<!-- versions -->` is filled
from the runs' `meta.json`, so never type a toolchain or library version into the template.

## Adding a benchmark

**Python.** Write a `test_*.py` in `jix-py/python/benches/` that parametrizes from `report_spec`
and calls `record(...)`:

```python
@pytest.mark.parametrize("library", report_spec.CORE)
@pytest.mark.parametrize("case", report_spec.MY_CASES)
def test_thing(benchmark, library, case):
    arr = build(library, "smooth", "f32", report_spec.SHAPE)
    record(benchmark, section="my_section", case=report_spec.my_label(case), library=library)
    assert benchmark(lambda: arr.thing(case)) is not None
```

`record` also takes:

- `value_field="peak_rss_bytes"` - plot that recorded field instead of the measured time, for a
  benchmark whose point is not how long it took.
- `also={"other_section": "ratio"}` - emit a second row under another section, for one run that
  yields two metrics. `test_compress.py` uses this for time and stored size.

**Rust.** Add to `jix/benches/vs_ndarray.rs`, in a group named `report_<section>` with benchmark
ids `<library>/<case_id>`. Case ids become directory names, so they must be filesystem-safe; put
the plot labels in `report_spec` under `case_ids`, parallel to `cases`.

The rest of `jix/benches/` is for optimizing the library and is deliberately not in the report;
`run.py --report` runs only `vs_ndarray`.

**Then** add a section to `report_spec.SECTIONS` and the markers to `report/report.template.md`.

## Section options

| key | meaning |
|---|---|
| `baseline` | the library every bar is divided by, or `None` for an absolute metric |
| `metric` | `time`, `stored`, `bytes` (all relative); `throughput`, `memory` (absolute) |
| `log` | `False` for a linear y axis - needed when zero is a meaningful value |
| `per_platform` | `False` when the measurement cannot vary by CPU, e.g. a stored size |
| `source` | draw from another section's rows, when one benchmark group feeds two plots |
| `case_ids` | Rust only: filesystem-safe ids, parallel to `cases` |

## Things that have already gone wrong here

Each of these produced a plausible-looking wrong number. They are the reasons for several
otherwise-odd choices in the code.

- **A benchmark that measured nothing.** The dim-order cases negated permuted views and reported
  the treatment and both controls as identical. `ndarray`'s contiguity check sorts strides first,
  so a permutation of a contiguous array still takes the fast path. **If a treatment agrees with
  its controls, believe the controls.**
- **A chain that constant-folded.** A Rust chain of `.map(|x| x * 2.0)` closures with literal
  constants collapses to one multiply-add, so it measured one operation at any depth. The chain
  now uses two array operands, which cannot be folded away.
- **A library computing something else.** blosc2 mis-evaluates `(nested expr) + -3.0`. The
  cross-check in `benches/tests/test_array_impls.py` runs every library's operations against NumPy
  on the same data and caught it. **Keep that test passing; nothing in the timings would show it.**
- **Unfaulted pages landing in a memory measurement.** `np.zeros` does not touch its pages, so an
  output buffer allocated that way faults in during the measurement and looks like the engine's
  overhead. Allocate with `np.empty` and `fill`.
- **Stale Criterion results.** Criterion keeps previous runs and CI caches `target`, so a benchmark
  that failed to run reports whatever it measured last time. `scripts/bench/run.py` deletes the
  tree first.
- **A run's fidelity going unrecorded.** `meta.json` carries `fast`; `build_report.py` stamps
  low-fidelity output. Without it a `--fast` run's numbers look exactly like publishable ones.

## Conventions worth keeping

- **`jix` and `blosc2` always mean the byte-shuffled build**, so a bar's identity never shifts
  between plots. The unfiltered variants appear only in the compression section, named
  `-noshuffle`, where the filter is the subject.
- **Compare against a competitor at its best.** `blosc2-chunked` exists because blosc2's default
  chunking is far worse than what it can do; the report shows both.
- **Losses are plotted, not written around.** `std` sits in the reduction plot next to `sum`.
- **Palette.** Seven fixed categorical slots, baselines in neutral gray. Each plot draws a subset,
  so drawn neighbours are not always palette neighbours - every section's subset was checked for
  contrast separately. **Re-check it if you change a plot's library set.**
