# Display vocabulary

The report is **visual first**. A reader scrolling it should get the whole story from the plots
alone. Tables are the ground truth and sit under every plot, but folded away.

## The one chart type

Every comparison in the report is the same chart, so the reader learns to read it once:

- **x axis: discrete cases.** Not array size. Each tick is a configuration we deliberately chose -
  a dtype, a data distribution, a block/read shape, a reduction axis.
- **bars within a case: one per library.** Same color for the same library everywhere in the
  report, so the legend stops being needed after the first plot.
- **y axis: cost relative to the baseline**, log scale. Bars rise from a common floor and a black
  rule is drawn at 1x, so the baseline's own bar tops out exactly on the rule and every other bar is
  read against it. **Shorter is better** - a bar at 4x took four times as long, used four times the
  memory, or stored four times the bytes. The subtitle says which.
- **One exception: compression throughput** is plotted in absolute MB/s with no baseline and no
  rule, where taller is faster. There is no meaningful NumPy arm to normalize against, because not
  compressing is not a compression speed.
- **The baseline is always drawn as its own gray 1x bar**, not just as a gridline, so a case never
  looks like it is missing a library. Its value label is suppressed: it is 1.00x by construction.
- **Every bar is labelled** with its ratio and its absolute time.

A log axis with a rule at 1x is what makes this work across the spread we actually have: a 0.27x
win and an 825x loss appear on the same plot at readable sizes, and "below the rule" is legible at a
glance without reading a single number. Costs read more naturally than speedups here - "blosc2 is
190x slower" lands better than "jix is 190x faster", and it keeps every metric pointing the same
way. There is always headroom above the rule, so it reads as a
reference line rather than the top edge of the plot.

Implemented in `jix-py/python/benches/report_bars.py`; every plot's cases and library order are
declared in `report_spec.py`, which the benchmarks also read, so the x axis is defined once.

One plot per operation, sometimes two - typically one for uncompressed storage and one for
compressed, or one for dtype and one for data distribution. Never more than two.

### What this replaces

`report.py::plot_throughput` currently draws one log-log ops/sec chart per case with array size on
x and one line per library. On log-log axes every library is a parallel straight line, so a 1.2x
difference and a 3x difference look identical - see `results/pipeline_elementwise.png`, where numpy
is 12% faster than jix-plain and 45x faster than blosc2 and the chart gives you no way to tell
those apart. It also answers a scaling question the report is not asking.

`plot_throughput` gets retired.

## Absolute numbers under every plot

Markdown supports collapsed sections, and both GitHub and MkDocs render them:

```markdown
<details>
<summary>Absolute numbers</summary>

| case | numpy | jix | blosc2 |
|---|---|---|---|
...
</details>
```

Every plot gets one, holding absolute means, the ratio, and the sample count. Visible by default
would bury the plots; absent entirely would make the report unciteable.

## Per-platform

Two runners: `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (aarch64). One PNG per operation, with
the platforms **stacked vertically as subplots** sharing the x cases and a single legend. One
image, one row per platform, so a platform-specific result is obvious rather than something the
reader has to reconstruct by opening several files. A section whose metric does not vary by CPU - compression
ratio - sets `per_platform: False` and renders one panel.

Any result that flips sign between platforms gets called out in the text of that section. Several
probably will - see the note on integer reductions in `PLAN.md`.

## Rules

- **No log-log throughput charts, no array-size x axis.**
- **Every relative number names its baseline in the same sentence.** "3.7x faster" is not a result;
  "3.7x faster than numpy" is.
- **Absolute numbers always accompany ratios**, even in prose. A 10x speedup on a 2 us operation is
  not the same claim as a 10x speedup on a 200 ms one.
- **Losses are plotted, not written around.** A bar below the line in the same chart as the wins.
  There is no separate section for them; `std` belongs in the reduction plot next to `sum`.
- **Colors are fixed report-wide.** jix, jix-shuffle, jix-plain, numpy, blosc2, blosc2-shuffle,
  zarr each keep one color in every plot on the page.
- **State the machine once, prominently** - CPU model, core count, single-threaded, library
  versions, from `meta.json`.

## Palette

Seven categorical slots, one per library, fixed across the whole report; baseline libraries
(`numpy`, `ndarray`) are neutral gray because they are a reference rather than a series. Slots are
assigned in `COLOR_ORDER`, which puts the libraries that appear on nearly every plot first, so the
common case draws consecutive palette slots. Validated
with the data-viz validator in light mode: all checks pass, worst adjacent CVD dE 9.1, worst
adjacent normal-vision dE 19.6. Three slots sit below 3:1 contrast on the light surface, which
obliges relief - satisfied by the per-bar value labels and the table under every plot.

Each plot draws a *subset* of the libraries, so bars that end up adjacent are not always adjacent
palette slots. Every section's drawn subset was validated separately and all pass. **Re-run that
check when the library set of any plot changes.**

## Library naming

`jix` and `blosc2` always mean the byte-shuffled build - the configuration anyone would actually
use - so a bar's identity never changes meaning between plots. The unfiltered builds appear only in
the compression section, where the filter is the thing being measured, and are named
`jix-noshuffle` / `blosc2-noshuffle` there.

`jix-plain` (jix over an ordinary uncompressed buffer) is on every plot except the compression
ones. It is what separates the cost of jix's operation machinery from the cost of decompression,
and without it every compressed-storage number is unattributable.
