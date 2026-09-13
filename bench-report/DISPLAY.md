# Display vocabulary

The report is **visual first**. A reader scrolling it should get the whole story from the plots
alone. Tables are the ground truth and sit under every plot, but folded away.

## The one chart type

Every comparison in the report is the same chart, so the reader learns to read it once:

- **x axis: discrete cases.** Not array size. Each tick is a configuration we deliberately chose -
  a dtype, a data distribution, a block/read shape, a reduction axis.
- **bars within a case: one per library.** Same color for the same library everywhere in the
  report, so the legend stops being needed after the first plot.
- **y axis: speedup relative to the baseline**, log scale, bars anchored at 1.0. Above the line is
  faster than the baseline, below is slower. numpy is the baseline in Python, ndarray in Rust.
- **The baseline is always drawn as its own 1x bar**, not just as a gridline, so a case never looks
  like it is missing a library.
- **Every bar is labelled** with its ratio and its absolute time.

Anchoring at 1.0 on a log axis is what makes this work across the spread we actually have. A 40x
win and a 3x loss appear on the same plot at readable sizes, and "above the line" is legible at a
glance without reading a single number.

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

Three runners: linux x86_64, linux aarch64, macos aarch64. One PNG per operation, with the
platforms **stacked vertically as subplots** sharing the x cases and a single legend. One image,
three rows, so a platform-specific result is obvious rather than something the reader has to
reconstruct by opening three files.

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
