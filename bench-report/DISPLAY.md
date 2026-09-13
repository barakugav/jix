# Display vocabulary

The measurements in this report span four orders of magnitude (0.3 us reads to 400 ms chains) and
the gaps between libraries are often 10x, not 20%. That combination breaks the default chart.

## What we are replacing

The current `report.py::plot_throughput` draws one log-log ops/sec chart per case, one line per
library. On a log-log axis every library is a straight parallel line, so a 1.2x difference and a
3x difference look identical, and the eye reads the vertical offsets as "roughly the same". Look
at `results/pipeline_elementwise.png`: numpy is 12% faster than jix-plain and 45x faster than
blosc2, and the chart gives you no way to tell those apart.

It is also the wrong question. "Ops per second at each array size" is a scaling study. We are not
making scaling claims; we are making "jix is Nx faster than X at this configuration" claims. The
chart should answer the claim.

`plot_throughput` gets retired.

## What replaces it

### 1. Claims table (top of the report)

The spine. One row per claim, generated from the benchmark JSON so it cannot drift from the data.

| claim | measured | configuration | where it stops holding |
|---|---|---|---|

Four columns, and the fourth is not optional. A claim without a stated limit reads as marketing.

### 2. Baseline-relative bars

The default chart for anything comparative. One bar per library, **log2 x-axis centered on 1.0**,
baseline (numpy or ndarray) drawn as a vertical line at 1.0, value labels on every bar showing
both the ratio and the absolute time.

Log2 is the point: it makes 1.2x and 3x visibly different while still fitting a 40x outlier on the
same axis. A shaded band at 1.0 +/- 5% marks "within noise" so nobody reads a 3% difference as a
result.

Bars to the right of 1.0 are faster than baseline. State that in the axis label, every time.

### 3. Absolute table underneath every chart

Every chart is followed by its numbers: absolute mean, the ratio, and the sample count. Charts are
for the shape of the result; the table is what someone quotes. Never ship one without the other.

### 4. Chain-length line chart

The only line chart in the report, and the only place a multi-point axis appears. x = number of
ops (1, 2, 4, 8), y = milliseconds, linear on both axes because the *slope* is the whole claim:
numpy climbs, jix stays flat. A log axis would hide exactly the thing we are showing.

The peak-RSS version is the same chart with MB on y, and it is the more convincing of the two
because memory measurements have no noise floor.

### 5. Break-even curve (C3 only)

x = compression ratio (log), y = jix time relative to numpy (log), with the 1.0 line drawn. The
answer is where the curve crosses. If it does not cross, the chart still shows how close it gets
and the gap is quoted as a memory saving instead.

### 6. Read-configuration table

Block shape and read shape as rows, libraries as columns, cells are absolute microseconds with the
ratio to jix underneath. A table, not a heatmap: three rows does not justify a colormap, and exact
microsecond values are more useful than a color here.

## Rules

- **No log-log throughput charts.** See above.
- **Every relative number names its baseline in the same sentence.** "3.7x faster" is not a
  result; "3.7x faster than numpy at chain length 8" is.
- **Absolute numbers always accompany ratios.** A 10x speedup on a 2 us operation is not the same
  claim as a 10x speedup on a 200 ms one.
- **Losses get the same chart treatment as wins.** "Where jix loses" uses the identical
  baseline-relative bars, not prose. A section of apologetic paragraphs reads as evasion; a chart
  with bars to the left of 1.0 reads as confidence.
- **State the machine once, prominently.** CPU model, core count, single-threaded, library
  versions, from `meta.json`. Not a footnote.
- **Colors stay fixed across the whole report.** jix keeps one color everywhere, numpy another, and
  so on, so the reader stops reading legends after the first chart.
