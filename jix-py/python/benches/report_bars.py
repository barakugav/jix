"""Grouped-bar report renderer.

One plot per operation. The x axis is discrete cases we chose deliberately; each case holds one
bar per library; the y axis is performance relative to a baseline library, so higher is always
better.

Bars rise from a common floor on a log axis - so the baseline library's bar tops out exactly on the
1x tick, a loss is a bar shorter than it, and a 40x win and a 300x loss share a readable plot.
Compressed size does not draw its baseline (the raw array), whose bar would just mark 1x.

Per-platform results stack vertically into a single PNG per section.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

THEME = {
    "font": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "surface": "#ffffff",
    "grid": "#e1e0d9",
    "ink": "#0b0b0b",
    "sub": "#52514e",
    "muted": "#898781",
    "axis": "#c3c2b7",
}
# Fixed across every plot. jix is the subject, so it gets the one saturated hue and bold labels;
# everything else is a soft tone that stays readable without competing, and the baseline libraries
# share one light stone. `jix` and `blosc2` always mean the byte-shuffled build; the unfiltered
# `-noshuffle` builds appear only in the compression section and reuse their library's color, hatched.
LIBRARY_COLORS = {
    "numpy": "#cfccc2",
    "ndarray": "#cfccc2",
    "jix-plain": "#a8431a",
    "jix": "#e8642c",
    "blosc2": "#a8bdd2",
    "blosc2-chunked": "#6f88a3",
    "zarr": "#e0c48e",
}
BASELINE_LIBRARIES = {"numpy", "ndarray"}


def _bar_style(library):
    """Fill, and hatching for the unfiltered variants, shared by the bars and the legend."""
    color = LIBRARY_COLORS[library.removesuffix("-noshuffle")]
    if library.endswith("-noshuffle"):
        return {"facecolor": color, "hatch": "////", "edgecolor": THEME["surface"]}
    return {"facecolor": color}


def load_python(path, platform):
    """Rows from a pytest-benchmark --benchmark-json file.

    Each benchmark's ``extra_info`` must carry ``section``, ``case`` and ``library``. Two optional
    fields let one run feed more than the obvious plot:

    - ``value_field`` names an ``extra_info`` field to use instead of the measured time, for
      benchmarks whose point is something other than how long they took.
    - ``also`` maps additional section keys to the ``extra_info`` field holding their value, for a
      run that yields two metrics at once - compressing produces both a time and a stored size.
    """
    rows = []
    for bench in json.loads(Path(path).read_text())["benchmarks"]:
        info = bench["extra_info"]
        if "section" not in info:
            continue  # a benchmark that predates the report spec, or is dev-only
        common = {"platform": platform, "case": info["case"], "library": info["library"]}
        field = info.get("value_field")
        value = info[field] if field else bench["stats"]["mean"]
        # Only the timing carries a spread; a recorded metric like a stored size has none.
        spread = None if field else bench["stats"].get("stddev")
        rows.append({**common, "section": info["section"], "value": value, "spread": spread})
        for section, extra_field in info.get("also", {}).items():
            rows.append({**common, "section": section, "value": info[extra_field]})
    return rows


def load_rust(criterion_root, platform, case_labels=None):
    """Rows from a Criterion output tree.

    Only groups named ``report_<section>`` are read; the rest of the Criterion suite exists to
    optimize the library and has no place in the report. Benchmark ids are ``<library>/<case>``,
    where the case is a filesystem-safe id. ``case_labels`` maps ``(section, case_id)`` to the
    label the plot shows, because Criterion ids become directory names and plot labels contain
    newlines and spaces.
    """
    rows = []
    case_labels = case_labels or {}
    for estimates in sorted(Path(criterion_root).glob("**/new/estimates.json")):
        parts = estimates.parent.parent.relative_to(criterion_root).parts
        if len(parts) < 3 or not parts[0].startswith("report_"):
            continue
        section = parts[0].removeprefix("report_")
        case_id = "/".join(parts[2:])
        mean = json.loads(estimates.read_text())["mean"]
        rows.append(
            {
                "platform": platform,
                "section": section,
                "library": parts[1],
                "case": case_labels.get((section, case_id), case_id),
                # Criterion reports nanoseconds; the rest of the report is in seconds.
                "value": mean["point_estimate"] * 1e-9,
                "spread": mean.get("standard_error", 0.0) * 1e-9 or None,
            }
        )
    return rows


# How each metric is turned into a bar height, and which direction is good. The relative metrics
# are a multiple of the baseline where higher is better - twice as fast, half the memory, or half
# the bytes stored. Each maps to (the subtitle, the word for above 1x, the word for below it).
RELATIVE = {
    "time": ("higher is faster", "faster", "slower"),
    "bytes": ("higher is less memory", "less", "more"),
    "stored": ("higher is smaller on disk", "smaller", "larger"),
}
ABSOLUTE = {
    "throughput": "taller is faster",
    "memory": "absolute peak above the arrays themselves; shorter is less",
}


def _score(value, baseline, metric, section):
    """Turn a raw measurement into a bar height."""
    if metric == "throughput":
        return section["raw_bytes"] / value / 1e6  # MB/s of original array bytes
    if metric == "memory":
        return value / 1e6  # MB
    if metric == "stored":
        return value  # recorded as raw/stored, the compression ratio
    return baseline / value  # time and bytes: how many times better than the baseline


def _format_score(score, metric="time"):
    """A bar's label: a multiple of the baseline, or an absolute rate for throughput."""
    if metric in ABSOLUTE:
        return f"{score:,.0f}" if score >= 1 or score == 0 else f"{score:,.2f}"
    for threshold, digits in ((100, 0), (10, 0), (2, 1), (0.1, 2), (0.01, 3)):
        if score >= threshold:
            return f"{score:.{digits}f}x"
    return f"{score:.4f}x"


def _format_cell(cell, metric):
    """One table cell: the measurement, plus its spread where the harness reported one.

    pytest-benchmark gives a standard deviation, Criterion a standard error of the mean. Both are
    taken as reported rather than converted, so the column means "how noisy was this" and not a
    confidence interval.
    """
    value, spread = cell
    text = _format_value(value, metric)
    if not spread:
        return text
    # Express the spread in the same unit as the value, so the two are comparable at a glance.
    unit = text.split(" ", 1)[1] if " " in text else ""
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(unit)
    return f"{text} +/- {spread / scale:.2f}" if scale else text


def _format_value(value, metric):
    if metric == "memory":
        return f"{value / 1e6:.1f} MB"
    if metric == "stored":
        return f"{value:.1f}x smaller"
    if metric == "throughput":
        return f"{value * 1e3:.0f} ms"
    if metric == "bytes":
        return f"{value / 1e6:.0f} MB"
    for unit, div in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if value >= div:
            return f"{value / div:.2f} {unit}"
    return f"{value * 1e9:.0f} ns"


@plt.rc_context({"font.family": THEME["font"]})
def plot_section(rows, section, out_dir):
    """Render one section to a single PNG, one subplot row per platform.

    ``section`` is a spec dict: ``key``, ``title``, ``subtitle``, ``baseline``, ``metric``,
    ``cases`` (x order) and ``libraries`` (bar order).
    """
    metric = section.get("metric", "time")
    relative = metric in RELATIVE
    cases = section["cases"]
    # Compressed size does not draw its baseline, the raw array, whose bar would just mark 1x.
    skip = section["baseline"] if metric == "stored" else None
    libraries = [lib for lib in section["libraries"] if lib != skip]
    by_platform = defaultdict(dict)  # platform -> (case, library) -> value
    source = section.get("source", section["key"])
    for row in rows:
        if row["section"] == source:
            by_platform[row["platform"]][(row["case"], row["library"])] = row["value"]
    platforms = [p for p in section.get("platforms", sorted(by_platform)) if p in by_platform]
    if not section.get("per_platform", True):
        platforms = platforms[:1]  # a platform-independent metric; one panel says it all
    if not platforms:
        return None

    # Score everything up front: the y range decides the bar floor, and a log axis needs one.
    scores = {}  # platform -> (case, library) -> score
    for platform in platforms:
        cells = by_platform[platform]
        got = {}
        for case in cases:
            baseline = cells.get((case, section["baseline"]))
            for library in libraries:
                value = cells.get((case, library))
                if value is None or (baseline is None and relative and metric != "stored"):
                    continue
                score = _score(value, baseline, metric, section)
                # A log axis cannot draw a zero; a linear one can, and for absolute memory zero is
                # the calibration value rather than missing data.
                if score > 0 or not section.get("log", True):
                    got[(case, library)] = score
        scores[platform] = got
    everything = [s for got in scores.values() for s in got.values()]
    if not everything:
        return None
    log_scale = section.get("log", True)
    if not log_scale:
        floor, ceiling = 0.0, max(everything) * 1.3
    else:
        floor = 10 ** (math.log10(min(everything)) - 0.45)
        ceiling = 10 ** (math.log10(max(everything)) + 0.45)
    if relative:
        # Keep headroom on both sides of 1x, so the baseline is never the top or bottom edge of the
        # plot when every library lands on one side of it.
        floor, ceiling = min(floor, 1 / 1.9), max(ceiling, 1.9)

    legend_line = f"relative to {section['baseline']}; {RELATIVE[metric][0]}" if relative else ABSOLUTE[metric]
    subtitle = f"{section.get('subtitle', '')} - {legend_line}"
    # Wide enough for the bars, but never so narrow that the header text runs off the edge.
    width = min(16.0, max(7.0, 0.30 * len(cases) * len(libraries) + 2.2, 0.058 * len(subtitle) + 0.4))
    height = 2.55 * len(platforms) + 1.4
    fig, axes = plt.subplots(len(platforms), 1, figsize=(width, height), squeeze=False, sharex=True)
    fig.set_facecolor(THEME["surface"])
    axes = [ax for (ax,) in axes]

    # A case's bar group spans 0.84, leaving a gap between cases; fewer than three bars stay that
    # wide each rather than stretching to fill it.
    bar_w = 0.84 / max(len(libraries), 3)
    for ax, platform in zip(axes, platforms):
        ax.set_facecolor(THEME["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(THEME["axis"])
        ax.tick_params(colors=THEME["muted"], labelsize=8, length=0)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylim(floor, ceiling)
        ax.grid(axis="y", color=THEME["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for ci, case in enumerate(cases):
            for li, library in enumerate(libraries):
                score = scores[platform].get((case, library))
                if score is None:
                    continue
                x = ci + (li + 0.5 - len(libraries) / 2) * bar_w
                # bottom/height on a log axis still draws the rectangle floor -> score.
                ax.bar(
                    x,
                    score - floor,
                    bottom=floor,
                    width=bar_w * 0.84,  # the gap between neighbouring bars
                    zorder=3,
                    linewidth=0,
                    **_bar_style(library),
                )
                if library in BASELINE_LIBRARIES and relative:
                    continue  # the bar tops out on the 1x tick, which is labelled already
                # A loss is labelled with how many times worse - "6.5x" under the "10x slower" tick
                # rather than "0.15x".
                jix = library.split("-")[0] == "jix"
                ax.annotate(
                    _format_score(1 / score if relative and score < 1 else score, metric),
                    (x, score),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    zorder=5,  # over the grid line a label just under a tick would cross
                    bbox={"boxstyle": "square,pad=0.1", "fc": THEME["surface"], "ec": "none"},
                    fontsize=6.8,
                    color=THEME["ink"] if jix else THEME["sub"],
                    fontweight="bold" if jix else "normal",
                )
        if log_scale:
            decades = range(math.floor(math.log10(floor)), math.ceil(math.log10(ceiling)) + 1)
            ticks = [10.0**d for d in decades if floor <= 10.0**d <= ceiling]
            if len(ticks) < 3:
                # A range inside one decade gets 2x and 5x steps too, both ways from each decade.
                steps = {round(s * 10.0**d, 12) for d in decades for s in (0.2, 0.5, 1, 2, 5)}
                ticks = sorted(t for t in steps if floor <= t <= ceiling)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.set_yticklabels([_tick_label(t, metric, section["baseline"]) for t in ticks], fontsize=7.5)
        ax.minorticks_off()
        title = f"{platform} ({section['unit']})" if "unit" in section else platform
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=THEME["sub"], pad=4)

    axes[-1].set_xticks(range(len(cases)))
    axes[-1].set_xticklabels(cases, fontsize=8, color=THEME["sub"])
    axes[-1].set_xlim(-0.5, len(cases) - 0.5)

    # Header offsets in inches, converted to figure fractions: a fraction would put the title and
    # subtitle on top of each other whenever the figure is short (one platform instead of two).
    fig.text(
        0.008,
        1 - 0.16 / height,
        section["title"],
        fontsize=13.5,
        fontweight="bold",
        color=THEME["ink"],
        va="top",
    )
    fig.text(0.008, 1 - 0.45 / height, subtitle, fontsize=8.2, color=THEME["muted"], va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, lw=0, **_bar_style(lib)) for lib in libraries]
    fig.legend(
        handles,
        libraries,
        loc="upper left",
        ncol=len(libraries),
        frameon=False,
        fontsize=8.5,
        labelcolor=THEME["sub"],
        handlelength=1.2,
        columnspacing=1.4,
        bbox_to_anchor=(0.003, 1 - 0.66 / height),
    )
    fig.tight_layout(rect=(0, 0.01, 1, 1 - 1.04 / height))
    out = Path(out_dir) / f"{section['key']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=THEME["surface"])
    plt.close(fig)
    return out


def _tick_label(tick, metric, baseline):
    """A y tick: the baseline at 1x, and how many times better or worse elsewhere."""
    if metric not in RELATIVE:
        return _format_score(tick, metric)
    if tick == 1:
        return f"{baseline} 1x"
    _, up, down = RELATIVE[metric]
    return f"{round(tick, 6):g}x {up}" if tick > 1 else f"{round(1 / tick, 6):g}x {down}"


def markdown_table(rows, section):
    """The collapsed ground-truth table that sits under a plot: absolute values, one row per case."""
    metric = section.get("metric", "time")
    cells = defaultdict(dict)
    source = section.get("source", section["key"])
    for row in rows:
        if row["section"] == source:
            cells[row["case"]][row["library"]] = (row["value"], row.get("spread"))
    libraries = section["libraries"]
    lines = [
        "<details>",
        "<summary>Absolute numbers</summary>",
        "",
        "| case | " + " | ".join(libraries) + " |",
        "|" + "---|" * (1 + len(libraries)),
    ]
    for case in section["cases"]:
        got = cells.get(case, {})
        values = [_format_cell(got[lib], metric) if lib in got else "-" for lib in libraries]
        # case labels are two-line on the plots; a markdown table row cannot contain a newline
        lines.append(f"| {case.replace(chr(10), ' ')} | " + " | ".join(values) + " |")
    lines += ["", "</details>"]
    return "\n".join(lines)


def natural_key(text):
    """Natural sort key: '20' before '100', 'a2' before 'a10'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(text))]


def render_report(rows, template, out_path, plots_dir, sections, banner="", versions="", machines=""):
    """Expand ``<!-- plot:KEY -->`` and ``<!-- table:KEY -->`` markers in a template.

    Keeps the report's numbers generated rather than typed: the tables come from the same rows the
    plots do, so a table can never drift from the chart above it.
    """
    text = Path(template).read_text().replace("<!-- banner -->", banner).replace("<!-- versions -->", versions)
    text = text.replace("<!-- machines -->", machines)
    for section in sections:
        key = section["key"]
        rel = f"{Path(plots_dir).name}/{key}.png"
        text = text.replace(f"<!-- plot:{key} -->", f"![{section['title']}]({rel})")
        text = text.replace(f"<!-- table:{key} -->", markdown_table(rows, section))
    leftover = [line for line in text.splitlines() if "<!-- plot:" in line or "<!-- table:" in line]
    if leftover:
        raise SystemExit(f"unknown markers in {template}: {leftover}")
    Path(out_path).write_text(text)
    return out_path


def headline(rows, section, case, library, baseline=None):
    """One (absolute, ratio-to-baseline) pair, for the handful of numbers the README quotes."""
    found = {
        (row["case"], row["library"]): row["value"] for row in rows if row["section"] == section and row["case"] == case
    }
    value = found.get((case, library))
    base = found.get((case, baseline)) if baseline else None
    return value, (base / value if value and base else None)
