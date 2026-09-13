"""Grouped-bar report renderer.

One plot per operation. The x axis is discrete cases we chose deliberately; each case holds one
bar per library; the y axis is performance relative to a baseline library, so a bar above the 1.0
rule is better and a bar below it is worse.

Bars rise from a common floor on a log axis, with a rule drawn at 1.0 - so the baseline library's
bar tops out exactly on the rule, and a 40x win and a 300x loss share a readable plot.

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

# Validated categorical palette; slots assigned in draw order so adjacent bars are adjacent slots.
# See bench-report/DISPLAY.md. The baseline library is neutral gray on purpose - it is a reference
# line drawn as a bar, not one of the series being compared.
THEME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "sub": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "baseline": "#a8a69c",
}
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

# Slot assignment order. Baselines are neutral gray - a reference, not a series. The four libraries
# that appear on nearly every plot come first, so in the common case the bars drawn next to each
# other are consecutive palette slots, which is the pairlist the palette was validated against.
# `jix` and `blosc2` always mean the byte-shuffled build; the unfiltered ones appear only in the
# compression section, where the filter is the subject.
COLOR_ORDER = [
    "jix-plain",
    "jix",
    "blosc2",
    "zarr",
    "blosc2-chunked",
    "jix-noshuffle",
    "blosc2-noshuffle",
]
BASELINE_LIBRARIES = {"numpy", "ndarray"}
assert len(COLOR_ORDER) <= len(SLOTS), "more series than validated palette slots"


def library_color(library):
    """Fixed color per library, shared by every plot in the report."""
    if library in BASELINE_LIBRARIES:
        return THEME["baseline"]
    return SLOTS[COLOR_ORDER.index(library)]


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
        rows.append({**common, "section": info["section"], "value": value})
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
        rows.append(
            {
                "platform": platform,
                "section": section,
                "library": parts[1],
                "case": case_labels.get((section, case_id), case_id),
                # Criterion reports nanoseconds; the rest of the report is in seconds.
                "value": json.loads(estimates.read_text())["mean"]["point_estimate"] * 1e-9,
            }
        )
    return rows


# How each metric is turned into a bar height, and which direction is good. Everything except
# throughput is a multiple of the baseline where shorter is better - so a bar twice the height of
# the 1x rule took twice as long, used twice the memory, or stored twice the bytes.
RELATIVE = {
    "time": "shorter is faster",
    "bytes": "shorter is less memory",
    "stored": "shorter is smaller on disk",
}
ABSOLUTE = {"throughput": "taller is faster"}


def _score(value, baseline, metric, section):
    """Turn a raw measurement into a bar height."""
    if metric == "throughput":
        return section["raw_bytes"] / value / 1e6  # MB/s of original array bytes
    if metric == "stored":
        return 1.0 / value  # recorded as raw/stored; shown as the fraction of the raw array kept
    return value / baseline  # time and bytes: a multiple of the baseline, less is better


def _format_score(score, metric="time"):
    """A bar's label: a multiple of the baseline, or an absolute rate for throughput."""
    if metric in ABSOLUTE:
        return f"{score:,.0f}"
    for threshold, digits in ((100, 0), (10, 0), (2, 1), (0.1, 2), (0.01, 3)):
        if score >= threshold:
            return f"{score:.{digits}f}x"
    return f"{score:.4f}x"


def _format_value(value, metric):
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


def plot_section(rows, section, out_dir):
    """Render one section to a single PNG, one subplot row per platform.

    ``section`` is a spec dict: ``key``, ``title``, ``subtitle``, ``baseline``, ``metric``,
    ``cases`` (x order) and ``libraries`` (bar order).
    """
    metric = section.get("metric", "time")
    cases, libraries = section["cases"], section["libraries"]
    by_platform = defaultdict(dict)  # platform -> (case, library) -> value
    for row in rows:
        if row["section"] == section["key"]:
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
                if value is None or (baseline is None and metric in RELATIVE and metric != "stored"):
                    continue
                score = _score(value, baseline, metric, section)
                if score > 0:
                    got[(case, library)] = score
        scores[platform] = got
    everything = [s for got in scores.values() for s in got.values()]
    if not everything:
        return None
    floor = 10 ** (math.floor(math.log10(min(everything))) - 0.15)
    ceiling = 10 ** (math.log10(max(everything)) + 0.55)
    if metric in RELATIVE:
        # Keep headroom on both sides of the 1x rule, so it reads as a reference line rather than
        # the top or bottom edge of the plot when every library lands on one side of it.
        floor, ceiling = min(floor, 1 / 1.5), max(ceiling, 1.9)

    if metric in RELATIVE:
        legend_line = f"relative to {section['baseline']}; {RELATIVE[metric]}"
    else:
        legend_line = ABSOLUTE[metric]
    subtitle = f"{section.get('subtitle', '')} - {legend_line}"
    # Wide enough for the bars, but never so narrow that the header text runs off the edge.
    width = min(16.0, max(7.0, 0.30 * len(cases) * len(libraries) + 2.2, 0.058 * len(subtitle) + 0.4))
    height = 2.55 * len(platforms) + 1.5
    fig, axes = plt.subplots(len(platforms), 1, figsize=(width, height), squeeze=False, sharex=True)
    fig.set_facecolor(THEME["surface"])
    axes = [ax for (ax,) in axes]

    span = 0.84  # width of a case's bar group, leaving a gap between cases
    bar_w = span / len(libraries)
    for ax, platform in zip(axes, platforms):
        ax.set_facecolor(THEME["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(THEME["axis"])
        ax.tick_params(colors=THEME["muted"], labelsize=8, length=0)
        ax.set_yscale("log")
        ax.set_ylim(floor, ceiling)
        ax.grid(axis="y", color=THEME["grid"], lw=0.7, zorder=0)
        ax.set_axisbelow(True)

        for ci, case in enumerate(cases):
            for li, library in enumerate(libraries):
                score = scores[platform].get((case, library))
                if score is None:
                    continue
                x = ci + (li + 0.5) * bar_w - span / 2
                # bottom/height on a log axis still draws the rectangle floor -> score.
                ax.bar(
                    x,
                    score - floor,
                    bottom=floor,
                    width=bar_w * 0.84,  # the gap between neighbouring bars
                    color=library_color(library),
                    zorder=3,
                    linewidth=0,
                )
                if library in BASELINE_LIBRARIES and metric in RELATIVE:
                    continue  # the bar tops out on the 1x rule; a "1.00x" label adds nothing
                ax.annotate(
                    _format_score(score, metric),
                    (x, score),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    rotation=90,
                    color=THEME["sub"],
                )
        if metric in RELATIVE:
            # Every bar is read against this line, and it is where the baseline's own bar tops out.
            ax.axhline(1.0, color=THEME["ink"], lw=1.2, zorder=4)
        decades = range(math.floor(math.log10(floor)), math.ceil(math.log10(ceiling)) + 1)
        ticks = [10.0**d for d in decades if floor <= 10.0**d <= ceiling]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([_format_score(t, metric) for t in ticks], fontsize=7.5)
        ax.minorticks_off()
        ax.set_ylabel(
            f"{platform}\n{section['unit']}" if "unit" in section else platform, color=THEME["sub"], fontsize=9
        )

    axes[-1].set_xticks(range(len(cases)))
    axes[-1].set_xticklabels(cases, fontsize=8, color=THEME["sub"])
    axes[-1].set_xlim(-0.5, len(cases) - 0.5)

    fig.text(0.008, 0.985, section["title"], fontsize=13.5, fontweight="bold", color=THEME["ink"], va="top")
    fig.text(
        0.008,
        0.951,
        subtitle,
        fontsize=8.2,
        color=THEME["muted"],
        va="top",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, fc=library_color(lib), ec="none") for lib in libraries]
    fig.legend(
        handles,
        libraries,
        loc="lower center",
        ncol=min(len(libraries), 7),
        frameon=False,
        fontsize=8.5,
        labelcolor=THEME["sub"],
        bbox_to_anchor=(0.5, 0.002),
    )
    fig.tight_layout(rect=(0, 0.04 + 0.2 / height, 1, 1 - 0.62 / height))
    out = Path(out_dir) / f"{section['key']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=THEME["surface"])
    plt.close(fig)
    return out


def markdown_table(rows, section):
    """The collapsed ground-truth table that sits under a plot: absolute values, one row per case."""
    metric = section.get("metric", "time")
    cells = defaultdict(dict)
    for row in rows:
        if row["section"] == section["key"]:
            cells[row["case"]][row["library"]] = row["value"]
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
        values = [_format_value(got[lib], metric) if lib in got else "-" for lib in libraries]
        # case labels are two-line on the plots; a markdown table row cannot contain a newline
        lines.append(f"| {case.replace(chr(10), ' ')} | " + " | ".join(values) + " |")
    lines += ["", "</details>"]
    return "\n".join(lines)


def natural_key(text):
    """Natural sort key: '20' before '100', 'a2' before 'a10'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(text))]


def render_report(rows, template, out_path, plots_dir, sections):
    """Expand ``<!-- plot:KEY -->`` and ``<!-- table:KEY -->`` markers in a template.

    Keeps the report's numbers generated rather than typed: the tables come from the same rows the
    plots do, so a table can never drift from the chart above it.
    """
    text = Path(template).read_text()
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
