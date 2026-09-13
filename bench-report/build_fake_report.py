"""Generate fake benchmark output and render the report from it.

Exists so the plotting code can be developed and its style iterated on without waiting for a real
benchmark run. It writes a `python.json` in pytest-benchmark's shape, drives the real loader and
renderer over it, and expands `report.template.md` into `report.md` - so anything that works here
works on real results.

    python bench-report/build_fake_report.py

Values are given per library as one entry per case, in the order `report_spec` declares the cases,
so the fake data cannot drift out of step with the plots. Times are in milliseconds. Some values
are real (measured locally, single run); the rest are invented.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "jix-py" / "python"))

from benches import report_bars, report_spec  # noqa: E402

# section -> library -> one value per case, in report_spec case order.
# Times in ms; compress_ratio in raw/stored; chain_memory in bytes.
FAKE = {
    # read: 1x200, 16x200, 16x16, 256x200, 4096x200, full
    "read": {
        "numpy": [0.0004, 0.0006, 0.00035, 0.012, 0.190, 5.2],
        "jix-plain": [0.0005, 0.0007, 0.00045, 0.013, 0.200, 5.5],
        "jix": [0.0016, 0.0026, 0.0036, 0.026, 0.380, 12.0],
        "blosc2": [0.076, 0.080, 0.052, 0.128, 0.880, 23.0],
        "blosc2-chunked": [0.034, 0.036, 0.042, 0.095, 0.780, 22.0],
        "zarr": [0.330, 0.340, 0.336, 0.420, 1.200, 40.0],
    },
    # random, smooth, 16 unique, 4 unique
    "compress_ratio": {
        "numpy": [1.0, 1.0, 1.0, 1.0],
        "jix-noshuffle": [1.00, 3.55, 24.0, 46.0],
        "jix": [1.00, 3.94, 31.0, 55.2],
        "blosc2-noshuffle": [1.00, 3.40, 22.5, 41.0],
        "blosc2": [1.00, 3.71, 28.4, 48.6],
        "zarr": [1.00, 3.71, 28.4, 48.6],
    },
    "compress": {
        "jix-noshuffle": [133, 163, 88, 71],
        "jix": [190, 133, 74, 62],
        "blosc2-noshuffle": [128, 176, 102, 84],
        "blosc2": [196, 147, 88, 72],
        "zarr": [540, 610, 420, 390],
    },
    # f32, i32. The f32 column is measured; the i32 arm does not exist in the suite yet.
    "negate": {
        "numpy": [2.157, 2.21],
        "jix-plain": [2.186, 2.25],
        "jix": [49.6, 45.0],
        "blosc2": [443, 420],
        "zarr": [30.8, 30.0],
    },
    "negate_dist": {
        "numpy": [2.157] * 4,
        "jix-plain": [2.186] * 4,
        "jix": [96.4, 49.6, 34.5, 21.3],
        "blosc2": [510, 443, 210, 155],
        "zarr": [44.0, 30.8, 21.0, 16.0],
    },
    "add": {
        "numpy": [2.886, 2.95],
        "jix-plain": [2.882, 2.94],
        "jix": [99.1, 94.0],
        "blosc2": [561, 540],
        "zarr": [59.2, 58.0],
    },
    # sum f32 axis0/axis1/all, sum i32 axis0/all, std f32 axis0/all - all measured
    "reduction": {
        "numpy": [2.33, 3.97, 3.21, 11.70, 4.39, 11.06, 10.74],
        "jix-plain": [2.03, 1.55, 1.31, 3.12, 1.36, 14.44, 34.81],
        "jix": [49.6, 49.0, 48.6, 22.0, 18.6, 61.7, 82.4],
        "blosc2": [34.7, 36.3, 39.6, 20.4, 14.2, 72.9, 74.0],
        "zarr": [30.9, 32.9, 31.5, 37.1, 29.6, 41.9, 39.1],
    },
    # 1, 2, 4, 8 cheap ops, then the exp/log chain - whose numbers are measured
    "chain": {
        "numpy": [2.9, 6.1, 12.8, 26.0, 80.9],
        "jix-plain": [2.9, 3.4, 4.6, 7.1, 90.9],
        "jix": [50.0, 51.0, 53.0, 57.0, 138.0],
    },
    "chain_memory": {
        "numpy": [218e6, 322e6, 428e6, 430e6],
        "jix-plain": [215e6] * 4,
        "jix": [135e6] * 4,
    },
    "rust_read": {
        "ndarray": [0.00028, 0.00035, 0.011],
        "jix-plain": [0.00030, 0.00038, 0.0115],
        "jix": [0.0016, 0.0105, 0.042],
    },
    "rust_op": {
        "ndarray": [11.0, 10.9, 16.1, 16.0, 7.1, 5.2, 6.8],
        "jix-plain": [11.2, 11.1, 16.4, 16.3, 5.4, 4.9, 2.8],
        "jix": [49.5, 45.0, 99.0, 95.0, 49.0, 48.0, 19.0],
    },
    "rust_chain": {
        "ndarray": [11.0, 22.4, 45.1, 90.3, 96.0, 104.0],
        "jix-plain": [11.2, 12.1, 13.8, 17.2, 38.0, 41.0],
        "jix": [50.0, 52.0, 55.0, 60.0, 88.0, 92.0],
    },
    "rust_axis_order": {
        "ndarray": [88.0, 86.0, 12.0, 2.0],
        "jix-plain": [12.0, 11.8, 12.2, 1.9],
    },
}

# Per-platform multipliers on jix only, so the fake data exercises the case the report has to
# handle honestly: a win on one platform that shrinks on another. The integer reduction is the live
# example - jix is tuned on arm64, numpy has had far more x86 attention. Keyed (section, case idx).
PLATFORM_TWEAKS = {
    "linux-x86_64": {("reduction", 3): 2.6, ("reduction", 4): 2.9},
    "linux-aarch64": {("reduction", 3): 1.1, ("reduction", 4): 1.15},
}
PLATFORM_SCALE = {"linux-x86_64": 1.18, "linux-aarch64": 1.32}


def rows_for(platform):
    """Fake normalized rows for one platform, in the shape the renderer consumes."""
    scale, tweaks = PLATFORM_SCALE[platform], PLATFORM_TWEAKS[platform]
    rows = []
    for key, by_library in FAKE.items():
        section = report_spec.BY_KEY[key]
        metric = section["metric"]
        for library, values in by_library.items():
            if len(values) != len(section["cases"]):
                raise SystemExit(f"{key}/{library}: {len(values)} values for {len(section['cases'])} cases")
            for index, (case, value) in enumerate(zip(section["cases"], values)):
                if metric in ("time", "throughput"):
                    value *= 1e-3 * scale  # ms -> s, then the platform's overall speed
                    if library.startswith("jix"):
                        value *= tweaks.get((key, index), 1.0)
                rows.append({"platform": platform, "section": key, "case": case, "library": library, "value": value})
    return rows


def write_fake_python_json(path, platform):
    """Write a pytest-benchmark shaped JSON, so `report_bars.load_python` gets exercised too."""
    benchmarks = []
    for row in rows_for(platform):
        if row["section"].startswith("rust_") or row["section"] in ("compress_ratio", "chain_memory"):
            continue
        benchmarks.append(
            {
                "name": f"{row['section']}[{row['library']}]",
                "stats": {"mean": row["value"], "rounds": 20},
                "extra_info": {"section": row["section"], "case": row["case"], "library": row["library"]},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmarks": benchmarks}, indent=1))
    return path


def render_report(rows, template, out_path, plots_dir):
    """Expand `<!-- plot:KEY -->` and `<!-- table:KEY -->` markers in a template.

    Keeps the report's numbers generated rather than typed: the tables come from the same rows the
    plots do, so a table can never drift from the chart above it.
    """
    text = Path(template).read_text()
    for section in report_spec.SECTIONS:
        key = section["key"]
        rel = f"{Path(plots_dir).name}/{key}.png"
        text = text.replace(f"<!-- plot:{key} -->", f"![{section['title']}]({rel})")
        text = text.replace(f"<!-- table:{key} -->", report_bars.markdown_table(rows, section))
    leftover = [line for line in text.splitlines() if "<!-- plot:" in line or "<!-- table:" in line]
    if leftover:
        raise SystemExit(f"unknown markers in {template}: {leftover}")
    Path(out_path).write_text(text)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render report plots from fake benchmark output.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "plots")
    parser.add_argument("--platform", action="append", default=None, help="repeatable; default all three")
    parser.add_argument("--template", type=Path, default=Path(__file__).parent / "report.template.md")
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "report.md")
    args = parser.parse_args(argv)
    platforms = args.platform or report_spec.PLATFORMS

    rows = [row for platform in platforms for row in rows_for(platform)]
    written = write_fake_python_json(args.out / "_fake" / "python.json", platforms[0])
    print(f"load_python round-trip: {len(report_bars.load_python(written, platforms[0]))} rows")

    for section in report_spec.SECTIONS:
        out = report_bars.plot_section(rows, {**section, "platforms": platforms}, args.out)
        print(out if out else f"(no data for {section['key']})")

    if args.template.exists():
        # Tables carry the first platform only; the plots carry the rest.
        single = [row for row in rows if row["platform"] == platforms[0]]
        print(render_report(single, args.template, args.report, args.out))


if __name__ == "__main__":
    main()
