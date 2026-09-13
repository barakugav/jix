"""Generate fake benchmark output and render the report plots from it.

Exists so the plotting code can be developed and its style iterated on without waiting for a real
benchmark run. It writes a `python.json` in pytest-benchmark's shape and a Criterion-shaped tree,
then drives the real renderer over them - so anything that works here works on real results.

    python bench-report/build_fake_report.py --out bench-report/plots

Numbers marked below as measured come from a real local run; the rest are invented and plausible.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "jix-py" / "python"))

from benches import report_bars, report_spec  # noqa: E402

US, MS = 1e-6, 1e-3

# (section, case) -> {library: value}. Times in seconds, ratios unitless, memory in bytes.
FAKE = {
    ("read", "b16x200\nr1x200"): {
        "numpy": 0.4 * US,
        "jix": 1.4 * US,
        "jix-shuffle": 1.6 * US,
        "blosc2": 78 * US,
        "blosc2-chunked": 34 * US,
        "zarr": 330 * US,
    },
    ("read", "b16x200\nr16x200"): {
        "numpy": 0.6 * US,
        "jix": 2.4 * US,
        "jix-shuffle": 2.6 * US,
        "blosc2": 82 * US,
        "blosc2-chunked": 36 * US,
        "zarr": 340 * US,
    },
    ("read", "b64x200\nr16x16"): {
        "numpy": 0.35 * US,
        "jix": 3.2 * US,
        "jix-shuffle": 3.6 * US,
        "blosc2": 54 * US,
        "blosc2-chunked": 42 * US,
        "zarr": 336 * US,
    },
    ("read", "b64x200\nr256x200"): {
        "numpy": 12 * US,
        "jix": 28 * US,
        "jix-shuffle": 26 * US,
        "blosc2": 130 * US,
        "blosc2-chunked": 95 * US,
        "zarr": 420 * US,
    },
    ("read", "b64x200\nr4096x200"): {
        "numpy": 190 * US,
        "jix": 420 * US,
        "jix-shuffle": 380 * US,
        "blosc2": 900 * US,
        "blosc2-chunked": 780 * US,
        "zarr": 1200 * US,
    },
    ("read", "b64x200\nfull"): {
        "numpy": 5.2 * MS,
        "jix": 14 * MS,
        "jix-shuffle": 12 * MS,
        "blosc2": 24 * MS,
        "blosc2-chunked": 22 * MS,
        "zarr": 40 * MS,
    },
    ("compress_ratio", "random"): {
        "numpy": 1.0,
        "jix": 1.00,
        "jix-shuffle": 1.00,
        "blosc2": 1.00,
        "blosc2-shuffle": 1.00,
        "zarr": 1.00,
    },
    ("compress_ratio", "smooth"): {
        "numpy": 1.0,
        "jix": 3.55,
        "jix-shuffle": 3.94,
        "blosc2": 3.40,
        "blosc2-shuffle": 3.71,
        "zarr": 3.71,
    },
    ("compress_ratio", "16 unique"): {
        "numpy": 1.0,
        "jix": 24.0,
        "jix-shuffle": 31.0,
        "blosc2": 22.5,
        "blosc2-shuffle": 28.4,
        "zarr": 28.4,
    },
    ("compress_ratio", "4 unique"): {
        "numpy": 1.0,
        "jix": 46.0,
        "jix-shuffle": 55.2,
        "blosc2": 41.0,
        "blosc2-shuffle": 48.6,
        "zarr": 48.6,
    },
    ("compress", "random"): {
        "numpy": 5.2 * MS,
        "jix": 133 * MS,
        "jix-shuffle": 190 * MS,
        "blosc2": 128 * MS,
        "blosc2-shuffle": 196 * MS,
        "zarr": 540 * MS,
    },
    ("compress", "smooth"): {
        "numpy": 5.2 * MS,
        "jix": 163 * MS,
        "jix-shuffle": 133 * MS,
        "blosc2": 176 * MS,
        "blosc2-shuffle": 147 * MS,
        "zarr": 610 * MS,
    },
    ("compress", "16 unique"): {
        "numpy": 5.2 * MS,
        "jix": 88 * MS,
        "jix-shuffle": 74 * MS,
        "blosc2": 102 * MS,
        "blosc2-shuffle": 88 * MS,
        "zarr": 420 * MS,
    },
    ("compress", "4 unique"): {
        "numpy": 5.2 * MS,
        "jix": 71 * MS,
        "jix-shuffle": 62 * MS,
        "blosc2": 84 * MS,
        "blosc2-shuffle": 72 * MS,
        "zarr": 390 * MS,
    },
    # negate/add f32 numbers are measured; the i32 arm does not exist yet.
    ("negate", "f32"): {
        "numpy": 2.157 * MS,
        "jix-plain": 2.186 * MS,
        "jix": 69.5 * MS,
        "jix-shuffle": 49.6 * MS,
        "blosc2": 470 * MS,
        "blosc2-shuffle": 443 * MS,
        "zarr": 30.8 * MS,
    },
    ("negate", "i32"): {
        "numpy": 2.21 * MS,
        "jix-plain": 2.25 * MS,
        "jix": 62.0 * MS,
        "jix-shuffle": 45.0 * MS,
        "blosc2": 480 * MS,
        "blosc2-shuffle": 420 * MS,
        "zarr": 30.0 * MS,
    },
    ("add", "f32"): {
        "numpy": 2.886 * MS,
        "jix-plain": 2.882 * MS,
        "jix": 136 * MS,
        "jix-shuffle": 99.1 * MS,
        "blosc2": 626 * MS,
        "blosc2-shuffle": 561 * MS,
        "zarr": 59.2 * MS,
    },
    ("add", "i32"): {
        "numpy": 2.95 * MS,
        "jix-plain": 2.94 * MS,
        "jix": 128 * MS,
        "jix-shuffle": 94 * MS,
        "blosc2": 610 * MS,
        "blosc2-shuffle": 540 * MS,
        "zarr": 58 * MS,
    },
    ("negate_dist", "random"): {"numpy": 2.157 * MS, "jix-shuffle": 96.4 * MS, "blosc2-shuffle": 510 * MS},
    ("negate_dist", "smooth"): {"numpy": 2.157 * MS, "jix-shuffle": 49.6 * MS, "blosc2-shuffle": 443 * MS},
    ("negate_dist", "16 unique"): {"numpy": 2.157 * MS, "jix-shuffle": 34.5 * MS, "blosc2-shuffle": 210 * MS},
    ("negate_dist", "4 unique"): {"numpy": 2.157 * MS, "jix-shuffle": 21.3 * MS, "blosc2-shuffle": 155 * MS},
    # every reduction number here is measured
    ("reduction", "sum f32\naxis 0"): {
        "numpy": 2.33 * MS,
        "jix-plain": 2.03 * MS,
        "jix-shuffle": 49.6 * MS,
        "blosc2-shuffle": 34.7 * MS,
        "zarr": 30.9 * MS,
    },
    ("reduction", "sum f32\naxis 1"): {
        "numpy": 3.97 * MS,
        "jix-plain": 1.55 * MS,
        "jix-shuffle": 49.0 * MS,
        "blosc2-shuffle": 36.3 * MS,
        "zarr": 32.9 * MS,
    },
    ("reduction", "sum f32\nall"): {
        "numpy": 3.21 * MS,
        "jix-plain": 1.31 * MS,
        "jix-shuffle": 48.6 * MS,
        "blosc2-shuffle": 39.6 * MS,
        "zarr": 31.5 * MS,
    },
    ("reduction", "sum i32\naxis 0"): {
        "numpy": 11.70 * MS,
        "jix-plain": 3.12 * MS,
        "jix-shuffle": 22.0 * MS,
        "blosc2-shuffle": 20.4 * MS,
        "zarr": 37.1 * MS,
    },
    ("reduction", "sum i32\nall"): {
        "numpy": 4.39 * MS,
        "jix-plain": 1.36 * MS,
        "jix-shuffle": 18.6 * MS,
        "blosc2-shuffle": 14.2 * MS,
        "zarr": 29.6 * MS,
    },
    ("reduction", "std f32\naxis 0"): {
        "numpy": 11.06 * MS,
        "jix-plain": 14.44 * MS,
        "jix-shuffle": 61.7 * MS,
        "blosc2-shuffle": 72.9 * MS,
        "zarr": 41.9 * MS,
    },
    ("reduction", "std f32\nall"): {
        "numpy": 10.74 * MS,
        "jix-plain": 34.81 * MS,
        "jix-shuffle": 82.4 * MS,
        "blosc2-shuffle": 74.0 * MS,
        "zarr": 39.1 * MS,
    },
    ("chain", "f32\n1 op"): {"numpy": 2.9 * MS, "jix-plain": 2.9 * MS, "jix-shuffle": 50 * MS},
    ("chain", "f32\n2 ops"): {"numpy": 6.1 * MS, "jix-plain": 3.4 * MS, "jix-shuffle": 51 * MS},
    ("chain", "f32\n4 ops"): {"numpy": 12.8 * MS, "jix-plain": 4.6 * MS, "jix-shuffle": 53 * MS},
    ("chain", "f32\n8 ops"): {"numpy": 26.0 * MS, "jix-plain": 7.1 * MS, "jix-shuffle": 57 * MS},
    # measured: the transcendental chain is the one jix loses
    ("chain", "f32\nexp/log"): {"numpy": 80.9 * MS, "jix-plain": 90.9 * MS, "jix-shuffle": 138 * MS},
    ("chain", "i32\n1 op"): {"numpy": 3.0 * MS, "jix-plain": 3.0 * MS, "jix-shuffle": 46 * MS},
    ("chain", "i32\n4 ops"): {"numpy": 13.1 * MS, "jix-plain": 4.8 * MS, "jix-shuffle": 49 * MS},
    ("chain", "i32\n8 ops"): {"numpy": 26.5 * MS, "jix-plain": 7.3 * MS, "jix-shuffle": 52 * MS},
    ("chain_memory", "f32\n1 op"): {"numpy": 218e6, "jix-plain": 215e6, "jix-shuffle": 135e6},
    ("chain_memory", "f32\n2 ops"): {"numpy": 322e6, "jix-plain": 215e6, "jix-shuffle": 135e6},
    ("chain_memory", "f32\n4 ops"): {"numpy": 428e6, "jix-plain": 215e6, "jix-shuffle": 135e6},
    ("chain_memory", "f32\n8 ops"): {"numpy": 430e6, "jix-plain": 215e6, "jix-shuffle": 135e6},
    ("rust_read", "b32x32\nr32x32"): {"ndarray": 0.28 * US, "jix": 1.4 * US, "jix-shuffle": 1.6 * US},
    ("rust_read", "b32x32\nr1x200"): {"ndarray": 0.35 * US, "jix": 9.8 * US, "jix-shuffle": 10.5 * US},
    ("rust_read", "b512x32\nr128x200"): {"ndarray": 11 * US, "jix": 46 * US, "jix-shuffle": 42 * US},
    ("rust_op", "negate\nf32"): {"ndarray": 11.0 * MS, "jix-plain": 11.2 * MS, "jix-shuffle": 49.5 * MS},
    ("rust_op", "negate\ni32"): {"ndarray": 10.9 * MS, "jix-plain": 11.1 * MS, "jix-shuffle": 45.0 * MS},
    ("rust_op", "add\nf32"): {"ndarray": 16.1 * MS, "jix-plain": 16.4 * MS, "jix-shuffle": 99.0 * MS},
    ("rust_op", "add\ni32"): {"ndarray": 16.0 * MS, "jix-plain": 16.3 * MS, "jix-shuffle": 95.0 * MS},
    ("rust_op", "sum f32\naxis 0"): {"ndarray": 7.1 * MS, "jix-plain": 5.4 * MS, "jix-shuffle": 49.0 * MS},
    ("rust_op", "sum f32\naxis 1"): {"ndarray": 5.2 * MS, "jix-plain": 4.9 * MS, "jix-shuffle": 48.0 * MS},
    ("rust_op", "sum i32\nall"): {"ndarray": 6.8 * MS, "jix-plain": 2.8 * MS, "jix-shuffle": 19.0 * MS},
    ("rust_chain", "1 op"): {"ndarray": 11.0 * MS, "jix-plain": 11.2 * MS},
    ("rust_chain", "2 ops"): {"ndarray": 22.4 * MS, "jix-plain": 12.1 * MS},
    ("rust_chain", "4 ops"): {"ndarray": 45.1 * MS, "jix-plain": 13.8 * MS},
    ("rust_chain", "8 ops"): {"ndarray": 90.3 * MS, "jix-plain": 17.2 * MS},
    ("rust_chain", "normalize\naxis 0"): {"ndarray": 96 * MS, "jix-plain": 38 * MS},
    ("rust_chain", "normalize\naxis 1"): {"ndarray": 104 * MS, "jix-plain": 41 * MS},
    ("rust_axis_order", "3-D rotate\n[1,2,0] f32"): {"ndarray": 88 * MS, "jix-plain": 12 * MS},
    ("rust_axis_order", "3-D rotate\n[1,2,0] i32"): {"ndarray": 86 * MS, "jix-plain": 11.8 * MS},
    ("rust_axis_order", "3-D reverse\n[2,1,0] f32"): {"ndarray": 12.0 * MS, "jix-plain": 12.2 * MS},
    ("rust_axis_order", "2-D\ntranspose f32"): {"ndarray": 2.0 * MS, "jix-plain": 1.9 * MS},
}

# Per-platform multipliers applied to jix only, so the fake data exercises the case the report has
# to handle honestly: a win on one platform that shrinks or inverts on another. The integer
# reduction is the live example - jix is tuned on arm64, numpy has had far more x86 attention.
PLATFORM_TWEAKS = {
    "macos-aarch64": {},
    "linux-x86_64": {("reduction", "sum i32\nall"): 2.9, ("reduction", "sum i32\naxis 0"): 2.6},
    "linux-aarch64": {("reduction", "sum i32\nall"): 1.15, ("reduction", "sum i32\naxis 0"): 1.1},
}
PLATFORM_SCALE = {"macos-aarch64": 1.0, "linux-x86_64": 1.18, "linux-aarch64": 1.32}


def rows_for(platform):
    """Fake normalized rows for one platform, in the shape the renderer consumes."""
    scale = PLATFORM_SCALE[platform]
    tweaks = PLATFORM_TWEAKS[platform]
    rows = []
    for (section, case), values in FAKE.items():
        metric = report_spec.BY_KEY[section].get("metric", "time")
        for library, value in values.items():
            if metric == "time":
                value *= scale
                if library.startswith("jix"):
                    value *= tweaks.get((section, case), 1.0)
            rows.append({"platform": platform, "section": section, "case": case, "library": library, "value": value})
    return rows


def write_fake_python_json(path, platform):
    """Write a pytest-benchmark shaped JSON, so `report_bars.load_python` can be exercised."""
    benchmarks = []
    for row in rows_for(platform):
        if row["section"].startswith("rust_") or row["section"].endswith(("_ratio", "_memory")):
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
        rel = Path(plots_dir).name + f"/{key}.png"
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
    # Exercise the real loader on a real file too, so its parsing stays honest.
    written = write_fake_python_json(args.out / "_fake" / "python.json", platforms[0])
    loaded = report_bars.load_python(written, platforms[0])
    print(f"load_python round-trip: {len(loaded)} rows from {written}")

    for section in report_spec.SECTIONS:
        section = {**section, "platforms": platforms}
        out = report_bars.plot_section(rows, section, args.out)
        print(out if out else f"(no data for {section['key']})")

    if args.template.exists():
        # Tables are rendered for the first platform only; the plots carry the rest.
        single = [row for row in rows if row["platform"] == platforms[0]]
        print(render_report(single, args.template, args.report, args.out))


if __name__ == "__main__":
    main()
