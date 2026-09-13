"""Build the report from downloaded benchmark artifacts.

Reads one or more artifact directories - whatever `gh run download` produced - and writes the
plots and `report.md`. Runs no benchmarks.

    python bench-report/build_report.py --artifacts ./bench-artifacts

Each artifact directory holds one `<sha>/` per benched ref, containing `meta.json`, `python/`, and
`rust/criterion/`. The platform each one ran on comes from `meta.json`, so the artifacts can be
passed in any order.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "jix-py" / "python"))

from benches import report_bars, report_spec  # noqa: E402


def load_run(run_dir):
    """Rows from one benched sha: its Python JSON, its Criterion tree, or both."""
    meta = json.loads((run_dir / "meta.json").read_text())
    platform = f"{meta['platform']['os']}-{meta['platform']['arch']}"
    rows = []
    python_json = run_dir / "python" / "python.json"
    if python_json.exists():
        rows += report_bars.load_python(python_json, platform)
    criterion = run_dir / "rust" / "criterion"
    if criterion.exists():
        rows += report_bars.load_rust(criterion, platform, report_spec.RUST_CASE_LABELS)
    return platform, rows, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the benchmark report from downloaded artifacts.")
    parser.add_argument("--artifacts", type=Path, required=True, help="directory holding downloaded artifacts")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "plots")
    parser.add_argument("--template", type=Path, default=Path(__file__).parent / "report.template.md")
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "report.md")
    args = parser.parse_args(argv)

    runs = sorted(args.artifacts.glob("**/meta.json"))
    if not runs:
        raise SystemExit(f"no meta.json under {args.artifacts} - is that the download directory?")

    rows, platforms = [], []
    for meta_path in runs:
        platform, run_rows, meta = load_run(meta_path.parent)
        if not run_rows:
            print(f"warning: {meta_path.parent} has no results, skipping")
            continue
        platforms.append(platform)
        rows += run_rows
        print(f"{platform}: {len(run_rows)} rows from {meta['sha'][:8]} ({meta['platform']['cpu_model']})")

    # Published reports come from the two CI runners, in that order. A local run is on whatever
    # machine you are sitting at, so fall back to plotting what is actually there.
    ordered = [p for p in report_spec.PLATFORMS if p in platforms]
    extra = sorted(set(platforms) - set(ordered))
    if extra:
        print(f"note: results from platforms the report does not publish: {extra}")
    ordered = ordered or extra
    if not ordered:
        raise SystemExit("no usable results found")

    for section in report_spec.SECTIONS:
        out = report_bars.plot_section(rows, {**section, "platforms": ordered}, args.out)
        print(out if out else f"(no data for {section['key']})")

    # Tables carry the first platform only; the plots carry the rest.
    single = [row for row in rows if row["platform"] == ordered[0]]
    print(report_bars.render_report(single, args.template, args.report, args.out, report_spec.SECTIONS))


if __name__ == "__main__":
    main()
