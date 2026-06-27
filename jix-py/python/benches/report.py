import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def load_pytest_json(path):
    """Return the list of benchmark entries from a pytest-benchmark --benchmark-json file."""
    return json.loads(Path(path).read_text())["benchmarks"]


def plot_throughput(benchmarks, out_dir):
    """One log-log PNG per `case`: x = array size, y = ops/sec (1/mean), one curve per library."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    for b in benchmarks:
        ei = b["extra_info"]
        groups[ei["case"]][ei["library"]].append((ei["size"], 1.0 / b["stats"]["mean"]))
    written = []
    for case, series in groups.items():
        fig, ax = plt.subplots()
        for library in sorted(series):
            pts = sorted(series[library])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=library)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("array size (rows)")
        ax.set_ylabel("operations/sec")
        ax.set_title(case)
        ax.legend()
        png = out_dir / f"{case}.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(png)
    return written


def write_ratio_table(benchmarks, out_dir, codec_desc):
    """Render the compression-ratio markdown table from compress-bench entries.

    Reads entries whose extra_info carries a `ratio` (recorded by test_compress). One row per
    case, one column per library; the largest measured size is used as the representative ratio
    (ratio is ~size-independent). Returns the written markdown path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_case = defaultdict(dict)  # case -> library -> (size, ratio)
    libraries = set()
    for b in benchmarks:
        ei = b["extra_info"]
        if "ratio" not in ei:
            continue
        case, lib, size = ei["case"], ei["library"], ei["size"]
        libraries.add(lib)
        cur = by_case[case].get(lib)
        if cur is None or size > cur[0]:
            by_case[case][lib] = (size, ei["ratio"])
    libraries = sorted(libraries)
    header = "| case | " + " | ".join(f"{lib} ratio" for lib in libraries) + " |"
    sep = "|" + "---|" * (1 + len(libraries))
    lines = [header, sep]
    for case in sorted(by_case):
        cells = [f"{by_case[case][lib][1]:.2f}" if lib in by_case[case] else "-" for lib in libraries]
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    md = out_dir / "compress_ratios.md"
    md.write_text(
        "# Compression ratio (raw/stored, higher is better)\n\n"
        f"Codec settings: {codec_desc}\n\n"
        "NumPy is the uncompressed baseline (ratio 1.00).\n\n" + "\n".join(lines) + "\n"
    )
    return md


def load_criterion_dir(root):
    """Walk criterion output for mean estimates. Rows: {group, bench, mean_ns}."""
    root = Path(root)
    rows = []
    for est in sorted(root.glob("**/new/estimates.json")):
        # .../<root>/<group>/<bench>/new/estimates.json
        bench_dir = est.parent.parent
        group = bench_dir.parent.name
        mean_ns = json.loads(est.read_text())["mean"]["point_estimate"]
        rows.append({"group": group, "bench": bench_dir.name, "mean_ns": mean_ns})
    return rows


def report_criterion(root, out_dir):
    """Render Rust criterion results: one bar PNG per group + a markdown summary table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_criterion_dir(root)
    written = []
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    for group, grows in by_group.items():
        grows = sorted(grows, key=lambda r: _natkey(r["bench"]))
        fig, ax = plt.subplots()
        ax.bar([r["bench"] for r in grows], [r["mean_ns"] for r in grows])
        ax.set_yscale("log")
        ax.set_ylabel("mean time (ns)")
        ax.set_title(f"rust: {group}")
        ax.tick_params(axis="x", rotation=90)
        png = out_dir / f"rust_{group}.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(png)
    lines = ["| group | bench | mean ns |", "|---|---|---|"]
    for r in sorted(rows, key=lambda r: (_natkey(r["group"]), _natkey(r["bench"]))):
        lines.append(f"| {r['group']} | {r['bench']} | {r['mean_ns']:.1f} |")
    md = out_dir / "rust_benchmarks.md"
    md.write_text("\n".join(lines) + "\n")
    written.append(md)
    return written


def _natkey(s):
    """Natural sort key: '20' before '100', 'a2' before 'a10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]
