"""Compare two jix benchmark runs (Rust Criterion + Python pytest-benchmark).

Read-side normalization: this tool parses the raw harness output uploaded by the
benchmarks workflow and computes Criterion-exact comparison statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BenchKey:
    suite: Literal["rust", "python"]
    group: str
    bench: str
    library: str


@dataclass
class BenchRecord:
    key: BenchKey
    samples: list[float]
    mean: float
    median: float
    stddev: float


type Records = dict[BenchKey, BenchRecord]
type Run = dict[str, tuple[dict, Records, Path]]  # platform -> (meta, records, result_dir)


def load_rust_records(criterion_dir: Path) -> list[BenchRecord]:
    """Parse a Criterion output tree into per-bench records (times converted to seconds)."""
    root = Path(criterion_dir)
    records = []
    for est_path in sorted(root.glob("**/new/estimates.json")):
        bench_dir = est_path.parent.parent  # <group>/<bench>/new/estimates.json -> <group>/<bench>
        group = bench_dir.parent.name
        bench = bench_dir.name
        est = json.loads(est_path.read_text())
        mean = est["mean"]["point_estimate"] / 1e9
        median = est["median"]["point_estimate"] / 1e9
        stddev = est.get("std_dev", {}).get("point_estimate", 0.0) / 1e9
        samples = []
        sample_path = est_path.parent / "sample.json"
        if sample_path.exists():
            s = json.loads(sample_path.read_text())
            samples = [t / it / 1e9 for it, t in zip(s["iters"], s["times"]) if it]
        records.append(BenchRecord(BenchKey("rust", group, bench, ""), samples, mean, median, stddev))
    return records


def load_python_records(python_json: Path) -> list[BenchRecord]:
    """Parse a pytest-benchmark json into per-bench records. Times are already seconds."""
    data = json.loads(Path(python_json).read_text())
    records = []
    for b in data.get("benchmarks", []):
        info = b.get("extra_info", {})
        stats = b.get("stats", {})
        if not all(k in info for k in ("case", "library", "size")):
            continue
        key = BenchKey("python", str(info["case"]), f"size={info['size']}", str(info["library"]))
        samples = [float(x) for x in stats.get("data", [])]
        records.append(
            BenchRecord(
                key,
                samples,
                float(stats.get("mean", 0.0)),
                float(stats.get("median", 0.0)),
                float(stats.get("stddev", 0.0)),
            )
        )
    return records


def find_artifact_dirs(run_dir: Path) -> list[Path]:
    """Return the result dirs under run_dir (each directly containing meta.json).

    A local run is a single result dir; a downloaded workflow run nests one per platform.
    """
    run_dir = Path(run_dir)
    if (run_dir / "meta.json").exists():
        return [run_dir]
    return sorted(p.parent for p in run_dir.glob("**/meta.json"))


def load_platform_records(result_dir: Path) -> tuple[dict, Records]:
    """Read one result dir (meta.json + rust/ + python/) -> (meta, {BenchKey: BenchRecord})."""
    result_dir = Path(result_dir)
    meta = json.loads((result_dir / "meta.json").read_text())
    records: Records = {}
    crit = result_dir / "rust" / "criterion"
    if crit.exists():
        for r in load_rust_records(crit):
            records[r.key] = r
    pj = result_dir / "python" / "python.json"
    if pj.exists():
        for r in load_python_records(pj):
            records[r.key] = r
    return meta, records


def load_run(run_dir: Path) -> Run:
    """Return platform -> (meta, {BenchKey: BenchRecord}, result_dir) for a run dir.

    result_dir is retained so callers can copy each platform's PNGs.
    """
    out: Run = {}
    for adir in find_artifact_dirs(run_dir):
        meta, records = load_platform_records(adir)
        plat = f"{meta['platform']['os']}-{meta['platform']['arch']}"
        out[plat] = (meta, records, adir)
    return out


DEFAULT_NRESAMPLES = 100_000
DEFAULT_SIGNIFICANCE = 0.05
DEFAULT_NOISE = 0.01
DEFAULT_CONFIDENCE = 0.95
_CHUNK = 8192  # bound bootstrap peak memory


@dataclass
class Comparison:
    key: BenchKey
    base_mean: float
    new_mean: float
    pct_change: float
    change_ci: tuple[float, float]
    p_value: float
    verdict: Literal["Improved", "Regressed", "WithinNoise", "NoChange"]


type PerPlatform = dict[str, tuple[list[Comparison], list[BenchKey], list[BenchKey]]]


def welch_t(base: np.ndarray, new: np.ndarray) -> float:
    """Two-sample Welch t-statistic (new vs base), matching Criterion's Sample::t."""
    b = np.asarray(base, dtype=float)
    n = np.asarray(new, dtype=float)
    denom = np.sqrt(n.var(ddof=1) / len(n) + b.var(ddof=1) / len(b))
    if denom == 0:
        return 0.0
    return float((n.mean() - b.mean()) / denom)


def bootstrap_p_value(base: np.ndarray, new: np.ndarray, nresamples: int, rng: np.random.Generator) -> float:
    """Two-tailed p-value via Criterion's pooled ("mixed") bootstrap of the t-statistic."""
    b = np.asarray(base, dtype=float)
    n = np.asarray(new, dtype=float)
    nn, nb = len(n), len(b)
    pooled = np.concatenate([n, b])
    t_obs = welch_t(b, n)
    if t_obs == 0.0:
        return 1.0
    extreme = 0
    finite = 0
    done = 0
    while done < nresamples:
        block = min(_CHUNK, nresamples - done)
        idx = rng.integers(0, nn + nb, size=(block, nn + nb))
        s = pooled[idx]
        a = s[:, :nn]
        c = s[:, nn:]
        denom = np.sqrt(a.var(axis=1, ddof=1) / nn + c.var(axis=1, ddof=1) / nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = (a.mean(axis=1) - c.mean(axis=1)) / denom
        t_star = t_star[np.isfinite(t_star)]
        extreme += int(np.count_nonzero(np.abs(t_star) >= abs(t_obs)))
        finite += int(t_star.size)
        done += block
    if finite == 0:
        return 1.0
    return extreme / finite


def change_estimates(
    base: np.ndarray, new: np.ndarray, nresamples: int, confidence: float, rng: np.random.Generator
) -> tuple[float, tuple[float, float]]:
    """Point estimate and CI of the mean relative change (new/base - 1).

    Uses an independent two-sample bootstrap (each sample resampled from its own data),
    matching Criterion's change-estimate procedure.
    """
    b = np.asarray(base, dtype=float)
    n = np.asarray(new, dtype=float)
    nn, nb = len(n), len(b)
    changes = np.empty(nresamples, dtype=float)
    done = 0
    while done < nresamples:
        block = min(_CHUNK, nresamples - done)
        mn = n[rng.integers(0, nn, size=(block, nn))].mean(axis=1)
        mb = b[rng.integers(0, nb, size=(block, nb))].mean(axis=1)
        changes[done : done + block] = mn / mb - 1.0
        done += block
    alpha = 1.0 - confidence
    lb = float(np.quantile(changes, alpha / 2.0))
    ub = float(np.quantile(changes, 1.0 - alpha / 2.0))
    point = float(n.mean() / b.mean() - 1.0)
    return point, (lb, ub)


def verdict(p_value: float, ci: tuple[float, float], significance: float, noise: float) -> str:
    """Criterion's decision: significance gate, then noise-threshold on the change CI."""
    if p_value >= significance:
        return "NoChange"
    lb, ub = ci
    if lb < -noise and ub < -noise:
        return "Improved"
    if lb > noise and ub > noise:
        return "Regressed"
    return "WithinNoise"


def compare_records(
    base: BenchRecord,
    new: BenchRecord,
    *,
    nresamples: int,
    significance: float,
    noise: float,
    confidence: float,
    rng: np.random.Generator,
) -> Comparison:
    """Compare two records; returns a Comparison keyed by the new record's key."""
    b = np.asarray(base.samples, dtype=float)
    n = np.asarray(new.samples, dtype=float)
    if len(b) < 2 or len(n) < 2:
        point = (new.mean / base.mean - 1.0) if base.mean else 0.0
        return Comparison(new.key, base.mean, new.mean, point, (point, point), 1.0, "NoChange")
    p = bootstrap_p_value(b, n, nresamples, rng)
    point, ci = change_estimates(b, n, nresamples, confidence, rng)
    return Comparison(new.key, base.mean, new.mean, point, ci, p, verdict(p, ci, significance, noise))


def _gh(args: list[str]) -> str:
    """Run a gh CLI command, returning stdout. Raises if gh is missing or errors."""
    try:
        return subprocess.check_output(["gh", *args], text=True, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise SystemExit("the 'gh' CLI is required for run-id/branch/sha resolution") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"gh {' '.join(args)} failed: {exc.stderr.strip()}") from exc


def list_runs(n: int, workflow: str = "benchmarks.yaml") -> list[dict]:
    """Return the newest n benchmark workflow runs as dicts (via gh run list --json)."""
    fields = "databaseId,headBranch,headSha,status,conclusion,createdAt,displayTitle"
    raw = _gh(["run", "list", "--workflow", workflow, "--limit", str(n), "--json", fields])
    return json.loads(raw)


def download_run(spec: str, cache_dir: Path, workflow: str = "benchmarks.yaml") -> Path:
    """Download a workflow run's artifacts (run-id | branch | sha) to a local dir. Separate from the
    main local-dir compare flow; only used by `compare --fetch`. A local dir is returned as-is."""
    p = Path(spec)
    if p.exists():
        return p
    cache_dir = Path(cache_dir)
    if spec.isdigit():
        run_id = spec
    else:
        fields = "databaseId,headSha,headBranch,conclusion"
        runs = json.loads(
            _gh(["run", "list", "--workflow", workflow, "--branch", spec, "--limit", "20", "--json", fields])
        )
        if not runs:
            # spec may be a sha rather than a branch: match on headSha prefix
            runs = json.loads(_gh(["run", "list", "--workflow", workflow, "--limit", "50", "--json", fields]))
            runs = [r for r in runs if r["headSha"].startswith(spec)]
        runs = [r for r in runs if r.get("conclusion") == "success"] or runs
        if not runs:
            raise SystemExit(f"no benchmark run found for {spec!r}")
        run_id = str(runs[0]["databaseId"])
    dest = cache_dir / run_id
    dest.mkdir(parents=True, exist_ok=True)
    _gh(["run", "download", run_id, "--dir", str(dest)])
    return dest


FAST_NRESAMPLES = 2000

_VERDICT_LABEL = {
    "Improved": "improved",
    "Regressed": "regressed",
    "WithinNoise": "within noise",
    "NoChange": "no change",
}


def _resolve_nresamples(nresamples: int | None, fast: bool) -> int:
    """Explicit --nresamples wins; otherwise pick the fast or full default."""
    if nresamples is not None:
        return nresamples
    return FAST_NRESAMPLES if fast else DEFAULT_NRESAMPLES


def compare_platform(
    base_records: Records, new_records: Records, *, only_library: str | None = None, **stats_opts
) -> tuple[list[Comparison], list[BenchKey], list[BenchKey]]:
    """Compare two platform record maps. Returns (comparisons, added_keys, removed_keys)."""
    comparisons, added, removed = [], [], []
    for key in sorted(set(base_records) | set(new_records), key=lambda k: (k.suite, k.group, k.bench, k.library)):
        if only_library and key.suite == "python" and key.library != only_library:
            continue
        if key not in base_records:
            added.append(key)
        elif key not in new_records:
            removed.append(key)
        else:
            comparisons.append(compare_records(base_records[key], new_records[key], **stats_opts))
    return comparisons, added, removed


def _severity(c: Comparison) -> tuple[int, float]:
    order = {"Regressed": 0, "WithinNoise": 1, "NoChange": 2, "Improved": 3}
    return (order[c.verdict], -c.pct_change)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _report_rows(comparisons: list[Comparison]) -> list[dict]:
    rows = []
    for c in sorted(comparisons, key=_severity):
        name = f"{c.key.suite}:{c.key.group}/{c.key.bench}" + (f" [{c.key.library}]" if c.key.library else "")
        rows.append(
            {
                "bench": name,
                "base_s": c.base_mean,
                "new_s": c.new_mean,
                "pct_change": c.pct_change,
                "p_value": c.p_value,
                "verdict": _VERDICT_LABEL[c.verdict],
            }
        )
    return rows


def _copy_run_pngs(run: Run, dest: Path):
    """Copy each platform's rust/python PNGs from a loaded run into dest/<platform>/."""
    dest = Path(dest)
    for plat, (_meta, _records, adir) in run.items():
        srcbase = adir
        for rel in ("rust/plots", "python"):
            src = srcbase / rel
            if not src.exists():
                continue
            target = dest / plat / rel
            target.mkdir(parents=True, exist_ok=True)
            for png in src.glob("*.png"):
                shutil.copy2(png, target / png.name)


_VERDICT_COLOR = {
    "Improved": "#2ca02c",
    "Regressed": "#d62728",
    "WithinNoise": "#7f7f7f",
    "NoChange": "#7f7f7f",
}


def _plt():
    """Lazily import matplotlib with the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bench_label(key: BenchKey) -> str:
    return f"{key.group}/{key.bench}" + (f" [{key.library}]" if key.library else "")


def plot_change_bars(comparisons: list[Comparison], platform: str, out_dir: Path) -> Path | None:
    """Horizontal per-bench %-change bars for one platform, colored by verdict, CI as error bars."""
    if not comparisons:
        return None
    plt = _plt()
    rows = sorted(comparisons, key=_severity)
    labels = [_bench_label(c.key) for c in rows]
    pct = [c.pct_change * 100 for c in rows]
    lo = [max(0.0, (c.pct_change - c.change_ci[0]) * 100) for c in rows]
    hi = [max(0.0, (c.change_ci[1] - c.pct_change) * 100) for c in rows]
    colors = [_VERDICT_COLOR[c.verdict] for c in rows]
    fig, ax = plt.subplots(figsize=(9, max(2.0, 0.28 * len(rows))))
    y = list(range(len(rows)))
    ax.barh(y, pct, xerr=[lo, hi], color=colors, error_kw={"elinewidth": 0.7})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()  # worst regression on top
    ax.set_xlabel("mean change (%), new vs base")
    ax.set_title(f"{platform}: per-bench change")
    png = Path(out_dir) / f"change_{platform}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_base_vs_new_scatter(comparisons: list[Comparison], platform: str, out_dir: Path) -> Path | None:
    """Log-log base-vs-new mean scatter, faceted by suite; diagonal = no change."""
    if not comparisons:
        return None
    plt = _plt()
    suites = sorted({c.key.suite for c in comparisons})
    fig, axes = plt.subplots(1, len(suites), figsize=(5 * len(suites), 4.5), squeeze=False)
    for ax, suite in zip(axes[0], suites):
        cs = [c for c in comparisons if c.key.suite == suite]
        xs = [c.base_mean for c in cs]
        ys = [c.new_mean for c in cs]
        ax.scatter(xs, ys, c=[_VERDICT_COLOR[c.verdict] for c in cs], s=18, alpha=0.8)
        lim_lo = min(xs + ys)
        lim_hi = max(xs + ys)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("base mean (s)")
        ax.set_ylabel("new mean (s)")
        ax.set_title(suite)
    fig.suptitle(f"{platform}: base vs new")
    png = Path(out_dir) / f"scatter_{platform}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_cross_platform_heatmap(per_platform: PerPlatform, out_dir: Path) -> Path | None:
    """benches (rows) x platforms (cols) heatmap of %change; red=slower, green=faster."""
    platforms = sorted(per_platform)
    keys = sorted(
        {c.key for comps, _a, _r in per_platform.values() for c in comps},
        key=lambda k: (k.suite, k.group, k.bench, k.library),
    )
    if not keys or not platforms:
        return None
    plt = _plt()
    m = np.full((len(keys), len(platforms)), np.nan)
    idx = {k: i for i, k in enumerate(keys)}
    for j, plat in enumerate(platforms):
        for c in per_platform[plat][0]:
            m[idx[c.key], j] = c.pct_change * 100
    vmax = float(np.nanmax(np.abs(m))) if np.isfinite(m).any() else 1.0
    vmax = vmax or 1.0
    fig, ax = plt.subplots(figsize=(max(4.0, 1.6 * len(platforms) + 3), max(2.5, 0.28 * len(keys))))
    im = ax.imshow(m, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(platforms)))
    ax.set_xticklabels(platforms, rotation=30, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k.suite}:{_bench_label(k)}" for k in keys], fontsize=7)
    fig.colorbar(im, ax=ax, label="mean change (%)")
    ax.set_title("cross-platform change")
    png = Path(out_dir) / "heatmap.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_sample_violins(
    base_records: Records, new_records: Records, platform: str, pattern: str, out_dir: Path
) -> list[Path]:
    """Overlay base vs new raw-sample distributions for benches whose label matches pattern."""
    plt = _plt()
    written = []
    keys = sorted(set(base_records) & set(new_records), key=lambda k: (k.suite, k.group, k.bench, k.library))
    for k in keys:
        if pattern not in f"{k.suite}:{_bench_label(k)}":
            continue
        b = base_records[k].samples
        n = new_records[k].samples
        if len(b) < 2 or len(n) < 2:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.violinplot([list(b), list(n)], showmeans=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["base", "new"])
        ax.set_ylabel("time (s)")
        ax.set_title(f"{platform}: {_bench_label(k)}")
        safe = f"violin_{platform}_{k.suite}_{k.group}_{k.bench}".replace("/", "_").replace(" ", "_")
        png = Path(out_dir) / f"{safe}.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(png)
    return written


def write_plots(
    out_dir: Path, per_platform: PerPlatform, base_run: Run, new_run: Run, bench_pattern: str | None = None
) -> list[Path]:
    """Generate all comparison plots into out_dir/plots/. Returns the written paths."""
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for plat, (comps, _a, _r) in per_platform.items():
        written.append(plot_change_bars(comps, plat, plots_dir))
        written.append(plot_base_vs_new_scatter(comps, plat, plots_dir))
        if bench_pattern and plat in base_run and plat in new_run:
            _, base_recs, _ = base_run[plat]
            _, new_recs, _ = new_run[plat]
            written += plot_sample_violins(base_recs, new_recs, plat, bench_pattern, plots_dir)
    written.append(plot_cross_platform_heatmap(per_platform, plots_dir))
    return [p for p in written if p]


def write_report(out_dir: Path, per_platform: PerPlatform, base_run: Run, new_run: Run, fmt: str):
    """Write report.md + report.json into out_dir. per_platform: plat -> (comparisons, added, removed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"platforms": {}}
    md = ["# Benchmark comparison", ""]
    for plat, (comparisons, added, removed) in sorted(per_platform.items()):
        rows = _report_rows(comparisons)
        counts = {v: 0 for v in ("regressed", "improved", "within noise", "no change")}
        for r in rows:
            counts[r["verdict"]] += 1
        doc["platforms"][plat] = {
            "summary": counts,
            "added": [f"{k.suite}:{k.group}/{k.bench}" for k in added],
            "removed": [f"{k.suite}:{k.group}/{k.bench}" for k in removed],
            "rows": rows,
        }
        base_meta = base_run.get(plat, ({}, {}, None))[0].get("platform", {})
        new_meta = new_run.get(plat, ({}, {}, None))[0].get("platform", {})
        md.append(f"## {plat}")
        if base_meta.get("cpu_model") != new_meta.get("cpu_model"):
            md.append(f"> WARNING: cpu mismatch: base={base_meta.get('cpu_model')} new={new_meta.get('cpu_model')}")
        md.append(
            f"regressed {counts['regressed']} | improved {counts['improved']} | "
            f"within noise {counts['within noise']} | no change {counts['no change']} | "
            f"added {len(added)} | removed {len(removed)}"
        )
        md.append("")
        md.append("| bench | base | new | change | p | verdict |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            md.append(
                f"| {r['bench']} | {r['base_s'] * 1e9:.1f} ns | {r['new_s'] * 1e9:.1f} ns | "
                f"{_fmt_pct(r['pct_change'])} | {r['p_value']:.3f} | {r['verdict']} |"
            )
        md.append("")
        links = []
        for name, caption in ((f"change_{plat}.png", "change bars"), (f"scatter_{plat}.png", "base vs new")):
            if (out_dir / "plots" / name).exists():
                links.append(f"[{caption}](plots/{name})")
        for violin in sorted((out_dir / "plots").glob(f"violin_{plat}_*.png")):
            links.append(f"[{violin.stem}](plots/{violin.name})")
        if links:
            md.append("Plots: " + " | ".join(links))
            md.append("")
    if (out_dir / "plots" / "heatmap.png").exists():
        md.append("## cross-platform")
        md.append("![heatmap](plots/heatmap.png)")
        md.append("")
    (out_dir / "report.json").write_text(json.dumps(doc, indent=2))
    (out_dir / "report.md").write_text("\n".join(md) + "\n")


def _print_human(per_platform: PerPlatform, base_run: Run, new_run: Run):
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def paint(text: str, verdict: str) -> str:
        if not color:
            return text
        code = {"regressed": "31", "improved": "32"}.get(verdict, "0")
        return f"\033[{code}m{text}\033[0m"

    for plat, (comparisons, added, removed) in sorted(per_platform.items()):
        print(f"\n== {plat} ==")
        base_cpu = base_run.get(plat, ({}, {}, None))[0].get("platform", {}).get("cpu_model")
        new_cpu = new_run.get(plat, ({}, {}, None))[0].get("platform", {}).get("cpu_model")
        if base_cpu != new_cpu:
            print(paint(f"WARNING: cpu mismatch base={base_cpu} new={new_cpu}", "regressed"))
        for r in _report_rows(comparisons):
            line = f"{r['bench']:<50} {_fmt_pct(r['pct_change']):>8}  p={r['p_value']:.3f}  {r['verdict']}"
            print(paint(line, r["verdict"]))
        if added or removed:
            print(f"  (+{len(added)} added, -{len(removed)} removed)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two jix benchmark runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("list", help="list recent benchmark workflow runs (via gh)")
    lp.add_argument("-n", type=int, default=20)
    cp = sub.add_parser("compare", help="compare two local result dirs (base then new)")
    cp.add_argument("base", help="base result dir (or run-id | branch | sha with --fetch)")
    cp.add_argument("new", help="new result dir (or run-id | branch | sha with --fetch)")
    cp.add_argument("--fetch", action="store_true", help="treat base/new as workflow runs and download them first")
    cp.add_argument("--out", type=Path, default=None)
    cp.add_argument("--only", default=None, help="restrict python rows to this library (e.g. jix)")
    cp.add_argument("--format", choices=["human", "json", "markdown"], default="human")
    cp.add_argument("--fast", action="store_true", help="quick comparison (fewer bootstrap resamples)")
    cp.add_argument("--nresamples", type=int, default=None, help="bootstrap resamples (default 100000)")
    cp.add_argument("--significance", type=float, default=DEFAULT_SIGNIFICANCE)
    cp.add_argument("--noise-threshold", type=float, default=DEFAULT_NOISE)
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--no-plots", action="store_true", help="skip generating comparison plots")
    cp.add_argument("--bench", default=None, help="substring: also emit per-bench sample violins for matches")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        for r in list_runs(args.n):
            print(
                f"{r['createdAt']}  {r['databaseId']!s:>12}  {r['conclusion'] or r['status']:<10} "
                f"{r['headBranch']:<20} {r['headSha'][:8]}  {r['displayTitle']}"
            )
        return 0

    # Separate remote pre-step: resolve workflow runs to local dirs before the main flow.
    base_dir, new_dir = Path(args.base), Path(args.new)
    if args.fetch:
        cache = Path(".bench-cache")
        base_dir = download_run(args.base, cache)
        new_dir = download_run(args.new, cache)

    # Main flow: compare two local result dirs, matched per platform.
    stats_opts = dict(  # noqa: C408
        nresamples=_resolve_nresamples(args.nresamples, args.fast),
        significance=args.significance,
        noise=args.noise_threshold,
        confidence=DEFAULT_CONFIDENCE,
        rng=np.random.default_rng(args.seed),
    )
    base_run = load_run(base_dir)
    new_run = load_run(new_dir)
    label = f"{base_dir.name}-vs-{new_dir.name}"

    per_platform: PerPlatform = {}
    for plat in sorted(set(base_run) & set(new_run)):
        _, base_recs, _ = base_run[plat]
        _, new_recs, _ = new_run[plat]
        per_platform[plat] = compare_platform(base_recs, new_recs, only_library=args.only, **stats_opts)

    out_dir = args.out or Path(f"bench-compare-{label}".replace("/", "_"))
    if not args.no_plots:
        write_plots(out_dir, per_platform, base_run, new_run, bench_pattern=args.bench)
    write_report(out_dir, per_platform, base_run, new_run, args.format)
    _copy_run_pngs(base_run, out_dir / "base")
    _copy_run_pngs(new_run, out_dir / "new")
    if args.format == "human":
        _print_human(per_platform, base_run, new_run)
    elif args.format == "json":
        print((out_dir / "report.json").read_text())
    else:
        print((out_dir / "report.md").read_text())
    print(f"\nreport written to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
