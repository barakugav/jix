"""Compare two `analyze.py` outputs kernel by kernel, per platform and CPU.

Usage:
    python scripts/analyze_llvm_mca/compare.py <new out dir> <base out dir> [--fn 'reduce_*' ...] [--kernels]

Speedup = base cycles / new cycles (> 1: new is faster). For each platform and CPU: the geomean
speedup over the kernels measured on both sides, the share faster / slower by more than 3%, and
the slowest and fastest kernels. `--kernels` adds a table of every kernel's speedups. Kernels whose
hot loop calls a function (free in llvm-mca) on either side are left out.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path

TOLERANCE = 0.03


def load(out_dir: Path) -> tuple[dict[tuple[str, str, str], float], set[tuple[str, str]]]:
    """(platform, cpu, kernel) -> cycles, and the (platform, kernel) with calls in the hot loop."""
    costs, calls = {}, set()
    for r in json.loads((out_dir / "summary.json").read_text()):
        for cpu, c in r["costs"].items():
            costs[(r["platform"], cpu, r["kernel"])] = c
        if r["calls"]:
            calls.add((r["platform"], r["kernel"]))
    return costs, calls


def geomean(xs: list[float]) -> float:
    return math.exp(sum(map(math.log, xs)) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new", type=Path)
    ap.add_argument("base", type=Path)
    ap.add_argument("--fn", action="append", dest="kernels", help="only the kernels matching these globs (repeatable)")
    ap.add_argument("--kernels", action="store_true", dest="table", help="also a table of every kernel's speedups")
    args = ap.parse_args()
    (new, new_calls), (base, base_calls) = load(args.new), load(args.base)

    speedups: dict[tuple[str, str], dict[str, float]] = {}  # (platform, cpu) -> kernel -> speedup
    for (p, cpu, k), c in new.items():
        if (
            (p, cpu, k) in base
            and (p, k) not in new_calls | base_calls
            and (not args.kernels or any(fnmatch.fnmatchcase(k, pat) for pat in args.kernels))
        ):
            speedups.setdefault((p, cpu), {})[k] = base[(p, cpu, k)] / c
    if not speedups:
        raise SystemExit("error: no kernel measured on both sides")

    lines = [
        f"## `{args.new.name}` vs `{args.base.name}`",
        "",
        f"Speedup = base cycles / new cycles (> 1: new is faster). faster / slower: by more than {TOLERANCE:.0%}.",
        "",
        "| platform | CPU | kernels | geomean | faster | slower | slowest | fastest |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for (p, cpu), s in speedups.items():
        v = list(s.values())
        ranked = sorted(s.items(), key=lambda kv: kv[1])
        faster, slower = sum(x > 1 + TOLERANCE for x in v) / len(v), sum(x < 1 - TOLERANCE for x in v) / len(v)
        slowest = ", ".join(f"{k} {x:.2f}" for k, x in ranked[:3])
        fastest = ", ".join(f"{k} {x:.2f}" for k, x in ranked[::-1][:3])
        lines.append(
            f"| {p} | {cpu} | {len(v)} | **{geomean(v):.2f}** | {faster:.0%} | {slower:.0%} | {slowest} | {fastest} |"
        )
    if args.table:
        cols = list(speedups)
        lines += ["", "| kernel | " + " | ".join(f"{p} {cpu}" for p, cpu in cols) + " |", "|---|" + "---:|" * len(cols)]
        for k in sorted({k for s in speedups.values() for k in s}):
            lines.append(
                f"| {k} | " + " | ".join(f"{speedups[c][k]:.2f}" if k in speedups[c] else "-" for c in cols) + " |"
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
