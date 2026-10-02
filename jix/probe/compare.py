#!/usr/bin/env python3
"""Compare analyze.py runs kernel by kernel, per platform and CPU.

Usage:
    python jix/probe/compare.py new=py-a base=py-b[,py-b-x86] [--out report.md]

Each `name=label[,label...]` is one configuration; several labels are merged (e.g. a baseline
build simulated on other CPUs with `--cpus`, plus the other platforms). Results are matched by
(platform, CPU, kernel); a run whose platform is `x86_64` but whose CPUs are another x86 platform's
(a build without multiversioning, which runs its SSE2 code everywhere) stands in for that platform.
The first configuration is compared with each of the others: for each platform and CPU, the
geomean speedup, the share of kernels faster / slower by more than 3%, the percentiles of the
per-kernel speedup, and the worst and best kernels. Kernels whose hot loop calls a function (whose
cost llvm-mca does not count) on either side are left out.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
# Platforms and their llvm-mca CPUs, newest first within each ISA (see analyze.py's PLATFORMS).
PLATFORMS = {
    "x86_64-v4": ["icelake-server", "sapphirerapids", "znver4"],
    "x86_64-v3": ["skylake", "alderlake", "znver3"],
    "x86_64-v2": ["sandybridge", "btver2"],
    "x86_64": ["sandybridge", "skylake", "znver3"],
    "aarch64-apple": ["apple-m1"],
    "aarch64": ["cortex-a72", "neoverse-n1", "neoverse-v2"],
    "i686": ["skylake"],
}
X86 = [p for p in PLATFORMS if p.startswith("x86_64")]
TOLERANCE = 0.03


def load(
    labels: list[str],
) -> tuple[dict[tuple[str, str, str], float], set[tuple[str, str]], dict[tuple[str, str], str]]:
    """(platform, cpu, kernel) -> cost; the (platform, kernel) with calls in the hot loop; and
    (platform, kernel) -> the label's results dir, for the gather count."""
    costs, calls, dirs = {}, set(), {}
    for label in labels:
        for r in json.loads((PROBE_DIR / "results" / label / "summary.json").read_text()):
            cpus = list(dict.fromkeys(m["cpu"] for m in r["mca"]))
            # A build for plain x86_64 simulated on other CPUs stands in for every x86 platform
            # of those CPUs.
            platforms = [r["platform"]]
            if r["platform"] == "x86_64":
                platforms = [p for p in X86 if set(PLATFORMS[p]) <= set(cpus)]
            for p in platforms:
                for cpu, c in zip(cpus, r["costs"]):
                    if cpu in PLATFORMS[p]:
                        costs[(p, cpu, r["kernel"])] = c
                if r["calls"]:
                    calls.add((p, r["kernel"]))
                dirs[(p, r["kernel"])] = str(PROBE_DIR / "results" / label / "mca" / r["platform"] / r["kernel"])
    return costs, calls, dirs


def gathers(prefix: str) -> int:
    """Gather instructions in the hot loop of a kernel's llvm-mca report(s)."""
    n = 0
    for f in Path(prefix).parent.glob(Path(prefix).name + ".*.txt"):
        text = f.read_text()
        listing = text.split("Instruction Info:")[-1]
        n = max(n, len(re.findall(r"\bv(?:p)?gather", listing)))
        break
    return n


def geo(xs: list[float]) -> float:
    return math.exp(sum(map(math.log, xs)) / len(xs))


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def compare(name: str, new: tuple, base_name: str, base: tuple) -> list[str]:
    (nc, ncalls, ndirs), (bc, bcalls, _) = new, base
    out = [f"## `{name}` vs `{base_name}`", ""]
    out += [
        (
            "Speedup = base cycles / new cycles (> 1: new is faster), per kernel, then geomean / percentiles"
            f" over the kernels measured on both sides without calls. faster / slower: > {TOLERANCE:.0%} apart."
        ),
        "",
        "| platform | CPU | kernels | geomean | faster | slower | p10 | median | p90 | min | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    details = []
    for p, cpus in PLATFORMS.items():
        for cpu in cpus:
            ks = sorted(
                k
                for (pp, cc, k) in nc
                if pp == p and cc == cpu and (p, cpu, k) in bc and (p, k) not in ncalls and (p, k) not in bcalls
            )
            if not ks:
                continue
            s = {k: bc[(p, cpu, k)] / nc[(p, cpu, k)] for k in ks}
            v = list(s.values())
            faster = sum(x > 1 + TOLERANCE for x in v) / len(v)
            slower = sum(x < 1 - TOLERANCE for x in v) / len(v)
            out.append(
                f"| {p} | {cpu} | {len(v)} | **{geo(v):.2f}x** | {faster:.0%} | {slower:.0%} | {pct(v, 0.1):.2f} "
                f"| {pct(v, 0.5):.2f} | {pct(v, 0.9):.2f} | {min(v):.2f} | {max(v):.2f} |"
            )
            if cpu == cpus[0]:
                worst = sorted(s.items(), key=lambda kv: kv[1])
                fmt = lambda kv: f"`{kv[0]}` {kv[1]:.2f}"
                line = f"- **{p}** ({cpu}): slowest " + ", ".join(map(fmt, worst[:8]))
                line += "; fastest " + ", ".join(map(fmt, worst[-5:][::-1]))
                if p == "x86_64-v4":
                    g = [k for k in ks if gathers(ndirs[(p, k)])]
                    line += f"; hot loops with gathers: {len(g)}" + (f" ({', '.join(g[:10])})" if g else "")
                details.append(line)
    return [*out, "", *details, ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="name=label[,label...]; the first is compared with the others")
    ap.add_argument("--out", help="also write the report to this file")
    args = ap.parse_args()
    configs = [(c.split("=")[0], load(c.split("=")[1].split(","))) for c in args.configs]
    (name, new), rest = configs[0], configs[1:]
    lines = [line for base_name, base in rest for line in compare(name, new, base_name, base)]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
