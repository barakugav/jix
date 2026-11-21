"""Render Gungraun `--output-format=json` output as a markdown table for the CI job summary."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

METRICS = ["Ir", "EstimatedCycles", "L1hits", "LLhits", "RamHits"]


@dataclass
class Cell:
    new: float
    old: float | None
    diff_pct: float | None


@dataclass
class Row:
    name: str
    metrics: dict[str, Cell]


def load_rows(path: Path) -> list[Row]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        bench = json.loads(line)
        metrics = {}
        for kind, result in bench["profiles"][0]["data"]["total"]["metrics"].items():
            values = result["values"]
            change = result.get("change")
            metrics[kind] = Cell(
                new=float(values["new"]),
                old=float(values["old"]) if "old" in values else None,
                diff_pct=float(change["diff_pct"]) if change else None,
            )
        rows.append(Row(bench["function_name"], metrics))
    return sorted(rows, key=lambda r: r.name)


def render_markdown(rows: list[Row], threshold: float) -> str:
    metrics = [m for m in METRICS if any(m in r.metrics for r in rows)]

    # Only Ir is marked: cache hits on small benches move by a few counts whenever code moves.
    def fmt(cell: Cell | None, marked: bool):
        if cell is None:
            return "-"
        if cell.diff_pct is None:
            return f"{cell.new:,.0f} (-)"
        pct = f"{cell.diff_pct:+.2f}%"
        return f"{cell.new:,.0f} ({f'**{pct}**' if marked and abs(cell.diff_pct) > threshold else pct})"

    changed = sum(1 for r in rows if "Ir" in r.metrics and abs(r.metrics["Ir"].diff_pct or 0) > threshold)
    lines = [
        f"{len(rows)} benchmarks, {changed} changed past +-{threshold:g}% (Ir)",
        "",
        "| bench | " + " | ".join(metrics) + " |",
        "|---" * (len(metrics) + 1) + "|",
    ]
    lines += [f"| {r.name} | " + " | ".join(fmt(r.metrics.get(m), m == "Ir") for m in metrics) + " |" for r in rows]
    lines += ["", "Counts are from Valgrind (Callgrind) via Gungraun. This job is report-only."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Gungraun JSON-lines output")
    parser.add_argument("--threshold", type=float, default=2.0, help="mark changes past this many percent")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    if not rows:
        sys.exit(f"no benchmarks in {args.summary}")
    print(render_markdown(rows, args.threshold), end="")


if __name__ == "__main__":
    main()
