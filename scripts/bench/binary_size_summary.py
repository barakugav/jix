"""Render the size of the jix-py shared library on the base and head commits as a markdown table for the CI report."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [".text", ".rodata", ".data.rel.ro", ".eh_frame"]


def sizes(path: Path) -> dict[str, int]:
    # `size -A -d` prints one "<section> <size> <addr>" line per section.
    out = subprocess.check_output(["size", "-A", "-d", path], text=True)
    sections = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit():
            sections[parts[0]] = int(parts[1])
    return {"file": path.stat().st_size} | {s: sections.get(s, 0) for s in SECTIONS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="shared library built on the base commit")
    parser.add_argument("head", type=Path, help="shared library built on the head commit")
    parser.add_argument("--threshold", type=float, default=1.0, help="mark changes past this many percent")
    args = parser.parse_args()
    base, head = sizes(args.base), sizes(args.head)

    def fmt_pct(old: int, new: int):
        if old == 0:
            return "-"
        pct = (new - old) / old * 100
        if abs(pct) > args.threshold:
            # GitHub renders LaTeX colors in markdown: red for bigger, green for smaller.
            color = "red" if pct > 0 else "green"
            return f"$\\color{{{color}}}{{\\text{{{pct:+.2f}\\%}}}}$"
        return f"{pct:+.2f}%"

    lines = [
        "### Binary size (jix-py)",
        "",
        "| | base | head | diff | % |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {base[name]:,} | {head[name]:,} | {head[name] - base[name]:+,} | {fmt_pct(base[name], head[name])} |"
        for name in base
    ]
    lines += ["", "Bytes of the release-built `libjix.so` (linux x86_64). This job is report-only."]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
