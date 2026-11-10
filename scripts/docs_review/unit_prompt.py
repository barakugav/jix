"""Print the prompt that launches the scanner or the verifier agent of a unit.

The prompt is short: it names the unit's files and points the agent at its brief under
`scripts/docs_review/prompts/`.

Usage (from the repo root):
    python scripts/docs_review/unit_prompt.py --list
    python scripts/docs_review/unit_prompt.py scanner UNIT
    python scripts/docs_review/unit_prompt.py verifier UNIT --workdir DIR --report FILE
"""

import argparse
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCANNER = """\
You are the docs scanner for unit `{id}` of the jix repository.

Read `scripts/docs_review/prompts/scanner.md` first and follow it exactly: it is your full brief.

{unit}

Work in the current directory, which is your own checkout of the repository. End with the report
the brief asks for.
"""

VERIFIER = """\
You are the docs verifier for unit `{id}` of the jix repository.

Work in `{workdir}`, a checkout that holds the scanner's uncommitted edits. Use that path for every
command and file access (for example `git -C {workdir} diff`).

Read `{workdir}/scripts/docs_review/prompts/verifier.md` first and follow it exactly: it is your
full brief.

{unit}

The scanner's report:

<scanner_report>
{report}
</scanner_report>

End with the report the brief asks for.
"""


def load_units() -> dict[str, dict]:
    with open(HERE / "units.toml", "rb") as f:
        return {unit["id"]: unit for unit in tomllib.load(f)["unit"]}


def describe(unit: dict) -> str:
    def bullets(paths: list[str]) -> str:
        return "\n".join(f"  - `{path}`" for path in paths) or "  - none"

    packs = ", ".join(f"`{name}.md`" for name in ["general", "api_index", *unit["context"]])
    lines = [
        "Your unit:",
        f"- Context packs, in `scripts/docs_review/context/`: {packs}",
        f"- Edit files:\n{bullets(unit['edit'])}",
        f"- Reference files, read-only:\n{bullets(unit.get('read', []))}",
    ]
    if notes := unit.get("notes", "").strip():
        lines.append(f"- Notes: {notes}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("role", nargs="?", choices=["scanner", "verifier"])
    parser.add_argument("unit", nargs="?")
    parser.add_argument("--list", action="store_true", help="list the units and exit")
    parser.add_argument("--workdir", help="verifier: the checkout with the scanner's edits")
    parser.add_argument("--report", help="verifier: file holding the scanner's report")
    args = parser.parse_args()

    units = load_units()
    if args.list:
        for unit in units.values():
            pilot = " (pilot)" if unit.get("pilot") else ""
            print(f"{unit['id']:<26} {len(unit['edit']):>2} files{pilot}")
        return 0
    if args.role is None or args.unit not in units:
        parser.error("give a role and a unit id from units.toml")

    unit = units[args.unit]
    if args.role == "scanner":
        print(SCANNER.format(id=unit["id"], unit=describe(unit)))
    else:
        if not (args.workdir and args.report):
            parser.error("the verifier prompt needs --workdir and --report")
        report = Path(args.report).read_text(encoding="utf-8").strip()
        print(VERIFIER.format(id=unit["id"], unit=describe(unit), workdir=args.workdir, report=report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
