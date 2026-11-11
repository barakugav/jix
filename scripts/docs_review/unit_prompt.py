"""Print the prompt that launches the scanner or the verifier agent of a unit.

The prompt is short: it names the unit's files and points the agent at its brief under
`scripts/docs_review/prompts/`.

Usage (from the repo root):
    python scripts/docs_review/unit_prompt.py --list
    python scripts/docs_review/unit_prompt.py scanner UNIT --workdir DIR --reports DIR
    python scripts/docs_review/unit_prompt.py verifier UNIT --workdir DIR --reports DIR

The agents write their output under the reports directory: `UNIT.scanner.md` (scanner report),
`UNIT.verifier.md` (verifier report), `UNIT.findings.md` and `UNIT.commit.md`.
"""

import argparse
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCANNER = """\
You are the docs scanner for unit `{id}` of the jix repository.

Work in `{workdir}`, your own checkout of the repository. Use that path for every command and file
access; the paths below are relative to it. Do not touch any other checkout.

Read `{workdir}/scripts/docs_review/prompts/scanner.md` first and follow it exactly: it is your
full brief.

{unit}

Write your report to `{reports}/{id}.scanner.md`.
"""

VERIFIER = """\
You are the docs verifier for unit `{id}` of the jix repository.

Work in `{workdir}`, a checkout that holds the scanner's uncommitted edits. Use that path for every
command and file access (for example `git -C {workdir} diff`); the paths below are relative to it.
Do not touch any other checkout.

Read `{workdir}/scripts/docs_review/prompts/verifier.md` first and follow it exactly: it is your
full brief.

{unit}

- The scanner's report: `{reports}/{id}.scanner.md`
- Write your report to `{reports}/{id}.verifier.md`
- Write the findings to `{reports}/{id}.findings.md`
- Write the commit summary to `{reports}/{id}.commit.md`
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
    parser.add_argument("--workdir", help="the agent's checkout: a worktree of the branch")
    parser.add_argument("--reports", help="directory the agents write their reports to")
    args = parser.parse_args()

    units = load_units()
    if args.list:
        for unit in units.values():
            pilot = " (pilot)" if unit.get("pilot") else ""
            print(f"{unit['id']:<26} {len(unit['edit']):>2} files{pilot}")
        return 0
    if args.role is None or args.unit not in units:
        parser.error("give a role and a unit id from units.toml")

    if not (args.workdir and args.reports):
        parser.error("give the agent's checkout with --workdir and the reports directory with --reports")
    unit = units[args.unit]
    template = SCANNER if args.role == "scanner" else VERIFIER
    print(template.format(id=unit["id"], unit=describe(unit), workdir=args.workdir, reports=args.reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
