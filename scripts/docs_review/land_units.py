"""Land verified units on the branch: apply their diffs, then commit them one by one.

    apply   copies the diff of each unit's worktree into the current checkout, after checking
            that it only touches the unit's own edit files. Run the gates after it.
    commit  makes one commit per unit: its edit files, plus the unit's findings appended to
            findings.md. The commit body is the summary the verifier wrote.

The gates run once between the two steps, on all applied units together. That is sound because
units own disjoint files and every gate judges a file, a doc or an example on its own.

Usage (from the repo root):
    python scripts/docs_review/land_units.py apply  --worktrees DIR --reports DIR UNIT ...
    scripts/docs_review/gates.sh
    python scripts/docs_review/land_units.py commit --reports DIR [--trailer TEXT] UNIT ...
"""

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINDINGS = HERE / "findings.md"


def load_units() -> dict[str, dict]:
    with open(HERE / "units.toml", "rb") as f:
        return {unit["id"]: unit for unit in tomllib.load(f)["unit"]}


def git(*args: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, input=stdin, check=True, capture_output=True, text=True).stdout


def apply(unit: dict, worktrees: Path, reports: Path) -> bool:
    worktree = worktrees / unit["id"]
    touched = git("diff", "--name-only", cwd=worktree).split()
    stray = sorted(set(touched) - set(unit["edit"])) if unit["edit"] else []
    if stray:
        print(f"{unit['id']}: SKIPPED, the diff touches files outside the unit: {', '.join(stray)}")
        return False
    patch = git("diff", cwd=worktree)
    (reports / f"{unit['id']}.patch").write_text(patch, encoding="utf-8")
    if patch:
        git("apply", "--index", stdin=patch)
    stat = git("diff", "--shortstat", cwd=worktree).strip() or "no changes"
    print(f"{unit['id']}: {stat}")
    return True


def append_findings(unit_id: str, findings: str):
    text = FINDINGS.read_text(encoding="utf-8")
    passes = sum(1 for line in text.splitlines() if line == f"## {unit_id}" or line.startswith(f"## {unit_id} ("))
    heading = f"## {unit_id}" if passes == 0 else f"## {unit_id} (pass {passes + 1})"
    FINDINGS.write_text(f"{text.rstrip()}\n\n{heading}\n\n{findings}\n", encoding="utf-8")


def commit(unit: dict, reports: Path, trailer: str) -> bool:
    unit_id = unit["id"]
    patch = reports / f"{unit_id}.patch"
    files = [line[6:] for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("+++ b/")]
    findings = (reports / f"{unit_id}.findings.md").read_text(encoding="utf-8").strip()
    if not files and not findings:
        print(f"{unit_id}: nothing to commit")
        return False
    if findings:
        append_findings(unit_id, findings)
        files.append(str(FINDINGS.relative_to(HERE.parents[1])))
    summary = (reports / f"{unit_id}.commit.md").read_text(encoding="utf-8").strip() if files else ""
    message = "\n\n".join(part for part in [f"Docs review: {unit_id}", summary, trailer.strip()] if part)
    git("add", "--", *files)
    git("commit", "--quiet", "--only", "--file", "-", "--", *files, stdin=message)
    print(f"{unit_id}: {git('log', '--oneline', '-1').strip()}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["apply", "commit"])
    parser.add_argument("units", nargs="+")
    parser.add_argument("--worktrees", type=Path, help="apply: directory holding one worktree per unit")
    parser.add_argument("--reports", type=Path, required=True, help="directory of the agents' reports")
    parser.add_argument("--trailer", default="", help="commit: text appended to every commit message")
    args = parser.parse_args()

    units = load_units()
    unknown = [unit for unit in args.units if unit not in units]
    if unknown:
        parser.error(f"unknown units: {', '.join(unknown)}")
    if args.step == "apply":
        if not args.worktrees:
            parser.error("apply needs --worktrees")
        done = [apply(units[unit], args.worktrees, args.reports) for unit in args.units]
    else:
        done = [commit(units[unit], args.reports, args.trailer) for unit in args.units]
    return 0 if all(done) or args.step == "commit" else 1


if __name__ == "__main__":
    sys.exit(main())
