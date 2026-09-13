"""Run all repository checks (format, lint, ASCII, docs, tests) with concise,
LLM-friendly output.

There is NO Cargo workspace: each crate is checked from its own manifest. This
runner wraps the commands documented in CLAUDE.md so an agent (or a human) can
gate the whole repo with one command and read a compact pass/fail summary.

Usage:
    python scripts/check_all.py                  # run everything
    python scripts/check_all.py --fast           # skip slow checks (see below)
    python scripts/check_all.py format lint      # run only these categories
    python scripts/check_all.py test --fast      # tests, minus the slow ones

Categories (default: all): format, lint, ascii, docs, test

Output contract (kept terse on purpose, so it is cheap to read):
  - each check prints exactly one result line on success: "[PASS] name (1.2s)"
  - on failure it prints the command, working dir, and the tail of its output
  - optional tools that are absent are reported as "[SKIP] name (reason)"
  - a final SUMMARY lists pass/fail/skip counts and every failing check name
  - exit code is 0 only if nothing failed (skips do not fail the run)

"--fast" skips the slow checks: the Cargo feature powerset, the maturin build,
and the Python test suite. Everything else still runs.

This file is tracked and must stay ASCII-only (scripts/check_only_ascii.py).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = ROOT / ".venv" / "bin"
TAIL_LINES = 40

CATEGORIES = ["format", "lint", "ascii", "docs", "test"]

# package-name -> Cargo.toml (no workspace; each crate is independent)
CRATES = {
    "jix": ROOT / "jix" / "Cargo.toml",
    "jix-schema": ROOT / "jix" / "schema" / "Cargo.toml",
    "jix-macros": ROOT / "jix-macros" / "Cargo.toml",
    "jix-py": ROOT / "jix-py" / "Cargo.toml",
}


def tool_available(name: str) -> bool:
    return (VENV_BIN / name).exists() or shutil.which(name) is not None


def runs_ok(cmd: list[str]) -> bool:
    """Whether a probe command exits 0 (used to detect optional tooling)."""
    try:
        return subprocess.run(cmd, capture_output=True, check=False).returncode == 0
    except OSError:
        return False


# Environment that mimics `source .venv/bin/activate` for maturin/pytest.
VENV_ENV = {
    **os.environ,
    "VIRTUAL_ENV": str(ROOT / ".venv"),
    "PATH": f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}",
}


@dataclass
class Check:
    name: str
    category: str
    argv: list[str]
    cwd: Path = ROOT
    env: dict | None = None
    slow: bool = False


@dataclass
class Result:
    name: str
    category: str
    status: str  # "pass" | "fail" | "skip"
    seconds: float = 0.0
    reason: str = ""
    warnings: int = 0


def build_checks() -> list[Check]:
    checks: list[Check] = []

    # ---- format ----
    for crate, mf in CRATES.items():
        checks.append(Check(f"fmt:{crate}", "format", ["cargo", "fmt", "--manifest-path", str(mf), "--", "--check"]))
    checks.append(
        Check(
            "ruff-format",
            "format",
            ["ruff", "--config", ".ruff.toml", "format", "--check", "jix-py/python"],
        )
    )

    # ---- lint ----
    for crate, mf in CRATES.items():
        checks.append(
            Check(f"clippy:{crate}", "lint", ["cargo", "clippy", "--manifest-path", str(mf), "--all-features"])
        )
    checks.append(
        Check(
            "ruff-check",
            "lint",
            ["ruff", "--config", ".ruff.toml", "check", "jix-py/python"],
        )
    )

    # ---- ascii ----
    checks.append(Check("ascii-only", "ascii", [sys.executable, "scripts/check_only_ascii.py"]))

    # ---- docs ----
    # Only the core `jix` crate is documented via rustdoc (RUSTDOCFLAGS=-D warnings catches
    # broken intra-doc links). The `jix-py` crate's Rust doc comments embed *Python* examples
    # in untagged code fences, which rustdoc cannot parse as Rust - its docs are surfaced via
    # the .pyi stubs and mkdocs (below), not rustdoc.
    checks.append(
        Check(
            "rustdoc:jix",
            "docs",
            ["cargo", "doc", "--no-deps", "--all-features", "--manifest-path", str(CRATES["jix"])],
        )
    )
    checks.append(
        Check(
            "mkdocs",
            "docs",
            ["mkdocs", "build", "--strict"],
            cwd=ROOT / "jix-py",
        )
    )

    # ---- test ----
    checks.append(Check("test:jix", "test", ["cargo", "test", "--all-features", "--manifest-path", str(CRATES["jix"])]))
    checks.append(Check("test:jix-schema", "test", ["cargo", "test", "--manifest-path", str(CRATES["jix-schema"])]))
    checks.append(Check("test:jix-macros", "test", ["cargo", "test", "--manifest-path", str(CRATES["jix-macros"])]))
    checks.append(
        Check(
            "test:jix-py-rust",
            "test",
            ["cargo", "test", "--all-features", "--all-targets", "--manifest-path", str(CRATES["jix-py"])],
        )
    )
    checks.append(
        Check(
            "feature-powerset",
            "test",
            ["cargo", "hack", "check", "--feature-powerset", "--depth", "2", "--manifest-path", str(CRATES["jix"])],
            slow=True,
        )
    )
    # The maturin build must precede pytest; pytest imports the freshly built extension.
    checks.append(
        Check(
            "maturin-develop",
            "test",
            ["maturin", "develop"],
            cwd=ROOT / "jix-py",
            env=VENV_ENV,
            slow=True,
        )
    )
    checks.append(
        Check(
            "pytest",
            "test",
            ["pytest", "python/tests", "--numprocesses", "auto", "-q"],
            cwd=ROOT / "jix-py",
            env=VENV_ENV,
            slow=True,
        )
    )
    return checks


def tail(text: str, n: int = TAIL_LINES) -> str:
    lines = text.rstrip("\n").splitlines()
    return "\n".join(lines[-n:])


def run_check(check: Check) -> Result:
    start = time.monotonic()
    try:
        proc = subprocess.run(check.argv, cwd=str(check.cwd), env=check.env, capture_output=True, text=True)  # noqa: PLW1510
    except OSError as exc:
        return Result(check.name, check.category, "fail", time.monotonic() - start, reason=f"could not launch: {exc}")
    seconds = time.monotonic() - start
    combined = (proc.stdout or "") + (proc.stderr or "")
    status = "pass" if proc.returncode == 0 else "fail"
    result = Result(check.name, check.category, status, seconds, warnings=combined.count("warning:"))
    if status == "fail":
        cmd = " ".join(check.argv)
        rel = check.cwd.relative_to(ROOT) if check.cwd != ROOT else Path(".")
        result.reason = f"exit {proc.returncode}\n  $ {cmd}   (cwd: {rel})\n" + _indent(tail(combined))
    return result


def _indent(text: str) -> str:
    return "\n".join("  | " + line for line in text.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repo checks with LLM-friendly output.")
    parser.add_argument("categories", nargs="*", choices=CATEGORIES, help="categories to run (default: all)")
    args = parser.parse_args()

    selected = args.categories or CATEGORIES

    bar = "=" * 60
    print(bar)
    print(f" jix repo checks  categories: {', '.join(selected)})")
    print(bar)

    all_checks = build_checks()
    results: list[Result] = []
    t0 = time.monotonic()

    for category in CATEGORIES:
        if category not in selected:
            continue
        group = [c for c in all_checks if c.category == category]
        if not group:
            continue
        print(f"\n-- {category.upper()} --")
        for check in group:
            res = run_check(check)
            results.append(res)
            if res.status == "pass":
                extra = f"  ({res.warnings} warnings)" if res.warnings else ""
                print(f"[PASS] {check.name}  ({res.seconds:.1f}s){extra}")
            else:
                print(f"[FAIL] {check.name}  ({res.seconds:.1f}s)")
                print(_indent(res.reason))

    passed = [r for r in results if r.status == "pass"]
    failed = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]

    print(f"\n{bar}")
    print(
        f" SUMMARY: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped"
        f"  (total {time.monotonic() - t0:.1f}s)"
    )
    if failed:
        print(" FAILED: " + ", ".join(r.name for r in failed))
    if skipped:
        print(" SKIPPED: " + ", ".join(r.name for r in skipped))
    print(bar)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
