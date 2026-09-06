"""Run every crate's and package's tests under coverage and report the result.

Usage:
    python scripts/coverage_report.py --out <dir>                  # everything
    python scripts/coverage_report.py --out <dir> jix jix-macros   # only these targets
    python scripts/coverage_report.py --out <dir> --fast           # skip the slow instrumented pytest leg
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # scripts/coverage_report.py -> repo root
JIX_PY_DIR = REPO / "jix-py"
JIX_PY_MANIFEST = JIX_PY_DIR / "Cargo.toml"


# ----------------------------------------------------------------------------------------------------
# Locating `#[cfg(test)]` blocks in Rust source
#
# Test code is self-covering, and this repo keeps its Rust tests inline in the file they
# test, so totals including those lines are meaningless. cargo-llvm-cov cannot drop them:
# `coverage(off)` is nightly-only and `--ignore-filename-regex` works per file.
# ----------------------------------------------------------------------------------------------------

# Matches `#[cfg(test)]` allowing arbitrary internal whitespace, e.g. `#[ cfg ( test ) ]`.
CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")


def _looks_like_char_literal(source: str, i: int, n: int) -> bool:
    """True if the `'` at index `i` opens a char literal rather than a lifetime.

    Lifetimes (`'a`, `'static`, `'_`) start with the same quote as char literals
    (`'a'`, `'\\n'`) but are never closed, so a full tokenizer isn't needed here: an
    escape right after the quote, or a closing quote exactly one character later, is
    enough to tell the two apart.
    """
    if i + 1 >= n:
        return False
    if source[i + 1] == "\\":
        return True
    return i + 2 < n and source[i + 1] != "'" and source[i + 2] == "'"


def _strip_noncode(source: str) -> str:
    """Replace comment and literal bodies with spaces, preserving every newline.

    Braces inside strings, chars and comments must not affect depth counting, but line
    numbers must stay exactly aligned with the original text, so every removed character
    is replaced by a space and every newline is kept.
    """
    out = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        two = source[i : i + 2]
        if two == "//":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
        elif two == "/*":
            depth = 0
            while i < n:
                if source[i : i + 2] == "/*":
                    depth += 1
                    out.append("  ")
                    i += 2
                elif source[i : i + 2] == "*/":
                    depth -= 1
                    out.append("  ")
                    i += 2
                    if depth == 0:
                        break
                else:
                    out.append("\n" if source[i] == "\n" else " ")
                    i += 1
        elif c == "r" and (raw := re.match(r'r(#*)"', source[i:])):
            hashes = len(raw.group(1))
            closing = '"' + "#" * hashes
            end = source.find(closing, i + hashes + 2)
            end = n if end == -1 else end + len(closing)
            out.extend("\n" if ch == "\n" else " " for ch in source[i:end])
            i = end
        elif c == "'" and not _looks_like_char_literal(source, i, n):
            # A bare `'` that doesn't open a char literal is a lifetime (`'a`, `'_`,
            # `'static`). Unlike a char literal it has no closing quote, so treating it
            # as one would swallow the rest of the file into a fake string body.
            out.append(c)
            i += 1
        elif c in ('"', "'"):
            quote = c
            out.append(" ")
            i += 1
            while i < n:
                if source[i] == "\\":
                    # Preserve a line continuation escape: eating that newline would shift
                    # every subsequent line number and corrupt which lines get excluded.
                    out.append(" ")
                    out.append("\n" if i + 1 < n and source[i + 1] == "\n" else " ")
                    i += 2
                    continue
                if source[i] == quote:
                    out.append(" ")
                    i += 1
                    break
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def cfg_test_ranges(source: str) -> list[tuple[int, int]]:
    """1-based inclusive (start, end) line ranges of every `#[cfg(test)]` block.

    The range starts at the attribute line and ends at the line closing the item the
    attribute applies to. An attribute whose block is never closed runs to end of file.
    """
    code = _strip_noncode(source)
    line_starts = [0]
    for idx, ch in enumerate(code):
        if ch == "\n":
            line_starts.append(idx + 1)

    def line_of(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    ranges = []
    for match in CFG_TEST.finditer(code):
        start_line = line_of(match.start())
        if ranges and start_line <= ranges[-1][1]:
            continue  # nested attribute already inside a recorded block
        depth = 0
        opened = False
        end_line = len(line_starts)
        for idx in range(match.end(), len(code)):
            ch = code[idx]
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth <= 0:
                    end_line = line_of(idx)
                    break
            elif ch == ";" and not opened:
                end_line = line_of(idx)  # e.g. `#[cfg(test)] use foo::bar;`
                break
        ranges.append((start_line, end_line))
    return ranges


def cfg_test_lines(source: str) -> set[int]:
    """Every 1-based line number that falls inside a `#[cfg(test)]` block."""
    lines: set[int] = set()
    for start, end in cfg_test_ranges(source):
        lines.update(range(start, end + 1))
    return lines


# ----------------------------------------------------------------------------------------------------
# Parsing the JSON that cargo-llvm-cov and pytest-cov export
# ----------------------------------------------------------------------------------------------------

# The export version tracks the LLVM shipped with the toolchain, not the format we parse. Major 3
# arrived in rustc 1.92 (LLVM 21.1.3, export 3.0.1) and rustc 1.98 emits 3.1.0; the two were checked
# to produce identical keys, segments and derived counts, so the minor is ignored. Any other major
# stops the run rather than guessing at a layout that is parsed by position.
LLVM_EXPORT_MAJOR = "3"
SEGMENT_FIELDS = 6
PYTEST_COV_FORMAT = 3


@dataclass(frozen=True)
class FileCoverage:
    """Coverage for one source file, as reported by one tool run."""

    path: Path
    counts: dict[int, int]
    tool_covered: int = 0
    tool_total: int = 0


def _is_start_of_region(seg: list) -> bool:
    """A segment that opens a newly counted region: not a gap, and it carries a count."""
    _line, _col, _count, has_count, is_region_entry, is_gap_region = seg
    return (not is_gap_region) and has_count and is_region_entry


def _lines_from_segments(segments: list[list]) -> dict[int, int]:
    """Derive per line execution counts from llvm-cov segments.

    A direct port of LLVM's own `LineCoverageStats` constructor
    (llvm/lib/ProfileData/Coverage/CoverageMapping.cpp): walk every line between the first and
    last segment, carrying the last segment seen from an earlier line ("wrapped"), because a
    region opened on one line continues to cover the lines beneath it until another segment
    closes it. Verbatim from that constructor (bindings renamed to match ours):

        bool StartOfSkippedRegion = !LineSegments.empty() &&
                                     !LineSegments.front()->HasCount &&
                                     LineSegments.front()->IsRegionEntry;
        Mapped = !StartOfSkippedRegion &&
                 ((WrappedSegment && WrappedSegment->HasCount) || (MinRegionCount > 0));
        // if there is any starting segment at this line with a counter, it must be mapped
        Mapped |= any_of(LineSegments, [](S) { return S->IsRegionEntry && S->HasCount; });
        if (!Mapped) return;
        if (WrappedSegment) ExecutionCount = WrappedSegment->Count;
        if (!MinRegionCount) return;
        for (LS : LineSegments)
          if (isStartOfRegion(LS)) ExecutionCount = std::max(ExecutionCount, LS->Count);

    Two details that are easy to get wrong:
    - The wrapped segment's count seeds `ExecutionCount` unconditionally whenever a wrapped
      segment exists at all - not only when the wrapped segment itself has_count. A closing
      segment with has_count False can still become `wrapped`, and its (possibly 0) count
      field is what later lines start from.
    - When the line has its own non-gap region-entry segment(s), their counts are *maxed
      against* the wrapped-seeded count, not used instead of it. A covered enclosing region
      with an untaken branch on the same line keeps reading as covered - the max, not the
      branch's own 0, wins. Gap regions never contribute to this max (they are excluded from
      `isStartOfRegion`), but a gap-region entry that carries a count still forces the line
      itself to be mapped via the second `any_of`, which does not exclude gap regions.

    Segments are sorted by (line, column) first: llvm-cov's own JSON is already in that order,
    but nothing in the format guarantees it, and both "first segment on the line" (for the
    skipped-region check) and "wrapped" (the last segment on the line) depend on that ordering.
    """
    if not segments:
        return {}
    ordered = sorted(segments, key=lambda seg: (seg[0], seg[1]))
    by_line: dict[int, list[list]] = {}
    for seg in ordered:
        by_line.setdefault(seg[0], []).append(seg)

    counts: dict[int, int] = {}
    wrapped = None
    for line in range(min(by_line), max(by_line) + 1):
        here = by_line.get(line, [])
        entries = [seg for seg in here if _is_start_of_region(seg)]
        start_of_skipped_region = bool(here) and (not here[0][3]) and here[0][4]
        mapped = (not start_of_skipped_region) and ((wrapped is not None and wrapped[3]) or bool(entries))
        mapped = mapped or any(seg[3] and seg[4] for seg in here)
        if mapped:
            count = wrapped[2] if wrapped is not None else 0
            if entries:
                count = max([count] + [seg[2] for seg in entries])
            counts[line] = count
        if here:
            wrapped = here[-1]
    return counts


def load_llvm_json(path: Path) -> list[FileCoverage]:
    """Load a `cargo llvm-cov report --json` export."""
    doc = json.loads(path.read_bytes())
    version = str(doc.get("version", ""))
    if version.split(".")[0] != LLVM_EXPORT_MAJOR:
        raise RuntimeError(
            f"{path}: llvm-cov export version {version} is not major {LLVM_EXPORT_MAJOR}, which needs "
            "rustc 1.92 or newer. Check scripts/coverage_report.py before trusting these numbers."
        )
    out = []
    for export in doc["data"]:
        for entry in export["files"]:
            segments = entry.get("segments", [])
            if segments and len(segments[0]) != SEGMENT_FIELDS:
                raise RuntimeError(
                    f"{path}: expected {SEGMENT_FIELDS}-field llvm-cov segments, got {len(segments[0])}. "
                    "The segment layout changed and is parsed by position."
                )
            summary = entry.get("summary", {}).get("lines", {})
            out.append(
                FileCoverage(
                    path=Path(entry["filename"]),
                    counts=_lines_from_segments(segments),
                    tool_covered=summary.get("covered", 0),
                    tool_total=summary.get("count", 0),
                )
            )
    return out


def load_pytest_json(path: Path, root: Path) -> list[FileCoverage]:
    """Load a `pytest --cov-report=json` report.

    coverage.py records paths relative to the directory pytest ran in, so `root` is
    joined onto each one to produce the absolute paths the aggregation layer keys on.
    Line counts are synthesised as 1 (executed) or 0 (missing): coverage.py tracks hit
    or not hit, not an execution count, which is all the aggregation layer needs.
    """
    doc = json.loads(path.read_bytes())
    fmt = doc.get("meta", {}).get("format")
    if fmt != PYTEST_COV_FORMAT:
        raise RuntimeError(
            f"{path}: expected pytest-cov JSON format {PYTEST_COV_FORMAT}, got {fmt}. "
            "The report format changed; check scripts/coverage_report.py before trusting these numbers."
        )
    out = []
    for name, entry in doc["files"].items():
        counts = {line: 1 for line in entry["executed_lines"]}
        counts.update({line: 0 for line in entry["missing_lines"]})
        summary = entry.get("summary", {})
        out.append(
            FileCoverage(
                path=(root / name).resolve(),
                counts=counts,
                tool_covered=summary.get("covered_lines", 0),
                tool_total=summary.get("num_statements", 0),
            )
        )
    return out


# ----------------------------------------------------------------------------------------------------
# Filtering, deduplicating and pooling per-target coverage
#
# Order matters: `#[cfg(test)]` lines are dropped, then files seen by more than one target
# are unioned by `(resolved path, line)` so a shared source counts once, then covered and
# total lines are pooled. The pooled ratio is the LOC-weighted average of the per-target
# percentages; an unweighted mean would let a 259-line crate weigh as much as a 47,000-line one.
# ----------------------------------------------------------------------------------------------------

SCHEMA_VERSION = 1

EXCLUDES = {
    "jix": [
        # prost output
        "src/archive/schema/_proto_gen/*",
        # external libraries
        "src/util/aligned_vec/*",
        "src/util/arrayvec.rs",
        # non production code
        "src/util/test_util.rs",
        "src/util/bench_util.rs",
    ],
    "jix-macros": [],
    "jix-py": [],
}


@dataclass
class TargetResult:
    """Everything one target contributed, after its report was parsed."""

    name: str
    files: list[FileCoverage]
    failures: list[str] = field(default_factory=list)


def _totals(pool: dict[Path, dict[int, int]]) -> tuple[int, int]:
    """Covered and total line counts for a pooled `{path: {line: count}}` mapping."""
    covered = sum(1 for lines in pool.values() for count in lines.values() if count > 0)
    return covered, sum(len(lines) for lines in pool.values())


def _percent(covered: int, total: int) -> float:
    return round(100.0 * covered / total, 2) if total else 0.0


@lru_cache
def _is_excluded_file(path: Path) -> bool:
    """True if `path` matches one of its crate's `EXCLUDES`."""
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        return False
    inner = Path(*rel.parts[1:]).as_posix()
    return any(fnmatch(inner, glob) for glob in EXCLUDES.get(rel.parts[0], []))


@lru_cache
def _excluded_lines(path: Path) -> frozenset[int]:
    """Lines inside `#[cfg(test)]` blocks. Cached: shared sources are filtered per target."""
    return frozenset(cfg_test_lines(path.read_text()))


def filter_cfg_test_lines(files: list[FileCoverage]) -> tuple[list[FileCoverage], int]:
    """Drop lines inside `#[cfg(test)]` blocks. Returns the files and how many were dropped."""
    out = []
    dropped = 0
    for fc in files:
        if fc.path.suffix != ".rs" or not fc.path.is_file():
            out.append(fc)
            continue
        excluded = _excluded_lines(fc.path)
        kept = {line: count for line, count in fc.counts.items() if line not in excluded}
        dropped += len(fc.counts) - len(kept)
        out.append(FileCoverage(fc.path, kept, fc.tool_covered, fc.tool_total))
    return out, dropped


def prepare_files(name: str, files: list[FileCoverage]) -> list[FileCoverage]:
    """Apply both filters to a freshly parsed report and say what they removed."""
    kept = [fc for fc in files if not _is_excluded_file(fc.path)]
    kept, dropped = filter_cfg_test_lines(kept)
    print(f"[{name}] excluded {len(files) - len(kept)} files and {dropped} lines of inline test code")
    return kept


def union_counts(results: list[TargetResult]) -> dict[Path, dict[int, int]]:
    """Merge files by `(resolved path, line)`, keeping the highest count seen.

    Paths are resolved because the same physical file reaches here spelled differently
    from different targets, and must not be counted twice.
    """
    merged: dict[Path, dict[int, int]] = {}
    for result in results:
        for fc in result.files:
            lines = merged.setdefault(fc.path.resolve(), {})
            for line, count in fc.counts.items():
                lines[line] = max(lines.get(line, 0), count)
    return merged


def build_summary(results: list[TargetResult], fast: bool = False, leg_failures: list[str] | None = None) -> dict:
    """Build the `summary.json` payload, the shape a badge job reads.

    `targets_run` and `fast` are recorded because a partial or `--fast` run otherwise
    produces a file indistinguishable from a complete one.
    """
    targets = []
    merged: dict[Path, dict[int, int]] = {}
    for result in results:
        pool = union_counts([result])
        for path, lines in pool.items():
            into = merged.setdefault(path, {})
            for line, count in lines.items():
                into[line] = max(into.get(line, 0), count)
        covered, total = _totals(pool)
        tool_covered = sum(fc.tool_covered for fc in result.files)
        tool_total = sum(fc.tool_total for fc in result.files)
        targets.append(
            {
                "name": result.name,
                "covered": covered,
                "total": total,
                "percent": _percent(covered, total),
                "unfiltered": {
                    "covered": tool_covered,
                    "total": tool_total,
                    "percent": _percent(tool_covered, tool_total),
                },
            }
        )

    all_covered, all_total = _totals(merged)
    return {
        "schema": SCHEMA_VERSION,
        "generated_by": "scripts/coverage_report.py",
        "targets_run": [result.name for result in results],
        "fast": fast,
        "leg_failures": list(leg_failures) if leg_failures else [],
        "targets": targets,
        "total": {"covered": all_covered, "total": all_total, "percent": _percent(all_covered, all_total)},
    }


def render_table(summary: dict) -> str:
    """One line per target plus a pooled total."""
    rows = [("TARGET", "COVERED", "LINES", "PERCENT", "UNFILTERED")]
    for target in summary["targets"]:
        rows.append(
            (
                target["name"],
                str(target["covered"]),
                str(target["total"]),
                f"{target['percent']:.1f}%",
                f"{target['unfiltered']['percent']:.1f}%",
            )
        )
    total = summary["total"]
    rows.append(("TOTAL", str(total["covered"]), str(total["total"]), f"{total['percent']:.1f}%", "-"))

    widths = [max(len(cell) for cell in col) for col in zip(*rows)]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if idx == 0 or idx == len(rows) - 2:
            lines.append("-" * len(lines[0]))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# Running the targets
# ----------------------------------------------------------------------------------------------------


def venv_env() -> dict[str, str]:
    """A copy of the environment with the repo venv activated."""
    env = {**os.environ}
    env["VIRTUAL_ENV"] = str(REPO / ".venv")
    env["PATH"] = f"{REPO / '.venv' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def check_leg(proc: subprocess.CompletedProcess, name: str, leg: str, failures: list[str]):
    """Record a non-zero test exit without aborting: the report is still worth building."""
    if proc.returncode != 0:
        msg = f"{name} {leg} leg exited {proc.returncode}"
        print(f"[{name}] WARNING: {msg}; building a report from whatever data exists")
        failures.append(msg)


def llvm_report(name: str, manifest_args: list[str], target_out: Path, failures: list[str]) -> TargetResult:
    """Render llvm-cov's html and json for a target, then load and filter the json.

    `cargo llvm-cov report --html` appends its own `html/` segment under `--output-dir`,
    so `target_out` (not `target_out / "html"`) lands the report at `html/index.html`.
    """
    subprocess.check_call(["cargo", "llvm-cov", "report", *manifest_args, "--html", "--output-dir", str(target_out)])
    json_path = target_out / "coverage.json"
    subprocess.check_call(["cargo", "llvm-cov", "report", *manifest_args, "--json", "--output-path", str(json_path)])
    return TargetResult(name, prepare_files(name, load_llvm_json(json_path)), failures)


def run_cargo_target(name: str, manifest: Path, out: Path, features: list[str], reuse: bool) -> TargetResult:
    """Run one crate's tests under llvm-cov and load the resulting report."""
    target_out = out / name
    target_out.mkdir(parents=True, exist_ok=True)
    manifest_args = ["--manifest-path", str(manifest)]

    if not reuse:
        subprocess.check_call(["cargo", "llvm-cov", "clean", *manifest_args])
    subprocess.check_call(["cargo", "llvm-cov", "--no-report", *manifest_args, *features])
    return llvm_report(name, manifest_args, target_out, [])


def run_jix_py_target(out: Path, fast: bool, reuse: bool) -> TargetResult:
    """Run jix-py's Rust tests and, unless --fast, the instrumented Python suite too.

    Both legs write into the same profile directory, so one report call covers them.
    """
    failures: list[str] = []
    target_out = out / "jix-py"
    target_out.mkdir(parents=True, exist_ok=True)
    manifest_args = ["--manifest-path", str(JIX_PY_MANIFEST)]
    crate_dir = JIX_PY_DIR

    if not reuse:
        subprocess.check_call(["cargo", "llvm-cov", "clean", *manifest_args])
    rust_tests = subprocess.run(
        ["cargo", "llvm-cov", "--no-report", *manifest_args, "--all-features", "--all-targets"], check=False
    )
    check_leg(rust_tests, "jix-py", "Rust test", failures)

    if not fast:
        # Plain `KEY=value` lines; `--export-prefix` would prefix each with `export `.
        shown_text = subprocess.check_output(["cargo", "llvm-cov", "show-env", *manifest_args], text=True)
        shown = parse_show_env(shown_text)
        env = {**venv_env(), **shown}
        env["LLVM_PROFILE_FILE"] = profile_file(shown)
        profile_dir = Path(env["LLVM_PROFILE_FILE"]).parent
        profile_dir.mkdir(parents=True, exist_ok=True)
        # `maturin develop` builds into the crate's ordinary `target/`, which `report` never
        # scans for object files. Without this redirect the cdylib is unfindable and `jix-py/src`
        # gets zero credit from the pytest leg even though its profraw files are valid.
        env["CARGO_TARGET_DIR"] = str(profile_dir)
        subprocess.check_call(["maturin", "develop"], cwd=crate_dir, env=env)
        # The profraw guard below proves profiles landed where `report` globs, not that
        # `report` can attribute them to `jix-py/src`. That needs the cdylib to exist here.
        built_libs = [p for p in profile_dir.rglob("libjix.*") if p.suffix in (".dylib", ".so")]
        if not built_libs:
            raise RuntimeError(
                f"[jix-py] maturin develop did not produce an instrumented libjix.dylib/.so "
                f"under {profile_dir}; jix-py/src coverage would silently fall back to "
                "uninstrumented (or missing) attribution - maturin/cargo's build layout may "
                "have changed"
            )
        profraw_before = len(list(profile_dir.glob("*.profraw")))
        pytest_run = subprocess.run(
            ["pytest", "python/tests", "--numprocesses", "auto", "-q"], cwd=crate_dir, env=env, check=False
        )
        check_leg(pytest_run, "jix-py", "pytest", failures)
        profraw_after = len(list(profile_dir.glob("*.profraw")))
        if profraw_after <= profraw_before:
            raise RuntimeError(
                f"[jix-py] pytest leg wrote no new profraw files into {profile_dir} "
                f"(before={profraw_before}, after={profraw_after}); coverage would be silently "
                "understated - cargo-llvm-cov's internal layout may have changed"
            )

    return llvm_report("jix-py", manifest_args, target_out, failures)


def parse_show_env(text: str) -> dict[str, str]:
    """Parse `cargo llvm-cov show-env` output into an environment dict.

    Lines look like `KEY='value'` or `KEY=value`. Informational lines that the tool
    writes without an `=` are ignored.
    """
    env = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith(("info:", "warning:")):
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'")
    return env


def profile_file(shown: dict[str, str]) -> str:
    """Path for `LLVM_PROFILE_FILE`, redirected to where `cargo llvm-cov report` reads.

    `show-env` points `LLVM_PROFILE_FILE` at `CARGO_LLVM_COV_TARGET_DIR`, but `report` only
    globs `<target>/llvm-cov-target/`. Keys come from `shown` rather than the environment,
    which may hold stale values from an earlier `show-env` eval.
    """
    missing = [key for key in ("CARGO_LLVM_COV_TARGET_DIR", "LLVM_PROFILE_FILE") if key not in shown]
    if missing:
        raise RuntimeError(f"cargo llvm-cov show-env is missing required key(s): {', '.join(missing)}")
    target_dir = shown["CARGO_LLVM_COV_TARGET_DIR"]
    raw_name = Path(shown["LLVM_PROFILE_FILE"]).name
    return str(Path(target_dir) / "llvm-cov-target" / raw_name)


def run_python_target(out: Path) -> TargetResult:
    """Measure the pure Python part of the jix package with pytest-cov.

    Thin today: `jix/__init__.py` is a five line re-export and the rest of the package is
    the compiled extension.

    `LLVM_PROFILE_FILE` is redirected into `--out` because the installed extension may be an
    instrumented build from a prior `jix-py` run; without it the LLVM runtime writes
    `default_*.profraw` into the source tree. This leg reads none of them - it is measured by
    pytest-cov.
    """
    target_out = out / "python"
    target_out.mkdir(parents=True, exist_ok=True)
    crate_dir = JIX_PY_DIR
    json_path = target_out / "coverage.json"
    scratch_profile_dir = target_out / "llvm_profile"
    scratch_profile_dir.mkdir(parents=True, exist_ok=True)

    env = venv_env()
    env["LLVM_PROFILE_FILE"] = str(scratch_profile_dir / "default_%m_%p.profraw")
    pytest_run = subprocess.run(
        [
            "pytest",
            "python/tests",
            "--numprocesses",
            "auto",
            "-q",
            "--cov=jix",
            f"--cov-report=json:{json_path}",
            f"--cov-report=html:{target_out / 'html'}",
        ],
        cwd=crate_dir,
        env=env,
        check=False,
    )
    failures: list[str] = []
    check_leg(pytest_run, "python", "pytest", failures)

    files = prepare_files("python", load_pytest_json(json_path, root=crate_dir))
    return TargetResult("python", files, failures)


RUNNERS = {
    "jix": lambda out, args: run_cargo_target("jix", REPO / "jix" / "Cargo.toml", out, ["--all-features"], args.reuse),
    "jix-macros": lambda out, args: run_cargo_target(
        "jix-macros", REPO / "jix-macros" / "Cargo.toml", out, [], args.reuse
    ),
    "jix-py": lambda out, args: run_jix_py_target(out, args.fast, args.reuse),
    "python": lambda out, args: run_python_target(out),
}
ALL_TARGETS = list(RUNNERS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the test suites under coverage and summarise the result.")
    parser.add_argument("targets", nargs="*", choices=ALL_TARGETS, help="targets to run (default: all)")
    parser.add_argument("--out", type=Path, required=True, help="output directory for reports (required)")
    parser.add_argument("--fast", action="store_true", help="skip the instrumented maturin+pytest leg")
    parser.add_argument("--open", action="store_true", help="open the HTML reports when finished")
    parser.add_argument("--reuse", action="store_true", help="reuse existing profile data instead of cleaning")
    args = parser.parse_args()

    selected = args.targets or ALL_TARGETS
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    results = [RUNNERS[name](out, args) for name in selected]
    leg_failures = [msg for result in results for msg in result.failures]

    summary = build_summary(results, fast=args.fast, leg_failures=leg_failures)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(render_table(summary))
    print()
    print(f"reports: {out}")

    if args.open:
        for name in selected:
            index = out / name / "html" / "index.html"
            if index.is_file():
                webbrowser.open(index.as_uri())

    if leg_failures:
        print()
        for msg in leg_failures:
            print(f"WARNING: {msg}")

    return 1 if leg_failures else 0


if __name__ == "__main__":
    sys.exit(main())
