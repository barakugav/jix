"""Gate: run the Python examples of the docs.

Extracts every fenced `python` block from the doc comments of `jix-py/src/**/*.rs` (the Python
docstrings) and from the markdown pages, and executes each one in a fresh namespace inside a
scratch directory. Only self-contained examples run: a block without `import jix`, or one that
reads a file it did not create, is a fragment and is reported as skipped.

Requires the extension to be installed (`maturin develop` in `jix-py`).

Usage (from the repo root):
    python scripts/docs_review/run_py_examples.py [PATH ...]
"""

import os
import re
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

DEFAULT_PATHS = ["jix-py/src", "jix-py/docs", "jix-py/README.md", "README.md"]
DOC_LINE = re.compile(r"^\s*//[/!] ?(.*)$")


def doc_lines(path: Path) -> list[tuple[int, str | None]]:
    """The documentation text of a file as (line number, text); the text of a code line is None."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".md":
        return list(enumerate(lines, 1))
    matches = ((lineno, DOC_LINE.match(line)) for lineno, line in enumerate(lines, 1))
    return [(lineno, m.group(1) if m else None) for lineno, m in matches]


def examples(path: Path) -> list[tuple[int, str]]:
    """The fenced python blocks of a file as (line number of the fence, code)."""
    found = []
    start, block = None, []
    for lineno, text in doc_lines(path):
        if text is None:
            start = None
            continue
        stripped = text.strip()
        if start is None:
            if stripped.startswith("```python"):
                start, block = lineno, []
        elif stripped.startswith("```"):
            found.append((start, textwrap.dedent("\n".join(block))))
            start = None
        else:
            block.append(text)
    return found


def main() -> int:
    roots = [Path(p) for p in (sys.argv[1:] or DEFAULT_PATHS)]
    files = []
    for root in roots:
        files += sorted(p for p in root.rglob("*") if p.suffix in (".rs", ".md")) if root.is_dir() else [root]
    todo = [(path, lineno, code) for path in files for lineno, code in examples(path)]

    passed = 0
    skipped, failures = [], []
    repo = os.getcwd()
    for path, lineno, code in todo:
        if "import jix" not in code:
            skipped.append((path, lineno, "fragment, no `import jix`"))
            continue
        # a fresh directory per example: several of them write the same file names
        with tempfile.TemporaryDirectory() as scratch:
            os.chdir(scratch)
            try:
                exec(compile(code, f"{path}:{lineno}", "exec"), {"__name__": "__main__"})  # noqa: S102
                passed += 1
            except FileNotFoundError:
                skipped.append((path, lineno, "reads a file it does not create"))
            except Exception:  # noqa: BLE001
                failures.append((path, lineno, traceback.format_exc(limit=-1).strip().splitlines()[-1]))
            finally:
                os.chdir(repo)

    print(f"{passed} passed, {len(failures)} failed, {len(skipped)} skipped")
    for path, lineno, reason in skipped:
        print(f"skip {path}:{lineno}: {reason}")
    for path, lineno, error in failures:
        print(f"FAIL {path}:{lineno}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
