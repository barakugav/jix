"""Gate: a docs-review change may only touch comments, docstrings and markdown.

Compares the working tree against a base revision and fails if any changed file differs in more
than its comments:

- `.rs`  - the code with all comments removed must be identical (doc comments, which hold the
  doc-tests, are comments too; `cargo test --doc` covers those).
- `.py`  - the AST with all docstrings removed must be identical.
- `.md`  - always allowed.
- anything else fails.

Also prints the net change in doc lines, to keep an eye on the docs not growing.

Usage (from the repo root):
    python scripts/docs_review/check_docs_only.py [--base REV] [--allow PATH ...] [PATH ...]
"""

import argparse
import ast
import re
import subprocess
import sys

RAW_STR = re.compile(r'r(#*)"')
DOC_LINE = re.compile(r"^\s*(///|//!)")


def strip_rust_comments(src: str) -> str:
    """Remove line and block comments, keeping string and char literals intact."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif src.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif c == "r" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_") or src[i - 1] == "b"):
            m = RAW_STR.match(src, i)
            if m is None:
                out.append(c)
                i += 1
                continue
            close = '"' + m.group(1)
            j = src.find(close, m.end())
            j = n if j < 0 else j + len(close)
            out.append(src[i:j])
            i = j
        elif c == "'":
            if src.startswith("\\", i + 1):  # escaped char literal: '\n', '\'', '\u{..}'
                j = src.find("'", i + 3)
                j = n if j < 0 else j + 1
                out.append(src[i:j])
                i = j
            elif src.startswith("'", i + 2):  # plain char literal: 'x'
                out.append(src[i : i + 3])
                i += 3
            else:  # lifetime
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def rust_code(src: str) -> list[str]:
    lines = (line.strip() for line in strip_rust_comments(src).splitlines())
    return [line for line in lines if line]


class _DropDocstrings(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def python_code(src: str) -> str:
    return ast.dump(_DropDocstrings().visit(ast.parse(src)))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def read_base(base: str, path: str) -> str | None:
    res = subprocess.run(["git", "show", f"{base}:{path}"], check=False, capture_output=True, text=True)
    return res.stdout if res.returncode == 0 else None


def read_tree(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def first_difference(old: list[str], new: list[str]) -> str:
    for k, (a, b) in enumerate(zip(old, new)):
        if a != b:
            return f"code line {k}:\n      - {a}\n      + {b}"
    return f"code has {len(old)} lines before, {len(new)} after"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="HEAD", help="revision to compare the working tree against")
    parser.add_argument("--allow", action="append", default=[], help="path prefix exempt from the check")
    parser.add_argument("paths", nargs="*", help="limit the check to these paths")
    args = parser.parse_args()

    changed = git("diff", "--name-only", args.base, "--", *args.paths).split()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *args.paths).split()
    failures = []
    doc_before = doc_after = 0
    for path in sorted(set(changed + untracked)):
        if any(path.startswith(prefix) for prefix in args.allow):
            continue
        old, new = read_base(args.base, path), read_tree(path)
        if old is None or new is None:
            failures.append(f"{path}: file added or removed")
        elif path.endswith(".md"):
            doc_before += len(old.splitlines())
            doc_after += len(new.splitlines())
        elif path.endswith(".rs"):
            old_code, new_code = rust_code(old), rust_code(new)
            if old_code != new_code:
                failures.append(f"{path}: code changed, {first_difference(old_code, new_code)}")
            doc_before += sum(1 for line in old.splitlines() if DOC_LINE.match(line))
            doc_after += sum(1 for line in new.splitlines() if DOC_LINE.match(line))
        elif path.endswith(".py"):
            if python_code(old) != python_code(new):
                failures.append(f"{path}: code changed")
        else:
            failures.append(f"{path}: not a docs file")

    print(f"{len(changed) + len(untracked)} changed files, doc lines {doc_before} -> {doc_after}")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
