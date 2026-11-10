#!/usr/bin/env bash
# Run the docs-review gates on the uncommitted changes of the working tree.
#
# Always: the change is docs-only, and the tree is ASCII-only.
# Per touched package: formatting, the docs build with warnings as errors, and the doc examples.
#
# Usage (from anywhere in the repo, with the venv active): scripts/docs_review/gates.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

changed=$(git diff --name-only HEAD)
touched() { grep -qE "$1" <<<"$changed"; }
step() { printf '\n== %s\n' "$1"; }

step "docs-only change"
python scripts/docs_review/check_docs_only.py --allow scripts/docs_review/

step "ascii only"
python scripts/check_only_ascii.py

if touched '^jix/src/'; then
    step "jix: fmt, rustdoc, doc-tests"
    (
        cd jix
        cargo fmt --all -- --check
        RUSTDOCFLAGS="-D warnings" cargo doc --all-features --quiet
        cargo test --doc --all-features --quiet
    )
fi

if touched '^jix-macros/'; then
    step "jix-macros: fmt, rustdoc, doc-tests"
    (
        cd jix-macros
        cargo fmt --all -- --check
        RUSTDOCFLAGS="-D warnings" cargo doc --quiet
        cargo test --doc --quiet
    )
fi

if touched '^(jix-py/|README\.md$)'; then
    step "jix-py: fmt, build, stubs, mkdocs, python examples"
    (
        cd jix-py
        cargo fmt --all -- --check
        # without debug info: the debug build of this crate needs more memory than CI-sized hosts have
        CARGO_PROFILE_DEV_DEBUG=0 maturin develop --uv --quiet
        CARGO_PROFILE_DEV_DEBUG=0 maturin generate-stubs --out python
        mkdocs build --strict --quiet
    )
    python scripts/docs_review/run_py_examples.py
fi

printf '\nall gates passed\n'
