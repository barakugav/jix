#!/usr/bin/env bash
# Prepare this checkout for a bug-hunt run.
#
# Hunters and verifiers run experiments in parallel against ONE shared checkout and build cache.
# Everything they use must be prebuilt here, so that no agent ever triggers a long build:
#   - the jix library (debug), which every `jix/tests/<name>.rs` repro links against (~2 s rebuild)
#   - the jix unit-test binary, so existing inline tests can be run by name
#   - the Miri build of the library (nightly), used sparingly for memory-safety repros
#   - the Python bindings (debug build) installed into .venv
#   - an i686-unknown-linux-musl (32-bit) build, so 32-bit claims can be tested natively
#
# Usage: bash bug-hunt/setup.sh [--no-miri] [--no-python] [--no-i686]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

MIRI=1
PYTHON=1
I686=1
for arg in "$@"; do
    case "$arg" in
        --no-miri) MIRI=0 ;;
        --no-python) PYTHON=0 ;;
        --no-i686) I686=0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

SMOKE=jix/tests/hunt_setup_smoke.rs
mkdir -p jix/tests
cp bug-hunt/templates/repro_template.rs "$SMOKE"
trap 'rm -f "$SMOKE"' EXIT

echo "== Rust: library + unit-test binary (debug, all features)"
cargo test --manifest-path jix/Cargo.toml --all-features --no-run
cargo test --manifest-path jix/Cargo.toml --all-features --test hunt_setup_smoke

if [ "$MIRI" = 1 ]; then
    echo "== Miri: nightly toolchain + cached Miri build of the library"
    if ! cargo +nightly miri --version >/dev/null 2>&1; then
        rustup toolchain install nightly --profile minimal --component miri,rust-src
    fi
    python jix/run_miri.py --all-features --test hunt_setup_smoke
fi

if [ "$PYTHON" = 1 ]; then
    echo "== Python: .venv + debug build of the bindings"
    if [ ! -d .venv ]; then
        uv venv --python 3.13 .venv
    fi
    export VIRTUAL_ENV="$ROOT/.venv"
    uv pip install --python .venv/bin/python -r scripts/dev_requirements.txt "numpy~=2.0"
    # A debug build keeps overflow checks and debug_asserts on. Without debuginfo it peaks at
    # ~7.5 GB of RAM; with full debuginfo it can get OOM-killed on a 16 GB machine. Do not run it
    # next to other large builds.
    CARGO_PROFILE_DEV_DEBUG=0 .venv/bin/maturin develop -m jix-py/Cargo.toml
    # Run from outside the repo root: there, the crate directory jix/ shadows the package.
    (cd /tmp && "$ROOT/.venv/bin/python" -c \
        "import jix, numpy as np; a = jix.compact(np.arange(6.0).reshape(2, 3)); print('python ok:', a.shape)")
fi

if [ "$I686" = 1 ]; then
    echo "== i686: 32-bit musl target (zig compiles zstd, clang + lld link, binaries run natively)"
    if ! command -v clang >/dev/null || ! command -v ld.lld >/dev/null; then
        echo "   skipped: needs clang and ld.lld on PATH"
    else
        rustup target add i686-unknown-linux-musl
        [ -d .venv ] || uv venv --python 3.13 .venv
        uv pip install --python .venv/bin/python ziglang
        bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test hunt_setup_smoke
    fi
fi

echo "== Ready. Start a run from Claude Code, e.g.: run the bug-hunt workflow with runId $(date +%F)"
