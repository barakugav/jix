"""F043 - SIMD level coverage of the test suite (test-infrastructure gap, no library defect).

Asserts the coverage the suite SHOULD have, so it exits 1 while the gap exists:
  1. for_each_simd_level! reports the levels it skips (today it skips them silently).
  2. Level-dependent driver code (the element-wise pipeline, the reduction folds and update_bulk)
     is run at more than the host's Level::new() level: either a per-level test in those modules
     or a CI leg that builds with fearless_simd's `--cfg disable_dispatch_*`.
  3. The bit-shuffle filter has a per-level test like byte_shuffle's.
  4. SimdTestValues pairs each edge case with more than one partner (the op2 body tests pair
     values(0)[i] with values(5)[i] only, so e.g. (MIN, -1) or (inf, inf) never meet).

Demonstration (not an assertion): if valgrind is installed, the prebuilt lib test binary is run
under valgrind, where fearless_simd's Level::new() is AVX2 (valgrind hides AVX-512). The per-level
tests still print "ok", although their Avx512 arm never ran.

Run from the repo root: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F043.py
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "jix" / "src"


def read(rel):
    return (ROOT / rel).read_text()


def test_module(text):
    """The `#[cfg(test)] mod tests` part of a source file (everything after it)."""
    i = text.find("#[cfg(test)]\nmod tests")
    return text[i:] if i >= 0 else ""


gaps = []

# 1. Silent skip in for_each_simd_level!.
tu = read("jix/src/util/test_util.rs")
m = re.search(r"macro_rules! for_each_simd_level \{(.*?)\npub\(crate\) use for_each_simd_level;", tu, re.S)
assert m, "for_each_simd_level! not found"
body = m.group(1)
reports = any(k in body for k in ("eprintln!", "println!", "panic!", "env::var", "std::env"))
if not reports:
    gaps.append(
        "for_each_simd_level! (test_util.rs) runs only the levels Level::new() proves and reports none "
        "of the ones it skips (no eprintln/panic/env override in the macro)"
    )

# 2. Driver code only at Level::new().
per_level_markers = ("for_each_simd_level", "as_sse2(", "as_sse4_2(", "as_avx2(", "as_avx512(", "as_neon(")
ci = "\n".join(p.read_text() for p in (ROOT / ".github" / "workflows").glob("*.y*ml"))
ci_forces_level = "disable_dispatch_" in ci
for rel, what in [
    ("jix/src/storage/elementwise_pipeline.rs", "element-wise pipeline (inner_loop_per_level lane count)"),
    ("jix/src/ops/reduction.rs", "reduction folds / update_bulk level branches"),
]:
    text = read(rel)
    dispatches = text.count("dispatch!(SimdLevel::new()") + text.count("dispatch_baseline!(")
    tested = any(k in test_module(text) for k in per_level_markers)
    if dispatches and not tested and not ci_forces_level:
        gaps.append(
            f"{what}: {dispatches} dispatch site(s) in {rel}, no per-level test and no CI "
            f"`--cfg disable_dispatch_*` leg -> only the host's Level::new() level is ever run"
        )

# Level-gated branches that no test on an AVX2 / AVX-512 host can reach.
red = read("jix/src/ops/reduction.rs")
gated = re.findall(r"impl_product!\((\w+), (\w+), simd = disable_on_x86_64\(([^)]*)\)\);", red)
if gated and not ci_forces_level:
    gaps.append(
        "reduction Product update_bulk falls back to the scalar loop only on "
        f"[{gated[0][2]}] for {', '.join(f'{a}->{b}' for a, b, _ in gated)}: that branch never runs "
        "on an AVX2/AVX-512 host (and the SIMD branch never runs on an SSE-only one)"
    )

# 3. bit-shuffle has no per-level test (byte-shuffle does).
bs = read("jix/src/codec/filter/bit_shuffle.rs")
bys = read("jix/src/codec/filter/byte_shuffle.rs")
if any(k in test_module(bys) for k in per_level_markers) and not any(
    k in test_module(bs) for k in per_level_markers
):
    gaps.append("bit_shuffle.rs has no per-level test (byte_shuffle.rs has encode/decode_simd_all_levels)")

# 4. SimdTestValues pairing: values(0)[i] vs values(5)[i], mirroring test_util.rs simd_test_values.
INT_EDGES = ["MIN", "MAX", "0", "1", "-1", "MIN+1", "MAX-1", "2", "7"]
FLOAT_EDGES = [
    "0", "-0", "0.5", "-0.5", "1.5", "-1.5", "2.5", "-2.5", "0.49999997", "-0.49999997", "1", "-1",
    "big-0.5", "-(big-0.5)", "big", "big+1", "1e30", "-1e30", "MAX", "MIN", "MIN_POSITIVE",
    "MIN_POSITIVE/4", "inf", "-inf", "NaN", "-NaN",
]
for name, edges, want in [("int", INT_EDGES, ("MIN", "-1")), ("float", FLOAT_EDGES, ("inf", "inf"))]:
    n = len(edges)
    pairs = {(edges[i], edges[(i + 5) % n]) for i in range(n)}
    if want not in pairs:
        gaps.append(
            f"SimdTestValues ({name}): only {len(pairs)} edge pairs of {n * n} reach each op2 kernel; "
            f"{want} is never paired"
        )

# Demonstration under valgrind (Level::new() == AVX2 there): per-level tests still say "ok".
if shutil.which("valgrind"):
    out = subprocess.run(
        ["cargo", "test", "--manifest-path", str(ROOT / "jix" / "Cargo.toml"), "--all-features", "--lib",
         "--no-run", "--message-format=json"],
        capture_output=True, text=True,
    )
    exe = None
    for line in out.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") and msg["target"]["name"] == "jix":
            exe = msg["executable"]
    if exe:
        r = subprocess.run(
            ["valgrind", "-q", exe, "--exact", "ops::cmp::tests::simd_bodies_all_levels"],
            capture_output=True, text=True, timeout=900,
        )
        last = [ln for ln in r.stdout.splitlines() if "simd_bodies_all_levels" in ln or "test result" in ln]
        print("under valgrind (Level::new() = AVX2, Avx512 arm skipped silently):")
        for ln in last:
            print("   ", ln)

for g in gaps:
    print("GAP ", g)
if gaps:
    print(f"FAIL: {len(gaps)} gap(s)")
    sys.exit(1)
print("ok: no gap")
