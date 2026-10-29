# Repro for F008 (run 2026-10-29-batch6). The crashing steps run in subprocesses: while the bug exists
# the child interpreter dies with SIGSEGV (exit status -11) instead of printing the value / finishing.
#
# A lazy chain built by `x = x + 1.0` in a loop is a linked list of type-erased op nodes; reading it
# (numpy/item/compact/write_array) and dropping it both recurse once per op level on the native stack,
# with no depth limit and no Python-level RecursionError.
#
# Expected (asserted): every child exits 0 and the read returns [n+1, n+2].
import subprocess
import sys

CHILD = r"""
import sys
import numpy as np
import jix
n = int(sys.argv[1]); what = sys.argv[2]
x = jix.asarray(np.array([1.0, 2.0]))
for _ in range(n):
    x = x + 1.0
if what == "read":
    print("value", x.numpy().tolist(), flush=True)
    import os; os._exit(0)  # skip the drop, only the read is under test
elif what == "drop":
    del x
    print("dropped", flush=True)
"""

CASES = [("read", 1000), ("read", 5000), ("read", 10000), ("drop", 20000), ("drop", 100000)]

failures = []
for what, n in CASES:
    p = subprocess.run([sys.executable, "-I", "-c", CHILD, str(n), what], capture_output=True, text=True, timeout=600)
    out = p.stdout.strip() or (p.stderr.strip().splitlines() or [""])[-1]
    print(f"{what:4s} n={n:6d}: exit={p.returncode} {out[:80]}")
    if what == "read":
        ok = p.returncode == 0 and out == f"value {[1.0 + n, 2.0 + n]}"
    else:
        ok = p.returncode == 0 and out == "dropped"
    if not ok:
        failures.append((what, n, p.returncode))

assert not failures, f"deep lazy chains crashed the interpreter: {failures}"
print("all ok")
