# Verifier repro (run 2026-11-15-batch7-1, finding K158).
# A left-deep lazy chain `x = x + base` (or `x = x + 1`) holds per-op cache-id sets that are
# deep-copied at every op, so its memory is quadratic in the chain depth.
# Asserts the CORRECT behavior (memory ~linear in depth); fails while the bug exists.
# Each measurement runs in a fresh subprocess and reports the growth of peak RSS while building.
import subprocess
import sys

CHILD = r"""
import resource, sys
import numpy as np
import jix

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

mode, n = sys.argv[1], int(sys.argv[2])
base = jix.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
x = base
before = rss_mb()
for i in range(n):
    x = x + 1 if mode == "scalar" else x + base
    if mode == "transpose":
        x = x.T
print(rss_mb() - before)
"""


def grow_mb(mode, n):
    out = subprocess.run([sys.executable, "-I", "-c", CHILD, mode, str(n)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


results = {}
for mode in ("array", "scalar", "transpose"):
    for n in (1000, 4000):
        results[mode, n] = grow_mb(mode, n)
        print(f"{mode:9s} n={n}: +{results[mode, n]:.1f} MB")
# "transpose" is the control: PermuteDims resets the id set after every add, so it stays linear.

for mode in ("array", "scalar"):
    small, big = results[mode, 1000], results[mode, 4000]
    # linear growth: 4x the ops -> about 4x the memory; quadratic -> about 16x
    assert big <= 6 * max(small, 1.0), f"{mode}: 4x deeper chain grew RSS {big:.1f} MB vs {small:.1f} MB"
print("OK")
