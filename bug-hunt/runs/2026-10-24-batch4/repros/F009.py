"""F009 repro: complex mean overflows for finite inputs whose mean is finite.

Complex mean finalizes with num-complex's textbook division sum / Complex(n, 0)
(jix/src/ops/reduction.rs:1766), which computes sum.re * n and sum.im * n first; those overflow
once |sum component| > f64::MAX / n. numpy (and jix's own float64 mean) return the finite mean.
Asserts the correct behavior, so it exits 1 while the bug exists.
"""

import sys

import numpy as np

import jix

cases = [
    np.array([1.7e308 + 0j, 0j], dtype=np.complex128),
    np.array([1.7e308 - 1.7e308j, 0j], dtype=np.complex128),
    np.full(1_000_000, 1e302 + 1e302j, dtype=np.complex128),
]
failed = 0
print("numpy", np.__version__)
for a in cases:
    want = np.mean(a)
    for kind, x in (("numpy-input", a), ("compact", jix.compact(a))):
        got = jix.mean(x).numpy()[()]
        ok = np.isfinite(got) and np.allclose(got, want, rtol=1e-12)
        failed += not ok
        print(f"len={a.size:>7} {kind:<11} first={a[0]!r}: numpy={want!r} jix={got!r} {'ok' if ok else 'WRONG'}")
# control: float64 mean of the real parts
re = jix.mean(np.array([1.7e308, 0.0])).numpy()[()]
print(f"control float64 mean([1.7e308, 0.0]) = {re!r}")
# control: complex64 is not reachable (its f64 accumulator never gets near f64::MAX)
c64 = jix.mean(np.array([3e38 + 3e38j, 3e38 + 3e38j], dtype=np.complex64)).numpy()[()]
print(f"control complex64 mean([3e38+3e38j]*2) = {c64!r}")
if failed:
    print(f"FAIL: {failed} wrong complex means")
    sys.exit(1)
print("PASS")
