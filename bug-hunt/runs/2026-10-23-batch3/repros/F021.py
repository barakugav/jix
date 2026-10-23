"""F021: an empty float / complex jix.sum returns -0.0 (sign bit set) instead of +0.0.

Asserts the correct (NumPy-matching, documented "returns 0") behavior, so it FAILS while the bug exists.
"""

import numpy as np

import jix

failures = []

for dtype in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
    a = np.zeros((3, 0), dtype=dtype)
    got = jix.sum(jix.asarray(a), dim=1).numpy()
    want = np.sum(a, axis=1)
    got_re, want_re = np.real(got), np.real(want)
    print(f"{np.dtype(dtype).name:>10}: jix {got!r:<40} signbit(re)={np.signbit(got_re)}  numpy {want!r}")
    if np.signbit(got_re).any() != np.signbit(want_re).any():
        failures.append(np.dtype(dtype).name)

# Reduce over all dims of an empty 1-d array.
s = jix.sum(jix.asarray(np.zeros(0))).numpy()
print("sum(empty 1-d f64):", repr(s), " 1/x:", np.float64(1.0) / s, " numpy:", np.sum(np.zeros(0)))
if np.signbit(s):
    failures.append("float64 all-dims")

assert not failures, f"empty sum returned -0.0 for: {failures}"
print("OK")
