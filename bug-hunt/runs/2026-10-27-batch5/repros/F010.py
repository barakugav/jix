"""Repro for F010 (run 2026-10-27-batch5).

jix.std of float16 / float32 / complex64 rounds the float64 variance to the narrow output dtype
before the square root: inf, 0 or tens of ulps off where the correctly rounded std is
representable. Asserts the CORRECT result (within 1 ulp of the exact std rounded once), so it
fails while the bug exists. NumPy's values are printed for context only.

Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F010.py
"""

import math
from fractions import Fraction

import numpy as np

import jix


def exact_std(xs):
    fr = [Fraction(float(v)) for v in xs]
    n = len(fr)
    m = sum(fr) / n
    var = sum((v - m) ** 2 for v in fr) / n
    # sqrt in high precision: float(var) is exact enough for these inputs once scaled
    return math.sqrt(var.numerator) / math.sqrt(var.denominator)


def ulps(a, b, dt):
    ia = np.array([a], dtype=dt).view({np.float16: np.int16, np.float32: np.int32}[dt])[0]
    ib = np.array([b], dtype=dt).view({np.float16: np.int16, np.float32: np.int32}[dt])[0]
    return abs(int(ia) - int(ib))


cases = []
x = np.array([1.0, 1.0 + 2**-10, 1.0 + 2**-9], np.float16)
cases.append(("f16 near 1.0", x, np.float16))
a = np.float16(0.01)
cases.append(("f16 near 0.01", np.array([a, np.nextafter(a, np.float16(1))], np.float16), np.float16))
cases.append(("f16 [-300, 300]", np.array([-300.0, 300.0], np.float16), np.float16))
cases.append(("f32 [-2e19, 2e19]", np.array([-2e19, 2e19], np.float32), np.float32))
cases.append(("f32 [1e-25, 2e-25]", np.array([1e-25, 2e-25], np.float32), np.float32))
cases.append(("c64 [-2e19, 2e19]", np.array([-2e19, 2e19], np.complex64), np.float32))

failures = []
for name, arr, out_dt in cases:
    got = jix.std(jix.asarray(arr)).numpy()[()]
    want = out_dt(exact_std(arr.real if np.iscomplexobj(arr) else arr))
    via_f64 = out_dt(jix.std(jix.asarray(arr.astype(np.complex128 if np.iscomplexobj(arr) else np.float64))).numpy()[()])
    npv = np.std(arr)
    ok = np.isfinite(got) and ulps(got, want, out_dt) <= 1
    print(
        f"{name:20s} jix.std={got!r:>14} want={want!r:>14} "
        f"jix.std(f64)->{out_dt.__name__}={via_f64!r:>14} numpy={npv!r:>14} {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(name)

assert not failures, f"std not correctly rounded for: {failures}"
