"""K186: since the K168 fix (#107), jix.isclose reports finite operands whose difference
overflows as NOT close even when the documented formula
|a - b| <= max(atol, rtol * max(|a|, |b|)) holds exactly: atol = inf, rtol = inf, any rtol >= 2,
(f16) any Python atol >= 65520, which casts to inf, and
rtol in (1, 2) when rtol >= 1 + min(|a|,|b|)/max(|a|,|b|). Asserts the documented result; fails
while the bug exists. Reference computed exactly with Python's arbitrary-precision Fractions."""

from fractions import Fraction

import numpy as np

import jix


def ref(a, b, rtol, atol):
    # Documented formula, evaluated exactly (finite a, b; tolerances may be inf).
    a, b = Fraction(float(a)), Fraction(float(b))
    diff = abs(a - b)
    if atol == np.inf or (rtol == np.inf and max(abs(a), abs(b)) > 0):
        return True
    return diff <= max(Fraction(float(atol)), Fraction(float(rtol)) * max(abs(a), abs(b)))


cases = [
    (np.float64, [np.finfo(np.float64).max, 1.0], [-np.finfo(np.float64).max, -1.0]),
    (np.float32, [np.finfo(np.float32).max, 1.0], [-np.finfo(np.float32).max, -1.0]),
    (np.float16, [40000.0, 1.0], [-40000.0, -1.0]),
    # rtol in (1, 2): exact |a - b| = |a| + |b| <= rtol * max iff rtol >= 1 + min/max (1.25, 1.4)
    (np.float64, [np.finfo(np.float64).max], [-np.finfo(np.float64).max / 4]),
    (np.float16, [50000.0], [-20000.0]),
]
tols = [
    {"rtol": 0.0, "atol": np.inf},
    {"rtol": np.inf, "atol": 0.0},
    {"rtol": 2.0, "atol": 0.0},
    {"rtol": 3.0, "atol": 0.0},
    {"rtol": 1.5, "atol": 0.0},  # K168 must stay fixed
    {"rtol": 0.0, "atol": 1e6},  # f16: atol casts to inf
    {"rtol": 0.0, "atol": 70000.0},  # f16: atol casts to inf, but 80000 > 70000 -> False
]
bad = []
for dt, av, bv in cases:
    a = np.array(av, dtype=dt)
    b = np.array(bv, dtype=dt)
    for kw in tols:
        exp = np.array([ref(x, y, kw["rtol"], kw["atol"]) for x, y in zip(a, b)])
        got = jix.isclose(jix.asarray(a), jix.asarray(b), **kw).numpy()
        with np.errstate(all="ignore"):
            npy = np.isclose(a, b, **kw)
        flag = "" if np.array_equal(got, exp) else "   <-- WRONG"
        print(f"{np.dtype(dt).name:8s} {kw}: jix {got} expected {exp} numpy {npy}{flag}")
        if flag:
            bad.append((np.dtype(dt).name, kw))
assert not bad, f"wrong for {bad}"
