"""F019: f64 jix.log(x, base) for a base other than e/2/10 multiplies ln(x) by a rounded
reciprocal 1/ln(base), so exact powers come out 1 ulp off and the error is larger than
ln(x)/ln(base).

Asserts the CORRECT behavior (exact results on exact powers of the base, as math.log(x, b),
np.log(x)/np.log(b) and jix's own f32 path give), so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F019.py
"""

import math
import random
from decimal import Decimal, getcontext

import numpy as np

import jix


def jlog(x, base, dtype=np.float64):
    return jix.log(np.array([x], dtype=dtype), base=base).numpy()[0]


failures = []

# 1. Concrete exact powers.
cases = [(64.0, 4, 3.0), (4.0**-7, 4, -7.0), (4096.0, 16, 3.0), (0.125, 0.5, 3.0), (2.0**-10, 0.25, 5.0)]
for x, base, want in cases:
    got = jlog(x, base)
    got32 = jlog(x, base, np.float32)
    ref_math = math.log(x, base)
    ref_np = np.log(x) / np.log(base)
    ok = got == want
    print(
        f"jix.log({x!r}, base={base}) f64 = {got!r}  f32 = {got32!r}  math.log = {ref_math!r}"
        f"  np.log/np.log = {ref_np!r}  {'OK' if ok else 'MISMATCH'}"
    )
    if not ok:
        failures.append(f"log({x!r}, base={base}) = {got!r} != {want!r}")

print(f"int(jix.log(64.0, base=4)) = {int(jlog(64.0, 4))}  (math: {int(math.log(64.0, 4))})")

# 2. Exact-power counts over k in [-40, 40].
for base in [4.0, 8.0, 16.0, 32.0, 0.5, 0.25, 3.0, 5.0, 6.0, 7.0]:
    ks = [k for k in range(-40, 41) if base**k != 0 and math.isfinite(base**k)]
    xs = np.array([base**k for k in ks])
    # only keep x that is an exact power (base**k exactly representable)
    exact = [(k, x) for k, x in zip(ks, xs) if Decimal(float(x)) == Decimal(base) ** k]
    xs = np.array([x for _, x in exact])
    kk = np.array([k for k, _ in exact], dtype=np.float64)
    got = jix.log(xs, base=base).numpy()
    lnln = np.log(xs) / np.log(base)
    l2l2 = np.log2(xs) / np.log2(base)
    print(
        f"base {base}: {len(xs)} exact powers; inexact: jix {int(np.sum(got != kk))}"
        f"  ln/ln {int(np.sum(lnln != kk))}  log2/log2 {int(np.sum(l2l2 != kk))}"
    )

# 3. Random inputs vs a 50-digit decimal reference.
getcontext().prec = 50


def ulp_err(v, ref):
    return float(abs(Decimal(float(v)) - ref) / Decimal(math.ulp(float(ref))))


rng = random.Random(0)
xs = np.array([math.exp(rng.uniform(-50, 50)) for _ in range(20000)])
for base in [5.0, 7.0]:
    got = jix.log(xs, base=base).numpy()
    lnln = np.log(xs) / np.log(base)
    lnb = Decimal(base).ln()
    errs_j, errs_n = [], []
    for x, g, n in zip(xs, got, lnln):
        ref = Decimal(float(x)).ln() / lnb
        if ref == 0:
            continue
        errs_j.append(ulp_err(g, ref))
        errs_n.append(ulp_err(n, ref))
    errs_j, errs_n = np.array(errs_j), np.array(errs_n)
    print(
        f"base {base}: >1 ulp: jix {int(np.sum(errs_j > 1))}  ln/ln {int(np.sum(errs_n > 1))} of {len(errs_j)};"
        f" max ulp jix {errs_j.max():.3f} ln/ln {errs_n.max():.3f}; mean jix {errs_j.mean():.3f} ln/ln {errs_n.mean():.3f}"
    )

assert not failures, failures
