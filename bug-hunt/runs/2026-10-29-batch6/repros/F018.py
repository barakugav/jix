"""F018 (run 2026-10-29-batch6): jix.log(f16, base) for a valid base near 1 rounds a finite
result to +-inf with no Overflow report.

For a base other than e / 2 / 10, jix-py/src/ops/op1.rs computes ln(x) in f64 and narrows
`ln(x) * (1 / ln(base))` to f16 inside `Array::map`, which has no ArithmeticOpStatus channel, and
the scalar Cast f64 -> f16 never reports. So a true value beyond f16's range (e.g. 69081 for
x = 1000, base = 1.0001) becomes inf silently, even under jix.errstate(over="raise"), while the
same quotient computed with jix.divide, and the f16 exp / tan / power kernels (K106 fix), report it.

This script asserts the CORRECT behavior (an Overflow report), so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F018.py
"""

import math
import warnings

import numpy as np

import jix

h = np.array([1000.0, 65504.0], dtype=np.float16)
failures = []


def check(name, f):
    """Return True if `f` reports an overflow both by default (warning) and under over='raise'."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = f()
    warned = any("overflow" in str(x.message) for x in w)
    with jix.errstate(over="raise"):
        try:
            f()
            raised = False
        except FloatingPointError:
            raised = True
    print(f"{name}: result {res!r}, default warning: {warned}, over='raise' raised: {raised}")
    return warned and raised


for base in [1.0001, 0.9999]:
    true = np.log(h.astype(np.float64)) / math.log(base)
    print(f"true log_{base}({h.tolist()}) = {true.tolist()} (f16 max 65504)")
    if not check(f"jix.log(f16, base={base})", lambda: jix.log(h, base=base).numpy()):
        failures.append(f"jix.log(f16, base={base}) overflowed to inf without an Overflow report")
    if not check(f"Array.log(f16, base={base})", lambda: jix.compact(h).log(base=base).numpy()):
        failures.append(f"Array.log(f16, base={base}) overflowed to inf without an Overflow report")

# Controls: the same quotient through jix.divide, and f16 kernels that round to inf, all report.
ln_base = np.float16(math.log(1.0001))
check("control jix.divide(jix.log(f16), f16 ln(1.0001))", lambda: jix.divide(jix.log(h), ln_base).numpy())
check("control jix.exp(f16 12)", lambda: jix.exp(np.array([12.0], np.float16)).numpy())
check("control jix.power(f16 300, 3)", lambda: jix.power(np.array([300.0], np.float16), 3).numpy())

if failures:
    raise AssertionError("; ".join(failures))
print("OK")
