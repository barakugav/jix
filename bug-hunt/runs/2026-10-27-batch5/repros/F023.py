"""F023 repro: jix.log / Array.log accept an invalid base (1, 0, -0.0, negative) silently.

The docstring of jix.log says base "Must be a positive number other than 1", but nothing checks
it: the base only becomes a multiplier 1/ln(base) (inf, -0.0 or NaN) applied with a status-less
multiply, so every element turns into inf / NaN / a signed zero without passing through
jix.errstate. This script asserts the CORRECT behavior (a ValueError for those bases, or at
least an error under the default errstate, where divide/invalid = raise), so it FAILS while the
bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F023.py
"""

import math

import numpy as np

import jix

x = np.array([2.0, 1.0, 0.5], dtype=np.float64)

CALLS = [
    ("jix.log f64", lambda b: jix.log(x, base=b).numpy()),
    ("Array.log f64", lambda b: jix.compact(x).log(base=b).numpy()),
    ("jix.log f32", lambda b: jix.log(x.astype(np.float32), base=b).numpy()),
    ("jix.log f16", lambda b: jix.log(x.astype(np.float16), base=b).numpy()),
]


def run(base):
    """Returns the list of calls that returned silently for this base."""
    silent = []
    for label, call in CALLS:
        try:
            out = call(base)
        except Exception as e:  # correct: invalid base rejected
            print(f"{label} base={base}: raised {type(e).__name__}: {e}")
            continue
        print(f"{label} base={base}: NO ERROR, result {out.tolist()}")
        silent.append((label, base, out.tolist()))
    return silent


failures = []
for base in [1.0, 0.0, -0.0, -2.0]:  # outside the documented domain (positive, != 1)
    failures += run(base)

print("--- informational (NaN propagates in math.log too; log base inf -> 0 is the limit) ---")
for base in [float("nan"), float("inf")]:
    run(base)

print("--- reference: Python's math.log rejects the same bases ---")
for base in [1.0, 0.0, -0.0, -2.0]:
    try:
        math.log(2.0, base)
        print(f"math.log(2, {base}): no error")
    except Exception as e:
        print(f"math.log(2, {base}): {type(e).__name__}: {e}")

print("--- under the same default errstate an invalid ELEMENT does raise ---")
try:
    jix.log(np.array([-1.0])).numpy()
    print("jix.log([-1.0]): no error")
except Exception as e:
    print(f"jix.log([-1.0]): raised {type(e).__name__}: {e}")

assert not failures, f"{len(failures)} invalid-base calls returned silently: {failures}"
