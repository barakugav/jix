"""F030 repro (verifier): shape/layout gaps in jix-py/python/tests/tests_util.py.

Draws from the REAL strategies (imported from jix-py/python/tests/tests_util.py) and asserts what
the docstring's "mirrors test_util.rs" claim implies: Rust's shape_strategy draws 0-d shapes
(test_util.rs:461, `vec(0..=3, 0..=8)`), and Rust's array_strategy_from_data draws non-row-major,
padded, offset and zero-stride layouts (layout_strategy). Each check FAILS while the gap exists.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F030.py
"""

import os
import sys

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "jix-py", "python", "tests"))
sys.path.insert(0, TESTS)
import tests_util  # noqa: E402

SETTINGS = settings(
    max_examples=3000,
    database=None,
    deadline=None,
    derandomize=True,
    suppress_health_check=list(HealthCheck),
)

failures = []

# 1. shape_strategy: does it ever draw a 0-d shape?
ndims = []


@SETTINGS
@given(tests_util.shape_strategy())
def collect_shapes(shape):
    ndims.append(len(shape))


collect_shapes()
n0 = sum(1 for n in ndims if n == 0)
print(f"shape_strategy: {len(ndims)} draws, ndim histogram:", np.bincount(ndims).tolist())
if n0 == 0:
    failures.append(f"shape_strategy drew no 0-d shape in {len(ndims)} draws (Rust: ~0.3%)")

# 2. carray_strategy: is the numpy input ever anything but a packed C-order buffer?
layouts = {"c_contig": 0, "other": 0}


@SETTINGS
@given(st.data())
def collect_layouts(data):
    np_a, _za = data.draw(tests_util.carray_strategy(np.float32))
    if np_a.ndim >= 1 and np_a.size > 1 and np_a.flags.c_contiguous and np_a.flags.aligned:
        layouts["c_contig"] += 1
    elif np_a.size > 1:
        layouts["other"] += 1


collect_layouts()
print("carray_strategy numpy-input layouts (size > 1):", layouts)
if layouts["other"] == 0:
    failures.append("carray_strategy never draws a non-C-contiguous / unaligned / strided numpy input")

# 3. For contrast: the Python float any_element_strategy DOES draw NaN/inf (Rust any_strategy does not).
vals = []


@SETTINGS
@given(tests_util.any_element_strategy(np.float32))
def collect_vals(x):
    vals.append(x)


collect_vals()
arr = np.array(vals, dtype=np.float64)
print(f"Python any_element_strategy(float32): NaN {int(np.isnan(arr).sum())}, inf {int(np.isinf(arr).sum())} of {len(arr)}")

if failures:
    print("FAIL:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("OK")
