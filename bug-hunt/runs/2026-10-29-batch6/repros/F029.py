"""F029 (run 2026-10-29-batch6): the Python test strategies never produce a bool array whose
bytes are anything but 0/1, although since #89 any non-zero byte is a valid True.

Asserts the CORRECT behavior for test infrastructure that mirrors the library contract: the bool
array strategies (carray_strategy / carrays2_strategy with each element strategy) must be able to
draw at least one non-canonical byte (2, 0x80, 0xFF, ...). It fails while the gap exists.

Part 2 is the library-side check the gap leaves missing (it passes today): long non-canonical
bool arrays, zero-copy asarray and compact, through the vectorized paths, against the same ops
on the canonicalized array.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F029.py
"""

import os
import sys

import numpy as np
from hypothesis import HealthCheck, find, settings
from hypothesis.errors import NoSuchExample

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "jix-py", "python", "tests"))
sys.path.insert(0, TESTS)

import jix  # noqa: E402
import tests_util as tu  # noqa: E402


def has_noncanonical(np_a):
    return np_a.size > 0 and int(np.asarray(np_a).view(np.uint8).max()) > 1


SETTINGS = settings(max_examples=3000, database=None, suppress_health_check=list(HealthCheck), deadline=None)

failures = []

# ---- Part 1: the strategy gap ------------------------------------------------------------------
element_strategies = {
    "op_safe (default)": None,
    "any_element_strategy": tu.any_element_strategy(np.bool_),
    "comparable_element_strategy": tu.comparable_element_strategy(np.bool_),
}
for name, est in element_strategies.items():
    try:
        find(tu.carray_strategy(np.bool_, est), lambda p: has_noncanonical(p[0]), settings=SETTINGS)
        print(f"carray_strategy(bool, {name}): found a non-canonical byte")
    except NoSuchExample:
        print(f"GAP: carray_strategy(bool, {name}): 3000 draws, every byte is 0 or 1")
        failures.append(f"carray_strategy {name}")
    try:
        find(
            tu.carrays2_strategy(np.bool_, est),
            lambda p: has_noncanonical(p[0][0]) or has_noncanonical(p[1][0]),
            settings=SETTINGS,
        )
        print(f"carrays2_strategy(bool, {name}): found a non-canonical byte")
    except NoSuchExample:
        print(f"GAP: carrays2_strategy(bool, {name}): 3000 draws, every byte is 0 or 1")
        failures.append(f"carrays2_strategy {name}")

# ---- Part 2: library behavior on long non-canonical input (passes today) ------------------------
rng = np.random.default_rng(0)
n = 4099
raw = rng.integers(1, 256, n).astype(np.uint8)
raw[rng.random(n) < 0.4] = 0
raw[5::13] = np.array([2, 0x80, 0xFE, 0xFF, 4, 0x40], np.uint8)[np.arange(len(raw[5::13])) % 6]
nc = raw.view(np.bool_)  # non-canonical bytes
cn = (raw != 0)  # canonical
raw2 = np.roll(raw, 7)
nc2, cn2 = raw2.view(np.bool_), (raw2 != 0)

ops1 = {
    "invert": lambda a: ~a,
    "sum": lambda a: jix.sum(a),
    "mean": lambda a: jix.mean(a),
    "any": lambda a: jix.any(a),
    "all": lambda a: jix.all(a),
    "argmax": lambda a: jix.argmax(a),
    "argmin": lambda a: jix.argmin(a),
    "cast_i32": lambda a: jix.cast(a, np.int32),
    "cast_f64": lambda a: jix.cast(a, np.float64),
    "cast_u8": lambda a: jix.cast(a, np.uint8),
}
ops2 = {
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "maximum": lambda a, b: jix.maximum(a, b),
    "where_i64": lambda a, b: jix.where(a, np.arange(n, dtype=np.int64), np.full(n, -1, np.int64)),
}


def as_truth(x):
    x = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
    return (x.view(np.uint8) != 0) if x.dtype == np.bool_ else x


lib_diffs = 0
for wrap_name, wrap in {"asarray": jix.asarray, "compact": jix.compact,
                        "compact_b7": lambda x: jix.compact(x, params={"block_shape": [7]})}.items():
    for name, f in ops1.items():
        got, want = as_truth(f(wrap(nc))), as_truth(f(wrap(cn)))
        if not np.array_equal(got, want):
            lib_diffs += 1
            print(f"LIBRARY DIFF {wrap_name} {name}")
    for name, f in ops2.items():
        got, want = as_truth(f(wrap(nc), wrap(nc2))), as_truth(f(wrap(cn), wrap(cn2)))
        if not np.array_equal(got, want):
            lib_diffs += 1
            print(f"LIBRARY DIFF {wrap_name} {name}")
print(f"library check on {n}-element non-canonical bool arrays: {lib_diffs} differences")

assert lib_diffs == 0, "library mishandles non-canonical bool8"
assert not failures, f"no bool test strategy draws a non-canonical byte: {failures}"
