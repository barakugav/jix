"""F028: jix-py/python/tests/tests_util.py::assert_array_matches is weaker than the Rust helper
it claims to mirror (jix/src/util/test_util.rs::assert_array_matches).

Asserts the CORRECT behavior of a test helper, so it fails while the gap exists:
  (1) a dtype mismatch must fail the assertion (the Rust helper does into_typed::<T>().unwrap()),
  (2) a 0-d vs (n,) shape mismatch must fail (the Rust helper's elementwise_eq checks shapes),
  (3) the sub-range check must be able to draw a slice that starts at the dim length (a[n:n]),
      as the Rust sub_range_strategy does (0..=len for both ends) - the K21 shape,
  (4) the sub-range check must not be skipped for arrays with a zero-length dim (K21 fixed in
      c009869, so the skip is stale).

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F028.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "jix-py", "python", "tests"))

import hypothesis  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import jix  # noqa: E402
from tests_util import assert_array_matches, sub_range_strategy  # noqa: E402

failures = []


def expect_mismatch_rejected(label, actual, expected):
    try:
        assert_array_matches(actual, expected)
    except AssertionError:
        print(f"ok   {label}: rejected")
        return
    print(f"GAP  {label}: assert_array_matches PASSED")
    failures.append(label)


# (1) dtype is not checked
expect_mismatch_rejected(
    "int64 result vs int8 expected",
    jix.asarray(np.array([1, 2], np.int64)),
    np.array([1, 2], np.int8),
)
expect_mismatch_rejected(
    "uint8 result vs bool expected (the K75 shape)",
    jix.asarray(np.array([1, 0, 1], np.uint8)),
    np.array([True, False, True]),
)

# (2) a 0-d operand is broadcast instead of shape-checked
expect_mismatch_rejected(
    "0-d result vs (3,) expected",
    jix.asarray(np.float32(2.0)),
    np.array([2.0, 2.0, 2.0], np.float32),
)
expect_mismatch_rejected(
    "(4,) result vs 0-d expected (a full reduction that kept a dim)",
    jix.asarray(np.full(4, 5.0)),
    np.float64(5.0),
)
expect_mismatch_rejected(
    "(1,1) result vs 0-d expected",
    jix.asarray(np.array([[7]], np.int32)),
    np.int32(7),
)

# (3) the Python sub_range_strategy can never start a slice at the dim length
try:
    found = hypothesis.find(
        sub_range_strategy((3,)),
        lambda s: s[0].start == 3,
        settings=settings(max_examples=2000, database=None),
    )
    print(f"ok   sub_range_strategy drew a start == len slice: {found}")
except hypothesis.errors.NoSuchExample:
    print("GAP  sub_range_strategy((3,)) never draws start == 3 (a[3:3]); Rust draws 0..=len")
    failures.append("sub_range start == len")

# K21 is fixed, so those slices now work on a real array:
z = jix.compact(np.zeros((0, 3), np.float32))
a = jix.compact(np.arange(6, dtype=np.float32).reshape(2, 3))
print("     K21 fixed: z[:] ->", z[:].shape, " z[0:0,1:2] ->", z[0:0, 1:2].shape,
      " a[2:2] ->", a[2:2].shape, " a[:,3:] ->", a[:, 3:].shape)


# (4) the sub-range check is skipped for any array with a zero-length dim
class Probe:
    """Stands in for a jix.Array: full read is right, counts sub-range reads."""

    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape
        self.subreads = 0

    def numpy(self):
        return self.arr

    def __getitem__(self, idx):
        self.subreads += 1
        return self.arr[idx]


probes = {}


@settings(max_examples=20, database=None)
@given(st.data())
def run_with_data(data):
    for shape in [(0, 3), (3, 0, 2), (4, 3)]:
        p = probes.setdefault(shape, Probe(np.zeros(shape, np.float32)))
        assert_array_matches(p, p.arr, data=data)


run_with_data()
for shape, p in probes.items():
    status = "ok  " if p.subreads > 0 else "GAP "
    print(f"{status} shape {shape}: {p.subreads} sub-range reads over 20 hypothesis examples")
    if p.subreads == 0:
        failures.append(f"no sub-range check for shape {shape}")

print()
if failures:
    print(f"FAIL: {len(failures)} gap(s): {failures}")
    sys.exit(1)
print("PASS")
