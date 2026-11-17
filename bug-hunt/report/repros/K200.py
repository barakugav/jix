"""K200: a Python bool is accepted as a shape entry by reshape / broadcast (True -> 1, False -> 0).

numpy rejects a bool shape entry with TypeError, and jix itself rejects bool for every dim
argument (type_hint.rs `Dim`, test_getitem.py::test_bool_dim_raises) and every index
(test_bool_index_raises). This repro asserts the CORRECT behavior (TypeError), so each
check FAILS while the bug exists.

Run: .venv/bin/python bug-hunt/report/repros/K200.py
"""

import sys

import numpy as np

import jix

print("numpy", np.__version__)

x = np.arange(6, dtype=np.int32)
a = jix.asarray(x)
z = np.int32(1)

# Reference: numpy rejects every one of these.
np_cases = [
    ("np.reshape(x, (True, -1))", lambda: np.reshape(x, (True, -1))),
    ("np.reshape(z, True)", lambda: np.reshape(z, True)),
    ("np.broadcast_to(x, (True, 6))", lambda: np.broadcast_to(x, (True, 6))),
    ("np.broadcast_to(x, (False, 6))", lambda: np.broadcast_to(x, (False, 6))),
]
for label, fn in np_cases:
    try:
        r = fn()
        print(f"REF  {label}: accepted, shape={np.shape(r)}")
    except TypeError as e:
        print(f"REF  {label}: TypeError: {e}")

# Control: jix's own dim policy rejects bool.
try:
    jix.sum(a, True)
    print("CTRL jix.sum(a, True): accepted")
except TypeError as e:
    print(f"CTRL jix.sum(a, True): TypeError: {e}")

cases = [
    ("jix.reshape(a, (True, -1))", lambda: jix.reshape(a, (True, -1))),
    ("jix.reshape(jix.asarray(z), True)", lambda: jix.reshape(jix.asarray(z), True)),
    ("a.reshape((2, True, 3))", lambda: a.reshape((2, True, 3))),
    ("jix.broadcast(a, (True, 6))", lambda: jix.broadcast(a, (True, 6))),
    ("a.broadcast((False, 6))", lambda: a.broadcast((False, 6))),
    ("jix.broadcast(a, (True, True, 6))", lambda: jix.broadcast(a, (True, True, 6))),
]
failures = 0
for label, fn in cases:
    try:
        r = fn()
    except TypeError as e:
        print(f"PASS {label}: TypeError: {e}")
        continue
    failures += 1
    print(f"FAIL {label}: expected TypeError, got shape={r.shape} values={r.numpy().ravel().tolist()}")

print(f"{failures} of {len(cases)} checks FAIL")
sys.exit(1 if failures else 0)
