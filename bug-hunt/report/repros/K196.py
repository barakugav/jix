"""K196: an integer index outside i64 raises TypeError ('only integers, slices ...') instead of
IndexError.

Asserts the CORRECT behavior (IndexError, as for an in-i64 out-of-bounds int and for a slice
bound of the same size, and as the jix.slice / Array.numpy docstrings promise), so it fails
while the bug exists.
"""

import sys

import numpy as np

import jix

a = jix.compact(np.arange(5))
a2 = jix.compact(np.arange(12).reshape(3, 4))

# Controls: in-i64 out-of-bounds int and huge slice bounds raise IndexError.
controls = [
    ("a[2**63 - 1]", lambda: a[2**63 - 1]),
    ("a[0:2**63]", lambda: a[0 : 2**63]),
    ("a[0:2**100]", lambda: a[0 : 2**100]),
]
cases = [
    ("a[2**63]", lambda: a[2**63]),
    ("a[-(2**63) - 1]", lambda: a[-(2**63) - 1]),
    ("a[2**100]", lambda: a[2**100]),
    ("a.numpy(2**64)", lambda: a.numpy(2**64)),
    ("jix.slice(a, 2**64)", lambda: jix.slice(a, 2**64)),
    ("jix.slice(a2, (0, 2**64))", lambda: jix.slice(a2, (0, 2**64))),
    ("a[np.uint64(2**63)]", lambda: a[np.uint64(2**63)]),
]

fails = 0
for name, fn in controls + cases:
    try:
        fn()
        got = "no error"
    except Exception as e:  # noqa: BLE001
        got = f"{type(e).__name__}: {e}"
    ok = got.startswith("IndexError")
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}")

# Reference behavior of Python lists and numpy for the same huge ints.
x = np.arange(5)
for name, fn in [
    ("list [1][2**64]", lambda: [1][2**64]),
    ("np x[2**100]", lambda: x[2**100]),
    ("np x[2**63]", lambda: x[2**63]),
    ("np x[-(2**63) - 1]", lambda: x[-(2**63) - 1]),
]:
    try:
        fn()
        got = "no error"
    except Exception as e:  # noqa: BLE001
        got = f"{type(e).__name__}: {e}"
    print(f"ref  {name}: {got}")

print(f"{fails} failure(s)")
sys.exit(1 if fails else 0)
