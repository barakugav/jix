"""K195 repro: strict slice bounds stop being strict on a dim of length PY_SSIZE_T_MAX.

slice_unpack (jix-py/src/util.rs:24-46) resolves the slice with PySlice_Unpack, which clamps an
explicit bound outside [PY_SSIZE_T_MIN, PY_SSIZE_T_MAX] to those limits. parse_basic_index
(jix-py/src/ops/shape_ops.rs:336-358) then checks `start_norm > len` / `stop_norm > len`, which a
clamped bound passes when len == PY_SSIZE_T_MAX.

Asserts the CORRECT behavior (IndexError, as for every other dim length and as the intended
"huge bounds raise" contract in test_slice.py::test_slice_huge_bounds_raise says), so it FAILS
while the bug exists. Run: .venv/bin/python bug-hunt/report/repros/K195.py
"""

import sys

import numpy as np

import jix

n = sys.maxsize  # PY_SSIZE_T_MAX on this platform (2**63-1 on 64-bit, 2**31-1 on 32-bit)

# Lazy broadcast: costs no memory until read.
big = jix.broadcast(jix.asarray(np.zeros(1, np.uint8)), (n,))
small = jix.broadcast(jix.asarray(np.zeros(1, np.uint8)), (10,))
assert big.shape == (n,), big.shape

failures = []


def check(label, arr, sl):
    # Only the lazy jix.slice path here: it never reads data, so a wrongly accepted
    # [5:2**100] does not try to allocate 8 EiB. __getitem__ shares parse_basic_index.
    try:
        r = jix.slice(arr, (sl,))
    except IndexError as e:
        print(f"OK   len={arr.shape[0]:>20} {label:<13} {sl!s:<45} -> IndexError: {e}")
        return
    print(f"BUG  len={arr.shape[0]:>20} {label:<13} {sl!s:<45} -> accepted, shape {r.shape}")
    failures.append((arr.shape[0], label, sl, r.shape))


for arr in (small, big):
    length = arr.shape[0]
    check("start 2**100", arr, slice(2**100, None))
    check("start len+1", arr, slice(length + 1, None))
    check("stop 2**100", arr, slice(5, 2**100))
    check("stop len+1", arr, slice(0, length + 1))

# __getitem__ (materializing) path, only the empty-result case so nothing large is read.
try:
    r = big[2**100 :]
    print(f"BUG  __getitem__ big[2**100:] -> accepted, returned {type(r).__name__} shape {r.shape}")
    failures.append((n, "__getitem__ start 2**100", slice(2**100, None), r.shape))
except IndexError as e:
    print(f"OK   __getitem__ big[2**100:] -> IndexError: {e}")

# Error message quotes the clamped bound, not the user's value (cosmetic part).
try:
    small[2**100 :]
except IndexError as e:
    print(f"msg  small[2**100:] -> {e}")

assert not failures, f"{len(failures)} out-of-bounds slices accepted on a dim of length {n}: {failures}"
print("all out-of-bounds slices raised IndexError")
