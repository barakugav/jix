"""F016 repro: slice_unpack (jix-py/src/util.rs:19-34) resolves slice bounds through Py_ssize_t.

Asserts the CORRECT behavior, so it fails while the bug exists.

Part A (any platform, runs here): explicit slice bounds beyond PY_SSIZE_T_MAX are silently clamped
by PySlice_Unpack, and a clamped stop is then mistaken for an omitted stop and replaced by the dim
length. So `a[0:2**63]` is accepted as `a[:]` while `a[0:len+1]` raises (documented strict bounds).

Part B (32-bit CPython only - i686 / armv7 / win32 wheels; skipped on 64-bit): Py_ssize_t is i32,
so `length as Py_ssize_t` truncates a dim length > 2**31-1, and parse_basic_index then treats the
negative truncated stop as end-relative: `a[:]` on a 2**31+10 dim has length 20.
"""

import sys

import numpy as np

import jix

failures = []


def expect_index_error(label, fn):
    try:
        r = fn()
    except IndexError as e:
        print(f"ok    {label}: IndexError: {e}")
        return
    shape = getattr(r, "shape", None)
    print(f"FAIL  {label}: expected IndexError, got result with shape {shape}")
    failures.append(label)


def expect_shape(label, fn, shape):
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {label}: expected shape {shape}, got {type(e).__name__}: {e}")
        failures.append(label)
        return
    if tuple(r.shape) != tuple(shape):
        print(f"FAIL  {label}: expected shape {shape}, got {tuple(r.shape)}")
        failures.append(label)
    else:
        print(f"ok    {label}: shape {tuple(r.shape)}")


# ---- Part A: 64-bit (and 32-bit) - clamped explicit bounds bypass the strict bounds check ----
a = jix.compact(np.arange(12, dtype=np.int64).reshape(3, 4))
expect_index_error("a[0:4] (stop = len + 1)", lambda: a[0:4])
expect_index_error("a[0:2**63]", lambda: a[0:2**63])
expect_index_error("a[0:2**100]", lambda: a[0:2**100])
expect_index_error("a[0:sys.maxsize]", lambda: a[0:sys.maxsize])
expect_index_error("jix.slice(a, slice(1, 2**64))", lambda: jix.slice(a, slice(1, 2**64)))
expect_index_error("a.numpy(index=slice(0, 2**63))", lambda: a.numpy(index=slice(0, 2**63)))
# The clamped start leaks into the error message (cosmetic):
try:
    a[2**100:]
except IndexError as e:
    print(f"info  a[2**100:]: IndexError: {e}")

# ---- Part B: 32-bit CPython only - dim length truncated to i32 ----
if sys.maxsize < 2**63 - 1:
    L1 = 2**31 + 10
    b = jix.broadcast(np.zeros((1, 2), np.uint8), (L1, 2))
    expect_shape("b[:, 0] with dim 2**31+10", lambda: b[:, 0], (L1,))
    expect_shape("b[:] with dim 2**31+10", lambda: b[:], (L1, 2))
    L2 = 3_000_000_000
    c = jix.broadcast(np.zeros((1,), np.uint8), (L2,))
    expect_shape("jix.slice(c, slice(100, 2_500_000_000))", lambda: jix.slice(c, slice(100, 2_500_000_000)), (2_499_999_900,))
    expect_shape("jix.slice(c, slice(-10, None))", lambda: jix.slice(c, slice(-10, None)), (10,))
    expect_shape("c[2_500_000_000:2_500_000_010]", lambda: c[2_500_000_000:2_500_000_010], (10,))
else:
    print("skip  Part B (32-bit CPython only): sys.maxsize =", sys.maxsize)

if failures:
    print(f"\n{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("\nall checks passed")
