"""F011: bitwise_rotate_left/right with a Python int value.

Asserts the CORRECT behavior, so it fails while the bug exists:
  (1) all-Python-int calls work like bitwise_left_shift(1, 1) (int64 default precision);
  (2) a Python int value next to a typed u32 count array is rotated in a width that holds it
      (not int8 picked from the first dispatch-table entry), so no wrap to -128 and no
      OverflowError for 200.
Run: .venv/bin/python bug-hunt/runs/2026-10-24-batch4/repros/F011.py
"""

import numpy as np

import jix

MASK64 = (1 << 64) - 1


def rotl64(x: int, s: int) -> int:
    s %= 64
    u = x & MASK64
    r = ((u << s) | (u >> (64 - s))) & MASK64 if s else u
    return r - (1 << 64) if r >= 1 << 63 else r


def rotr64(x: int, s: int) -> int:
    return rotl64(x, (64 - s % 64) % 64)


failures = []


def check(name, fn, expected_values, expected_dtype=None):
    try:
        r = fn()
        got = r.numpy()
        print(f"{name} -> {r.dtype} {np.asarray(got).tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"{name} -> ERR {type(e).__name__}: {e}")
        failures.append(f"{name}: raised {type(e).__name__}: {e}")
        return
    if expected_dtype is not None and r.dtype != np.dtype(expected_dtype):
        failures.append(f"{name}: dtype {r.dtype}, expected {np.dtype(expected_dtype)}")
    if np.asarray(got).tolist() != expected_values:
        failures.append(f"{name}: values {np.asarray(got).tolist()}, expected {expected_values}")


# Baseline: shift with all-Python-int operands works and defaults to int64.
check("bitwise_left_shift(1, 1)", lambda: jix.bitwise_left_shift(1, 1), 2, np.int64)
# Baseline: shift with a Python int value and a u32 count takes the typed operand's dtype.
check(
    "bitwise_left_shift(64, u32[1])",
    lambda: jix.bitwise_left_shift(64, np.array([1], np.uint32)),
    [128],
)
# Baseline: a typed scalar value works.
check("bitwise_rotate_left(np.int64(64), 1)", lambda: jix.bitwise_rotate_left(np.int64(64), 1), 128, np.int64)

# (1) all-Python-int rotate: should behave like the shifts (int64), not raise TypeError.
check("bitwise_rotate_left(1, 1)", lambda: jix.bitwise_rotate_left(1, 1), rotl64(1, 1), np.int64)
check("bitwise_rotate_right(1, 1)", lambda: jix.bitwise_rotate_right(1, 1), rotr64(1, 1), np.int64)

# (2) Python int value, typed u32 count: the value should not be squeezed into int8.
check(
    "bitwise_rotate_left(64, u32[1])",
    lambda: jix.bitwise_rotate_left(64, np.array([1], np.uint32)),
    [128],
)
check(
    "bitwise_rotate_left(200, u32[1])",
    lambda: jix.bitwise_rotate_left(200, np.array([1], np.uint32)),
    [400],
)
check(
    "bitwise_rotate_right(1, u32[1])",
    lambda: jix.bitwise_rotate_right(1, np.array([1], np.uint32)),
    [rotr64(1, 1)],
)

if failures:
    print("\nFAILURES:")
    for f in failures:
        print("  " + f)
    raise AssertionError(f"F011: {len(failures)} check(s) failed")
print("\nall checks passed")
