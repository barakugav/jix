"""K181: power() with a FLOAT base and an untyped Python int exponent outside i32 raises
OverflowError ('out of bounds for i32'), though the result dtype is the base's float dtype either
way and the later (float, float) slot holds the exponent.

Asserts the CORRECT behavior (numpy's values, base dtype kept), so it fails while the bug exists.
Not asserted (intended, same as the pinned power(int_arr, -1)): an INT base with a Python int
exponent outside u32, e.g. int64_arr ** 2**32 -> OverflowError 'out of bounds for u32'.

Run: .venv/bin/python bug-hunt/report/repros/K181.py
"""

import warnings

import numpy as np

import jix

warnings.simplefilter("ignore", RuntimeWarning)  # numpy's overflow warnings for the references

f64 = np.array([1.0, 0.5, -1.0])
f32 = f64.astype(np.float32)
N = 2**31

failures = []


def check(label, fn, expected, dtype):
    try:
        r = fn()
        got = r.numpy()
        ok = r.dtype == np.dtype(dtype) and np.array_equal(got, expected)
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {r.dtype} {got}  (expected {np.dtype(dtype)} {expected})")
    except Exception as ex:  # noqa: BLE001
        ok = False
        print(f"FAIL {label}: {type(ex).__name__}: {ex}  (expected {np.dtype(dtype)} {expected})")
    if not ok:
        failures.append(label)


# Control: the same value as float, as a typed np.int64, or with an all-untyped call works.
check("power(f64_arr, float(2**31))", lambda: jix.power(jix.asarray(f64), float(N)), f64 ** float(N), np.float64)
check(
    "power(f64_arr, np.int64(2**31))", lambda: jix.power(jix.asarray(f64), np.int64(N)), f64 ** np.int64(N), np.float64
)
check("power(0.5, 2**31) (all untyped)", lambda: jix.power(0.5, N), np.array(0.0), np.float64)
# Control: every other op accepts a Python int of any i64 value next to a float array.
check("f64_arr + 2**31", lambda: jix.asarray(f64) + N, f64 + N, np.float64)
check("maximum(f64_arr, 2**31)", lambda: jix.maximum(jix.asarray(f64), N), np.maximum(f64, N), np.float64)

# The bug: the Python int exponent lands in the (float, i32) slot and its range check raises.
check("f64_arr ** 2**31", lambda: jix.asarray(f64) ** N, f64**N, np.float64)
check("power(f64_arr, -2**31 - 1)", lambda: jix.power(jix.asarray(f64), -N - 1), f64 ** (-N - 1), np.float64)
check("power(f32_arr, 2**31)", lambda: jix.power(jix.asarray(f32), N), f32**N, np.float32)
check("power(np.float64(0.5), 2**31)", lambda: jix.power(np.float64(0.5), N), np.array(0.0), np.float64)
check("f64_arr ** 10**10", lambda: jix.asarray(f64) ** 10**10, f64**10**10, np.float64)

# Informational only (intended): integer base, Python int exponent outside u32.
i64 = np.array([1, -1, 0], np.int64)
try:
    print("info i64_arr ** 2**32:", (jix.asarray(i64) ** 2**32).numpy())
except Exception as ex:  # noqa: BLE001
    print(f"info i64_arr ** 2**32: {type(ex).__name__}: {ex} (intended, like power(int_arr, -1))")

if failures:
    raise SystemExit(f"K181 reproduced: {len(failures)} check(s) failed: {failures}")
print("K181 not reproduced: all checks pass")
