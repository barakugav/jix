"""F012: asarray/compact/cast with dtype= silently wrap out-of-range Python ints.

Asserts the CORRECT behavior (OverflowError, like `u8_arr + 256` in jix and like
numpy 2's np.asarray / np.array with dtype), so it fails while the bug exists.

Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F012.py
"""

import numpy as np

import jix


def outcome(fn):
    try:
        r = fn()
    except OverflowError as e:
        return f"OverflowError: {e}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    if isinstance(r, jix.Array):
        return f"returned {r.numpy()!r}"
    return f"returned {r!r}"


# Controls: jix ops bounds-check Python ints, numpy bounds-checks them at creation.
u8 = jix.compact(np.array([1, 2], dtype=np.uint8))
controls = {
    "jix: u8_arr + 256": lambda: u8 + 256,
    "jix: where(c, u8_arr, -1)": lambda: jix.where(jix.asarray(np.array([True, False])), u8, -1),
    "numpy: np.asarray(300, dtype=np.uint8)": lambda: np.asarray(300, dtype=np.uint8),
    "numpy: np.array([0, 255, 256], dtype=np.uint8)": lambda: np.array([0, 255, 256], dtype=np.uint8),
    "numpy: np.asarray(-1, dtype=np.uint32)": lambda: np.asarray(-1, dtype=np.uint32),
}
for name, fn in controls.items():
    print(f"[control] {name:50s} -> {outcome(fn)}")

# Python-int literals have no source dtype to cast from: an out-of-range value must raise.
cases = {
    "jix.asarray(300, dtype=np.uint8)": lambda: jix.asarray(300, dtype=np.uint8),
    "jix.asarray(-1, dtype=np.uint8)": lambda: jix.asarray(-1, dtype=np.uint8),
    "jix.compact([0, 255, 256], dtype=np.uint8)": lambda: jix.compact([0, 255, 256], dtype=np.uint8),
    "jix.compact([-1, 5], dtype=np.uint32)": lambda: jix.compact([-1, 5], dtype=np.uint32),
    "jix.compact(2**40, dtype=np.int32)": lambda: jix.compact(2**40, dtype=np.int32),
    "jix.asarray([[128]], dtype=np.int8)": lambda: jix.asarray([[128]], dtype=np.int8),
    "jix.cast(300, np.uint8)": lambda: jix.cast(300, np.uint8),
}
failures = []
for name, fn in cases.items():
    got = outcome(fn)
    ok = got.startswith("OverflowError")
    print(f"[{'ok' if ok else 'BUG'}] {name:50s} -> {got}")
    if not ok:
        failures.append(name)

# In-range values must keep working (guard against an over-eager fix).
np.testing.assert_array_equal(jix.compact([0, 255], dtype=np.uint8).numpy(), np.array([0, 255], dtype=np.uint8))
assert jix.asarray(-128, dtype=np.int8).numpy() == -128

assert not failures, f"{len(failures)} out-of-range Python int inputs were silently wrapped: {failures}"
print("all out-of-range Python ints raised OverflowError")
