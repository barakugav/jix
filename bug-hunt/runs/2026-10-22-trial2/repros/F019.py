"""F019 repro: permute_dims with more than 8 entries panics instead of raising.

Asserts the CORRECT behavior (a ValueError / RuntimeError, like permute_dims(a, [0] * 3)),
so it fails while the bug exists. A pyo3 PanicException derives from BaseException, not
Exception, so `except Exception` below does not swallow it.

Run: .venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F019.py
"""

import numpy as np

import jix

a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))

# Control: wrong length within ArrayVec capacity -> clean error from core.
try:
    jix.permute_dims(a, [0] * 3)
    raise AssertionError("expected an error for [0]*3")
except Exception as e:  # noqa: BLE001
    print(f"control permute_dims(a, [0]*3): {type(e).__name__}: {e}")

failures = []
cases = [
    ("jix.permute_dims(a, [0]*9)", lambda: jix.permute_dims(a, [0] * 9)),
    ("jix.permute_dims(a, [0, 1]*5)", lambda: jix.permute_dims(a, [0, 1] * 5)),
    ("a.permute_dims([0]*9)", lambda: a.permute_dims([0] * 9)),
    ("jix.permute_dims(np.zeros((2, 3)), [0]*9)", lambda: jix.permute_dims(np.zeros((2, 3)), [0] * 9)),
]
for name, fn in cases:
    try:
        fn()
        failures.append(f"{name}: no error raised")
    except (ValueError, RuntimeError) as e:
        print(f"OK {name}: {type(e).__name__}: {e}")
    except BaseException as e:  # noqa: BLE001
        print(f"BUG {name}: {type(e).__module__}.{type(e).__name__}: {e}")
        print(f"    isinstance(e, Exception) = {isinstance(e, Exception)}")
        failures.append(f"{name}: {type(e).__name__}: {e}")

assert not failures, "permute_dims with >8 dims did not raise a clean error:\n" + "\n".join(failures)
print("all cases raised clean errors")
