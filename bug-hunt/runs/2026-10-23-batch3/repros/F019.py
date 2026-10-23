"""F019: jix.power(x, 2) (and x ** 2, power(x, 2.0), power(x, np.int32(2))) is routed to Square
(jix-py/src/ops/op2.rs:688), which has no status, so the Overflow that the Pow kernel reports for a
finite base with an infinite result (documented in the jix.power docstring) is silently dropped.

The repro asserts the CORRECT (documented) behavior: under over='raise' every spelling of
"x to the power 2" raises FloatingPointError, as power(x, 3) and power(x, array_of_2) do.
It fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-23-batch3/repros/F019.py
"""

import warnings

import numpy as np

import jix


def kind(f):
    try:
        f()
    except FloatingPointError as e:
        return f"FloatingPointError({e})"
    return None


failures = []
for dtype, big in [(np.float64, 1e200), (np.float32, 1e30), (np.float16, 300.0)]:
    x = np.array([big, 3.0], dtype=dtype)
    y = jix.compact(x)
    exp_arr = jix.compact(np.array([2, 2], dtype=np.int32))
    spellings = {
        "power(y, 2)": lambda: jix.power(y, 2).numpy(),
        "y ** 2": lambda: (y**2).numpy(),
        "power(y, 2.0)": lambda: jix.power(y, 2.0).numpy(),
        "power(y, np.int32(2))": lambda: jix.power(y, np.int32(2)).numpy(),
        # Controls: the same mathematical operation through the Pow kernel.
        "power(y, int32 array [2, 2])": lambda: jix.power(y, exp_arr).numpy(),
        "power(y, 3)": lambda: jix.power(y, 3).numpy(),
    }
    with np.errstate(over="raise"):
        np_kind = kind(lambda: np.power(x, 2))
    print(f"{np.dtype(dtype).name}: numpy power(x, 2) -> {np_kind}")
    with jix.errstate(over="raise"):
        for label, f in spellings.items():
            k = kind(f)
            print(f"  {label:32s} -> {k}")
            if k is None:
                failures.append(f"{np.dtype(dtype).name} {label}: no overflow reported")

# Default modes (over='warn'): power(y, 2) should warn like power(y, 3).
y = jix.compact(np.array([1e200, 3.0]))
for label, f in [("power(y, 2)", lambda: jix.power(y, 2).numpy()), ("power(y, 3)", lambda: jix.power(y, 3).numpy())]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = f()
    msgs = [str(m.message) for m in w]
    print(f"default modes {label}: result={r!r} warnings={msgs}")
    if not any("overflow" in m for m in msgs):
        failures.append(f"default modes {label}: no overflow warning")

print()
if failures:
    print("FAIL:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("PASS")
