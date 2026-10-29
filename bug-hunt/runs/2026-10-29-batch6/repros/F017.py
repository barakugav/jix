"""F017 repro: an untyped Python float/complex next to a bool/u8/i8/u16/i16 operand is converted to
f32 / complex64 at dispatch (Operand::cast -> Scalar::cast, jix-py/src/ops/common/operand.rs:164-166,
385) with no range check, so a finite value beyond f32 range becomes +-inf and a nonzero value below
the f32 subnormal range becomes 0, silently. An out-of-range Python int raises OverflowError instead.

Asserts the CORRECT behavior, so it exits 1 while the issue exists. Correct = for every case,
either the call raises OverflowError (like `int8 + 300`), or the result matches the exact value
(numpy, which computes these in float64) - in particular it is never a silently lost operand, and
`u8 / 1e-46` never raises 'divide by zero' for a nonzero divisor.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F017.py
"""

import numpy as np

import jix

i8n = np.array([1, 2], np.int8)
u8n = np.array([1, 2], np.uint8)
u8bign = np.array([255, 200], np.uint8)
u8_100n = np.array([100], np.uint8)
i16n = np.array([1, 2], np.int16)
condn = np.array([True, False])
i8, u8, u8big, u8_100, i16, cond = map(jix.compact, (i8n, u8n, u8bign, u8_100n, i16n, condn))

# (label, jix thunk, exact reference computed by numpy in float64/complex128)
CASES = [
    ("int8 + 1e300", lambda: i8 + 1e300, i8n + 1e300),
    ("int8 - 1e300", lambda: i8 - 1e300, i8n - 1e300),
    ("int8 * 1e39", lambda: i8 * 1e39, i8n * 1e39),
    ("int16 + 1e39", lambda: i16 + 1e39, i16n + 1e39),
    ("uint8 * 1e-46", lambda: u8 * 1e-46, u8n * 1e-46),
    # 100 * 1e-46 = 1e-44 is representable in f32 (subnormal); jix multiplies by 0.0 instead
    ("uint8 [100] * 1e-46", lambda: u8_100 * 1e-46, u8_100n * 1e-46),
    # 255 / 1e39 = 2.55e-37 is a NORMAL f32; jix divides by inf instead and returns 0
    ("uint8 [255, 200] / 1e39", lambda: u8big / 1e39, u8bign / 1e39),
    # nonzero divisor: jix raises FloatingPointError 'divide by zero encountered'
    ("uint8 / 1e-46", lambda: u8 / 1e-46, u8n / 1e-46),
    ("jix.divide(True, 1e-46)", lambda: jix.divide(True, 1e-46), np.divide(True, 1e-46)),
    ("int8 + 1e300j", lambda: i8 + 1e300j, i8n + 1e300j),
    ("where(c, uint8, 1e300)", lambda: jix.where(cond, u8, 1e300), np.where(condn, u8n, 1e300)),
    (
        "stack([uint8 0-d, 1e300])",
        lambda: jix.stack([jix.compact(np.array(1, np.uint8)), 1e300]),
        np.stack([np.array(1, np.uint8), 1e300]),
    ),
]

wrong = 0
for label, thunk, ref in CASES:
    try:
        got = thunk().numpy()
    except OverflowError as e:
        print(f"ok   : {label}: raised OverflowError: {e}")
        continue
    except Exception as e:  # noqa: BLE001
        wrong += 1
        print(f"WRONG: {label}: raised {type(e).__name__}: {e} (exact: {ref!r})")
        continue
    if np.allclose(got.astype(np.complex128), np.asarray(ref, np.complex128), rtol=1e-6, atol=0):
        print(f"ok   : {label}: {got!r}")
    else:
        wrong += 1
        print(f"WRONG: {label}: jix={got!r} exact={ref!r}")

# Contrast (not counted): an out-of-range Python int is range-checked and raises.
try:
    i8 + 300
    print("contrast: int8 + 300 did not raise")
except OverflowError as e:
    print(f"contrast: int8 + 300 raises OverflowError: {e}")
# Control (not counted): a typed np.float64 scalar keeps its value (dispatch picks f64).
print(f"control: int8 + np.float64(1e300) -> {(i8 + np.float64(1e300)).numpy()!r}")

print(f"{wrong} wrong of {len(CASES)} cases")
assert wrong == 0, f"{wrong} cases lost the Python float's value"
