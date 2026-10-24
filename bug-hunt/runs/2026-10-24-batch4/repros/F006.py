# Repro for bug-hunt run 2026-10-24-batch4, finding F006 (incomplete K28 fix).
# f64 -> f16 in jix uses half::f16::from_f64, whose no_std software fallback (used on every
# shipped target except aarch64-apple-darwin) drops the low 32 mantissa bits before rounding.
# Asserts the CORRECT rounding (numpy astype(float16) and CPython struct 'e', both single-rounded),
# so it fails while the bug exists. Run: .venv/bin/python bug-hunt/runs/2026-10-24-batch4/repros/F006.py
import struct

import numpy as np

import jix


def ref(x):
    """Correctly rounded f64 -> f16 via CPython's PyFloat_Pack2 (struct 'e')."""
    return np.frombuffer(struct.pack("<e", x), dtype=np.float16)[0]


failures = []


def check(name, got, want):
    got = np.asarray(got, dtype=np.float16)
    want = np.asarray(want, dtype=np.float16)
    ok = np.array_equal(got.view(np.uint16), want.view(np.uint16))
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r} want {want!r}")
    if not ok:
        failures.append(name)


x = 1 + 2.0**-11 + 2.0**-40  # just above the midpoint of f16 neighbours 1.0 and 1 + 2^-10
s = 2.0**-25 + 2.0**-60  # just above half the smallest f16 subnormal
assert ref(x) == np.float16(x) == 1 + 2.0**-10
assert ref(s) == np.float16(s) == 2.0**-24

check("jix.cast(f64 array, float16)", jix.cast(np.array([x, -x, s]), np.float16).numpy(), [ref(x), ref(-x), ref(s)])
check("jix.asarray(f64 array, dtype=float16)", jix.asarray(np.array([x]), dtype=np.float16).numpy(), [ref(x)])
# Python float scalar mixed with an f16 array (jix-py/src/ops/common/operand.rs:286)
check("jix.add(zeros(f16), python float)", jix.add(np.zeros(2, np.float16), x).numpy(), [ref(x), ref(x)])
check("jix.add(zeros(f16), python float subnormal)", jix.add(np.zeros(1, np.float16), s).numpy(), [ref(s)])

rng = np.random.default_rng(2026)
v = (rng.random(1_000_000) - 0.5) * 400.0
got = jix.cast(v, np.float16).numpy()
want = v.astype(np.float16)
bad = np.nonzero(got.view(np.uint16) != want.view(np.uint16))[0]
old = (v.astype(np.float32).astype(np.float16).view(np.uint16) != want.view(np.uint16)).sum()
print(f"random sweep (uniform +-200): {len(bad)} of {len(v)} mismatch numpy; first: "
      f"{[(float(v[i]), float(got[i]), float(want[i])) for i in bad[:3]]}")
print(f"(info) pre-#87 formula (via f32) would mismatch {old} of {len(v)}")
if len(bad):
    failures.append("random sweep")

if failures:
    raise SystemExit(f"FAILED: {failures}")
print("all ok")
