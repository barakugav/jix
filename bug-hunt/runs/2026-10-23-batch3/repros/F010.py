"""F010 verifier repro: float ** integer exponent goes through powi.

Asserts the CORRECT behavior (agreement with numpy / with the float-exponent path within 2 ulps),
so it fails while the bug exists. Run with .venv/bin/python.
"""

import numpy as np

import jix


def ulps(a, b):
    a, b = np.asarray(a), np.asarray(b)
    it = np.int32 if a.dtype == np.float32 else np.int64
    return np.abs(a.view(it).astype(np.int64) - b.view(it).astype(np.int64))


failures = []


def check(label, got, want):
    got, want = np.asarray(got), np.asarray(want)
    u = ulps(got, want)
    zero_mismatch = (got == 0) != (want == 0)
    ok = bool(np.all(u <= 2) and not np.any(zero_mismatch))
    shown = f"got {got!r} want {want!r} " if got.size <= 4 else ""
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {shown}max ulps {int(u.max())}")
    if not ok:
        failures.append(label)


with jix.errstate(all="raise"):  # no status is raised for any of these
    # 1. Subnormal results of a negative integer exponent are flushed to 0.
    x64 = jix.compact(np.array([2.0, 10.0]))
    check("f64 ** python int -1074", (x64 ** -1074).numpy()[:1], np.array([2.0]) ** -1074)
    check(
        "f64 ** int32 [-1074, -309]",
        jix.power(x64, jix.compact(np.array([-1074, -309], np.int32))).numpy(),
        np.power(np.array([2.0, 10.0]), np.array([-1074, -309], np.int32)),
    )
    # Same value, float exponent: correct. The int and float exponent paths disagree.
    check("2.0 ** -1074 (int) vs ** -1074.0 (float)", (x64 ** -1074).numpy()[:1], (x64 ** -1074.0).numpy()[:1])
    x32 = jix.compact(np.array([2.0], np.float32))
    check("f32 ** int16(-130)", jix.power(x32, np.int16(-130)).numpy(), np.array([2.0], np.float32) ** -130)

    # 2. Error grows with |n|.
    rng = np.random.default_rng(0)
    for dt in (np.float32, np.float64):
        for n in (100, 1000, 10000):
            # Bases near 1 so base**n stays in [e^-8, e^8] (no overflow / underflow).
            base = (1.0 + rng.uniform(-8.0, 8.0, 20000) / n).astype(dt)
            jb = jix.compact(base)
            got = (jb**n).numpy()
            want = (base.astype(np.float64) ** float(n)).astype(dt)  # numpy pow, rounded once
            check(f"{np.dtype(dt).name} ** {n} (python int), 20000 random bases", got, want)

print("FAILURES:", failures)
assert not failures, failures
