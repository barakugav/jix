"""F008 repro: jix.log(x, base) / jix.log2 / jix.log10 are not correctly rounded.

`log(x, base)` computes `Ln(x)` rounded to T, then multiplies by `1/ln(base)` cast to T: three
roundings. Exact powers come out 1 ulp low. This script asserts the CORRECT behavior (numpy's
np.log2 / np.log10 results), so it fails while the bug exists.
"""

import math

import numpy as np

import jix

failures = []


def check(name, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    ok = np.array_equal(got, want)
    print(f"{name}: jix={got.tolist()} numpy={want.tolist()} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(name)


# Headline triggers.
check("f64 log10(1000)", jix.log10(np.array([1000.0])).numpy(), np.log10(np.array([1000.0])))
check("f64 log2(8)", jix.log2(np.array([8.0])).numpy(), np.log2(np.array([8.0])))
check(
    "f16 log2(32)",
    jix.log2(np.array([32.0], dtype=np.float16)).numpy(),
    np.log2(np.array([32.0], dtype=np.float16)),
)
check("f32 log2(8)", jix.log2(np.array([8.0], dtype=np.float32)).numpy(), np.log2(np.array([8.0], dtype=np.float32)))

# Integer-valued uses: digit count / bit length.
x = np.array([1000.0, 1e6, 1e15])
print("floor(jix.log10(x)) =", np.floor(jix.log10(x).numpy()).tolist(), " numpy:", np.floor(np.log10(x)).tolist())
print("int(jix.log2(8.0)) =", int(jix.log2(np.array([8.0])).numpy()[0]), " numpy:", int(np.log2(8.0)))

# Python's own math.log with a base has the same 3-rounding behavior; math.log10/log2 do not.
print("math.log(1000, 10) =", math.log(1000, 10), " math.log10(1000) =", math.log10(1000))

# Counts over exact powers.
p10 = np.array([10.0**k for k in range(1, 23)])
n10 = int(np.sum(jix.log10(p10).numpy() != np.arange(1, 23)))
print(f"f64 log10: {n10} of 22 powers 10**1..10**22 inexact (numpy: {int(np.sum(np.log10(p10) != np.arange(1, 23)))})")
p2 = np.array([2.0**k for k in range(-60, 61)])
n2 = int(np.sum(jix.log2(p2).numpy() != np.arange(-60, 61)))
print(f"f64 log2: {n2} of 121 powers 2**-60..2**60 inexact (numpy: {int(np.sum(np.log2(p2) != np.arange(-60, 61)))})")
p2f = np.array([2.0**k for k in range(-20, 21)], dtype=np.float32)
n2f = int(np.sum(jix.log2(p2f).numpy() != np.arange(-20, 21)))
print(f"f32 log2: {n2f} of 41 powers 2**-20..2**20 inexact")
p2h = np.array([2.0**k for k in range(-14, 16)], dtype=np.float16)
n2h = int(np.sum(jix.log2(p2h).numpy() != np.arange(-14, 16)))
print(f"f16 log2: {n2h} of 30 powers 2**-14..2**15 inexact")

# f16 max ulp error over every positive finite f16.
allx = np.arange(1, 0x7C00, dtype=np.uint16).view(np.float16)
for name, jf, nf in (("log2", jix.log2, np.log2), ("log10", jix.log10, np.log10)):
    j = jf(allx).numpy().astype(np.float64)
    ref = nf(allx.astype(np.float64))
    ulp = np.spacing(np.abs(nf(allx)).astype(np.float16)).astype(np.float64)
    err = np.abs(j - ref) / ulp
    i = int(np.argmax(err))
    nerr = float(np.max(np.abs(nf(allx).astype(np.float64) - ref) / ulp))
    print(f"f16 {name}: jix max err {err[i]:.2f} ulp at x={allx[i]} (numpy max {nerr:.2f} ulp)")

# The same three-rounding path also affects arbitrary values (not only exact powers): f64 ulp stats.
rng = np.random.default_rng(0)
xs = rng.uniform(0.5, 1e6, 200000)
for name, jf, nf in (("log2", jix.log2, np.log2), ("log10", jix.log10, np.log10)):
    diff = jf(xs).numpy() != nf(xs)
    print(f"f64 {name}: differs from numpy on {diff.mean() * 100:.1f}% of 200k random inputs")

assert not failures, f"mismatches vs numpy: {failures}"
