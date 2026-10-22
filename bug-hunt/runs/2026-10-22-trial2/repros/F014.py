"""F014 repro: complex division uses num-complex's textbook formula.

jix computes (a+bi)/(c+di) as ((ac+bd) + (bc-ad)i) / (c*c + d*d) (num-complex 0.4.6 Div,
called from jix/src/ops/op2.rs:781). The denominator c*c+d*d overflows for |divisor| above
~sqrt(MAX) and underflows below ~sqrt(MIN), so finite operands whose quotient is perfectly
representable come back as nan/inf/0 or with large relative error. numpy (Smith's
algorithm) gets them right.

The script asserts the CORRECT result (the exact quotient, computed with Fractions and
rounded), so it exits 1 while the bug exists.
"""

import sys
from fractions import Fraction

import jix
import numpy as np

failures = []


def exact_div(a: complex, b: complex, dtype) -> complex:
    ar, ai = Fraction(a.real), Fraction(a.imag)
    br, bi = Fraction(b.real), Fraction(b.imag)
    den = br * br + bi * bi
    re = (ar * br + ai * bi) / den
    im = (ai * br - ar * bi) / den
    ft = np.float32 if dtype == np.complex64 else np.float64
    # Fraction -> float is correctly rounded to f64; for complex64 round once more to f32
    # (double rounding is at most 1 ulp of f32, well within the tolerance below).
    return complex(ft(float(re)), ft(float(im)))


def check(dtype, a, b, how):
    a_np = np.array([a], dtype)
    b_np = np.array([b], dtype)
    # Cast the operands as stored (complex64 rounds the inputs).
    a_s, b_s = complex(a_np[0]), complex(b_np[0])
    expected = exact_div(a_s, b_s, dtype)
    if how == "compact":
        got = jix.divide(jix.compact(a_np), jix.compact(b_np)).numpy()[0]
    elif how == "operator":
        got = (jix.compact(a_np) / jix.compact(b_np)).numpy()[0]
    else:
        got = jix.divide(a_np, b_np).numpy()[0]
    with np.errstate(all="ignore"):
        np_res = (a_np / b_np)[0]
    rtol = 1e-6 if dtype == np.complex64 else 1e-15
    ok = np.isfinite(got) and abs(complex(got) - expected) <= rtol * abs(expected)
    print(
        f"{np.dtype(dtype).name:10s} {how:8s} ({a_s}) / ({b_s}): exact={expected} "
        f"numpy={np_res} jix={got} {'ok' if ok else 'WRONG'}"
    )
    if not ok:
        failures.append((dtype, a, b, how))


print("numpy", np.__version__)

c64 = np.complex64
c128 = np.complex128

# |b|^2 overflows: the quotient is 1
check(c64, 1e20 + 1e20j, 1e20 + 1e20j, "compact")
check(c64, 1e20 + 1e20j, 1e20 + 1e20j, "operator")
check(c64, 1e20 + 1e20j, 1e20 + 1e20j, "numpy-in")
check(c128, 1e160 + 1e160j, 1e160 + 1e160j, "compact")
# |b|^2 overflows: small quotient flushed to 0
check(c64, 1 + 1j, 1e20 + 1e20j, "compact")
check(c128, 1 + 1j, 1e160 + 1e160j, "compact")
# |b|^2 underflows to 0: inf/nan for a representable quotient
check(c64, 1 + 1j, 1e-23 + 1e-23j, "compact")
check(c128, 1 + 1j, 1e-170 + 1e-170j, "compact")
# |b|^2 subnormal: precision loss
check(c64, 3 + 4j, 1e-20 + 0j, "compact")
check(c128, 3 + 4j, 1e-160 + 0j, "compact")
# Moderate magnitudes stay right (control)
check(c64, 3 + 4j, 1 + 2j, "compact")
check(c128, 3 + 4j, 1 + 2j, "compact")

# Special values: numpy follows C99 Annex G-like results.
with np.errstate(all="ignore"):
    for a, b in [(1 + 2j, complex(np.inf, 0)), (1 + 2j, 0j)]:
        a_np, b_np = np.array([a], c128), np.array([b], c128)
        try:
            with jix.errstate(all="ignore"):
                got = jix.divide(jix.compact(a_np), jix.compact(b_np)).numpy()[0]
        except Exception as e:  # noqa: BLE001
            got = repr(e)
        np_res = (a_np / b_np)[0]
        print(f"special    ({a}) / ({b}): numpy={np_res} jix={got} (informational)")

# Error reporting: numpy warns/raises 'divide' for complex x / 0; jix's float divide raises
# FloatingPointError by default, complex divide reports nothing.
try:
    r = jix.divide(jix.compact(np.array([1 + 2j], c128)), jix.compact(np.array([0j], c128))).numpy()
    print(f"complex128 (1+2j)/0j with default errstate: no error, result {r}")
except FloatingPointError as e:
    print(f"complex128 (1+2j)/0j raised FloatingPointError: {e}")
try:
    r = jix.divide(jix.compact(np.array([1.0])), jix.compact(np.array([0.0]))).numpy()
    print(f"float64 1/0 with default errstate: no error, result {r}")
except FloatingPointError as e:
    print(f"float64 1/0 raised FloatingPointError: {e}")

if failures:
    print(f"FAIL: {len(failures)} wrong complex quotients")
    sys.exit(1)
print("PASS")
