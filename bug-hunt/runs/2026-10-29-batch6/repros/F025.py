"""F025 repro: jix.Array binary dunders never return NotImplemented.

Asserts the CORRECT (Python data-model / NEP 13) behavior, so it fails while the bug exists:
- an operand that opts out with `__array_ufunc__ = None` must get its reflected method called
  (as `ndarray OP x` does),
- `a == None` / `a != "x"` must not raise ValueError (identity fallback or elementwise result),
- an operand type jix cannot handle must raise TypeError ("unsupported operand type(s)"),
  not ValueError.
Run with: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F025.py
"""

import fractions
import operator
import sys

import numpy as np

import jix

a = jix.compact(np.array([1, 2, 3], np.int32))
n = np.array([1, 2, 3], np.int32)


class Foo:
    """A third-party type with reflected operators that opts out of numpy (NEP 13)."""

    __array_ufunc__ = None

    def __radd__(self, o):
        return "Foo.__radd__"

    def __rsub__(self, o):
        return "Foo.__rsub__"

    def __rmul__(self, o):
        return "Foo.__rmul__"

    def __rtruediv__(self, o):
        return "Foo.__rtruediv__"

    def __eq__(self, o):
        return "Foo.__eq__"

    def __lt__(self, o):
        return "Foo.__lt__"

    def __gt__(self, o):
        return "Foo.__gt__"

    __hash__ = object.__hash__


failures = []


def show(label, fn):
    try:
        r = fn()
        print(f"{label:40s} -> {r!r}")
        return ("ok", r)
    except Exception as e:  # noqa: BLE001
        print(f"{label:40s} !! {type(e).__name__}: {e}")
        return ("err", e)


print("--- reference: numpy ndarray on the left ---")
show("np + Foo()", lambda: n + Foo())
show("np - Foo()", lambda: n - Foo())
show("np < Foo()", lambda: n < Foo())  # reflected: Foo.__gt__
show("np == None", lambda: n == None)  # noqa: E711
show("np + 'x'", lambda: n + "x")
print()
print("--- jix Array on the left ---")

# 1. NEP 13 opt-out: the reflected method of the right operand must be tried.
for name, op, expected in [
    ("+", operator.add, "Foo.__radd__"),
    ("-", operator.sub, "Foo.__rsub__"),
    ("*", operator.mul, "Foo.__rmul__"),
    ("/", operator.truediv, "Foo.__rtruediv__"),
    ("==", operator.eq, "Foo.__eq__"),
    ("<", operator.lt, "Foo.__gt__"),
]:
    kind, r = show(f"a {name} Foo()", lambda op=op: op(a, Foo()))
    if kind != "ok" or r != expected:
        failures.append(f"a {name} Foo(): expected {expected!r}, got {kind} {r!r}")

# 2. Comparison with None / str must not raise ValueError.
for label, fn in [
    ("a == None", lambda: a == None),  # noqa: E711
    ("a != None", lambda: a != None),  # noqa: E711
    ("a == 'x'", lambda: a == "x"),
]:
    kind, r = show(label, fn)
    if kind == "err":
        failures.append(f"{label}: raised {type(r).__name__}: {r}")

# 3. An unsupported operand type must give TypeError, not ValueError.
for label, fn in [
    ("a * Fraction(1, 2)", lambda: a * fractions.Fraction(1, 2)),
    ("a + 'x'", lambda: a + "x"),
    ("a + object()", lambda: a + object()),
    ("a + None", lambda: a + None),
]:
    kind, r = show(label, fn)
    if kind == "err" and not isinstance(r, TypeError):
        failures.append(f"{label}: expected TypeError, got {type(r).__name__}: {r}")

print()
if failures:
    print(f"{len(failures)} failures:")
    for f in failures:
        print("  ", f)
    sys.exit(1)
print("all ok")
