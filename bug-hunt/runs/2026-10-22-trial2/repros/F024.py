"""F024 repro: numpy ndarray (or numpy scalar) on the LEFT of a binary operator with a jix.Array.

jix.Array defines neither `__array_ufunc__` nor `__array_priority__`, so numpy's ndarray.__add__
(etc.) does not defer to jix's reflected dunders: it wraps the jix Array as an opaque 0-d object,
broadcasts it, and returns an object ndarray whose every element is a whole lazy jix Array.
Comparisons then call bool() on those element arrays and raise ValueError. For a numpy scalar on
the left the result is a jix Array, but numpy first demotes the scalar to an untyped Python
number, so the result dtype differs from `jix OP scalar` (np.int16(3) + int8 jix -> int8).

The reference is what Python would produce if numpy deferred: jix's own reflected operator
(`j.__radd__(a)`, `j.__gt__(a)` for `a < j`, ...). Asserts the CORRECT behavior, so it FAILS while
the bug exists. Run: .venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F024.py
"""

import operator

import jix
import numpy as np

a = np.array([1, 2, 3], dtype=np.int32)
j = jix.compact(np.array([10, 20, 30], dtype=np.int32))

r = a + j
print("type(a + j):", type(r).__name__, "dtype:", getattr(r, "dtype", None), "shape:", getattr(r, "shape", None))
if isinstance(r, np.ndarray) and r.dtype == object:
    print("  r[0] =", repr(r[0]), "-> values", r[0].numpy())
print("j.__radd__(a) (what Python calls if numpy defers):", j.__radd__(a).numpy())

# (symbol, forward operator, name of the reflected jix method Python falls back to)
ops = [
    ("+", operator.add, "__radd__"),
    ("-", operator.sub, "__rsub__"),
    ("*", operator.mul, "__rmul__"),
    ("/", operator.truediv, "__rtruediv__"),
    ("//", operator.floordiv, "__rfloordiv__"),
    ("%", operator.mod, "__rmod__"),
    ("&", operator.and_, "__rand__"),
    ("|", operator.or_, "__ror__"),
    ("^", operator.xor, "__rxor__"),
    ("<<", operator.lshift, "__rlshift__"),
    (">>", operator.rshift, "__rrshift__"),
    ("<", operator.lt, "__gt__"),
    ("<=", operator.le, "__ge__"),
    (">", operator.gt, "__lt__"),
    (">=", operator.ge, "__le__"),
    ("==", operator.eq, "__eq__"),
    ("!=", operator.ne, "__ne__"),
]


def check(lhs, label):
    failures = []
    for name, op, rname in ops:
        expected = getattr(j, rname)(lhs)
        exp_np = expected.numpy()
        try:
            got = op(lhs, j)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{label} {name} j raised {type(e).__name__}: {e}")
            continue
        ok = (
            isinstance(got, jix.Array)
            and got.dtype == expected.dtype
            and got.shape == expected.shape
            and np.array_equal(got.numpy(), exp_np)
        )
        if not ok:
            failures.append(
                f"{label} {name} j -> {type(got).__name__} dtype={getattr(got, 'dtype', None)} "
                f"shape={getattr(got, 'shape', None)}; expected jix Array {expected.dtype} {exp_np.tolist()}"
            )
    return failures


failures = check(a, "ndarray[int32]")

# numpy scalar on the left: same root cause, visible as a dtype difference
j8 = jix.compact(np.array([1, 2, 3], dtype=np.int8))
s = np.int16(3)
got, expected = s + j8, j8.__radd__(s)
print(f"np.int16(3) + int8 jix -> {got.dtype}; j8.__radd__(np.int16(3)) -> {expected.dtype}")
if got.dtype != expected.dtype:
    failures.append(f"np.int16(3) + int8 jix -> {got.dtype}, expected {expected.dtype}")

print("hasattr(jix.Array, '__array_ufunc__'):", hasattr(jix.Array, "__array_ufunc__"))
print(f"{len(failures)} failures:")
for f in failures:
    print("  ", f)
assert not failures, "numpy operand on the left of an operator does not defer to the jix reflected operator"
print("OK")
