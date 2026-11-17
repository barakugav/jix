"""K176: layout-identical numpy struct dtypes (default vs align=True) are unequal jix dtypes.

P = [('a','<i4'),('b','<f4')] (numpy default, alignment 1, mapped to a PACKED jix dtype) and
A = the same list with align=True (alignment 4, mapped to an ALIGNED jix dtype) have the same names,
offsets (0, 4) and itemsize (8). numpy calls them equal (P == A, can_cast(P, A, 'no')), and so does
jix's own Python surface (jix_array.dtype == other.dtype, numpy(out=)). But jix-py's cast and
promotion compare the core Dtype, which includes the alignment, so the identity cast and every
multi-operand op between the two raise TypeError.

Asserts the CORRECT behavior (the layout-identical cast is a no-op, promotion succeeds and values
match numpy), so it fails while the inconsistency exists.

Run: .venv/bin/python bug-hunt/report/repros/K176.py
"""

import numpy as np

import jix

P = np.dtype([("a", "<i4"), ("b", "<f4")])
A = np.dtype([("a", "<i4"), ("b", "<f4")], align=True)
assert P == A and np.can_cast(P, A, casting="no") and np.can_cast(A, P, casting="no")
assert (P.itemsize, A.itemsize) == (8, 8) and (P.alignment, A.alignment) == (1, 4)

x = np.zeros(2, dtype=P)
x["a"], x["b"] = [1, 2], [0.5, 1.5]
y = np.zeros(2, dtype=A)
y["a"], y["b"] = [3, 4], [2.5, 3.5]
jx, jy = jix.asarray(x), jix.asarray(y)

# Two jix paths already treat the layouts as the same dtype:
assert jx.dtype == jy.dtype  # numpy descr equality on the jix arrays' own dtypes
np.testing.assert_array_equal(jx.numpy(out=np.zeros(2, dtype=A)), x)  # out= of the other layout
np.testing.assert_array_equal(jy.numpy(out=np.zeros(2, dtype=P)), y)


def same(got, want):
    assert got.tobytes() == want.tobytes(), (got, want)


cond = np.array([True, False])
checks = [
    ("asarray(x, dtype=A)", lambda: same(jix.asarray(x, dtype=A).numpy(), x)),
    ("cast(y, P)", lambda: same(jix.cast(y, P).numpy(), y)),
    ("concatenate([x, jy])", lambda: same(jix.concatenate([x, jy]).numpy(), np.concatenate([x, y]))),
    ("stack([jx, jy])", lambda: same(jix.stack([jx, jy]).numpy(), np.stack([x, y]))),
    ("where(cond, jx, y)", lambda: same(jix.where(cond, jx, y).numpy(), np.where(cond, x, y))),
]
failures = []
for label, check in checks:
    try:
        check()
        print(f"ok    {label}")
    except Exception as e:  # noqa: BLE001
        msg = f"{label}: {type(e).__name__}: {str(e)[:160]}"
        print(f"FAIL  {msg}")
        failures.append(msg)
if failures:
    raise AssertionError(f"{len(failures)} failure(s):\n" + "\n".join(failures))
print("OK")
