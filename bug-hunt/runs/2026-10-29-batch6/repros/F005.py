"""F005 repro (asserts the CORRECT behavior; exits 1 while the bug exists).

Comparing a bool/uint8/int8/uint16/int16 operand with an untyped Python float (or complex)
dispatches to the f32 (Complex<f32>) comparison kernel: the Python float is rounded to f32 before
the comparison, so the returned booleans are wrong whenever that rounding lands on or crosses an
array value. The reference is Python's own exact int-vs-float comparison (numpy 2 agrees: it
compares these pairs in float64, where every 16-bit int is exact).
"""
import operator

import numpy as np

import jix

OPS = {
    "equal": operator.eq,
    "not_equal": operator.ne,
    "less": operator.lt,
    "less_equal": operator.le,
    "greater": operator.gt,
    "greater_equal": operator.ge,
}

failures = []
checked = 0


def check(label, got, expected):
    global checked
    checked += 1
    got = np.asarray(got).tolist()
    if got != expected:
        failures.append(f"{label}: jix={got} exact={expected}")


vals = [0, 1, 3]
scalars = [1.00000001, 0.99999999, 3 * 0.1 * 10, 1e-50, -1e-50]
for dt in [np.uint8, np.int8, np.uint16, np.int16]:
    a_np = np.array(vals, dtype=dt)
    a = jix.compact(a_np)
    for s in scalars:
        for name, pyop in OPS.items():
            expected = [pyop(v, s) for v in vals]
            check(f"jix.{name}({np.dtype(dt).name} {vals}, {s!r})", getattr(jix, name)(a, s).numpy(), expected)
    # the dunder operators take the same path
    check(f"({np.dtype(dt).name} {vals}) == 1.00000001", (a == 1.00000001).numpy(), [v == 1.00000001 for v in vals])
    check(f"({np.dtype(dt).name} {vals}) < 3*0.1*10", (a < 3 * 0.1 * 10).numpy(), [v < 3 * 0.1 * 10 for v in vals])
    # a plain numpy input (Plain storage) too
    check(f"jix.equal(np {np.dtype(dt).name} {vals}, 1.00000001)", jix.equal(a_np, 1.00000001).numpy(),
          [v == 1.00000001 for v in vals])
    # complex scalar -> Complex<f32>
    check(f"jix.equal({np.dtype(dt).name} {vals}, complex(1.00000001, 0))",
          jix.equal(a, complex(1.00000001, 0)).numpy(), [v == complex(1.00000001, 0) for v in vals])

# bool array / Python bool (typed P1) / typed numpy int8 scalar
b = jix.compact(np.array([False, True]))
check("jix.equal(bool [F, T], 1.00000001)", jix.equal(b, 1.00000001).numpy(), [False, False])
check("jix.less(bool [F, T], 1e-50)", jix.less(b, 1e-50).numpy(), [True, False])
check("jix.equal(True, 1.00000001)", jix.equal(True, 1.00000001).numpy(), False)
check("jix.equal(np.int8(1), 1.00000001)", jix.equal(np.int8(1), 1.00000001).numpy(), False)

# isclose with zero tolerances is an exact equality test
check("jix.isclose(uint8 [1], 1.00000001, rtol=0, atol=0)",
      jix.isclose(jix.compact(np.array([1], np.uint8)), 1.00000001, rtol=0, atol=0).numpy(), [False])

# Controls (exact today): int32 arrays and typed float64 scalars take the f64 slot.
assert jix.equal(jix.compact(np.array([1], np.int32)), 1.00000001).numpy().tolist() == [False]
assert jix.equal(jix.compact(np.array([1], np.int16)), np.float64(1.00000001)).numpy().tolist() == [False]
# numpy agrees with the exact reference for the narrow ints
assert np.equal(np.array([1], np.int16), 1.00000001).tolist() == [False]
assert np.less(np.array([3], np.int8), 3 * 0.1 * 10).tolist() == [True]

for f in failures:
    print("WRONG:", f)
print(f"{len(failures)} wrong of {checked} checks")
assert not failures, f"{len(failures)} wrong comparison results"
