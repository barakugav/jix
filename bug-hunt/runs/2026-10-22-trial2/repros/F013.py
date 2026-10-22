"""F013 repro: comparisons between uint64 and signed-int arrays run in float64.

jix dispatches equal/not_equal/less/less_equal/greater/greater_equal for a
(uint64, int64) pair to the f64 impl (no Safe-cast integer slot holds both), so
integers above 2**53 collapse before comparing and the boolean result is wrong.

The script asserts the CORRECT (exact integer) results, so it fails while the bug
exists. Reference = Python int comparison (exact), which numpy 2.x matches.
"""

import operator

import jix
import numpy as np

failures = []


def check(name, a_np, b_np, make_a, make_b):
    ops = [
        ("equal", jix.equal, operator.eq),
        ("not_equal", jix.not_equal, operator.ne),
        ("less", jix.less, operator.lt),
        ("less_equal", jix.less_equal, operator.le),
        ("greater", jix.greater, operator.gt),
        ("greater_equal", jix.greater_equal, operator.ge),
    ]
    for op_name, jop, pyop in ops:
        expected = np.array([pyop(int(x), int(y)) for x, y in zip(a_np, b_np)], dtype=bool)
        got = jop(make_a(a_np), make_b(b_np)).numpy()
        np_res = getattr(np, op_name)(a_np, b_np)
        status = "ok" if np.array_equal(got, expected) else "WRONG"
        print(f"{name:28s} {op_name:13s} expected={expected} numpy={np_res} jix={got} {status}")
        if status != "ok":
            failures.append((name, op_name))


print("numpy", np.__version__)

# Case 1 (hunter 1): adjacent values just above 2**53, uint64 vs int64, compact arrays.
a = np.array([2**53, 2**53 + 1, 5, 2**64 - 1], dtype=np.uint64)
b = np.array([2**53 + 1, 2**53, 5, -1], dtype=np.int64)
check("u64 vs i64 (compact)", a, b, jix.compact, jix.compact)

# Case 2 (hunter 2): int64 vs uint64 near 2**63, numpy arrays passed directly (Plain path).
a = np.array([2**53 + 1, 2**63 - 1], dtype=np.int64)
b = np.array([2**53, 2**63], dtype=np.uint64)
check("i64 vs u64 (numpy in)", a, b, lambda x: x, lambda x: x)

# Case 3: the dunder operators go through the same dispatch.
a_np = np.array([2**60 + 1], dtype=np.int64)
b_np = np.array([2**60], dtype=np.uint64)
got = (jix.compact(a_np) == jix.compact(b_np)).numpy()
print(f"{'__eq__ i64 vs u64':28s} expected=[False] numpy={a_np == b_np} jix={got}")
if got.tolist() != [False]:
    failures.append(("__eq__", "equal"))

# Case 4: a typed numpy uint64 scalar vs an int64 array, and a narrower signed type (int8).
a_np = np.array([2**62 + 1, 2**62], dtype=np.int64)
got = jix.equal(jix.compact(a_np), np.uint64(2**62)).numpy()
print(f"{'i64 arr vs np.uint64 scalar':28s} expected=[False  True] numpy={a_np == np.uint64(2**62)} jix={got}")
if got.tolist() != [False, True]:
    failures.append(("scalar", "equal"))

a_np = np.array([2**63 + 1], dtype=np.uint64)
b_np = np.array([1], dtype=np.int8)
got = jix.greater(jix.compact(a_np), jix.compact(b_np)).numpy()
print(f"{'u64 vs i8 (ok, sanity)':28s} expected=[True] jix={got}")

# For contrast: same-signedness pairs compare exactly.
a_np = np.array([2**53, 2**53 + 1], dtype=np.uint64)
b_np = np.array([2**53 + 1, 2**53], dtype=np.uint64)
print("u64 vs u64 equal (exact):", jix.equal(jix.compact(a_np), jix.compact(b_np)).numpy())
print("dtype jix.add(u64, i64):", jix.add(jix.compact(a_np), jix.compact(b_np.astype(np.int64))).dtype)

print()
if failures:
    print(f"FAIL: {len(failures)} wrong comparison results: {failures}")
    raise SystemExit(1)
print("PASS")
