# F022: parse_basic_index (jix-py/src/ops/shape_ops.rs:309/317/333/370) and flatten (:821) cast
# u64 dim lengths / sizes `as i64`. A dim longer than i64::MAX (valid for 1-byte dtypes, reachable
# with repeat/tile/insert_dim) then cannot be indexed at all: every int / slice index raises
# IndexError with a negative "size", x[-2**63] overflows (PanicException in debug builds), an
# ellipsis / trailing fill hands core Slice a negative end, and flatten raises ValueError.
# Asserts the CORRECT behavior, so it fails while the bug exists.
import numpy as np

import jix

failures = []


def check(name, fn, expected):
    try:
        got = fn()
    except BaseException as e:  # PanicException derives from BaseException
        failures.append(f"{name}: raised {type(e).__module__}.{type(e).__name__}: {e}")
        print(f"FAIL  {name}: raised {type(e).__name__}: {e}")
        return
    if got != expected:
        failures.append(f"{name}: got {got!r}, expected {expected!r}")
        print(f"FAIL  {name}: got {got!r}, expected {expected!r}")
    else:
        print(f"ok    {name}: {got!r}")


def vals(a):
    # Array.__getitem__ already returns a numpy array; jix.slice returns a lazy jix.Array
    a = a.numpy() if isinstance(a, jix.Array) else a
    return np.asarray(a).ravel().tolist()


# x: len 2**64 - 1 (u64::MAX), x[j] = j // K (K = (2**64 - 1) // 3)
K = (2**64 - 1) // 3
x = jix.repeat(np.arange(3, dtype=np.uint8), K)
assert x.shape == (2**64 - 1,), x.shape
# Reference path that avoids the index parser's huge dim: reshape to (3, K) (all dims < 2**63).
xr = jix.reshape(x, (3, K))
check("reference xr[0, 0:2] (dims < 2**63 work)", lambda: vals(xr[0, 0:2]), [0, 0])

check("x[0]", lambda: vals(x[0]), [0])
check("x[-1]", lambda: vals(x[-1]), [2])
check("x[0:2]", lambda: vals(x[0:2]), [0, 0])
check("x[:].shape", lambda: x[:].shape, (2**64 - 1,))
# -2**63 + (2**64 - 1) = 2**63 - 1, which lies in [K, 2K) -> value 1
check("x[-2**63]", lambda: vals(x[-(2**63)]), [1])
check("x.numpy(index=slice(0, 3))", lambda: x.numpy(index=slice(0, 3)).tolist(), [0, 0, 0])

# Ellipsis / trailing fill on a huge dim: end = shape as i64 = -1 reaches core Slice
x2 = jix.insert_dim(x, 0)  # shape (1, 2**64 - 1)
check("jix.slice(x2, 0).shape", lambda: jix.slice(x2, 0).shape, (2**64 - 1,))
check("jix.slice(x2, (0, ...)).shape", lambda: jix.slice(x2, (0, ...)).shape, (2**64 - 1,))
check("jix.slice(x, slice(0, 2))", lambda: vals(jix.slice(x, slice(0, 2))), [0, 0])

# y: len 2**63 + 2, y[j] = 0 for j < 2**62 + 1, else 1
y = jix.repeat(np.array([0, 1], np.uint8), 2**62 + 1)
assert y.shape == (2**63 + 2,), y.shape
check("y[0]", lambda: vals(y[0]), [0])
check("y[-1]", lambda: vals(y[-1]), [1])
# flatten of a 1-D array is the array itself; reshape(y, -1) works, flatten(y) should too
check("reshape(y, -1).shape", lambda: jix.reshape(y, -1).shape, (2**63 + 2,))
check("flatten(y).shape", lambda: jix.flatten(y).shape, (2**63 + 2,))
check("flatten(insert_dim(y, 0)).shape", lambda: jix.flatten(jix.insert_dim(y, 0)).shape, (2**63 + 2,))

print()
if failures:
    print(f"{len(failures)} check(s) failed")
    raise SystemExit(1)
print("all checks passed")
