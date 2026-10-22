# F009: jix.roll reduces shift with rem_euclid(len as i64) (jix/src/ops/shape_ops/roll.rs:54).
# Dims longer than i64::MAX (valid for uint8) get a wrong modulus; len == 2**64-1 with
# shift == -2**63 panics. Asserts the CORRECT behavior, so it fails while the bug exists.
import numpy as np

import jix

failures = []


def check(name, fn, expected):
    try:
        got = fn()
    except BaseException as e:  # PanicException derives from BaseException
        failures.append(f"{name}: raised {type(e).__name__}: {e}")
        return
    if got != expected:
        failures.append(f"{name}: got {got}, expected {expected}")
    else:
        print(f"{name}: ok")


base = jix.compact(np.array([0, 1, 2], dtype=np.uint8))
K = (2**64 - 1) // 3
t_max = jix.tile(base, K)  # len 2**64 - 1, in[j] = j % 3
assert t_max.shape == (2**64 - 1,), t_max.shape
# roll(1): out[0] = in[L-1] = (2**64 - 2) % 3 = 2. Read element 0 via reshape (3, K)[0, 0:1].
check(
    "roll(1) len 2**64-1",
    lambda: int(np.asarray(jix.reshape(jix.roll(t_max, 1), (3, K))[0, 0:1])[0]),
    2,
)
# Should succeed: -2**63 mod (2**64-1) is well defined. Read element 0 the same way.
# out[0] = in[(0 - (-2**63)) mod L] = in[2**63] = 2**63 % 3 = 2.
check(
    "roll(-2**63) len 2**64-1",
    lambda: int(np.asarray(jix.reshape(jix.roll(t_max, -(2**63)), (3, K))[0, 0:1])[0]),
    (2**63) % 3,
)

for f in failures:
    print("FAIL", f)
raise SystemExit(1 if failures else 0)
