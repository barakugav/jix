"""F021 repro: reshape with -1 on a zero-size array.

Asserts the CORRECT behavior (numpy-compatible inference of the -1 dim when the
known dims multiply to a nonzero value), so this script fails while the bug exists.
"""

import sys

import numpy as np

import jix

failures = []


def check(label, fn, expected_shape):
    try:
        got = fn()
    except Exception as e:  # noqa: BLE001
        failures.append(f"{label}: raised {type(e).__name__}: {e} (expected shape {expected_shape})")
        print(f"{label}: FAIL raised {type(e).__name__}: {e}")
        return
    if tuple(got.shape) != expected_shape:
        failures.append(f"{label}: shape {tuple(got.shape)} != {expected_shape}")
        print(f"{label}: FAIL shape {tuple(got.shape)}")
        return
    np.testing.assert_array_equal(got.numpy(), np.zeros(expected_shape, np.int32))
    print(f"{label}: OK shape={tuple(got.shape)}")


src = np.zeros((0, 3), np.int32)
z = jix.compact(src)

# References: numpy and jix.flatten (which passes the explicit size 0 instead of -1).
print("numpy reshape(-1, 3):", src.reshape(-1, 3).shape)
print("numpy reshape(-1):", src.reshape(-1).shape)
print("numpy reshape(2, -1, 3):", src.reshape(2, -1, 3).shape)
print("jix.flatten(z):", jix.flatten(z).shape)
print("jix.reshape(z, 0):", jix.reshape(z, 0).shape)

check("jix.reshape(z, (-1, 3))", lambda: jix.reshape(z, (-1, 3)), (0, 3))
check("jix.reshape(z, -1)", lambda: jix.reshape(z, -1), (0,))
check("z.reshape((-1, 3))", lambda: z.reshape((-1, 3)), (0, 3))
check("jix.reshape(z, (2, -1, 3))", lambda: jix.reshape(z, (2, -1, 3)), (2, 0, 3))
check("jix.reshape(np.zeros((0,3)), (-1, 3))", lambda: jix.reshape(np.zeros((0, 3), np.int32), (-1, 3)), (0, 3))

# Genuinely ambiguous case: known_size == 0. numpy raises here too, so raising is correct.
try:
    src.reshape(0, -1)
    print("numpy reshape(0, -1): no error")
except ValueError as e:
    print("numpy reshape(0, -1): ValueError:", e)
try:
    jix.reshape(z, (0, -1))
    print("jix reshape(0, -1): no error")
except ValueError as e:
    print("jix reshape(0, -1): ValueError (ok, ambiguous):", e)

if failures:
    print("\nFAILED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nall OK")
