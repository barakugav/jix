"""F015: a negative stride on a never-stepped (length 0/1) dim is rejected on input.

NumPy keeps a negative stride on a length-1 axis after x[::-1] / np.flip, still flags the
array C_CONTIGUOUS, and np.ascontiguousarray returns it unchanged. jix rejects it as an input
(asarray / compact / binary-op operand) with "Negative strides are not supported", while the
numpy(out=) path accepts the same layout. This script asserts the CORRECT behavior (inputs
accepted, values equal), so it fails while the bug exists.
"""

import numpy as np

import jix

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        failures.append(name)


x = np.arange(3, dtype=np.float32).reshape(1, 3)[::-1]
print("x strides", x.strides, "C_CONTIGUOUS", x.flags["C_CONTIGUOUS"])
xc = np.ascontiguousarray(x)
print("ascontiguousarray(x) is x:", xc is x, "strides", xc.strides)
one = np.array([7.0])[::-1]
print("one strides", one.strides, "C_CONTIGUOUS", one.flags["C_CONTIGUOUS"])
flipped = np.flip(np.arange(4, dtype=np.int64).reshape(1, 4), axis=0)
print("flipped strides", flipped.strides)


def eq(a, b):
    assert np.array_equal(np.asarray(a), np.asarray(b)), (a, b)


check("asarray(x)", lambda: eq(jix.asarray(x).numpy(), x))
check("compact(x)", lambda: eq(jix.compact(x).numpy(), x))
check("asarray(ascontiguousarray(x))", lambda: eq(jix.asarray(xc).numpy(), x))
check("asarray(np.array([7.0])[::-1])", lambda: eq(jix.asarray(one).numpy(), one))
check("asarray(np.flip(len-1 axis))", lambda: eq(jix.asarray(flipped).numpy(), flipped))
check(
    "jix array + x",
    lambda: eq((jix.asarray(np.ones((1, 3), np.float32)) + x).numpy(), x + 1),
)

# The out= path accepts exactly the same layout.
out = np.zeros((1, 3), np.float32)[::-1]
print("out strides", out.strides)
src = jix.asarray(np.arange(3, dtype=np.float32).reshape(1, 3))
src.numpy(out=out)
print("numpy(out=<strides (-12, 4)>) accepted:", out.tolist())

if failures:
    raise SystemExit(f"FAILED: {failures}")
print("all OK")
