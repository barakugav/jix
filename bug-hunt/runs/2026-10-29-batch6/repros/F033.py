"""F033: an empty numpy array whose length>1 dim has a negative stride is rejected on input
(asarray / compact / operands), although no element is ever addressed, numpy flags it
C-contiguous, and numpy(out=) accepts the very same array. Asserts the CORRECT behavior.

Run with: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F033.py   (exit 1 while the bug exists)
"""

import sys

import numpy as np

import jix

fails = 0

x = np.arange(12.0).reshape(4, 3)[::-1][:, 1:1]
print("x", x.shape, x.strides, "C_CONTIGUOUS:", x.flags.c_contiguous)
c = np.ascontiguousarray(x)
print("ascontiguousarray(x) is x:", c is x, "strides", c.strides)


def check(name, f):
    global fails
    try:
        r = f()
        assert r.shape == (4, 0), r.shape
        print(f"  ok   {name}")
    except BaseException as e:  # noqa: BLE001
        fails += 1
        print(f"  FAIL {name}: {type(e).__name__}: {e}")


check("asarray(x)", lambda: jix.asarray(x).numpy())
check("compact(x)", lambda: jix.compact(x).numpy())
check("asarray(ascontiguousarray(x))", lambda: jix.asarray(np.ascontiguousarray(x)).numpy())
check("jix array + x", lambda: (jix.asarray(np.zeros((4, 0))) + x).numpy())

# The output path accepts the same array as a destination.
jix.compact(np.zeros((4, 0))).numpy(out=x)
print("numpy(out=<shape (4, 0) strides", x.strides, ">) accepted")

print(f"{fails} failures")
sys.exit(1 if fails else 0)
