# Repro for K161 (run 2026-11-15-batch7-2). Run with .venv/bin/python.
#
# A Python scalar operand becomes a Scalar storage of the broadcast shape (jix-py operand.rs:209).
# Scalar::new runs ArrayParams::tune, whose padded-shape check rejects a shape whose element
# product fits u64 (so jix.broadcast accepts it) but whose product padded to the AUTO block shape
# overflows u64 - although a Scalar has no blocks and overwrites that block shape with all-ones.
#
# Asserts the correct behavior, so it fails (exit 1) while the bug exists. The auto block shape
# depends on the host L1d size; B = 2**31 + 1 is odd, so any power-of-two-multiple block dim
# pads it, and A * B is within B of u64::MAX, so any padding overflows.
import sys

import numpy as np

import jix

B = 2**31 + 1
A = (2**64 - 1) // B
assert A * B <= 2**64 - 1 and A < 2**63

x = jix.asarray(np.full((1, 1), 7, np.uint8))
y = jix.broadcast(x, (A, B))
print("broadcast ok:", y.shape)

# Control: an op between two lazy arrays of this shape works, and reads back.
z = y + y
print("y + y ok:", z[0:2, B - 3 : B].tolist())

failures = []
for expr, expected in [("y + 1", 8), ("y * 2", 14), ("y == 7", True), ("jix.maximum(y, 3)", 7)]:
    try:
        r = eval(expr)
        got = r[A - 2 : A, 0:3]
        ok = bool(np.all(got == expected))
        print(f"{expr}: ok, read {got.tolist()}")
        if not ok:
            failures.append(f"{expr}: wrong values {got.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"{expr}: raised {type(e).__name__}: {e}")
        failures.append(f"{expr}: {type(e).__name__}: {e}")

if failures:
    print("FAIL:", len(failures), "of 4 scalar ops rejected a valid lazy shape")
    sys.exit(1)
print("PASS")
