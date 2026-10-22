# F018 repro: Array.numpy(out=x) where the lazy chain reads x's own memory (a Plain view made
# by jix.asarray, or by the implicit numpy -> jix conversion of an operator).
#
# Correct behavior (what this script asserts): either the result equals the out-of-place result
# (what NumPy gives for ufuncs with an overlapping out=), or a ValueError for the overlapping out.
# While the bug exists the script reports wrong values and exits 1.
#
# The identity case (asarray(x).numpy(out=x)) runs in a subprocess because a Plain push copy with
# src == dst may trip std's copy_nonoverlapping precondition check and abort the process.
import subprocess
import sys

import numpy as np

import jix

failures = []


def check(name, run, expected, out):
    """`run()` writes into `out`; correct iff out == expected afterwards, or ValueError."""
    before = out.copy()
    try:
        r = run()
    except ValueError as e:
        print(f"[ok]   {name}: ValueError: {e}")
        return
    assert r is out
    if np.array_equal(out, expected):
        print(f"[ok]   {name}: correct values")
    else:
        print(f"[FAIL] {name}: wrong values")
        print("       before:  ", before.tolist())
        print("       expected:", expected.tolist())
        print("       got:     ", out.tolist())
        failures.append(name)


# A. Hunter's trigger: transpose of an asarray view written back into the same array.
x = np.arange(16, dtype=np.float32).reshape(4, 4)
exp = x.T + 0
a = jix.asarray(x)
check("A: (asarray(x).T + 0).numpy(out=x)", lambda: (a.T + 0).numpy(out=x), exp, x)

# B. No explicit asarray: the numpy operand of `+` is wrapped zero-copy by the operator itself.
x = np.arange(16, dtype=np.float32).reshape(4, 4)
exp = x.T + 1
c = jix.compact(np.ones((4, 4), dtype=np.float32))
check("B: (compact(ones) + x.T).numpy(out=x)", lambda: (c + x.T).numpy(out=x), exp, x)

# C. Shifted 1-d view, an op whose result differs from the overwritten neighbour.
y = np.arange(100, dtype=np.int64)
exp = y[:-1] * 2
b = jix.asarray(y[:-1])
y_out = y[1:]
check("C: (asarray(y[:-1]) * 2).numpy(out=y[1:])", lambda: (b * 2).numpy(out=y_out), exp, y_out)

# D. Pure shape op (no element-wise pipeline): a Plain push copy whose src overlaps dst.
x = np.arange(16, dtype=np.float32).reshape(4, 4)
exp = x.T.copy()
check("D: asarray(x).T.numpy(out=x)", lambda: jix.asarray(x).T.numpy(out=x), exp, x)

# E. Identity: src == dst exactly, in a subprocess (may abort on copy_nonoverlapping's check).
code = """
import numpy as np, jix
x = np.arange(16, dtype=np.float32).reshape(4, 4)
jix.asarray(x).numpy(out=x)
assert np.array_equal(x, np.arange(16, dtype=np.float32).reshape(4, 4))
print("identity: values ok")
"""
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/")
tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
if p.returncode == 0:
    print("[ok]   E: asarray(x).numpy(out=x):", tail)
else:
    print(f"[FAIL] E: asarray(x).numpy(out=x): subprocess exit status {p.returncode}")
    for line in tail:
        print("       ", line)
    failures.append("E")

print("failures:", failures)
sys.exit(1 if failures else 0)
