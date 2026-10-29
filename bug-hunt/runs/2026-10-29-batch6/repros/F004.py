"""F004 (run 2026-10-29-batch6): Array.__len__ truncates a dim >= 2**32 on 32-bit wheels.

jix-py/src/array.rs:417 returns `Ok(*len as usize)`. On a 32-bit CPython (x86 / armv7 / win32
wheels) usize is 32 bits, so a dim of 2**32 + 5 is returned as 5, which pyo3 passes through as a
valid Py_ssize_t (it only raises OverflowError above sys.maxsize).

Asserts the CORRECT behavior: len(x) == x.shape[0], or OverflowError. Passes on a 64-bit build
(this checkout's .venv); expected to FAIL on a 32-bit wheel with "len() returned 5".
"""

import struct
import sys

import numpy as np

import jix

N = 2**32 + 5
print(f"python pointer width: {struct.calcsize('P') * 8} bits, sys.maxsize = {sys.maxsize}")

base = jix.compact(np.zeros(1, np.uint8))
cases = {
    "broadcast": jix.broadcast(base, (N,)),
    "tile": jix.tile(base, N),
    "repeat": jix.repeat(base, N),
}

failures = []
for name, x in cases.items():
    try:
        got = len(x)
    except OverflowError:
        print(f"{name}: shape[0]={x.shape[0]} size={x.size} len() -> OverflowError (acceptable)")
        continue
    print(f"{name}: shape[0]={x.shape[0]} size={x.size} len() -> {got}")
    if got != x.shape[0]:
        failures.append(f"{name}: len() returned {got}, shape[0] is {x.shape[0]}")

assert not failures, "; ".join(failures)
print("OK")
