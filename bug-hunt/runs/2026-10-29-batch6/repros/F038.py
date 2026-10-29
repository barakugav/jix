"""F038: write_array of a lazy view whose read fails part-way (integer divide by zero in the last
block, which raises under the default errstate) leaves a partial, unreadable .jix file behind.
With append=False a retry under errstate(divide="ignore") then raises FileExistsError; with
append=True the failed array's bytes stay appended after the existing arrays.

Asserts the CORRECT behavior, so it fails while the bug exists.
"""

import os
import tempfile

import numpy as np

import jix

N = 200_000
P = {"block_shape": [1000]}


def make_q():
    a = jix.compact(np.arange(1, N + 1, dtype=np.int32), params=P)
    b = np.ones(N, dtype=np.int32)
    b[-1] = 0  # divide by zero in the LAST block only
    return a // jix.compact(b, params=P)


failures = []
with tempfile.TemporaryDirectory() as d:
    # --- append=False ---
    p = os.path.join(d, "q.jix")
    q = make_q()
    try:
        jix.write_array(q, p)
        print("A write_array: no error (unexpected)")
    except FloatingPointError as e:
        print("A write_array raises:", type(e).__name__, e)
    exists = os.path.exists(p)
    print("A file exists after error:", exists, "size:", os.path.getsize(p) if exists else None)
    if exists:
        try:
            jix.read_array(p)
            print("A read back partial: ok")
        except Exception as e:
            print("A read back partial:", type(e).__name__, e)
        failures.append("partial file left at a new path after a failed write_array")
    try:
        with jix.errstate(divide="ignore"):
            jix.write_array(q, p)
        print("A retry with divide=ignore: ok")
        np.testing.assert_array_equal(jix.read_array(p).numpy()[-3:], [N - 2, N - 1, 0])
    except Exception as e:
        print("A retry with divide=ignore:", type(e).__name__, e)
        failures.append(f"retry after fixing errstate raises {type(e).__name__}")

    # --- append=True ---
    p2 = os.path.join(d, "packed.jix")
    first = jix.compact(np.arange(10, dtype=np.int32))
    jix.write_array(first, p2)
    size0 = os.path.getsize(p2)
    try:
        jix.write_array(make_q(), p2, append=True)
        print("B append: no error (unexpected)")
    except FloatingPointError as e:
        print("B append raises:", type(e).__name__, e)
    size1 = os.path.getsize(p2)
    print("B sizes before/after failed append:", size0, size1)
    if size1 != size0:
        failures.append(f"failed append left {size1 - size0} garbage bytes after the existing array")

print()
print("FAILURES:", failures)
assert not failures, failures
