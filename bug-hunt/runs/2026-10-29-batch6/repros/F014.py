# Repro for F014 (run 2026-10-29-batch6): jix.asarray(params={"block_shape": ...}) drops the
# block shape, although its docstring says arrays later created from it (compact(), write_array())
# inherit the params. Asserts the documented behavior, so it FAILS while the bug exists.
#
# Python has no block-shape accessor, so the layout is compared through the serialized archive:
# compacting the same data with the same block shape and codec must give the same .jix bytes.
import io
import time

import numpy as np

import jix

n = 1024
src = np.arange(n * n, dtype=np.float32).reshape(n, n)
params = {"block_shape": [n, 1]}  # column blocks


def archive_bytes(arr):
    buf = io.BytesIO()
    arr.write_to(buf)
    return buf.getvalue()


def col_read_time(x, reps=20):
    t = time.perf_counter()
    for _ in range(reps):
        np.asarray(x[:, 7])  # __getitem__ materializes
    return (time.perf_counter() - t) / reps


reference = jix.compact(src, params=params)  # params given to compact directly
via_asarray = jix.asarray(src, params=params).compact()  # documented to inherit them
auto = jix.compact(src)  # default params

ref_b, via_b, auto_b = archive_bytes(reference), archive_bytes(via_asarray), archive_bytes(auto)
via_w = archive_bytes(jix.asarray(src, params=params))  # write_array/write_to of the plain view
print("archive bytes: reference=%d via_asarray=%d auto=%d asarray.write_to=%d" % (len(ref_b), len(via_b), len(auto_b), len(via_w)))
print("via_asarray == reference:", via_b == ref_b, "| via_asarray == auto:", via_b == auto_b)
print("asarray.write_to == reference:", via_w == ref_b, "| asarray.write_to == auto:", via_w == auto_b)
print("column read: reference %.6f s, via_asarray %.6f s" % (col_read_time(reference), col_read_time(via_asarray)))

np.testing.assert_array_equal(via_asarray.numpy(), src)  # values are fine
assert via_b == ref_b, "asarray(params={'block_shape': [n, 1]}).compact() did not inherit the block shape"
assert via_w == ref_b, "asarray(params={'block_shape': [n, 1]}).write_to() did not inherit the block shape"
print("OK")
