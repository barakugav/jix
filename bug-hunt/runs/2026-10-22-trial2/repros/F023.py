# F023: Repeat computes `out_strides[k] * repeats` (jix/src/ops/shape_ops/repeat.rs:181) even when
# output dim k has length 1. With an ordinary numpy `out=` (one row of a wider 2-D array) and a
# large lazy repeat count the product overflows usize: the debug build raises
# pyo3_runtime.PanicException. Asserts the CORRECT result, so it fails while the bug exists.
import numpy as np

import jix

a = jix.compact(np.array([[1, 2, 3, 4]], dtype=np.uint8))
r = jix.repeat(a, 2**50, dim=0)  # shape (2**50, 4), lazy

# Plain numpy basic slicing, no stride tricks: shape (1, 4), strides (16384, 1).
big = np.zeros((2, 2**14), dtype=np.uint8)
out = big[0:1, 0:4]
print("out shape/strides:", out.shape, out.strides)

failed = False
try:
    r.numpy(slice(5, 6), out=out)
except BaseException as e:  # PanicException derives from BaseException
    print("FAIL (row pitch):", type(e).__name__, e)
    failed = True
else:
    print("row pitch result:", big[0, :4])
    assert big[0, :4].tolist() == [1, 2, 3, 4]

# Also with a stepped slice: np.zeros((8, 4))[::2**60] -> shape (1, 4), strides (2**62, 1).
r4 = jix.repeat(a, 4, dim=0)
big2 = np.zeros((8, 4), dtype=np.uint8)
out2 = big2[:: 2**60]
print("out2 shape/strides:", out2.shape, out2.strides)
try:
    r4.numpy(slice(1, 2), out=out2)
except BaseException as e:
    print("FAIL (stepped slice):", type(e).__name__, e)
    failed = True
else:
    print("stepped slice result:", big2[0])
    assert big2[0].tolist() == [1, 2, 3, 4]

assert not failed, "repeat with a length-1 out dim panicked"
print("OK")
