# F017: an empty array that jix itself writes cannot be read back when its padded shape product
# (zero dims counted as 1) overflows u64. Asserts the CORRECT behavior: fails while the bug exists.
import io

import numpy as np

import jix

shape = (0, 3, 0x5555555555555555)  # 3 * 0x5555555555555555 == 2**64 - 1
a = jix.broadcast(jix.compact(np.zeros((0, 1, 1), np.uint8)), shape)
print("source shape:", a.shape)
print("in-memory compact shape:", jix.compact(a).shape)

buf = io.BytesIO()
jix.write_array(a, buf)
print("written", len(buf.getvalue()), "bytes")
buf.seek(0)
try:
    b = jix.read_array(buf)
except Exception as e:
    print("read back FAILED:", type(e).__name__, e)
    raise
print("read back OK:", b.shape)
assert tuple(b.shape) == shape
