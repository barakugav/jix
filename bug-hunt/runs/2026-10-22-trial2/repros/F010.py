"""F010: jix.concatenate of two u8 arrays whose concat-dim lengths sum past 2**64.

Asserts the CORRECT behavior (a regular Python exception for an overflowing shape), so it fails
while the bug exists: debug builds raise pyo3 PanicException ("attempt to add with overflow" at
jix/src/ops/shape_ops/concatenate.rs:111), release builds return a shape-(4,) array.
"""

import numpy as np

import jix

y = jix.repeat(np.array([0, 1], np.uint8), 2**62 + 1)  # length 2**63 + 2, lazy
print("y.shape =", y.shape)

try:
    c = jix.concatenate([y, y])
except Exception as e:  # ValueError / RuntimeError etc. = correct
    print("OK: raised", type(e).__name__, e)
except BaseException as e:  # PanicException derives from BaseException
    print("BUG: raised", type(e).__module__, type(e).__name__, e)
    raise SystemExit(1)
else:
    print("BUG: returned shape", c.shape, "contents", c.numpy() if c.size <= 16 else "<big>")
    raise SystemExit(1)
