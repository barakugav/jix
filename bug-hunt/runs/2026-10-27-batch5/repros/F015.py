"""F015 repro: Array.plain() of a dtype with an inner shape changes shape and dtype.

plain()'s docstring promises "the same dtype as self and shape determined by the index argument
(or self.shape when no index is supplied)", and presents it as the uncompressed counterpart of
compact(). For a dtype with an inner shape it instead returns the base dtype with the inner dims
appended to the array shape.

Asserts the CORRECT (documented) behavior, so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F015.py
"""

import numpy as np

import jix

dt = np.dtype([("a", "i4", (3,)), ("b", "i4")])
x = np.zeros((2, 2), dtype=dt)
x["a"] = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
f = jix.dtype_sub_field(jix.compact(x), "a")  # shape (2, 2), dtype ('<i4', (3,))
c = f.compact()
p = f.plain()
print("self   : shape", f.shape, "ndim", f.ndim, "dtype", f.dtype)
print("compact: shape", c.shape, "ndim", c.ndim, "dtype", c.dtype)
print("plain  : shape", p.shape, "ndim", p.ndim, "dtype", p.dtype)

# Index forms: f[1] keeps the inner shape in its numpy() result; p[1] is a different selection.
print("f[1].shape", f[1].shape, " c[1].shape", c[1].shape, " p[1].shape", p[1].shape)
print("f[:, 1].shape", f[:, 1].shape, " p[:, 1].shape", p[:, 1].shape)
try:
    print("p[:, :, 0] ->", p[:, :, 0].shape, "(f[:, :, 0] would be an IndexError)")
except Exception as e:  # noqa: BLE001
    print("p[:, :, 0] raised", type(e).__name__, e)
try:
    f[:, :, 0]
except Exception as e:  # noqa: BLE001
    print("f[:, :, 0] raised", type(e).__name__)

# Silent divergence: an index anchored at the end selects a different dim on plain().
print("f[..., 1] ->", f[..., 1].shape, f[..., 1].ravel().tolist())
print("p[..., 1] ->", p[..., 1].shape, p[..., 1].ravel().tolist())

# With an index: plain(index) vs compact()[index]-style shape.
pi = f.plain(1)
print("f.plain(1): shape", pi.shape, "dtype", pi.dtype, "(expected shape (2,), dtype", f.dtype, ")")

# Ops that apply differ.
for name, arr in (("self", f), ("compact", c), ("plain", p)):
    try:
        print(f"{name}.sum() ->", arr.sum().item())
    except Exception as e:  # noqa: BLE001
        print(f"{name}.sum() raised", type(e).__name__, e)

# compact() keeps shape and dtype - the reference for "materialize this view".
assert c.shape == f.shape and c.dtype == f.dtype

# The documented contract of plain(): same dtype as self, shape determined by the index.
assert p.dtype == f.dtype, f"plain() dtype {p.dtype} != self dtype {f.dtype}"
assert p.shape == f.shape, f"plain() shape {p.shape} != self shape {f.shape}"
assert pi.dtype == f.dtype and pi.shape == (2,), f"plain(1): {pi.shape} {pi.dtype}"
np.testing.assert_array_equal(p.numpy(), f.numpy())
print("OK")
