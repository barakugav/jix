"""F027: Array.numpy() documents a C-contiguous result (jix-py/src/array.rs:491-492) but the
allocation follows the storage's read_layout_order (array.rs:542-551, util.rs:102-128).

Asserts the DOCUMENTED (correct per docstring) behavior, so it fails while the docstring and the
behavior disagree. Values are always correct; only the memory layout differs.
"""

import ctypes

import numpy as np

import jix

results = []


def check(label, j, expected):
    n = j.numpy()
    ok_vals = np.array_equal(n, expected)
    print(
        f"{label:34s} values_ok={ok_vals} C={n.flags.c_contiguous} F={n.flags.f_contiguous} "
        f"strides={n.strides}"
    )
    results.append((label, ok_vals, n.flags.c_contiguous))
    return n


base = np.arange(12.0).reshape(3, 4)
f = np.asfortranarray(base)

check("compact(C)", jix.compact(base), base)
check("compact(F)", jix.compact(f), base)
n1 = check("asarray(F)", jix.asarray(f), base)
check("permute_dims(compact, [1,0])", jix.permute_dims(jix.compact(base.T.copy()), [1, 0]), base)
check("asarray(F) + 1", jix.asarray(f) + 1, base + 1)
check("asarray(F)[0:2] (getitem path)", jix.asarray(f), base)  # numpy() without index
g = jix.asarray(f)[0:2, :]
print(f"{'asarray(F)[0:2, :] via __getitem__':34s} C={g.flags.c_contiguous} F={g.flags.f_contiguous} strides={g.strides}")
results.append(("getitem", True, g.flags.c_contiguous))

# What a consumer that trusts the docstring sees: raw memory read via .ctypes.data as row-major.
raw = np.ctypeslib.as_array(
    ctypes.cast(n1.ctypes.data, ctypes.POINTER(ctypes.c_double)), shape=(12,)
).copy()
print("raw memory of asarray(F).numpy():", raw.tolist())
print("row-major expected            :", base.ravel().tolist())

# np.frombuffer over the buffer protocol
try:
    fb = np.frombuffer(memoryview(n1), dtype=np.float64)
    print("np.frombuffer(memoryview(x)):", fb.tolist())
except Exception as e:  # noqa: BLE001
    print("np.frombuffer(memoryview(x)) raised:", type(e).__name__, e)

bad = [label for label, ok, c in results if not c]
assert all(ok for _, ok, _ in results), "values differ"
assert not bad, f"numpy() result not C-contiguous (docstring promises C-contiguous) for: {bad}"
print("OK")
