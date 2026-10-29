# Repro for F015 (run 2026-10-29-batch6): `block_shape_fixed_dims` without `block_shape` is rejected
# for a numpy input but, for a jix input, silently replaced by the source's flags.
# Asserts the CORRECT behavior for the jix input - rejected like the numpy input, or accepted with
# the caller's flags honored (dim 0 pinned at the inherited 8, dim 1 rescaled to block_size) - so it
# fails while the bug exists. Python exposes no block-shape getter, so the stored block shape is
# read from the archive header (ArrayHeader.block_shape = field 2, packed varints).
import io

import numpy as np

import jix

src = np.arange(64 * 64, dtype=np.int32).reshape(64, 64)
params = {"block_shape_fixed_dims": [True, False], "block_size": 8 * 64 * 4}


def header_block_shape(arr):
    buf = io.BytesIO()
    arr.write_to(buf)
    data = buf.getvalue()
    for shape in ([8, 32], [8, 64], [64, 64]):
        if bytes([0x12, 2, *shape]) in data:
            return shape
    return "unknown"


try:
    jix.compact(src, params=params)
    print("numpy input: accepted")
except Exception as e:  # noqa: BLE001
    print(f"numpy input: {type(e).__name__}: {e}")

jix_src = jix.compact(src, params={"block_shape": [8, 32]})
print("jix source block shape:", header_block_shape(jix_src))
try:
    out = jix.compact(jix_src, params=params)
except Exception as e:  # noqa: BLE001
    print(f"jix input: {type(e).__name__}: {e}")  # consistent with the numpy path: OK
else:
    assert (out.numpy() == src).all()
    got = header_block_shape(out)
    print("jix input: accepted, stored block shape", got)
    assert got == [8, 64], f"caller's block_shape_fixed_dims [True, False] silently dropped: block shape {got}"
