"""F013 repro: Array.numpy(out=) accepts a self-overlapping destination (positive, non-zero strides
that map two different indices onto the same bytes). check_output_array (jix-py/src/array.rs)
only rejects negative and zero strides, so the StridedBuf::from_raw_parts_mut no-alias
precondition is broken. Reductions fold their running states straight into `out`
(state_in_out_buf, jix/src/ops/reduction.rs), so cells that share bytes accumulate each other's
partial results and the buffer ends up holding values that belong to no output cell.

Asserts the CORRECT behavior, so it fails while the bug exists: either numpy(out=) raises
ValueError (like it does for reversed / broadcast views), or every element of the backing buffer
equals what NumPy's own reduction leaves there (last-write-wins over the true per-cell results).
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view

import jix

failures = []


def check(name, jix_read, np_reduce, src, make_out):
    """Run the jix read and the NumPy reduction into identical overlapping destinations."""
    buf, out = make_out()
    ref_buf, ref_out = make_out()
    np_reduce(src, ref_out)
    try:
        jix_read(out)
    except ValueError as e:
        print(f"{name}: rejected with ValueError: {e}")
        return
    print(f"{name}: jix buf   = {buf}")
    print(f"{name}: numpy buf = {ref_buf}")
    if not np.array_equal(buf, ref_buf):
        failures.append(name)


src = np.arange(1, 2 * 4 * 3 + 1, dtype=np.int64).reshape(2, 4, 3)
a = jix.compact(src)
print("true per-cell sums:", src.sum(0).ravel())


def sliding():
    # shape (4, 3), strides (8, 8): cell (i, j) lives at buf[i + j]
    buf = np.zeros(6, dtype=np.int64)
    out = sliding_window_view(buf, 3, writeable=True)
    assert out.flags.writeable and out.strides == (8, 8)
    return buf, out


def strided():
    # the same layout built with as_strided
    buf = np.zeros(6, dtype=np.int64)
    return buf, as_strided(buf, shape=(4, 3), strides=(8, 8))


check(
    "sum/sliding_window_view",
    lambda out: a.sum(dim=0).numpy(out=out),
    lambda s, out: np.sum(s, axis=0, out=out),
    src,
    sliding,
)
check(
    "prod/as_strided",
    lambda out: a.prod(dim=0).numpy(out=out),
    lambda s, out: np.prod(s, axis=0, out=out),
    src,
    strided,
)

# A plain copy into the same view is last-write-wins, identical to NumPy (not a failure).
buf, out = sliding()
a.numpy(0, out=out)
ref_buf, ref_out = sliding()
np.copyto(ref_out, src[0])
print("copy: jix buf =", buf, " numpy buf =", ref_buf)

if failures:
    raise AssertionError(f"self-overlapping out accepted and result differs from NumPy: {failures}")
print("OK")
