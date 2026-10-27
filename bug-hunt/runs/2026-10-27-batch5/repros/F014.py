"""F014: numpy(out=) with an EMPTY out whose dim of extent > 1 has a negative stride.

check_output_array (jix-py/src/array.rs:251) returns Ok for any shape containing a 0 before the
negative-stride check, and numpy() then rebuilds the strides (array.rs:588-617) before its
`nitems == 0` early return (array.rs:620). A negative stride on a non-dropped dim with extent > 1
reaches `debug_assert!(read_shape[dim] <= 1)` at array.rs:607.

Correct behavior (asserted here): the empty out is returned as-is, nothing is written, and no
exception is raised - exactly what happens for an empty out without negative strides, and what
release builds do. In the debug build installed in .venv this script FAILS with PanicException.
Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F014.py
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

import jix


def check(label, arr, dst, host, index=None):
    before = host.copy()
    print(f"{label}: dst shape {dst.shape} strides {dst.strides} writeable {dst.flags.writeable}")
    try:
        r = arr.numpy(index, out=dst)
    except BaseException as e:  # PanicException derives from BaseException
        print(f"  FAIL: raised {type(e).__module__}.{type(e).__name__}: {str(e)[:200]}")
        return False
    ok = r is dst and r.shape == dst.shape and np.array_equal(host, before)
    print(f"  {'ok' if ok else 'FAIL'}: returned out unchanged = {ok}")
    return ok


a = jix.compact(np.zeros((0, 3), dtype=np.float64))
results = []

# Control: an ordinary empty out (numpy C-order strides) - passes.
host = np.zeros(8)
results.append(check("control C-order empty out", a, np.empty((0, 3)), host))

# Control: an ordinary reversed slice of an empty array.
host = np.zeros(8)
results.append(check("control empty[:, ::-1]", a, np.empty((0, 3))[:, ::-1], host))

# Trigger 1: as_strided, empty dim 0, negative stride on dim 1 (extent 3).
host = np.zeros(8)
results.append(check("as_strided (24, -8)", a, as_strided(host[4:], shape=(0, 3), strides=(24, -8)), host))

# Trigger 2: np.ndarray with explicit strides over a buffer.
host = np.zeros(8)
results.append(
    check(
        "np.ndarray strides=(24, -8)",
        a,
        np.ndarray(shape=(0, 3), dtype=np.float64, buffer=host, offset=16, strides=(24, -8)),
        host,
    )
)

# Trigger 3: same thing through an index selection that is empty (a[:, 0:0] style is not
# needed - an empty selection on a non-empty array with a negative stride on the other dim).
b = jix.compact(np.arange(12, dtype=np.float64).reshape(3, 4))
host = np.zeros(8)
results.append(
    check(
        "b[0:0, :] with np.ndarray strides=(32, -8)",
        b,
        np.ndarray(shape=(0, 4), dtype=np.float64, buffer=host, offset=24, strides=(32, -8)),
        host,
        index=(slice(0, 0), slice(None)),
    )
)

# Trigger 4 (same root cause, different debug panic): the negative stride sits on the empty dim,
# so the debug_assert holds, but the C-order replacement `strides[1] * read_shape[1]`
# (array.rs:609) multiplies an unchecked huge stride of the next dim: "attempt to multiply
# with overflow" in debug builds.
host = np.zeros(8)
results.append(
    check(
        "np.ndarray strides=(-8, 2**63 - 8)",
        a,
        np.ndarray(shape=(0, 3), dtype=np.float64, buffer=host, offset=16, strides=(-8, 2**63 - 8)),
        host,
    )
)

print("ALL OK" if all(results) else "FAILED")
assert all(results), "numpy(out=) raised on a valid empty out with a negative stride"
