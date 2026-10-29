"""F034 repro: an empty numpy operand whose non-empty dims have large strides panics in debug
builds (overflow in strided_span_bytes, jix/src/util/mod.rs:111) instead of reading as empty.

Asserts the CORRECT behavior, so it fails while the bug exists.
Run with: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F034.py
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

import jix

# NumPy accepts any strides for a zero-element array: nothing is ever addressed.
e = as_strided(np.zeros(1), shape=(3, 3, 0), strides=(2**62, 2**62, 8))
y = np.ones((3, 3, 2))

fails = 0


def check(name, f, expected_shape, fill=None):
    global fails
    try:
        r = f()
        assert r.shape == expected_shape, r.shape
        if fill is not None:
            assert np.all(r == fill)
        print(f"ok   {name}")
    except BaseException as ex:  # noqa: BLE001
        fails += 1
        print(f"FAIL {name}: {type(ex).__name__}: {ex}")


check("asarray(e).numpy()", lambda: jix.asarray(e).numpy(), (3, 3, 0))
check("concatenate([y, e, y], 2)", lambda: jix.concatenate([y, e, y], 2).numpy(), (3, 3, 4), 1.0)
check("asarray(e) + 1", lambda: (jix.asarray(e) + 1).numpy(), (3, 3, 0))
check("compact(e)", lambda: jix.compact(e).numpy(), (3, 3, 0))

print(f"{fails} failures")
assert fails == 0
