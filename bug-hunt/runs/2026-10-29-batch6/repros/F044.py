"""F044: free-threaded TOCTOU between the validated and the used layout of numpy arrays.

`numpy(out=)` validates `out`'s shape/strides in check_output_array (jix-py/src/array.rs:218-306)
and then re-reads the strides (array.rs:617) and the data pointer (array.rs:653) for the write;
`plain_from_numpy` (jix-py/src/ops/common/operand.rs:424-482) reads ndim, shape and strides
separately. On a GIL build no Python code runs between those reads, so the window does not exist
and this script PASSES. On a free-threaded build (3.14t/3.15t, module gil_used=false) a second
thread toggling `.shape` can land in the window; the asserts below then fail (canary overwritten,
wrong values read from past the view, or a PanicException).

Part 1 runs everywhere and shows the facts the argument rests on:
  - `x.shape = ...` is still accepted (DeprecationWarning only), keeps the data pointer, and
    changes the strides in place;
  - the setter swaps in a freshly allocated dims/strides array (the old one is freed), so even a
    single read of shape+strides can tear on a free-threaded build;
  - validated shape (4, 2) + post-reshape strides (16, 4) spans 56 bytes of a 32-byte view.
Part 2 is the race itself (only meaningful on a free-threaded interpreter).
"""

import ctypes
import sys
import sysconfig
import threading
import time
import warnings

import numpy as np

import jix

warnings.simplefilter("ignore", DeprecationWarning)

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED")) and not getattr(
    sys, "_is_gil_enabled", lambda: True
)()
print(f"python {sys.version.split()[0]}, numpy {np.__version__}, free-threaded={FREE_THREADED}")

# ---------------------------------------------------------------- part 1: facts
x = np.empty((4, 2), np.int32)
if not FREE_THREADED:
    # PyArrayObject_fields on a GIL build: refcnt, type, data, nd(+pad), dimensions, strides
    dims_ptr = lambda a: ctypes.c_void_p.from_address(id(a) + 32).value  # noqa: E731
    before_data, before_dims = x.ctypes.data, dims_ptr(x)
    x.shape = (2, 4)
    print("after x.shape = (2, 4): strides", x.strides, "data ptr unchanged:", x.ctypes.data == before_data,
          "dims array reallocated:", dims_ptr(x) != before_dims)
else:
    x.shape = (2, 4)
    print("after x.shape = (2, 4): strides", x.strides)

read_shape, strides, itemsize = (4, 2), (16, 4), 4
span = sum((n - 1) * s for n, s in zip(read_shape, strides)) + itemsize
print(f"validated shape {read_shape} + re-read strides {strides}: span {span} bytes of a 32-byte view")

# ---------------------------------------------------------------- part 2: the race
N = 20000 if FREE_THREADED else 2000
SHAPES = [(4, 2), (2, 4), (8,)]
stop = threading.Event()


def toggler(arr):
    i = 0
    while not stop.is_set():
        arr.shape = SHAPES[i % len(SHAPES)]
        i += 1
        if not FREE_THREADED:
            time.sleep(0)  # hand the GIL over so every shape is seen at check time


def race_numpy_out():
    a = jix.compact(np.arange(8, dtype=np.int32).reshape(4, 2))
    big = np.full(16, -1, np.int32)  # out = big[:8], canary = big[8:]
    out = big[:8].reshape(4, 2)
    t = threading.Thread(target=toggler, args=(out,))
    stop.clear()
    t.start()
    ok = errors = 0
    try:
        for _ in range(N):
            try:
                a.numpy(out=out)
                ok += 1
            except ValueError:
                errors += 1  # shape mismatch at check time: the clean outcome
            assert (big[8:] == -1).all(), f"numpy(out=) wrote past out: canary {big[8:]}"
    finally:
        stop.set()
        t.join()
    print(f"numpy(out=): {ok} ok, {errors} ValueError, canary intact")


def race_asarray():
    big = np.full(16, 99, np.int32)
    big[:8] = np.arange(8)
    x = big[:8].reshape(4, 2)
    good = {np.arange(8, dtype=np.int32).reshape(s).tobytes() + str(s).encode() for s in SHAPES}
    t = threading.Thread(target=toggler, args=(x,))
    stop.clear()
    t.start()
    try:
        for _ in range(N):
            v = jix.asarray(x).numpy()
            key = v.tobytes() + str(v.shape).encode()
            assert key in good, f"asarray read past x: shape {v.shape}, values {v.ravel()}"
    finally:
        stop.set()
        t.join()
    print("asarray: all reads consistent")


race_numpy_out()
race_asarray()
if not FREE_THREADED:
    print("GIL build: no Python code runs between the reads, so the race window does not exist here")
print("PASS")
