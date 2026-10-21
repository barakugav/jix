# Verifier repro for F013 (run 2026-10-21-pilot): unchecked C-order stride products on shapes that
# contain a 0 dim but whose other dims multiply past u64. Debug build of the bindings: a Rust
# overflow panic surfaces as pyo3_runtime.PanicException.
# Run: .venv/bin/python bug-hunt/runs/2026-10-21-pilot/repros/F013.py
import numpy as np

import jix

BIG = 1 << 40


def attempt(name, fn):
    try:
        r = fn()
        print(f"[{name}] OK: {r!r}")
    except BaseException as e:  # PanicException derives from BaseException
        print(f"[{name}] {type(e).__module__}.{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")


empty = np.zeros(0, np.float32)
attempt("reshape (0, 2^40, 2^40)", lambda: jix.reshape(empty, (0, BIG, BIG)).shape)
attempt("reshape (2^40, 2^40, 0)", lambda: jix.reshape(empty, (BIG, BIG, 0)).shape)
attempt("reshape (0, 2^31, 2^31) control", lambda: jix.reshape(empty, (0, 1 << 31, 1 << 31)).shape)
attempt(
    "broadcast (0, 2^40, 2^40) then compact()",
    lambda: jix.compact(jix.broadcast(np.zeros((0, 1, 1), np.float32), (0, BIG, BIG))).shape,
)
