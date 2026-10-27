"""Repro for F021 (run 2026-10-27-batch5), Python API.

A source with an explicit (all-fixed) block shape of 2^26 + 8 uint8 elements (64 MiB per block)
is valid. Casting it to complex128 keeps the block shape in elements, so jix.compact() with
default params re-checks a 1 GiB + 128 B block and raises, although the view is valid.

Asserts the correct behavior (compact succeeds), so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F021.py
"""

import numpy as np

import jix

n = (1 << 26) + 8
a = jix.compact(np.zeros(n, dtype=np.uint8), params={"block_shape": [n]})
b = jix.cast(a, np.complex128)
try:
    c = jix.compact(b)
except Exception as e:  # noqa: BLE001
    print(f"jix.compact(cast view) raised {type(e).__name__}: {e}")
    raise AssertionError("compact() of a valid widened view failed") from e
print("ok:", c.shape, c.dtype)
