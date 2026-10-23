# Repro for F008 (run 2026-10-23-batch3): jix.write_array into a file object opened in append
# mode ("ab") returns without error, but the archive cannot be read back. The writer patches the
# block locations and the TOC with seek-then-write, which O_APPEND redirects to EOF.
#
# Correct behavior (asserted here): write_array either raises, or the array reads back.
# Exits non-zero while the bug exists.
import os
import sys
import tempfile

import numpy as np

import jix

failures = []
d = tempfile.mkdtemp()

# Case 1: append a second array to an existing file through an "ab" handle.
p = os.path.join(d, "packed.jix")
a_np = np.arange(10, dtype=np.float32)
b_np = np.arange(6, dtype=np.int16) * 7
jix.write_array(jix.compact(a_np), p)
off = os.path.getsize(p)
try:
    with open(p, "ab") as f:
        jix.write_array(jix.compact(b_np), f)
        end = f.tell()
except Exception as e:  # noqa: BLE001
    print("case 1: write_array raised (acceptable):", type(e).__name__, e)
else:
    size = os.path.getsize(p)
    print(f"case 1: write_array returned; offset {off}, f.tell() {end}, file size {size}")
    if end != size:
        failures.append(f"case 1: f.tell() after write_array is {end}, file size is {size}")
    np.testing.assert_array_equal(jix.read_array(p, len=off).numpy(), a_np)
    try:
        got = jix.read_array(p, offset=off).numpy()
        if not np.array_equal(got, b_np):
            failures.append(f"case 1: second array read back as {got}, expected {b_np}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"case 1: second array read failed: {type(e).__name__}: {e}")

# Case 2: a fresh file opened with "ab".
q = os.path.join(d, "fresh.jix")
try:
    with open(q, "ab") as f:
        jix.write_array(jix.compact(b_np), f)
except Exception as e:  # noqa: BLE001
    print("case 2: write_array raised (acceptable):", type(e).__name__, e)
else:
    try:
        got = jix.read_array(q).numpy()
        if not np.array_equal(got, b_np):
            failures.append(f"case 2: read back as {got}, expected {b_np}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"case 2: read failed: {type(e).__name__}: {e}")

for msg in failures:
    print("FAIL:", msg)
sys.exit(1 if failures else 0)
