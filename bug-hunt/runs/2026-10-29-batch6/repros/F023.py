"""F023 repro: read_array(file_like) leaves the stream at an arbitrary position, so arrays
written back-to-back with write_array(f) cannot be read back with successive read_array(f).

Asserts the CORRECT behavior (stream at the end of each archive after read_array, like
write_array leaves it and like np.load does), so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F023.py
"""

import io
import tempfile
import os

import numpy as np

import jix

failures = []


def check(name, cond, msg):
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}: {msg}")
    if not cond:
        failures.append(name)


def packed(arrays):
    buf = io.BytesIO()
    ends = []
    for a in arrays:
        jix.write_array(jix.compact(a), buf)
        ends.append(buf.tell())
    return buf, ends


# --- writer side: write_array leaves the stream at the end of each archive -------------------
a = np.arange(10, dtype=np.int32)
b = np.arange(5.0)
buf, ends = packed([a, b])
check("writer", ends[1] == len(buf.getvalue()), f"archive ends {ends}, total {len(buf.getvalue())}")

# --- (1) small arrays, BytesIO, len=None ------------------------------------------------------
buf.seek(0)
a2 = jix.read_array(buf)
pos = buf.tell()
check("small: tell after read", pos == ends[0], f"buf.tell() = {pos}, end of a = {ends[0]}")
try:
    b2 = jix.read_array(buf)
    check("small: second read", np.array_equal(b2.numpy(), b), f"got {b2.numpy()}")
except Exception as e:  # noqa: BLE001
    check("small: second read", False, f"raised {type(e).__name__}: {e}")

# --- (2) len= given (exact archive length): BufReader still reads past it --------------------
buf.seek(0)
a2 = jix.read_array(buf, len=ends[0])
pos = buf.tell()
check("small len=: tell after read", pos == ends[0], f"buf.tell() = {pos}, end of a = {ends[0]}")

# --- (3) larger first array: position is INSIDE the first archive ----------------------------
big = np.arange(50000, dtype=np.float64)
buf, ends = packed([big, b])
buf.seek(0)
jix.read_array(buf)
pos = buf.tell()
check("big: tell after read", pos == ends[0], f"buf.tell() = {pos}, end of a = {ends[0]}")
try:
    b2 = jix.read_array(buf)
    check("big: second read", np.array_equal(b2.numpy(), b), f"got {b2.numpy()}")
except Exception as e:  # noqa: BLE001
    check("big: second read", False, f"raised {type(e).__name__}: {e}")

# --- (4) real file object, three arrays ------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "packed.jix")
    arrs = [np.arange(3, dtype=np.int64), np.ones((2, 3), np.float32), np.arange(7, dtype=np.uint8)]
    with open(p, "wb") as f:
        for x in arrs:
            jix.write_array(jix.compact(x), f)
    got = []
    try:
        with open(p, "rb") as f:
            for _ in arrs:
                got.append(jix.read_array(f).numpy())
        ok = all(np.array_equal(g, x) for g, x in zip(got, arrs))
        check("file: three successive reads", ok, f"got {got}")
    except Exception as e:  # noqa: BLE001
        check("file: three successive reads", False, f"read #{len(got) + 1} raised {type(e).__name__}: {e}")

# --- reference: numpy's np.save / np.load on a packed stream ---------------------------------
nbuf = io.BytesIO()
np.save(nbuf, a)
np.save(nbuf, b)
nbuf.seek(0)
np.load(nbuf)
print(f"[ref ] numpy: second np.load -> {np.load(nbuf)}")

if failures:
    raise SystemExit(f"FAILED: {failures}")
print("all ok")
