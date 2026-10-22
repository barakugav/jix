"""F007 repro: non-native (big-endian) numpy dtypes are treated as little-endian.

Asserts the CORRECT behavior (values preserved, or the non-native byte order rejected
with TypeError/ValueError), so each check FAILS while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F007.py
"""

import os
import tempfile

import numpy as np

import jix

be = np.array([1, 256, -2], dtype=">i4")
assert be.tolist() == [1, 256, -2]


def check(label, fn, expected):
    try:
        got = fn()
    except (TypeError, ValueError) as e:  # rejecting non-native byte order is also correct
        print(f"PASS  {label}: rejected ({type(e).__name__}: {e})")
        return True
    got = np.asarray(got)
    ok = np.array_equal(got, expected)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: got {got.tolist()} dtype={got.dtype.str}, expected {np.asarray(expected).tolist()}")
    return ok


def write_read_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jix")
        jix.write_array(be, p)
        return jix.read_array(p).numpy()


results = [
    # 0-d big-endian goes through numpy .item() and is correct - control case
    check("asarray(0-d >i4 = 256).numpy()", lambda: jix.asarray(np.array(256, ">i4")).numpy(), 256),
    check("asarray(>i4).numpy()", lambda: jix.asarray(be).numpy(), [1, 256, -2]),
    check("compact(>i4).numpy()", lambda: jix.compact(be).numpy(), [1, 256, -2]),
    check("add(>i4, 1).numpy()", lambda: jix.add(be, 1).numpy(), [2, 257, -1]),
    check("sum(>i4).numpy()", lambda: jix.sum(be).numpy(), 255),
    check("write_array(>i4) -> read_array", write_read_roundtrip, [1, 256, -2]),
    check("asarray(<i4 [1,2], dtype='>f8').numpy()",
          lambda: jix.asarray(np.array([1, 2], "<i4"), dtype=">f8").numpy(), [1.0, 2.0]),
    check("cast(f8 [1.5,2.0], '>f8').numpy()",
          lambda: jix.cast(np.array([1.5, 2.0]), np.dtype(">f8")).numpy(), [1.5, 2.0]),
    check("compact([1,2], dtype='>i4').numpy()",
          lambda: jix.compact(np.array([1, 2], "<i4"), dtype=">i4").numpy(), [1, 2]),
]
n_fail = results.count(False)
print(f"\n{n_fail} of {len(results)} checks failed")
assert n_fail == 0, "F007: big-endian numpy dtypes silently misread / mislabeled"
