"""F003 (run 2026-10-29-batch6): Python reachability of the whole-span aliasing UB.

This script cannot observe the UB itself (the values come out right; only Miri sees it). It shows
that the Python API accepts calls where `out` and an asarray'd input share NO element
(np.shares_memory is False, so the numpy() docstring - "out must not overlap memory the array
itself reads from" - permits them) while their strided spans interleave
(np.may_share_memory is True). For each one jix builds a `&[u8]` over the input's whole span
(jix/src/storage/plain.rs:468) and a `&mut [u8]` over out's whole span
(jix-py/src/array.rs:651), so the two Rust slices cover each other's elements.

The UB is shown by the Miri run of
jix/tests/repro_2026_10_29_batch6__f003.rs::jixpy_numpy_out_column_into_column_pattern,
which replays jix-py's numpy(out=) steps for case 2 below.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F003.py  (it passes; see above)
"""

import numpy as np

import jix


def case(name, make_in, make_out, build, expect_row):
    x = np.arange(12, dtype=np.uint8).reshape(3, 4)
    src, out = make_in(x), make_out(x)
    shares = np.shares_memory(src, out)
    may = np.may_share_memory(src, out)
    r = build(jix.asarray(src)).numpy(out=out)
    print(f"{name}: shares_memory={shares} may_share_memory={may} accepted={r is out} x={x.tolist()}")
    assert not shares and may and r is out
    assert x.tolist() == expect_row, x.tolist()


# 1. out (row 1, cols 1..4: bytes 5..7) lies inside the input column's span [0, 9).
case("plain push, out inside input span", lambda x: x[:, 0], lambda x: x[1, 1:4], lambda a: a,
     [[0, 1, 2, 3], [4, 0, 4, 8], [8, 9, 10, 11]])
# 2. column into the next column: out's span [1, 10) covers input elements 4, 8, and the input's
#    span [0, 9) covers out elements 1, 5.
case("plain push, column -> column", lambda x: x[:, 0], lambda x: x[:, 1], lambda a: a,
     [[0, 0, 2, 3], [4, 4, 6, 7], [8, 8, 10, 11]])
# 3. same through the element-wise pipeline.
case("pipeline, column -> column", lambda x: x[:, 0], lambda x: x[:, 1], lambda a: a * np.uint8(2),
     [[0, 0, 2, 3], [4, 8, 6, 7], [8, 16, 10, 11]])
print("all accepted: the Python API reaches the aliasing violation Miri reports for the Rust replay")
