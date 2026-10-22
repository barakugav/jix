"""F026 repro: a packed (align=False) numpy struct dtype whose fields all have alignment 1
trips the debug_assert in jix-py/src/dtype.rs:208-211 (dtype_from_numpy).

Asserts the CORRECT behavior (asarray / compact succeed and round-trip the values), so it fails
while the bug exists. In the debug build (maturin develop) it raises PanicException.
"""

import numpy as np

import jix

CASES = [
    np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")]),  # RGB pixel
    np.dtype([("x", "i1")]),
    np.dtype([("x", "?"), ("y", "u1")]),
    # nested: inner struct of 1-byte fields inside a packed multi-byte struct
    np.dtype([("rgb", [("r", "u1"), ("g", "u1")]), ("w", "<i4")]),
]

failures = []
for dt in CASES:
    assert not dt.isalignedstruct
    a = np.zeros(4, dt)
    a.view(np.uint8)[:] = np.arange(a.nbytes, dtype=np.uint8)
    for name, fn in [("asarray", jix.asarray), ("compact", jix.compact)]:
        try:
            got = fn(a).numpy()
            assert got.tobytes() == a.tobytes(), (name, dt)
            print(f"{name}({dt}) ok, result dtype {got.dtype} isalignedstruct={got.dtype.isalignedstruct}")
        except BaseException as e:  # PanicException derives from BaseException
            msg = " ".join(str(e).split())
            print(f"{name}({dt}) -> {type(e).__name__}: {msg[:160]}")
            failures.append((name, str(dt), type(e).__name__))

assert not failures, f"{len(failures)} failure(s): {failures}"
print("all ok")
