"""F025 repro: an 8-d array (NDIM_MAX) cannot be indexed with a full 8-item index plus an Ellipsis.

numpy accepts `a[(0,)*8 + (...,)]` and `a[(...,) + (slice(None),)*8]` on an 8-d array (the Ellipsis
fills 0 dims). jix's `parse_basic_index` (jix-py/src/ops/shape_ops.rs:250) rejects any tuple
longer than NDIM_MAX before discounting the Ellipsis. Asserts the CORRECT (numpy) behavior, so it
fails while the bug exists.
"""

import numpy as np

import jix

e8 = np.arange(256, dtype=np.int64).reshape((2,) * 8)
j8 = jix.compact(e8)

cases = {
    "8 ints + Ellipsis": (0,) * 8 + (Ellipsis,),
    "Ellipsis + 8 full slices": (Ellipsis,) + (slice(None),) * 8,
    "4 ints + Ellipsis + 4 ints": (1,) * 4 + (Ellipsis,) + (1,) * 4,
}

failures = []
for name, idx in cases.items():
    expected = e8[idx]
    for how, fn in [
        ("__getitem__", lambda i: j8[i]),
        ("numpy(index)", lambda i: j8.numpy(i)),
        ("jix.slice", lambda i: jix.slice(j8, i).numpy()),
    ]:
        try:
            got = np.asarray(fn(idx))
            ok = got.shape == np.shape(expected) and np.array_equal(got, expected)
            print(f"{name:28s} {how:13s} -> shape {got.shape} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                failures.append((name, how, "mismatch"))
        except Exception as e:  # noqa: BLE001
            print(f"{name:28s} {how:13s} -> {type(e).__name__}: {e} | numpy: {np.shape(expected)}")
            failures.append((name, how, repr(e)))

# Control: one fewer dim (7-d) with 7 ints + Ellipsis works.
e7 = np.arange(128).reshape((2,) * 7)
print("control 7-d, 7 ints + Ellipsis ->", np.shape(jix.compact(e7)[(0,) * 7 + (Ellipsis,)]))

assert not failures, f"{len(failures)} index forms rejected/mismatched: {failures}"
