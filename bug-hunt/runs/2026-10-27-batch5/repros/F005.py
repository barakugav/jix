"""F005 repro: Array.numpy() / __getitem__ (and plain(), which calls numpy()) raise
pyo3_runtime.PanicException ('ArrayVec: capacity exceeded in extend/from_iter') when the array's
ndim plus its dtype's inner ndim exceeds NDIM_MAX = 8.

numpy() appends the dtype's inner dims to the numpy shape (jix-py/src/array.rs:545-549) and
allocates through numpy_empty_ordered / numpy_empty, which collect that shape into
DimArray = ArrayVec<_, 8> (jix-py/src/util.rs:103, 130, 133).

Asserts the CORRECT behavior (the result equals numpy's own x['a']), so it fails while the bug
exists. Each case runs in-process; a PanicException is a BaseException, caught per case so every
combination is reported, then the script fails if any case failed.

Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F005.py
"""

import numpy as np

import jix


def make(outer, inner):
    dt = np.dtype([("a", "i4", inner), ("b", "i4")])
    x = np.zeros(outer, dtype=dt)
    x["a"] = np.arange(x["a"].size, dtype=np.int32).reshape(x["a"].shape)
    f = jix.dtype_sub_field(jix.compact(x), "a")
    assert f.shape == outer, (f.shape, outer)
    return x["a"], f


CASES = [
    # (outer shape, inner shape) - outer ndim + inner ndim
    ((1, 1, 1, 1, 2), (2, 2, 2)),  # 5 + 3 = 8: control, must pass
    ((1, 1, 1, 1, 1, 2), (2, 2, 2)),  # 6 + 3 = 9
    ((1, 1, 1, 1, 1, 1, 1, 2), (2,)),  # 8 + 1 = 9
    ((1, 1, 1, 1, 1, 1, 2), (2, 2)),  # 7 + 2 = 9
    ((1, 1, 1, 1, 2), (2, 2, 2, 2)),  # 5 + 4 = 9
]

failures = []
for outer, inner in CASES:
    ref, f = make(outer, inner)
    nd = len(outer)
    perm = tuple(reversed(range(nd)))
    ref_perm = ref.transpose(perm + tuple(range(nd, ref.ndim)))
    checks = [
        ("numpy()", lambda: f.numpy(), ref),
        ("[...]", lambda: f[...], ref),
        ("permuted numpy()", lambda: jix.permute_dims(f, perm).numpy(), ref_perm),
        # numpy(out=) does not allocate, so it shows the read itself is fine.
        ("numpy(out=)", lambda: f.numpy(out=np.empty(ref.shape, np.int32)), ref),
    ]
    for name, fn, want in checks:
        try:
            got = fn()
            np.testing.assert_array_equal(got, want)
            status = "ok"
        except BaseException as e:  # noqa: BLE001 - PanicException derives from BaseException
            status = f"FAIL {type(e).__module__}.{type(e).__name__}: {e}"
            failures.append((outer, inner, name, status))
        print(f"outer ndim {nd} + inner ndim {len(inner)} = {nd + len(inner)}: {name}: {status}")

assert not failures, f"{len(failures)} case(s) failed"
print("OK")
