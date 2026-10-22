"""F008 repro: parse_basic_index rejects slice start == dim length.

Asserts the CORRECT behavior (empty slices at the end of a dim, and any slice on a
zero-length dim, are valid and yield length-0 dims), so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F008.py
"""

import traceback

import numpy as np

import jix

np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
a = jix.compact(np_a)
np_z = np.zeros((0, 3), np.int32)
z = jix.compact(np_z)

cases = [
    # (label, thunk producing a numpy array, numpy reference)
    ("a[2:2] (control, start < len)", lambda: a[2:2], np_a[2:2]),
    ("z[...] (control)", lambda: z[...], np_z[...]),
    ("z.numpy() (control)", lambda: z.numpy(), np_z),
    ("a[3:3]", lambda: a[3:3], np_a[3:3]),
    ("a[3:]", lambda: a[3:], np_a[3:]),
    ("a[3:3, 1:2]", lambda: a[3:3, 1:2], np_a[3:3, 1:2]),
    ("a[:, 4:]", lambda: a[:, 4:], np_a[:, 4:]),
    ("a.numpy(slice(3, None))", lambda: a.numpy(slice(3, None)), np_a[3:]),
    ("jix.slice(a, slice(3, None))", lambda: jix.slice(a, slice(3, None)).numpy(), np_a[3:]),
    ("z[:]", lambda: z[:], np_z[:]),
    ("z[0:0]", lambda: z[0:0], np_z[0:0]),
    ("z[:, 0]", lambda: z[:, 0], np_z[:, 0]),
    ("z.numpy((slice(None), 0))", lambda: z.numpy((slice(None), 0)), np_z[:, 0]),
    ("jix.slice(z, slice(None))", lambda: jix.slice(z, slice(None)).numpy(), np_z[:]),
    ("jix.asarray(z_np)[:]", lambda: jix.asarray(np_z)[:], np_z[:]),
]

failures = []
for label, thunk, ref in cases:
    try:
        got = thunk()
        ok = got.shape == ref.shape and np.array_equal(got, ref)
        print(f"{label:45s} OK={ok} shape={got.shape} expected={ref.shape}")
        if not ok:
            failures.append(label)
    except Exception as e:  # noqa: BLE001
        print(f"{label:45s} RAISED {type(e).__name__}: {e}")
        failures.append(label)

if failures:
    print(f"\nFAIL: {len(failures)} case(s) wrong: {failures}")
    raise SystemExit(1)
print("\nPASS")
