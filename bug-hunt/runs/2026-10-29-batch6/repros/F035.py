"""
F035 repro: jix-py rejects a padding-free numpy struct dtype whose `names` are not in offset
order (e.g. numpy's multi-field view x[['b', 'a']]) with an error that blames padding.

Asserts the CORRECT behavior, so it fails while the bug exists. Either fix passes:
  - the dtype is accepted (e.g. fields sorted by offset) and the data reads back unchanged, or
  - it is rejected with an error that does not claim the dtype has padding.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F035.py
"""

import numpy as np

import jix

x = np.zeros(3, dtype=[("a", "<i4"), ("b", "<f4")])
x["a"] = [1, 2, 3]
x["b"] = [0.5, 1.5, 2.5]

CASES = {
    "multi-field view x[['b','a']]": x[["b", "a"]],
    "explicit packed reorder": np.frombuffer(
        x.tobytes(), dtype=np.dtype({"names": ["b", "a"], "formats": ["<f4", "<i4"], "offsets": [4, 0]})
    ),
    "explicit aligned reorder": np.frombuffer(
        x.tobytes(),
        dtype=np.dtype({"names": ["b", "a"], "formats": ["<f4", "<i4"], "offsets": [4, 0]}, align=True),
    ),
}

failures = []
for name, arr in CASES.items():
    d = arr.dtype
    print(f"{name}: dtype={d} itemsize={d.itemsize} alignment={d.alignment} isalignedstruct={d.isalignedstruct}")
    # Sanity: the layout has no padding - fields tile [0, itemsize) back to back.
    spans = sorted((d.fields[n][1], d.fields[n][1] + d.fields[n][0].itemsize) for n in d.names)
    assert spans == [(0, 4), (4, 8)] and d.itemsize == 8, spans
    for fn in (jix.asarray, jix.compact):
        try:
            j = fn(arr)
            out = j.numpy()
            ok = all(np.array_equal(out[f], arr[f]) for f in arr.dtype.names)
            print(f"  {fn.__name__}: accepted, dtype={j.dtype}, fields equal: {ok}")
            if not ok:
                failures.append(f"{name}/{fn.__name__}: wrong field values")
        except ValueError as e:
            msg = str(e)
            print(f"  {fn.__name__}: ValueError: {msg}")
            if "padding" in msg:
                failures.append(f"{name}/{fn.__name__}: error blames padding the dtype does not have")

# Contrast: the core's own constructor that orders fields itself accepts the same layout
# (shown in the write-up via the Rust doc/test `from_fields_sorts_by_offset`).

print()
print("numpy: x[['b','a']].dtype == x.dtype:", x[["b", "a"]].dtype == x.dtype)
if failures:
    raise AssertionError("F035 reproduced:\n  " + "\n  ".join(failures))
print("OK")
