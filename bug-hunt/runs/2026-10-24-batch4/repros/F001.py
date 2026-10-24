"""F001: Array.numpy() of an array whose dtype has an inner shape (e.g. the field
('pos', '<f4', (3,)) selected with jix.dtype_sub_field) returns wrong values.

numpy expands the sub-array descr ('<f4', (3,)) into trailing dims, so the allocated
array has ndim + inner_ndim dims, but jix-py/src/array.rs:568-596 takes one numpy stride
per array dim from the END of np_arr.strides - i.e. the inner sub-array strides.

Asserts the CORRECT behavior; every check below fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-24-batch4/repros/F001.py
"""

import numpy as np

import jix

dt = np.dtype([("id", "<u4"), ("pos", "<f4", (3,))])
a = np.zeros(4, dtype=dt)
a["id"] = [1, 2, 3, 4]
a["pos"] = np.arange(12, dtype=np.float32).reshape(4, 3)
expected = a["pos"]

failures = []


def check(name, fn, want):
    try:
        got = fn()
        ok = np.array_equal(np.asarray(got), want)
        print(f"{name}: {'OK' if ok else 'WRONG'}")
        if not ok:
            print("  got     :", np.asarray(got).reshape(-1).tolist())
            print("  expected:", np.asarray(want).reshape(-1).tolist())
            failures.append(name)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: ERROR {type(e).__name__}: {e}")
        failures.append(name)


pos_plain = jix.dtype_sub_field(jix.asarray(a), "pos")
pos_compact = jix.dtype_sub_field(jix.compact(a), "pos")
print("jix dtype:", pos_plain.dtype, "shape:", pos_plain.shape)

# 1-D, Plain (numpy) parent and compact parent
check("plain-parent numpy()", lambda: pos_plain.numpy(), expected)
check("compact-parent numpy()", lambda: pos_compact.numpy(), expected)
# (np.asarray(jix_array) is NOT a trigger: jix.Array has no __array__, so numpy wraps it in an
# object array without calling numpy() at all.)
check("__getitem__ [1:4]", lambda: pos_compact[1:4], expected[1:4])
check("plain()", lambda: pos_compact.plain().numpy(), expected)

# 2-D struct array
b = np.zeros((2, 3), dtype=dt)
b["pos"] = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
q = jix.dtype_sub_field(jix.compact(b), "pos")
check("2-D numpy()", lambda: q.numpy(), b["pos"])
check("2-D q[1, 0:2]", lambda: q[1, 0:2], b["pos"][1, 0:2])

# Permuted read layout -> numpy_empty_ordered transposes with only ndim axes
p = jix.dtype_sub_field(jix.permute_dims(jix.compact(b), [1, 0]), "pos")
check("permuted numpy()", lambda: p.numpy(), b["pos"].transpose(1, 0, 2))

# Downstream ops reading through numpy()
check("stack", lambda: jix.stack([q, q]).numpy(), np.stack([b["pos"], b["pos"]]))


# A .jix file whose array dtype has an inner shape (e.g. written from a Rust [f32; 3] element
# type, or from the view above) read back from Python
import os
import tempfile

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "pos.jix")
    jix.write_array(pos_compact, path)
    r = jix.read_array(path)
    print("read_array dtype:", r.dtype, "shape:", r.shape)
    check("read_array(file).numpy()", lambda: r.numpy(), expected)

# out= with the natural destination (4, 3) float32 can never be accepted
def with_out():
    out = np.empty((4, 3), np.float32)
    pos_compact.numpy(out=out)
    return out


check("numpy(out=(4,3) f32)", with_out, expected)

# Sanity: 0-d (all dims dropped) reads correctly - no numpy stride is consumed
check("0-d q[1, 2] (control)", lambda: q[1, 2], b["pos"][1, 2])

print()
print("FAILURES:", failures)
assert not failures, f"{len(failures)} check(s) failed: {failures}"
