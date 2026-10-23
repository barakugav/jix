"""F011: a .jix archive whose struct dtype has two fields with the same name loads without error.

The archive reader builds struct dtypes with Dtype::new_struct, which (unlike Dtype::from_fields)
never checks for duplicate field names. From Python the defect surfaces late: read_array succeeds,
then every use of the dtype (arr.dtype, arr.numpy()) fails inside numpy with a confusing error.

The script asserts the CORRECT behavior (read_array rejects the archive), so it fails while the bug
exists. Run with .venv/bin/python.
"""

import io

import numpy as np

import jix

src = np.array([(1, 2), (3, 4)], dtype=[("xa", "u1"), ("xb", "u1")])
buf = io.BytesIO()
jix.write_array(src, buf)
data = buf.getvalue()

# Rename field "xb" to "xa" in the protobuf header (same length, so no offsets move).
needle = b"\n\x02xb"
assert data.count(needle) == 1, data.count(needle)
data = data.replace(needle, b"\n\x02xa")

try:
    arr = jix.read_array(io.BytesIO(data))
except Exception as e:  # expected: the archive is rejected
    print(f"read_array rejected the archive: {type(e).__name__}: {e}")
else:
    print("read_array ACCEPTED an archive with duplicate field names")
    print("  shape:", arr.shape)
    for what, fn in [("arr.dtype", lambda: arr.dtype), ("arr.numpy()", lambda: arr.numpy())]:
        try:
            print(f"  {what} ->", fn())
        except Exception as e:
            print(f"  {what} raised {type(e).__name__}: {e}")
    raise AssertionError("archive with duplicate struct field names was accepted by read_array")
