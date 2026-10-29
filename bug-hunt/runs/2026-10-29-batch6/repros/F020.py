"""F020: a numpy struct dtype nested 33+ levels deep is written by jix.write_array without error,
but jix.read_array cannot read the file back (prost decode recursion limit, 100 message levels;
each struct level costs 3: Dtype -> DtypeStruct -> Field -> Dtype).

Asserts the CORRECT behavior: either write_array rejects the dtype, or the array round-trips.
Fails (AssertionError) while the bug exists.
"""

import io

import numpy as np

import jix


def nested(depth):
    d = np.dtype("u1")
    for _ in range(depth):
        d = np.dtype([("a", d)])
    return d


def roundtrip(depth):
    x = np.arange(1, 5, dtype=np.uint8).view(nested(depth))
    b = io.BytesIO()
    try:
        jix.write_array(x, b)
    except Exception as e:  # rejecting up front is acceptable
        return f"write rejected: {type(e).__name__}: {str(e)[:100]}"
    b.seek(0)
    try:
        y = jix.read_array(b)
    except Exception as e:
        msg = str(e)
        raise AssertionError(
            f"depth {depth}: write_array succeeded ({len(b.getvalue())} bytes) but read_array failed: "
            f"{type(e).__name__}: {msg[:60]} ... {msg[-60:]}"
        ) from None
    y = y.numpy()
    assert y.dtype == x.dtype, (y.dtype, x.dtype)
    assert y.tobytes() == x.tobytes()
    return "roundtrip ok"


for depth in (32, 33, 50, 200):
    try:
        print(depth, roundtrip(depth))
    except AssertionError as e:
        print("FAIL", e)

print(32, roundtrip(32))
roundtrip(33)  # raises AssertionError while the bug exists
