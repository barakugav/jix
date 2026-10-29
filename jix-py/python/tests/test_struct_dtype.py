"""
Tests for numpy structured dtypes: which layouts jix accepts, and that accepted ones round trip.

jix does not support padding bytes in struct dtypes (they are uninitialized, and jix reads elements
as raw bytes). A layout that needs padding is accepted only when the padding is an explicit field.
"""

import io
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import jix

SCALAR_FORMATS = ["?", "<i1", "<i2", "<i4", "<i8", "<u1", "<u2", "<u4", "<u8", "<f2", "<f4", "<f8", "<c8", "<c16"]

ACCEPTED_DTYPES = [
    # packed (numpy's default, align=False)
    np.dtype([("a", "<u1"), ("b", "<f4"), ("c", "<i2")]),
    np.dtype([("x", "<f8")]),
    # aligned, with no padding needed
    np.dtype([("x", "<i4"), ("y", "<i4")], align=True),
    np.dtype([("x", "<f8"), ("y", "<i4"), ("z", "<u2"), ("w", "<u1"), ("v", "<u1")], align=True),
    # aligned, with the padding as explicit fields
    np.dtype([("a", "<u1"), ("_pad", "(3,)u1"), ("b", "<f4")], align=True),
    np.dtype([("a", "<f4"), ("b", "<u1"), ("_pad", "(3,)u1")], align=True),
    np.dtype([("a", "<u1"), ("_pad0", "<u1"), ("b", "<i2"), ("_pad1", "(4,)u1"), ("c", "<f8")], align=True),
    # sub-array fields
    np.dtype([("pos", "<f4", (3,)), ("id", "<u4")], align=True),
    np.dtype([("m", "<i2", (2, 3))]),
    # nested structs
    np.dtype([("rgb", [("r", "u1"), ("g", "u1"), ("b", "u1")]), ("w", "<i4")]),
    np.dtype(
        [("p", np.dtype([("x", "<f4"), ("y", "<f4")], align=True)), ("w", "<f4"), ("_pad", "(4,)u1"), ("t", "<f8")],
        align=True,
    ),
]

PADDED_DTYPES = [
    # numpy inserts padding between fields
    np.dtype([("a", "<u1"), ("b", "<f4")], align=True),
    np.dtype([("a", "<u1"), ("b", "<i2"), ("c", "<f8")], align=True),
    # trailing padding only
    np.dtype([("a", "<f4"), ("b", "<u1")], align=True),
    np.dtype([("a", "<f8"), ("b", "<i4")], align=True),
    # explicit offsets with a gap
    np.dtype({"names": ["a", "b"], "formats": ["<u1", "<u1"], "offsets": [0, 2]}),
    # explicit itemsize larger than the fields
    np.dtype({"names": ["a", "b"], "formats": ["<u1", "<u1"], "offsets": [0, 1], "itemsize": 4}),
    # overlapping fields
    np.dtype({"names": ["a", "b"], "formats": ["<i4", "<i4"], "offsets": [0, 2], "itemsize": 6}),
    # padding inside a nested struct
    np.dtype([("inner", np.dtype([("a", "<u1"), ("b", "<i4")], align=True)), ("c", "<u8")]),
    # padding inside a struct used as a sub-array field
    np.dtype([("arr", np.dtype([("a", "<i2"), ("b", "<u1")], align=True), (2,))]),
]


def _filled(dtype, shape=(5, 3)):
    """An array of `dtype` whose bytes are all distinct-ish, so any shuffling of bytes shows up."""
    a = np.zeros(shape, dtype)
    a.view(np.uint8).reshape(-1)[:] = np.arange(a.nbytes, dtype=np.uint64).astype(np.uint8)
    # Bools must hold 0 or 1.
    _fix_bools(a)
    return a


def _fix_bools(a):
    for name in a.dtype.names or ():
        field = a[name]
        if field.dtype.names is not None:
            _fix_bools(field)
        elif field.dtype == np.bool_:
            a[name] = np.arange(field.size).reshape(field.shape) % 2 == 1


def _assert_same(jix_arr, np_arr):
    assert jix_arr.dtype == np_arr.dtype
    out = jix_arr.numpy()
    assert out.dtype == np_arr.dtype
    assert out.shape == np_arr.shape
    assert out.tobytes() == np_arr.tobytes()


@pytest.mark.parametrize("dtype", ACCEPTED_DTYPES, ids=str)
@pytest.mark.parametrize("fn", [jix.asarray, jix.compact])
def test_accepted_struct_round_trips(dtype, fn):
    a = _filled(dtype)
    _assert_same(fn(a), a)


@pytest.mark.parametrize("dtype", ACCEPTED_DTYPES, ids=str)
def test_accepted_struct_archive_round_trips(dtype):
    a = _filled(dtype)
    buf = io.BytesIO()
    jix.write_array(jix.compact(a), buf)
    buf.seek(0)
    _assert_same(jix.read_array(buf), a)


@pytest.mark.parametrize("dtype", ACCEPTED_DTYPES, ids=str)
def test_accepted_struct_slice(dtype):
    a = _filled(dtype, shape=(7, 4))
    out = jix.compact(a)[1:6, 1:3]
    assert out.dtype == a.dtype
    assert out.tobytes() == a[1:6, 1:3].tobytes()


def test_aligned_flag_is_preserved():
    aligned = np.dtype([("a", "<u1"), ("_pad", "(3,)u1"), ("b", "<f4")], align=True)
    packed = np.dtype([("a", "<u1"), ("_pad", "(3,)u1"), ("b", "<f4")])
    assert jix.compact(_filled(aligned)).dtype.isalignedstruct
    assert not jix.compact(_filled(packed)).dtype.isalignedstruct
    assert jix.compact(_filled(aligned)).dtype.alignment == 4
    assert jix.compact(_filled(packed)).dtype.alignment == 1


@pytest.mark.parametrize("dtype", PADDED_DTYPES, ids=str)
@pytest.mark.parametrize("fn", [jix.asarray, jix.compact])
def test_padded_struct_is_rejected(dtype, fn):
    a = np.zeros(4, dtype)
    with pytest.raises(ValueError, match="padding|offset"):
        fn(a)


def test_padded_struct_error_suggests_explicit_padding():
    a = np.zeros(4, np.dtype([("a", "<u1"), ("b", "<f4")], align=True))
    with pytest.raises(ValueError, match="explicit padding field"):
        jix.asarray(a)


def test_sub_field_of_struct_with_explicit_padding():
    dt = np.dtype([("id", "<u1"), ("_pad", "(3,)u1"), ("grade", "<f4")], align=True)
    a = np.zeros(4, dt)
    a["id"] = [1, 2, 3, 4]
    a["grade"] = [0.5, 1.5, 2.5, 3.5]
    ja = jix.compact(a)
    np.testing.assert_array_equal(jix.dtype_sub_field(ja, "id").numpy(), a["id"])
    np.testing.assert_array_equal(jix.dtype_sub_field(ja, "grade").numpy(), a["grade"])


def test_sub_array_field_reads_as_trailing_dims():
    # Like numpy, the inner shape of the dtype comes back as trailing dims of the base dtype.
    a = np.zeros((2, 4), np.dtype([("id", "<u4"), ("pos", "<f4", (3,))]))
    a["pos"] = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    pos = jix.dtype_sub_field(jix.compact(a), "pos")
    assert pos.numpy().dtype == np.float32
    np.testing.assert_array_equal(pos.numpy(), a["pos"])
    np.testing.assert_array_equal(pos[1, 1:3], a["pos"][1, 1:3])
    pos_t = jix.dtype_sub_field(jix.permute_dims(jix.compact(a), [1, 0]), "pos")
    np.testing.assert_array_equal(pos_t.numpy(), a["pos"].transpose(1, 0, 2))

    out = np.zeros((2, 4, 3), np.float32)
    assert pos.numpy(out=out) is out
    np.testing.assert_array_equal(out, a["pos"])
    with pytest.raises(ValueError, match="inner dim"):
        pos.numpy(out=np.zeros((2, 4, 3), np.float32, order="F"))


def test_sub_array_field_plain_keeps_the_dtype():
    dt = np.dtype([("a", "<i4", (3,)), ("b", "<i4")])
    a = np.zeros((2, 2), dt)
    a["a"] = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    f = jix.dtype_sub_field(jix.compact(a), "a")
    p = f.plain()
    assert (p.shape, p.dtype) == (f.shape, f.dtype)
    np.testing.assert_array_equal(p.numpy(), a["a"])
    p = f.plain(1)
    assert (p.shape, p.dtype) == ((2,), f.dtype)
    np.testing.assert_array_equal(p.numpy(), a["a"][1])
    # 8 array dims, the most jix supports, plus the inner dim.
    f = jix.dtype_sub_field(jix.compact(np.zeros((1,) * 7 + (2,), dt)), "a")
    assert f.plain().shape == f.shape


def test_sub_array_field_plain_of_non_c_order_read():
    # numpy() allocates in the storage's read order, so the read of a permuted view is not
    # C-contiguous, while each inner sub-array still is.
    dt = np.dtype([("a", "<i4", (2, 3)), ("b", "<i4")])
    a = np.zeros((3, 4), dt)
    a["a"] = np.arange(72, dtype=np.int32).reshape(3, 4, 2, 3)
    f = jix.permute_dims(jix.dtype_sub_field(jix.compact(a), "a"), (1, 0))
    assert not f.numpy().flags.c_contiguous
    p = f.plain()
    assert (p.shape, p.dtype) == (f.shape, f.dtype)
    np.testing.assert_array_equal(p.numpy(), a["a"].transpose(1, 0, 2, 3))
    np.testing.assert_array_equal(f.plain((slice(1, 3), 2)).numpy(), a["a"][2, 1:3])


def test_numpy_dtype_import_hook_can_use_the_array():
    # The numpy dtype of an aligned struct imports numpy, running any Python __import__ hook. A
    # subprocess, so that a deadlock fails on the timeout.
    code = """
import builtins, numpy as np, jix
arr = jix.compact(np.zeros(4, np.dtype([("a", "i4"), ("b", "i4")], align=True)))
orig_import = builtins.__import__
def hook(name, *args, **kwargs):
    if name == "numpy":
        builtins.__import__ = orig_import
        arr[0]
    return orig_import(name, *args, **kwargs)
builtins.__import__ = hook
assert arr.dtype.isalignedstruct
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_sub_array_field_more_than_8_numpy_dims():
    a = np.zeros((1, 1, 1, 1, 1, 2), np.dtype([("a", "<i4", (2, 2, 2)), ("b", "<i4")]))
    a["a"] = np.arange(16, dtype=np.int32).reshape(a["a"].shape)
    f = jix.dtype_sub_field(jix.compact(a), "a")
    np.testing.assert_array_equal(f.numpy(), a["a"])
    np.testing.assert_array_equal(f[...], a["a"])
    perm = (5, 4, 3, 2, 1, 0)
    np.testing.assert_array_equal(jix.permute_dims(f, perm).numpy(), a["a"].transpose(*perm, 6, 7, 8))


@st.composite
def _packed_struct_dtype(draw):
    n = draw(st.integers(1, 6))
    formats = draw(st.lists(st.sampled_from(SCALAR_FORMATS), min_size=n, max_size=n))
    shapes = draw(st.lists(st.sampled_from([(), (2,), (1, 3)]), min_size=n, max_size=n))
    return np.dtype([(f"f{i}", fmt, shape) for i, (fmt, shape) in enumerate(zip(formats, shapes))])


@given(dtype=_packed_struct_dtype(), n=st.integers(0, 20))
def test_packed_struct_property_round_trips(dtype, n):
    a = _filled(dtype, shape=(n,))
    _assert_same(jix.compact(a), a)


@given(dtype=_packed_struct_dtype())
def test_aligned_struct_property_accepted_iff_no_padding(dtype):
    aligned = np.dtype([(name, dtype.fields[name][0]) for name in dtype.names], align=True)
    has_padding = aligned.itemsize != sum(aligned.fields[name][0].itemsize for name in aligned.names)
    a = _filled(aligned, shape=(3,))
    if has_padding:
        with pytest.raises(ValueError):
            jix.compact(a)
    else:
        _assert_same(jix.compact(a), a)
