"""
Tests for asarray. Python-only: no core counterpart.
"""

import numpy as np
import pytest

import jix

# ---------------------------------------------------------------------------
# asarray
# ---------------------------------------------------------------------------


def test_asarray_from_list():
    a = jix.asarray([1, 2, 3])
    assert a.shape == (3,)
    np.testing.assert_array_equal(a.numpy(), [1, 2, 3])


def test_asarray_from_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = jix.asarray(arr)
    assert a.shape == (2, 2)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a.numpy(), arr)


@pytest.mark.parametrize(
    "value, dtype",
    [
        (True, np.bool_),
        (42, np.int64),
        (2.5, np.float64),
        (1 - 2j, np.complex128),
        (np.bool_(False), np.bool_),
        (np.int8(-5), np.int8),
        (np.int32(-7), np.int32),
        (np.uint8(255), np.uint8),
        (np.float32(1.5), np.float32),
        (np.complex64(1 - 2j), np.complex64),
        (np.array(3, dtype=np.int16), np.int16),
    ],
)
def test_asarray_from_scalar(value, dtype):
    a = jix.asarray(value)
    assert a.shape == ()
    assert a.dtype == dtype
    assert a.numpy()[()] == value


def test_asarray_from_jix_array_is_noop():
    arr = jix.compact([1, 2, 3], dtype=np.int32)
    assert jix.asarray(arr) is arr


def test_asarray_dtype_string():
    a = jix.asarray([1, 2, 3], dtype="float32")
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a.numpy(), [1.0, 2.0, 3.0])


def test_asarray_dtype_numpy_type():
    src = np.array([1.9, 2.1, -3.7], dtype=np.float64)
    a = jix.asarray(src, dtype=np.int32)
    assert a.dtype == np.int32
    np.testing.assert_array_equal(a.numpy(), src.astype(np.int32))


def test_asarray_dtype_to_bool():
    a = jix.asarray(np.array([0, 1, -2, 0], dtype=np.int32), dtype=np.bool_)
    assert a.dtype == np.bool_
    np.testing.assert_array_equal(a.numpy(), [False, True, True, False])


def test_asarray_dtype_none_preserves_dtype():
    a = jix.asarray(np.array([1, 2], dtype=np.int16))
    assert a.dtype == np.int16


def test_asarray_dtype_casts_existing_jix_array():
    arr = jix.asarray([1, 2, 3], dtype="float32")
    cast = jix.asarray(arr, dtype=np.int32)
    assert cast.dtype == np.int32
    np.testing.assert_array_equal(cast.numpy(), [1, 2, 3])


def test_asarray_dtype_unsupported_cast_raises():
    with pytest.raises(TypeError):
        jix.asarray(np.array([1 + 2j, 3 + 4j]), dtype="float64")


@pytest.mark.parametrize("fn", [lambda v, d: jix.asarray(v, dtype=d), lambda v, d: jix.compact(v, dtype=d), jix.cast])
def test_python_ints_are_range_checked_at_the_dtype(fn):
    # Python ints have no dtype to cast from, so like numpy they must fit the requested one.
    for value, dtype in [(300, np.uint8), (-1, np.uint8), ([0, 255, 256], np.uint8), ([[128]], np.int8)]:
        with pytest.raises(OverflowError):
            fn(value, dtype)
    assert fn(2**63, np.uint64).numpy() == 2**63
    np.testing.assert_array_equal(fn([0, 255], np.uint8).numpy(), [0, 255])
    # Typed values are cast, wrapping like numpy's astype.
    assert fn(np.int64(300), np.uint8).numpy() == 44
    # Typed scalars are cast like 0-d arrays of the same type.
    for value, dtype in [(np.float64(-1.5), np.uint8), (np.float32(1e10), np.int32)]:
        assert fn(value, dtype).numpy() == fn(np.array(value), dtype).numpy()
    with pytest.raises(TypeError):
        fn(np.complex128(1 + 2j), np.float64)
