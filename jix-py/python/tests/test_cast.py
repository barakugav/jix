"""
Tests for cast. Mirrors jix-py/src/ops/elementwise/casting.rs.
"""

import numpy as np
import pytest

import jix

# ---------------------------------------------------------------------------
# cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_dtype, dst_dtype",
    [
        (np.int32, np.float32),
        (np.float32, np.float64),
        (np.int8, np.int64),
        (np.uint8, np.int16),
        (np.float64, np.int32),
        (np.bool_, np.uint8),
        (np.int32, np.bool_),
        (np.complex64, np.complex128),
        (np.complex128, np.complex64),
    ],
)
def test_cast_scalar_dtypes(src_dtype, dst_dtype):
    data = np.array([1, 2, 3, 4], dtype=src_dtype)
    za = jix.compact(data)
    result = jix.cast(za, dst_dtype)
    assert result.dtype == np.dtype(dst_dtype)
    np.testing.assert_array_equal(result.numpy(), data.astype(dst_dtype))


@pytest.mark.parametrize(
    "src_dtype, dst_dtype",
    [(np.int8, np.uint8), (np.uint16, np.int16), (np.int32, np.uint32), (np.uint64, np.int64)],
)
def test_cast_same_size_int(src_dtype, dst_dtype):
    info = np.iinfo(src_dtype)
    data = np.array([info.min, 0, 1, info.max], dtype=src_dtype)
    result = jix.cast(jix.compact(data), dst_dtype)
    assert result.dtype == np.dtype(dst_dtype)
    np.testing.assert_array_equal(result.numpy(), data.astype(dst_dtype))


def test_cast_preserves_shape():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    za = jix.compact(arr)
    result = jix.cast(za, np.float64)
    assert result.shape == (3, 4)


def test_cast_float_to_bool():
    data = np.array([0.0, 1.0, -3.5], dtype=np.float32)
    za = jix.compact(data)
    result = jix.cast(za, np.bool_)
    np.testing.assert_array_equal(result.numpy(), [False, True, True])


# ---------------------------------------------------------------------------
# Relaxed inputs: shape ops + cast accept anything `jix.asarray` accepts
# (numpy arrays, Python lists, tuples, scalars), not just `jix.Array` instances.
# ---------------------------------------------------------------------------


def test_cast_accepts_numpy_array():
    result = jix.cast(np.array([1, 2, 3], dtype=np.int32), np.float64)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])


def test_cast_accepts_python_list():
    result = jix.cast([1, 2, 3], np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])
