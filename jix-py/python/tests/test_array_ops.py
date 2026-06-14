"""Tests for shape/axis manipulation, compact, astype, asarray, concatenate, stack, where,
flatten, reshape, broadcast, permute_axes, squeeze/unsqueeze, insert_axis/remove_axis,
read_array/write_array."""

import tempfile
from pathlib import Path

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


def test_asarray_from_scalar():
    a = jix.asarray(42)
    assert a.numpy()[()] == 42


def test_asarray_from_jix_array_is_noop():
    arr = jix.compact([1, 2, 3], dtype=np.int32)
    a = jix.asarray(arr)
    np.testing.assert_array_equal(a.numpy(), [1, 2, 3])


# ---------------------------------------------------------------------------
# astype
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
def test_astype_scalar_casts(src_dtype, dst_dtype):
    data = np.array([1, 2, 3, 4], dtype=src_dtype)
    za = jix.compact(data)
    result = jix.astype(za, dst_dtype)
    assert result.dtype == np.dtype(dst_dtype)
    np.testing.assert_array_equal(result.numpy(), data.astype(dst_dtype))


def test_astype_preserves_shape():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    za = jix.compact(arr)
    result = jix.astype(za, np.float64)
    assert result.shape == (3, 4)


def test_astype_float_to_bool():
    data = np.array([0.0, 1.0, -3.5], dtype=np.float32)
    za = jix.compact(data)
    result = jix.astype(za, np.bool_)
    np.testing.assert_array_equal(result.numpy(), [False, True, True])


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def test_compact_produces_equal_array():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    za = jix.compact(arr)
    copied = jix.compact(za)
    assert copied.shape == za.shape
    assert copied.dtype == za.dtype
    np.testing.assert_array_equal(copied.numpy(), arr)


def test_compact_is_independent():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    copied = jix.compact(za)
    np.testing.assert_array_equal(copied.numpy(), za.numpy())


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def test_flatten_2d():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    za = jix.compact(arr)
    flat = jix.flatten(za)
    assert flat.shape == (6,)
    np.testing.assert_array_equal(flat.numpy(), arr.flatten())


def test_flatten_3d():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    za = jix.compact(arr)
    flat = jix.flatten(za)
    assert flat.shape == (24,)
    np.testing.assert_array_equal(flat.numpy(), arr.flatten())


def test_flatten_1d_noop():
    arr = np.array([10, 20, 30], dtype=np.int32)
    za = jix.compact(arr)
    flat = jix.flatten(za)
    assert flat.shape == (3,)
    np.testing.assert_array_equal(flat.numpy(), arr)


def test_flatten_lazy():
    arr = np.arange(6, dtype=np.int32)
    za = jix.compact(arr)
    flat = jix.flatten(za, copy=False)
    assert flat.shape == (6,)
    np.testing.assert_array_equal(flat.numpy(), arr)


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------


def test_reshape_1d_to_2d():
    arr = np.arange(6, dtype=np.int32)
    za = jix.compact(arr)
    r = jix.reshape(za, [2, 3])
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), arr.reshape(2, 3))


def test_reshape_2d_to_1d():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    za = jix.compact(arr)
    r = jix.reshape(za, [12])
    assert r.shape == (12,)
    np.testing.assert_array_equal(r.numpy(), arr.flatten())


def test_reshape_lazy():
    arr = np.arange(6, dtype=np.int32)
    za = jix.compact(arr)
    r = jix.reshape(za, [3, 2], copy=False)
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), arr.reshape(3, 2))


def test_reshape_wrong_size_raises():
    arr = np.arange(6, dtype=np.int32)
    za = jix.compact(arr)
    with pytest.raises(Exception):
        jix.reshape(za, [2, 4])


# ---------------------------------------------------------------------------
# insert_axis / remove_axis / squeeze / unsqueeze
# ---------------------------------------------------------------------------


def test_insert_axis_front():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.insert_axis(za, 0)
    assert r.shape == (1, 3)


def test_insert_axis_back():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.insert_axis(za, 1)
    assert r.shape == (3, 1)


def test_insert_axis_multiple():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    # gap 0 = before dim0, gap 2 = after dim1 -> (1, 2, 3, 1)
    r = jix.insert_axis(za, [0, 2])
    assert r.shape == (1, 2, 3, 1)


def test_remove_axis():
    arr = np.array([[1, 2, 3]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.remove_axis(za, 0)
    assert r.shape == (3,)
    np.testing.assert_array_equal(r.numpy(), [1, 2, 3])


def test_remove_axis_non_one_raises():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    za = jix.compact(arr)
    with pytest.raises(Exception):
        jix.remove_axis(za, 0)


def test_squeeze_all():
    arr = np.array([[[42]]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.squeeze(za)
    assert r.shape == ()
    assert r.numpy()[()] == 42


def test_squeeze_specific_axis():
    arr = np.zeros((2, 1, 3), dtype=np.float32)
    za = jix.compact(arr)
    r = jix.squeeze(za, axis=1)
    assert r.shape == (2, 3)


def test_squeeze_no_size_one_dims_is_noop():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    r = jix.squeeze(za)
    assert r.shape == (2, 3)


def test_unsqueeze_single():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.unsqueeze(za, 0)
    assert r.shape == (1, 3)


def test_unsqueeze_same_as_insert_axis():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    r1 = jix.unsqueeze(za, 1)
    r2 = jix.insert_axis(za, 1)
    assert r1.shape == r2.shape
    np.testing.assert_array_equal(r1.numpy(), r2.numpy())


# ---------------------------------------------------------------------------
# permute_axes
# ---------------------------------------------------------------------------


def test_permute_axes_2d_transpose():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.permute_axes(za, [1, 0])
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), arr.T)


def test_permute_axes_3d():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    za = jix.compact(arr)
    r = jix.permute_axes(za, [2, 0, 1])
    assert r.shape == (4, 2, 3)
    np.testing.assert_array_equal(r.numpy(), np.transpose(arr, [2, 0, 1]))


def test_permute_axes_identity():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    r = jix.permute_axes(za, [0, 1])
    np.testing.assert_array_equal(r.numpy(), arr)


def test_permute_axes_none_reverses():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    za = jix.compact(arr)
    r = jix.permute_axes(za)
    assert r.shape == (4, 3, 2)
    np.testing.assert_array_equal(r.numpy(), arr.T)


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------


def test_broadcast_expand_dim():
    arr = np.array([[1], [2], [3]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.broadcast(za, [3, 4])
    assert r.shape == (3, 4)
    np.testing.assert_array_equal(r.numpy(), np.broadcast_to(arr, (3, 4)))


def test_broadcast_scalar_to_shape():
    arr = np.array([[5]], dtype=np.float32)
    za = jix.compact(arr)
    r = jix.broadcast(za, [2, 3])
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), np.full((2, 3), 5.0, dtype=np.float32))


def test_broadcast_identity():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    r = jix.broadcast(za, [2, 3])
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), arr)


def test_broadcast_non_one_dim_raises():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    za = jix.compact(arr)
    with pytest.raises(Exception):
        jix.broadcast(za, [3, 2])


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------


def test_concatenate_axis0():
    a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
    b = jix.compact([[5, 6]], dtype=np.int32)
    r = jix.concatenate([a, b], axis=0)
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2], [3, 4], [5, 6]]))


def test_concatenate_axis1():
    a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
    b = jix.compact([[5], [6]], dtype=np.int32)
    r = jix.concatenate([a, b], axis=1)
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2, 5], [3, 4, 6]]))


def test_concatenate_negative_axis():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    b = jix.compact([4, 5], dtype=np.int32)
    r = jix.concatenate([a, b], axis=-1)
    assert r.shape == (5,)
    np.testing.assert_array_equal(r.numpy(), [1, 2, 3, 4, 5])


def test_concatenate_three_arrays():
    arrays = [jix.compact([i, i + 1], dtype=np.float32) for i in range(3)]
    r = jix.concatenate(arrays, axis=0)
    assert r.shape == (6,)


def test_concatenate_default_axis():
    a = jix.compact([1, 2], dtype=np.int32)
    b = jix.compact([3, 4], dtype=np.int32)
    r = jix.concatenate([a, b])
    np.testing.assert_array_equal(r.numpy(), [1, 2, 3, 4])


def test_concatenate_dtype_mismatch_raises():
    a = jix.compact([1, 2], dtype=np.int32)
    b = jix.compact([3.0, 4.0], dtype=np.float32)
    with pytest.raises(Exception):
        jix.concatenate([a, b])


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------


def test_stack_axis0():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    b = jix.compact([4, 5, 6], dtype=np.int32)
    r = jix.stack([a, b], axis=0)
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2, 3], [4, 5, 6]]))


def test_stack_axis1():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    b = jix.compact([4, 5, 6], dtype=np.int32)
    r = jix.stack([a, b], axis=1)
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 4], [2, 5], [3, 6]]))


def test_stack_default_axis():
    a = jix.compact([1, 2], dtype=np.int32)
    b = jix.compact([3, 4], dtype=np.int32)
    r = jix.stack([a, b])
    assert r.shape == (2, 2)


def test_stack_2d_arrays():
    a = jix.compact(np.zeros((2, 3), dtype=np.float32))
    b = jix.compact(np.ones((2, 3), dtype=np.float32))
    r = jix.stack([a, b], axis=0)
    assert r.shape == (2, 2, 3)


def test_stack_shape_mismatch_raises():
    a = jix.compact([1, 2], dtype=np.int32)
    b = jix.compact([3, 4, 5], dtype=np.int32)
    with pytest.raises(Exception):
        jix.stack([a, b])


def test_stack_dtype_mismatch_raises():
    a = jix.compact([1, 2], dtype=np.int32)
    b = jix.compact([3.0, 4.0], dtype=np.float32)
    with pytest.raises(Exception):
        jix.stack([a, b])


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------


def test_where_basic():
    cond = jix.compact([True, False, True, False], dtype=bool)
    x = jix.compact([1, 2, 3, 4], dtype=np.int32)
    y = jix.compact([10, 20, 30, 40], dtype=np.int32)
    r = jix.where(cond, x, y)
    np.testing.assert_array_equal(r.numpy(), [1, 20, 3, 40])


def test_where_float():
    cond = jix.compact([True, False, True], dtype=bool)
    x = jix.compact([1.0, 2.0, 3.0], dtype=np.float32)
    y = jix.compact([0.1, 0.2, 0.3], dtype=np.float32)
    r = jix.where(cond, x, y)
    np.testing.assert_allclose(r.numpy(), [1.0, 0.2, 3.0])


def test_where_matches_numpy():
    rng = np.random.default_rng(0)
    cond = rng.integers(0, 2, size=10).astype(bool)
    x = rng.integers(-10, 10, size=10).astype(np.int32)
    y = rng.integers(-10, 10, size=10).astype(np.int32)
    r = jix.where(jix.compact(cond), jix.compact(x), jix.compact(y))
    np.testing.assert_array_equal(r.numpy(), np.where(cond, x, y))


def test_where_2d():
    cond = np.array([[True, False], [False, True]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[10, 20], [30, 40]], dtype=np.int32)
    r = jix.where(jix.compact(cond), jix.compact(x), jix.compact(y))
    np.testing.assert_array_equal(r.numpy(), np.where(cond, x, y))


# ---------------------------------------------------------------------------
# read_array / write_array
# ---------------------------------------------------------------------------


def test_write_and_read_array_roundtrip():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path)
        assert loaded.shape == za.shape
        assert loaded.dtype == za.dtype
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_write_and_read_integer_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path)
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_write_and_read_bool_array():
    arr = np.array([True, False, True, True], dtype=bool)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path)
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_read_array_mmap():
    arr = np.arange(50, dtype=np.float64)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path, mmap=True)
        np.testing.assert_array_equal(loaded.numpy(), arr)


# ---------------------------------------------------------------------------
# Relaxed inputs: shape ops + astype now accept anything `jix.asarray` accepts
# (numpy arrays, Python lists, tuples, scalars), not just `jix.Array` instances.
# ---------------------------------------------------------------------------


def test_astype_accepts_numpy_array():
    result = jix.astype(np.array([1, 2, 3], dtype=np.int32), np.float64)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])


def test_astype_accepts_python_list():
    result = jix.astype([1, 2, 3], np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])


def test_reshape_accepts_numpy_array():
    np_a = np.arange(6, dtype=np.int32)
    result = jix.reshape(np_a, [2, 3])
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result.numpy(), np_a.reshape(2, 3))


def test_reshape_accepts_python_list():
    result = jix.reshape([1, 2, 3, 4], [2, 2])
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), [[1, 2], [3, 4]])


def test_flatten_accepts_numpy_array():
    np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    result = jix.flatten(np_a)
    assert result.shape == (6,)
    np.testing.assert_array_equal(result.numpy(), [1, 2, 3, 4, 5, 6])


def test_broadcast_accepts_numpy_array():
    np_a = np.array([[1, 2, 3]], dtype=np.int32)  # shape (1, 3)
    result = jix.broadcast(np_a, [2, 3])
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result.numpy(), [[1, 2, 3], [1, 2, 3]])


def test_permute_axes_accepts_numpy_array():
    np_a = np.arange(6, dtype=np.int32).reshape(2, 3)
    result = jix.permute_axes(np_a, [1, 0])
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result.numpy(), np_a.T)


def test_squeeze_accepts_numpy_array():
    np_a = np.array([[[1, 2, 3]]], dtype=np.int32)  # shape (1, 1, 3)
    result = jix.squeeze(np_a)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result.numpy(), [1, 2, 3])


def test_insert_axis_accepts_python_list():
    result = jix.insert_axis([1, 2, 3], 0)
    assert result.shape == (1, 3)


def test_remove_axis_accepts_numpy_array():
    np_a = np.array([[[1, 2, 3]]], dtype=np.int32)  # shape (1, 1, 3)
    result = jix.remove_axis(np_a, 0)
    assert result.shape == (1, 3)
