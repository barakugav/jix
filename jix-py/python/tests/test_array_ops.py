"""Tests for shape/dim manipulation, compact, cast, asarray, concatenate, stack, where,
flatten, reshape, broadcast, permute_dims, squeeze/unsqueeze, insert_dim/remove_dim,
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
# plain
# ---------------------------------------------------------------------------


def test_plain_materializes_to_equal_array():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    za = jix.compact(arr)
    # Materializing returns an equal array.
    plain = za.plain()
    assert plain.shape == za.shape
    assert plain.dtype == za.dtype
    np.testing.assert_array_equal(plain.numpy(), arr)
    # It also materializes a lazy view.
    np.testing.assert_array_equal((za + 1).plain().numpy(), arr + 1)


# ---------------------------------------------------------------------------
# item
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, expected_type",
    [
        (np.array([[1, 2], [3, 4]], dtype=np.int32), int),
        (np.array([1.5, 2.25], dtype=np.float64), float),
        (np.array([1 + 2j, 3 - 1j], dtype=np.complex64), complex),
    ],
)
def test_item_of_full_reduction(data, expected_type):
    result = jix.sum(jix.compact(data)).item()
    assert type(result) is expected_type
    assert result == data.sum().item()


def test_item_bool():
    assert jix.all(jix.compact(np.array([True, False]))).item() is False


@pytest.mark.parametrize("shape", [(1,), (2, 3)])
def test_item_requires_zero_dim(shape):
    with pytest.raises(ValueError, match="zero-dimensional"):
        jix.compact(np.zeros(shape)).item()


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
    flat = jix.flatten(za)
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
    r = jix.reshape(za, [3, 2])
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), arr.reshape(3, 2))


def test_reshape_wrong_size_raises():
    arr = np.arange(6, dtype=np.int32)
    za = jix.compact(arr)
    with pytest.raises(Exception):
        jix.reshape(za, [2, 4])


# ---------------------------------------------------------------------------
# insert_dim / remove_dim / squeeze / unsqueeze
# ---------------------------------------------------------------------------


def test_insert_dim_front():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.insert_dim(za, 0)
    assert r.shape == (1, 3)
    np.testing.assert_array_equal(r.numpy(), np.expand_dims(arr, 0))


def test_insert_dim_back():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    # numpy expand_dims: dim indexes the output shape; 1 is the last dim of a 2-d result.
    r = jix.insert_dim(za, 1)
    assert r.shape == (3, 1)
    np.testing.assert_array_equal(r.numpy(), np.expand_dims(arr, 1))


def test_insert_dim_negative():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    # -1 refers to the last position of the (larger) output shape, like numpy.
    r = jix.insert_dim(za, -1)
    assert r.shape == (3, 1)
    np.testing.assert_array_equal(r.numpy(), np.expand_dims(arr, -1))


def test_insert_dim_multiple():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    # numpy expand_dims semantics: dims index the output shape -> (1, 2, 1, 3)
    r = jix.insert_dim(za, [0, 2])
    assert r.shape == (1, 2, 1, 3)
    np.testing.assert_array_equal(r.numpy(), np.expand_dims(arr, (0, 2)))


def test_insert_dim_repeated_dim_raises():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    # like numpy.expand_dims, repeated output dims are rejected
    with pytest.raises(Exception):
        jix.insert_dim(za, [0, 0])


def test_insert_dim_out_of_range_raises():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    # output ndim is 2 for a single insertion; dim 2 is out of range (valid -2..1)
    with pytest.raises(Exception):
        jix.insert_dim(za, 2)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((3,), 0),
        ((3,), 1),
        ((3,), -1),
        ((3,), -2),
        ((2, 3), [0, 2]),
        ((2, 3), [0, 1]),
        ((2, 3), -1),
        ((2, 3, 4), [0, 2, 5]),
        ((2, 3, 4), [-1, -3]),
    ],
)
def test_insert_dim_matches_numpy_expand_dims(shape, dim):
    arr = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    za = jix.compact(arr)
    np_dim = tuple(dim) if isinstance(dim, list) else dim
    expected = np.expand_dims(arr, np_dim)
    got = jix.insert_dim(za, dim).numpy()
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_remove_dim():
    arr = np.array([[1, 2, 3]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.remove_dim(za, 0)
    assert r.shape == (3,)
    np.testing.assert_array_equal(r.numpy(), [1, 2, 3])


def test_remove_dim_non_one_raises():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    za = jix.compact(arr)
    with pytest.raises(Exception):
        jix.remove_dim(za, 0)


def test_squeeze_all():
    arr = np.array([[[42]]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.squeeze(za)
    assert r.shape == ()
    assert r.numpy()[()] == 42


def test_squeeze_specific_dim():
    arr = np.zeros((2, 1, 3), dtype=np.float32)
    za = jix.compact(arr)
    r = jix.squeeze(za, dim=1)
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


def test_unsqueeze_same_as_insert_dim():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    r1 = jix.unsqueeze(za, [0, 2])
    r2 = jix.insert_dim(za, [0, 2])
    assert r1.shape == r2.shape
    np.testing.assert_array_equal(r1.numpy(), r2.numpy())


def test_unsqueeze_method_matches_numpy_expand_dims():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    # the Array.unsqueeze method uses the same output-world semantics
    r = za.unsqueeze([0, 2])
    np.testing.assert_array_equal(r.numpy(), np.expand_dims(arr, (0, 2)))


# ---------------------------------------------------------------------------
# permute_dims
# ---------------------------------------------------------------------------


def test_permute_dims_2d_transpose():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    za = jix.compact(arr)
    r = jix.permute_dims(za, [1, 0])
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), arr.T)


def test_permute_dims_3d():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    za = jix.compact(arr)
    r = jix.permute_dims(za, [2, 0, 1])
    assert r.shape == (4, 2, 3)
    np.testing.assert_array_equal(r.numpy(), np.transpose(arr, [2, 0, 1]))


def test_permute_dims_identity():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    za = jix.compact(arr)
    r = jix.permute_dims(za, [0, 1])
    np.testing.assert_array_equal(r.numpy(), arr)


def test_permute_dims_none_reverses():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    za = jix.compact(arr)
    r = jix.permute_dims(za)
    assert r.shape == (4, 3, 2)
    np.testing.assert_array_equal(r.numpy(), arr.T)


@pytest.mark.parametrize("dims", [[-1, 0, 1], [2, -3, -2], [-1, -2, -3], [0, -2, 2]])
def test_permute_dims_negative(dims):
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    za = jix.compact(arr)
    r = jix.permute_dims(za, dims)
    np.testing.assert_array_equal(r.numpy(), np.transpose(arr, dims))
    np.testing.assert_array_equal(za.permute_dims(dims).numpy(), np.transpose(arr, dims))


@pytest.mark.parametrize("dims", [[0, -2], [1, -1]])
def test_permute_dims_negative_duplicate_raises(dims):
    # -2 resolves to 0 and -1 to 1 for a 2-D array, so these repeat a dim.
    with pytest.raises(RuntimeError, match="duplicate dim"):
        jix.permute_dims(jix.compact(np.zeros((2, 3))), dims)


def test_permute_dims_out_of_bounds_raises():
    with pytest.raises(ValueError, match="out of bounds"):
        jix.permute_dims(jix.compact(np.zeros((2, 3))), [0, -3])


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim0, dim1", [(0, 1), (1, 0), (0, 2), (1, 2), (-1, -2), (-3, 1), (1, 1)])
def test_transpose_swaps_two_dims(dim0, dim1):
    arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    za = jix.compact(arr)
    expected = np.swapaxes(arr, dim0, dim1)
    np.testing.assert_array_equal(jix.transpose(za, dim0, dim1).numpy(), expected)
    np.testing.assert_array_equal(za.transpose(dim0, dim1).numpy(), expected)


def test_transpose_2d_matches_t():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    za = jix.compact(arr)
    np.testing.assert_array_equal(jix.transpose(za, 0, 1).numpy(), za.T.numpy())


def test_transpose_out_of_bounds_raises():
    with pytest.raises(ValueError, match="out of bounds"):
        jix.transpose(jix.compact(np.zeros((2, 3))), 0, 2)


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


def test_concatenate_dim0():
    a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
    b = jix.compact([[5, 6]], dtype=np.int32)
    r = jix.concatenate([a, b], dim=0)
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2], [3, 4], [5, 6]]))


def test_concatenate_dim1():
    a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
    b = jix.compact([[5], [6]], dtype=np.int32)
    r = jix.concatenate([a, b], dim=1)
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2, 5], [3, 4, 6]]))


def test_concatenate_negative_dim():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    b = jix.compact([4, 5], dtype=np.int32)
    r = jix.concatenate([a, b], dim=-1)
    assert r.shape == (5,)
    np.testing.assert_array_equal(r.numpy(), [1, 2, 3, 4, 5])


def test_concatenate_three_arrays():
    arrays = [jix.compact([i, i + 1], dtype=np.float32) for i in range(3)]
    r = jix.concatenate(arrays, dim=0)
    assert r.shape == (6,)


def test_concatenate_default_dim():
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


def test_stack_dim0():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    b = jix.compact([4, 5, 6], dtype=np.int32)
    r = jix.stack([a, b], dim=0)
    assert r.shape == (2, 3)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2, 3], [4, 5, 6]]))


def test_stack_dim1():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    b = jix.compact([4, 5, 6], dtype=np.int32)
    r = jix.stack([a, b], dim=1)
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 4], [2, 5], [3, 6]]))


def test_stack_default_dim():
    a = jix.compact([1, 2], dtype=np.int32)
    b = jix.compact([3, 4], dtype=np.int32)
    r = jix.stack([a, b])
    assert r.shape == (2, 2)


def test_stack_2d_arrays():
    a = jix.compact(np.zeros((2, 3), dtype=np.float32))
    b = jix.compact(np.ones((2, 3), dtype=np.float32))
    r = jix.stack([a, b], dim=0)
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


def test_permute_dims_accepts_numpy_array():
    np_a = np.arange(6, dtype=np.int32).reshape(2, 3)
    result = jix.permute_dims(np_a, [1, 0])
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result.numpy(), np_a.T)


def test_squeeze_accepts_numpy_array():
    np_a = np.array([[[1, 2, 3]]], dtype=np.int32)  # shape (1, 1, 3)
    result = jix.squeeze(np_a)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result.numpy(), [1, 2, 3])


def test_insert_dim_accepts_python_list():
    result = jix.insert_dim([1, 2, 3], 0)
    assert result.shape == (1, 3)


def test_remove_dim_accepts_numpy_array():
    np_a = np.array([[[1, 2, 3]]], dtype=np.int32)  # shape (1, 1, 3)
    result = jix.remove_dim(np_a, 0)
    assert result.shape == (1, 3)


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------


def test_repeat_1d():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    r = jix.repeat(a, 2, dim=0)
    assert r.shape == (6,)
    np.testing.assert_array_equal(r.numpy(), [1, 1, 2, 2, 3, 3])


def test_repeat_2d_dim0():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.repeat(a, 3, dim=0)
    assert r.shape == (6, 2)
    np.testing.assert_array_equal(r.numpy(), np.repeat(arr, 3, axis=0))


def test_repeat_2d_dim1():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.repeat(a, 3, dim=1)
    assert r.shape == (2, 6)
    np.testing.assert_array_equal(r.numpy(), np.repeat(arr, 3, axis=1))


def test_repeat_negative_dim():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.repeat(a, 2, dim=-1)
    assert r.shape == (2, 4)
    np.testing.assert_array_equal(r.numpy(), np.repeat(arr, 2, axis=-1))


def test_repeat_identity():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    a = jix.compact(arr)
    r = jix.repeat(a, 1, dim=0)
    np.testing.assert_array_equal(r.numpy(), arr)


def test_repeat_zero_yields_empty():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    r = jix.repeat(a, 0, dim=0)
    assert r.shape == (0,)


def test_repeat_3d_middle_dim():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a = jix.compact(arr)
    r = jix.repeat(a, 2, dim=1)
    assert r.shape == (2, 6, 4)
    np.testing.assert_array_equal(r.numpy(), np.repeat(arr, 2, axis=1))


def test_repeat_dim_out_of_bounds_raises():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    with pytest.raises(Exception):
        jix.repeat(a, 2, dim=1)


def test_repeat_method_on_array():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    a = jix.compact(arr)
    r = a.repeat(2, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.repeat(arr, 2, axis=0))


def test_repeat_accepts_python_list():
    r = jix.repeat([1, 2, 3], 2, dim=0)
    np.testing.assert_array_equal(r.numpy(), [1, 1, 2, 2, 3, 3])


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------


def test_flip_1d():
    a = jix.compact([1, 2, 3, 4], dtype=np.int32)
    np.testing.assert_array_equal(jix.flip(a).numpy(), [4, 3, 2, 1])


def test_flip_dim_int():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.flip(a, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.flip(arr, axis=0))


def test_flip_dim_list():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.flip(a, dim=[1])
    np.testing.assert_array_equal(r.numpy(), np.flip(arr, axis=1))


def test_flip_negative_dim():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.flip(a, dim=-1)
    np.testing.assert_array_equal(r.numpy(), np.flip(arr, axis=-1))


def test_flip_dim_none_reverses_all():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.flip(a)
    np.testing.assert_array_equal(r.numpy(), np.flip(arr))


def test_flip_multiple_dims():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a = jix.compact(arr)
    r = jix.flip(a, dim=[0, 2])
    np.testing.assert_array_equal(r.numpy(), np.flip(arr, axis=(0, 2)))


def test_flip_duplicate_dim_raises():
    a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(Exception):
        jix.flip(a, dim=[0, 0])


def test_flip_out_of_bounds_dim_raises():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    with pytest.raises(Exception):
        jix.flip(a, dim=1)


def test_flip_method_on_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = a.flip(dim=0)
    np.testing.assert_array_equal(r.numpy(), np.flip(arr, axis=0))


def test_flip_accepts_python_list():
    r = jix.flip([1, 2, 3], dim=0)
    np.testing.assert_array_equal(r.numpy(), [3, 2, 1])


# ---------------------------------------------------------------------------
# roll
# ---------------------------------------------------------------------------


def test_roll_1d_positive():
    a = jix.compact([0, 1, 2, 3, 4], dtype=np.int32)
    np.testing.assert_array_equal(jix.roll(a, 2).numpy(), [3, 4, 0, 1, 2])


def test_roll_1d_negative():
    a = jix.compact([0, 1, 2, 3, 4], dtype=np.int32)
    np.testing.assert_array_equal(jix.roll(a, -1).numpy(), [1, 2, 3, 4, 0])


def test_roll_dim_int():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(arr)
    r = jix.roll(a, 1, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.roll(arr, 1, axis=0))


def test_roll_negative_dim():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(arr)
    r = jix.roll(a, 1, dim=-1)
    np.testing.assert_array_equal(r.numpy(), np.roll(arr, 1, axis=-1))


def test_roll_dim_none_on_non_1d_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(Exception):
        jix.roll(a, 1)


def test_roll_out_of_bounds_dim_raises():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    with pytest.raises(Exception):
        jix.roll(a, 1, dim=1)


def test_roll_method_on_array():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(arr)
    r = a.roll(1, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.roll(arr, 1, axis=0))


def test_roll_accepts_python_list():
    r = jix.roll([1, 2, 3, 4], 1)
    np.testing.assert_array_equal(r.numpy(), [4, 1, 2, 3])


def test_roll_large_shift_wraps():
    arr = np.arange(5, dtype=np.int32)
    a = jix.compact(arr)
    r = jix.roll(a, 12, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.roll(arr, 12, axis=0))


# ---------------------------------------------------------------------------
# tile
# ---------------------------------------------------------------------------


def test_tile_1d():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(jix.tile(a, 2).numpy(), [1, 2, 3, 1, 2, 3])


def test_tile_dim0():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.tile(a, 3, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.tile(arr, (3, 1)))


def test_tile_dim1():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.tile(a, 2, dim=1)
    np.testing.assert_array_equal(r.numpy(), np.tile(arr, (1, 2)))


def test_tile_negative_dim():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.tile(a, 2, dim=-1)
    np.testing.assert_array_equal(r.numpy(), np.tile(arr, (1, 2)))


def test_tile_dim_none_on_non_1d_raises():
    a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(Exception):
        jix.tile(a, 2)


def test_tile_out_of_bounds_dim_raises():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    with pytest.raises(Exception):
        jix.tile(a, 2, dim=1)


def test_tile_reps_zero_yields_empty():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    r = jix.tile(a, 0)
    assert r.numpy().shape == (0,)


def test_tile_reps_one_is_identity():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = jix.tile(a, 1, dim=0)
    np.testing.assert_array_equal(r.numpy(), arr)


def test_tile_method_on_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = jix.compact(arr)
    r = a.tile(2, dim=0)
    np.testing.assert_array_equal(r.numpy(), np.tile(arr, (2, 1)))


def test_tile_accepts_python_list():
    r = jix.tile([1, 2, 3], 2)
    np.testing.assert_array_equal(r.numpy(), [1, 2, 3, 1, 2, 3])


def test_tile_large_reps():
    arr = np.arange(4, dtype=np.int32)
    a = jix.compact(arr)
    r = jix.tile(a, 7)
    np.testing.assert_array_equal(r.numpy(), np.tile(arr, 7))
