"""
Use-case tests for zix.dtype_sub_field.
"""

import numpy as np
import pytest

import zix


def _struct_1d():
    """Return a 1-D numpy structured array and its zix counterpart."""
    dt = np.dtype([("x", np.int32), ("y", np.int32)])
    pts = np.array([(1, 10), (2, 20), (3, 30)], dtype=dt)
    return pts, zix.asarray(pts)


def test_extract_first_field():
    pts, za = _struct_1d()
    xs = zix.dtype_sub_field(za, "x")
    assert xs.dtype == np.int32
    assert list(xs.shape) == [3]
    np.testing.assert_array_equal(xs.numpy(), [1, 2, 3])


def test_extract_second_field():
    pts, za = _struct_1d()
    ys = zix.dtype_sub_field(za, "y")
    assert ys.dtype == np.int32
    np.testing.assert_array_equal(ys.numpy(), [10, 20, 30])


def test_mixed_field_dtypes():
    """Fields with different scalar dtypes are each extracted correctly."""
    dt = np.dtype([("a", np.int16), ("b", np.float64)])
    data = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dt)
    za = zix.asarray(data)

    a = zix.dtype_sub_field(za, "a")
    b = zix.dtype_sub_field(za, "b")

    assert a.dtype == np.int16
    assert b.dtype == np.float64
    np.testing.assert_array_equal(a.numpy(), [1, 2, 3])
    np.testing.assert_allclose(b.numpy(), [1.5, 2.5, 3.5])


def test_shape_preserved_2d():
    """Outer shape is unchanged; only the element dtype changes."""
    dt = np.dtype([("x", np.int32), ("y", np.int32)])
    pts2d = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=dt)
    za = zix.asarray(pts2d)

    xs = zix.dtype_sub_field(za, "x")
    assert list(xs.shape) == [2, 2]
    np.testing.assert_array_equal(xs.numpy(), [[1, 3], [5, 7]])


def test_compact_array_input():
    """dtype_sub_field also works when the input is a zix compact array."""
    dt = np.dtype([("x", np.int32), ("y", np.int32)])
    pts = np.array([(7, 8), (9, 10)], dtype=dt)
    za = zix.compact(pts)

    xs = zix.dtype_sub_field(za, "x")
    np.testing.assert_array_equal(xs.numpy(), [7, 9])


def test_numpy_array_input():
    """dtype_sub_field accepts a raw numpy structured array (implicit asarray)."""
    dt = np.dtype([("x", np.int32), ("y", np.int32)])
    pts = np.array([(4, 5)], dtype=dt)

    xs = zix.dtype_sub_field(pts, "x")
    np.testing.assert_array_equal(xs.numpy(), [4])


def test_error_non_struct_dtype():
    """Raises ValueError when the array has a plain (non-struct) dtype."""
    za = zix.compact(np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(Exception):
        zix.dtype_sub_field(za, "x")


def test_error_field_not_found():
    """Raises ValueError when the field name does not exist in the struct."""
    dt = np.dtype([("x", np.int32), ("y", np.int32)])
    pts = np.array([(1, 2)], dtype=dt)
    za = zix.asarray(pts)
    with pytest.raises(Exception):
        zix.dtype_sub_field(za, "z")
