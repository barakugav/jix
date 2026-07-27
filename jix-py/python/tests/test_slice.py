"""
Tests for `jix.slice`: a lazy view-producing counterpart to `__getitem__` / `.numpy()`.

Slicing here returns a new `jix.Array` rather than materializing to numpy. The
parsing logic for the index (handled by `parse_basic_index` in shape_ops.rs) is
shared with `.numpy()`, so most of the index-syntax surface is also exercised by
the existing `test_array_indexing` block in test_array_ops.py / test_archive.py.
This file focuses on the slice op itself: laziness, view semantics, axis
dropping, and the strict-bounds errors introduced in the core `resolve` and the
Python-side index parser.
"""

import numpy as np
import pytest

import jix

# ---------------------------------------------------------------------------
# Array.slice() method form (delegates to jix.slice).
# ---------------------------------------------------------------------------


def test_array_slice_method_basic():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    result = a.slice((slice(0, 2), slice(1, 3)))
    assert isinstance(result, jix.Array)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), np_a[0:2, 1:3])


def test_array_slice_method_int_index():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    result = a.slice(1)
    assert result.shape == (4,)
    np.testing.assert_array_equal(result.numpy(), np_a[1])


def test_array_slice_method_matches_free_function():
    np_a = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    a = jix.compact(np_a)
    idx = (slice(0, 2), 2, slice(1, 4))
    np.testing.assert_array_equal(a.slice(idx).numpy(), jix.slice(a, idx).numpy())


# ---------------------------------------------------------------------------
# Relaxed input: slice accepts anything asarray() accepts (list, ndarray, scalar).
# ---------------------------------------------------------------------------


def test_slice_accepts_numpy_array():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = jix.slice(np_a, (slice(0, 2), slice(1, 3)))
    assert isinstance(result, jix.Array)
    np.testing.assert_array_equal(result.numpy(), np_a[0:2, 1:3])


def test_slice_accepts_python_list():
    result = jix.slice([[1, 2, 3], [4, 5, 6]], (0, slice(1, 3)))
    np.testing.assert_array_equal(result.numpy(), [2, 3])


# ---------------------------------------------------------------------------
# Return type and laziness
# ---------------------------------------------------------------------------


def test_slice_returns_jix_array():
    """slice() must return a jix.Array, not a numpy array (that's what numpy()/[] does)."""
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, (slice(0, 2), slice(1, 3)))
    assert isinstance(result, jix.Array)


def test_slice_preserves_dtype():
    a = jix.compact(np.arange(12, dtype=np.float32).reshape(3, 4))
    result = jix.slice(a, slice(1, 3))
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Shape: slice keeps the axis; integer index drops the axis.
# ---------------------------------------------------------------------------


def test_slice_keeps_axis():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, (slice(0, 2), slice(1, 3)))
    assert result.shape == (2, 2)


def test_slice_int_drops_axis():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, 1)
    assert result.shape == (4,)


def test_slice_int_on_each_axis_yields_scalar():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, (1, 2))
    assert result.shape == ()


def test_slice_mixed_int_and_slice():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, (1, slice(1, 3)))
    assert result.shape == (2,)


def test_slice_ellipsis_only_is_identity():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, ...)
    assert result.shape == (3, 4)


def test_slice_ellipsis_fills_missing_axes():
    """`(..., i)` on a 3-D array applies i to the last axis only."""
    a = jix.compact(np.arange(24, dtype=np.int32).reshape(2, 3, 4))
    result = jix.slice(a, (..., 2))
    assert result.shape == (2, 3)


def test_slice_implicit_full_range_on_missing_axes():
    """Fewer index items than ndim: remaining axes get implicit full slices."""
    a = jix.compact(np.arange(24, dtype=np.int32).reshape(2, 3, 4))
    result = jix.slice(a, 1)  # only axis 0 indexed; axes 1, 2 implicitly full
    assert result.shape == (3, 4)


def test_slice_empty_slice_produces_length_zero_axis():
    """`arr[2:2]` is a valid empty slice."""
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, slice(2, 2))
    assert result.shape == (0, 4)


def test_slice_negative_indices_resolve_against_axis_length():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    result = jix.slice(a, slice(-2, None))
    assert result.shape == (2, 4)


# ---------------------------------------------------------------------------
# Values: numpy()-on-slice must equal __getitem__ on the source array.
# ---------------------------------------------------------------------------


def test_slice_values_match_getitem_basic():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    result = jix.slice(a, (slice(0, 2), slice(1, 3)))
    np.testing.assert_array_equal(result.numpy(), np_a[0:2, 1:3])


def test_slice_values_match_getitem_int_axis_dropped():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    result = jix.slice(a, 1)
    np.testing.assert_array_equal(result.numpy(), np_a[1])


def test_slice_values_match_getitem_ellipsis_int():
    np_a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    a = jix.compact(np_a)
    result = jix.slice(a, (..., 2))
    np.testing.assert_array_equal(result.numpy(), np_a[..., 2])


def test_slice_values_match_getitem_negative():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    result = jix.slice(a, slice(-2, None))
    np.testing.assert_array_equal(result.numpy(), np_a[-2:])


def test_slice_values_match_getitem_3d_mixed():
    np_a = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    a = jix.compact(np_a)
    result = jix.slice(a, (slice(1, 3), 2, slice(0, 3)))
    np.testing.assert_array_equal(result.numpy(), np_a[1:3, 2, 0:3])


# ---------------------------------------------------------------------------
# Composition with other lazy ops.
# ---------------------------------------------------------------------------


def test_slice_composes_with_op():
    """A slice view can feed into an element-wise op without materializing."""
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    sub = jix.slice(a, (slice(0, 2), slice(1, 3)))  # shape (2, 2)
    result = jix.add(sub, np.int32(100))
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), np_a[0:2, 1:3] + 100)


def test_slice_then_slice():
    np_a = np.arange(20, dtype=np.int32).reshape(4, 5)
    a = jix.compact(np_a)
    first = jix.slice(a, slice(1, 4))  # shape (3, 5)
    second = jix.slice(first, (slice(0, 2), slice(1, 4)))  # shape (2, 3)
    assert second.shape == (2, 3)
    np.testing.assert_array_equal(second.numpy(), np_a[1:4][0:2, 1:4])


# ---------------------------------------------------------------------------
# Error cases.
# ---------------------------------------------------------------------------


def test_slice_int_out_of_bounds_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, 5)


def test_slice_negative_int_out_of_bounds_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, -4)  # axis size 3, -4 normalizes to -1 (oob)


def test_slice_start_out_of_bounds_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, slice(10, 12))


def test_slice_stop_out_of_bounds_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, slice(0, 10))


def test_slice_start_greater_than_stop_raises():
    """Strict bounds: reversed slices (post-normalization) are rejected."""
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, slice(2, 1))


def test_slice_step_not_one_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(ValueError):
        jix.slice(a, slice(0, 3, 2))


def test_slice_too_many_indices_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, (1, 2, 3))  # array is 2-D


def test_slice_multiple_ellipsis_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError):
        jix.slice(a, (..., ...))


def test_slice_unsupported_index_type_raises():
    a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(TypeError):
        jix.slice(a, "not a valid index")


# ---------------------------------------------------------------------------
# Equivalence with __getitem__ for parametrized cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index",
    [
        slice(0, 2),
        slice(1, None),
        slice(None, 2),
        slice(None, None),
        slice(-2, None),
        slice(None, -1),
        1,
        -1,
        ...,
        (slice(0, 2), slice(1, 3)),
        (1, slice(1, 3)),
        (slice(0, 2), 2),
        (..., 1),
        (0, ...),
    ],
)
def test_slice_matches_getitem(index):
    """For any index that both forms accept, the values must match exactly."""
    np_a = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    a = jix.compact(np_a)
    np.testing.assert_array_equal(
        jix.slice(a, index).numpy(),
        np_a[index],
    )
