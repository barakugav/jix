"""
Tests for where. Mirrors the test block in jix/src/ops/elementwise/select.rs.
"""

import numpy as np
import pytest

import jix

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


def test_where_scalar_condition_returns_operand():
    x = jix.compact([1, 2, 3], dtype=np.int32)
    y = jix.compact([10, 20, 30], dtype=np.int32)
    assert jix.where(True, x, y) is x
    assert jix.where(np.False_, x, y) is y
    # broadcasting and type promotion still apply
    y = jix.compact([[10], [20]], dtype=np.int64)
    r = jix.where(True, x, y)
    assert r.dtype == np.int64
    np.testing.assert_array_equal(r.numpy(), [[1, 2, 3], [1, 2, 3]])
    with pytest.raises(TypeError):
        jix.where(1, x, y)


def test_where_same_operands_returns_operand():
    x = jix.compact([1, 2, 3], dtype=np.int32)
    assert jix.where(jix.compact([True, False, True]), x, x) is x
    with pytest.raises(TypeError):
        jix.where(jix.compact([1, 0, 1]), x, x)
