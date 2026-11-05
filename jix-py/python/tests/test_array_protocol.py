"""
Tests for `jix.Array.__array__`, the hook NumPy calls from `np.asarray(a)` / `np.array(a)`.
"""

import numpy as np
import pytest

import jix


def test_asarray():
    np_a = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    for res in [np.asarray(a), np.array(a), a.__array__()]:
        assert res.dtype == np.int32
        np.testing.assert_array_equal(res, np_a)


def test_lazy_op():
    np_a = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(np.asarray(jix.compact(np_a) * 2), np_a * 2)


def test_dtype():
    np_a = np.arange(6, dtype=np.int32)
    res = np.asarray(jix.compact(np_a), dtype=np.float32)
    assert res.dtype == np.float32
    np.testing.assert_array_equal(res, np_a)


def test_copy():
    np_a = np.arange(6, dtype=np.int32)
    a = jix.compact(np_a)
    np.testing.assert_array_equal(np.array(a, copy=True), np_a)
    with pytest.raises(ValueError):
        np.asarray(a, copy=False)
