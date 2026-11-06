"""
Property tests for element-wise maximum, minimum and clamp.
Mirrors the test block in jix/src/ops/elementwise/min_max.rs.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    any_element_strategy,
    assert_array_matches,
    carray_strategy,
    carrays2_strategy,
    floats,
    ints,
    op_safe_element_strategy,
    uints,
)

import jix

_ordered_dtypes = ints + uints + floats + [np.bool_]

# clamp_min_only / clamp_max_only share identical input arrays: int32 dtype min/max
# plus values on both sides of the bound; float32 +/-inf, a value at the bound, and
# NaN passthrough (comparisons with NaN are always false).
_CLAMP_INT_VALS = np.array([np.iinfo(np.int32).min, -5, 0, 5, np.iinfo(np.int32).max], dtype=np.int32)
_CLAMP_FLOAT_VALS = np.array([float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")], dtype=np.float32)


def _max_min_st(dtype):
    """any for integers/bool, op_safe for floats (NaN in output breaks assert_allclose)."""
    if np.issubdtype(dtype, np.floating):
        return op_safe_element_strategy(dtype)
    return any_element_strategy(dtype)


# maximum / minimum: NaN propagates to float output, breaking assert_allclose.
# Use op_safe_strategy for floats to keep all outputs finite.


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_maximum(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_max_min_st(dtype)), label="arrays")
    assert_array_matches(jix.maximum(za, zb), np.maximum(np_a, np_b), data=data)


# minimum edge cases: signed negatives, equal values,
# inf ordering (finite outputs), and a NaN-propagation case matching the kernel.
def test_minimum_concrete():
    a_i = np.array([1, -5, 3, 3], dtype=np.int32)
    b_i = np.array([4, -2, 3, -3], dtype=np.int32)
    za_i, zb_i = jix.compact(a_i), jix.compact(b_i)
    assert_array_matches(jix.minimum(za_i, zb_i), np.minimum(a_i, b_i))

    # float32: inf/-inf ordering, all finite outputs (no NaN yet).
    a_f = np.array([float("-inf"), 1.0, float("inf"), -1.0], dtype=np.float32)
    b_f = np.array([5.0, -1.0, 2.0, -1.0], dtype=np.float32)
    za_f, zb_f = jix.compact(a_f), jix.compact(b_f)
    assert_array_matches(jix.minimum(za_f, zb_f), np.minimum(a_f, b_f))

    # NaN-propagation edge: minimum(NaN, x) is NaN for any x, matching np.minimum.
    a_n = np.array([float("nan"), 1.0], dtype=np.float32)
    b_n = np.array([2.0, float("nan")], dtype=np.float32)
    za_n, zb_n = jix.compact(a_n), jix.compact(b_n)
    assert_array_matches(jix.minimum(za_n, zb_n), np.minimum(a_n, b_n))


# clamp: prop tests against numpy.clip, plus explicit edge-case tests.

_clamp_dtypes = ints + uints + floats + [np.bool_]


@pytest.mark.parametrize("dtype", _clamp_dtypes)
@given(st.data())
def test_clamp_both(dtype: np.dtype, data: DataObject):
    # Unordered bounds: where lo > hi both numpy and jix return hi.
    np_a, za = data.draw(carray_strategy(dtype, element_st=_max_min_st(dtype)), label="array")
    lo = data.draw(_max_min_st(dtype), label="lo")
    hi = data.draw(_max_min_st(dtype), label="hi")
    if dtype == np.uint64:  # a Python int above int64 raises
        lo, hi = np.uint64(lo), np.uint64(hi)
    assert_array_matches(jix.clamp(za, min=lo, max=hi), np.clip(np_a, lo, hi), data=data)


# clamp_min_only / clamp_max_only / clamp_none use np.clip as the reference
# (with the corresponding bound set to None).
def test_clamp_min_only_concrete():
    # int32: dtype min/max, 0, values on both sides of the bound.
    a_i = _CLAMP_INT_VALS
    za_i = jix.compact(a_i)
    assert_array_matches(jix.clamp(za_i, min=0), np.clip(a_i, 0, None))

    # float32: +/-inf, NaN passthrough (comparisons with NaN are always false),
    # and a value exactly at the bound.
    a_f = _CLAMP_FLOAT_VALS
    za_f = jix.compact(a_f)
    assert_array_matches(jix.clamp(za_f, min=0.0), np.clip(a_f, 0.0, None))


def test_clamp_max_only_concrete():
    # int32: dtype min/max, 0, values on both sides of the bound.
    a_i = _CLAMP_INT_VALS
    za_i = jix.compact(a_i)
    assert_array_matches(jix.clamp(za_i, max=0), np.clip(a_i, None, 0))

    # float32: +/-inf, NaN passthrough, and a value exactly at the bound.
    a_f = _CLAMP_FLOAT_VALS
    za_f = jix.compact(a_f)
    assert_array_matches(jix.clamp(za_f, max=0.0), np.clip(a_f, None, 0.0))


def test_clamp_none_concrete():
    # No bounds set: clamp is a passthrough regardless of dtype or edge values.
    a_i = np.array([np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max], dtype=np.int32)
    za_i = jix.compact(a_i)
    assert_array_matches(jix.clamp(za_i), a_i)

    a_f = np.array([float("-inf"), 0.0, float("inf"), float("nan")], dtype=np.float32)
    za_f = jix.compact(a_f)
    assert_array_matches(jix.clamp(za_f), a_f)


@pytest.mark.parametrize("lo, hi", [(5, 2), (2, 2), (-1, 10)])
def test_clamp_min_greater_than_max_gives_max(lo, hi):
    a = np.array([1, 2, 3], dtype=np.int32)
    assert_array_matches(jix.clamp(jix.compact(a), min=lo, max=hi), np.clip(a, lo, hi))


@pytest.mark.parametrize("lo, hi", [(np.nan, 1.0), (0.0, np.nan), (np.nan, np.nan), (np.nan, None), (None, np.nan)])
def test_clamp_nan_bound_propagates(lo, hi):
    a = np.array([1.0, 2.0], dtype=np.float64)
    assert_array_matches(jix.clamp(jix.compact(a), min=lo, max=hi), np.clip(a, lo, hi))


def test_clamp_array_bounds_broadcast():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    lo = np.array([[1.0], [4.0]], dtype=np.float32)
    hi = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    expected = np.clip(a, lo, hi)
    assert_array_matches(jix.clamp(jix.compact(a), min=jix.compact(lo), max=hi), expected)
    assert_array_matches(jix.compact(a).clamp(min=lo, max=jix.compact(hi)), expected)


def test_clamp_promotes_like_maximum():
    a = np.array([1, 2, 5, 9], dtype=np.int32)
    result = jix.clamp(jix.compact(a), min=0.5, max=7.7)
    assert result.dtype == np.float64
    assert_array_matches(result, np.clip(a, 0.5, 7.7))


def test_clamp_complex_limit_raises():
    a = jix.compact([1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError):
        jix.clamp(a, min=complex(1, 0), max=2.0)


def test_clamp_equal_bounds():
    a = jix.compact([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = jix.clamp(a, min=0.5, max=0.5).numpy()
    assert np.all(result == 0.5)


def test_clamp_inf_max():
    a = jix.compact([0.0, 1e10, float("inf")], dtype=np.float64)
    result = jix.clamp(a, min=0.0, max=float("inf")).numpy()
    assert np.array_equal(result, [0.0, 1e10, float("inf")])


def test_clamp_neg_inf_min():
    a = jix.compact([float("-inf"), -1.0, 0.0], dtype=np.float64)
    result = jix.clamp(a, min=float("-inf"), max=0.0).numpy()
    assert np.array_equal(result, [float("-inf"), -1.0, 0.0])


def test_clamp_method():
    a = jix.compact([0, 5, 10], dtype=np.int32)
    assert np.array_equal(a.clamp(min=2, max=8).numpy(), [2, 5, 8])
