"""
Property tests for element-wise comparison ops.
Mirrors the test block in jix/src/ops/cmp.rs.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    any_element_strategy,
    assert_array_matches,
    carrays2_strategy,
    carray_strategy,
    comparable_element_strategy,
    complexes,
    floats,
    ints,
    maybe_non_finite_element_strategy,
    op_safe_element_strategy,
    uints,
)

import jix

_int_bool_dtypes = ints + uints + [np.bool_]
_all_cmp_dtypes = ints + uints + floats + complexes + [np.bool_]
_ordered_dtypes = ints + uints + floats + [np.bool_]


def _ordering_st(dtype):
    """any for integers/bool, maybe_non_finite for floats (exercises NaN -> False paths)."""
    if np.issubdtype(dtype, np.floating):
        return maybe_non_finite_element_strategy(dtype)
    return any_element_strategy(dtype)


def _max_min_st(dtype):
    """any for integers/bool, op_safe for floats (NaN in output breaks assert_allclose)."""
    if np.issubdtype(dtype, np.floating):
        return op_safe_element_strategy(dtype)
    return any_element_strategy(dtype)


# equal / not_equal: comparable_strategy gives ~33 % equal pairs and exercises NaN != NaN.
# Output is bool, so NaN float inputs are safe for assert_array_matches.


@pytest.mark.parametrize("dtype", _all_cmp_dtypes)
@given(st.data())
def test_equal(dtype: np.dtype, data: DataObject):
    cmp_st = comparable_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=cmp_st), label="arrays")
    assert_array_matches(jix.equal(za, zb), np_a == np_b, data=data)


@pytest.mark.parametrize("dtype", _all_cmp_dtypes)
@given(st.data())
def test_not_equal(dtype: np.dtype, data: DataObject):
    cmp_st = comparable_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=cmp_st), label="arrays")
    assert_array_matches(jix.not_equal(za, zb), np_a != np_b, data=data)


# Ordering ops: integers use any_strategy; floats use maybe_non_finite to cover NaN -> False paths.


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_greater(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_ordering_st(dtype)), label="arrays")
    assert_array_matches(jix.greater(za, zb), np_a > np_b, data=data)


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_greater_equal(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_ordering_st(dtype)), label="arrays")
    assert_array_matches(jix.greater_equal(za, zb), np_a >= np_b, data=data)


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_less(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_ordering_st(dtype)), label="arrays")
    assert_array_matches(jix.less(za, zb), np_a < np_b, data=data)


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_less_equal(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_ordering_st(dtype)), label="arrays")
    assert_array_matches(jix.less_equal(za, zb), np_a <= np_b, data=data)


# maximum / minimum: NaN propagates to float output, breaking assert_allclose.
# Use op_safe_strategy for floats to keep all outputs finite.


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_maximum(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_max_min_st(dtype)), label="arrays")
    assert_array_matches(jix.maximum(za, zb), np.maximum(np_a, np_b), data=data)


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_minimum(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_max_min_st(dtype)), label="arrays")
    assert_array_matches(jix.minimum(za, zb), np.minimum(np_a, np_b), data=data)


# clamp: prop tests against numpy.clip, plus explicit edge-case tests.

_clamp_dtypes = ints + uints + floats


@pytest.mark.parametrize("dtype", _clamp_dtypes)
@given(st.data())
def test_clamp_both(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype, element_st=_max_min_st(dtype)), label="array")
    lo, hi = sorted([data.draw(_max_min_st(dtype), label="lo"), data.draw(_max_min_st(dtype), label="hi")])
    assert_array_matches(jix.clamp(za, min=lo, max=hi), np.clip(np_a, lo, hi), data=data)


@pytest.mark.parametrize("dtype", _clamp_dtypes)
@given(st.data())
def test_clamp_min_only(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype, element_st=_max_min_st(dtype)), label="array")
    lo = data.draw(_max_min_st(dtype), label="lo")
    assert_array_matches(jix.clamp(za, min=lo), np.clip(np_a, lo, None), data=data)


@pytest.mark.parametrize("dtype", _clamp_dtypes)
@given(st.data())
def test_clamp_max_only(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype, element_st=_max_min_st(dtype)), label="array")
    hi = data.draw(_max_min_st(dtype), label="hi")
    assert_array_matches(jix.clamp(za, max=hi), np.clip(np_a, None, hi), data=data)


@pytest.mark.parametrize("dtype", _clamp_dtypes)
@given(st.data())
def test_clamp_none(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype, element_st=_max_min_st(dtype)), label="array")
    assert_array_matches(jix.clamp(za), np_a, data=data)


# --- invalid limit tests (non-property) ---


def test_clamp_min_greater_than_max_raises():
    a = jix.compact([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="min"):
        jix.clamp(a, min=5, max=2).numpy()


def test_clamp_nan_min_raises():
    a = jix.compact([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        jix.clamp(a, min=float("nan"), max=1.0).numpy()


def test_clamp_nan_max_raises():
    a = jix.compact([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        jix.clamp(a, min=0.0, max=float("nan")).numpy()


def test_clamp_nan_both_raises():
    a = jix.compact([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        jix.clamp(a, min=float("nan"), max=float("nan")).numpy()


def test_clamp_complex_limit_raises():
    a = jix.compact([1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError):
        jix.clamp(a, min=complex(1, 0)).numpy()


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
