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
