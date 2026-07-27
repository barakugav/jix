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
    carray_strategy,
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

# greater_equal / less_equal share identical float32 operands: NaN comparisons are
# always false, inf compared to itself is true (matters for >=/<=), and one operand
# sits below the other's bound.
_GE_LE_FLOAT_A = np.array([float("nan"), 1.0, float("inf"), -1.0], dtype=np.float32)
_GE_LE_FLOAT_B = np.array([1.0, float("nan"), float("inf"), 1.0], dtype=np.float32)

# clamp_min_only / clamp_max_only share identical input arrays: int32 dtype min/max
# plus values on both sides of the bound; float32 +/-inf, a value at the bound, and
# NaN passthrough (comparisons with NaN are always false).
_CLAMP_INT_VALS = np.array([np.iinfo(np.int32).min, -5, 0, 5, np.iinfo(np.int32).max], dtype=np.int32)
_CLAMP_FLOAT_VALS = np.array([float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")], dtype=np.float32)


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


# not_equal edge cases: NaN != NaN is true (IEEE 754)
# while equal ints/infinities are false, plus signed negatives.
def test_not_equal_concrete():
    a_i = np.array([1, -5, 3, 0], dtype=np.int32)
    b_i = np.array([1, 5, 3, 0], dtype=np.int32)
    za_i, zb_i = jix.compact(a_i), jix.compact(b_i)
    assert_array_matches(jix.not_equal(za_i, zb_i), a_i != b_i)

    # float32: NaN != NaN is true; same infinities are equal (so not_equal is false).
    a_f = np.array([float("nan"), float("inf"), 1.0, -1.0], dtype=np.float32)
    b_f = np.array([float("nan"), float("inf"), 1.0, 1.0], dtype=np.float32)
    za_f, zb_f = jix.compact(a_f), jix.compact(b_f)
    assert_array_matches(jix.not_equal(za_f, zb_f), a_f != b_f)


# Ordering ops: integers use any_strategy; floats use maybe_non_finite to cover NaN -> False paths.


@pytest.mark.parametrize("dtype", _ordered_dtypes)
@given(st.data())
def test_greater(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=_ordering_st(dtype)), label="arrays")
    assert_array_matches(jix.greater(za, zb), np_a > np_b, data=data)


# greater_equal / less / less_equal edge cases per op:
# NaN operands (always false), inf/-inf ordering, equal values (matters for >= and <=),
# signed negatives, and a bool path since these ops support bool.
def test_greater_equal_concrete():
    a_i = np.array([3, -1, 2, 2], dtype=np.int32)
    b_i = np.array([1, 1, 2, 3], dtype=np.int32)
    za_i, zb_i = jix.compact(a_i), jix.compact(b_i)
    assert_array_matches(jix.greater_equal(za_i, zb_i), a_i >= b_i)

    # float32: NaN comparisons are false; inf ordering and inf >= inf (true).
    a_f, b_f = _GE_LE_FLOAT_A, _GE_LE_FLOAT_B
    za_f, zb_f = jix.compact(a_f), jix.compact(b_f)
    assert_array_matches(jix.greater_equal(za_f, zb_f), a_f >= b_f)

    # bool: true>=false, false>=false, true>=true, false>=true.
    a_b = np.array([True, False, True, False])
    b_b = np.array([False, False, True, True])
    za_b, zb_b = jix.compact(a_b), jix.compact(b_b)
    assert_array_matches(jix.greater_equal(za_b, zb_b), a_b >= b_b)


def test_less_concrete():
    a_i = np.array([1, 1, 3, -5], dtype=np.int32)
    b_i = np.array([3, 1, 1, -1], dtype=np.int32)
    za_i, zb_i = jix.compact(a_i), jix.compact(b_i)
    assert_array_matches(jix.less(za_i, zb_i), a_i < b_i)

    # float32: NaN comparisons are false; -inf < finite, inf < inf is false.
    a_f = np.array([float("nan"), 1.0, float("-inf"), float("inf")], dtype=np.float32)
    b_f = np.array([1.0, float("nan"), 1.0, float("inf")], dtype=np.float32)
    za_f, zb_f = jix.compact(a_f), jix.compact(b_f)
    assert_array_matches(jix.less(za_f, zb_f), a_f < b_f)

    # bool: false<true (true), true<false (false), equal values are false.
    a_b = np.array([False, True, True, False])
    b_b = np.array([True, False, True, False])
    za_b, zb_b = jix.compact(a_b), jix.compact(b_b)
    assert_array_matches(jix.less(za_b, zb_b), a_b < b_b)


def test_less_equal_concrete():
    a_i = np.array([1, 2, 2, 5], dtype=np.int32)
    b_i = np.array([2, 2, 1, 5], dtype=np.int32)
    za_i, zb_i = jix.compact(a_i), jix.compact(b_i)
    assert_array_matches(jix.less_equal(za_i, zb_i), a_i <= b_i)

    # float32: NaN <= x is false; inf ordering.
    a_f, b_f = _GE_LE_FLOAT_A, _GE_LE_FLOAT_B
    za_f, zb_f = jix.compact(a_f), jix.compact(b_f)
    assert_array_matches(jix.less_equal(za_f, zb_f), a_f <= b_f)

    # bool on a 2x2 shape with a non-default 1x1 block shape, so a block-boundary
    # bug in the ordering kernel would still show up.
    a_b = np.array([[False, True], [True, False]])
    b_b = np.array([[True, False], [True, True]])
    za_b = jix.compact(a_b, params={"block_shape": [1, 1]})
    zb_b = jix.compact(b_b, params={"block_shape": [1, 1]})
    assert_array_matches(jix.less_equal(za_b, zb_b), a_b <= b_b)


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

_clamp_dtypes = ints + uints + floats


@pytest.mark.parametrize("dtype", _clamp_dtypes)
@given(st.data())
def test_clamp_both(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype, element_st=_max_min_st(dtype)), label="array")
    lo, hi = sorted([data.draw(_max_min_st(dtype), label="lo"), data.draw(_max_min_st(dtype), label="hi")])
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
