"""
Tests for element-wise float classification ops. Mirrors the test block in
jix/src/ops/logical1.rs. is_nan is kept as a property test; is_finite/is_infinite
use fixed-input `test_*_concrete` functions ([nan, inf, -inf, 0.0, 1.0] per dtype).
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    assert_array_matches,
    carray_strategy,
    maybe_non_finite_element_strategy,
)

import jix

_float_dtypes = [np.float16, np.float32, np.float64]


@pytest.mark.parametrize("dtype", _float_dtypes)
@given(st.data())
def test_is_nan(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=maybe_non_finite_element_strategy(dtype)),
        label="array",
    )
    assert_array_matches(jix.is_nan(za), np.isnan(np_a), data=data)


@pytest.mark.parametrize("dtype", _float_dtypes)
def test_is_finite_concrete(dtype: np.dtype):
    np_a = np.array([float("nan"), float("inf"), float("-inf"), 0.0, 1.0], dtype=dtype)
    za = jix.compact(np_a)
    assert_array_matches(jix.is_finite(za), np.isfinite(np_a))


@pytest.mark.parametrize("dtype", _float_dtypes)
def test_is_infinite_concrete(dtype: np.dtype):
    np_a = np.array([float("nan"), float("inf"), float("-inf"), 0.0, 1.0], dtype=dtype)
    za = jix.compact(np_a)
    assert_array_matches(jix.is_infinite(za), np.isinf(np_a))
