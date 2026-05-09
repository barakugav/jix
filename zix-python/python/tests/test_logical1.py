"""
Property tests for element-wise float classification ops.
Mirrors the test block in zix/src/ops/logical1.rs.
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

import zix

_float_dtypes = [np.float16, np.float32, np.float64]


@pytest.mark.parametrize("dtype", _float_dtypes)
@given(st.data())
def test_is_nan(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=maybe_non_finite_element_strategy(dtype)),
        label="array",
    )
    assert_array_matches(zix.is_nan(za), np.isnan(np_a), data=data)


@pytest.mark.parametrize("dtype", _float_dtypes)
@given(st.data())
def test_is_finite(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=maybe_non_finite_element_strategy(dtype)),
        label="array",
    )
    assert_array_matches(zix.is_finite(za), np.isfinite(np_a), data=data)


@pytest.mark.parametrize("dtype", _float_dtypes)
@given(st.data())
def test_is_infinite(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=maybe_non_finite_element_strategy(dtype)),
        label="array",
    )
    assert_array_matches(zix.is_infinite(za), np.isinf(np_a), data=data)
