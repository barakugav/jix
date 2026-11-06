"""
Tests for element-wise exponential, logarithm and square root ops. Mirrors the test block in
jix/src/ops/elementwise/exp_log.rs.

exp/log are kept as property tests, one per dtype; the others use fixed-input
`test_*_concrete` functions.

Python name differences vs Rust:
  ln -> log
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    assert_array_matches,
    carray_strategy,
    check_op1_concrete,
    floats,
    op_safe_positive_element_strategy,
)

import jix


def test_sqrt_concrete():
    # Domain is non-negative: 0.0, a perfect square, a non-perfect square, and the
    # op_safe_non_negative_strategy bound (100.0).
    values = [[0.0, 4.0, 2.0], [9.0, 0.25, 100.0]]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.sqrt, np.sqrt, [(dtype, values, [1, 2])], rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_exp(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    with np.errstate(over="ignore"):
        assert_array_matches(jix.exp(za), np.exp(np_a).astype(dtype), data=data, rtol=rtol)


def test_log_natural_no_base():
    a = jix.compact([1.0, np.e, np.e**2], dtype=np.float64)
    result = jix.log(a).numpy()
    assert abs(result[0]) < 1e-12  # ln(1) = 0
    assert abs(result[1] - 1.0) < 1e-12  # ln(e) = 1
    assert abs(result[2] - 2.0) < 1e-12  # ln(e^2) = 2


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_log(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_positive_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.log(za), np.log(np_a).astype(dtype), data=data, rtol=rtol)


def test_log_base2_concrete():
    # Domain is non-negative (incl. 0.0, -inf with a divide by zero): 0, 1, a perfect power of 2,
    # and a non-power value, up to the op_safe_non_negative_strategy bound (100.0).
    values = [[0.0, 1.0, 2.0], [4.0, 0.25, 100.0]]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        with np.errstate(divide="ignore"):
            check_op1_concrete(lambda za: jix.log(za, base=2), np.log2, [(dtype, values)], rtol=rtol)
            check_op1_concrete(jix.log2, np.log2, [(dtype, values)], rtol=rtol)


def test_log_base10_concrete():
    # Domain is non-negative (incl. 0.0): 0, 1, powers of 10, and a non-power value.
    values = [[0.0, 1.0, 10.0], [100.0, 0.25, 50.0]]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        with np.errstate(divide="ignore"):
            check_op1_concrete(lambda za: jix.log(za, base=10), np.log10, [(dtype, values)], rtol=rtol)
            check_op1_concrete(jix.log10, np.log10, [(dtype, values)], rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_log2_log10_exact_on_powers(dtype):
    # Exact, as numpy for float64 (numpy's float32 log10 is not).
    k = np.arange(-14, 16)
    np.testing.assert_array_equal(jix.log2(np.exp2(k).astype(dtype)).numpy(), k)
    k = np.arange(5)
    np.testing.assert_array_equal(jix.log10((10.0**k).astype(dtype)).numpy(), k)
    x = np.array([0.5, 3.0, 81.0], dtype=dtype)
    np.testing.assert_allclose(jix.log(x, base=3).numpy(), np.log(x) / np.log(3), rtol=1e-3)


def test_log_base_divides():
    # Divides by ln(base) like math.log(x, base), multiplying by its reciprocal gives 2.9999999999999996
    np.testing.assert_array_equal(jix.log(np.array([16.0, 64.0]), base=4).numpy(), [2.0, 3.0])


@pytest.mark.parametrize("base", [1.0, 0.0, -0.0, -2.0])
def test_log_invalid_base(base):
    with pytest.raises(ValueError, match="base"):
        jix.log(np.array([2.0]), base=base)
