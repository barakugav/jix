"""
Property tests for element-wise unary ops. Mirrors the test block in zix/src/ops/op1.rs.

One test per dtype, parametrized via @pytest.mark.parametrize, analogous to the
test_op1! macro which expands to one proptest per (op, dtype) pair.

Python name differences vs Rust:
  ln  → log
  neg → negative
  abs → absolute
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    assert_array_matches,
    carray_strategy,
    complexes,
    floats,
    ints,
    op_safe_non_negative_element_strategy,
    unit_element_strategy,
)

import zix


@pytest.mark.parametrize("dtype", ints + floats + complexes)
@given(st.data())
def test_negative(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(zix.negative(za), -np_a, data=data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_floor(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(zix.floor(za), np.floor(np_a), data=data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_ceil(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(zix.ceil(za), np.ceil(np_a), data=data)


def _round_half_away_from_zero(a: np.ndarray) -> np.ndarray:
    """Reference for zix.round: ties round away from zero (Rust f32/f64::round semantics)."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_round(dtype: np.dtype, data: DataObject):
    # numpy.round uses banker's rounding (half-to-even); zix.round rounds half away from zero.
    # Use a reference that matches Rust's f32::round / f64::round semantics.
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(zix.round(za), _round_half_away_from_zero(np_a), data=data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_sqrt(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.sqrt(za), np.sqrt(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_exp(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.exp(za), np.exp(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_log(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.log(za), np.log(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_sin(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.sin(za), np.sin(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_cos(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.cos(za), np.cos(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_tan(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.tan(za), np.tan(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_asin(dtype: np.dtype, data: DataObject):
    # Domain [-1, 1]; use unit_element_strategy to avoid NaN comparison failures.
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=unit_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.asin(za), np.arcsin(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_acos(dtype: np.dtype, data: DataObject):
    # Domain [-1, 1]; use unit_element_strategy to avoid NaN comparison failures.
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=unit_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.acos(za), np.arccos(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_atan(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(zix.atan(za), np.arctan(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_signum(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    # zix.signum: +1.0 for positive and +0.0, -1.0 for negative and -0.0.
    # np.sign returns 0.0 for 0.0; np.copysign(1, x) matches Rust's f32::signum.
    assert_array_matches(zix.signum(za), np.copysign(np.ones_like(np_a), np_a), data=data)


@pytest.mark.parametrize("dtype", ints + floats)
@given(st.data())
def test_absolute_scalar(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(zix.absolute(za), np.abs(np_a), data=data)


@pytest.mark.parametrize("dtype", complexes)
@given(st.data())
def test_absolute_complex(dtype: np.dtype, data: DataObject):
    # Complex absolute returns the real component dtype, not the input dtype.
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    result = zix.absolute(za)
    expected = np.abs(np_a)
    # Output dtype: complex64 → float32, complex128 → float64
    expected_dtype = np.float32 if dtype == np.complex64 else np.float64
    assert result.dtype == expected_dtype
    assert_array_matches(result, expected, data=data, rtol=1e-5)
