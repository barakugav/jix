"""
Property tests for element-wise unary ops. Mirrors the test block in jix/src/ops/op1.rs.

One test per dtype, parametrized via @pytest.mark.parametrize, analogous to the
test_op1! macro which expands to one proptest per (op, dtype) pair.

Python name differences vs Rust:
  ln  -> log
  neg -> negative
  abs -> absolute
"""

import numpy as np
import pytest
import jix
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
    uints,
)


@pytest.mark.parametrize("dtype", ints + floats + complexes)
@given(st.data())
def test_negative(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(jix.negative(za), -np_a, data=data)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_floor(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(jix.floor(za), np.floor(np_a), data=data)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_ceil(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(jix.ceil(za), np.ceil(np_a), data=data)


def _round_half_away_from_zero(a: np.ndarray) -> np.ndarray:
    """Reference for jix.round: ties round away from zero (Rust f32/f64::round semantics)."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_round(dtype: np.dtype, data: DataObject):
    # numpy.round uses banker's rounding (half-to-even); jix.round rounds half away from zero.
    # Use a reference that matches Rust's f32::round / f64::round semantics.
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(jix.round(za), _round_half_away_from_zero(np_a), data=data)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_sqrt(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.sqrt(za), np.sqrt(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_exp(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.exp(za), np.exp(np_a).astype(dtype), data=data, rtol=rtol)


def test_log_natural_no_base():
    a = jix.compact([1.0, np.e, np.e**2], dtype=np.float64)
    result = jix.log(a).numpy()
    assert abs(result[0]) < 1e-12       # ln(1) = 0
    assert abs(result[1] - 1.0) < 1e-12 # ln(e) = 1
    assert abs(result[2] - 2.0) < 1e-12 # ln(e^2) = 2


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_log(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.log(za), np.log(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_log_base2(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.log(za, base=2), np.log2(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_log_base10(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.log(za, base=10), np.log10(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_sin(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.sin(za), np.sin(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_cos(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.cos(za), np.cos(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_tan(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.tan(za), np.tan(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_asin(dtype: np.dtype, data: DataObject):
    # Domain [-1, 1]; use unit_element_strategy to avoid NaN comparison failures.
    # f16 excluded: asin is in [-pi/2, pi/2], not representable precisely in f16.
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=unit_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(jix.asin(za), np.arcsin(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_acos(dtype: np.dtype, data: DataObject):
    # Domain [-1, 1]; use unit_element_strategy to avoid NaN comparison failures.
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=unit_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    assert_array_matches(jix.acos(za), np.arccos(np_a), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_atan(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.atan(za), np.arctan(np_a).astype(dtype), data=data, rtol=rtol)


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_sign_float(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    # jix.sign on floats: +1.0 for positive and +0.0, -1.0 for negative and -0.0.
    # np.sign returns 0.0 for 0.0; np.copysign(1, x) matches Rust's f32::signum.
    assert_array_matches(jix.sign(za), np.copysign(np.ones_like(np_a), np_a), data=data)


@pytest.mark.parametrize("dtype", ints)
@given(st.data())
def test_sign_int(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    # jix.sign on signed integers: -1, 0, or +1 of the same dtype.
    assert_array_matches(jix.sign(za), np.sign(np_a).astype(dtype), data=data)


@pytest.mark.parametrize("dtype", uints)
@given(st.data())
def test_sign_uint(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    # unsigned: sign is 0 when zero, 1 otherwise; output keeps the same uint dtype.
    expected = np.where(np_a == 0, np.zeros_like(np_a), np.ones_like(np_a))
    assert_array_matches(jix.sign(za), expected, data=data)


def test_sign_auto_cast_bool():
    """sign(bool) auto-casts to int8."""
    np_a = np.array([True, False, True], dtype=np.bool_)
    za = jix.compact(np_a)
    result = jix.sign(za)
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result.numpy(), np.sign(np_a.astype(np.int8)))


@pytest.mark.parametrize("dtype", ints + floats)
@given(st.data())
def test_absolute_scalar(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(jix.absolute(za), np.abs(np_a), data=data)


@pytest.mark.parametrize("dtype", complexes)
@given(st.data())
def test_absolute_complex(dtype: np.dtype, data: DataObject):
    # Complex absolute returns the real component dtype, not the input dtype.
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    result = jix.absolute(za)
    expected = np.abs(np_a)
    # Output dtype: complex64 -> float32, complex128 -> float64
    expected_dtype = np.float32 if dtype == np.complex64 else np.float64
    assert result.dtype == expected_dtype
    assert_array_matches(result, expected, data=data, rtol=1e-5)


# ---------------------------------------------------------------------------
# Auto-cast (Safe dispatch) for op1
#
# When the input dtype is not directly in the dispatch list, the first impl
# whose CastKind::Safe rule accepts the input dtype is selected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        # uint types auto-cast to the smallest signed type with higher precision
        (np.uint8, np.int16),  # UInt P1 -> Int P2 (higher prec needed)
        (np.uint16, np.int32),  # UInt P2 -> Int P4
        (np.uint32, np.int64),  # UInt P4 -> Int P8
    ],
)
def test_negative_auto_cast_uint(dtype, expected_dtype):
    """negative() on uint dtypes auto-casts to the next larger signed integer."""
    np_a = np.array([1, 2, 3], dtype=dtype)
    za = jix.compact(np_a)
    result = jix.negative(za)
    assert result.dtype == np.dtype(expected_dtype), (
        f"negative({dtype.__name__}): got {result.dtype}, expected {expected_dtype.__name__}"
    )
    np.testing.assert_array_equal(result.numpy(), -np_a.astype(expected_dtype))


def test_negative_on_bool_auto_casts():
    """negative(bool) auto-casts to the first signed integer impl (i8)."""
    np_a = np.array([True, False, True], dtype=np.bool_)
    za = jix.compact(np_a)
    result = jix.negative(za)
    # bool is Rank::Bool -> first Int impl: i8
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result.numpy(), -(np_a.astype(np.int8)))
