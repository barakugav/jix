"""
Tests for element-wise unary ops. Mirrors the test block in jix/src/ops/op1.rs.

negative/exp/log are kept as property tests, one per dtype, parametrized via
@pytest.mark.parametrize (analogous to the test_op1! macro expanding to one proptest
per (op, dtype) pair). The remaining ops use fixed-input `test_*_concrete` functions
covering the same edge cases over 2-3 representative dtypes each.

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
    check_op1_concrete,
    complexes,
    floats,
    ints,
    op_safe_non_negative_element_strategy,
)


@pytest.mark.parametrize("dtype", ints + floats + complexes)
@given(st.data())
def test_negative(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype), label="array")
    assert_array_matches(jix.negative(za), -np_a, data=data)


def test_floor_concrete():
    # Edge inputs: exact ints, +/- fractions, .5 tie; float32 + float64 paths;
    # a non-default block shape to cross block boundaries.
    values = [[-2.5, -0.5, 0.0], [0.5, 2.5, 3.9]]
    check_op1_concrete(jix.floor, np.floor, [(np.float32, values, [1, 2]), (np.float64, values, [1, 2])])


def test_ceil_concrete():
    # Same edge inputs as floor: exact ints, +/- fractions, .5 tie.
    values = [[-2.5, -0.5, 0.0], [0.5, 2.5, 3.9]]
    check_op1_concrete(jix.ceil, np.ceil, [(np.float32, values), (np.float64, values)])


def _round_half_away_from_zero(a: np.ndarray) -> np.ndarray:
    """Reference for jix.round: ties round away from zero (Rust f32/f64::round semantics)."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def test_round_concrete():
    # numpy.round uses banker's rounding (half-to-even); jix.round rounds half away from zero.
    # .5 ties round away from zero: 0.5 -> 1.0, -0.5 -> -1.0, 2.5 -> 3.0, -2.5 -> -3.0.
    values = [[-2.5, -0.5, 0.0], [0.5, 2.5, 1.4]]
    check_op1_concrete(jix.round, _round_half_away_from_zero, [(np.float32, values), (np.float64, values)])


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
        carray_strategy(dtype, element_st=op_safe_non_negative_element_strategy(dtype)),
        label="array",
    )
    rtol = 1e-2 if dtype == np.float16 else (1e-5 if dtype == np.float32 else 1e-12)
    assert_array_matches(jix.log(za), np.log(np_a).astype(dtype), data=data, rtol=rtol)


def test_log_base2_concrete():
    # Domain is non-negative (incl. 0.0): 0, 1, a perfect power of 2, and a non-power value,
    # up to the op_safe_non_negative_strategy bound (100.0).
    values = [[0.0, 1.0, 2.0], [4.0, 0.25, 100.0]]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(lambda za: jix.log(za, base=2), np.log2, [(dtype, values)], rtol=rtol)


def test_log_base10_concrete():
    # Domain is non-negative (incl. 0.0): 0, 1, powers of 10, and a non-power value.
    values = [[0.0, 1.0, 10.0], [100.0, 0.25, 50.0]]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(lambda za: jix.log(za, base=10), np.log10, [(dtype, values)], rtol=rtol)


def test_sin_concrete():
    # 0, +/-pi/2, pi, and a couple of interior op_safe_strategy values.
    values = [0.0, np.pi / 2, np.pi, -np.pi / 2, 1.0, -1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.sin, np.sin, [(dtype, values)], rtol=rtol)


def test_cos_concrete():
    values = [0.0, np.pi / 2, np.pi, -np.pi / 2, 1.0, -1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.cos, np.cos, [(dtype, values)], rtol=rtol)


def test_tan_concrete():
    # Avoid drawing exactly at the +/-pi/2 asymptote; op_safe_strategy never hits it either.
    values = [0.0, np.pi / 4, -np.pi / 4, 1.0, -1.0, 10.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.tan, np.tan, [(dtype, values)], rtol=rtol)


def test_asin_concrete():
    # Domain [-1, 1]: both endpoints plus interior points.
    # f16 excluded: asin is in [-pi/2, pi/2], not representable precisely in f16.
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.asin, np.arcsin, [(dtype, values)], rtol=rtol)


def test_acos_concrete():
    # Domain [-1, 1]: both endpoints plus interior points.
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.acos, np.arccos, [(dtype, values)], rtol=rtol)


def test_atan_concrete():
    values = [0.0, 1.0, -1.0, 100.0, -100.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.atan, np.arctan, [(dtype, values)], rtol=rtol)


def test_sign_float_concrete():
    # jix.sign on floats: +1.0 for positive and +0.0, -1.0 for negative and -0.0.
    # np.sign returns 0.0 for 0.0; np.copysign(1, x) matches Rust's f32::signum.
    values = [-3.0, -0.0, 0.0, 5.0]
    check_op1_concrete(
        jix.sign,
        lambda a: np.copysign(np.ones_like(a), a),
        [(np.float32, values, [2]), (np.float64, values, [2])],
    )


def test_sign_int_concrete():
    # jix.sign on signed integers: -1, 0, or +1 of the same dtype.
    check_op1_concrete(jix.sign, np.sign, [(np.int32, [-5, 0, 7])])


def test_sign_uint_concrete():
    # unsigned: sign is 0 when zero, 1 otherwise; output keeps the same uint dtype.
    check_op1_concrete(
        jix.sign,
        lambda a: np.where(a == 0, np.zeros_like(a), np.ones_like(a)),
        [(np.uint32, [0, 1, 7])],
    )


def test_sign_auto_cast_bool():
    """sign(bool) auto-casts to int8."""
    np_a = np.array([True, False, True], dtype=np.bool_)
    za = jix.compact(np_a)
    result = jix.sign(za)
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result.numpy(), np.sign(np_a.astype(np.int8)))


def test_absolute_scalar_concrete():
    # int32: negative, zero, positive, and the op_safe_strategy bound (+/-100).
    # float32/float64: negative, zero, positive.
    check_op1_concrete(
        jix.absolute,
        np.abs,
        [
            (np.int32, [-5, 0, 7, -100, 100]),
            (np.float32, [-5.0, 0.0, 7.5, -100.0]),
            (np.float64, [-5.0, 0.0, 7.5, -100.0]),
        ],
    )


def test_absolute_complex_concrete():
    # Complex absolute returns the real-component dtype, not the input dtype
    # (complex64 -> float32, complex128 -> float64). 3+4j / -3-4j exercise an exact
    # abs()==5 via the 3-4-5 triangle; 0+0j and pure-real/pure-imaginary round out the domain.
    values = [0 + 0j, 3 + 4j, -3 - 4j, 0 + 5j, 5 + 0j]
    cases = [(np.complex64, np.float32), (np.complex128, np.float64)]
    for dtype, expected_dtype in cases:
        za = jix.compact(np.array(values, dtype=dtype))
        assert jix.absolute(za).dtype == expected_dtype
    check_op1_concrete(jix.absolute, np.abs, [(dtype, values) for dtype, _ in cases], rtol=1e-5)


def test_square_int_concrete():
    # int32: negative, zero, positive, and the op_safe_strategy bound (+/-100) so the squared
    # result (10000) still fits comfortably in int32.
    # uint32: zero, and positive values up to the op_safe_strategy bound (30).
    cases = [(np.int32, [-100, -1, 0, 1, 5, 100]), (np.uint32, [0, 1, 5, 30])]
    for dtype, values in cases:
        za = jix.compact(np.array(values, dtype=dtype))
        assert jix.square(za).dtype == dtype  # square preserves the input dtype
    check_op1_concrete(jix.square, np.square, cases)


def test_square_float_complex_concrete():
    # float32/float64: negative, zero, positive, and the op_safe_strategy bound (+/-100.0).
    # complex64/complex128: zero, pure-real, pure-imaginary, and a mixed value.
    float_vals = [[-100.0, -0.5, 0.0], [0.5, 5.0, 100.0]]
    complex_vals = [0 + 0j, 3 + 0j, 0 + 4j, 3 + 4j]
    rtol_groups = [
        (1e-5, [(np.float32, float_vals), (np.complex64, complex_vals)]),
        (1e-12, [(np.float64, float_vals), (np.complex128, complex_vals)]),
    ]
    for rtol, cases in rtol_groups:
        for dtype, values in cases:
            za = jix.compact(np.array(values, dtype=dtype))
            assert jix.square(za).dtype == dtype  # square preserves the input dtype
        check_op1_concrete(jix.square, np.square, cases, rtol=rtol)


def test_square_method_matches_function():
    """Array.square() matches the jix.square() free function."""
    np_a = np.array([1.5, -2.0, 3.0], dtype=np.float32)
    za = jix.compact(np_a)
    np.testing.assert_array_equal(za.square().numpy(), jix.square(za).numpy())


def test_square_unsupported_bool_raises():
    """square() does not auto-cast bool (CastKind::None); it raises."""
    za = jix.compact(np.array([True, False], dtype=np.bool_))
    with pytest.raises(Exception):
        jix.square(za)


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
