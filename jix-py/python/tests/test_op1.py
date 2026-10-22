"""
Tests for element-wise unary ops. Mirrors the test block in jix/src/ops/op1.rs.

negative/exp/log are kept as property tests, one per dtype, parametrized via
@pytest.mark.parametrize. The remaining ops use fixed-input `test_*_concrete` functions
covering the same edge cases over 2-3 representative dtypes each.

Python name differences vs Rust:
  ln  -> log
  neg -> negative
  abs -> absolute
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
    complexes,
    floats,
    ints,
    op_safe_positive_element_strategy,
)

import jix


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


@pytest.mark.parametrize("op", [jix.floor, jix.ceil, jix.round])
def test_rounding_returns_ints_as_is(op):
    for dtype in [np.int8, np.int64, np.uint8, np.uint64]:
        za = jix.compact(np.array([0, 5], dtype=dtype))
        assert op(za) is za
    assert op(7).numpy() == 7
    for dtype in [np.bool_, np.complex64]:
        with pytest.raises(TypeError):
            op(jix.compact(np.zeros(2, dtype=dtype)))


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
    # Its overflows (float16 past 11.1, float32 past 88.8) are tested in test_errstate.py.
    with jix.errstate(over="ignore"), np.errstate(over="ignore"):
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
        with jix.errstate(divide="ignore"), np.errstate(divide="ignore"):
            check_op1_concrete(lambda za: jix.log(za, base=2), np.log2, [(dtype, values)], rtol=rtol)
            check_op1_concrete(jix.log2, np.log2, [(dtype, values)], rtol=rtol)


def test_log_base10_concrete():
    # Domain is non-negative (incl. 0.0): 0, 1, powers of 10, and a non-power value.
    values = [[0.0, 1.0, 10.0], [100.0, 0.25, 50.0]]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        with jix.errstate(divide="ignore"), np.errstate(divide="ignore"):
            check_op1_concrete(lambda za: jix.log(za, base=10), np.log10, [(dtype, values)], rtol=rtol)
            check_op1_concrete(jix.log10, np.log10, [(dtype, values)], rtol=rtol)


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
    # jix.sign on floats matches np.sign: +-1.0, 0.0 for both zeros, NaN for NaN.
    values = [-3.0, -0.0, 0.0, 5.0, np.nan]
    check_op1_concrete(jix.sign, np.sign, [(np.float16, values), (np.float32, values, [2]), (np.float64, values, [2])])


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


def test_sign_never_casts():
    for dtype in (np.int8, np.uint16, np.float16):
        assert jix.sign(jix.compact(np.array([1, 0], dtype=dtype))).dtype == dtype
    assert jix.sign(-5).numpy() == -1  # untyped Python int -> int64
    with pytest.raises(TypeError, match="sign"):
        jix.sign(jix.compact(np.array([True, False])))


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


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64])
def test_absolute_uint_bool_returns_self(dtype):
    # abs is a no-op on unsigned and bool: the same Array object comes back, for the free
    # function, the method and the builtin.
    np_a = np.array([0, 1, 1], dtype=dtype) if dtype == np.bool_ else np.array([0, 1, 200], dtype=dtype)
    za = jix.compact(np_a)
    assert jix.absolute(za) is za
    assert za.abs() is za
    assert abs(za) is za
    assert jix.absolute(za).dtype == dtype
    np.testing.assert_array_equal(jix.absolute(za).numpy(), np.abs(np_a))


def test_absolute_method_matches_function():
    za = jix.compact(np.array([-1.5, 0.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(za.abs().numpy(), jix.absolute(za).numpy())
    np.testing.assert_array_equal(abs(za).numpy(), jix.absolute(za).numpy())


@pytest.mark.parametrize(
    "value, expected, expected_dtype",
    [
        # Python scalars take the default (64-bit) precision.
        (True, True, np.bool_),
        (False, False, np.bool_),
        (-5, 5, np.int64),
        (-(2**63), -(2**63), np.int64),  # i64 min wraps, like numpy
        (-2.5, 2.5, np.float64),
        (3 - 4j, 5.0, np.float64),
        # NumPy scalars keep their precision.
        (np.uint8(7), 7, np.uint8),
        (np.int8(-7), 7, np.int8),
        (np.int8(-128), -128, np.int8),  # i8 min wraps
        (np.float16(-1.5), 1.5, np.float16),
        (np.float32(-1.5), 1.5, np.float32),
        (np.complex64(3 + 4j), 5.0, np.float32),
    ],
)
def test_absolute_scalar_input(value, expected, expected_dtype):
    result = jix.absolute(value)
    assert result.dtype == expected_dtype
    assert result.numpy()[()] == expected


def test_absolute_unsupported_dtype_raises():
    za = jix.compact(np.zeros(3, dtype=[("a", np.int32), ("b", np.float32)]))
    with pytest.raises(TypeError, match="Op 'absolute' does not support"):
        jix.absolute(za)


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
    [(np.uint8, np.int8), (np.uint16, np.int16), (np.uint32, np.int32), (np.uint64, np.int64)],
)
def test_negative_uint_is_int_of_same_width(dtype, expected_dtype):
    """The bits are those of numpy's wrapping unsigned negation."""
    np_a = np.array([0, 1, 3, np.iinfo(dtype).max], dtype=dtype)
    result = jix.negative(jix.compact(np_a))
    assert result.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), (-np_a).view(expected_dtype))


@pytest.mark.parametrize("dtype", ints + floats + complexes + [np.uint8, np.uint64])
def test_positive_returns_self(dtype):
    za = jix.compact(np.array([1, 2, 3], dtype=dtype))
    assert +za is za


def test_positive_rejects_bool():
    with pytest.raises(TypeError, match="Op 'positive' does not support"):
        +jix.compact(np.array([True, False]))


def test_negative_on_bool_auto_casts():
    """negative(bool) auto-casts to the first signed integer impl (i8)."""
    np_a = np.array([True, False, True], dtype=np.bool_)
    za = jix.compact(np_a)
    result = jix.negative(za)
    # bool is Rank::Bool -> first Int impl: i8
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result.numpy(), -(np_a.astype(np.int8)))
