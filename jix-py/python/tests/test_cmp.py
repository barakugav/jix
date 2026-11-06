"""
Property tests for element-wise comparison ops.
Mirrors the test block in jix/src/ops/elementwise/cmp.rs.
"""

import cmath

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


def _ordering_st(dtype):
    """any for integers/bool, maybe_non_finite for floats (exercises NaN -> False paths)."""
    if np.issubdtype(dtype, np.floating):
        return maybe_non_finite_element_strategy(dtype)
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


@pytest.mark.parametrize("op", ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"])
def test_uint64_vs_int64_is_exact(op):
    # A float64 common type would round these values above 2**53.
    u = np.array([2**53 + 1, 2**63, 2**64 - 1, 0, 5], np.uint64)
    i = np.array([2**53, -1, 2**63 - 1, -(2**63), 5], np.int64)
    np.testing.assert_array_equal(getattr(jix, op)(u, i).numpy(), getattr(np, op)(u, i))
    np.testing.assert_array_equal(getattr(jix, op)(i, u).numpy(), getattr(np, op)(i, u))


def _mixed_dtype_operands(dtype_a, dtype_b):
    """Small fixed (np_a, np_b) pair for a mixed-dtype case, with values kept within each
    dtype's safe range (no overflow/underflow). Same-category dtypes share a matching
    value at index 0 (equal), the rest differ, so equal() coverage exercises both
    True and False branches."""

    def vals(dtype, variant):
        if dtype == np.bool_:
            return [True, False, True, False] if variant == 0 else [True, True, False, False]
        if np.issubdtype(dtype, np.unsignedinteger):
            return [0, 1, 2, 3] if variant == 0 else [0, 5, 2, 9]
        if np.issubdtype(dtype, np.signedinteger):
            return [-3, 0, 2, 3] if variant == 0 else [-3, 5, 2, -9]
        if np.issubdtype(dtype, np.complexfloating):
            return [1 + 1j, -2 - 2j, 0 + 0j, 3 - 1j] if variant == 0 else [1 + 1j, 2 + 2j, 0 + 0j, -3 + 1j]
        return [-2.5, 0.0, 2.0, 3.5] if variant == 0 else [-2.5, 9.0, 2.0, -1.0]  # float

    np_a = np.array(vals(dtype_a, 0), dtype=dtype_a)
    np_b = np.array(vals(dtype_b, 1), dtype=dtype_b)
    return np_a, np_b


_EQUAL_MIXED_CASES = [
    (np.uint8, np.uint16, np.uint16),
    (np.uint8, np.int32, np.int32),
    (np.uint8, np.float32, np.float32),
    (np.int8, np.int32, np.int32),
    (np.int32, np.float64, np.float64),
    (np.float32, np.float64, np.float64),
    (np.bool_, np.int32, np.int32),
]


def test_equal_mixed_dtypes_concrete():
    """equal(a, b) with different dtypes casts both to expected_dtype, output is bool."""
    for dtype_a, dtype_b, expected_dtype in _EQUAL_MIXED_CASES:
        np_a, np_b = _mixed_dtype_operands(dtype_a, dtype_b)
        za, zb = jix.compact(np_a), jix.compact(np_b)
        result = jix.equal(za, zb)
        assert result.dtype == np.bool_
        expected = np_a.astype(expected_dtype) == np_b.astype(expected_dtype)
        np.testing.assert_array_equal(result.numpy(), expected)


# ---------------------------------------------------------------------------
# Reference implementation (mirrors the Rust algorithm element-wise)
# ---------------------------------------------------------------------------


def _ref_isclose_scalar(a, b, rtol, atol):
    """Reference for a single pair of same-dtype numpy scalars (any float or complex)."""
    if a == b:
        return True
    if cmath.isinf(a) or cmath.isinf(b):
        return False
    diff = abs(a - b)
    if diff <= atol:
        return True
    largest = max(abs(a), abs(b))
    return bool(diff <= largest * rtol)


def _ref_isclose_array(np_a, np_b, rtol, atol, dtype):
    """Elementwise reference for real float ndarrays, computed in dtype precision."""
    rtol_t, atol_t = dtype(rtol), dtype(atol)
    a_t = np_a.astype(dtype)
    b_t = np_b.astype(dtype)
    # a and b share a shape (carrays2_strategy), so zip the flattened views.
    out = [_ref_isclose_scalar(a, b, rtol_t, atol_t) for a, b in zip(a_t.reshape(-1), b_t.reshape(-1))]
    return np.array(out, dtype=bool).reshape(a_t.shape)


# ---------------------------------------------------------------------------
# Tolerance strategies
# ---------------------------------------------------------------------------

_tol_st = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property tests: float dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", floats)
@given(st.data())
def test_isclose_float(dtype: np.dtype, data: DataObject):
    element_st = maybe_non_finite_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=element_st), label="arrays")
    rtol = data.draw(_tol_st, label="rtol")
    atol = data.draw(_tol_st, label="atol")
    expected = _ref_isclose_array(np_a, np_b, rtol, atol, dtype)
    assert_array_matches(jix.isclose(za, zb, rtol=rtol, atol=atol), expected, data=data)


# ---------------------------------------------------------------------------
# Concrete: complex dtypes, real atol (same tolerance for re and im)
# ---------------------------------------------------------------------------


def test_isclose_complex_real_atol_concrete():
    """The same tolerance value applies to both the real and imaginary component.
    Edge cases: exact match, within atol, outside atol (both plain and
    magnitude-scaled so rtol behaves differently from atol), NaN component,
    matching inf, opposite inf; complex64 and complex128 paths.
    """
    for dtype in (np.complex64, np.complex128):
        a = np.array(
            [
                1.0 + 2.0j,  # exact match
                1.0 + 2.0j,  # small diff: within atol AND within rtol
                1.0 + 2.0j,  # diff of 1: outside atol AND outside rtol
                100.0 + 100.0j,  # diff of 9: outside atol BUT within rtol
                complex(float("nan"), 0.0),  # NaN real component
                complex(float("inf"), 0.0),  # matching inf
                complex(float("inf"), 0.0),  # opposite inf
            ],
            dtype=dtype,
        )
        b = np.array(
            [
                1.0 + 2.0j,
                1.005 + 2.005j,
                2.0 + 3.0j,
                109.0 + 109.0j,
                complex(float("nan"), 0.0),
                complex(float("inf"), 0.0),
                complex(-float("inf"), 0.0),
            ],
            dtype=dtype,
        )
        za = jix.compact(a)
        zb = jix.compact(b)

        expected_atol = np.array([True, True, False, False, False, True, False])
        assert_array_matches(jix.isclose(za, zb, rtol=0.0, atol=0.01), expected_atol)

        expected_rtol = np.array([True, True, False, True, False, True, False])
        assert_array_matches(jix.isclose(za, zb, rtol=0.1, atol=0.0), expected_rtol)


# ---------------------------------------------------------------------------
# Concrete: complex dtypes, complex atol (independent re/im tolerances)
# ---------------------------------------------------------------------------


def test_isclose_complex_atol_concrete():
    """Real and imaginary components each get their own atol. Edge cases: exact match,
    diff that only clears the real tolerance, diff that only clears the
    imaginary tolerance, diff that clears both, NaN component, matching inf,
    opposite inf, and a magnitude-scaled diff that only clears via rtol;
    complex64 and complex128 paths.
    """
    for dtype in (np.complex64, np.complex128):
        a = np.array(
            [
                1.0 + 2.0j,  # exact match
                1.0 + 2.0j,  # real diff exceeds atol_re; imag diff within atol_im
                1.0 + 2.0j,  # real diff within atol_re; imag diff exceeds atol_im
                1.0 + 2.0j,  # both diffs within their own atol
                complex(float("nan"), 2.0),  # NaN real component, imag matches
                complex(float("inf"), 2.0),  # matching inf (real)
                1.0 + complex(0.0, float("inf")),  # opposite inf (imag)
            ],
            dtype=dtype,
        )
        b = np.array(
            [
                1.0 + 2.0j,
                1.01 + 2.1j,
                1.001 + 2.5j,
                1.001 + 2.1j,
                complex(float("nan"), 2.0),
                complex(float("inf"), 2.0),
                1.0 + complex(0.0, -float("inf")),
            ],
            dtype=dtype,
        )
        za = jix.compact(a)
        zb = jix.compact(b)

        atol = complex(0.005, 0.2)
        expected = np.array([True, False, False, True, False, True, False])
        assert_array_matches(jix.isclose(za, zb, rtol=0.0, atol=atol), expected)

        # Magnitude-scaled diff: too big for atol_re but cleared via rtol; the
        # imaginary component clears atol_im directly regardless of rtol.
        c = np.array([100.0 + 100.0j], dtype=dtype)
        d = np.array([109.0 + 100.1j], dtype=dtype)
        zc = jix.compact(c)
        zd = jix.compact(d)
        expected_rtol = np.array([True])
        assert_array_matches(jix.isclose(zc, zd, rtol=0.1, atol=atol), expected_rtol)


# ---------------------------------------------------------------------------
# Exact-match edge cases
# ---------------------------------------------------------------------------


def test_isclose_exact_match():
    a = jix.compact([1.0, -2.5, 0.0], dtype=np.float32)
    b = jix.compact([1.0, -2.5, 0.0], dtype=np.float32)
    result = jix.isclose(a, b, rtol=0.0, atol=0.0).numpy()
    assert np.all(result)


def test_isclose_within_atol():
    a = jix.compact([1.0, 2.0, 3.0], dtype=np.float32)
    b = jix.compact([1.0, 2.01, 4.0], dtype=np.float32)
    result = jix.isclose(a, b, rtol=0.0, atol=0.02).numpy()
    assert np.array_equal(result, [True, True, False])


def test_isclose_within_rtol():
    a = jix.compact([100.0, 1.0], dtype=np.float64)
    b = jix.compact([109.0, 1.12], dtype=np.float64)
    result = jix.isclose(a, b, rtol=0.1, atol=0.0).numpy()
    assert np.array_equal(result, [True, False])


def test_isclose_nan_always_false():
    a = jix.compact([float("nan"), 1.0], dtype=np.float32)
    b = jix.compact([float("nan"), 1.0], dtype=np.float32)
    result = jix.isclose(a, b, rtol=0.0, atol=0.0).numpy()
    assert np.array_equal(result, [False, True])


def test_isclose_same_inf():
    a = jix.compact([float("inf"), float("-inf")], dtype=np.float32)
    b = jix.compact([float("inf"), float("-inf")], dtype=np.float32)
    result = jix.isclose(a, b, rtol=0.0, atol=0.0).numpy()
    assert np.all(result)


def test_isclose_opposite_inf():
    a = jix.compact([float("inf"), float("-inf")], dtype=np.float32)
    b = jix.compact([float("-inf"), float("inf")], dtype=np.float32)
    result = jix.isclose(a, b, rtol=0.0, atol=0.0).numpy()
    assert not np.any(result)


def test_isclose_inf_vs_finite():
    a = jix.compact([float("inf"), 1.0], dtype=np.float32)
    b = jix.compact([1.0, float("inf")], dtype=np.float32)
    result = jix.isclose(a, b, rtol=1e10, atol=1e10).numpy()
    assert np.array_equal(result, [False, False])


# ---------------------------------------------------------------------------
# Complex edge cases
# ---------------------------------------------------------------------------


def test_isclose_complex_scalar_atol():
    # Different per-component tolerances via complex atol.
    a = jix.compact([1.0 + 2.0j], dtype=np.complex64)
    b = jix.compact([1.01 + 2.1j], dtype=np.complex64)
    # re diff = 0.01, im diff = 0.1
    tight = jix.isclose(a, b, rtol=0.0, atol=complex(0.005, 0.2)).numpy()
    loose = jix.isclose(a, b, rtol=0.0, atol=complex(0.02, 0.2)).numpy()
    assert np.array_equal(tight, [False])  # re diff 0.01 > 0.005
    assert np.array_equal(loose, [True])  # both within tolerance


def test_isclose_complex_real_atol_applies_to_both():
    a = jix.compact([1.0 + 2.0j], dtype=np.complex64)
    b = jix.compact([1.01 + 2.01j], dtype=np.complex64)
    result = jix.isclose(a, b, rtol=0.0, atol=0.02).numpy()
    assert np.array_equal(result, [True])


def test_isclose_complex_nan_component():
    a = jix.compact([complex(float("nan"), 0.0)], dtype=np.complex64)
    b = jix.compact([complex(float("nan"), 0.0)], dtype=np.complex64)
    result = jix.isclose(a, b, rtol=0.0, atol=0.0).numpy()
    assert np.array_equal(result, [False])


# ---------------------------------------------------------------------------
# Type-error cases
# ---------------------------------------------------------------------------


def test_isclose_complex_rtol_raises():
    a = jix.compact([1.0, 2.0], dtype=np.float32)
    b = jix.compact([1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError):
        jix.isclose(a, b, rtol=complex(1e-5, 0), atol=0.0)


def test_isclose_complex_atol_on_real_dtype_raises():
    a = jix.compact([1.0, 2.0], dtype=np.float32)
    b = jix.compact([1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError):
        jix.isclose(a, b, rtol=0.0, atol=complex(1e-8, 0))


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


def test_isclose_broadcasting():
    a = jix.compact([[0.0], [1.0], [2.0]], dtype=np.float32)
    b = jix.compact([[0.0, 0.01, 0.1]], dtype=np.float32)
    result = jix.isclose(a, b, rtol=0.0, atol=0.02).numpy()
    expected = np.array(
        [
            [True, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    assert np.array_equal(result, expected)
