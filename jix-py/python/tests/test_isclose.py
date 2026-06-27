"""
Property tests and edge-case tests for jix.isclose.
Mirrors the approx_equal test block in jix/src/ops/cmp.rs.
"""

import cmath

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    assert_array_matches,
    carrays2_strategy,
    floats,
    complexes,
    maybe_non_finite_element_strategy,
)

import jix


# ---------------------------------------------------------------------------
# Reference implementation (mirrors the Rust algorithm element-wise)
# ---------------------------------------------------------------------------

# Maps each supported dtype to its float component type (for tolerance casting).
_COMPONENT_DTYPE = {
    np.float16: np.float16,
    np.float32: np.float32,
    np.float64: np.float64,
    np.complex64: np.float32,
    np.complex128: np.float64,
}


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


def _ref_isclose_complex_scalar(a, b, rtol, atol_re, atol_im):
    """Per-component reference for a single complex pair."""
    re_type = type(a.real)
    return _ref_isclose_scalar(re_type(a.real), re_type(b.real), rtol, atol_re) and _ref_isclose_scalar(
        re_type(a.imag), re_type(b.imag), rtol, atol_im
    )


def _ref_isclose_array(np_a, np_b, rtol, atol, dtype):
    """Vectorised reference for real float ndarrays, computed in dtype precision."""
    ft = _COMPONENT_DTYPE[dtype]
    rtol_t, atol_t = ft(rtol), ft(atol)
    a_t = np_a.astype(ft)
    b_t = np_b.astype(ft)
    return np.vectorize(lambda a, b: _ref_isclose_scalar(a, b, rtol_t, atol_t), otypes=[bool])(a_t, b_t)


def _ref_isclose_complex_array(np_a, np_b, rtol, atol_re, atol_im, dtype):
    """Vectorised reference for complex ndarrays, computed in component precision."""
    ft = _COMPONENT_DTYPE[dtype]
    rtol_t, atol_re_t, atol_im_t = ft(rtol), ft(atol_re), ft(atol_im)
    return np.vectorize(
        lambda a, b: _ref_isclose_complex_scalar(a, b, rtol_t, atol_re_t, atol_im_t),
        otypes=[bool],
    )(np_a, np_b)


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
# Property tests: complex dtypes, real atol (same tolerance for re and im)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", complexes)
@given(st.data())
def test_isclose_complex_real_atol(dtype: np.dtype, data: DataObject):
    component = np.float32 if dtype == np.complex64 else np.float64
    element_st = st.tuples(
        maybe_non_finite_element_strategy(component),
        maybe_non_finite_element_strategy(component),
    ).map(lambda x: complex(x[0], x[1]))
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=element_st), label="arrays")
    rtol = data.draw(_tol_st, label="rtol")
    atol = data.draw(_tol_st, label="atol")
    expected = _ref_isclose_complex_array(np_a, np_b, rtol, atol, atol, dtype)
    assert_array_matches(jix.isclose(za, zb, rtol=rtol, atol=atol), expected, data=data)


# ---------------------------------------------------------------------------
# Property tests: complex dtypes, complex atol (independent re/im tolerances)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", complexes)
@given(st.data())
def test_isclose_complex_atol(dtype: np.dtype, data: DataObject):
    component = np.float32 if dtype == np.complex64 else np.float64
    element_st = st.tuples(
        maybe_non_finite_element_strategy(component),
        maybe_non_finite_element_strategy(component),
    ).map(lambda x: complex(x[0], x[1]))
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=element_st), label="arrays")
    rtol = data.draw(_tol_st, label="rtol")
    atol_re = data.draw(_tol_st, label="atol_re")
    atol_im = data.draw(_tol_st, label="atol_im")
    atol = complex(atol_re, atol_im)
    expected = _ref_isclose_complex_array(np_a, np_b, rtol, atol_re, atol_im, dtype)
    assert_array_matches(jix.isclose(za, zb, rtol=rtol, atol=atol), expected, data=data)


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
