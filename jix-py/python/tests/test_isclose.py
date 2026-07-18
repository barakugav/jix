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
    maybe_non_finite_element_strategy,
)

import jix


# ---------------------------------------------------------------------------
# Reference implementation (mirrors the Rust algorithm element-wise)
# ---------------------------------------------------------------------------

# Maps each supported float dtype to its component type (for tolerance casting).
# Only used by test_isclose_float (real dtypes); the complex property tests were
# converted to concrete tests that compute expected results by hand, so no complex
# entries are needed here.
_COMPONENT_DTYPE = {
    np.float16: np.float16,
    np.float32: np.float32,
    np.float64: np.float64,
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


def _ref_isclose_array(np_a, np_b, rtol, atol, dtype):
    """Vectorised reference for real float ndarrays, computed in dtype precision."""
    ft = _COMPONENT_DTYPE[dtype]
    rtol_t, atol_t = ft(rtol), ft(atol)
    a_t = np_a.astype(ft)
    b_t = np_b.astype(ft)
    return np.vectorize(lambda a, b: _ref_isclose_scalar(a, b, rtol_t, atol_t), otypes=[bool])(a_t, b_t)


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
    """Concrete replacement for the complex real_atol property test: the same
    tolerance value applies to both the real and imaginary component.
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
    """Concrete replacement for the complex atol property test: real and
    imaginary components each get their own atol. Edge cases: exact match,
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
