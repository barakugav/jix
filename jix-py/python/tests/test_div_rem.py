"""
Property tests for element-wise division and remainder ops. Mirrors the test block in
jix/src/ops/elementwise/div_rem.rs.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    assert_array_matches,
    carrays2_strategy,
    complexes,
    floats,
    ints,
    op_safe_non_zero_element_strategy,
    uints,
)

import jix


@pytest.mark.parametrize("dtype", floats + complexes)
@given(st.data())
def test_divide(dtype: np.dtype, data: DataObject):
    # True division dispatches only on float/complex dtypes; integer quotients come from a
    # rounding_mode (see test_integer_division). Non-zero strategy avoids inf/NaN from
    # division-by-zero for the float case.
    nz = op_safe_non_zero_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=nz), label="arrays")
    result = za / zb
    expected = np_a / np_b
    # Complex division algorithms differ slightly between jix (Rust) and numpy;
    # allow a few ULP of tolerance. Float32 eps ~1.2e-7, float64 eps ~2.2e-16.
    rtol = 1e-5 if np.issubdtype(dtype, np.complexfloating) else 0.0
    assert_array_matches(result, expected, data=data, rtol=rtol)


def test_divide_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # float64
    df = np.array([10.0, 20.0, 30.0])
    zaf = jix.compact(df)
    check(zaf / 2.0, df / 2.0)
    check(60.0 / zaf, 60.0 / df)  # __rtruediv__ (float64 natural)
    check(zaf / [2.0, 4.0, 5.0], df / [2.0, 4.0, 5.0])
    check(zaf / (2.0, 4.0, 5.0), df / (2.0, 4.0, 5.0))
    check(jix.divide(zaf, 2.0), df / 2.0)
    check(jix.divide(60.0, zaf), 60.0 / df)  # scalar first via free-function

    # float32: use jix.divide() for scalar-first to bypass numpy promotion
    df32 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    zaf32 = jix.compact(df32)
    check(zaf32 / np.float32(2.0), df32 / np.float32(2.0))
    check(jix.divide(np.float32(60.0), zaf32), np.float32(60.0) / df32)  # scalar first via free-function
    check(
        zaf32 / np.array([2.0, 4.0, 5.0], dtype=np.float32),
        df32 / np.array([2.0, 4.0, 5.0], dtype=np.float32),
    )

    # complex128
    dc = np.array([2 + 4j, 6 + 8j], dtype=np.complex128)
    zac = jix.compact(dc)
    check(zac / complex(2, 0), dc / complex(2, 0))
    check(zac / [1 + 0j, 2 + 0j], dc / np.array([1 + 0j, 2 + 0j]))


def _remainder_euclid_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Python's `%` is floored, so by `|b|` it is never negative.
    values = [x % abs(y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist())]
    return np.array(values, dtype=a.dtype).reshape(a.shape)


def _quotient(a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The quotient of `a / b` whose remainder is `r`."""
    if np.issubdtype(a.dtype, np.integer):
        return (a - r) // b
    exact = (a.astype(np.float64) - r.astype(np.float64)) / b.astype(np.float64)
    return exact.round().astype(a.dtype)


ROUNDING_MODES = ["trunc", "floor", "euclid"]

# Each rounding mode and a reference remainder.
CONVENTIONS = [
    pytest.param("trunc", np.fmod, id="trunc"),
    pytest.param("floor", np.remainder, id="floor"),
    pytest.param("euclid", _remainder_euclid_ref, id="euclid"),
]


@pytest.mark.parametrize(("rounding_mode", "remainder_ref"), CONVENTIONS)
@pytest.mark.parametrize("dtype", ints + uints + floats)
@given(st.data())
def test_integer_division(rounding_mode, remainder_ref, dtype: np.dtype, data: DataObject):
    # The remainder against the reference, and the quotient against the one of that remainder,
    # which is exact. Non-zero divisors avoid the integer divide-by-zero error.
    nz = op_safe_non_zero_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=nz), label="arrays")
    r = remainder_ref(np_a, np_b)
    assert_array_matches(jix.remainder(za, zb, rounding_mode=rounding_mode), r, data=data)
    q = jix.divide(za, zb, rounding_mode=rounding_mode)
    assert_array_matches(q, _quotient(np_a, np_b, r), data=data)


@pytest.mark.parametrize("dtype", ints + uints + floats)
@given(st.data())
def test_floordiv_and_mod_operators_match_numpy(dtype: np.dtype, data: DataObject):
    nz = op_safe_non_zero_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=nz), label="arrays")
    assert_array_matches(za // zb, np_a // np_b, data=data)
    assert_array_matches(za % zb, np_a % np_b, data=data)


def test_integer_division_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # Every sign combination: the conventions differ only when the operands have opposite signs,
    # or (Euclidean vs floored) when the divisor is negative. Each quotient pairs with its remainder.
    a = np.array([7, -7, 7, -7, 0], dtype=np.int32)
    b = np.array([2, 2, -2, -2, 5], dtype=np.int32)
    za, zb = jix.compact(a), jix.compact(b)
    for mode, q, r in [
        ("trunc", [3, -3, -3, 3, 0], [1, -1, 1, -1, 0]),
        ("floor", [3, -4, -4, 3, 0], [1, 1, -1, -1, 0]),
        ("euclid", [3, -4, -3, 4, 0], [1, 1, 1, 1, 0]),
    ]:
        zq = jix.divide(za, zb, rounding_mode=mode)
        zr = jix.remainder(za, zb, rounding_mode=mode)
        check(zq, q)
        check(zr, r)
        check(zq * zb + zr, a)
        check(za.divide(zb, rounding_mode=mode), q)  # the Array methods
        check(za.remainder(zb, rounding_mode=mode), r)
    check(za // zb, a // b)
    check(za % zb, a % b)
    check(jix.remainder(za, zb), a % b)  # floored by default
    check(za.remainder(zb), a % b)
    check(za.divide(zb), a / b)  # true division by default

    # Scalars and lists on either side.
    d = np.array([[10, 20, 30], [40, 50, 61]], dtype=np.int64)
    zd = jix.compact(d)
    check(zd // np.int64(-7), d // -7)
    check(zd % np.int64(-7), d % -7)
    check(np.int64(65) // zd, 65 // d)  # __rfloordiv__
    check(np.int64(-65) % zd, -65 % d)  # __rmod__
    check(65 % zd, 65 % d)
    check(zd // [[5, 4, 3], [2, 5, 6]], d // [[5, 4, 3], [2, 5, 6]])
    check(jix.remainder(zd, [[3, 6, 7], [9, 8, 10]], rounding_mode="trunc"), np.fmod(d, [[3, 6, 7], [9, 8, 10]]))

    # Floats: of the exact quotient, so `0.1` fits in `1.0` 9 times, as in numpy.
    f = np.array([1.0, -7.5, 7.5, -1e-20], dtype=np.float64)
    g = np.array([0.1, 2.0, -2.0, 1.0], dtype=np.float64)
    zf, zg = jix.compact(f), jix.compact(g)
    check(zf // zg, f // g)
    check(zf % zg, f % g)
    check(jix.divide(zf, zg, rounding_mode="trunc"), [9.0, -3.0, -3.0, 0.0])
    check(jix.divide(zf, zg, rounding_mode="floor"), [9.0, -4.0, -4.0, -1.0])
    check(jix.divide(zf, zg, rounding_mode="euclid"), [9.0, -4.0, -3.0, -1.0])
    check(jix.remainder(zf, zg, rounding_mode="trunc"), np.fmod(f, g))
    check(jix.remainder(zf, zg, rounding_mode="euclid"), _remainder_euclid_ref(f, g))


@pytest.mark.parametrize("dtype", floats)
@pytest.mark.parametrize(
    ("name", "rounding_mode", "numpy_fn"),
    [("remainder", "trunc", np.fmod), ("remainder", "floor", np.remainder), ("divide", "floor", np.floor_divide)],
)
def test_float_special_values(name, rounding_mode, numpy_fn, dtype: np.dtype):
    # Every pair of special values, bit for bit (zero signs included) as numpy.
    values = [0.0, -0.0, 1.0, -1.0, 5.0, -2.5, np.inf, -np.inf, np.nan]
    a = np.array([x for x in values for _ in values], dtype=dtype)
    b = np.array([y for _ in values for y in values], dtype=dtype)
    result = getattr(jix, name)(jix.compact(a), jix.compact(b), rounding_mode=rounding_mode).numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = numpy_fn(a, b)
    np.testing.assert_array_equal(result, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(np.signbit(result[finite]), np.signbit(expected[finite]))


@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("name", ["divide", "remainder"])
def test_integer_division_rejects_complex(name, rounding_mode):
    a = jix.compact(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
    with pytest.raises(TypeError, match=f"Op '{name}' does not support"):
        _ = getattr(jix, name)(a, a, rounding_mode=rounding_mode).numpy()


@pytest.mark.parametrize("name", ["divide", "remainder"])
def test_invalid_rounding_mode_raises(name):
    a = jix.compact(np.array([7, -7], dtype=np.int32))
    with pytest.raises(ValueError, match="invalid rounding_mode value: 'round'"):
        getattr(jix, name)(a, a, rounding_mode="round")
    with pytest.raises(ValueError, match="invalid rounding_mode value: 'round'"):
        getattr(a, name)(a, rounding_mode="round")
    with pytest.raises(TypeError):
        getattr(jix, name)(a, a, "floor")  # keyword-only


def test_remainder_rejects_none_rounding_mode():
    # There is no true remainder, as there is a true division.
    a = jix.compact(np.array([7, -7], dtype=np.int32))
    with pytest.raises(TypeError):
        jix.remainder(a, a, rounding_mode=None)
