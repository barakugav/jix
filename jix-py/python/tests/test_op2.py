"""
Property tests for element-wise binary ops. Mirrors the test block in jix/src/ops/op2.rs.

One test per dtype, parametrized via @pytest.mark.parametrize, analogous to the
test_op2! macro which expands to one proptest per (op, dtype) pair.

Mixed-dtype section verifies the automatic casting / dispatch rules:
- Safe cast: the first impl in the dispatch table that can accept both operands wins.
- Scalars without explicit precision (Python int, float) are untyped and match any
  same-rank impl; scalars with precision (np.int32, np.float32) match precisely.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    assert_array_matches,
    carrays2_mixed_strategy,
    carrays2_strategy,
    check_op2_concrete,
    complexes,
    floats,
    ints,
    op_safe_non_zero_element_strategy,
    uints,
)

import jix


@pytest.mark.parametrize("dtype", ints + uints + floats + complexes)
@given(st.data())
def test_add(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype), label="arrays")
    result = za + zb
    assert_array_matches(result, np_a + np_b, data=data)


def test_add_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64: Python ints coerce to int64 naturally
    d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    za = jix.compact(d)
    check(za + np.int64(10), d + 10)  # typed scalar, broadcast
    check(np.int64(10) + za, 10 + d)  # __radd__ (int64 natural)
    check(za + np.array([[10, 20, 30], [40, 50, 60]]), d + [[10, 20, 30], [40, 50, 60]])  # numpy array
    check(za + [[10, 20, 30], [40, 50, 60]], d + [[10, 20, 30], [40, 50, 60]])  # list
    check(za + ((10, 20, 30), (40, 50, 60)), d + ((10, 20, 30), (40, 50, 60)))  # tuple
    check(jix.add(za, np.int64(10)), d + 10)  # free-function form
    check(jix.add(np.int64(10), za), 10 + d)  # free-function, scalar first

    # float64: Python floats coerce to float64 naturally
    df = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    zaf = jix.compact(df)
    check(zaf + 10.0, df + 10.0)  # Python float scalar
    check(10.0 + zaf, 10.0 + df)  # __radd__ (float64 natural)
    check(
        zaf + [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        df + [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    )
    check(
        zaf + ((10.0, 20.0, 30.0), (40.0, 50.0, 60.0)),
        df + ((10.0, 20.0, 30.0), (40.0, 50.0, 60.0)),
    )

    # float32: Python float infers float64, so use typed numpy scalars / arrays.
    # numpy scalar's __add__ promotes its type before calling __radd__, so test
    # "scalar first" via jix.add() to bypass Python's operator dispatch.
    df32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    zaf32 = jix.compact(df32)
    check(zaf32 + np.float32(10.0), df32 + np.float32(10.0))  # typed scalar, broadcast
    check(jix.add(np.float32(10.0), zaf32), np.float32(10.0) + df32)  # scalar first via free-function
    check(
        zaf32 + np.array([10.0, 20.0, 30.0], dtype=np.float32),
        df32 + np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    # complex128: Python complex coerces to complex128 naturally
    dc = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    zac = jix.compact(dc)
    check(zac + complex(1, 1), dc + complex(1, 1))  # Python complex scalar
    check(zac + [1 + 1j, 2 + 2j], dc + np.array([1 + 1j, 2 + 2j]))  # list of complex


def test_subtract_concrete():
    # int32: negatives, zero, positive, at the op_safe bound (+/-100); differences stay
    # well within int32 range so there's no overflow panic. Unsigned dtypes are excluded
    # (the jix extension is a debug build that panics on unsigned underflow).
    # float64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default
    # block shape so this case crosses a block boundary.
    check_op2_concrete(
        lambda a, b: a - b,
        lambda a, b: a - b,
        [
            (np.int32, [[-100, -1, 0], [1, 50, 100]], [[100, 1, 0], [-1, -50, -100]]),
            (
                np.float64,
                [[-100.0, -0.5, 0.0], [0.5, 50.0, 100.0]],
                [[100.0, 0.5, 0.0], [-0.5, -50.0, -100.0]],
                [1, 2],
            ),
        ],
    )


def test_subtract_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64
    d = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    za = jix.compact(d)
    check(za - np.int64(5), d - 5)
    check(np.int64(100) - za, 100 - d)  # __rsub__ (int64 natural)
    check(za - [[1, 2, 3], [4, 5, 6]], d - [[1, 2, 3], [4, 5, 6]])
    check(za - ((1, 2, 3), (4, 5, 6)), d - ((1, 2, 3), (4, 5, 6)))
    check(jix.subtract(za, np.int64(5)), d - 5)
    check(jix.subtract(np.int64(100), za), 100 - d)  # scalar first via free-function

    # float64
    df = np.array([10.0, 20.0, 30.0])
    zaf = jix.compact(df)
    check(zaf - 5.0, df - 5.0)
    check(100.0 - zaf, 100.0 - df)  # __rsub__ (float64 natural)
    check(zaf - [1.0, 2.0, 3.0], df - [1.0, 2.0, 3.0])
    check(zaf - (1.0, 2.0, 3.0), df - (1.0, 2.0, 3.0))

    # float32: use jix.subtract() for scalar-first to bypass numpy promotion
    df32 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    zaf32 = jix.compact(df32)
    check(zaf32 - np.float32(5.0), df32 - np.float32(5.0))
    check(jix.subtract(np.float32(100.0), zaf32), np.float32(100.0) - df32)  # scalar first via free-function
    check(
        zaf32 - np.array([1.0, 2.0, 3.0], dtype=np.float32),
        df32 - np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_multiply_concrete():
    # int32: negatives, zero, positive, at the op_safe bound (+/-100); products stay well
    # within int32 range so there's no overflow panic.
    # float64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default
    # block shape so this case crosses a block boundary.
    check_op2_concrete(
        lambda a, b: a * b,
        lambda a, b: a * b,
        [
            (np.int32, [[-100, -1, 0], [1, 10, 100]], [[100, -1, 0], [-1, 10, -100]]),
            (
                np.float64,
                [[-100.0, -0.5, 0.0], [0.5, 10.0, 100.0]],
                [[100.0, -0.5, 0.0], [-0.5, 10.0, -100.0]],
                [1, 2],
            ),
        ],
    )

    # complex64: full complex multiplication with negative and zero components on both
    # operands. Complex multiplication of large values can differ by a few ULP, hence rtol.
    check_op2_concrete(
        lambda a, b: a * b,
        lambda a, b: a * b,
        [
            (
                np.complex64,
                [[-3.0 + 4.0j, 0.0 + 0.0j], [2.5 - 1.5j, 100.0 - 100.0j]],
                [[1.0 - 2.0j, 5.0 + 5.0j], [-2.5 + 1.5j, -100.0 + 100.0j]],
            ),
        ],
        rtol=1e-5,
    )


def test_multiply_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64
    d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    za = jix.compact(d)
    check(za * np.int64(3), d * 3)
    check(np.int64(3) * za, 3 * d)  # __rmul__ (int64 natural)
    check(za * [[2, 3, 4], [5, 6, 7]], d * [[2, 3, 4], [5, 6, 7]])
    check(za * ((2, 3, 4), (5, 6, 7)), d * ((2, 3, 4), (5, 6, 7)))
    check(jix.multiply(za, np.int64(3)), d * 3)

    # float64
    df = np.array([1.0, 2.0, 3.0])
    zaf = jix.compact(df)
    check(zaf * 2.0, df * 2.0)
    check(2.0 * zaf, 2.0 * df)  # __rmul__ (float64 natural)
    check(zaf * [2.0, 3.0, 4.0], df * [2.0, 3.0, 4.0])
    check(zaf * (2.0, 3.0, 4.0), df * (2.0, 3.0, 4.0))

    # float32: use jix.multiply() for scalar-first to bypass numpy promotion
    df32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    zaf32 = jix.compact(df32)
    check(zaf32 * np.float32(2.0), df32 * np.float32(2.0))
    check(jix.multiply(np.float32(2.0), zaf32), np.float32(2.0) * df32)  # scalar first via free-function

    # complex128: Python complex scalar
    dc = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    zac = jix.compact(dc)
    check(zac * complex(2, 0), dc * complex(2, 0))
    check(zac * [1 + 1j, 2 + 0j], dc * np.array([1 + 1j, 2 + 0j]))


@pytest.mark.parametrize("dtype", floats + complexes)
@given(st.data())
def test_divide(dtype: np.dtype, data: DataObject):
    # divide dispatches only on float/complex dtypes; integer dtypes go through
    # floor_divide (see test_floor_divide). Non-zero strategy avoids inf/NaN from
    # division-by-zero for the float case (mirrors Rust's op_safe_non_zero_strategy).
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


def _floor_divide_ref(np_a, np_b):
    return (np_a.astype(np.float64) / np_b.astype(np.float64)).astype(np_a.dtype)


def test_floor_divide_concrete():
    # floor_divide dispatches only on integer/unsigned dtypes. Non-zero divisors avoid
    # divide-by-zero panics. jix's `//` truncates toward zero (Rust `/` semantics) - for
    # signed-negative quotients this differs from numpy's `//` which floors toward -inf,
    # so the expected is computed via float / and cast (mirroring the original
    # op_safe_non_zero_strategy-based test). A non-default block shape crosses a block
    # boundary for the signed-int case. uint32/uint64: unsigned floor division has no
    # sign/truncation subtlety.
    check_op2_concrete(
        lambda a, b: a // b,
        _floor_divide_ref,
        [
            (np.int32, [[-100, -1, 7], [1, 50, 100]], [[3, -1, 2], [-4, 7, -100]], [1, 2]),
            (np.int64, [[-100, -1, 7], [1, 50, 100]], [[3, -1, 2], [-4, 7, -100]], [1, 2]),
            (np.uint32, [[1, 20, 100], [7, 50, 30]], [[1, 3, 7], [2, 6, 4]]),
            (np.uint64, [[1, 20, 100], [7, 50, 30]], [[1, 3, 7], [2, 6, 4]]),
        ],
    )


def test_floor_divide_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64: Python ints coerce to int64 naturally. For non-negative operands //
    # and truncation-toward-zero agree, so numpy's // is the reference.
    d = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    za = jix.compact(d)
    check(za // np.int64(3), d // 3)
    check(np.int64(60) // za, 60 // d)  # __rfloordiv__ (int64 natural)
    check(za // [[5, 4, 3], [2, 5, 6]], d // [[5, 4, 3], [2, 5, 6]])
    check(za // ((5, 4, 3), (2, 5, 6)), d // ((5, 4, 3), (2, 5, 6)))
    check(jix.floor_divide(za, np.int64(3)), d // 3)
    check(jix.floor_divide(np.int64(60), za), 60 // d)  # scalar first via free-function

    # int32: signed-negative quotient - jix truncates toward zero, numpy floors
    # toward -inf, so they diverge. Compute expected via Python int truncation.
    neg = np.array([-7, -8, 9], dtype=np.int32)
    zn = jix.compact(neg)
    trunc = np.array([int(x / 2) for x in neg], dtype=np.int32)  # [-3, -4, 4]
    check(zn // np.int32(2), trunc)

    # uint32: unsigned floor div has no sign issue
    du = np.array([10, 20, 30], dtype=np.uint32)
    zu = jix.compact(du)
    check(zu // np.uint32(3), du // np.uint32(3))
    check(zu // np.array([2, 7, 4], dtype=np.uint32), du // np.array([2, 7, 4], dtype=np.uint32))


def test_floor_divide_rejects_floats():
    """floor_divide dispatches only on integer/unsigned dtypes; floats must error."""
    a = jix.compact([1.0, 2.0, 3.0], dtype=np.float32)
    b = jix.compact([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(Exception):
        _ = jix.floor_divide(a, b).numpy()


# Every (base_dtype, exponent_dtype) pair the power dispatch table accepts directly.
# Each pair maps onto exactly one impl, so the result dtype is the base dtype.
# Integer bases require an unsigned exponent; float bases accept int or float.
_POWER_DIRECT_PAIRS = [
    (np.uint8, np.uint8),
    (np.int8, np.uint8),
    (np.uint16, np.uint16),
    (np.int16, np.uint16),
    (np.uint32, np.uint32),
    (np.int32, np.uint32),
    (np.uint64, np.uint32),
    (np.int64, np.uint32),
    (np.float32, np.int32),
    (np.float32, np.float32),
    (np.float64, np.int32),
    (np.float64, np.float64),
]


@pytest.mark.parametrize("base_dtype,exp_dtype", _POWER_DIRECT_PAIRS)
@given(st.data())
def test_power(base_dtype, exp_dtype, data: DataObject):
    """For a directly-supported pair the result keeps the base dtype and values match.

    Bases and exponents are kept small so no integer power overflows its base type
    (the extension is a debug build that panics on overflow).
    """
    base_is_int = np.issubdtype(base_dtype, np.integer)
    if base_is_int:
        base_st = st.integers(-3, 3) if np.issubdtype(base_dtype, np.signedinteger) else st.integers(0, 3)
        exp_st = st.integers(0, 4)
    else:
        base_st = st.integers(1, 5).map(float)  # positive, integer-valued -> exact results
        exp_st = st.integers(0, 4) if np.issubdtype(exp_dtype, np.integer) else st.integers(0, 4).map(float)

    (np_a, za), (np_b, zb) = data.draw(
        carrays2_mixed_strategy(base_dtype, exp_dtype, element_st_a=base_st, element_st_b=exp_st),
        label="arrays",
    )
    result = jix.power(za, zb)
    assert result.dtype == np.dtype(base_dtype), f"dtype: {result.dtype} != {base_dtype}"
    expected = np.power(np_a, np_b).astype(base_dtype)
    rtol = 0.0 if base_is_int else 1e-6
    assert_array_matches(result, expected, data=data, rtol=rtol)


# (base_dtype, exp_dtype, expected_result_dtype) for pairs that do NOT match the
# dispatch table directly and so promote. Hand-derived from the Safe-cast rules and
# verified against the implementation.
_POWER_PROMOTE_CASES = [
    # Signed exponent can't fill the unsigned-exponent slot -> promote to float.
    (np.int8, np.int8, np.float32),
    (np.int16, np.int16, np.float32),
    (np.int32, np.int32, np.float64),  # 32-bit base needs f64, not f32
    (np.int64, np.int64, np.float64),
    (np.uint32, np.int8, np.float64),  # u32 base only fits f64
    # Widening within the unsigned-exponent family stays integer.
    (np.uint8, np.uint16, np.uint16),
    (np.uint8, np.uint32, np.uint32),
    (np.int8, np.uint16, np.int16),
    # 64-bit exponent has no slot (widest is 32-bit) -> promote to float.
    (np.uint8, np.uint64, np.float64),
    (np.uint64, np.uint64, np.float64),
    # f16 has no impl; it promotes to f32 (or f64).
    (np.float16, np.float16, np.float32),
    (np.float32, np.float64, np.float64),
]


def _power_promote_operands(base_dtype, exp_dtype):
    """Small fixed (base, exponent) arrays within the safe range used by the original
    hypothesis strategy (base in [1, 3], exponent in [0, 3])."""
    base_vals = [1, 2, 3, 1] if np.issubdtype(base_dtype, np.integer) else [1.0, 2.0, 3.0, 1.0]
    exp_vals = [0, 1, 2, 3] if np.issubdtype(exp_dtype, np.integer) else [0.0, 1.0, 2.0, 3.0]
    np_a = np.array(base_vals, dtype=base_dtype)
    np_b = np.array(exp_vals, dtype=exp_dtype)
    return np_a, np_b


def test_power_promotes_mixed_dtypes_concrete():
    """Mixed (base, exponent) dtypes promote to expected_dtype and values still match."""
    for base_dtype, exp_dtype, expected_dtype in _POWER_PROMOTE_CASES:
        np_a, np_b = _power_promote_operands(base_dtype, exp_dtype)
        za, zb = jix.compact(np_a), jix.compact(np_b)
        result = jix.power(za, zb)
        assert result.dtype == np.dtype(expected_dtype), f"dtype: {result.dtype} != {expected_dtype}"
        expected = np.power(np_a, np_b).astype(expected_dtype)
        rtol = 0.0 if np.issubdtype(expected_dtype, np.integer) else 1e-6
        assert_array_matches(result, expected, rtol=rtol)


def test_power_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int32 base: an unsigned-typed scalar exponent keeps the base dtype, but a plain
    # Python int (signed) promotes to float (it can't fill the unsigned exponent slot).
    di32 = np.array([2, 3, 4], dtype=np.int32)
    zai32 = jix.compact(di32)
    r = jix.power(zai32, np.uint32(3))  # typed unsigned scalar, broadcast
    assert r.dtype == np.int32
    check(r, di32**3)
    check(
        jix.power(zai32, np.array([1, 2, 3], dtype=np.uint32)),  # uint32 array exponent
        di32 ** np.array([1, 2, 3]),
    )
    assert jix.power(zai32, 3).dtype == np.float64  # plain Python int exponent -> float

    # Broadcasting: (2, 1) base ** (1, 3) exponent -> (2, 3)
    base_2d = jix.compact(np.array([[2], [3]], dtype=np.int32))
    exp_2d = jix.compact(np.array([[1, 2, 3]], dtype=np.uint32))
    assert jix.power(base_2d, exp_2d).shape == (2, 3)

    # float32: typed numpy scalars/arrays required
    df32 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    zaf32 = jix.compact(df32)
    check(jix.power(zaf32, np.float32(2.0)), df32 ** np.float32(2.0))  # scalar, broadcast
    check(jix.power(np.float32(2.0), zaf32), np.float32(2.0) ** df32)  # scalar first
    check(jix.power(zaf32, np.int32(2)), (df32**2).astype(np.float32))  # integer exponent
    check(
        jix.power(zaf32, np.array([3.0, 2.0, 0.5], dtype=np.float32)),
        df32 ** np.array([3.0, 2.0, 0.5], dtype=np.float32),
    )

    # float64: Python floats coerce to float64 naturally
    df64 = np.array([2.0, 3.0, 4.0])
    zaf64 = jix.compact(df64)
    check(jix.power(zaf64, 2.0), df64**2.0)  # Python float scalar
    check(jix.power(2.0, zaf64), 2.0**df64)  # scalar first
    check(jix.power(zaf64, [3.0, 2.0, 0.5]), df64 ** [3.0, 2.0, 0.5])  # list
    check(jix.power(zaf64, (3.0, 2.0, 0.5)), df64 ** (3.0, 2.0, 0.5))  # tuple


# ---------------------------------------------------------------------------
# Mixed-dtype op2 tests
#
# The dispatch table is consulted left-to-right; the first impl where *both*
# operands pass the CastKind::Safe rules is selected. Expected result dtypes:
#   u8  + u16  -> u16   (UInt P1 -> UInt P2, safe same-rank)
#   u8  + i32  -> i32   (UInt P1 -> Int P4, needs higher prec: P2 <= P4, ok)
#   u8  + f32  -> f32   (UInt P1 -> Float P4, needs higher prec: P2 <= P4, ok)
#   i8  + i32  -> i32   (Int P1 -> Int P4, P1 <= P4)
#   i32 + f32  -> f64   (Int P4 -> Float P4, needs higher prec: P8 <= P4? no -> f64)
#   i32 + f64  -> f64   (Int P4 -> Float P8, P8 <= P8, ok)
#   f32 + f64  -> f64   (Float P4 -> Float P8, P4 <= P8)
#   bool + i32 -> i32   (Bool -> anything is always safe)
#   f32 + c64  -> c64   (Float P4 -> Complex P4, P4 <= P4)
# ---------------------------------------------------------------------------

_MIXED_ARITH_CASES = [
    # (dtype_a, dtype_b, expected_result_dtype)
    (np.uint8, np.uint16, np.uint16),
    (np.uint8, np.int32, np.int32),
    (np.uint8, np.float32, np.float32),
    (np.int8, np.int32, np.int32),
    (np.int16, np.int64, np.int64),
    (np.int32, np.float64, np.float64),
    (np.float32, np.float64, np.float64),
    (np.bool_, np.int32, np.int32),
    (np.bool_, np.float64, np.float64),
    (np.float32, np.complex64, np.complex64),
]


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


def test_add_mixed_dtypes_concrete():
    """add(a, b) with different dtypes casts both to expected_dtype, values match."""
    for dtype_a, dtype_b, expected_dtype in _MIXED_ARITH_CASES:
        np_a, np_b = _mixed_dtype_operands(dtype_a, dtype_b)
        za, zb = jix.compact(np_a), jix.compact(np_b)
        result = jix.add(za, zb)
        assert result.dtype == np.dtype(expected_dtype), f"dtype: {result.dtype} != {expected_dtype}"
        expected = np_a.astype(expected_dtype) + np_b.astype(expected_dtype)
        np.testing.assert_array_equal(result.numpy(), expected)


def test_multiply_mixed_dtypes_concrete():
    """multiply(a, b) with different dtypes casts both to expected_dtype, values match."""
    for dtype_a, dtype_b, expected_dtype in _MIXED_ARITH_CASES:
        np_a, np_b = _mixed_dtype_operands(dtype_a, dtype_b)
        za, zb = jix.compact(np_a), jix.compact(np_b)
        result = jix.multiply(za, zb)
        assert result.dtype == np.dtype(expected_dtype), f"dtype: {result.dtype} != {expected_dtype}"
        expected = np_a.astype(expected_dtype) * np_b.astype(expected_dtype)
        rtol = 1e-5 if np.issubdtype(expected_dtype, np.complexfloating) else 0.0
        np.testing.assert_allclose(result.numpy(), expected, rtol=rtol)


def test_subtract_mixed_dtypes_concrete():
    """subtract(a, b) with different dtypes casts both to expected_dtype, values match."""
    for dtype_a, dtype_b, expected_dtype in _MIXED_ARITH_CASES:
        # Unsigned subtypes that would underflow: restrict to uint cases
        if np.issubdtype(expected_dtype, np.unsignedinteger):
            # skip: unsigned underflow panics in debug mode
            continue
        np_a, np_b = _mixed_dtype_operands(dtype_a, dtype_b)
        za, zb = jix.compact(np_a), jix.compact(np_b)
        result = jix.subtract(za, zb)
        assert result.dtype == np.dtype(expected_dtype)
        expected = np_a.astype(expected_dtype) - np_b.astype(expected_dtype)
        np.testing.assert_array_equal(result.numpy(), expected)


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


def test_mixed_dtype_result_dtype_determinism():
    """The result dtype for common mixed-type pairs is stable and matches documented rules."""

    def check(da, db, expected):
        a = jix.compact([1, 2, 3], dtype=da)
        b = jix.compact([4, 5, 6], dtype=db)
        result = jix.add(a, b)
        assert result.dtype == np.dtype(expected), (
            f"add({da.__name__}, {db.__name__}): got {result.dtype}, expected {np.dtype(expected)}"
        )

    check(np.uint8, np.uint16, np.uint16)
    check(np.uint8, np.int32, np.int32)
    check(np.uint8, np.float32, np.float32)
    check(np.int8, np.int32, np.int32)
    check(np.int32, np.float64, np.float64)
    check(np.float32, np.float64, np.float64)
    check(np.bool_, np.int32, np.int32)


def test_mixed_dtype_op2_does_not_error_on_safe_combos():
    """All 'safe' up-cast pairs must succeed, not raise UnsupportedDtype."""
    combos = [
        (np.uint8, np.uint16),
        (np.uint8, np.uint32),
        (np.uint8, np.int16),
        (np.uint8, np.int32),
        (np.uint8, np.float32),
        (np.uint8, np.float64),
        (np.int8, np.int16),
        (np.int8, np.int32),
        (np.int8, np.int64),
        (np.int16, np.int32),
        (np.int32, np.int64),
        (np.int32, np.float64),
        (np.float32, np.float64),
        (np.bool_, np.uint8),
        (np.bool_, np.int32),
        (np.bool_, np.float32),
    ]
    for da, db in combos:
        a = jix.compact([1, 2], dtype=da)
        b = jix.compact([3, 4], dtype=db)
        try:
            r = jix.add(a, b)
            _ = r.numpy()
        except Exception as e:
            pytest.fail(f"add({da.__name__}, {db.__name__}) raised: {e}")


def test_mixed_dtype_op2_complex_plus_large_int_upcasts_to_complex128():
    """complex64 + int64/uint64 upcasts to complex128 (f64-precision complex)."""
    for int_dtype in [np.int64, np.uint64]:
        a = jix.compact([1 + 2j, 3 + 4j], dtype=np.complex64)
        b = jix.compact([1, 2], dtype=int_dtype)
        result = jix.add(a, b)
        assert result.dtype == np.complex128
        np.testing.assert_array_equal(result.numpy(), np.array([2 + 2j, 5 + 4j]))


def test_complex_plus_small_int_upcasts_to_complex128():
    """complex64 + int32 upcast to complex128 (the only fitting impl)."""
    a = jix.compact([1 + 2j, 3 + 0j], dtype=np.complex64)
    b = jix.compact([10, 20], dtype=np.int32)
    result = jix.add(a, b)
    assert result.dtype == np.complex128
    np.testing.assert_array_equal(result.numpy(), np.array([11 + 2j, 23 + 0j]))


# ---------------------------------------------------------------------------
# Mixed dtype + broadcasting combined
# ---------------------------------------------------------------------------


def test_mixed_dtype_broadcasting():
    """Mixed-dtype inputs also get properly broadcast before the op."""
    # (3, 1) i8 + (1, 4) i32  -> (3, 4) i32
    np_a = np.arange(3, dtype=np.int8).reshape(3, 1)
    np_b = np.arange(4, dtype=np.int32).reshape(1, 4)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)

    result = jix.add(za, zb)
    expected = np_a.astype(np.int32) + np_b

    assert result.dtype == np.int32
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_mixed_dtype_broadcasting_float():
    """Mixed dtype broadcast: (3,) u8 + (2, 3) f32 -> (2, 3) f32."""
    np_a = np.array([1, 2, 3], dtype=np.uint8)
    np_b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)

    result = jix.add(za, zb)
    expected = np_a.astype(np.float32) + np_b

    assert result.dtype == np.float32
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result.numpy(), expected)
