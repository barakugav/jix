"""
Property tests for element-wise binary ops. Mirrors the test block in zix/src/ops/op2.rs.

One test per dtype, parametrized via @pytest.mark.parametrize, analogous to the
test_op2! macro which expands to one proptest per (op, dtype) pair.
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

import zix


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
    za = zix.compact(d)
    check(za + np.int64(10), d + 10)  # typed scalar, broadcast
    check(np.int64(10) + za, 10 + d)  # __radd__ (int64 natural)
    check(
        za + np.array([[10, 20, 30], [40, 50, 60]]), d + [[10, 20, 30], [40, 50, 60]]
    )  # numpy array
    check(za + [[10, 20, 30], [40, 50, 60]], d + [[10, 20, 30], [40, 50, 60]])  # list
    check(za + ((10, 20, 30), (40, 50, 60)), d + ((10, 20, 30), (40, 50, 60)))  # tuple
    check(zix.add(za, np.int64(10)), d + 10)  # free-function form
    check(zix.add(np.int64(10), za), 10 + d)  # free-function, scalar first

    # float64: Python floats coerce to float64 naturally
    df = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    zaf = zix.compact(df)
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
    # "scalar first" via zix.add() to bypass Python's operator dispatch.
    df32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    zaf32 = zix.compact(df32)
    check(zaf32 + np.float32(10.0), df32 + np.float32(10.0))  # typed scalar, broadcast
    check(
        zix.add(np.float32(10.0), zaf32), np.float32(10.0) + df32
    )  # scalar first via free-function
    check(
        zaf32 + np.array([10.0, 20.0, 30.0], dtype=np.float32),
        df32 + np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    # complex128: Python complex coerces to complex128 naturally
    dc = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    zac = zix.compact(dc)
    check(zac + complex(1, 1), dc + complex(1, 1))  # Python complex scalar
    check(zac + [1 + 1j, 2 + 2j], dc + np.array([1 + 1j, 2 + 2j]))  # list of complex


@pytest.mark.parametrize("dtype", ints + floats + complexes)
@given(st.data())
def test_subtract(dtype: np.dtype, data: DataObject):
    # Unsigned types are excluded: the zix extension is a debug build that panics on
    # unsigned underflow, matching Rust's test_op2!(sub, ..., [i8, i16, i32, i64, ...]).
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype), label="arrays")
    result = za - zb
    assert_array_matches(result, np_a - np_b, data=data)


def test_subtract_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64
    d = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    za = zix.compact(d)
    check(za - np.int64(5), d - 5)
    check(np.int64(100) - za, 100 - d)  # __rsub__ (int64 natural)
    check(za - [[1, 2, 3], [4, 5, 6]], d - [[1, 2, 3], [4, 5, 6]])
    check(za - ((1, 2, 3), (4, 5, 6)), d - ((1, 2, 3), (4, 5, 6)))
    check(zix.subtract(za, np.int64(5)), d - 5)
    check(zix.subtract(np.int64(100), za), 100 - d)  # scalar first via free-function

    # float64
    df = np.array([10.0, 20.0, 30.0])
    zaf = zix.compact(df)
    check(zaf - 5.0, df - 5.0)
    check(100.0 - zaf, 100.0 - df)  # __rsub__ (float64 natural)
    check(zaf - [1.0, 2.0, 3.0], df - [1.0, 2.0, 3.0])
    check(zaf - (1.0, 2.0, 3.0), df - (1.0, 2.0, 3.0))

    # float32: use zix.subtract() for scalar-first to bypass numpy promotion
    df32 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    zaf32 = zix.compact(df32)
    check(zaf32 - np.float32(5.0), df32 - np.float32(5.0))
    check(
        zix.subtract(np.float32(100.0), zaf32), np.float32(100.0) - df32
    )  # scalar first via free-function
    check(
        zaf32 - np.array([1.0, 2.0, 3.0], dtype=np.float32),
        df32 - np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


@pytest.mark.parametrize("dtype", ints + uints + floats + complexes)
@given(st.data())
def test_multiply(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype), label="arrays")
    result = za * zb
    # Complex multiplication of large values can differ by a few ULP across implementations.
    rtol = 1e-5 if np.issubdtype(dtype, np.complexfloating) else 0.0
    assert_array_matches(result, np_a * np_b, data=data, rtol=rtol)


def test_multiply_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64
    d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    za = zix.compact(d)
    check(za * np.int64(3), d * 3)
    check(np.int64(3) * za, 3 * d)  # __rmul__ (int64 natural)
    check(za * [[2, 3, 4], [5, 6, 7]], d * [[2, 3, 4], [5, 6, 7]])
    check(za * ((2, 3, 4), (5, 6, 7)), d * ((2, 3, 4), (5, 6, 7)))
    check(zix.multiply(za, np.int64(3)), d * 3)

    # float64
    df = np.array([1.0, 2.0, 3.0])
    zaf = zix.compact(df)
    check(zaf * 2.0, df * 2.0)
    check(2.0 * zaf, 2.0 * df)  # __rmul__ (float64 natural)
    check(zaf * [2.0, 3.0, 4.0], df * [2.0, 3.0, 4.0])
    check(zaf * (2.0, 3.0, 4.0), df * (2.0, 3.0, 4.0))

    # float32: use zix.multiply() for scalar-first to bypass numpy promotion
    df32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    zaf32 = zix.compact(df32)
    check(zaf32 * np.float32(2.0), df32 * np.float32(2.0))
    check(
        zix.multiply(np.float32(2.0), zaf32), np.float32(2.0) * df32
    )  # scalar first via free-function

    # complex128: Python complex scalar
    dc = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    zac = zix.compact(dc)
    check(zac * complex(2, 0), dc * complex(2, 0))
    check(zac * [1 + 1j, 2 + 0j], dc * np.array([1 + 1j, 2 + 0j]))


@pytest.mark.parametrize("dtype", ints + uints + floats + complexes)
@given(st.data())
def test_divide(dtype: np.dtype, data: DataObject):
    # Use non-zero strategy for both operands to avoid integer division-by-zero panics.
    # Mirrors Rust's test_op2!(div, ..., op_safe_non_zero_strategy).
    nz = op_safe_non_zero_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=nz), label="arrays"
    )
    result = za / zb
    # zix integer division truncates toward zero (Rust semantics); numpy // is floor division.
    # Cast through float64 and back to get truncation-toward-zero for all signed/unsigned combos.
    if np.issubdtype(dtype, np.integer):
        expected = (np_a.astype(np.float64) / np_b.astype(np.float64)).astype(dtype)
    else:
        expected = np_a / np_b
    # Complex division algorithms differ slightly between zix (Rust) and numpy;
    # allow a few ULP of tolerance. Float32 eps ~1.2e-7, float64 eps ~2.2e-16.
    rtol = 1e-5 if np.issubdtype(dtype, np.complexfloating) else 0.0
    assert_array_matches(result, expected, data=data, rtol=rtol)


def test_divide_custom_inputs():
    def check_int(result, expected_int):
        # zix integer division truncates (same as //); numpy / would give float
        np.testing.assert_array_equal(result.numpy(), expected_int)

    def check_float(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # int64 (use // for reference: floor div == truncating for positive values)
    d = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    za = zix.compact(d)
    check_int(za / np.int64(10), d // 10)
    check_int(np.int64(60) / za, 60 // d)  # __rtruediv__ (int64 natural)
    check_int(za / [[5, 4, 3], [2, 5, 6]], d // [[5, 4, 3], [2, 5, 6]])
    check_int(za / ((5, 4, 3), (2, 5, 6)), d // ((5, 4, 3), (2, 5, 6)))
    check_int(zix.divide(za, np.int64(10)), d // 10)

    # float64
    df = np.array([10.0, 20.0, 30.0])
    zaf = zix.compact(df)
    check_float(zaf / 2.0, df / 2.0)
    check_float(60.0 / zaf, 60.0 / df)  # __rtruediv__ (float64 natural)
    check_float(zaf / [2.0, 4.0, 5.0], df / [2.0, 4.0, 5.0])
    check_float(zaf / (2.0, 4.0, 5.0), df / (2.0, 4.0, 5.0))

    # float32: use zix.divide() for scalar-first to bypass numpy promotion
    df32 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    zaf32 = zix.compact(df32)
    check_float(zaf32 / np.float32(2.0), df32 / np.float32(2.0))
    check_float(
        zix.divide(np.float32(60.0), zaf32), np.float32(60.0) / df32
    )  # scalar first via free-function
    check_float(
        zaf32 / np.array([2.0, 4.0, 5.0], dtype=np.float32),
        df32 / np.array([2.0, 4.0, 5.0], dtype=np.float32),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@given(st.data())
def test_power(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=st.integers(1, 5).map(float)),
        label="arrays",
    )
    result = zix.power(za, zb)
    assert_array_matches(result, np_a**np_b, data=data)


def test_power_custom_inputs():
    def check(result, expected):
        np.testing.assert_array_equal(result.numpy(), expected)

    # float32: typed numpy scalars/arrays required
    df32 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    zaf32 = zix.compact(df32)
    check(
        zix.power(zaf32, np.float32(2.0)), df32 ** np.float32(2.0)
    )  # scalar, broadcast
    check(zix.power(np.float32(2.0), zaf32), np.float32(2.0) ** df32)  # scalar first
    check(
        zix.power(zaf32, np.array([3.0, 2.0, 0.5], dtype=np.float32)),
        df32 ** np.array([3.0, 2.0, 0.5], dtype=np.float32),
    )

    # float64: Python floats coerce to float64 naturally
    df64 = np.array([2.0, 3.0, 4.0])
    zaf64 = zix.compact(df64)
    check(zix.power(zaf64, 2.0), df64**2.0)  # Python float scalar
    check(zix.power(2.0, zaf64), 2.0**df64)  # scalar first
    check(zix.power(zaf64, [3.0, 2.0, 0.5]), df64 ** [3.0, 2.0, 0.5])  # list
    check(zix.power(zaf64, (3.0, 2.0, 0.5)), df64 ** (3.0, 2.0, 0.5))  # tuple
