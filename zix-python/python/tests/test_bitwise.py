"""
Property tests for bitwise and logical ops.
Mirrors the test block in zix/src/ops/bitwise.rs.

Auto-cast section verifies that Safe/Unsafe dispatch rules work for non-matching inputs:
- bitwise_and/or/xor use CastKind::Safe, so e.g. u8+u16 -> u16.
- logical_and/or/xor use CastKind::Unsafe, so any non-complex input casts to bool.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import (
    any_element_strategy,
    assert_array_matches,
    carray_strategy,
    carrays2_mixed_strategy,
    carrays2_strategy,
    complexes,
    floats,
    ints,
    logical_op_element_strategy,
    shift_safe_element_strategy,
    uints,
)

import zix

_int_dtypes = ints + uints
_int_bool_dtypes = ints + uints + [np.bool_]
_logical_dtypes = ints + uints + floats + complexes + [np.bool_]
_multibyte_int_dtypes = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]

# ---------------------------------------------------------------------------
# Reference implementations for ops with no numpy equivalent
# ---------------------------------------------------------------------------


def _uint_type(a: np.ndarray) -> np.dtype:
    return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize]


def _as_uint(x: int, bits: int) -> int:
    return x & ((1 << bits) - 1)


def _ref_count_ones(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    return np.vectorize(
        lambda x: bin(_as_uint(int(x), bits)).count("1"), otypes=[np.uint32]
    )(a)


def _ref_count_zeros(a: np.ndarray) -> np.ndarray:
    return np.full(a.shape, a.itemsize * 8, dtype=np.uint32) - _ref_count_ones(a)


def _ref_leading_zeros(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8

    def _lz(x):
        x = _as_uint(int(x), bits)
        return bits if x == 0 else bits - x.bit_length()

    return np.vectorize(_lz, otypes=[np.uint32])(a)


def _ref_trailing_zeros(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8

    def _tz(x):
        x = _as_uint(int(x), bits)
        if x == 0:
            return bits
        n = 0
        while (x & 1) == 0:
            x >>= 1
            n += 1
        return n

    return np.vectorize(_tz, otypes=[np.uint32])(a)


def _ref_reverse_bits(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    ut = _uint_type(a)

    def _rb(x):
        x = _as_uint(int(x), bits)
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    return np.vectorize(_rb, otypes=[ut])(a).view(a.dtype)


def _ref_rotate_left(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    ut = _uint_type(a)

    def _rl(x, y):
        x = _as_uint(int(x), bits)
        sh = int(y) % bits
        return ((x << sh) | (x >> (bits - sh))) & ((1 << bits) - 1) if sh else x

    return np.vectorize(_rl, otypes=[ut])(a, b).view(a.dtype)


def _ref_rotate_right(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    ut = _uint_type(a)

    def _rr(x, y):
        x = _as_uint(int(x), bits)
        sh = int(y) % bits
        return ((x >> sh) | (x << (bits - sh))) & ((1 << bits) - 1) if sh else x

    return np.vectorize(_rr, otypes=[ut])(a, b).view(a.dtype)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


# logical_not: any_strategy + zeros to exercise true branch (bool output, safe for NaN inputs)
@pytest.mark.parametrize("dtype", _logical_dtypes)
@given(st.data())
def test_logical_not(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=logical_op_element_strategy(dtype)),
        label="array",
    )
    assert_array_matches(zix.logical_not(za), np.logical_not(np_a), data=data)


# bitwise_not: full range is valid (no overflow for bitwise complement)
@pytest.mark.parametrize("dtype", _int_bool_dtypes)
@given(st.data())
def test_bitwise_not(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.bitwise_not(za), ~np_a, data=data)


# bit-counting ops: output is u32
@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_count_ones(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.count_ones(za), _ref_count_ones(np_a), data=data)


@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_count_zeros(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.count_zeros(za), _ref_count_zeros(np_a), data=data)


@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_leading_zeros(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.leading_zeros(za), _ref_leading_zeros(np_a), data=data)


@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_trailing_zeros(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.trailing_zeros(za), _ref_trailing_zeros(np_a), data=data)


# byte/bit permutation: same output type, full range valid
@pytest.mark.parametrize("dtype", _multibyte_int_dtypes)
@given(st.data())
def test_swap_bytes(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.swap_bytes(za), np_a.byteswap(), data=data)


@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_reverse_bits(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(
        carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array"
    )
    assert_array_matches(zix.reverse_bits(za), _ref_reverse_bits(np_a), data=data)


# ---------------------------------------------------------------------------
# Binary bitwise ops: same output type, full range valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _int_bool_dtypes)
@given(st.data())
def test_bitwise_and(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=any_element_strategy(dtype)), label="arrays"
    )
    assert_array_matches(zix.bitwise_and(za, zb), np_a & np_b, data=data)


@pytest.mark.parametrize("dtype", _int_bool_dtypes)
@given(st.data())
def test_bitwise_or(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=any_element_strategy(dtype)), label="arrays"
    )
    assert_array_matches(zix.bitwise_or(za, zb), np_a | np_b, data=data)


@pytest.mark.parametrize("dtype", _int_bool_dtypes)
@given(st.data())
def test_bitwise_xor(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=any_element_strategy(dtype)), label="arrays"
    )
    assert_array_matches(zix.bitwise_xor(za, zb), np_a ^ np_b, data=data)


# shift ops: shift amount must be in [0, bit_width) to avoid debug panic
@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_bitwise_left_shift(dtype: np.dtype, data: DataObject):
    shift_st = shift_safe_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=shift_st), label="arrays"
    )
    assert_array_matches(zix.bitwise_left_shift(za, zb), np_a << np_b, data=data)


@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_bitwise_right_shift(dtype: np.dtype, data: DataObject):
    shift_st = shift_safe_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=shift_st), label="arrays"
    )
    assert_array_matches(zix.bitwise_right_shift(za, zb), np_a >> np_b, data=data)


# rotate ops: LHS is any integer dtype; RHS (rotation amount) is always u32
@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_bitwise_rotate_left(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_mixed_strategy(
            dtype,
            np.uint32,
            element_st_a=any_element_strategy(dtype),
            element_st_b=any_element_strategy(np.uint32),
        ),
        label="arrays",
    )
    assert_array_matches(
        zix.bitwise_rotate_left(za, zb), _ref_rotate_left(np_a, np_b), data=data
    )


@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_bitwise_rotate_right(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_mixed_strategy(
            dtype,
            np.uint32,
            element_st_a=any_element_strategy(dtype),
            element_st_b=any_element_strategy(np.uint32),
        ),
        label="arrays",
    )
    assert_array_matches(
        zix.bitwise_rotate_right(za, zb), _ref_rotate_right(np_a, np_b), data=data
    )


# ---------------------------------------------------------------------------
# Logical binary ops: bool output; reference uses np.logical_* to match cast::<T, bool>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _logical_dtypes)
@given(st.data())
def test_logical_and(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=logical_op_element_strategy(dtype)),
        label="arrays",
    )
    assert_array_matches(zix.logical_and(za, zb), np.logical_and(np_a, np_b), data=data)


@pytest.mark.parametrize("dtype", _logical_dtypes)
@given(st.data())
def test_logical_or(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=logical_op_element_strategy(dtype)),
        label="arrays",
    )
    assert_array_matches(zix.logical_or(za, zb), np.logical_or(np_a, np_b), data=data)


@pytest.mark.parametrize("dtype", _logical_dtypes)
@given(st.data())
def test_logical_xor(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_strategy(dtype, element_st=logical_op_element_strategy(dtype)),
        label="arrays",
    )
    assert_array_matches(zix.logical_xor(za, zb), np.logical_xor(np_a, np_b), data=data)


# ---------------------------------------------------------------------------
# Mixed-dtype bitwise ops (CastKind::Safe auto-cast)
# ---------------------------------------------------------------------------

_BITWISE_MIXED_CASES = [
    # (dtype_a, dtype_b, expected_result_dtype)
    (np.uint8, np.uint16, np.uint16),
    (np.uint8, np.int32, np.int32),
    (np.int8, np.uint16, np.int32),  # i8->u16: needs higher: P2.higher=P4, i32
    (np.uint16, np.uint32, np.uint32),
    (np.int16, np.int64, np.int64),
    (np.bool_, np.uint8, np.uint8),
    (np.bool_, np.int32, np.int32),
]


@pytest.mark.parametrize("dtype_a,dtype_b,expected_dtype", _BITWISE_MIXED_CASES)
def test_bitwise_and_mixed_dtypes(dtype_a, dtype_b, expected_dtype):
    """bitwise_and with different integer dtypes casts both to the expected result dtype."""
    np_a = np.array([0b1010, 0b1100, 0b1111], dtype=dtype_a)
    np_b = np.array([0b1111, 0b1010, 0b0000], dtype=dtype_b)
    za = zix.compact(np_a)
    zb = zix.compact(np_b)
    result = zix.bitwise_and(za, zb)
    assert result.dtype == np.dtype(expected_dtype), (
        f"bitwise_and({dtype_a.__name__}, {dtype_b.__name__}): "
        f"got {result.dtype}, expected {expected_dtype.__name__}"
    )
    expected = np_a.astype(expected_dtype) & np_b.astype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("dtype_a,dtype_b,expected_dtype", _BITWISE_MIXED_CASES)
def test_bitwise_or_mixed_dtypes(dtype_a, dtype_b, expected_dtype):
    """bitwise_or with different integer dtypes casts both to the expected result dtype."""
    np_a = np.array([0b1010, 0b0000, 0b1111], dtype=dtype_a)
    np_b = np.array([0b0101, 0b1111, 0b1010], dtype=dtype_b)
    za = zix.compact(np_a)
    zb = zix.compact(np_b)
    result = zix.bitwise_or(za, zb)
    assert result.dtype == np.dtype(expected_dtype)
    expected = np_a.astype(expected_dtype) | np_b.astype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("dtype_a,dtype_b,expected_dtype", _BITWISE_MIXED_CASES)
def test_bitwise_xor_mixed_dtypes(dtype_a, dtype_b, expected_dtype):
    """bitwise_xor with different integer dtypes casts both to the expected result dtype."""
    np_a = np.array([0b1010, 0b1100, 0b0000], dtype=dtype_a)
    np_b = np.array([0b1111, 0b1010, 0b1111], dtype=dtype_b)
    za = zix.compact(np_a)
    zb = zix.compact(np_b)
    result = zix.bitwise_xor(za, zb)
    assert result.dtype == np.dtype(expected_dtype)
    expected = np_a.astype(expected_dtype) ^ np_b.astype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), expected)


# ---------------------------------------------------------------------------
# Mixed-dtype logical ops (CastKind::Unsafe auto-cast to bool)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        (np.float32, np.int32),
        (np.float64, np.uint8),
        (np.int8, np.float32),
        (np.uint16, np.bool_),
        (np.float32, np.float64),
    ],
)
def test_logical_and_mixed_dtypes(dtype_a, dtype_b):
    """logical_and with different dtypes: both cast to bool (Unsafe), output is bool."""
    np_a = np.array([0, 1, 0, 1], dtype=dtype_a)
    np_b = np.array([1, 1, 0, 0], dtype=dtype_b)
    za = zix.compact(np_a)
    zb = zix.compact(np_b)
    result = zix.logical_and(za, zb)
    assert result.dtype == np.bool_
    expected = np.logical_and(np_a, np_b)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        (np.float32, np.int32),
        (np.float64, np.uint8),
        (np.int16, np.float64),
    ],
)
def test_logical_or_mixed_dtypes(dtype_a, dtype_b):
    """logical_or with different dtypes: both cast to bool (Unsafe), output is bool."""
    np_a = np.array([0, 1, 0, 1], dtype=dtype_a)
    np_b = np.array([0, 0, 1, 1], dtype=dtype_b)
    za = zix.compact(np_a)
    zb = zix.compact(np_b)
    result = zix.logical_or(za, zb)
    assert result.dtype == np.bool_
    expected = np.logical_or(np_a, np_b)
    np.testing.assert_array_equal(result.numpy(), expected)
