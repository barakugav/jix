"""
Property tests for bitwise and logical ops.
Mirrors the test block in jix/src/ops/bitwise.rs.

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
    check_op1_concrete,
    check_op2_concrete,
    ints,
    shift_safe_element_strategy,
    uints,
)

import jix

_int_dtypes = ints + uints
_int_bool_dtypes = ints + uints + [np.bool_]


_LOGICAL_EXTRA_UNARY_CASES = {
    np.float64: np.array([float("nan"), -0.0, 0.0, 2.5], dtype=np.float64),
    np.complex128: np.array([0j, 2 + 0j, 3j, 1 + 1j], dtype=np.complex128),
    np.bool_: np.array([True, False], dtype=np.bool_),
}
_LOGICAL_EXTRA_BINARY_CASES = {
    np.float64: (
        np.array([0.0, 0.0, 2.5, float("nan")], dtype=np.float64),
        np.array([-0.0, 2.5, 0.0, 2.5], dtype=np.float64),
    ),
    np.complex128: (
        np.array([0j, 0j, 2 + 0j, 3j], dtype=np.complex128),
        np.array([0j, 2 + 0j, 0j, 1 + 1j], dtype=np.complex128),
    ),
    np.bool_: (
        np.array([False, False, True, True], dtype=np.bool_),
        np.array([False, True, False, True], dtype=np.bool_),
    ),
}

# ---------------------------------------------------------------------------
# Reference implementations for ops with no numpy equivalent
# ---------------------------------------------------------------------------


def _uint_type(a: np.ndarray) -> np.dtype:
    return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize]


def _as_uint(x: int, bits: int) -> int:
    return x & ((1 << bits) - 1)


def _ref_count_ones(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    out = [_as_uint(int(x), bits).bit_count() for x in a.reshape(-1)]
    return np.array(out, dtype=np.uint32).reshape(a.shape)


def _ref_rotate_left(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    ut = _uint_type(a)

    def _rl(x, y):
        x = _as_uint(int(x), bits)
        sh = int(y) % bits
        return ((x << sh) | (x >> (bits - sh))) & ((1 << bits) - 1) if sh else x

    # a and b share a shape (carrays2_mixed_strategy), so zip the flattened views.
    out = [_rl(x, y) for x, y in zip(a.reshape(-1), b.reshape(-1))]
    return np.array(out, dtype=ut).reshape(a.shape).view(a.dtype)


def _alt_pattern(bits: int) -> int:
    """Alternating bit pattern for the given width, e.g. bits=8 -> 0xAA."""
    return int("10" * (bits // 2), 2)


def _popcount_int(x: int, bits: int) -> int:
    return _as_uint(x, bits).bit_count()


def _leading_zeros_int(x: int, bits: int) -> int:
    x = _as_uint(x, bits)
    return bits if x == 0 else bits - x.bit_length()


def _trailing_zeros_int(x: int, bits: int) -> int:
    x = _as_uint(x, bits)
    if x == 0:
        return bits
    n = 0
    while (x & 1) == 0:
        x >>= 1
        n += 1
    return n


def _reverse_bits_int(x: int, bits: int) -> int:
    x = _as_uint(x, bits)
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _rotate_right_int(x: int, sh: int, bits: int) -> int:
    x = _as_uint(x, bits)
    sh %= bits
    return ((x >> sh) | (x << (bits - sh))) & ((1 << bits) - 1) if sh else x


def _bit_quad(dtype) -> tuple:
    """(0, dtype max, a single bit, an alternating pattern) for the given byte width,
    e.g. bits=8 -> (0, 255, 1, 0xAA). Shared building block for the fixed edge-value
    cases below."""
    maxv = np.iinfo(dtype).max
    alt = _alt_pattern(np.dtype(dtype).itemsize * 8)
    return 0, maxv, 1, alt


def _std_cases(dtypes=uints) -> list:
    """(dtype, [0, max, 1, alt]) cases: the standard fixed edge values shared by the
    byte-width unary concrete tests below."""
    return [(dtype, list(_bit_quad(dtype))) for dtype in dtypes]


def _bit_pair_vals(dtype) -> tuple:
    """(a_vals, b_vals) for binary byte-width concrete tests: 0<->max and
    single-bit<->alternating-pattern are swapped between a and b."""
    z, m, o, a = _bit_quad(dtype)
    return [z, m, o, a], [m, z, a, o]


def _shift_pair_vals(dtype) -> tuple:
    """(value, shift-amount) pairs: shift-by-0 and shift-by-(width-1), the extremes
    shift_safe_element_strategy draws from."""
    bits = np.dtype(dtype).itemsize * 8
    maxv = np.iinfo(dtype).max
    alt = _alt_pattern(bits)
    return [maxv, alt, 1, maxv], [0, bits - 1, bits - 1, 1]


def _logical_unary_vals(dtype) -> list:
    """[0, 1, dtype max, alternating pattern] - false, then three truthy variants."""
    z, m, o, a = _bit_quad(dtype)
    return [z, o, m, a]


def _logical_binary_vals(dtype) -> tuple:
    """(a_vals, b_vals) covering all four truthiness combinations:
    (False, False) / (False, True) / (True, False) / (True, True)."""
    z, m, o, a = _bit_quad(dtype)
    return [z, z, m, a], [z, m, z, o]


def _ref_count_zeros(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    return np.array([bits - _popcount_int(int(v), bits) for v in a], dtype=np.uint32)


def _ref_leading_zeros(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    return np.array([_leading_zeros_int(int(v), bits) for v in a], dtype=np.uint32)


def _ref_trailing_zeros(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    return np.array([_trailing_zeros_int(int(v), bits) for v in a], dtype=np.uint32)


def _ref_reverse_bits(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    return np.array([_reverse_bits_int(int(v), bits) for v in a], dtype=a.dtype)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


# Fixed inputs per byte width: 0 (false), a
# single bit, dtype max, and an alternating pattern (all truthy) - bool output.
def test_logical_not_concrete():
    cases = [(dtype, _logical_unary_vals(dtype)) for dtype in uints]
    cases += list(_LOGICAL_EXTRA_UNARY_CASES.items())
    check_op1_concrete(jix.logical_not, np.logical_not, cases)


def test_bitwise_not_concrete():
    check_op1_concrete(jix.bitwise_not, lambda a: ~a, _std_cases())


# bit-counting ops: output is u32.
@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_count_ones(dtype: np.dtype, data: DataObject):
    np_a, za = data.draw(carray_strategy(dtype, element_st=any_element_strategy(dtype)), label="array")
    assert_array_matches(jix.count_ones(za), _ref_count_ones(np_a), data=data)


def test_count_zeros_concrete():
    check_op1_concrete(jix.count_zeros, _ref_count_zeros, _std_cases())


def test_leading_zeros_concrete():
    check_op1_concrete(jix.leading_zeros, _ref_leading_zeros, _std_cases())


def test_trailing_zeros_concrete():
    check_op1_concrete(jix.trailing_zeros, _ref_trailing_zeros, _std_cases())


# byte/bit permutation: same output type, full range valid.
def test_swap_bytes_concrete():
    check_op1_concrete(jix.swap_bytes, lambda a: a.byteswap(), _std_cases())


def test_reverse_bits_concrete():
    check_op1_concrete(jix.reverse_bits, _ref_reverse_bits, _std_cases())


# ---------------------------------------------------------------------------
# Binary bitwise ops: same output type, full range valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _int_bool_dtypes)
@given(st.data())
def test_bitwise_and(dtype: np.dtype, data: DataObject):
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=any_element_strategy(dtype)), label="arrays")
    assert_array_matches(jix.bitwise_and(za, zb), np_a & np_b, data=data)


def test_bitwise_or_concrete():
    cases = [(dtype, *_bit_pair_vals(dtype)) for dtype in uints]
    # A 2x2 shape with a non-default 1x1 block shape, so a block-boundary bug in the
    # bitwise kernel would still show up (multi-block coverage for this op family).
    cases.append((np.uint16, [[0, 0xFFFF], [1, 0xAAAA]], [[0xFFFF, 0], [0xAAAA, 1]], [1, 1]))
    check_op2_concrete(jix.bitwise_or, lambda a, b: a | b, cases)


def test_bitwise_xor_concrete():
    cases = [(dtype, *_bit_pair_vals(dtype)) for dtype in uints]
    check_op2_concrete(jix.bitwise_xor, lambda a, b: a ^ b, cases)


# shift ops: shift amount must be in [0, bit_width) to avoid debug panic.
@pytest.mark.parametrize("dtype", _int_dtypes)
@given(st.data())
def test_bitwise_left_shift(dtype: np.dtype, data: DataObject):
    shift_st = shift_safe_element_strategy(dtype)
    (np_a, za), (np_b, zb) = data.draw(carrays2_strategy(dtype, element_st=shift_st), label="arrays")
    assert_array_matches(jix.bitwise_left_shift(za, zb), np_a << np_b, data=data)


# Shift amounts include shift-by-0 and shift-by-(width-1).
def test_bitwise_right_shift_concrete():
    cases = [(dtype, *_shift_pair_vals(dtype)) for dtype in uints]
    check_op2_concrete(jix.bitwise_right_shift, lambda a, b: a >> b, cases)


# rotate ops: LHS is any integer dtype; RHS (rotation amount) is always u32.
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
    assert_array_matches(jix.bitwise_rotate_left(za, zb), _ref_rotate_left(np_a, np_b), data=data)


# rotate_right: kept as a manual loop (not check_op2_concrete) because, like rotate_left,
# the rotation-amount array is always u32 regardless of the value dtype, while
# check_op2_concrete builds both operands from a single shared dtype.
def test_bitwise_rotate_right_concrete():
    for dtype in uints:
        bits = np.dtype(dtype).itemsize * 8
        vals, amounts = _shift_pair_vals(dtype)
        np_a = np.array(vals, dtype=dtype)
        np_b = np.array(amounts, dtype=np.uint32)
        za = jix.compact(np_a)
        zb = jix.compact(np_b)
        expected = np.array([_rotate_right_int(v, s, bits) for v, s in zip(vals, amounts)], dtype=dtype)
        assert_array_matches(jix.bitwise_rotate_right(za, zb), expected)


# ---------------------------------------------------------------------------
# Logical binary ops: bool output; reference uses np.logical_* to match cast::<T, bool>
# ---------------------------------------------------------------------------


# logical_and/or/xor: np_a/np_b truth values cover all four
# (False, False) / (False, True) / (True, False) / (True, True) combinations, with the
# True side using dtype-max / alternating-pattern nonzero values (not just 1).
def test_logical_and_concrete():
    cases = [(dtype, *_logical_binary_vals(dtype)) for dtype in uints]
    cases += [(dtype, a, b) for dtype, (a, b) in _LOGICAL_EXTRA_BINARY_CASES.items()]
    check_op2_concrete(jix.logical_and, np.logical_and, cases)


def test_logical_or_concrete():
    cases = [(dtype, *_logical_binary_vals(dtype)) for dtype in uints]
    cases += [(dtype, a, b) for dtype, (a, b) in _LOGICAL_EXTRA_BINARY_CASES.items()]
    check_op2_concrete(jix.logical_or, np.logical_or, cases)


def test_logical_xor_concrete():
    cases = [(dtype, *_logical_binary_vals(dtype)) for dtype in uints]
    cases += [(dtype, a, b) for dtype, (a, b) in _LOGICAL_EXTRA_BINARY_CASES.items()]
    check_op2_concrete(jix.logical_xor, np.logical_xor, cases)


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
    za = jix.compact(np_a)
    zb = jix.compact(np_b)
    result = jix.bitwise_and(za, zb)
    assert result.dtype == np.dtype(expected_dtype), (
        f"bitwise_and({dtype_a.__name__}, {dtype_b.__name__}): got {result.dtype}, expected {expected_dtype.__name__}"
    )
    expected = np_a.astype(expected_dtype) & np_b.astype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("dtype_a,dtype_b,expected_dtype", _BITWISE_MIXED_CASES)
def test_bitwise_or_mixed_dtypes(dtype_a, dtype_b, expected_dtype):
    """bitwise_or with different integer dtypes casts both to the expected result dtype."""
    np_a = np.array([0b1010, 0b0000, 0b1111], dtype=dtype_a)
    np_b = np.array([0b0101, 0b1111, 0b1010], dtype=dtype_b)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)
    result = jix.bitwise_or(za, zb)
    assert result.dtype == np.dtype(expected_dtype)
    expected = np_a.astype(expected_dtype) | np_b.astype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("dtype_a,dtype_b,expected_dtype", _BITWISE_MIXED_CASES)
def test_bitwise_xor_mixed_dtypes(dtype_a, dtype_b, expected_dtype):
    """bitwise_xor with different integer dtypes casts both to the expected result dtype."""
    np_a = np.array([0b1010, 0b1100, 0b0000], dtype=dtype_a)
    np_b = np.array([0b1111, 0b1010, 0b1111], dtype=dtype_b)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)
    result = jix.bitwise_xor(za, zb)
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
    za = jix.compact(np_a)
    zb = jix.compact(np_b)
    result = jix.logical_and(za, zb)
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
    za = jix.compact(np_a)
    zb = jix.compact(np_b)
    result = jix.logical_or(za, zb)
    assert result.dtype == np.bool_
    expected = np.logical_or(np_a, np_b)
    np.testing.assert_array_equal(result.numpy(), expected)
