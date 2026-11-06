"""
Property tests for the bit counting and byte order ops.
Mirrors the test block in jix/src/ops/elementwise/bit_count.rs.
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
    check_op1_concrete,
    ints,
    uints,
)

import jix

_int_dtypes = ints + uints


def _as_uint(x: int, bits: int) -> int:
    return x & ((1 << bits) - 1)


def _ref_count_ones(a: np.ndarray) -> np.ndarray:
    bits = a.itemsize * 8
    out = [_as_uint(int(x), bits).bit_count() for x in a.reshape(-1)]
    return np.array(out, dtype=np.uint32).reshape(a.shape)


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
def test_byteswap_concrete():
    cases = _std_cases() + [(np.int8, [0, -1, 1, -128, 127]), (np.bool_, [True, False])]
    check_op1_concrete(jix.byteswap, lambda a: a.byteswap(), cases)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.bool_])
def test_byteswap_single_byte_returns_self(dtype):
    np_a = np.array([0, 1], dtype=dtype)
    za = jix.compact(np_a)
    assert jix.byteswap(za) is za
    # non-jix inputs still come back as a jix array
    for value in [np_a, np_a[0]]:
        result = jix.byteswap(value)
        assert isinstance(result, jix.Array)
        np.testing.assert_array_equal(result.numpy(), value)


def test_reverse_bits_concrete():
    check_op1_concrete(jix.reverse_bits, _ref_reverse_bits, _std_cases())
