"""
Hypothesis strategies and assertion helpers for jix property tests.

Mirrors the test utilities in jix/src/util/test_util.rs:
  shape_strategy           <-->  shape_strategy()
  op_safe_element_strategy <-->  ScalarStrategy::op_safe_strategy()
  carrays2_strategy        <-->  carrays2_strategy_generic()
  sub_range_strategy       <-->  sub_range_strategy()
  assert_array_matches     <-->  assert_array_matches()
"""

from typing import Optional

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from hypothesis.strategies import DataObject


import jix

ints = [np.int8, np.int16, np.int32, np.int64]
uints = [np.uint8, np.uint16, np.uint32, np.uint64]
floats = [np.float16, np.float32, np.float64]
complexes = [np.complex64, np.complex128]


def shape_strategy():
    """Random array shapes across a variety of dimensionalities and sizes."""
    return st.one_of(
        st.lists(st.integers(1, 100), min_size=1, max_size=1),  # 1D small
        st.lists(st.integers(100, 1000), min_size=1, max_size=1),  # 1D large
        st.lists(st.integers(1, 20), min_size=2, max_size=2),  # 2D small
        st.lists(st.integers(20, 50), min_size=2, max_size=2),  # 2D medium
        st.lists(st.integers(1, 16), min_size=3, max_size=3),  # 3D
        st.lists(st.integers(1, 8), min_size=4, max_size=4),  # 4D
        st.lists(st.integers(1, 4), min_size=1, max_size=8),  # many dims
        st.lists(st.integers(0, 3), min_size=1, max_size=8),  # zero-length dims
    )


def any_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Full domain strategy. Mirrors ScalarStrategy::any_strategy() in Rust."""
    _f32 = st.floats(width=32)
    _f64 = st.floats(width=64)
    return {
        np.int8: st.integers(-128, 127),
        np.int16: st.integers(-32768, 32767),
        np.int32: st.integers(-(2**31), 2**31 - 1),
        np.int64: st.integers(-(2**63), 2**63 - 1),
        np.uint8: st.integers(0, 255),
        np.uint16: st.integers(0, 65535),
        np.uint32: st.integers(0, 2**32 - 1),
        np.uint64: st.integers(0, 2**64 - 1),
        np.float16: st.floats(width=16),
        np.float32: _f32,
        np.float64: _f64,
        np.bool_: st.booleans(),
        np.complex64: st.tuples(_f32, _f32).map(lambda x: complex(x[0], x[1])),
        np.complex128: st.tuples(_f64, _f64).map(lambda x: complex(x[0], x[1])),
    }[dtype]


def logical_op_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """any_strategy plus extra zero/default. Mirrors ScalarStrategy::logical_op_strategy()."""
    _zero = {
        np.int8: 0,
        np.int16: 0,
        np.int32: 0,
        np.int64: 0,
        np.uint8: 0,
        np.uint16: 0,
        np.uint32: 0,
        np.uint64: 0,
        np.float16: 0.0,
        np.float32: 0.0,
        np.float64: 0.0,
        np.bool_: False,
        np.complex64: complex(0.0, 0.0),
        np.complex128: complex(0.0, 0.0),
    }[dtype]
    return st.one_of(any_element_strategy(dtype), st.just(_zero))


def comparable_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Small fixed set for ~33% equal pairs; floats include NaN for NaN!=NaN coverage.
    Mirrors ScalarStrategy::comparable_strategy()."""
    _i = [0, 1, 2]
    _f = [0.0, 1.0, 2.4, float("nan")]
    _c = [
        complex(0, 0),
        complex(1, 0),
        complex(0, 1),
        complex(float("nan"), 0),
        complex(0, float("nan")),
    ]
    return {
        np.int8: st.sampled_from(_i),
        np.int16: st.sampled_from(_i),
        np.int32: st.sampled_from(_i),
        np.int64: st.sampled_from(_i),
        np.uint8: st.sampled_from(_i),
        np.uint16: st.sampled_from(_i),
        np.uint32: st.sampled_from(_i),
        np.uint64: st.sampled_from(_i),
        np.float16: st.sampled_from(_f),
        np.float32: st.sampled_from(_f),
        np.float64: st.sampled_from(_f),
        np.bool_: st.booleans(),
        np.complex64: st.sampled_from(_c),
        np.complex128: st.sampled_from(_c),
    }[dtype]


def maybe_non_finite_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Full float domain with extra inf/nan weight. Mirrors ScalarStrategy::maybe_non_finite_strategy()."""
    _extra = st.sampled_from([float("inf"), float("-inf"), float("nan")])
    return {
        np.float16: st.one_of(st.floats(width=16), _extra),
        np.float32: st.one_of(st.floats(width=32), _extra),
        np.float64: st.one_of(st.floats(width=64), _extra),
    }[dtype]


def shift_safe_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Shift amounts in [0, bit_width). Mirrors ScalarStrategy::shift_safe_strategy()."""
    return {
        np.int8: st.integers(0, 7),
        np.int16: st.integers(0, 15),
        np.int32: st.integers(0, 31),
        np.int64: st.integers(0, 63),
        np.uint8: st.integers(0, 7),
        np.uint16: st.integers(0, 15),
        np.uint32: st.integers(0, 31),
        np.uint64: st.integers(0, 63),
    }[dtype]


def _float_op_safe_st() -> st.SearchStrategy:
    """float op_safe: (0..=20000).map(|x| (x - 10000) / 100.0) -> [-100.0, 100.0]."""
    return st.integers(0, 2 * 100 * 100).map(lambda x: float((x - 100 * 100) / 100.0))


def op_safe_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Bounded element strategy mirroring ScalarStrategy::op_safe_strategy() in Rust."""
    _f = _float_op_safe_st()
    return {
        np.int8: st.integers(-4, 4),
        np.int16: st.integers(-22, 22),
        np.int32: st.integers(-100, 100),
        np.int64: st.integers(-100, 100),
        np.uint8: st.integers(0, 4),
        np.uint16: st.integers(0, 27),
        np.uint32: st.integers(0, 30),
        np.uint64: st.integers(0, 30),
        np.float16: _f,
        np.float32: _f,
        np.float64: _f,
        np.complex64: st.tuples(_f, _f).map(lambda x: complex(x[0], x[1])),
        np.complex128: st.tuples(_f, _f).map(lambda x: complex(x[0], x[1])),
        np.bool_: st.booleans(),
    }[dtype]


@st.composite
def carray_strategy(draw, dtype: np.dtype, element_st=None):
    """
    Generate a single (numpy_arr, jix_arr) pair. Mirrors Rust's carray_strategy_from_shape().
    """
    if element_st is None:
        element_st = op_safe_element_strategy(dtype)

    shape = tuple(draw(shape_strategy(), label="shape"))
    ndim = len(shape)

    np_a = draw(np_arrays(dtype=dtype, shape=shape, elements=element_st), label="np_a")
    block_shape = draw(st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim), label="block_shape")
    za = jix.compact(np_a, params={"block_shape": block_shape})
    return np_a, za


def op_safe_non_zero_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Non-zero element strategy mirroring ScalarStrategy::op_safe_non_zero_strategy() in Rust."""
    _f_nz = _float_op_safe_st().filter(lambda x: x != 0.0)
    return {
        np.int8: st.one_of(st.integers(-4, -1), st.integers(1, 4)),
        np.int16: st.one_of(st.integers(-22, -1), st.integers(1, 22)),
        np.int32: st.one_of(st.integers(-100, -1), st.integers(1, 100)),
        np.int64: st.one_of(st.integers(-100, -1), st.integers(1, 100)),
        np.uint8: st.integers(1, 4),
        np.uint16: st.integers(1, 27),
        np.uint32: st.integers(1, 30),
        np.uint64: st.integers(1, 30),
        np.float16: _f_nz,
        np.float32: _f_nz,
        np.float64: _f_nz,
        np.complex64: st.tuples(_f_nz, _f_nz).map(lambda x: complex(x[0], x[1])),
        np.complex128: st.tuples(_f_nz, _f_nz).map(lambda x: complex(x[0], x[1])),
    }[dtype]


def op_safe_non_negative_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Non-negative float elements mirroring ScalarStrategy::op_safe_non_negative_strategy().

    (0..=10000).map(|x| x / 100.0) -> [0.0, 100.0]
    """
    _st = st.integers(0, 100 * 100).map(lambda x: float(x) / 100.0)
    return {
        np.float16: _st,
        np.float32: _st,
        np.float64: _st,
    }[dtype]


def unit_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Float elements in [-1, 1] mirroring ScalarStrategy::unit_strategy().

    (-100..=100).map(|x| x / 100.0) -> [-1.0, 1.0]
    """
    _st = st.integers(-100, 100).map(lambda x: float(x) / 100.0)
    return {
        np.float32: _st,
        np.float64: _st,
    }[dtype]


@st.composite
def carrays2_strategy(draw, dtype: np.dtype, element_st=None):
    """
    Generate two (numpy_arr, jix_arr) pairs sharing a shape but with independent
    data and block shapes. Mirrors Rust's carrays2_strategy_generic().
    """
    return draw(carrays2_mixed_strategy(dtype, dtype, element_st_a=element_st, element_st_b=element_st))


@st.composite
def carrays2_mixed_strategy(draw, dtype_a: np.dtype, dtype_b: np.dtype, element_st_a=None, element_st_b=None):
    """
    Generate two (numpy_arr, jix_arr) pairs sharing a shape but with different dtypes.
    Useful for ops where LHS and RHS have distinct types (e.g. rotate: value=T, amount=u32).
    """
    if element_st_a is None:
        element_st_a = op_safe_element_strategy(dtype_a)
    if element_st_b is None:
        element_st_b = op_safe_element_strategy(dtype_b)

    shape = tuple(draw(shape_strategy(), label="shape"))
    ndim = len(shape)

    np_a = draw(np_arrays(dtype=dtype_a, shape=shape, elements=element_st_a), label="np_a")
    np_b = draw(np_arrays(dtype=dtype_b, shape=shape, elements=element_st_b), label="np_b")

    block_shape_a = draw(st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim), label="block_shape_a")
    block_shape_b = draw(st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim), label="block_shape_b")

    za = jix.compact(np_a, params={"block_shape": block_shape_a})
    zb = jix.compact(np_b, params={"block_shape": block_shape_b})

    return (np_a, za), (np_b, zb)


@st.composite
def sub_range_strategy(draw, shape: tuple):
    """
    Random sub-range (tuple of slices) for an array of the given shape.
    jix requires 0 <= start < size, so only call this when all dims are non-zero.
    """
    slices = []
    for size in shape:
        start = draw(st.integers(0, size - 1), label="sub_range_start")
        stop = draw(st.integers(start, size), label="sub_range_stop")
        slices.append(slice(start, stop))
    return tuple(slices)


def assert_array_matches(
    actual: jix.Array,
    expected: np.ndarray,
    *,
    data: Optional[DataObject] = None,
    rtol: float = 0.0,
    atol: float = 0.0,
):
    """
    Assert actual matches expected for the full array. When data is provided
    (from @given(st.data())), also checks one random sub-range.
    Mirrors Rust's assert_array_matches().
    """

    def assert_arr_equal(a, b):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    assert_arr_equal(actual.numpy(), expected)

    # jix rejects slice(0, 0) on zero-length dimensions; skip sub-range for those.
    if data is not None and actual.shape and all(s > 0 for s in actual.shape):
        sub_idx = data.draw(sub_range_strategy(actual.shape), label="sub_range")
        assert_arr_equal(actual[sub_idx], expected[sub_idx])
