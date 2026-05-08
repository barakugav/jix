"""
Hypothesis strategies and assertion helpers for zix property tests.

Mirrors the test utilities in zix/src/util/test_util.rs:
  shape_strategy          ←→  shape_strategy()
  op_safe_element_strategy ←→ ScalarStrategy::op_safe_strategy()
  carrays2_strategy       ←→  carrays2_strategy_generic()
  sub_range_strategy      ←→  sub_range_strategy()
  assert_array_matches    ←→  assert_array_matches()
"""

from typing import Optional

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from hypothesis.strategies import DataObject

import zix

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


def op_safe_element_strategy(dtype: np.dtype) -> st.SearchStrategy:
    """Bounded element strategy for dtype that avoids overflow in arithmetic ops."""
    # Bounded element ranges that keep arithmetic results in-range for each dtype.
    # Integer ranges are kept small so that e.g. a + b doesn't overflow the type.
    # Float/complex use small integers to avoid precision issues.
    return {
        np.int8: st.integers(1, 4),
        np.int16: st.integers(1, 22),
        np.int32: st.integers(1, 100),
        np.int64: st.integers(1, 100),
        np.uint8: st.integers(1, 4),
        np.uint16: st.integers(1, 27),
        np.uint32: st.integers(1, 30),
        np.uint64: st.integers(1, 30),
        np.float16: st.integers(1, 100).map(float),
        np.float32: st.integers(1, 100).map(float),
        np.float64: st.integers(1, 100).map(float),
        np.complex64: st.tuples(st.integers(1, 15), st.integers(1, 15)).map(
            lambda x: complex(float(x[0]), float(x[1]))
        ),
        np.complex128: st.tuples(st.integers(1, 15), st.integers(1, 15)).map(
            lambda x: complex(float(x[0]), float(x[1]))
        ),
    }[dtype]


@st.composite
def carrays2_strategy(draw, dtype: np.dtype, element_st=None):
    """
    Generate two (numpy_arr, zix_arr) pairs sharing a shape but with independent
    data and block shapes. Mirrors Rust's carrays2_strategy_generic().
    """
    if element_st is None:
        element_st = op_safe_element_strategy(dtype)

    shape = tuple(draw(shape_strategy(), label="shape"))
    ndim = len(shape)

    np_a = draw(np_arrays(dtype=dtype, shape=shape, elements=element_st), label="np_a")
    np_b = draw(np_arrays(dtype=dtype, shape=shape, elements=element_st), label="np_b")

    block_shape_a = draw(
        st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim), label="block_shape_a"
    )
    block_shape_b = draw(
        st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim), label="block_shape_b"
    )

    za = zix.compact(np_a, params={"block_shape": block_shape_a})
    zb = zix.compact(np_b, params={"block_shape": block_shape_b})

    return (np_a, za), (np_b, zb)


@st.composite
def sub_range_strategy(draw, shape: tuple):
    """
    Random sub-range (tuple of slices) for an array of the given shape.
    zix requires 0 <= start < size, so only call this when all dims are non-zero.
    """
    slices = []
    for size in shape:
        start = draw(st.integers(0, size - 1), label="sub_range_start")
        stop = draw(st.integers(start, size), label="sub_range_stop")
        slices.append(slice(start, stop))
    return tuple(slices)


def assert_array_matches(
    actual: zix.Array,
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

    # zix rejects slice(0, 0) on zero-length dimensions; skip sub-range for those.
    if data is not None and actual.shape and all(s > 0 for s in actual.shape):
        sub_idx = data.draw(sub_range_strategy(actual.shape), label="sub_range")
        assert_arr_equal(actual[sub_idx], expected[sub_idx])
